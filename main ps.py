import cv2
import numpy as np
import matplotlib.pyplot as plt


class PhotometricStereoSolver:
    EPS = 1e-12

    @staticmethod
    def normalize(arr, eps: float | None = None):
        if eps is None:
            eps = PhotometricStereoSolver.EPS
        return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + eps)

    @staticmethod
    def show_normal(norm, gain: float = 10):
        # enhance x/y by
        plt.imshow(((PhotometricStereoSolver.normalize(norm * np.array([gain, gain, 1])) / 2 + .5) * 255).astype(np.uint8))
        plt.show()

    def __init__(self, camera_pos: np.ndarray, light_pos: np.ndarray):
        self.camera_pos = camera_pos[:, None, None, :]
        self.light_pos = light_pos[:, None, None, :]

    # luminance
    def normal_solve(self, luminance: np.ndarray, iteration: int = 2):
        *_, img_h, img_w = luminance.shape

        # get pixel location
        pixel_cord = np.moveaxis(np.mgrid[:img_h, :img_w], 0, -1) - np.array([img_h, img_w]) / 2
        pixel_cord = np.concatenate([pixel_cord, np.zeros([img_h, img_w, 1])], axis=-1)

        # calculate the direction vector to camera&light for each pixel
        camera_dir = self.normalize(self.camera_pos - pixel_cord)
        light_dir = self.normalize(self.light_pos - pixel_cord)

        # get light-camera Tensor
        A_i = np.einsum('...i,...j->...ij', light_dir, camera_dir)
        A = np.sum(luminance[..., None, None] * (A_i + np.moveaxis(A_i, -2, -1)), axis=0)

        # do power iteration
        n = np.ones((img_h, img_w, 3), dtype=np.float64)
        n *= np.array([0, 0, 1])
        for _ in range(iteration):
            n = self.normalize(np.einsum('...ij,...i->...j', A, n))
            n *= np.sign(n[..., -1:])

        return self.normalize(n)

    def height_solve(self, normal_map: np.ndarray):
        *_, img_h, img_w, _ = normal_map.shape

        # FFT transform
        x, y, z = np.moveaxis(normal_map, -1, 0)
        X = np.fft.fftshift(np.fft.fft2(- x / (z + self.EPS)))
        Y = np.fft.fftshift(np.fft.fft2(- y / (z + self.EPS)))

        # get U / V (frequency variable) for derivative & integral & filter
        V, U = np.mgrid[:img_h, :img_w]
        V, U = 2 * np.pi * (V / img_h - .5), 2 * np.pi * (U / img_w - .5)

        # derivative & integral & filter
        H = - 1j * (U * X + V * Y) / (U ** 2 + V ** 2 + 1e-2)
        H *= 1 / (1 + 1e-3 * np.sqrt(U ** 2 + V ** 2))

        return np.fft.ifft2(np.fft.ifftshift(H)).real

    def solve(self, luminance: np.ndarray, debug: bool = False):
        normal_map = self.normal_solve(luminance)
        if debug:
            self.show_normal(normal_map)
        height_map = self.height_solve(normal_map)
        return height_map


def get_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = cap.read()
    return frame


video = cv2.VideoCapture("./data/TDM/IMG_2765.MOV")

# read in images from video and cut out ROI
frame_idx_init, frame_idx_step = 250, 120
imgs = [
    get_frame(video, frame_idx_init + 0 * frame_idx_step).mean(axis=-1),
    get_frame(video, frame_idx_init + 1 * frame_idx_step).mean(axis=-1),
    get_frame(video, frame_idx_init + 2 * frame_idx_step).mean(axis=-1),
    get_frame(video, frame_idx_init + 3 * frame_idx_step).mean(axis=-1),
    get_frame(video, frame_idx_init + 4 * frame_idx_step).mean(axis=-1),
]
imgs = np.stack(imgs, axis=0).astype(np.float64) / 255.
(_, h, w), r = imgs.shape, 1100
imgs = imgs[..., (h - r) // 2:(h + r) // 2, (w - r) // 2:(w + r) // 2]

# show gray image
plt.imshow(np.mean(imgs, axis=0), cmap='gray')
plt.show()

# set up the configuration
l_r, l_h, c_h = 4000, 10000, 10000
light_angle = np.deg2rad(np.array([-30, 30, 90, 150, 210]))
solver = PhotometricStereoSolver(
    camera_pos=np.array([0, 0, c_h]) * np.ones(len(imgs))[:, None],
    light_pos=np.stack([l_r * np.cos(light_angle), l_r * np.sin(light_angle), l_h * np.ones_like(light_angle)], axis=-1)
)

# solve height
height = solver.solve(imgs, debug=True)

# save result
np.save("height.npy", height)

# normalization
height = (height - np.min(height)) / (np.max(height) - np.min(height))
cv2.imwrite('h.png', (height * 255).astype(np.uint8))

# show
clip = .005
plt.imshow(np.clip(height, np.quantile(height, clip), np.quantile(height, 1 - clip)))
plt.show()
