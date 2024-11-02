import cv2
import scipy
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

    def __init__(
            self,
            camera_pos: np.ndarray,
            light_pos: np.ndarray,
            iteration: int = 2,
            high_pass: float = 1e-4,
            low_pass: float = 1e-5,
    ):
        self.camera_pos = camera_pos[:, None, None, :]
        self.light_pos = light_pos[:, None, None, :]

        self.iteration = iteration
        self.high_pass = high_pass
        self.low_pass = low_pass

        self.debug = True

    # luminance
    def normal_solve(self, luminance: np.ndarray):
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
        for _ in range(self.iteration):
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
        H = - 1j * (U * X + V * Y) / (U ** 2 + V ** 2 + self.high_pass)
        H *= 1 / (1 + self.low_pass * np.sqrt(U ** 2 + V ** 2))

        if self.debug:
            normal_map = np.stack(
                [
                    - np.fft.ifft2(np.fft.ifftshift(- 1j * U * H)).real,
                    - np.fft.ifft2(np.fft.ifftshift(- 1j * V * H)).real,
                    np.ones_like(H.real)
                ], axis=-1
            )
            normal_map = self.normalize(normal_map)
            self.show_normal(normal_map)

        return np.fft.ifft2(np.fft.ifftshift(H)).real

    def solve(self, luminance: np.ndarray):
        normal_map = self.normal_solve(luminance)
        if self.debug:
            self.show_normal(normal_map)
        height_map = self.height_solve(normal_map)
        return height_map


def get_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = cap.read()
    return frame


def get_groups_from_video(cap: cv2.VideoCapture):
    # get light curve of video
    avg_light = []
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        avg_light.append(np.mean(frame))
    avg_light = np.array(avg_light)

    # find edges of the light curve
    diff_avg_light = np.diff(avg_light)
    rising_edge, _ = scipy.signal.find_peaks(diff_avg_light, threshold=1, distance=5)
    falling_edge, _ = scipy.signal.find_peaks(-diff_avg_light, threshold=1, distance=5)

    # get rise edge with pulse width before and after
    state_edge = [(r, np.min((r - falling_edge)[falling_edge < r]), np.min((falling_edge - r)[falling_edge > r])) for r in rising_edge if np.any(falling_edge < r) and np.any(falling_edge > r)]

    # split groups by black length
    groups = []
    tmp_group = []
    for frame_idx, width_before, width_after in state_edge:
        if width_before >= 15:
            tmp_group = [frame_idx - width_before // 2, ]
        tmp_group.append(frame_idx + width_after // 2)
        if len(tmp_group) == 5 + 1:
            groups.append(tmp_group)
            tmp_group = []

    return groups


video = cv2.VideoCapture("./data/TDM/IMG_2842.MOV")
groups = get_groups_from_video(video)

for group_idx, group in enumerate(groups):
    # read in images from video and cut out ROI
    gamma, blur_r = 0.92, 15
    imgs = [(get_frame(video, idx).astype(np.float64) / 255.).mean(axis=-1) ** (1 / gamma) for idx in group]
    dark_field, *imgs = imgs
    dark_field = cv2.blur(dark_field, (blur_r, blur_r))
    imgs = np.stack(imgs, axis=0)
    imgs = np.maximum(imgs - dark_field, 0)
    (_, h, w), r = imgs.shape, 1300
    imgs = imgs[..., (h - r) // 2:(h + r) // 2, (w - r) // 2:(w + r) // 2]

    # show gray image
    plt.imshow(np.mean(imgs, axis=0), cmap='gray')
    plt.title(f'Group {group_idx} gray image')
    plt.show()

    # set up the configuration
    l_r, l_h, c_h = 1000, 6000, 6000
    light_angle = np.deg2rad(np.array([-30, 30, 90, 150, 210]))
    solver = PhotometricStereoSolver(
        camera_pos=np.array([0, 0, c_h]) * np.ones(len(imgs))[:, None],
        light_pos=np.stack([l_r * np.cos(light_angle), l_r * np.sin(light_angle), l_h * np.ones_like(light_angle)], axis=-1)
    )

    # solve height
    height = solver.solve(imgs)

    # save result
    np.save(f"height_{group_idx}.npy", height)

    # normalization
    height = (height - np.min(height)) / (np.max(height) - np.min(height))
    cv2.imwrite(f"height_{group_idx}.png", (height * 255).astype(np.uint8))

    # show
    clip = .005
    plt.imshow(np.clip(height, np.quantile(height, clip), np.quantile(height, 1 - clip)))
    plt.title(f'Group {group_idx} height image')
    plt.show()
