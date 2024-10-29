import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


def normalize(arr, eps=EPS):
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + eps)


def show_normal(norm):
    plt.imshow(((normalize(norm) / 2 + .5) * 255).astype(np.uint8))

# load a RGB image and separate the channels into three 2D arrays, then convert each channel to grayscale, respectively
I = cv2.imread('example_images/IMG_2586.JPG', cv2.IMREAD_COLOR)

# 先进行裁剪 (从 750 到 2500)
I = I[750:2500, 750:2500]

B, G, R = cv2.split(I)

I_i = [
    B, G, R
]
I_i = np.stack(I_i, axis=0).astype(np.float64) / 255.
I_i -= 0.85 * np.mean(I_i, axis=0)
I_i = I_i[..., 1000:2500, 400:1800]

plt.imshow(np.mean(I_i, axis=0))
plt.show()

h, w = I_i.shape[-2:]

cord = np.moveaxis(np.mgrid[:h, :w], 0, -1) - np.array([h, w]) / 2
cord = np.concatenate([cord, np.zeros([h, w, 1])], axis=-1)

l_r, l_h, c_h = 4000, 10000, 10000
s = normalize(np.array([0, 0, c_h]) - cord)
l = np.array(
    [
        [0, -l_r, l_h],
        [0, +l_r, l_h],
        [+l_r, 0, l_h],
        [-l_r, 0, l_h],
    ]
)
l = normalize(l[:, None, None, :] - cord)
A_i = np.einsum('...i,...j->...ij', l, s)
A = np.sum(I_i[..., None, None] * (A_i + np.moveaxis(A_i, -2, -1)), axis=0)
C = np.sum(I_i[..., None] * l, axis=0)

n = np.ones((h, w, 3), dtype=np.float64)
n *= np.array([0, 0, 1])
for _ in range(2):
    n = normalize(np.einsum('...ij,...i->...j', A, n))
    n *= np.sign(n[..., -1:])
    show_normal(n)
    plt.show()

# n = normalize(C)
# show_normal(n)
# plt.show()

x, y, z = np.moveaxis(n, -1, 0)
X = np.fft.fftshift(np.fft.fft2(- x / (z + EPS)))
Y = np.fft.fftshift(np.fft.fft2(- y / (z + EPS)))

V, U = np.mgrid[:h, :w]
V, U = 2 * np.pi * (V / h - .5), 2 * np.pi * (U / w - .5)

H = - 1j * (U * X + V * Y) / (U ** 2 + V ** 2 + 1e-3)
H *= 1 / (1 + 1e-3 * np.sqrt(U ** 2 + V ** 2))
h = np.fft.ifft2(np.fft.ifftshift(H)).real

clip = .05
plt.imshow(np.clip(h, np.quantile(h, clip), np.quantile(h, 1 - clip)))
plt.show()
