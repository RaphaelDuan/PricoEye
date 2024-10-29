import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

phone_model = 'iPhone13ProMax'

# read R, G, B images
R = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/R_1.JPG'), cv2.IMREAD_COLOR)
G = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/G_1.JPG'), cv2.IMREAD_COLOR)
B = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/B_1.JPG'), cv2.IMREAD_COLOR)

R = cv2.cvtColor(R, cv2.COLOR_BGR2RGB)
G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

# calcluate the average RGB values of the whole image, respectively
R = np.mean(R.reshape(-1, 3), axis=0)[np.newaxis, :] # shape (1, 3)
G = np.mean(G.reshape(-1, 3), axis=0)[np.newaxis, :] # shape (1, 3)
B = np.mean(B.reshape(-1, 3), axis=0)[np.newaxis, :] # shape (1, 3)

# convert the images to float32 and normalize them to [0, 1]
R = R.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
B = B.astype(np.float32) / 255.0

# create the M matrices using the RGB values when only one light source is on
# the first column of M means the response of the camera when only red light is on
M_matrices = np.column_stack((R.flatten(), G.flatten(), B.flatten()))

# 计算 M 矩阵的逆矩阵
try:
    M_inv_matrices = np.linalg.inv(M_matrices)
except np.linalg.LinAlgError:
    raise ValueError("Matrix is singular and could not be inverted.")

# save the M_matrices and its inverse
save_path = os.path.join('calibration', phone_model, 'calib_results')
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'M_matrices.npy'), M_matrices)
np.save(os.path.join(save_path, 'M_inv_matrices.npy'), M_inv_matrices)

# print the M_matrices and its inverse
print("M_matrices:", M_matrices)
print("M_inv_matrices:", M_inv_matrices)

# visualize the M_inv_matrices
plt.figure(figsize=(6, 6))
plt.imshow(M_inv_matrices, cmap='jet', interpolation='none')
plt.title('M_inv_matrices for calibration (iPhone13ProMax)')
plt.colorbar()

# show the values of M_inv_matrices
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{M_inv_matrices[i, j]:.2f}', ha='center', va='center', color='white')

plt.show()