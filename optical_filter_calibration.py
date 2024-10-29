import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

phone_model = 'HuaweiP50'

# read R, G, B images
R = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/R_1.JPG'), cv2.IMREAD_COLOR)
G = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/G_1.JPG'), cv2.IMREAD_COLOR)
B = cv2.imread(os.path.join('calibration', phone_model, 'RGB_for_calib/B_1.JPG'), cv2.IMREAD_COLOR)

R = cv2.cvtColor(R, cv2.COLOR_BGR2RGB)
G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

# crop the images
R = R[0:100, 0:100]
G = G[0:100, 0:100]
B = B[0:100, 0:100]

# get the height and width of the image
height, width, _ = R.shape

# convert the images to float32 and normalize them to [0, 1]
R = R.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
B = B.astype(np.float32) / 255.0

# initialize the M_matrices (M means the spectral sensitivity matrix)
# each pixel has a 3x3 matrix
M_matrices = np.zeros((height, width, 3, 3), dtype=np.float32)

# calculate the 3x3 matrix for each pixel
for i in range(height):
    for j in range(width):
        # the RGB response under red light
        I_R = R[i, j]
        # the RGB response under green light
        I_G = G[i, j]
        # the RGB response under blue light
        I_B = B[i, j]
        
        # stack the RGB responses to a 3x3 matrix
        M = np.column_stack((I_R, I_G, I_B))  # each column represents the RGB response under a specific light
        
        # store the 3x3 matrix to the M_matrices
        M_matrices[i, j] = M

# calculate the inverse matrix of the M_matrices with error handling
M_inv_matrices = np.zeros_like(M_matrices)  # Initialize inverse matrices with the same shape

for i in range(height):
    for j in range(width):
        try:
            M_inv_matrices[i, j] = np.linalg.inv(M_matrices[i, j])
        except np.linalg.LinAlgError:
            # Handle singular matrix (non-invertible matrix) by setting it to zero or another fallback
            M_inv_matrices[i, j] = np.zeros((3, 3), dtype=np.float32)
            print(f"Matrix at pixel ({i}, {j}) is singular and could not be inverted.")

# save the M_matrices and M_inv_matrices to the disk
save_path = os.path.join('calibration', phone_model, 'calib_results')
np.save(os.path.join(save_path, 'M_matrices.npy'), M_matrices)
np.save(os.path.join(save_path, 'M_inv_matrices.npy'), M_inv_matrices)

# visualize the M_matrices
# visualize the RGB responses under different light conditions
R_responses = np.zeros((100, 100, 3))
G_responses = np.zeros((100, 100, 3))
B_responses = np.zeros((100, 100, 3))

# extract each pixel's RGB responses
for i in range(100):
    for j in range(100):
        R_responses[i, j] = M_matrices[i, j][:, 0]  # Red Light Response
        G_responses[i, j] = M_matrices[i, j][:, 1]  # Green Light Response
        B_responses[i, j] = M_matrices[i, j][:, 2]  # Blue Light Response

# visualize the responses to Red light
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(R_responses)
plt.title('Red Light RGB Response')

# visualize the responses to Green light
plt.subplot(1, 3, 2)
plt.imshow(G_responses)
plt.title('Green Light RGB Response')

# visualize the responses to Blue light
plt.subplot(1, 3, 3)
plt.imshow(B_responses)
plt.title('Blue Light RGB Response')

plt.show()