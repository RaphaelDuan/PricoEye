import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义一个函数来简化重复的显示过程
def display_channel(intensity, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(intensity, cmap='gray')
    plt.title(title)
    plt.show()

  
# load M_inv_matrices
phone_model = 'iPhone13ProMax'
M_inv_matrices = np.load(os.path.join('calibration', phone_model, 'calib_results/M_inv_matrices.npy'))

# load the captured image
FILE_NAME = './example_images/IMG_2643.JPG'
SAVE_DIR = FILE_NAME.replace('.JPG', '')
os.makedirs(SAVE_DIR, exist_ok=True)

I = cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)

# crop the image (from 750 to 2500)
I = I[750:2500, 750:2500]  # shape in (1750, 1750, 3)

# normalize the image to [0, 1]
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I = I.astype(np.float32) / 255.0

# initialize the light intensity matrix, shape in (1750, 1750, 3), 3 means RGB channels
height, width, _ = I.shape

# use einsum to recover the real light intensities
L = np.einsum('ijk,kl->ijl', I, M_inv_matrices)

# save and display the captured RGB image
original_image_path = os.path.join(SAVE_DIR, 'captured_RGB.bmp')
cv2.imwrite(original_image_path, cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 10))
plt.imshow(I)
plt.title('Captured RGB Image')
plt.show()

# save the display the recovered RGB image
inversed_image_path = os.path.join(SAVE_DIR, 'recovered_RGB.bmp')
cv2.imwrite(inversed_image_path, cv2.cvtColor((L * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 10))
plt.imshow(L)
plt.title('Inversed RGB Image')
plt.show()


# ------------------------- Split RGB Intensities -------------------------------------
# split the recovered RGB image into R, G, B channels
Red_intensity = L[:, :, 0]
Green_intensity = L[:, :, 1]
Blue_intensity = L[:, :, 2]

# save and display the grayscale images of R, G, B channels
Red_intensity_path = os.path.join(SAVE_DIR, 'R_intensity.bmp')
Green_intensity_path = os.path.join(SAVE_DIR, 'G_intensity.bmp')
Blue_intensity_path = os.path.join(SAVE_DIR, 'B_intensity.bmp')

# save the intensities from different light sources as grayscale images, *.bmp format without any compression
cv2.imwrite(Red_intensity_path, (Red_intensity * 255).astype(np.uint8))
cv2.imwrite(Green_intensity_path, (Green_intensity * 255).astype(np.uint8))
cv2.imwrite(Blue_intensity_path, (Blue_intensity * 255).astype(np.uint8))

# 分别显示 R、G、B 通道的灰度图像
display_channel(Red_intensity, 'Red Intensity')
display_channel(Green_intensity, 'Green Intensity')
display_channel(Blue_intensity, 'Blue Intensity')


# ------------------------- Generate Mask -------------------------------------
# generate the mask image (all pixels are set to 0, the shape is the same as the original image)
# generate ones matrix with the same shape as the original image, dtype is np.uint8
mask = np.zeros((height, width))
# set target area to 1
mask[500:1250, 500:1250] = 1

# save as bmp format without any compression
mask_path = os.path.join(SAVE_DIR, 'mask.bmp')
cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
print(mask.shape)

# display the mask image
display_channel(mask, 'Mask')

print('Done!')


# ------------------------- Photometric Stereo -------------------------------------
from tools.photometric_stereo.photostereo_py.script.photostereo import photometry

IMAGES = 3
root_fold = SAVE_DIR + "/"
suffix_name = "_intensity"
format = ".bmp"
light_manual = True

#Load input image array
image_array = []
for id in ["R", "G", "B"]:
    try:
        filename = root_fold + str(id) + str(suffix_name) + format
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image_array.append(im)
    except cv2.error as err:
        print(err)

myps = photometry(IMAGES, True)

if light_manual:
    # radius is 3cm, distance is 13cm, slant is 12.99 degrees ~ 13 degrees
    # 设置光源的 slants 和 tilts
    slants = [13, 13, 13]  # R, G, B 光源的倾斜角分别是 13, 13, 13
    tilts = [60, 180, -60]  # R, G, B 方位角分别是 60, 180, -60

    myps.setlmfromts(tilts, slants)
    print(myps.settsfromlm())
else:
    # LOADING LIGHTS FROM FILE
    fs = cv2.FileStorage(root_fold + "LightMatrix.yml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("Lights")
    light_mat = fn.mat()
    myps.setlightmat(light_mat)
    #print(myps.settsfromlm())

tic = time.process_time()
mask = cv2.imread(root_fold + "mask" + format, cv2.IMREAD_GRAYSCALE)
normal_map = myps.runphotometry(image_array, np.asarray(mask, dtype=np.uint8))
normal_map = cv2.normalize(normal_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
albedo = myps.getalbedo()
albedo = cv2.normalize(albedo, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
gauss = myps.computegaussian()
med = myps.computemedian()

cv2.imwrite(os.path.join(SAVE_DIR, 'normal_map.png'), normal_map)
cv2.imwrite(os.path.join(SAVE_DIR, 'albedo.png'), albedo)
cv2.imwrite(os.path.join(SAVE_DIR, 'gauss.png'), gauss)
cv2.imwrite(os.path.join(SAVE_DIR, 'med.png'), med)

# cv2.imwrite('albedo.png',albedo)
# cv2.imwrite('gauss.png',gauss)
# cv2.imwrite('med.png',med)

toc = time.process_time()
print("Process duration: " + str(toc - tic))

# TEST: 3d reconstruction
myps.computedepthmap()
myps.computedepth2()
myps.display3dobj()
cv2.imshow("normal", normal_map)
cv2.imshow("mean", med)
cv2.imshow("gauss", gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()