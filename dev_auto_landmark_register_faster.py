import skimage
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import core_functions

target = skimage.io.imread('./images/259270_0_crop.tif')
target_gray = skimage.color.rgb2gray(target)
target_gray = skimage.util.img_as_ubyte(target_gray)

moving = skimage.io.imread('./images/261082_0_crop.tif')
moving_gray = skimage.color.rgb2gray(moving)
moving_gray = skimage.util.img_as_ubyte(moving_gray)

# Condition the images to have the same shape
height_A, width_A = target_gray.shape
height_B, width_B = moving_gray.shape

target_height = max(height_A, height_B)
target_width = max(width_A, width_B)

if height_A < target_height:
    diff = target_height - height_A
    target = np.pad(target, ((0, diff), (0, 0)), 'constant')

if width_A < target_width:
    diff = target_width - width_A
    target = np.pad(target, ((0, 0), (0, diff)), 'constant')

if height_B < target_height:
    diff = target_height - height_B
    moving = np.pad(moving, ((0, diff), (0, 0)), 'constant')

if width_B < target_width:
    diff = target_width - width_B
    moving = np.pad(moving, ((0, 0), (0, diff)), 'constant')

# First do a coarse alignment
shift, _, _ = skimage.registration.phase_cross_correlation(target_gray, moving_gray)
tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))

corrected = skimage.transform.warp(moving_gray, tform)
corrected = skimage.util.img_as_ubyte(corrected)

reg_im = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
reg_im[..., 0] = target_gray
reg_im[..., 1] = corrected
reg_im[..., 2] = target_gray

plt.imshow(reg_im)
plt.show()

landmark_size = [400, 400]  # Height, Width

# Generate a list of landmark grid locations - there might need to be overlap
num_lm_grid = [40, 40]

H, W = moving.shape[:2]

# To avoid errors, the first edge has to at least fit the landmark region - note this might cause errors along the edge but we can live with that for now
lm_grid_spacing_x = np.round(W / num_lm_grid[0])
lm_grid_spacing_y = np.round(H / num_lm_grid[1])

lm_grid_edges_x = np.uint(np.arange(landmark_size[1]/2, W, lm_grid_spacing_x))
lm_grid_edges_y = np.uint(np.arange(landmark_size[0]/2, H, lm_grid_spacing_y))

# lm_grid_edges_x = np.arange(0, W, landmark_size[1])
# lm_grid_edges_y = np.arange(0, H, landmark_size[0])

# For each landmark, crop the coarsely-aligned image and register it

# Calculate the landmark grid centers
xx = lm_grid_edges_x[:-1] + landmark_size[1]/2
yy = lm_grid_edges_y[:-1] + landmark_size[0]/2

xg, yg = np.meshgrid(xx, yy)

dX = np.zeros(xg.shape, dtype=np.float32)
dY = np.zeros(yg.shape, dtype=np.float32)

#ft_target = np.fft.fft2(target_gray)

for iX in tqdm(range(len(lm_grid_edges_x) - 1), position=0):
    for iY in tqdm(range(len(lm_grid_edges_y) - 1), position=1, leave=False):

        template = corrected[lm_grid_edges_y[iY]:(lm_grid_edges_y[iY] + landmark_size[1]), lm_grid_edges_x[iX]:(lm_grid_edges_x[iX] + landmark_size[0])]

        shift = core_functions.faster_template_match(template, target_gray)
        dX[iX, iY] = shift[0]
        dY[iX, iY] = shift[1]

        # res = cv2.matchTemplate(target_gray, template, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # dX[iX, iY] = max_loc[0] - lm_grid_edges_x[iX]
        
        # TODO: Limit the search region - but cap for now
        if dX[iX, iY] > 30:
            dX[iX, iY] = 0

        # dY[iX, iY] = max_loc[1] - lm_grid_edges_y[iY]

        # TODO: Limit the search region - but cap for now
        if dY[iX, iY] > 30:
            dY[iX, iY] = 0
        
        # # The top-left corner of the best match is max_loc
        # top_left = max_loc
        # bottom_right = (top_left[0] + template.shape[0], top_left[1] + template.shape[1])

        # reg_template = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        # reg_template[..., 0] = target_gray
        # reg_template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 1] = template
        # reg_template[..., 2] = target_gray

        # plt.imshow(reg_template)
        # plt.show()

        # print(f"(dx,dy)={dX[iX,iY], dY[iX,iY]}")

# Interpolate the displacement field to the whole image
interp_dX = RegularGridInterpolator((xx, yy), dX, bounds_error=False, fill_value=None)
interp_dY = RegularGridInterpolator((xx, yy), dY, bounds_error=False, fill_value=None)

# Generate a grid for the image coordinates
xi = np.arange(0, corrected.shape[1])
yi = np.arange(0, corrected.shape[0])

X, Y = np.meshgrid(xi, yi)
X_float32 = np.float32(X)
Y_float32 = np.float32(Y)

dX_upsampled = interp_dX((X, Y))
dY_upsampled = interp_dY((X, Y))

dX_upsampled_float32 = np.float32(dX_upsampled)
dY_upsampled_float32 = np.float32(dY_upsampled)
corrected_final = cv2.remap(corrected, X_float32 - dX_upsampled_float32, Y_float32 - dY_upsampled_float32, cv2.INTER_LINEAR)


reg_final = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
reg_final[..., 0] = target_gray
reg_final[..., 1] = corrected_final
reg_final[..., 2] = target_gray

plt.imshow(reg_final)
plt.show()

# # Draw the result on a color version of the target image
# cv2.rectangle(target, top_left, bottom_right, (0, 255, 0), 2)
# #cv2.rectangle(target, (lm_grid_edges_x[curr_iter], lm_grid_edges_x[curr_iter + 1]), (lm_grid_edges_y[curr_iter], lm_grid_edges_y[curr_iter + 1]), (255, 0, 255), 2)
# plt.subplot(1, 2, 1)
# plt.imshow(target)
# plt.subplot(1, 2, 2)
# plt.imshow(template)
# plt.show()

# plt.imshow(template)
# plt.show()