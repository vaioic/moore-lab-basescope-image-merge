import core_functions
import skimage
from matplotlib import pyplot as plt
import scipy
import numpy as np
import cv2
import tifffile

target = skimage.io.imread("../data/259267_261081/259267_bottom.tif")
moving = skimage.io.imread("../data/259267_261081/261081_bottom.tif")

# Match image sizes
target, moving = core_functions.match_image_size(target, moving)

print(target.shape)

# plt.subplot(1, 2, 1)
# plt.imshow(target)
# plt.subplot(1, 2, 2)
# plt.imshow(moving)
# plt.show()
# exit()

# Perform a coarse alignment
shift = core_functions.get_shift(target, moving)
print(shift)

corrected = core_functions.translate_image(moving, shift)
crop_moving, crop_target = core_functions.match_translated_images(corrected, target, shift)

# merged = core_functions.merge_images(crop_target, crop_moving)
# plt.imshow(merged)
# plt.show()
# exit()

# TODO: Images cannot exceed 32767x32767

hh, ww, _ = crop_target.shape

if ww > 32500:
    crop_target = crop_target[:, :32500, :]
    crop_moving = crop_moving[:, :32500, :]

if hh > 32500:
    crop_target = crop_target[:32500, :, :]
    crop_moving = crop_moving[:32500, :, :]

print(crop_target.shape)

# # For development, crop the resulting image further for speed of testing
# crop_moving = crop_moving[1500:3000, 1500:3000, :]
# crop_target = crop_target[1500:3000, 1500:3000, :]

# merge = core_functions.merge_images(crop_moving, crop_target)
# plt.imshow(merge)
# plt.show()

# Perform fine-tuned alignments
dX, dY, dX_c, dY_c = core_functions.get_fine_shift(crop_moving, crop_target, 60, 400)

# Might want to apply some kind of smoothing filter

# Interpolate the displacements to the whole image
interp_dX = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dX.T, bounds_error=False, fill_value=0.0)
interp_dY = scipy.interpolate.RegularGridInterpolator((dX_c, dY_c), dY.T, bounds_error=False, fill_value=0.0)

# Generate image coordinates
iX = np.arange(0, crop_moving.shape[1])
iY = np.arange(0, crop_moving.shape[0])

iXX, iYY = np.meshgrid(iX, iY, indexing='xy')

# Upsample the displacement field
dX_upsampled = interp_dX((iXX, iYY))
dY_upsampled = interp_dY((iXX, iYY))

print(np.max(np.abs(dX_upsampled)))
print(np.max(np.abs(dY_upsampled)))

# plt.subplot(2, 2, 1)
# plt.imshow(dX)
# plt.colorbar()
# plt.subplot(2, 2, 2)
# plt.imshow(dX_upsampled)
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.imshow(dY)
# plt.colorbar()
# plt.subplot(2, 2, 4)
# plt.imshow(dY_upsampled)
# plt.colorbar()
# plt.show()

corrected_final = core_functions.remap_image(crop_moving, iXX, iYY, dX_upsampled, dY_upsampled)

corrected_final = skimage.util.img_as_float(corrected_final)
crop_target = skimage.util.img_as_float(crop_target)

merged = core_functions.merge_images(crop_target, corrected_final)

skimage.io.imsave("../processed/259267_261081_bottom_merged_falsecolor.png", skimage.util.img_as_ubyte(merged))
# merged_original = core_functions.merge_images(crop_target, crop_moving)

# print(corrected_final.dtype)
# print(crop_target.dtype)

# plt.subplot(1, 2, 1)
# plt.imshow(merged_original)
# plt.subplot(1, 2, 2)
# plt.imshow(merged)
# plt.show()

# Generate merged image
alpha = 0.5

composite = np.zeros(corrected_final.shape, dtype=corrected_final.dtype)
for iC in range(3):
    composite[:, :, iC] = alpha * corrected_final[:, :, iC] + (1 - alpha) * crop_target[:, :, iC]

# plt.imshow(composite)
# plt.show()

# Save output images
tifffile.imwrite("../processed/259267_261081_bottom_merged.tiff", composite, compression="lzw")

skimage.io.imsave("../processed/259267_261081_bottom_merged.png", skimage.util.img_as_ubyte(composite))
