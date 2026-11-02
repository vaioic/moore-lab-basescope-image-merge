""" A SimpleITK example demonstrating fast symmetric forces Demons image
    registation. """

import sys
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import skimage
import numpy as np

def srgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]    
    channels = [sitk.VectorIndexSelectionCast(image,i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
    #linear mapping
    I = 1/255.0*(0.2126*channels[0] + 0.7152*channels[1] + 0.0722*channels[2])
    #nonlinear gamma correction
    I = I*sitk.Cast(I<=0.0031308,sitk.sitkFloat32)*12.92 + I**(1/2.4)*sitk.Cast(I>0.0031308,sitk.sitkFloat32)*1.055-0.055
    return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkFloat32)

def command_iteration(sitk_filter):
    """ Callback invoked when the filter progresses. """
    print(f"{sitk_filter.GetElapsedIterations():3} = {sitk_filter.GetMetric():10.5f}")

# if len(sys.argv) < 4:
#     print(
#         "Usage:",
#         sys.argv[0],
#         "<fixedImageFilter> <movingImageFile>",
#         "[initialTransformFile] <outputTransformFile>",
#     )
#     sys.exit(1)

# fixed = srgb2gray(sitk.ReadImage('./images/259270_0_crop.tif'))
# moving = srgb2gray(sitk.ReadImage('./images/261082_0_crop.tif'))

target = skimage.io.imread('./images/259270_0_crop.tif')
target_gray = skimage.color.rgb2gray(target)
target_gray = skimage.util.img_as_ubyte(target_gray)

moving = skimage.io.imread('./images/261082_0_crop.tif')
moving_gray = skimage.color.rgb2gray(moving)
moving_gray = skimage.util.img_as_ubyte(moving_gray)

# First do a coarse alignment
shift, _, _ = skimage.registration.phase_cross_correlation(target_gray, moving_gray)
tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))

corrected = skimage.transform.warp(moving_gray, tform)
corrected = skimage.util.img_as_ubyte(corrected)

print(corrected.shape)
# Crop to the corrected part
corrected = corrected[0:(corrected.shape[0] + int(shift[0])),0:(corrected.shape[1] + int(shift[1]))]
print(corrected.shape[0] + int(shift[1]))
print(corrected.shape)

target_gray = target_gray[0:(target_gray.shape[0]+int(shift[0])),0:(target_gray.shape[1]+int(shift[1]))]

reg_final = np.zeros((target_gray.shape[0], target_gray.shape[1], 3), dtype=np.uint8)
reg_final[..., 0] = target_gray
reg_final[..., 1] = corrected
reg_final[..., 2] = target_gray

plt.imshow(reg_final)
plt.show()


# Start here
target_gray_itk = sitk.GetImageFromArray(target_gray)
corrected_itk = sitk.GetImageFromArray(corrected)

matcher = sitk.HistogramMatchingImageFilter()
if target_gray_itk.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
    matcher.SetNumberOfHistogramLevels(128)
else:
    matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
corrected_itk = matcher.Execute(corrected_itk, target_gray_itk)

# The basic Demons Registration Filter
demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(200)
# Standard deviation for Gaussian smoothing of displacement field
demons.SetStandardDeviations(15)
demons.SetSmoothUpdateField(False)
demons.SetSmoothDisplacementField(True)
demons.SetUseImageSpacing(False)
demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

displacementField = demons.Execute(target_gray_itk, corrected_itk)

print("-------")
print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
print(f" RMS: {demons.GetRMSChange()}")

outTx = sitk.DisplacementFieldTransform(displacementField)
sitk.WriteTransform(outTx, 'transform.txt')

# outTx = sitk.ReadTransform('transform.txt')

if "SITK_NOSHOW" not in os.environ:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_gray_itk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(corrected_itk)

    simg1 = sitk.Cast(sitk.RescaleIntensity(target_gray_itk), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)

    orimg1 = sitk.Cast(sitk.RescaleIntensity(target_gray_itk), sitk.sitkUInt8)
    orimg2 = sitk.Cast(sitk.RescaleIntensity(corrected_itk), sitk.sitkUInt8)
    dimg = sitk.Compose(orimg1, orimg2, orimg1 // 2.0 + orimg2 // 2.0)
    
    try:
        sitk.Show(cimg, "DeformableRegistration1 Composition")
    except:
        nda = sitk.GetArrayFromImage(cimg)
        nda_d = sitk.GetArrayFromImage(dimg)
        plt.subplot(1, 2, 1)
        plt.imshow(nda)
        plt.subplot(1, 2, 2)
        plt.imshow(nda_d)
        plt.show()
