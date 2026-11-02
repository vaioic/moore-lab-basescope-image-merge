from numpy.fft import ifft2, fft2
from numpy import conj, pad, argmax, unravel_index, abs
from matplotlib import pyplot as plt
import pyfftw
import numpy as np
import skimage
import scipy
from cv2 import remap, INTER_LINEAR
from tqdm import tqdm

def match_image_size(imageA, imageB):
    """Returns images cropped to the smallest matching dimensions

    Only the row and columns of the images are cropped

    Parameters
    ----------
    imageA : array-like
        First image
    imageB : array-like
        Second image

    Returns
    -------
    imageA_cropped : array-like
        Cropped first image
    imageB_cropped : array-like
        Cropped second image
    """

    # Find the smallest value of each dimension
    min_lengths = np.minimum(imageA.shape, imageB.shape)

    # print(f"Image A shape:{imageA.shape}")
    # print(f"Image B shape:{imageB.shape}")
    # print(f"Minimum shape:{min_lengths}")
    
    # Crop the images
    imageA_cropped = imageA[:min_lengths[0],:min_lengths[1],...]
    imageB_cropped = imageB[:min_lengths[0],:min_lengths[1],...]

    # print(f"Image A shape (cropped):{imageA_cropped.shape}")
    # print(f"Image B shape (cropped):{imageB_cropped.shape}")

    return imageA_cropped, imageB_cropped

def get_shift(moving, target, downsample_factor=2):
    """Register images with optional downsampling

    This function returns the pixel shift between the moving and
    target images. The function optionally downsamples the images
    to reduce computation load.

    Parameters
    ----------
    moving : array-like
        Moving image
    target : array-like
        Target (or reference) image
    downsample_factor : int, optional
        Factor to downsample images, by default 2

    Returns
    -------
    shift : list of [int, int]
        [row, col] pixel shifts
    """

    if len(moving.shape) == 3:
        # Convert to grayscale
        moving = skimage.color.rgb2gray(moving)
    
    if len(target.shape) == 3:
        target = skimage.color.rgb2gray(target)

    if not (downsample_factor is None or downsample_factor == 1):
        moving = moving[::downsample_factor, ::downsample_factor]
        target = target[::downsample_factor, ::downsample_factor]
    else:
        downsample_factor = 1

    shift = faster_xcorrreg(moving, target)

    return np.multiply(shift, downsample_factor)

def faster_xcorrreg(moving, target, debug_plot=False):
    """Calculate the pixel shift using the fftw algorithm

    This function calculates the pixel shift by calculating the phase correlation 
    using a normalized Fourier Transform. The (inverse) Fourier Transforms are
    calculated using the ``pyfftw`` package.

    Parameters
    ----------
    moving : array-like
        The moving image
    target : array-like
        The target (or reference) image

    Returns
    -------
    shift : list of [int, int]
        The [row, col] of the shift between the two images

    Raises
    ------
    ValueError
        If the images do not have the same shapes.
    ValueError
        If the images are not 2D (i.e., grayscale)
    """

    if not (moving.shape == target.shape):
        raise ValueError("Both images must have the same shape.")
    
    if (len(moving.shape) != 2) or (len(target.shape) != 2):
        raise ValueError("Images must be 2D (grayscale or normalized)")
    
    # Check if all values are zero (occurs if images were translated but not cropped)
    if not ((moving > 0).any() or (target > 0).any()):
        return [0, 0]

    # Byte-align the images for computational efficiency
    moving = pyfftw.byte_align(moving, n = 16)
    target = pyfftw.byte_align(target, n = 16)

    xcorr = pyfftw.interfaces.numpy_fft.fft2(moving) * np.conjugate(pyfftw.interfaces.numpy_fft.fft2(target))
    phase_xcorr = pyfftw.interfaces.numpy_fft.ifft2(xcorr / np.abs(xcorr))    
    phase_xcorr = pyfftw.interfaces.numpy_fft.fftshift(phase_xcorr)

    if debug_plot:
        plt.imshow(np.abs(phase_xcorr))
        plt.show()

    # Find the location of the maximum correlation
    idx_max = np.argmax(np.abs(phase_xcorr))
    row, col = np.unravel_index(idx_max, target.shape)

    # Need to recalculate the pixel shifts to take into account the fftshift
    H, W = target.shape
    y_shift = row - H // 2
    x_shift = col - W // 2
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(moving)
    # plt.subplot(1, 2, 2)
    # plt.imshow(target)
    # plt.show()

    # plt.figure()
    # plt.imshow(np.abs(phase_xcorr))
    # plt.show()

    return [int(y_shift), int(x_shift)]
    

def translate_image(moving, shift):

    tform = skimage.transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
    corrected = skimage.transform.warp(moving, tform)

    return corrected

def match_translated_images(moving, target, shift):
    # note Shift = [y_shift, x_shift] and -ve shift moves up and left
    print(f"Moving shape (start): {moving.shape}")

    if shift[1] < 0:
        moving = moving[:, :(moving.shape[1] + shift[1]), ...]
        target = target[:, :(target.shape[1] + shift[1]), ...]
        print(f"Moving shape (after shift[0]): {moving.shape}")
    elif shift[1] > 0:
        moving = moving[:, shift[1]:, ...]
        target = target[:, shift[1]:, ...]
    
    if shift[0] < 0:
        moving = moving[:(moving.shape[0] + shift[0]), :, ...]
        target = target[:(target.shape[0] + shift[0]), :, ...]
    elif shift[0] > 0:
        moving = moving[shift[0]:, :, ...]
        target = target[shift[0]:, :, ...]

    try:
        assert moving.shape == target.shape
    except:
        print(f"Final images were not the same shape.")
        print(f"Moving: {moving.shape} | Target {target.shape}")
        exit()

    return moving, target

def merge_images(imageA, imageB):
    """Generate a composite images to evaluate differences

    Image A will be displayed in magenta and image B is displayed in green. 
    If the images are identical and properly overlapped, the image will
    appear as gray.

    Parameters
    ----------
    imageA : array-like
        First image (displayed in magenta)
    imageB : array-like
        Second image (displayed in green)

    Returns
    -------
    composite: ndarray
        Merged image
    """

    if len(imageA.shape) == 3:
        imageA = skimage.color.rgb2gray(imageA)
    
    if len(imageB.shape) == 3:
        imageB = skimage.color.rgb2gray(imageB)

    composite = np.zeros((imageA.shape[0], imageA.shape[1], 3), dtype=imageA.dtype)
    composite[:, :, 0] = imageA
    composite[:, :, 1] = imageB
    composite[:, :, 2] = imageA

    return composite

def generate_grid_centers(image, num_points, offset=(50,50)):

    height, width, _ = image.shape

    if isinstance(num_points, int):
        num_rows = num_points
        num_cols = num_points
    else:
        num_rows, num_cols = num_points

    grid_centers_x = np.linspace(offset[1], width - offset[1], num_cols, dtype=int)    
    grid_centers_y = np.linspace(offset[0], height - offset[0], num_rows, dtype=int)

    return grid_centers_x, grid_centers_y

def get_fine_shift(moving, target, num_points, window):

    grid_centers_x, grid_centers_y = generate_grid_centers(moving, num_points, offset=(window/2, window/2))

    displacement_X = np.zeros((len(grid_centers_y), len(grid_centers_x)), dtype=np.float32)
    displacement_Y = np.zeros((len(grid_centers_y), len(grid_centers_x)), dtype=np.float32)

    window_half_size = int(window/2)

    moving = skimage.color.rgb2gray(moving)
    target = skimage.color.rgb2gray(target)

    # Match intensities by histogram to see if it generates better fits
    moving = skimage.exposure.match_histograms(moving, target)
    
    col = 0
    for xc in tqdm(grid_centers_x):
        row = 0
        for yc in grid_centers_y:
            curr_moving = moving[(yc - window_half_size):(yc + window_half_size), (xc - window_half_size):(xc + window_half_size)]
            curr_target = target[(yc - window_half_size):(yc + window_half_size), (xc - window_half_size):(xc + window_half_size)]

            # plt.subplot(1, 2, 1)
            # plt.imshow(curr_moving)
            # plt.subplot(1, 2, 2)
            # plt.imshow(curr_target)
            # plt.show()

            shift = faster_xcorrreg(curr_moving, curr_target)

            # print(shift)
            if abs(shift[0]) > 12:
                shift[0] = 0
            if abs(shift[1]) > 12:
                shift[1] = 0


            displacement_X[row, col] = shift[1]
            displacement_Y[row, col] = shift[0]
            
            row += 1

            # # Test
            # if abs(shift[0]) > 5 or abs(shift[1]) > 5:
            #     print(f"{(row, col)} = {(shift[1], shift[0])}")
            #     corrected = translate_image(curr_moving, shift)
            #     # crop_moving, crop_target = match_translated_images(corrected, curr_target, shift)

            #     merged = merge_images(curr_target, corrected)
            #     plt.imshow(merged)
            #     plt.show()
        col += 1

    return displacement_X, displacement_Y, grid_centers_x, grid_centers_y

def remap_image(image, XX, YY, dXX, dYY):

    # Convert the inputs into the format required by openCV
    XX = np.float32(XX)
    YY = np.float32(YY)

    dXX = np.float32(dXX)
    dYY = np.float32(dYY)
    
    corrected = np.zeros(image.shape, dtype=image.dtype)
    for iC in range(image.shape[2]):
        curr_channel = skimage.util.img_as_ubyte(image[:, :, iC])  
        curr_corrected = remap(curr_channel, XX + dXX, YY + dYY, INTER_LINEAR)
        corrected[:, :, iC] = skimage.util.img_as_float(curr_corrected)

    return corrected

def fast_template_match(template, target, target_istransformed=False):
    """Fast template matching using Fourier Transforms

    :param template: Template image
    :param target: Target image
    """

    # Pad the template to match the target size
    if not (template.shape == target.shape):
        template_H, template_W = template.shape[:2]
        target_H, target_W = target.shape[:2]   

        template_padded = pad(template, [(0, target_H - template_H), (0, target_W - template_W)], mode='constant', constant_values=0)
    else:
        template_padded = template

    # Compute the phase correlation
    if not target_istransformed:
        xcorr = fft2(target) * conj(fft2(template_padded))
    else:
        xcorr = target * conj(fft2(template_padded))
    phase_xcorr = ifft2(xcorr / abs(xcorr))
    
    # Find the location of maximum correlation
    idx_max = argmax(phase_xcorr)
    row, col = unravel_index(idx_max, target.shape)

    return (row, col)

def faster_template_match(template, target, target_istransformed=False):
    """Faster template matching using pyfftw

    :param template: Template image
    :param target: Target image
    """

    # Pad the template to match the target size
    template_H, template_W = template.shape[:2]
    target_H, target_W = target.shape[:2]   

    template_padded = pad(template, [(0, target_H - template_H), (0, target_W - template_W)], mode='constant', constant_values=0)

    template_padded = pyfftw.byte_align(template_padded, n=16)

    # Compute the phase correlation
    if not target_istransformed:
        target = pyfftw.byte_align(target, n=16)
        xcorr = pyfftw.interfaces.numpy_fft.fft2(target) * conj(pyfftw.interfaces.numpy_fft.fft2(template_padded))
    else:
        xcorr = target * conj(pyfftw.interfaces.numpy_fft.fft2(template_padded))
    phase_xcorr =  pyfftw.interfaces.numpy_fft.ifft2(xcorr / abs(xcorr))
    
    # Find the location of maximum correlation
    idx_max = argmax(phase_xcorr)
    row, col = unravel_index(idx_max, target.shape)

    return (row, col)
