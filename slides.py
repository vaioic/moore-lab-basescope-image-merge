'''
Contains functions used to work with and register slide images
'''

import openslide
from  matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm
import skimage
from pybioimageutils import visualize
import cv2

def filter_image_by_pink(image):

    image_hsv = skimage.color.rgb2hsv(image)

    # target_color_rgb = np.zeros((1, 3), np.uint8)
    # target_color_rgb[:,0] = 120
    # target_color_rgb[:,1] = 33
    # target_color_rgb[:,2] = 67
    # print(target_color_rgb.shape)

    # target_color_hsv = skimage.color.rgb2hsv(target_color_rgb)
    # print(target_color_hsv)

    # Define the lower and upper bounds for the color (example for red)
    # Hue values for red are split between the high and low ends of the spectrum
    lower_red_hue = 0.85
    upper_red_hue = 0.95  # A small range near 0

    # Also, define saturation and value thresholds to avoid grey/white/black pixels
    min_saturation = 0.4
    min_value = 0.3

    # For red specifically, we combine two masks:
    hue_mask_1 = image_hsv[:, :, 0] >= lower_red_hue
    hue_mask_2 = image_hsv[:, :, 0] <= upper_red_hue
    hue_mask = hue_mask_1 & hue_mask_2 # Combine the two red ranges

    # Create saturation and value masks
    saturation_mask = image_hsv[:, :, 1] >= min_saturation
    value_mask = image_hsv[:, :, 2] >= min_value

    # Combine all masks
    color_mask = hue_mask & saturation_mask & value_mask

    color_mask = skimage.morphology.binary_dilation(color_mask, skimage.morphology.disk(4))

    # filtered_image = image.copy()
    # filtered_image[~color_mask] = np.mean(image)
    color_mask = (color_mask * 255).astype(np.uint8)

    return color_mask

def print_slide_properties(slide):
    print("All properties:")

    for prop_name, prop_value in slide.properties.items():
        print(f"{prop_name}: {prop_value}")

def get_registered_regions(moving_slide, ref_slide, shift, region_size, topleft=(0,0), ds_level=2):

    # region_size = (width, height)
    downsample_factor = ref_slide.level_downsamples[ds_level]

    #Read the reference slide
    ref = ref_slide.read_region(topleft, ds_level, size=region_size)
    ref = np.array(ref.convert('RGB'))

    # Read the corrected image
    moving = moving_slide.read_region(
        (topleft[0] - shift[1], topleft[1] - shift[0]),
        ds_level,
        size=region_size
    )
    moving = np.array(moving.convert('RGB'))

    return ref, moving



def calculate_shift(moving_slide, ref_slide, ds_level=2):
    '''
    Calculates the pixel shift of a slide image. Returns the shift in the original image resolution.
    
    :param moving_slide: Description
    :param ref_slide: Description
    :param ds_level: Description
    '''

    moving = moving_slide.read_region((0,0), ds_level, ref_slide.level_dimensions[ds_level])
    moving = np.array(moving.convert('L'))


    ref = ref_slide.read_region((0,0), ds_level, ref_slide.level_dimensions[ds_level])
    ref = np.array(ref.convert('L'))
    
    shift, _, _= skimage.registration.phase_cross_correlation(ref, moving)

    print(shift)

    ds_factor = ref_slide.level_downsamples[ds_level]

    # Return the shift, corrected to the original resolution and as integers as required by read_region.
    shift = (int(shift[0] * ds_factor), int(shift[1] * ds_factor))
    print(f"Corrected {shift}")

    return shift

def register_tiled_image(moving_slide, ref_slide, shift, ds_level=2, tile_size=256):

     # Calculate number of tiles
    width, height = ref_slide.level_dimensions[ds_level]    
    downsample_factor = ref_slide.level_downsamples[ds_level]
    
    num_cols = int(np.ceil(width / tile_size))
    num_rows = int(np.ceil(height / tile_size))

    print(f"Num Rows: {num_rows}, Num Cols: {num_cols}")

    for row in tqdm(range(num_rows)):
        for col in range(num_cols):
            
            x0 = int(col * tile_size * downsample_factor)
            y0 = int(row * tile_size * downsample_factor)

            tile_width = min(tile_size, width - col * tile_size)
            tile_height = min(tile_size, height - row * tile_size)
                
            #Read the reference slide
            ref = ref_slide.read_region(
                    (x0, y0),
                    ds_level,
                    size=(tile_width, tile_height)
            )
            ref = np.array(ref.convert('RGB'))
           
            # Read the corrected image
            moving = moving_slide.read_region(
                (x0 - shift[1], y0 - shift[0]),
                ds_level,
                size=(tile_width, tile_height)
            )
            moving = np.array(moving.convert('RGB'))
            
            output = visualize.composite(ref, moving, normalize_images=False)
            # output = (output * 255).astype(np.uint8)
            output = output.astype(np.uint8)

            # print((np.max(output), np.min(output)))

            yield output


def get_region(slide, ds_level=2):
    '''
    Returns an image region interactively

    
    :param slide: Description
    :param ds_level: Description
    '''

    # Initialize variables to hold the final selection
    topleft = (0,0)
    bottomright = (0,0)

    # Define callback function for rectangle selection
    def onselect_callback(eclick, erelease):
        nonlocal topleft, bottomright

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        topleft = (min(x1,x2), min(y1,y2))
        bottomright = (max(x1,x2), max(y1,y2))

    # Read in the image at the current downsample level
    img = slide.read_region((0,0), ds_level, slide.level_dimensions[ds_level])

    fig, ax = plt.subplots()
    ax.imshow(img)
    selector = RectangleSelector(
        ax, onselect_callback,
        useblit=True,
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)
    plt.title('Drag to select a region. Close the window when done.')
    plt.show()

    # Return coordinates scaled by the downsample level
    downsample_factor = slide.level_downsamples[ds_level]
    topleft = np.floor(np.array(topleft) * downsample_factor)
    bottomright = np.ceil(np.array(bottomright) * downsample_factor)

    # Threshold the coordinates so they always return a valid region
    topleft[topleft < 0] = 0
    region_size = (bottomright - topleft) + 1  #[Width, Height]

    #region_size[0] = min(region_size[0], slide.dimensions[0])
    #region_size[1] = min(region_size[1], slide.dimensions[1])
    
    return tuple(topleft.astype(int)), tuple(region_size.astype(int))

def compute_descriptors(moving_gray, ref_gray, landmarks, window_size=31):

    # 1. Convert manual landmarks (x, y) to OpenCV KeyPoint objects
    # The 'size' parameter defines the diameter of the neighborhood considered.
    kp = []
    for x, y in landmarks:
        # Parameters: x, y, size, angle, response, octave, class_id
        # Size can be adjusted based on the scale of features you expect
        keypoint = cv2.KeyPoint(x, y, window_size) 
        kp.append(keypoint)

    # 2. Initialize a descriptor extractor (e.g., SIFT or ORB)
    # SIFT is a common choice for its robustness
    try:
        descriptor_extractor = cv2.SIFT_create()
    except AttributeError:
        # Handle cases where SIFT might be in xfeatures2d (older OpenCV versions)
        print("SIFT not found directly, trying xfeatures2d.SIFT_create()")
        descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    # 3. Compute descriptors for the created keypoints
    # The compute method returns the same keypoints (potentially refined) and the descriptors
    kpM, desM = descriptor_extractor.compute(moving_gray, kp)
    kpR, desR = descriptor_extractor.compute(ref_gray, kp)

    if desM is None:
        print("No descriptors were computed.")
        return kp, None

    return kpM, desM, kpR, desR


def compute_descriptors_skimage(moving_gray, ref_gray, landmarks, window_size=31):

    # 1. Convert manual landmarks (x, y) to OpenCV KeyPoint objects
    # The 'size' parameter defines the diameter of the neighborhood considered.
    kp = []
    for x, y in landmarks:
        # Parameters: x, y, size, angle, response, octave, class_id
        # Size can be adjusted based on the scale of features you expect
        keypoint = cv2.KeyPoint(x, y, window_size) 
        kp.append(keypoint)

    # 2. Initialize a descriptor extractor (e.g., SIFT or ORB)
    # SIFT is a common choice for its robustness
    try:
        descriptor_extractor = cv2.SIFT_create()
    except AttributeError:
        # Handle cases where SIFT might be in xfeatures2d (older OpenCV versions)
        print("SIFT not found directly, trying xfeatures2d.SIFT_create()")
        descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    # 3. Compute descriptors for the created keypoints
    # The compute method returns the same keypoints (potentially refined) and the descriptors
    kpM, desM = descriptor_extractor.compute(moving_gray, kp)
    kpR, desR = descriptor_extractor.compute(ref_gray, kp)

    if desM is None:
        print("No descriptors were computed.")
        return kp, None

    return kpM, desM, kpR, desR