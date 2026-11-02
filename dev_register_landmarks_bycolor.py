# Perform coarse registration

import openslide
import slides
from tifffile import imwrite
import numpy as np
from pybioimageutils import visualize
import skimage
from matplotlib import pyplot as plt
import cv2
import sys


# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

# image_A_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/259270.svs"

# image_B_path = "/primary/projects/moore/vari-core-generated-data/PBC-Aperio Images/261082.svs"

# image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

# image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

# # ---Begin processing---
# # Create slide objects
# slide_A = openslide.OpenSlide(image_A_path)
# slide_B = openslide.OpenSlide(image_B_path)

# #shift = slides.calculate_shift(slide_B, slide_A, ds_level=2)
# shift = (-80, 592)

# ref, moving = slides.get_registered_regions(slide_B, slide_A, shift, (5000, 5000), topleft=(10000,5000), ds_level=0)

# imwrite('test_ref.tiff', ref)
# imwrite('test_moving.tiff', moving)

ref = skimage.io.imread('test_ref.tiff')
moving = skimage.io.imread('test_moving.tiff')

# Get regions
ref_lab = skimage.color.rgb2lab(ref)
moving_lab = skimage.color.rgb2lab(moving)

target_color_rgb = np.zeros((1, 1, 3))
target_color_rgb[0, 0, 0] = 154.0/255.0
target_color_rgb[0, 0, 1] = 50.0/255.0
target_color_rgb[0, 0, 2] = 70./255.0

target_color_lab = skimage.color.rgb2lab(target_color_rgb)

match_radius = 30
black_thresh = 30

mask_ref = ((ref_lab[:, :, 1] - target_color_lab[:, :, 1]) ** 2 + (ref_lab[:, :, 2] - target_color_lab[:, :, 2]) ** 2) <= (match_radius ** 2)
mask_ref_black = moving_lab[:, :, 0] <= black_thresh
mask_ref_final = mask_ref | mask_ref_black
mask_ref_final = skimage.morphology.remove_small_holes(mask_ref_final, 50)

filtered_ref = np.zeros(ref.shape, dtype=np.uint8)

for c in range(3):
    curr_channel = ref[:, :, c].copy()
    curr_channel[~mask_ref_final] = np.mean(curr_channel)
    filtered_ref[:, :, c] = curr_channel  # Use the inverse mask (~) to set other pixels to black

target_color_rgb = np.zeros((1, 1, 3))
target_color_rgb[0, 0, 0] = 120.0/255.0
target_color_rgb[0, 0, 1] = 61.0/255.0
target_color_rgb[0, 0, 2] = 83.0/255.0

mask_moving = ((moving_lab[:, :, 1] - target_color_lab[:, :, 1]) ** 2 + (moving_lab[:, :, 2] - target_color_lab[:, :, 2]) ** 2) <= (match_radius ** 2)

mask_moving_black = moving_lab[:, :, 0] <= black_thresh
mask_moving_final = mask_moving | mask_moving_black
mask_moving_final = skimage.morphology.remove_small_holes(mask_moving_final, 50)
filtered_moving = np.zeros(moving.shape, dtype=np.uint8)

for c in range(3):
    curr_channel = moving[:, :, c].copy()
    curr_channel[~mask_moving_final] = np.mean(curr_channel)
    filtered_moving[:, :, c] = curr_channel  # Use the inverse mask (~) to set other pixels to black

# Initiate ORB detector
orb = cv2.SIFT_create(2000)
 
filtered_moving_gray = skimage.color.rgb2gray(filtered_moving)
filtered_moving_gray = (filtered_moving_gray * 255).astype(np.uint8)

# find the keypoints with ORB
kp1, des1 = orb.detectAndCompute(filtered_moving_gray,None)
 
# # draw only keypoints location,not size and orientation
# imgout = cv2.drawKeypoints(filtered_moving, kp1, None, color=(0,255,0), flags=0)
# plt.imshow(imgout), plt.show()

# sys.exit()
filtered_ref_gray = skimage.color.rgb2gray(filtered_ref)
filtered_ref_gray = (filtered_ref_gray * 255).astype(np.uint8)

# find the keypoints with ORB
kp2, des2 = orb.detectAndCompute(filtered_ref_gray,None)
 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good)>=MIN_MATCH_COUNT:

    ref_gray = skimage.color.rgb2gray(ref)
    ref_gray = (ref_gray * 255).astype(np.uint8)

    moving_gray = skimage.color.rgb2gray(moving)
    moving_gray = (moving_gray * 255).astype(np.uint8)  

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
 
    h,w = moving_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
 
    img2 = cv2.polylines(ref_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
 
    img3 = cv2.drawMatches(moving_gray,kp1,img2,kp2,good,None,**draw_params)
    
    plt.imshow(img3, 'gray'),plt.show()
 
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


# # draw only keypoints location,not size and orientation
# imgout = cv2.drawKeypoints(filtered_ref, kp2, None, color=(0,255,0), flags=0)
# plt.imshow(imgout), plt.show()


# ref_lab = skimage.color.rgb2lab(ref)
# moving_lab = skimage.color.rgb2lab(moving)

# target_color_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
# target_color_rgb[0, 0, 0] = 63
# target_color_rgb[0, 0, 1] = 33
# target_color_rgb[0, 0, 2] = 67

# target_color_hsv = skimage.color.rgb2hsv(target_color_rgb)
# print(target_color_hsv)
# # Define the lower and upper bounds for the color (example for red)
# # Hue values for red are split between the high and low ends of the spectrum
# lower_red_hue = 0.35
# upper_red_hue = 0.30  # A small range near 0

# # Also, define saturation and value thresholds to avoid grey/white/black pixels
# min_saturation = 0.4
# min_value = 0.3

# # For red specifically, we combine two masks:
# hue_mask_1 = ref_hsv[:, :, 0] >= lower_red_hue
# hue_mask_2 = ref_hsv[:, :, 0] <= upper_red_hue
# hue_mask = hue_mask_1 | hue_mask_2 # Combine the two red ranges

# # Create saturation and value masks
# saturation_mask = ref_hsv[:, :, 1] >= min_saturation
# value_mask = ref_hsv[:, :, 2] >= min_value

# # Combine all masks
# color_mask = hue_mask & saturation_mask & value_mask

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ref[2500:3000,2500:3000,:])
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_ref[2500:3000,2500:3000,:])
plt.title('Filtered Color (Red)')
plt.axis('off')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(moving[2500:3000,2500:3000,:])
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_moving[2500:3000,2500:3000,:])
plt.title('Filtered Color (Red)')
plt.axis('off')

plt.show()

sys.exit()

ref_gray = skimage.color.rgb2gray(ref)
ref_gray = (ref_gray * 255).astype(np.uint8)

moving_gray = skimage.color.rgb2gray(moving)
moving_gray = (moving_gray * 255).astype(np.uint8)

output = visualize.composite(ref, moving, normalize_images=False)
# output = (output * 255).astype(np.uint8)
output = output.astype(np.uint8)
# imwrite('./export/merged_registered.tif', output)

fig, ax = plt.subplots()
ax.set_title('Click on points to select them')
# Set the 'picker' property to a tolerance value (e.g., 5 points)
ax.imshow(output, picker=5) 

selected_points = []

def onclick(event):
    """
    Event handler for mouse clicks.
    
    Draws a circle at the click location and updates the plot.
    """
    if event.xdata is not None and event.ydata is not None:
        # Get the coordinates
        x, y = event.xdata, event.ydata
        selected_points.append((x, y))
        
        print(f"Point selected: ({x:.2f}, {y:.2f})")
        
        # Draw a red circle at the clicked point
        circle = plt.Circle((x, y), radius=31, color='r', fill=False)
        ax.add_patch(circle)
        
        # Refresh the display to show the new circle
        plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

kpM, desM, kpR, desR = slides.compute_descriptors(moving_gray, ref_gray, selected_points, window_size=31)

print(desM.dtype)

# imwrite('./export/ref_registered_gray.tif', ref_gray)
# imwrite('./export/moving_registered_gray.tif', moving_gray)

# output = visualize.composite(ref, moving, normalize_images=False)
# # output = (output * 255).astype(np.uint8)
# output = output.astype(np.uint8)
# imwrite('./export/merged_registered.tif', output)
# 4. Initialize the Brute Force Matcher
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# # 5. Match the descriptors
# matches = bf.match(desM, desR)

# # 6. Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(desR,desM,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 5

if len(good)>MIN_MATCH_COUNT:

    # 7. Draw the top N matches
    img_matches = cv2.drawMatches(ref_gray, kpR, moving_gray, kpM, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    src_pts = np.float32([kpM[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = ref_gray.shape[:2]
    img_registered = cv2.warpPerspective(moving_gray, M, (w, h))

    overlay = cv2.addWeighted(ref_gray, 0.5, img_registered, 0.5, 0)
    cv2.imshow("Registered Image", overlay)

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None