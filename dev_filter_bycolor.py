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

#image = skimage.io.imread('test_ref.tiff')
# print(image.shape)
# print(image.dtype)
# print(np.max(image))
# print(np.min(image))
moving = skimage.io.imread('test_moving.tiff')
moving_mask = slides.filter_image_by_pink(moving)

filtered_moving_gray = skimage.color.rgb2gray(moving)
filtered_moving_gray = (filtered_moving_gray * 255).astype(np.uint8)

ref = skimage.io.imread('test_ref.tiff')
ref_mask = slides.filter_image_by_pink(ref)

filtered_ref_gray = skimage.color.rgb2gray(ref)
filtered_ref_gray = (filtered_ref_gray * 255).astype(np.uint8)

#sift = cv2.ORB_create(nlevels=10, firstLevel=3)
sift = cv2.SIFT_create()

filtered_moving_gray_down = skimage.transform.rescale(filtered_moving_gray, 0.25)
filtered_moving_gray_down = (filtered_moving_gray_down * 255).astype(np.uint8)

moving_mask_down = skimage.transform.rescale(moving_mask, 0.25, preserve_range=True)
moving_mask_down[moving_mask_down > 0] = 255

plt.imshow(moving_mask_down)
plt.show()


kpM, desM = sift.detectAndCompute(filtered_moving_gray_down, moving_mask_down)
kpR, desR = sift.detectAndCompute(filtered_ref_gray, ref_mask)

print(len(kpM))
print(len(kpR))



img_with_keypoints = cv2.drawKeypoints(filtered_moving_gray_down, kpM, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.imshow(img_with_keypoints)
plt.show()

sys.exit()

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors.
# matches = bf.match(desM,desR)
 
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
 
# # Draw first 10 matches.
# img3 = cv2.drawMatches(filtered_moving_gray, kpM, filtered_ref_gray, kpR, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
# plt.imshow(img3),plt.show()

# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
 
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
 
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 # plt.imshow(img3),plt.show()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(desM, desR, k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(filtered_moving_gray, kpM, filtered_ref_gray, kpR,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()


MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kpM[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpR[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
 
    h,w = filtered_moving_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
 
    img2 = cv2.polylines(filtered_ref_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
 
    img3 = cv2.drawMatches(filtered_moving_gray,kpM,filtered_ref_gray,kpR,good,None,**draw_params)
    
    plt.imshow(img3, 'gray'),plt.show()
 
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image[4000:5000,4000:5000,:])
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(filtered_image[4000:5000,4000:5000,:])
# plt.title('Filtered Color (Red)')
# plt.axis('off')

# gray = skimage.color.rgb2gray(filtered_image)
# gray = (gray * 255).astype(np.uint8)
 
