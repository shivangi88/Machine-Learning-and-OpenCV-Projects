"""
The following code was referred and highly inspired by the given below links. 
References:-
1.1:- https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
1.2:- https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
1.3:- https://www.programcreek.com/python/example/70413/cv2.RANSAC
1.4:- https://www.programcreek.com/python/example/70413/cv2.RANSAC
1.5:- https://github.com/cbuntain/stitcher/blob/master/alignImagesRansac.py
"""

UBIT = 'shivangi'; 
import numpy as np; 
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import numpy as np
#Reading Images
image1 = cv2.imread('mountain1.jpg',0)
image2 = cv2.imread('mountain2.jpg',0)

MIN_MATCH_COUNT = 10

sift = cv2.xfeatures2d.SIFT_create()
s1 = sift.detect(image1,None)
s2 = sift.detect(image2,None)

kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

#Task 1.1
kpimg1 = cv2.drawKeypoints(image1, s1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kpimg2 = cv2.drawKeypoints(image2, s2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('task1_sift1.jpg',kpimg1)
cv2.imwrite('task1_sift2.jpg',kpimg2)

#Task 1.2
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,None, flags=2)
cv2.imwrite('task1_matches_knn.jpg', img3)

#Task 1.3:- Homography Matrix "M" calculation
good_1 = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_1.append(m)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_1]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_1]).reshape(-1, 1, 2)
np.random.shuffle(good_1)
#Homographgy Matrix:- M
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print('Homographic matrix:',M)

#Task 1.4 Inliers Matches :- 10 matches only
if len(good_1)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_1 ]).reshape(-1,1,2)
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    matchesMask = mask.ravel().tolist()
    
    h,w = image1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    image2 = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good_1),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask[:11], # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(image1,kp1,image2,kp2,good_1[:11],None,**draw_params)
cv2.imwrite('task1_matches.jpg', img3)

#Task 1.5
#Height and width of input images	
w1,h1 = image2.shape[:2]
w2,h2 = image1.shape[:2]

# Get the image dimesions
img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

# Performing perspective transform on second image
img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

# Resulting dimensions
result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

# Getting images together # Calculate dimensions of match points
[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	
# Output array after affine transformation 
transform_dist = [-x_min,-y_min]
transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 
inverseM = np.linalg.inv(M)
# Warp images to get the resulting image
result_img = cv2.warpPerspective(image1, transform_array.dot(M), 
									(x_max-x_min, y_max-y_min))
result_img[transform_dist[1]:w1+transform_dist[1], 
				transform_dist[0]:h1+transform_dist[0]] =image2

cv2.imwrite('task1_pano.jpg',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


