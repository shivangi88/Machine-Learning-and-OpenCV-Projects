"""
The following code was referred and highly inspired by the given below links. 
References:-
1.https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
2.https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
3.https://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereobm-stereobm
"""
UBIT = 'shivangi'; 
import numpy as np; 
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import numpy as np

#Reading Images
img1 = cv2.imread('tsucuba_left.png',0)
img2 = cv2.imread('tsucuba_right.png',0)

#performing SIFT operator 
sift=cv2.xfeatures2d.SIFT_create()
kypt1,comp1=sift.detectAndCompute(img1,None)
kypt2,comp2=sift.detectAndCompute(img2,None)

#Task 2.1 part 1:- drawing keypoints detected in the above lines 
draw1=cv2.drawKeypoints(img1,kypt1,None)
draw2=cv2.drawKeypoints(img2,kypt2,None)
cv2.imwrite('task2_sift1.jpg',draw1)
cv2.imwrite('task2_sift2.jpg',draw2)

#Task 2.1 part 2
#Key Point Matching
bfm=cv2.BFMatcher()
matches=bfm.knnMatch(comp1,comp2,k=2)
good_list=[]
good_not_list = []
for m,n in matches:
    if m.distance<0.75*n.distance:
        good_list.append([m])
        good_not_list.append(m)

#Drawing all the matching  points
matches_knn=cv2.drawMatchesKnn(img1,kypt1,img2,kypt2,good_list,None,flags=2)
cv2.imwrite('task2_matches_knn.jpg',matches_knn)
np.random.shuffle(good_not_list)
#Task 2.2 Find Fundamental Matrix
points1 = np.float32([ kypt1[m.queryIdx].pt for m in good_not_list ]).reshape(-1,1,2)
points2 = np.float32([ kypt2[m.trainIdx].pt for m in good_not_list ]).reshape(-1,1,2)
#Calculation of Fundamental matrix F
F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)
matchesMask = mask.ravel().tolist()
print('Fundamental Matrix:-',F)

#Task 2.3 Creating Epilines of right and left images
# For selecting only inlier points    
points1=points1[mask.ravel()==1]
points2=points2[mask.ravel()==1]
points1 = np.float32([ kypt1[m.queryIdx].pt for m in good_not_list[:11] ]).reshape(-1,1,2)
points2 = np.float32([ kypt2[m.trainIdx].pt for m in good_not_list[:11] ]).reshape(-1,1,2)

#Method to draw EpiLines on the images
def drawlines(img1,img2,lines,points1,points2):
    row,col = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for row,points1,points2 in zip(lines,points1,points2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -row[2]/row[1] ])
        x1,y1 = map(int, [col, -(row[2]+row[0]*col)/row[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(points1.flatten()),5,color,-1)
        img2 = cv2.circle(img2,tuple(points2.flatten()),5,color,-1)
    return img1,img2
# Find epilines for points in right image and draw lines on left image
epiLine1 = cv2.computeCorrespondEpilines(points2.reshape(-1,1,2), 2,F)
epiLine1 = epiLine1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,epiLine1,points1,points2)
# Find epilines for points in left image and draw lines on right image
epiLine2 = cv2.computeCorrespondEpilines(points1.reshape(-1,1,2), 1,F)
epiLine2 = epiLine2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,epiLine2,points1,points1)
cv2.imwrite('task2_epi_left.jpg',img5)
cv2.imwrite('task2_epi_right.jpg',img3)

#Task 2.4 :- Show Disparity map of the images
stereo = cv2.StereoSGBM_create( minDisparity = 0,
 								numDisparities =64,
 								blockSize = 7,
 								uniquenessRatio = 15,
 								speckleWindowSize = 50,
 								speckleRange = 16,
 								disp12MaxDiff = 1,
 								P1 = 8*3*7**2,
 								P2 = 32*3*7**2)
#Calculating disparity value for both the images
disp = stereo.compute(img1,img2).astype(np.float32)/16.0
min_disp = disp.min()
max_disp = disp.max()
#Finding difference in disparity min max values
disparity = np.uint8(255 * (disp - min_disp) / (np.abs(max_disp) - np.abs(min_disp)))
cv2.imwrite('task2_disparity.jpg',disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()