# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:22:50 2018

@author: shivangi
"""
#Reference:- http://aishack.in/tutorials/convolutions/
#https://plot.ly/python/normalization/
import cv2
import math


img = cv2.imread("C:\\Users\\yashi\\.spyder-py3\\task1.png", 0)
L = img.shape[0] 
K = img.shape[1]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite()

sbl_x =[[1,0,-1],[2,0,-2],[1,0,-1]]
sbl_y =[[1,2,1],[0,0,0],[-1,-2,-1]]
temp_x=img.copy()
temp_y=img.copy()
temp_xy=img.copy()
maxval_x=0
minval_x=0
maxval_y=0
minval_y=0
maxval_xy=0
minval_xy=0

#method to calculate edges of x direction
def sobelx(img, sbl_x):
    Gausx = 0
    img[0,:] = img[:,0] = img[:,-1] =  img[-1,:] = 0
    for i in range(1, L-2):  
        for j in range (1,K-2):
            Gausx = ( sbl_x[0][2]* img[i-1][j-1]) + ( sbl_x[0][1]* img[i-1][j]) + \
             (sbl_x[0][0] * img[i-1][j+1]) + (sbl_x[1][2] * img[i][j-1]) + \
             (sbl_x[1][1] * img[i][j]) + (sbl_x[1][0] * img[i][j+1]) + \
             (sbl_x[2][2]* img[i+1][j-1]) + (sbl_x[2][1]* img[i+1][j]) + \
             (sbl_x[2][0]* img[i+1][j+1])
            temp_x[i-1][j-1] = Gausx
    return temp_x
#method to calculate edges of y direction
def sobely(img, sbl_y):
    Gausy = 0
    img[0,:] = img[:,0] = img[:,-1] =  img[-1,:] = 0
    for i in range(1, L-2):  
        for j in range (1,K-2):
            Gausy = (sbl_y[0][2] * img[i-1][j-1]) + (sbl_y[0][1] * img[i-1][j]) + \
             (sbl_y[0][0] * img[i-1][j+1]) + (sbl_y[1][2] * img[i][j-1]) + \
             (sbl_y[1][1]* img[i][j]) + (sbl_y[1][0] * img[i][j+1]) + \
             (sbl_y[2][2] * img[i+1][j-1]) + (sbl_y[2][1] * img[i+1][j]) + \
             (sbl_y[2][0] * img[i+1][j+1])          
            temp_y[i-1][j-1] = Gausy
    return temp_y

#method to calculate edges of x and y direction
def sobelxy(img,sbl_x,sbl_y):
    Gausx = 0
    Gausy = 0
    img[0,:] = img[:,0] = img[:,-1] =  img[-1,:] = 0
    for i in range(1, L-2):  
        for j in range (1,K-2):
            Gausx = ( sbl_x[0][2]* img[i-1][j-1]) + ( sbl_x[0][1]* img[i-1][j]) + \
             (sbl_x[0][0] * img[i-1][j+1]) + (sbl_x[1][2] * img[i][j-1]) + \
             (sbl_x[1][1] * img[i][j]) + (sbl_x[1][0] * img[i][j+1]) + \
             (sbl_x[2][2]* img[i+1][j-1]) + (sbl_x[2][1]* img[i+1][j]) + \
             (sbl_x[2][0]* img[i+1][j+1])
            
            Gausy = (sbl_y[0][2] * img[i-1][j-1]) + (sbl_y[0][1] * img[i-1][j]) + \
             (sbl_y[0][0] * img[i-1][j+1]) + (sbl_y[1][2] * img[i][j-1]) + \
             (sbl_y[1][1]* img[i][j]) + (sbl_y[1][0] * img[i][j+1]) + \
             (sbl_y[2][2] * img[i+1][j-1]) + (sbl_y[2][1] * img[i+1][j]) + \
             (sbl_y[2][0] * img[i+1][j+1])
            temp_x[i-1][j-1] = Gausx 
            temp_y[i-1][j-1] = Gausy
            edge_mag =math.sqrt(Gausx ** 2 + Gausy ** 2)
            temp_xy[i-1][j-1]=edge_mag
    return  temp_xy

# Computing vertical edges
edge_x = sobelx(img,sbl_x)
#cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
# =============================================================================
# cv2.imshow('edge_x_dir', edge_x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 
# =============================================================================
# Computing horizontal edges
edge_y = sobely(img,sbl_y)
#cv2.namedWindow('edge_y_dir', cv2.WINDOW_NORMAL)
# =============================================================================
# cv2.imshow('edge_y_dir', edge_y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  
# =============================================================================

#Computing vertical and horizontal edges
edge_xy= sobelxy(img,sbl_x,sbl_y)           
# =============================================================================
# cv2.imshow('edge_xy_dir', edge_xy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

#for Normalizing Images FINDING Maximum or minimum
for i in range(1, L): 
    for j in range (1,K):
         if(maxval_x<edge_x[i-1][j-1]):
             maxval_x=edge_x[i-1][j-1]
         if(minval_x>edge_x[i-1][j-1]):
             minval_x=edge_x[i-1][j-1] 
for i in range(1, L): 
    for j in range (1,K):
         if(maxval_y<edge_y[i-1][j-1]):
             maxval_y=edge_y[i-1][j-1]
         if(minval_y>edge_y[i-1][j-1]):
             minval_y=edge_y[i-1][j-1] 
for i in range(1, L): 
    for j in range (1,K):
         if(maxval_xy<edge_xy[i-1][j-1]):
             maxval_xy=edge_xy[i-1][j-1]
         if(minval_xy>edge_xy[i-1][j-1]):
             minval_xy=edge_xy[i-1][j-1] 

# Eliminate zero values with method 1
pos_edge_x = (edge_x -minval_x) / math.fabs( maxval_x-minval_x)
#cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Eliminate zero values with method 1
pos_edge_y = (edge_x -minval_y) / math.fabs( maxval_y-minval_y) 
#cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# magnitude of edges (conbining horizontal and vertical edges)
pos_edge_xy = (edge_xy -minval_xy) / math.fabs( maxval_xy-minval_xy)
cv2.namedWindow('edge_magnitude', cv2.WINDOW_NORMAL)
cv2.imshow('edge_magnitude', pos_edge_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()


print("Original image size: {:4d} x {:4d}".format(img.shape[0], img.shape[1]))
print("Resulting image size: {:4d} x {:4d}".format(pos_edge_xy.shape[0], pos_edge_xy.shape[1]))


