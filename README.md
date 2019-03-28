# Projects
1. Edge Detection using OpenCV in Python:- 
    Edge detection is the method which incolves finding points in an image where the brightness of pixels intensity changes sharply.
    Sobel is used to find derivatives of Laplacian operator. 
    Sobel x and Sobel y are calculated to find derivatives in x and y directions respectively.
2. Panorama Stitching using OpenCV in Python:-
    The two images are taken from same camera but differet angles and goal is to stitch both images to form a panorama.
    Sift operator is used to find keypoints in both images. KNN Matcher is sued to match these keypoints. 
    OpenCV method findHomography() returns the homographic matrix. Inlier points are matched using the homographic matrix.
    OpenCV method warpPerspective() and perspectiveTransform() are used to perform Affine transformation on the images to create panorama.
3. SIFT Operator- Epipolar Line matching:-
    SIFT operator is used to find keypoints, draw matching points and then used OpenCV method computeCorrespondEpilines() to drawn epipolar lines.
    Epipolar geometry is used to calculate depth of the image taken by pin hole camera. Two cameras are used to find depth in epipolar geometry.
4. K means Algorithm:-
    K means Algorithm is a unsupervised clustering algorithm in Machine Learning.  
    K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points 
    and gradually learn how to cluster them into groups by computing the mean of the distance between different points.