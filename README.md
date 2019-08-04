# Projects
  ## Machine Learning Projects
   ## MultiLayer Perceptron Neural Network using Tensorflow in Python:-
    Implemented a Multilayer Perceptron Neural Network on MNIST (Modified National Institute of Standards and Technology) handwritten digit
    dataset and evaluate its performance in classifying the handwritten digits. Used the same Neural Network on a Face dataset and
    evaluate its performance against a deep neural network and a convolutional neural network.

   ## Logistic Regression and SVM on MNIST using sklearn in Python:-
    Logistic Regression on MNIST data to classify the handwritten digits and implement Support Vector Machine from sklearn to perform
    classification.

   ## K means Algorithm:-
    K means Algorithm is a unsupervised clustering algorithm in Machine Learning.  
    K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points 
    and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

   ## Loan Claim Status Classification using Tensorflow in Python:-
    Implemented Neural Network to predict whether a loan claim will be accepted or rejected based on given input features. Input dataset
    contains following input columns: Loan id, Gender, Married, Dependents, Education, Self Employed, Applicant income, Coapplicant income,
    Loan Amount,Credit History, Property_Area. Target: Loan_Status. 

 ## OPENCV/Computer Vision Projects:
   ## Edge Detection using OpenCV in Python:- 
    Edge detection is the method which incolves finding points in an image where the brightness of pixels intensity changes sharply.
    Sobel is used to find derivatives of Laplacian operator. 
    Sobel x and Sobel y are calculated to find derivatives in x and y directions respectively.

   ## Panorama Stitching using OpenCV in Python:-
    The two images are taken from same camera but differet angles and goal is to stitch both images to form a panorama.
    Sift operator is used to find keypoints in both images. KNN Matcher is sued to match these keypoints. 
    OpenCV method findHomography() returns the homographic matrix. Inlier points are matched using the homographic matrix.
    OpenCV method warpPerspective() and perspectiveTransform() are used to perform Affine transformation on the images to create panorama.

   ## SIFT Operator- Epipolar Line matching:-
    SIFT operator is used to find keypoints, draw matching points and then used OpenCV method computeCorrespondEpilines() to drawn epipolar
    lines.Epipolar geometry is used to calculate depth of the image taken by pin hole camera. Two cameras are used to find depth in epipolar
    geometry
