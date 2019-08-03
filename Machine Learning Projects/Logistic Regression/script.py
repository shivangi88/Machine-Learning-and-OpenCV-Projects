import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(W,x):
    mult1 = np.dot(x,W)
    expo1 = np.exp(mult1)
    sum1 = np.sum(expo1,axis=1)
    sum1 = sum1.reshape(sum1.shape[0],1)
    output = expo1/sum1
    return output

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    W = initialWeights.reshape((n_features+1,1))   #adding bias
    ones_data = np.ones((n_data,1))
    X = np.hstack((ones_data,train_data))
    mult1 = np.dot(X,W)
    theta_n = sigmoid(mult1)

    mult2 = labeli*np.log(theta_n)
    sub1 = 1.0-labeli
    logr1 = np.log(1.0-theta_n)
    mult3 = sub1*logr1
    sum1 = mult2 + mult3
    error = -np.sum(sum1)
    error = error/n_data    #computing error
    
    sub2 = theta_n-labeli
    sub2 = sub2.reshape(n_data,1)
    error_grad = sub2*X
    error_grad = np.sum(error_grad, axis=0)
    error_grad = error_grad/n_data    #computing error gradient

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    rows = data.shape[0]
    label = np.zeros((rows,1))
    one_rows = np.ones((rows, 1))
    X = np.hstack((one_rows,data))   #adding bias

    mult1 = np.dot(X, W)
    out1 = sigmoid(mult1)   #compute sigmoid to get output
    out2 = np.argmax(out1, axis=1)    
    label = out2.reshape((rows,1))
    
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    W = params.reshape((n_feature+1,10))     #adding bias
    data_ones = np.ones((n_data,1))
    X = np.hstack((data_ones,train_data))
    Y = softmax(W,X)                       #computing softmax to get output
    
    log_y=np.log(Y)
    mult1=labeli*log_y
    sum1=np.sum(mult1)
    error = -np.sum(sum1)
    error = error/n_data               #computing error
    
    sub1 = Y-labeli
    e1 = np.dot(X.T,sub1)
    error_grad = e1.flatten()/n_data   #computing error gradient

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    n_data = data.shape[0];
    data_ones = np.ones((n_data, 1))
    X = np.hstack((data_ones,data))     #adding bias

    probab1 = softmax(W,X)                  #compute softmax to get output
    probab2 = np.argmax(probab1, axis=1)    
    label = probab2.reshape((n_data,1))
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
print("blr obj function done")
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
cmatrix = confusion_matrix(y_true=train_label,y_pred=predicted_label)
print(cmatrix)
# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
cmatrix = confusion_matrix(y_true=test_label,y_pred=predicted_label)
print(cmatrix)

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
#Sampling 10,000 training data samples
svm_train_data_indices_1=np.random.permutation(50000)
svm_train_data_indices = svm_train_data_indices_1[0:10000]
svm_train_data=train_data[svm_train_data_indices,:]
svm_train_label=train_label[svm_train_data_indices]

#Linear kernel
print('SVM with linear kernel')
calculated_svm_l = SVC(kernel='linear')
calculated_svm_l.fit(svm_train_data, svm_train_label.flatten())
train_score_val=100*calculated_svm_l.score(svm_train_data, svm_train_label)
validation_score_val=100*calculated_svm_l.score(validation_data, validation_label)
test_score_val=100*calculated_svm_l.score(validation_data, test_label)
print('\n Training set Accuracy:' + str(train_score_val) + '%')
print('\n Validation set Accuracy:' + str(validation_score_val) + '%')
print('\n Testing set Accuracy:' + str(test_score_val) + '%')


# Radial basis function with gamma = 1
print('\n\n SVM with radial basis function, gamma = 1')
calculated_svm_rbf_1 = SVC(kernel='rbf', gamma=1.0)
calculated_svm_rbf_1.fit(svm_train_data, svm_train_label.flatten())
train_score_val=100*calculated_svm_rbf_1.score(svm_train_data, svm_train_label)
validation_score_val=100*calculated_svm_rbf_1.score(validation_data, validation_label)
test_score_val=100*calculated_svm_rbf_1.score(test_data, test_label)
print('\n Training set Accuracy:' + str(train_score_val) + '%')
print('\n Validation set Accuracy:' + str(validation_score_val) + '%')
print('\n Testing set Accuracy:' + str(test_score_val) + '%')

#Radial basis function with gamma = default
print('\n\n SVM with radial basis function, gamma = 0')
calculated_svm_rbf_0 = SVC(kernel='rbf')
calculated_svm_rbf_0.fit(svm_train_data, svm_train_label.flatten())
train_score_val=100*calculated_svm_rbf_0.score(svm_train_data, svm_train_label)
validation_score_val=100*calculated_svm_rbf_0.score(validation_data, validation_label)
test_score_val=100*calculated_svm_rbf_0.score(test_data, test_label)
print('\n Training set Accuracy:' + str(train_score_val) + '%')
print('\n Validation set Accuracy:' + str(validation_score_val) + '%')
print('\n Testing set Accuracy:' + str(test_score_val) + '%')

print('\n\n SVM with radial basis function, different values of C')
print("SVM")
train_acc = np.zeros(11)
valid_acc = np.zeros(11)
test_acc = np.zeros(11)
C_values = np.zeros(10)
C_1 = 1.0   # When C=1
#different values of C from 10 to 100 with difference of 10 
C_values = [ 10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0, 100.0]    
#Computing SVM for each value of C
calculated_svm = SVC(C=C_1,kernel='rbf')
calculated_svm.fit(svm_train_data, svm_train_label.flatten())
train_score_val=100*calculated_svm.score(svm_train_data, svm_train_label)
validation_score_val=100*calculated_svm.score(validation_data, validation_label)
test_score_val=100*calculated_svm.score(test_data, test_label)
print('\n Training set Accuracy for C=1:' + str(train_score_val) + '%')
print('\n Validation set Accuracy for C=1:' + str(validation_score_val) + '%')
print('\n Testing set Accuracy for C=1:' + str(test_score_val) + '%')

train_acc[0] = 100*calculated_svm.score(svm_train_data, svm_train_label)
valid_acc[0] = 100*calculated_svm.score(validation_data, validation_label)
test_acc[0] = 100*calculated_svm.score(test_data, test_label) 
train_acc[1:] = np.zeros(10)
valid_acc[1:] = np.zeros(10)
test_acc[1:] = np.zeros(10)
for i in range(10):
    calculated_svm = SVC(C=C_values[i],kernel='rbf')
    calculated_svm.fit(svm_train_data, svm_train_label.flatten())
    train_acc[i+1] = 100*calculated_svm.score(svm_train_data, svm_train_label)
    valid_acc[i+1] = 100*calculated_svm.score(validation_data, validation_label)
    test_acc[i+1] = 100*calculated_svm.score(test_data, test_label)
C_manyValues=np.zeros(11)
C_manyValues=np.append(C_1,C_values)

plt.plot(C_manyValues, train_acc, 'o-',C_manyValues, valid_acc,'o-',C_manyValues, test_acc, 'o-')
plt.xlabel('C values')
plt.ylabel('Accuracy (%)')
plt.legend(('Training','Validation','Test'), loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

#Computing SVM for C=10 after analysis
calculated_svm = SVC(C=10,kernel='rbf')
calculated_svm.fit(train_data, train_label.flatten())
train_score_val=100*calculated_svm.score(train_data, train_label)
validation_score_val=100*calculated_svm.score(validation_data, validation_label)
test_score_val=100*calculated_svm.score(test_data, test_label)
print('\n Training set Accuracy for C=10:' + str(train_score_val) + '%')
print('\n Validation set Accuracy for C=10:' + str(validation_score_val) + '%')
print('\n Testing set Accuracy for C=10:' + str(test_score_val) + '%')


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))
print("mlr obj function done")
# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
cmatrix = confusion_matrix(y_true=train_label,y_pred=predicted_label_b)
print(cmatrix)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
cmatrix = confusion_matrix(y_true=test_label,y_pred=predicted_label_b)
print(cmatrix)
