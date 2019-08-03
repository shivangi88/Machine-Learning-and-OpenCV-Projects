import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # your code here
    sigma=1.0/(1.0+np.exp(-z))
    return sigma 
    


def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    matkeys = mat.keys()
    l_keys=list(matkeys)
    l_keys=l_keys[3:]
    n=0
    m=0
    list_train_keys=[]
    list_test_keys=[]
    
    # Fetching and Separating train and test keys
    for l in l_keys:
        if 'train'+str(n) in l:
            list_train_keys.append('train'+str(n))
            n=n+1
        if 'test'+str(m) in l:
            list_test_keys.append('test'+str(m))
            m=m+1
        
    train_data_temp=np.zeros((1,784))
    train_data_label = np.full((1), 10)
    i=0
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    # Fetching data from keys having train data
    for key in list_train_keys:
        temp_train = np.array(mat[key])
        train_data_temp = np.vstack((train_data_temp, temp_train))
        train_lab = np.full((temp_train.shape[0]), i)
        train_data_label = np.concatenate((train_data_label, train_lab))
        i = i + 1
    train_data_label = train_data_label[1:60001] 
    
    train_data_temp = train_data_temp[1:60001,:]
    
    # Fetching the randomly sampled indices for the train and validation set
    indices = np.random.permutation(train_data_temp.shape[0])
    
    #Splitting data into validation data and training data
    train_indices, validation_indices = np.split(indices,[50000])        
    
    train_data = train_data_temp[train_indices,:]
   
    train_label = train_data_label[train_indices]

    validation_data = train_data_temp[validation_indices,:]
    
    validation_label = train_data_label[validation_indices]

   
    
    # Block for test data
    test_data_temp = np.zeros((1,784))
    test_data_label = np.full((1), 10)
    i = 0

    # fetching the data from keys having test data    
    for key in list_test_keys:
        temp_test = np.array(mat[key])
        test_data_temp = np.vstack((test_data_temp, temp_test))
        #label
        test_label_rows = temp_test.shape[0]
        test_lab = np.full((test_label_rows), i)
        test_data_label = np.concatenate((test_data_label, test_lab))
        i = i + 1

    # 
    test_label = test_data_label[1:10001]
    
    #
    test_data = test_data_temp[1:10001,:]
    
    # Feature selection
    data = np.vstack((train_data, validation_data))
    
    a = np.all(data == data[0,:], axis=0)
    
    ind = []
    for i in range(0,len(a)):
        if (a[i] == False):
            ind.append(i)
    
    train_data = train_data[:,ind]
    train_data=train_data/255.0
    validation_data = validation_data[:,ind]
    validation_data=validation_data/255.0
    test_data = test_data[:,ind]
    test_data=test_data/255.0
    print('preprocess done')

    return ind,train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    
    #Feedforward Propagation:-
    
    #Input Layer  to Hidden Layer
    train_data_rows=training_data.shape[0]
    bias_training_input=np.ones(train_data_rows) 
    training_data=np.column_stack((training_data,bias_training_input)) #adding bias
    trans_w1=np.transpose(w1)
    mult1=np.dot(training_data,trans_w1)
    out_hid=sigmoid(mult1) 
    
    #Hidden Layer to Output Layer
    out_hid_rows=out_hid.shape[0]
    bias_hid=np.ones(out_hid_rows)
    out_hid=np.column_stack((out_hid,bias_hid))
    trans_w2=np.transpose(w2)
    mult2=np.dot(out_hid,trans_w2)
    output=sigmoid(mult2) #Final Output
    
    #Derivation of Error calculated
    label_array = np.array(training_label)
    size_label_array = len(label_array)
    rowIndex = np.arange(size_label_array)
    target = np.zeros((size_label_array,10))
    target[rowIndex,train_label]=1
    del_Error = output - target
    
    del_Error_w2 = np.dot(del_Error,w2)
    mult_out_hid_1 = 1-out_hid
    mult_out_hid_2 = mult_out_hid_1 * out_hid
    mult3 = mult_out_hid_2 * del_Error_w2
    trans_del_Error_w2 = np.transpose(mult3)
   
    #Error Gradient for w1
    grad_w1 = np.dot(trans_del_Error_w2,training_data) 
    grad_w1 = np.delete(grad_w1,n_hidden,0)
    
    #Error Gradient for w2
    trans_del_Error = del_Error.T
    grad_w2 = np.dot(trans_del_Error,out_hid)
    
    #Error value calculation:-
    new_target = 1-target
    new_output = 1-output
    
    #Negative Log Likelihood Calculation
    log_output=np.log(output)
    log_new_output=np.log(new_output)
    error_val_1 = target*log_output
    error_val_2 = new_target*log_new_output
    error_val_1 = np.sum(-1*(error_val_1+error_val_2))
    error_val_1 = error_val_1/train_data_rows
    
    #Calculating Regularization
    square_w1 = np.square(w1)
    square_w2 = np.square(w2)
    sum_w1_w2 = np.sum(square_w1) + np.sum(square_w2)
    n_2 = 2*train_data_rows
    error_val_2 = (lambdaval/n_2) * sum_w1_w2
    
    #Final Error Value
    obj_val = error_val_1 + error_val_2

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad=obj_grad/train_data_rows
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data_rows=data.shape[0]
    
    #INPUT LAYER
    bias=np.ones(data_rows)
    data=np.column_stack((data,bias))
    trans_w1=np.transpose(w1)
    mult1=np.dot(data,trans_w1)
    out_hid=sigmoid(mult1)
    
    #HIDDEN LAYER 
    hid_rows=out_hid.shape[0]
    bias_hid=np.ones(hid_rows)
    out_hid=np.column_stack((out_hid,bias_hid))
    trans_w2=np.transpose(w2)
    mult2=np.dot(out_hid,trans_w2)
    output=sigmoid(mult2)
    
    #FINAL OUTPUT
    labels=np.argmax(output,1)
    
    return labels


"""**************Neural Network Script Starts here********************************"""

start=time.time()
selected_features,train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 12

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
end=time.time()

predicted_label = nnPredict(w1, w2, validation_data)


# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


total_time=end-start
print(str(total_time))


pickle.dump((selected_features,n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
file = pickle.load(open('params.pickle','rb'))
print(file)




