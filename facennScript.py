'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    sigma=1.0/(1.0+np.exp(-z))
    return sigma
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    train_data_rows=training_data.shape[0]
    bias_training_input=np.ones(train_data_rows)
    training_data=np.column_stack((training_data,bias_training_input))
    trans_w1=np.transpose(w1)
    mult1=np.dot(training_data,trans_w1)
    out_hid=sigmoid(mult1)
    
    #HIdden Layer
    out_hid_rows=out_hid.shape[0]
    bias_hid=np.ones(out_hid_rows)
    out_hid=np.column_stack((out_hid,bias_hid))
    trans_w2=np.transpose(w2)
    mult2=np.dot(out_hid,trans_w2)
    output=sigmoid(mult2)
    
    label_array=np.array(training_label)
    size_label_array=len(label_array)
    rowIndex = np.arange(size_label_array)
    target = np.zeros((size_label_array,2))
    target[rowIndex,train_label]=1
    del_Error = output - target
    #new weights
    
    
    
    del_Error_w2 = np.dot(del_Error,w2)
    mult_out_hid_1 = 1-out_hid
    mult_out_hid_2 = mult_out_hid_1*out_hid
    mult3=mult_out_hid_2* del_Error_w2
    trans_del_Error_w2 = np.transpose(mult3)
    
    grad_w1 = np.dot(trans_del_Error_w2,training_data) 
    grad_w1 = np.delete(grad_w1,n_hidden,0)
    
    trans_del_Error = del_Error.T
    grad_w2 = np.dot(trans_del_Error,out_hid)
    
    #Error value calculation:-
    new_target = 1-target
    new_output = 1-output
    
    log_output=np.log(output)
    error_val_1 = target*log_output
    
    error_val_2 = new_target*np.log(new_output)
    
    error_val_1 = np.sum(-1*(error_val_1+error_val_2))
    
    error_val_1 = error_val_1/train_data_rows
    
    #Calculating Regularization
    square_w1 = np.square(w1)
    square_w2 = np.square(w2)
    sum_w1_w2 = np.sum(square_w1) + np.sum(square_w2)
    n_2 = 2*train_data_rows
    error_val_2 = (lambdaval/n_2) * sum_w1_w2
    obj_val = error_val_1 + error_val_2

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad=obj_grad/train_data_rows
    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
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

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
start=time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
end=time.time()
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

total_time=end-start
print(str(total_time))