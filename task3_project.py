
"""
@author: shivangi
"""

import matplotlib.pyplot as plt
import copy
import numpy as np
#import cv2

f1=[5.9, 4.6, 6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
f2=[3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]
classification_vector=[]
# Getting the data points
X = np.array(list(zip(f1, f2)))
# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
# Number of clusters
k = 3
# X coordinates of given centroids
C_x = [6.2,6.6,6.5]
# Y coordinates of given centroids
C_y = [3.2,3.7,3.0]
# Getting the centroids
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
# Storing the value of centroids when they gets updated
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error functionwhich is calculated asd istance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will runs till the error becomes zero 
temp=0  #Checking condition for producing output every iteration
colors = ['r', 'g', 'b']
fig, ax = plt.subplots()
def kMeansClusteringAlgo(X,error,temp,k,isImage=0):
	while error != 0:
		temp=temp+1
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		for i in range(len(C)):
			C_old = copy.deepcopy(C)
		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [X[j] for j in range(len(X)) if clusters[j] == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)
		if(temp==1 and isImage==0):
			print(clusters)
			plt.figure(1)
			for i in range(0,k):
				points =np.array([X[j] for j in range(len(X)) if clusters[j] == i])
				plt.scatter(points[:, 0], points[:, 1], marker='^', s=50, facecolors='none',edgecolors=colors[i])
				plt.scatter(C_old[i, 0], C_old[i, 1], marker='o', s=100, c=colors[i])
			plt.title('task3_iter1_a.jpg')
			plt.savefig('task3_iter1_a.jpg')
			plt.figure(2)
			for i in range(0,k):
				points =np.array([X[j] for j in range(len(X)) if clusters[j] == i])
				plt.scatter(points[:, 0], points[:, 1], marker='^', s=50, facecolors='none',edgecolors=colors[i])
				plt.scatter(C[i, 0], C[i, 1], marker='o', s=100,  c=colors[i])
			plt.title('task3_iter1_b.jpg')
			plt.savefig('task3_iter1_b.jpg')
		if(temp==2 and isImage==0):
			plt.figure(3)
			for i in range(0,k):
				points =np.array([X[j] for j in range(len(X)) if clusters[j] == i])
				plt.scatter(points[:, 0], points[:, 1], marker='^', facecolors='none',edgecolors=colors[i])
				plt.scatter(C_old[i, 0], C_old[i, 1], marker='o', s=100,  c=colors[i])
			plt.title('task3_iter2_a.jpg')
			plt.savefig('task3_iter2_a.jpg')
			plt.figure(4)
			for i in range(0,k):
				points =np.array([X[j] for j in range(len(X)) if clusters[j] == i])
				plt.scatter(points[:, 0], points[:, 1], marker='^', s=50, facecolors='none',edgecolors=colors[i])
				plt.scatter(C[i, 0], C[i, 1], marker='o', s=100, c=colors[i])
			plt.title('task3_iter2_b.jpg')
			plt.savefig('task3_iter2_b.jpg')
	return C,clusters
centres,clust=kMeansClusteringAlgo(X,error,temp,3)


