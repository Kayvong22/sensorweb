import itertools
import math
import os
import random
import scipy.spatial.distance 
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastdtw import fastdtw
from functools import partial
from itertools import chain
from scipy import stats


sys.setrecursionlimit(10000)


def init_centers(data,K=2):
	'''
	Randomly initialize the  K centers

	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	'''
	return([random.randint(0,len(data)-1) for k in range(K)])


def assign_points(data,centroids,K):
	'''
	Calculate the distance of each point to the centroids and affect them to the closest 

	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	'''
	for k in range(K):
		if k==0:
			dist=fastdtw(data,centroids[k],dist=euclidean)
		else:
			dist=np.vstack((dist,fastdtw(data,centroids[k],dist=scipy.spatial.distance.euclidean)))

	return([np.argmin(dist,axis=0),dist]) # for each consumer in data affect it to the closest cluster.

def compute_means(data,assigned,K):
	'''
	Calculate the mean for each cluster to generate the centroids

	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	assigned: list of index specifying to which cluster each individual is assigned to
	'''
	return(np.array([np.nanmean(np.array([data[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0) for k in range(K)]))
	

def objective_function(data,centroids,assigned,K):
	'''
	Calculate the sum of the distance of the points to their centroids and return the sum of it

	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	assigned: list of index specifying to which cluster each individual is assigned to
	'''
	sumcort=0
	for k in range(K):
	 	subdata=[data[i] for i in np.where(assigned==k)[0]]
	 	sumcort+=np.nansum(fastdtw(subdata,centroids[k],dist=euclidean) )
	return(sumcort)

def clustering(data,centroids,K=2,eps=1.e-4):
	'''
	main K-means clustering function (used for the consensus clustering)
	
	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	eps: the threshold used to break the loop

	'''	
	WCSS=[]
	# iterate up to 300 loop; stop if the Winthin sum of square is not reducing anymore (<1.e-4)
	for loop in xrange(500):
		print('loop:'+str(loop))
		#assign points to a cluster (center defined by centroid)
		assigned,dist=assign_points(data,centroids,K)
		#recalculate the centroid
		centroids=compute_means(data,assigned,K)
		#check convergence
		WCSS.append(objective_function(data,centroids,assigned,K))
		if len(WCSS)>2:
			if (WCSS[len(WCSS)-2]-WCSS[-1])<eps and (WCSS[len(WCSS)-2]-WCSS[-1])>=0:
				break
		print(WCSS[-1])
	return([WCSS,assigned,centroids,dist])


def Kmpp(data,K,outlierT=2.5,npoints=2):
	'''
	K means++ seeding; find the most extreme point of the data to start as centers (unless they are isolated)
	
	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	outlierT
	npoints
	'''
	# first point randomly seeded
	centroids=[random.randint(0,len(data)-1)]
	outliers=[]
	k=1
	while k<(K+1): # get the K farthest points as cluster centres. the +1 is to make sure the last one is not an outlier.
		print(k)
		if k==1:
			dist=fastdtw(data,data[centroids[-1]],dist=scipy.spatial.distance.euclidean)[0]
			centroids.append(np.argmax(dist))
			k+=1
		else:
			disttemp=fastdtw(data,data[centroids[-1]],dist=scipy.spatial.distance.euclidean)[0]	
			# if a center is too far from the closest point, it is classified as outlier and cannot be seeded
			if sorted(disttemp)[npoints]>outlierT:
				# classified as outlier
				outliers.append(centroids[-1])
				# set the distance to 0 so that it does not get taken in next rounds
				dist[-1][outliers[-1]]=0
				# delete it from centroids' list
				del centroids[-1]
				dist=np.vstack((dist,disttemp))
				# get one step back
				k-=1
			else:
				dist=np.vstack((dist,disttemp))

			centroids.append(np.argmax(np.min(dist,axis=0)))
			k+=1
			
	del centroids[-1]		
	return(centroids,outliers)


