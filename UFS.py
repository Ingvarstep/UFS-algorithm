import numpy as np
import math
from  sklearn.linear_model import LassoLarsCV
import scipy

def get_euclidian_norm(a):
    x = np.expand_dims(a, axis=1)
    y = np.sum((a-x)**2, axis=-1)
    return y**0.5

def get_cos_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def weighting(a,b, delta, type):
    if type == 'hkw':#Heat kernel weighting
        return math.exp(-math.pow(np.linalg.norm(a-b),2)/delta)
    elif type == 'cos_sim':
        return get_cos_sim(a,b)
    elif type == 'bin':
        return 1
    else:
        return 0

def UFS(data, K, d):
    '''
    Cai D, Zhang C, He X. Unsupervised feature selection for multi-cluster data[C]
    //Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2010: 333-342.
    This algorithm implements an unsupervised feature selection.
    data: is an N*M matrix, each row represents a sample, and each column represents a feature;
    K: number of clusters;
    d: number of features to select;
    return :  data:The data N*d after feature selection, each row represents a sample, and each column represents a feature;   
              seq:Indicates the selected feature sequence number 1*d each element in seq represents the feature index from the original data.
    '''
    k = int(max(0.2*data.shape[0],10))#Number of neighbors
    data = np.array(data)
    M = data.shape[1]#number of features
    N = data.shape[0]#number of data samples
    dist = get_euclidian_norm(data)
    data = data.transpose()#Convert the matrix to column form
    #Find the K-nearest neighbors of a sample of the data
    delta = 2#The parameters of the thermal kernel function in the original text
    W = np.zeros((N, N))#Adjacency matrix weights
    D = np.zeros((N, N))#matrix initialization
    for i in range(N):#Find the k-nearest neighbors for each sample
        neigbors = np.argsort(dist[i,:], axis=0)
        neigbors = neigbors[1:k+1]#Only the first k samples are selected, because the 0th shortest matrix is the distance between itself and itself, and its distance is 0
        for j in neigbors:
            W[i,j] = weighting(data[:,i],data[:,j], delta, type = 'hkw')#Calculate the elements in the weight matrix
            W[j,i] = W[i,j]
        D[i,i] = sum(W[i,:])#diagonal elements at degree matrix
    L = D - W #Calculate the Laplace matrix
    feature_values, vectors = scipy.linalg.eig(L,D)#Find generalized eigenvalues and eigenvectors
    seq = np.argsort(feature_values)#Sort eigenvalues
    seq = seq[1:K+1]#Select the K eigenvalues from the next smallest
    Y = vectors[:,seq]#get feature vector
    Y = np.real(Y)
    # Using least angle regression to get the parameters of the balloon section a
    score = np.zeros((1,M))#Record the score for each feature
    model = LassoLarsCV()#train a model
    print(Y[:,0].shape)
    for i in range(K):
        model.fit(data.transpose(),Y[:,i])
        a = model.coef_#Get the coefficients of a linear regression model
        score[0,i] = max(a)
    seq = np.argsort(-score)#Sort the scores from largest to smallest
    seq = seq[0,0:d]#Select the first d largest scores
    data = data.transpose()
    data = data[:,seq]#get the final result
    return data,seq,score
