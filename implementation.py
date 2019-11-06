import os, sys
import numpy as np
import time
import seaborn as sns 
import matplotlib.pyplot as plt
import cv2
import sys

from PIL import Image
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import inv
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance 
from itertools import combinations

################# Question 1 #################
def resize_and_blur_image(file_name, output_name, h_size=100, v_size=100):
    """
    Resizes and blurs image given image's name
    """
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) 
    img_res = cv2.resize(img, (h_size, v_size))
    img_blur = cv2.GaussianBlur(img_res,(5,5),cv2.BORDER_DEFAULT) 

    # Uncomment if you'd like to see the blurred picture - by default the image is saved
    #plt.imshow(img_blur)
    #plt.show()
    
    # save the picture into a file
    cv2.imwrite(output_name, img_blur)

################# /Question 1 #################


################# Question 2 #################
def get_sq_spatial_dist(neigh_size, square_dim):
    """
    Calculates spatial squared euclidean distance between a central point and 
    its neighbours. Neighbourhood size is constant for each pixel => output of
    this function is constant and hence only calculated once
    """
    eucl_distances = np.zeros((square_dim, square_dim))
    
    for x in range(-neigh_size, neigh_size+1):
        for y in range(-neigh_size, neigh_size+1):
            eucl_distances[neigh_size-x,neigh_size-y] = np.power(x/100, 2) + np.power(y/100, 2)

    return eucl_distances


def get_sq_rgb_dist(neighbours, pixel):
    """
    Returns array of squared euclidean distances of the central pixel's RGB values
    and RGB values of its neighbourhood
    """
    rgb_dist = np.subtract( neighbours, pixel )
    rgb_dist = rgb_dist**2

    return np.sum(rgb_dist, axis=2)


def get_sq_euclidean_dist(spatial_distances, rgb_distances):
    """
    Returns sum of spatial distances and RGB distances between central pixel and its neighbours
    """
    return np.add(spatial_distances, rgb_distances)


def calculate_weights(eucl_dist, threshold=0.9):
    """
    Applies specified function on euclidean distances (both spatial and RGB together)
    to calculate weight between central pixel and its neighbours. Applies threshold.
    """
    weights =  np.exp(-4 * eucl_dist)
    weights[weights < threshold] = 0

    return weights


def construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    """
    Constructs an adjacency matrix for the specified image
    For each pixel of the image, it considers +-5 rows and columns in each direction as a neighbourhood
    of the pixel and applies the "wrap around" effect
    """
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    xx = []
    yy = []
    data = []
    neigh_size = 5
    square_dim = 2 * neigh_size + 1

    # compute spatial dimensions only once - they repeat for each pixel
    spatial_distances = get_sq_spatial_dist(neigh_size, square_dim)

    # normalize data
    img = img/255

    # go through each pixel of image one by one
    for x in range(len(img)):
        for y in range(len(img)):
            # take neighbours of central pixel with coordinates (x,y)
            neighbours = np.take(img, range(x-neigh_size,x+neigh_size+1), axis=0, mode='wrap').take(range(y-neigh_size,y+neigh_size+1), axis=1, mode='wrap')
            
            # calculate sq rgb distances between neighbours and central point
            rgb_distances = get_sq_rgb_dist(neighbours, img[x,y])

            #calculate squared euclidean distance between central point and its neighbours
            eucl_dist = get_sq_euclidean_dist(spatial_distances, rgb_distances)

            # calculate_weights
            weights = calculate_weights(eucl_dist, threshold=threshold)

            # add weights to data
            data.extend(weights.flatten())

            # row_ind is the same as inserting edges from (x,y) vertex
            row_ind = np.array([100*x + y]*121)

            if (y < neigh_size):
                x_ind = np.repeat( 100*np.array(range(x-neigh_size, x+neigh_size+1)) , square_dim)
                y_ind = np.tile( np.array(range(y-neigh_size, y+neigh_size+1)) , (square_dim) )
                col_ind = np.add(x_ind, y_ind).reshape((square_dim,square_dim))
                col_ind[:,:neigh_size-y] += 100
            elif (y >= neigh_size and y < 100-neigh_size):
                x_ind = np.repeat( 100*np.array(range(x-neigh_size, x+neigh_size+1)) , square_dim)
                y_ind = np.tile( np.array(range(y-neigh_size, y+neigh_size+1)) , (square_dim) )
                col_ind = np.add(x_ind, y_ind).reshape((square_dim,square_dim))
            elif (y >= 100-neigh_size):
                x_ind = np.repeat( 100*np.array(range(x-neigh_size, x+neigh_size+1)) , square_dim)
                y_ind = np.tile( np.array(range(y-neigh_size, y+neigh_size+1)) , (square_dim) )
                col_ind = np.add(x_ind, y_ind).reshape((square_dim,square_dim))
                col_ind[:,100+neigh_size-y:] -= 100
            
            col_ind = col_ind.flatten()
            col_ind[col_ind>=10000] -= 10000
            col_ind[col_ind<0]      += 10000

            # get indices
            xx.extend(row_ind)
            yy.extend(col_ind)

    return csr_matrix( (data, (xx, yy)), shape=(10000, 10000) )

################# /Question 2 #################


################# Question 3 #################
def compute_D(A):
    """
    Computes vector D where D = (d_1, ...., d_n) and each d_i
    is the degree of vertex i (thus sum of the row of A[i]) 
    """
    return A.sum(axis=1)
    

def normalised_laplacian(A):
    """
    Computes normalised laplacian matrix given adjacency matrix A
    """
    # L = I - inv(sqrt(D)) A inv(sqrt(D))
    D = compute_D(A)
    D_inv = np.reciprocal(D.astype(float))

    I = identity(A.shape[0])

    L = np.subtract(I, D_inv*A)
    
    return L


def desired_matrix(A, D):
    """
    Input: Adjacency matrix (A) and D in the form of vector
    Returns: the desired matrix (I + inv(D)*A ) which holds as D is a diagonal matrix
    """
    D_inv = np.reciprocal(D.astype(float))
    D_inv_matr = np.diag(D_inv)
    
    return identity(A.shape[0]) + D_inv_matr * A


def power_method(B, k=100):
    """
    Estimates the largest eigenvalue of matrix B
    """
    B = B.todense()
    x = np.random.normal(0, 1, B.shape[0])

    for i in range(k):
        x = np.asarray(B.dot(x))
        x = x.reshape((x.shape[1],))
        x = x/np.linalg.norm(x)
        print(x)

    return x, np.linalg.norm(B.dot(x))


def power_method_2nd_eigenvalue(B, v1, k=1000):
    """
    Given a matrix B, estimates the 2nd largest eigenvalue of B
    using power method
    Input: matrix B and v1 - vector corresponding to the largest eigenvalue of B
    Output: vector corresponding to the 2nd largest eigenvalue of B and number of iterations k 
    """
    x0 = np.random.normal(0, 1, B.shape[0])
    x0 = x0/np.linalg.norm(x0)

    # reshape v1 to shape (n,) from (n,1)
    v1 = v1.reshape( (v1.shape[0],) )
    v1 = v1/np.linalg.norm(v1)
    dot_product = np.dot(v1, x0)
    
    x = x0 - dot_product * v1
    x = x/np.linalg.norm(x)

    for i in range(k):
        x = np.asarray(B.dot(x))
        x = x.reshape((x.shape[1],))
        x = x/np.linalg.norm(x)         # normalise the vector x

    return x, k

################# /Question 3 #################


################# Question 4 #################
def sort_eigenvector(f2):
    """
    Sorts elements of given vector and returns both sorted elements and an array of indices
    to retrieve original elements
    """
    V_inds = f2.argsort()
    f2.sort()

    return f2, V_inds


def sum_old_weights(S, A):
    """
    When S is updated with a new vertex v, this vertex is no longer in 
    V\S. Hence, it is necessary to subtract all edges between v and all elements
    of S. This function computes sum of these edges. 
    """
    # create edges from all vertices in S with this new latest index
    indices = tuple([[S[-1]]*len(S), S])
    
    # sum adjacency matrix over given indices of edges that are no longer in w(S, V\S)
    sum_val = np.sum(A[indices])

    # dont forget to multiply by 2 due to symmetry of edges (edge(1,2) == edge(2,1) as A is symmetric)
    return sum_val*2


def phi_function(S, A, S_vol, Conj_vol, nominator):
    """
    Applies Phi function on a given set of vertices S
    Uses other arguments to apply dynamic programming approach for faster computation
    """
    # no need to compute volume every time, only needs to add d_u for newly added vertex u=S[-1]
    increment  = A[S[-1]].sum()
    
    # increase volume of S
    S_vol += increment
    # conjugate volume needs to be decreased by increment that was added to S_vol
    Conj_vol -= increment

    # choose minimum between the volumes
    denominator = min(S_vol, Conj_vol)
    
    # compute w(S, conj(S))
    nominator += increment
    decrement = sum_old_weights(S, A)
    nominator -= decrement

    return nominator/denominator, S_vol, Conj_vol, nominator


def find_sparse_cut(f2, A):
    """
    Finds sparse cut in the image given second smallest eigenvector f2 and 
    adjacency matrix representation of the graph A
    """
    # sort vertices based on their f2 values in ascending order
    f2, V_inds = sort_eigenvector(f2)
    
    # intialise sets S and S*
    S      = []
    S_star = [ V_inds[0] ]

    # initialise volumes
    S_vol         = 0
    Conj_vol      = A.sum()

    # initialise nominator values for computing w(S, V\S)
    S_nom      = 0

    # initialise phi(S*) to a very large value     
    S_star_phi = 1000000

    t = 0
    # iterate through whole graph except for the last index
    # this is because result might obtain very small negative value due to numpy computations
    while t < 9999:
        S.append(V_inds[t])
        S_phi, S_vol, Conj_vol, S_nom = phi_function(S, A, S_vol, Conj_vol, S_nom)

        # Update S* if phi(S) < phi(S*) 
        if S_phi < S_star_phi:
            S_star = S.copy()
            S_star_phi = S_phi

        t += 1
        
    return S_star

################# /Question 4 #################


# Call this method to execute the code
def main(file_name):
    #start the timer
    start = time.time()
    print("Starting: ")

    # set threshold for weights
    threshold = 0.90
    k = 1000

    #file_name = "images/sheep.png"
    output_file_name = "question1_blur.jpg"

    # Question 1
    resize_and_blur_image(file_name, output_file_name)
    
    # Question 2
    A = construct_adjacency_matrix(output_file_name, threshold=threshold)
    A.setdiag(0)        # ensure the diagonal elements are 0

    # Question 3
    D = np.asarray(compute_D(A)).reshape((10000,))

    # compute the desired matrix (I - sqrt(inv(D))*A*sqrt(inv(D)) )
    des_matr = desired_matrix(A, D)
    
    # get the 1st eigenvector
    v1 = np.sqrt(D)  # D represented as a vector

    # Compute the 2nd smallest eigenvector of the normalised laplacian 
    f2, k = power_method_2nd_eigenvalue(des_matr, v1, k=k)
    print("Value of k: ", k)
    
    img_repr = f2.reshape((100, 100))
    heat_map = sns.heatmap(img_repr)
    plt.savefig("question3_heatmap.jpg")
    # Uncomment if you'd like to plot the heat map - by default it is saved into a file 
    #plt.show()

    # Question 4
    S_star = find_sparse_cut(f2, A) 
    f2[S_star] = 1

    img_repr = f2.reshape((100, 100))
    heat_map = sns.heatmap(img_repr)
    plt.savefig("question4_heatmap.jpg")
    # Uncomment if you'd like to plot the heat map - by default it is saved into a file 
    #plt.show()
    
    end = time.time()
    print("Time from start till the end: ", end - start)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please specify name of the file to be processed")
    else:
        main(sys.argv[1])