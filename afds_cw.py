import cv2
from PIL import Image
import os, sys
import numpy as np
import time
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import inv
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance 

file_name = "images/bear.png"
threshold = 0.9
h_size, v_size = 100, 100

"""
Resizes and blurs image given image's name
"""
def resize_and_blur_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) 
    img_res = cv2.resize(img, (h_size, v_size))
    img_blur = cv2.GaussianBlur(img_res,(5,5),cv2.BORDER_DEFAULT) 

    cv2.imwrite("test_blur_2.jpg", img_blur)


def compute_weight(point_1, point_2, threshold=0.9, x_max=100, y_max=100):
    weight_val = np.exp( -4 * np.linalg.norm( point_2 - point_1 )**2 )

    if weight_val < threshold: 
        weight_val = 0

    return weight_val


def calculate_weights_without_wrapeffect(file_name):
    xvalues = np.arange(1, 101)
    yvalues = np.arange(1, 101)

    yy, xx = np.meshgrid(xvalues, yvalues)
    xx = xx/100
    yy = yy/100

    img = cv2.imread("test_blur_2.jpg", cv2.IMREAD_UNCHANGED)
    img = img.reshape((10000, 3))    
    img = img/255

    points = np.concatenate( (xx.flatten().reshape((10000,1)), yy.flatten().reshape((10000,1)), img), axis=1) 
    print(points.shape)
    res = distance.cdist(points, points, metric='sqeuclidean')
    res = np.exp(-4*res)
    res[res < .9] = 0


def normalise_pixel(x, y, r, g, b, x_max=100, y_max=100):
    return x/x_max, y/y_max, r/255, g/255, b/255


"""
Constructs a square 10000 x 10000 adjacency matrix
"""
def naive_construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    adj_matr =  np.zeros( (h_size*v_size, h_size*v_size) )
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    # go through each pixel of image one by one
    for i in range(len(img)):
        for j in range(len(img)):
            r, g, b = img[i][j]
            x, y, r, g, b = normalise_pixel(i+1, j+1, r, g, b)
            point_1 = np.array([x, y, r, g, b])
            
            # need to find an alternative for the loop
            for row_ind in range(-5, 6):
                for col_ind in range(-5, 6):
                    x2 = (i+1) + row_ind
                    y2 = (j+1) + col_ind
                    r2, g2, b2 = img[row_ind, col_ind]
                    point_2 = np.array([normalise_pixel(x2, y2, r2, g2, b2)])
                    
                    # compute weight between two points in the grid
                    weight_val = compute_weight(point_1, point_2) 
                    
                    snd_ind = (i+row_ind)*h_size +  j+col_ind
                    if snd_ind >= 10000:
                        snd_ind -= 10000  
                    
                    adj_matr[i*h_size+j, snd_ind] = weight_val

    num_nonzero = np.count_nonzero(adj_matr)

    print("Non zero: " + str(num_nonzero))
    print("Average degree: " + str(num_nonzero/10000) )

    return adj_matr

# TODO: make this function smarter and more flexible
def get_sq_spatial_dist():
    eucl_distances = np.zeros((11, 11))
    
    for x in range(-5, 6):
        for y in range(-5, 6):
            eucl_distances[5-x,5-y] = np.power(x/100, 2) + np.power(y/100, 2)

    return eucl_distances

def get_sq_rgb_dist(neighbours, pixel):
    rgb_dist = np.subtract( neighbours, pixel )
    #rgb_dist = np.square(rgb_dist)
    rgb_dist = rgb_dist**2

    return np.sum(rgb_dist, axis=2)

def get_sq_euclidean_dist(spatial_distances, rgb_distances):
    return np.add(spatial_distances, rgb_distances)

def calculate_weights(eucl_dist, threshold=0.9):
    weights =  np.exp(-4 * eucl_dist)
    weights[weights < threshold] = 0

    return weights

def construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    xx = []
    yy = []
    data = []
    spatial_distances = get_sq_spatial_dist()
    
    # normalize data
    img = img/255

    # go through each pixel of image one by one
    for x in range(len(img)):
        for y in range(len(img)):
            neighbours = np.take(img, range(x-5,x+6), axis=0, mode='wrap').take(range(y-5,y+6), axis=1, mode='wrap')
    
            # calculate sq rgb distances between neighbours and central point
            rgb_distances = get_sq_rgb_dist(neighbours, img[x,y])
            #calculate squared euclidean distance between central point and its neighbours
            eucl_dist = get_sq_euclidean_dist(spatial_distances, rgb_distances)
            # calculate_weights
            weights = calculate_weights(eucl_dist)

            # add weights to data
            data.extend(weights.flatten())

            # row_ind is the same as inserting edges from (x,y) vertex
            row_ind = np.array([100*x + y]*121)

            x_ind = np.repeat( 100*np.array(range(x-5, x+6)) , 11)
            y_ind = np.tile( np.array(range(y-5, y+6)) , (11) )
            col_ind = np.add(x_ind, y_ind)
            
            # TODO: check these methods
            col_ind[col_ind>=10000] -= 10000
            col_ind[col_ind<0]      += 10000

            # get indices
            xx.extend(row_ind)
            yy.extend(col_ind)
             
    return csr_matrix( (data, (xx, yy)), shape=(10000, 10000) )


################# Question 3 #################
"""
Given a sparse matrix D, returns squared root form of its inverse
"""
def get_sqroot_inv(D):
    D = inv(D)      # use scipy's inv function for sparse matrix
    D = sqrtm(D)    
    
    return D


"""
Computes inverse squared root of matrix D given an adjacency matrix A
Returns only the diagonal values 
"""
def compute_sqrt_inv_D(D):
    row_sums = np.asarray( csr_matrix.sum(A, axis=1) )
    row_sums = np.sqrt(row_sums)        # take square root
    row_sums = np.reciprocal(row_sums)  # take inverse

    return np.diag(row_sums.reshape((len(row_sums),)))


def compute_D(A):
    return A.sum(axis=1)


def normalised_laplacian(A):
    # L = I - inv(sqrt(D)) A inv(sqrt(D))
    sqrt_inv_D = compute_sqrt_inv_D(A)
    
    # inv(sqrt(D)) A
    DA = A.dot(sqrt_inv_D)
    # inv(sqrt(D)) A inv(sqrt(D))
    ADA = DA.dot(sqrt_inv_D)
   
    I = identity(A.shape[0])

    L = np.subtract(I.toarray(), ADA)
    
    return L


"""
Input: Adjacency matrix (A) and D in the form of vector
Returns: the desired matrix (I + inv(sqrt(D))*A*inv(sqrt(D)) )
"""
def desired_matrix(A, D):
    D_inv = np.reciprocal(D)

    return identity(A.shape[0]) + D_inv * A


def power_method(B):
    B = B.todense()
    print(B)
    x = np.random.normal(0, 1, B.shape[0])
    k = 10

    for i in range(k):
        x = np.asarray(B.dot(x))
        x = x.reshape((x.shape[1],))
        x = x/np.linalg.norm(x)
        print(x)

    return x, np.linalg.norm(B.dot(x))

"""
Given a square root of matrix D and a matrix B,
computes using power method the 2nd largest eigenvalue of B
"""
def power_method_2nd_eigenvalue(B, D_sq_rt, k=10):
    x0 = np.random.normal(0, 1, B.shape[0])
    
    # assume supplied matrix is already squared root of D
    v1 = D_sq_rt  
    x = x0 - np.dot(v1, x0) * v1

    for i in range(k):
        x = np.asarray(B.dot(x))
        x = x.reshape((x.shape[1],))
        x = x/np.linalg.norm(x)

    return x, np.linalg.norm(B.dot(x))


def compute_eigenvalue(A, eigen_vector):
    pass


def main():
    start = time.time()
    #resize_and_blur_image(file_name)
    A = construct_adjacency_matrix("test_blur_2.jpg")
    D = compute_D(A)
    
    print(type(D))
    print(D.shape)
    
    #D_sq_rt = np.sqrt(np.reciprocal(D))
    #des_matr = desired_matrix(A, D)
    #power_method_2nd_eigenvalue(des_matr, D_sq_rt)
    
    #L = compute_normalised_laplacian(A)
    

    end = time.time()
    print(end - start)

main()


#print( np.take(arr, [range(i-2,i+3), range(-2,3)], mode='wrap') )

# works for small indices
#print( np.take(arr, range(i-2,i+3), axis=0).take(range(j-2,j+3), axis=1)  )

#print( arr[i-2:i+3, j-2:j+3] )

""" start = time.time()
naive_construct_adjacency_matrix("test_blur_2.jpg")
    
end = time.time()
print(end - start) """


""" matrix = np.zeros((2, 2))
val = 0
for i in range(2):
    for j in range(2):
        matrix[i][j] = val
        val+=1
"""
""" row = np.array([0,2,2,0,1,2])
col = np.array([0,0,1,2,2,2])
data = np.array([1,2,3,4,5,6])
print( csr_matrix( (data,(row,col)), shape=(3,3) ).todense() ) 
A = csr_matrix( (data,(row,col)), shape=(3,3) )
compute_matrix_D(A)
"""