import cv2
from PIL import Image
import os, sys
import numpy as np
import time
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import inv
from scipy.linalg import sqrtm


file_name = "images/bear.png"
threshold = 0.9
h_size, v_size = 100, 100

### Question 1 ###
def resize_and_blur_image(file_name):
    # alternative using Image library
    #img = Image.open(file_name)
    #img = img.resize((h_size,v_size), Image.ANTIALIAS)

    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) 
    img_res = cv2.resize(img, (h_size, v_size))
    img_blur = cv2.GaussianBlur(img_res,(5,5),cv2.BORDER_DEFAULT) 
    

    cv2.imwrite("test_blur_2.jpg", img_blur)


def compute_weight(point_1, point_2, threshold=0.9, x_max=100, y_max=100):
    x1, y1 = point_1
    x2, y2 = point_2
    
    norm_val = np.power((x2-x1), 2) + np.power((y2-y1), 2)

    weight_val = np.exp( -4 * norm_val )
    if weight_val < threshold: 
        weight_val = 0

    return weight_val


def compute_sq_eucl_dist(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    
    return np.power((x2-x1), 2) + np.power((y2-y1), 2)


def normalise_pixel(x, y, x_max=100, y_max=100):
    return x/x_max, y/y_max


"""
Constructs a square 10000 x 10000 adjacency matrix
"""
def naive_construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    adj_matr =  np.zeros( (h_size*v_size, h_size*v_size) )
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    # go through each pixel of image one by one
    for i in range(len(img)):
        for j in range(len(img)):
            x_norm, y_norm = normalise_pixel( (i+1), (j+1), h_size, v_size )
            point_1 = (x_norm, y_norm)
            
            # need to find an alternative for the loop
            for row_ind in range(-5, 6):
                for col_ind in range(-5, 6):
                    x_temp = (i+1) + row_ind
                    y_temp = (j+1) + col_ind
                    point_2 = ( normalise_pixel(x_temp, y_temp, h_size, v_size) )

                    #eucl_dist = compute_sq_eucl_dist(point_1, point_2)
                    weight_val = compute_weight(point_1, point_2) 
                    
                    snd_ind = i*h_size + j - row_ind*h_size + col_ind
                    if snd_ind >= 10000:
                        snd_ind -= 10000  
                    adj_matr[i*h_size+j, snd_ind] = weight_val 

    num_nonzero = np.count_nonzero(adj_matr)

    print("Non zero: " + str(num_nonzero))
    print("Average degree: " + str(num_nonzero/10000) )

    return csr_matrix(adj_matr)


"""
Computes inverse squared root of matrix D given an adjacency matrix A
Returns only the diagonal values 
"""
def compute_sqrt_inv_D(A):
    row_sums = np.asarray( csr_matrix.sum(A, axis=1) )
    row_sums = np.sqrt(row_sums)        # take square root
    row_sums = np.reciprocal(row_sums)  # take inverse

    return np.diag(row_sums.reshape((len(row_sums),)))


def compute_normalised_laplacian(A):
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
Given a sparse matrix D, returns squared root form of its inverse
TODO: check if order in which operations are executed is correct
"""
def get_sqroot_inv(D):
    D = inv(D)      # use scipy's inv function for sparse matrix
    D = sqrtm(D)    
    
    return D

def construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    row = []
    col = []
    data = []

    # go through each pixel of image one by one
    for i in range(len(img)):
        for j in range(len(img)):
            x_norm, y_norm = normalise_pixel( (i+1), (j+1), h_size, v_size )
            point_1 = (x_norm, y_norm)
            
            row.append(i)
            col.append(j)

            for ind in range(120):

            # need to find an alternative for the loop
                    x_temp = (i+1) + row_ind
                    y_temp = (j+1) + col_ind
                    point_2 = ( normalise_pixel(x_temp, y_temp, h_size, v_size) )

                    #eucl_dist = compute_sq_eucl_dist(point_1, point_2)
                    weight_val = compute_weight(point_1, point_2) 
                    
                    snd_ind = i*h_size + j - row_ind*h_size + col_ind
                    if snd_ind >= 10000:
                        snd_ind -= 10000  
                    adj_matr[i*h_size+j, snd_ind] = weight_val 

    num_nonzero = np.count_nonzero(adj_matr)

    print("Non zero: " + str(num_nonzero))
    print("Average degree: " + str(num_nonzero/10000) )

    return csr_matrix(adj_matr)


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

def compute_eigenvalue(A, eigen_vector):
    pass


""" 
def construct_adjacency_graph(file_name, threshold=0.9, h_size=100, v_size=100):
    adj_matr =  np.zeros( (h_size, v_size) )
    # adjacency matrix will have shape 100x100
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    # go through each pixel of image one by one
    for i in range(len(img)):
        for j in range(len(img)):
"""            

def main():
    start = time.time()
    #resize_and_blur_image(file_name)
    A = naive_construct_adjacency_matrix("test_blur_2.jpg")
    L = compute_normalised_laplacian(A)

    end = time.time()
    print(end - start)

#main()

A = csr_matrix( [[4, 0], [0, 16]] )

print( power_method(A) )

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

