import cv2
from PIL import Image
import os, sys
import numpy as np
from scipy.sparse import csr_matrix

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

def compute_weight(x, y, threshold):
    pass

def construct_adjacency_graph(file_name):
    # 1) 

    # adjacency matrix will have shape 100x100
    adj_matr = np.zeros( (100, 100) )
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    
    for i in range(len(img)):
        for j in range(len(img)):
            print(img[i][j])


def main():
    #resize_and_blur_image(file_name)
    construct_adjacency_graph("test_blur_2.jpg")


main()