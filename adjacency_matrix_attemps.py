def construct_adjacency_matrix(file_name, threshold=0.9, h_size=100, v_size=100):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    # go through each pixel of image one by one
    for i in range(len(img)):
        for j in range(len(img)):
            r, g, b = img[i][j]
            x, y, r, g, b = normalise_pixel(i+1, j+1, r, g, b, h_size, v_size)
            point_1 = (x, y, r, g, b)
            
            neighbors = img[i-3:i+4, j-3:j+4]

            # need to find an alternative for the loop
            x_temp = (i+1) + row_ind
            y_temp = (j+1) + col_ind
            point_2 = ( normalise_pixel(x_temp, y_temp, h_size, v_size) )

            #weight_val = compute_weight(point_1, point_2) 
            
            snd_ind = i*h_size + j + row_ind*h_size + col_ind
            if snd_ind >= 10000:
                snd_ind -= 10000  
            adj_matr[i*h_size+j, snd_ind] = weight_val 

    num_nonzero = np.count_nonzero(adj_matr)

    print("Non zero: " + str(num_nonzero))
    print("Average degree: " + str(num_nonzero/10000) )

    return csr_matrix(adj_matr)

"""
spatial_distances = get_sq_spatial_dist()
i=5
j=5
neighbours = img[i-5:i+6, j-5:j+6]

# row_ind is the same as inserting edges from (i,j) vertex
row_ind = np.array([j + i*100]*121)

xx = np.repeat( 100*np.array(range(i-5, i+6)) , 11)
yy = np.tile( np.array(range(i-5, i+6)) , (11) )
col_ind = np.add(xx, yy)

# calculate sq rgb distances between neighbours and central point
rgb_distances = get_sq_rgb_dist(neighbours, img[i,j])
#calculate squared euclidean distance between central point and its neighbours
eucl_dist = get_sq_euclidean_dist(spatial_distances, rgb_distances)
# calculate_weights
weights = calculate_weights(eucl_dist)
"""

# create edges from all vertices in S with this new latest index
#old_edges = [(val, latest_index) for val in S]

# calculate sum of these edges
#indices = [int(x*10000+y) for (x,y) in old_edges]

#TODO: need to find another way - line below takes ages
#A_flat = A.todense().flatten().reshape((100000000, 1))

"""
if (x > 3 and x < 97 and y > 3 and y < 97) or (x < 3 and y > 3 and y < 97) or (x > 97 and y > 3 and y < 97):
    x_ind = np.repeat( 100*np.array(range(x-3, x+4)) , 7)
    y_ind = np.tile( np.array(range(y-3, y+4)) , (7) )
    col_ind = np.add(x_ind, y_ind).reshape((7,7))
elif (x > 3 and x < 97 and y < 3) or (x < 3 and y < 3) or (x > 97 and y < 3):
    x_ind = np.repeat( 100*np.array(range(x-3, x+4)) , 7)
    y_ind = np.tile( np.array(range(y-3, y+4)) , (7) )
    col_ind = np.add(x_ind, y_ind).reshape((7,7))
    col_ind[:,:3] += 100
    col_ind = col_ind.flatten()
elif (x > 3 and x < 97 and y > 97) or (x > 97 and y > 97) or(x < 3 and y > 97):
    x_ind = np.repeat( 100*np.array(range(x-3, x+4)) , 7)
    y_ind = np.tile( np.array(range(y-3, y+4)) , (7) )
    col_ind = np.add(x_ind, y_ind).reshape((7,7))
    col_ind[:,4:] -= 100
    col_ind = col_ind.flatten()
"""