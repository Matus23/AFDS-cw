Implementation of my solution to this coursework is in script implementation.py
This can be run as follows:
    python implementation.py "images/sheep.png"

assuming that you are in the directory of this file and image that the program
should be run on is under folder "images" and is called "sheep.png".

The program runs with default threshold value for computing weights and neighbourhood
of size +- 5 rows and columns from a given pixel. When using power method to compute the
second smallest eigenvalue and corresponding eigenvector of the normalised laplacian matrix, 
default value for k, the number of iterations, was set to 1000. This is a rather large value
and is there to ensure as accurate results as the implementation can get. 

With these values for parameters specified above, the execution time is around 70 seconds. This
is substantially more than the limit, however, code optimisation has still been done and initial
execution time has been cut down a lot. 

Please note that if value of k is set to a lower value, e.g. 500, the execution time dropped to around 55 seconds.
However, this is not such a significant difference which is why I kept the value of k so large. 

By default, the images produced at each stage of the assessment are not displayed but saved instead. If you 
would like to display these images, just uncomment lines that are indicated.