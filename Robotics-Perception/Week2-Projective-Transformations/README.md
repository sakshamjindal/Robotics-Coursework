## Perspective Projection

### Homography Estimation

The mathematical name for homography concept is "projective transformation" (source) and in computer vision it refers to transforming images such as if they were taken under different perspective.

An example from opencv: we have two images of the same place, taken from different angle. We will compute homography H. If we now select one pixel with coordinates (x1, y1) from the first image and another pixel (x2, y2) that represents the same point on another image, we can transform the latter pixel to have the same viewing perspective as the first one by applying H.

homography is just a special case of transformation. Most examples that I have seen, consider homography only for 2D (i.e., for images). Still, homography can be extended to larger dimension (source).


**Why do we need 4 points** ?

Consider the number of DOFs and number of constraints. On the one hand, the matrix H contains 9 entries, but is defined only up to scale. Thus, the total number of degrees of freedom in a 2D projective transformation is 8.

On the other hand, each point-to-point correspondence accounts for two constraints, since for each point x_i in the first image the two DOFs of the point in the second image must correspond to the mapped point H*x_i. A 2D point has two DOFs to (x,y) components, each of which may be specified separately.

Alternatively, the point is specified as a homogeneous 3-vector, which also has 2 DOFs since scale is arbitrary. As a consequence, it is necessary to specify four point correspondences in order to constrain H fully.

### Transformation

Transformation is very general concept and includes all kinds of conversions, including conversion between coordinate frames, homography is a subset of it, mostly only applied when rotation is needed (source). In computer vision it is a technical term that describes above-mentioned case of transformation. 

 *For planar surfaces, 3D to 2D perspective, projection reduces to a 2D to 2D transformation. Punchline2: This transformation is INVERTIBLE!*

#### Links : 
- Best Notes to solve for Homography Matrix (http://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf)
- https://docs.opencv.org/master/d9/dab/tutorial_homography.html
- https://medium.com/all-things-about-robotics-and-computer-vision/homography-and-how-to-calculate-it-8abf3a13ddc5
- https://dsp.stackexchange.com/questions/40289/why-do-we-need-4-points-for-homography-but-7-8-points-for-fundamental-matrix-cal#:~:text=In%202D%20each%20corresponding%20point,of%20DOFs%20of%20the%20homography.
- https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
- https://math.stackexchange.com/questions/2388259/differences-between-homography-and-transformation-matrix 
- http://www.cs.columbia.edu/~allen/F17/NOTES/homography_pka.pdf
- cse.psu.edu/~rtc12/CSE486/lecture16.pdf
- https://stackoverflow.com/questions/28961743/homography-and-projective-transformation
- http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf