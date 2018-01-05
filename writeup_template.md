## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup/undistorted_2.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image3]: ./writeup/combined_binary_3.jpg "Binary Example"
[image4]: ./writeup/warped_2.jpg "Warp Example"
[image5]: ./writeup/lanes_2.jpg "Fit Visual"
[image6]: ./writeup/result_2.jpg "Output"
[video1]: ./output_images/lanes_detected.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibration.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

I then included the mtx and dist values in the `calibration_pickle.p` file to be used in the image processing pipeline.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


In the `image_generation.py` file I used a combination of color and gradient thresholds to generate a binary image. I implemented a number of procedures to perform various thresholding transformations form lines 14-80. the functions are an absolute soble threshold, a color threshold, using the S and V channels from the HLS and HSV colour spaces.I also included a threshold function for the magnitude of the gradient and another for the direction of the gradient.

I tried a number of combinations of the above procedures but settled on building the binary image from all of them, it is in the `threshold_binary()` procedure on line 83

Here's an example of my output for this step.

![alt text][image3]

The code for my perspective transform includes a function called `perspective_transformation()`, which appears in lines 128 through 133 in the file `image_generation.py`.  The `perspective_transformation()` function takes as inputs an image (`img`). It uses a function `perspective_src_dst()` to generate the source (`src`) and destination (`dst`) points from the image and returns M and Minv.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used and operation `find_lane_lines_with_sliding_windows(binary_warped)` which takes a binary warped image and uses sliding windows to identify the left and right lane lines the procedure is on lines: (147-260).

it finds the base of each line by generating a histogram of the pixel intensities. the highest intensity representing the line.

I identify the base of each line and then using a series of nine sliding windows I identify the nonzero pixels in each window. These are the collection of pixels present in each line.

I used np.polyfit to fit a line 2nd order polynomial to the pixels in each line. and stored them in the leftx_fit and rightx_fit variables.

I then used cv2.fillpoly to illustrate the search area on the image coloured the pixels left for blue and red for left.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the `draw_lane_on_image()` function lines 239 through 288 in my code in `image_generation.py`
I performed a conversion to move from pixel space to real space, meters. i then calulated the curvature for the lines in the code on lines 269 - 272.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `draw_lane_on_image()` function lines  239 through 288 in my code in `image_generation.py`   Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I built up the solution using the info I'd picked up in the class. for the most part the work in class aligned well with the solution. I attempted to put some structure on the solution by including several functions. But would like to look further at the design of such solutions.

The pipeline fluctuates a bit when there are bright spots on the image.


