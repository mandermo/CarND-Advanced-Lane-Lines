## Writeup Template

### My solution to the "Advance Lane Lines" project in the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). Here you can find a [link to Udacity's upstream project](https://github.com/udacity/CarND-Advanced-Lane-Lines).
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

[image1]: ./output_images/undistorted_chessboard.png "Undistorted"
[image2]: ./output_images/distorted.jpg "Road Transformed"
[image3]: ./output_images/thresholded_image.jpg "Binary Example"
[image4]: ./output_images/unwarping.png "Warp Example"
[image5]: ./output_images/lane_location.jpg "Lane Line Search"
[polyfit]: ./output_images/birdseye_fitted_polygon.jpg "Fit Visual"
[polyfitnonrobust]: ./birdseye_fitted_polygon_nonrobust.jpg "Fit Visual Non Robust"
[polyrescale]: ./polynom_rescale_algebra.jpg "Polynom Rescale Algebra"
[image6]: ./output_images/lanestats.jpg "Output"
[video1]: ./output_images/processed_project_video.mp4 "Video"

[//]: #[image1]: ./examples/undistort_output.png "Undistorted"
[//]: #[image2]: ./test_images/test1.jpg "Road Transformed"
[//]: #[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[//]: #[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[//]: #[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[//]: #[image6]: ./examples/example_output.jpg "Output"
[//]: #[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/AdvancedLaneFinding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates. `imgpts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. `objpts` will be initialized to the same size as `imgpt`, with each entry a copy of `objp`. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. In the next cell I have to code to do undistortion. I started with using cv2.undistort(), but had trouble figuring out calling cv2.undistortPoints() later with same transform used by cv2.undistort(). Therefore I decided to use cv2.getOptimalNewCameraMatrix(), cv2.initUndistortRectifyMap() and cv2.remap() for undistortion. Then give the return value from cv2.getOptimalNewCameraMatrix() to cv2.undistortPoints() for undistortion of pixel coordnates, which I later needed to translate lane positions for the unwarping. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the Jupyter notebook under heading Thresholding).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp_lane()`, which appears in the under heading "Unwarped view of lane area" in the Jupyter notebook.  The `unwarp_lane()` function takes as inputs an image (`img`), the source and destination points are not taken as parameters, but hardcoded in the same cell in the notebook. I choose the hardcode the source and destination points in the following manner:

```python
# Distorted coordinates for straight_lines2.jpg
# Positions on lane from test_images/straight_lines2.jpg. The positions are distorted.
srcpts = np.array([(308,656), (588,455), (697,455), (1012,656)], np.float32)
srcpts_undistorted = undistort_road_img_points(srcpts)

sidemargin = 260
#dstpts = np.multiply(np.array([(0,1), (0,0), (1,0), (1,1)], np.float32), np.float32(unwrapped_dim))
dstpts = np.array([
    (sidemargin,unwrapped_dim[1]),
    (sidemargin,0),
    (unwrapped_dim[0]-sidemargin,0),
    (unwrapped_dim[0]-sidemargin,unwrapped_dim[1])],
    np.float32)
```

This resulted in the following source and destination points:

| Source         | Destination    |
-----------------------------------
| 362.8, 623.4   | 260.0, 720.0   |
| 612.6, 445.5   | 260.0, 0.0     |
| 704.9, 445.4   | 1020.0, 0.0    |
| 982.8, 622.6   | 1020.0, 720.0  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I did searching for possible lane area using the code from the section "Sliding Window Search", because I imagined using a convolution could give more stable result. The code for that is under "Finding of lane lines". My own spices to the pipeline is that I used a morphological closing operation to the thresholded image before unwarping it. The code for the morphological operation can be found in the function `close_binary_lane_img()`. The effect of that is that the pixels between the lane marking contours are also set to white. The result of that is shown here.
![alt text][image5]
I have been quite agressive with very tolerant thresholding values, therefore there are some false positives. When I fit a polygon using least squares using `np.polyfit()`, it is not very accurate.
![alt text][polyfitnonrobust]
To combat that I choose to use some method that better accomdates outliers (my false positives are outliers). I based my fitting on what I learned here http://scipy-cookbook.readthedocs.io/items/robust_regression.html. The code can be found under "Fit polygon to the points" The result is this.
![alt text][polyfit]
I use robust fitting when tweaking the video pipeline, but did it on a shorter video. The robust fitting is significantly slower and I would have tweaked my thresholds if I did it again.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature under heading "Curvature and position in lane". Instead of rescaling the points to meters and then doing a new polynom fitting a I choose to resale the polynom using simple algebra shown here.
![alt text][polyrescale]

Then I applied the formula from the section "Measuring Curvature" to calulcate the curvature from the polynomial equation. I applied that formula to both left and right lane line polynom. To calculate the average of that I choose to average the inverse of the radius of left and right, and the invert the averaged inverse to get the radius. I imagine that the inverse radius is  more stable than the radius and therefore I imagine it should be a better to average the inverse.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell under heading "Lane on Image" in the function `draw_lane_on_image`.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If the found lane is wrong it could have trouble to recover in the video. If lightness would change a lot, it could in practice affect the saturation enough to make the thresholding wrong. If too many pixels are flaged as possible lane pixels, maybe some more adaptive thresholding could be used. Doing some approx SLAM where lane pixels are fitted to a the flat road plane (same plane used for doing birdseye view. Then tracking the position using feature descriptors on the the lane markings. Even if feature points are easiest to extract for the dotted line, it could be fine. If the right lane line is dotted and the left is solid and is not found in current frame, then the right lane line can be used to update how our position has changed and map the previously found left lane line to the position of this frame. That should perform better than averaging the old frames, but is much more complicated to implement.