# **Finding Lane Lines on the Road**

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the `draw_lines()` function.

The pipeline accepts an RGB image as input, applies guassian blur, canny edge detection, and then mask the region of interest with a triangle covering the middle section.

[region](/md/region.png)

Then result is then passed to Hough transform to identify the lane lines.

In order to draw a single line on the left and right lanes, I created a new function `draw_lines2()` which separates lines into 2 groups based on their slope.

Briefly, given the line characteristics, slopes is filterd by discarding overly small values (nearly horizontal) and overly large values (more likely to be noise). Then lines are separated into `left` and `right` groups based on their sign. Positive slopes are right lines and negative slopes are left lines.

Once we have left and right line groups, further slope filtering is done by discarding outliers. Based on experimentation, outlier detection by median works best for the test cases here.

Finally, the remainig line points are then fed into `cv2.fitLine` to get a single line on left and right for display overlay.

In order to tackle the challenge video, an additional state variable is used to track left line and right line slope and y intercept over time in order to smooth lane detection and transition. An running average is used to construct final lane overlay which results in significantly less line jumps.

[example output](test_images_output/solidYellowCurve2.jpg)



### 2. Identify potential shortcomings with your current pipeline


The pipeline is hand tuned to the examples at hand and only handle daylight conditions with clear lane markers.

Another potential issue is weather conditions and scenery variety, which is completely unaccounted for.

Finally, the smoothing via running average also prevents sharp lane changes, which is possible in real life.


### 3. Suggest possible improvements to your pipeline

To improve the lane detection pipeline, a good first step is collect more lane images from a variety of conditions.

Next, choose an evalution criteria to give the pipeline a score so we can apply a search method such as gridsearch to optimize the parameters for the conditions we like to tackle. In other words, to be more robust, we find need different pipelines with different parameters under different conditions.

One interesting option is try lane detection via some other sensor input such as x-ray, infra red camera etc.
