## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![hog image](output_images/hog_visualization/Car_HLS.PNG)  | ![hog image](output_images/hog_visualization/Not_Car_HLS.PNG)
-----------------------------------------------------------| ---------------------------------------------------------
![hog image](output_images/hog_visualization/Car_HSV.PNG)  | ![hog image](output_images/hog_visualization/Not_Car_HSV.PNG)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

| Color Space        | HOG Channel           | Features Length  | SVC Test Accuracy  |  SVC Training Time(s) | Extract HOG (s) |
| ------------------ |:---------------------:| ----------------:|-------------------:|----------------------:|-----------------:
| RGB                | 0                     |1764              | 0.96               | 0.42                  |9.54             |
| RGB                | 1                     |1764              | 0.97               | 1.37                  |8.56             |
| RGB                | 2                     |1764              | 0.97               | 0.44                  |8.35             |
| RGB                | ALL                   |5292              | 0.98               | 0.87                  |19.0             |
| HSV                | 0                     |1764              | 0.94               | 0.94                  |8.36             |
| HSV                | 1                     |1764              | 0.95               | 1.51                  |8.54             |
| HSV                | 2                     |1764              | 0.96               | 1.36                  |8.50             |
| HSV                | ALL                   |5292              | 1.00               | 0.58                  |19.13            |
| LUV                | 0                     |1764              | 0.97               | 1.35                  |8.40             |
| LUV                | 1                     |1764              | 0.96               | 1.62                  |8.46             |
| LUV                | 2                     |1764              | 0.97               | 1.47                  |8.53             |
| LUV                | ALL                   |5292              | 0.99               | 0.53                  |18.92            |
| HLS                | 0                     |1764              | 0.92               | 0.75                  |8.22             |
| HLS                | 1                     |1764              | 0.96               | 1.33                  |8.34             |
| HLS                | 2                     |1764              | 0.95               | 1.59                  |8.50             |
| HLS                | ALL                   |5292              | 0.99               | 0.55                  |19.06            |
| YUV                | 0                     |1764              | 0.98               | 1.34                  |8.18             |
| YUV                | 1                     |1764              | 0.93               | 0.52                  |8.46             |
| YUV                | 2                     |1764              | 0.93               | 1.93                  |8.48             |
| YUV                | ALL                   |5292              | 0.99               | 0.57                  |19.45            |
| YCrCb              | 0                     |1764              | 0.98               | 1.26                  |8.45             |
| YCrCb              | 1                     |1764              | 0.96               | 1.81                  |8.44             |
| YCrCb              | 2                     |1764              | 0.96               | 1.81                  |8.51             |
| YCrCb              | ALL                   |5292              | 0.99               | 0.52                  |19.30            |
 
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

