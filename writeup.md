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


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for HOG feature extraction is in function `get_hog_features.py` lines 18 through 37 of the file `extractfeatures.py'

The vehicles and non-vehicles present in the given image dataset was balanced and approximately 8600 images and decided to proceed further to find out the accuracy of model.

Various color space images both in vehicle and non-vehicle dataset was visualized using `skimage.hog()` as below


![hog image](output_images/hog_visualization/Car_YCrCb.png)  | ![hog image](output_images/hog_visualization/Not_Car_YCrCb.png)
-----------------------------------------------------------| -------------------------------------------------------------


#### 2. Explain how you settled on your final choice of HOG parameters.

```python
self.orient = 9
self.pix_per_cell = 8
self.cell_per_block = 2
self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
```

These parameters were chosen based on the SVC model accuracy and having hog extraction performed with "ALL" channels for the color space increased the number of features available for training the model

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As suggested in the lecture video Linear SVM was used to classify the vehicle images and the table below lists the various combination of parameters tried on subset of the training data to gather accuracy and training times. The parameters listed above was finalized in order to get higher accuray of ~99% so that the vehicles could be detected from the video with minimum false positives.

With the parameters finalzied the SVM model was training using the entire dataset and was able to acheive an accuracy of 99.7% in 'YCrCb' colorspace.

The code for SVM Linear classifer training is in function `classify_images` lines 135 through 213 of the file `hog_classify.py'

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
 
### HOG Subsampling approach

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use HOG Subsampling approach to search for cars in the images. 

The search space was reduced along the Y ( 400,656 ) dimension and was limited to the near vicinity of the vehicle and also sampling rate of 64 was chosen after trying various values (64,96,128) as it gave better results. I also tried various scale factor 1.0, 1.5, 2.0 and 2.5 and decided upon 1.5 on which the cars were identified with minimum false detection

The code for hog subsampling is in `find_cars` lines 36 through 106 of `hogsubsampler.py`

![subsample image](output_images/hog_subsampling/test1.png)  | ![subsample image](output_images/hog_subsampling/test2.png)
-------------------------------------------------------------| -------------------------------------------------------------
![subsample image](output_images/hog_subsampling/test3.png)  | ![subsample image](output_images/hog_subsampling/test4.png)
![subsample image](output_images/hog_subsampling/test5.png)  | ![subsample image](output_images/hog_subsampling/test6.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Using the above determined Linear SVM parameters and with HOG subsampler approach as suggested in lecture was able to optimize the vehicle detection performance bycalculating HOG features once and sub sample per window. The following is the output for test image pipeline and was able to identify the vehicles successfully. The vehicle in test4.jpg wasn't detected as it was outside the sampled Y dimension near vicinity (400,566)

![subsample image](output_images/heat_visualization/test1.png)  | ![subsample image](output_images/heat_visualization/test2.png)
----------------------------------------------------------------| ---------------------------------------------------------------
![subsample image](output_images/heat_visualization/test3.png)  | ![subsample image](output_images/heat_visualization/test4.png)
![subsample image](output_images/heat_visualization/test5.png)  | ![subsample image](output_images/heat_visualization/test6.png)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. For the video heat map values across 10 frames is computed and the detected blobs are drawn after meeting a threshold of 4 to avoid false positive detection and also to ensure smooth drawing of the detected blobs.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Vehicles were detected in the opposite direction and probably to get better perception for the autonomous vehicle these need to be differentiated as against traffic
* In certain angle the vehicle wasn't detected for fewer frames and need to be handled as this was happening at the near vehicle vicinity
* Having the input features passed at incorrect scale and incorrect color space resulted in lot of false positives.

