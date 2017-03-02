##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding-window.png
[image4]: ./output_images/final_image1.png
[image5]: ./output_images/heatmap.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/color_spaces.png
[image9]: ./output_images/final_image2.png
[image10]: ./output_images/final_image3.png
[image11]: ./output_images/final_image4.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the in lines 25 through 42 of the file called `vehicle-detection.py`.  The function `get_hog_features()` takes in the parameters `pixels_per_cell`, `cells_per_block`, `orient` as input and outputs the hog features of the image. I call this function inside the function `extract_features` to get the hog features of an image.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is the HOG visualization of an example car image for the parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Apart from HOG feature exploration I also looked into various color spaces. After comparing the 3d plots of various colorspaces (RGB, HSV and YCrCb), I decided to use YCrCb color space for extracting color space information from cares and later using it along with HOG features to train my classifier.

Here is an example of color space plots for an example car image in RGB, YCrCb and HSV space.

![alt-text][image8]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and compared the training and testing accuracy of the model. I chose the model which was most accurate without compromising on time. I found out that `color_space = YCrCb`, `orientation=9`, `pixels_per_cell = 8`, `cells_per_block = 2`, `hog_channel = ALL`, `spatial_size = 16` and `hist_bins = 16` yields best results in terms of SVM model accuracy and time. The parameters can be seen in the lines 258 through 267 of the file called `vehicle-detection.py`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG and color features. The color features are extracted using the functions `bin_spatial` and `color_hist`. These functions are shown in the lines 45 through 49, and 52 through 60 of the file `vehicle-detection.py`. The HOG features are extracted using the function `get_hog_features` which is shown in the lines 25 through 42 of the file `vehicle-detection.py`. Finally a write a function called `extract_features()`, which takes a list of images as input and extracts the color and HOG features, concatenates them, and outputs a feature array. This function is shows in the lines 65 through 114 of the file `vehicle-detection.py`.

After extracting the car and notcar features and stacking them in an array, I used `StandardScaler().fit()` to normalize the feature vector. This feature vector and labels are then fed to `svc.fit()` to train the classifier. The training process can be seen in the lines 282 through 315 of the file `vehicle-detection.py`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the function `find_cars()`, which is in lines 152 through 214 of the file `vehicle-detection.py`. First we define the ystart, ystop, xstart and xtart values, which indicates where to look for the cars. Next step is to search the image. To improve the efficiency and accuracy we only search that portion of the image which might have cars. Whilst searching we set the parameters such as window_size and overlap, which indicates the size of the bounding boxes that we search, and also the overlap between to consecutive searches. Once we successfully find some cars in the image, we can store them in an array called bboxes.

I used y_start = 400 and y_stop = 656 and a scale of 1.5. This ensures that I'm only searching in the areas where i would find a car. The overlap helps in searching through all the pixels and looking for the car. I tried various values of overlap on a sample image and chose the one which gave me most accurate boxes.

Here is an example of the bounding boxes that I drew on an image after searching using sliding windows technique.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I still encountered some false positives in the result. To get rid of the false positives I built a heat map from these detections. The functions `add_heat()` and `apply_threshold()` are used to detect the hot parts of the map (where the cars are), and then reject false positives by applying a threshold. After removing false positives, here are some example images:

![alt text][image4]
![alt text][image9]
![alt text][image10]
![alt text][image11]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example of a test image with bounding box and heatmap corresponding to the image

![alt text][image5]
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced problems in deciding the right colorspace and HOG parameters. It was a lot of hit and trial and running the model and checking accuracy. After I was done with manually setting all the parameters, I had the problem of false positives. To eliminate false positives I used the heatmap technique and then carefully tuned the threshold value.

To make the code more robust, I would try to use deep learning so that the parameters are tuned by the neural net. I can also implement other methods to remove false positives. To make the program more efficient I would like to optimize the search space.

These are some of the works that I really liked in this area:
1. You only look once: https://arxiv.org/abs/1506.02640
2. https://github.com/diyjac/SDC-P5
