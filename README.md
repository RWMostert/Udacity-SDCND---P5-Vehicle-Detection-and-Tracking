# **Vehicle Detection Project**

![demo](result.gif)
(https://youtu.be/wg0ZnZdV7m0)

---

### Histogram of Oriented Gradients (HOG)

The code for extracting the HOG features is contained in the second code cell of the IPython notebook (the `get_hog_features` method). This method makes use of SKImage's hog function to extract the HOG features from the vehicle / non-vehicle images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![png](output_2_0.png)


![png](output_2_1.png)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

```python
from skimage.feature import hog
import cv2

featimg_vehicle = cv2.cvtColor(vehicle, cv2.COLOR_RGB2YCrCb)
featimg_non_vehicle = cv2.cvtColor(non_vehicle, cv2.COLOR_RGB2YCrCb)

_, hog_img_vehicle = hog(featimg_vehicle[:,:,0], orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualise=True, feature_vector=True)
_, hog_img_non_vehicle = hog(featimg_non_vehicle[:,:,0], orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualise=True, feature_vector=True)

```


![png](output_5_0.png)

I used spatial features, the colour histogram of each image, as well as the HOG features to train my models. (See the `extract_features` function) <br>
For classification, I used an ensemble of - firstly - a linear support vector machine (from sklearn - LinearSVC), as well as a very simple feedforward neural network. Using these two classifiers in ensemble produced the best results.

### Sliding Window Search

The sliding-window search is implemented in the `slide_window` and `search_windows` functions in my the Vehicle Detection iPython notebook. <br> The functions work by sliding a window, or set of windows, across the image being searched (within specific boundaries for where the actual road is in the image). These windows then get fed to the `search_windows` function, which extracts the features (hog, histogram and spatial features) from the image bounded by the window, and feeds them to our classifier(s). We use the result from the classifier to say whether the window in question contains a vehicle or not. <br>

I implemented the sliding window approach at 3 different sizes: 96x96, 128x128 and 192x192.  Smaller window sizes took a lot of time to process, and yielded results that left much to be desired. <br>
Ultimately, I needed to get my window sizes to represent the images I trained on; i.e. a nicely bounded box around most of the vehicles I would like to identify. These three sizes worked well in combination to identify cars at different depths in the camera images. <br>
I kept my overlap at 0,5 - which seemed to work best. I wanted the overlap to be such that it would be unlikely to miss a vehicle (because it might be in the center of two boxes) while not being too inefficient on processing time.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


![png](output_10_1.png)


---

### Video Implementation

[link to the video result](./output_video_nn_ensemble.mp4)


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, which I averaged over the last 10 frames of the video to get a result (this smooths out unsure, once-off, false positives) and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


![png](output_13_0.png)


### Discussion

What makes this project challenging, is the sheer amount of knobs that can be tuned. They can seem overwhelming, and - at first - very daunting to start the tuning process as you try to make the project work.
<br> <br>
The pipeline is likely to fail under very different lighting conditions, or bad weather conditions. We haven't trained enough on images from night driving, and we should expect issues, the same holds for all other sub-ideal weather conditions. <br>
The pipeline could be made more robust by incorporating different models, for example a convolutional neural network fed on the images themselves (not the hog features). Ensembling more models is also an option to consider, as they tend to compliment each other (and add "knowledge").
