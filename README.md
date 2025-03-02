#VR-Assignment1

##Description

###1) Image Stitching



This code is used to stitch images with overlapping features together. I have tried two approaches for stitching images:

(a) Using the Stitcher class from OpenCV (cv2).

(b) Using SIFT/ORB-based feature matching:
Keypoints are detected using SIFT/ORB.
A homography matrix is computed, allowing adjacent images in the panorama to be warped into a single image.
This process is repeated iteratively until all images are stitched together.
Finally, black borders (caused by perspective differences) are removed using different techniques.



###2) Coin Edge Detection and Segmentation
   

This code is used to detect, segment, and count coins using computer vision techniques.

Techniques Used:

Image Preprocessing
The input image is converted to grayscale for simplified processing.

Gaussian Blur is applied to reduce noise and improve edge detection.

Edge DetectionCanny Edge Detection is used. Morphological operations (dilation and closing) refine the edges.

Contour Detection Coins are identified using cv2.findContours().

Segmentation and Cropping Each detected coin is extracted using a binary mask and bitwise operations. The segmented coins are cropped using their bounding box and saved as individual images.

Counting the Coins The total number of detected coins is displayed on the image using cv2.putText(). The count is also printed in the terminal.






##How to Use

For Image Stitching
The scripts part2_stitching_method1.py and part2_stitching_method2.py stitch multiple images together to create a single panoramic image.

Steps to Use:

Place the images you want to stitch inside the panorama photos directory.
Run the script.
The output images will be saved in their respective directories:
Output Files:

For part2_stitching_method1.py → Output in part2_stitching_method1_output/
stitchedOutput.png
stitchedOutputProcessed.png
For part2_stitching_method2.py → Output in part2_stitching_method2_output/
panorama_result_blend.png
panorama_result_crop.png
panorama_result_inpaint.png




For Coin Detection
The Jupyter Notebook part1_coin_count.ipynb is used for this task.

Steps to Use:

Place the images inside the coin_photos directory.
Run the notebook up to the 5th block.
Depending on the result you want to see, run the respective blocks:
To visualize coin edges → Run 6th block
For coin segmentation → Run 7th block
To get only the coin count → Run 8th block





Installation

Clone the repository:

git clone https://github.com/DKRJ13/VR-Assignment1




Dependencies

Install the required libraries using:

pip install opencv-python numpy
