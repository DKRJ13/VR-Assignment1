# VR-Assignment1

Description

1)Image Stitching : This code is used to stitch images with overlapping features together. I have tried two approaches for stitching images.
  a)One used stitcher class from the cv2 library.
  b)Second one uses SIFT/ORB to detect key points. After this step a homography matrix is computed which allows adjacent images in the panaroma to be warped into a single image.
  This step happens iteratively till all the images have been stitched.After this step we remove the black borders which appear because of the perspective differences of the
  individual images.


2)Coin Edge detection and Segmentation : This code is used to detect segment and count coins using computer vision techniques.
  Some of the techniques used are:
  
  Image Preprocessing:The input image is converted to grayscale to simplify processing. Gaussian blur is applied to reduce noise and improve edge detection.
  Edge Detection: The Canny Edge Detection and Morphological operations like dilation and closing are used
  Contour Detection: The contours of the coins are identified using cv2.findContours().
  Segmentation and Cropping:Each detected coin is segmented using a binary mask and extracted using bitwise operations.The extracted coins are cropped based on their bounding box and saved as individual images.
  Counting the Coins:The total number of detected coins is counted and displayed on the image using cv2.putText().The count is also printed in the terminal.


How to use:

For Image Stitching 
The part2_stitching_method1.py and part2_stitching_method2.py scripts stitch multiple images together to create a single panoramic image. To use this script:
Place the images you want to stitch in the panaroma photos directory.
Run the script.
The stitched image for part2_stitching_method1.py is saved in part2_stitching_method1_output folder. Two files will be present stitchedOutput.png and stitchedOutputProcessed.png .
The stitched image for part2_stitching_method2.py is saved in part2_stitching_method2_output folder. Three files will be present.
a)panorama_result_blend.png b)panorama_result_crop.png c)panorama_result_inpaint.png


For Coin Detection
The jupyter notebook part1_coin_count.ipynb is used for this task.
Place the images you want to stitch in the coin_photos directory.
Run the notebook till 5th block.
Depending on the result you want to see run differnt block.
To visualize coins using edge detection : 6th block
For coin segmentation : 7th block
To just get the number of coins: 8th block

Installation
git clone https://github.com/DKRJ13/VR-ssignment

Dependencies
pip install opencv-python numpy



