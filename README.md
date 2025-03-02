
---

# **VR-Assignment1**

## **Description**

### **1) Image Stitching**  
This code is used to stitch images with overlapping features together. I have tried two approaches for stitching images:  

- **Method 1:** Uses the `cv2.Stitcher` class from OpenCV.  
- **Method 2:** Uses **SIFT/ORB** to detect key points. After this step, a **homography matrix** is computed, which allows adjacent images in the panorama to be warped into a single image.  
  - This step happens iteratively until all the images have been stitched.  
  - After this step, we remove the black borders that appear due to perspective differences in the individual images.

---

### **2) Coin Edge Detection and Segmentation**  
This code is used to **detect, segment, and count** coins using computer vision techniques.  

#### **Techniques Used:**
- **Image Preprocessing:**  
  - The input image is converted to **grayscale** to simplify processing.  
  - **Gaussian blur** is applied to reduce noise and improve edge detection.  
- **Edge Detection:**  
  - **Canny Edge Detection** and **morphological operations** (dilation & closing) are used.  
- **Contour Detection:**  
  - The contours of the coins are identified using `cv2.findContours()`.  
- **Segmentation and Cropping:**  
  - Each detected coin is segmented using a **binary mask** and extracted using **bitwise operations**.  
  - The extracted coins are **cropped** based on their bounding box and **saved as individual images**.  
- **Counting the Coins:**  
  - The **total number of detected coins** is counted and displayed on the image using `cv2.putText()`.  
  - The count is also printed in the terminal.

---

## **How to Use**

### **For Image Stitching**  
The `part2_stitching_method1.py` and `part2_stitching_method2.py` scripts stitch multiple images together to create a single panoramic image.  

**Steps to run:**  
1. Place the images you want to stitch in the `panorama photos` directory.  
2. Run the script.  
3. The stitched image will be saved in the respective output folder:  

   - **For `part2_stitching_method1.py`:**  
     - Saved in **`part2_stitching_method1_output`**  
     - Output files:  
       - `stitchedOutput.png`  
       - `stitchedOutputProcessed.png`  

   - **For `part2_stitching_method2.py`:**  
     - Saved in **`part2_stitching_method2_output`**  
     - Output files:  
       - `panorama_result_blend.png`  
       - `panorama_result_crop.png`  
       - `panorama_result_inpaint.png`  

---

### **For Coin Detection**  
The Jupyter notebook `part1_coin_count.ipynb` is used for this task.  

**Steps to run:**  
1. Place the images in the `coin_photos` directory.  
2. Run the notebook **up to the 5th block**.  
3. Depending on what result you want to see, run the appropriate block:  
   - **To visualize coins using edge detection:** Run **Block 6**  
   - **For coin segmentation:** Run **Block 7**  
   - **To get the number of coins:** Run **Block 8**  

---

## **Installation**
To clone this repository and set up dependencies, run the following commands:

```bash
git clone https://github.com/DKRJ13/VR-Assignment1.git
```

---

## **Dependencies**
Install the required libraries using:

```bash
pip install opencv-python numpy
```

---

This version keeps everything the same while improving formatting for readability and emphasis. Copy-paste this into your README file, and it will maintain its structure. Let me know if you need further adjustments! 🚀
