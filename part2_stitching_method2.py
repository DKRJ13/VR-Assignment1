import os
import cv2
import numpy as np

# Choose method to handle black borders: "crop", "inpaint", or "blend"
BLACK_BORDER_REMOVAL_METHOD = "blend"  # Options: "crop", "inpaint", "blend"

def load_images(folder_path):
    """Load all images from a folder."""
    image_paths = sorted([os.path.join(folder_path, f) 
                          for f in os.listdir(folder_path) 
                          if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]
    return images

def detect_features(images, method="SIFT"):
    """Detect keypoints and compute descriptors using ORB or SIFT."""
    method = method.upper()
    detector = cv2.SIFT_create() if method == "SIFT" else cv2.ORB_create(nfeatures=2000)

    keypoints_list, descriptors_list = [], []
    for img in images:
        kp, des = detector.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    return keypoints_list, descriptors_list

def match_keypoints(descriptors_list, method="SIFT", ratio_thresh=0.6):
    """Match keypoints between consecutive images."""
    bf = cv2.BFMatcher(cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING)
    
    matches_list = []
    for i in range(len(descriptors_list) - 1):
        if descriptors_list[i] is None or descriptors_list[i+1] is None:
            matches_list.append(None)
            continue
        
        raw_matches = bf.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
        good_matches = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]

        matches_list.append(good_matches if len(good_matches) >= 4 else None)

    return matches_list

def compute_homographies(matches_list, keypoints_list):
    """Compute cumulative homographies from all images to the reference image (first image)."""
    homographies = [np.eye(3)]  # First image has identity homography

    for i, matches in enumerate(matches_list):
        if matches is None:
            print(f"Skipping homography computation for image {i+1}. Not enough matches.")
            homographies.append(None)
            continue

        src_pts = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            homographies.append(H @ homographies[-1])  # Multiply with previous homography
        else:
            homographies.append(None)

    return homographies

def warp_images(images, homographies):
    """Warp images using computed homographies and blend them."""
    ref_h, ref_w = images[0].shape[:2]

    # Compute final panorama size
    corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, homographies[i])
        corners.append(transformed_pts)

    all_corners = np.vstack(corners)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Adjust translation to keep everything in positive coordinates
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama_size = (x_max - x_min, y_max - y_min)

    # Initialize panorama canvas
    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)

    for i, img in enumerate(images):
        warped_img = cv2.warpPerspective(img, translation @ homographies[i], panorama_size)

        # Ensure mask has correct size & type
        mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Use proper blending (avoid bitwise_or issue)
        panorama[mask > 0] = warped_img[mask > 0]

    return panorama

### **Methods to Remove Black Borders** ###
def remove_black_borders(image):
    """Auto-crop the stitched image to remove black areas."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the largest contour (non-black region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image  # Return original if no contours found

def fill_black_regions(image):
    """Use inpainting to fill black regions in the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype(np.uint8) * 255
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def blend_black_regions(image):
    """Resize and blend the image with itself to reduce black borders."""
    height, width = image.shape[:2]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(image, 0.7, resized, 0.3, 0)

def apply_black_border_removal(image, method="blend"):
    """Applies the selected black border removal technique."""
    if method == "crop":
        return remove_black_borders(image)
    elif method == "inpaint":
        return fill_black_regions(image)
    elif method == "blend":
        return blend_black_regions(image)
    return image  # Default return original image

def main():
    input_folder = "panorama photos"
    output_dir = "part2_stitching_method2_output"
    os.makedirs(output_dir, exist_ok=True)

    images = load_images(input_folder)
    if len(images) < 2:
        print("Not enough images to stitch.")
        return

    print(f"Loaded {len(images)} images.")

    keypoints_list, descriptors_list = detect_features(images, method="SIFT")
    matches_list = match_keypoints(descriptors_list, method="SIFT", ratio_thresh=0.6)
    homographies = compute_homographies(matches_list, keypoints_list)

    panorama = warp_images(images, homographies)

    # Apply selected black border removal method
    panorama = apply_black_border_removal(panorama, method=BLACK_BORDER_REMOVAL_METHOD)

    output_file = os.path.join(output_dir, f"panorama_result_{BLACK_BORDER_REMOVAL_METHOD}.jpg")
    cv2.imwrite(output_file, panorama)
    print(f"Panorama saved as {output_file}")

if __name__ == "__main__":
    main()
