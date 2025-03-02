import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_panorama(images):
    # Convert images to grayscale for feature detection only
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors for each image
    keypoints = []
    descriptors = []
    
    for gray in gray_images:
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
    
    # Create images with keypoints drawn on them
    images_with_keypoints = []
    for i in range(len(images)):
        img_with_kp = cv2.drawKeypoints(images[i], keypoints[i], None, 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        images_with_keypoints.append(img_with_kp)
    
    # Match features between images
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
    
    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints[0][match.queryIdx].pt
        points2[i, :] = keypoints[1][match.trainIdx].pt
    
    # Find homography matrix
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    # Use homography matrix to warp and stitch images
    h1, w1 = images[0].shape[:2]
    h2, w2 = images[1].shape[:2]
    
    # Calculate the dimensions of the final panorama
    # Find the corners of the second image in the first image's space
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)
    
    # Find minimum and maximum coordinates
    all_corners = np.concatenate((np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2), corners2_transformed))
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Create translation matrix to shift the image to positive coordinates
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    H_adjusted = translation @ H
    
    # Create the final panorama with the correct size
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    
    # Warp both images to the new panorama space
    cv2.warpPerspective(images[1], H_adjusted, (panorama_width, panorama_height), 
                        dst=panorama, borderMode=cv2.BORDER_TRANSPARENT)
    
    # Create a mask for the first image
    mask1 = np.zeros((panorama_height, panorama_width), dtype=np.uint8)
    warped_first = np.zeros_like(panorama)
    
    # Apply the same translation to the first image - FIX: Ensure H1 is float32
    H1 = translation.copy()  # This will now inherit the float32 type
    cv2.warpPerspective(images[0], H1, (panorama_width, panorama_height), 
                        dst=warped_first, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8), H1, (panorama_width, panorama_height), 
                        dst=mask1, borderMode=cv2.BORDER_TRANSPARENT)
    
    # Create the final blended panorama
    mask1_3d = np.stack([mask1] * 3, axis=2)
    panorama = np.where(mask1_3d > 0, warped_first, panorama)
    
    # Crop black edges if needed
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (the main content area)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the panorama to remove black borders
        panorama = panorama[y:y+h, x:x+w]
    
    return images_with_keypoints, panorama, good_matches, keypoints

def main():
    # Create output directory if it doesn't exist
    import os
    output_dir = "./output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load your overlapping images
    left_img = cv2.imread('./input_images/panaroma_left.jpg')
    right_img = cv2.imread('./input_images/panaroma_right.jpg')
    
    # Check if images were loaded
    if left_img is None or right_img is None:
        print("Error: Could not load one or more images. Please check the file paths.")
        return
    
    # Create a list of images to stitch
    images = [left_img, right_img]
    
    # Get the images with keypoints and the final panorama
    images_with_keypoints, panorama, good_matches, keypoints = create_panorama(images)
    
    # Save images to output directory
    cv2.imwrite('./output_images/left_image_keypoints.jpg', images_with_keypoints[0])
    cv2.imwrite('./output_images/right_image_keypoints.jpg', images_with_keypoints[1])
    cv2.imwrite('./output_images/final_panorama.jpg', panorama)
    
    # Print information about keypoints and matches
    print(f"Total number of keypoints detected in left image: {len(keypoints[0])}")
    print(f"Total number of keypoints detected in right image: {len(keypoints[1])}")
    print(f"Number of good matches found: {len(good_matches)}")
    print("Panorama created successfully!")
    
    # Display images in separate windows
    # Convert images to RGB for display
    left_keypoints_rgb = cv2.cvtColor(images_with_keypoints[0], cv2.COLOR_BGR2RGB)
    right_keypoints_rgb = cv2.cvtColor(images_with_keypoints[1], cv2.COLOR_BGR2RGB)
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    
    # Display left image with keypoints
    plt.figure("Left Image with Keypoints")
    plt.imshow(left_keypoints_rgb)
    plt.axis('off')
    plt.title('Left Image with Keypoints')
    
    # Display right image with keypoints
    plt.figure("Right Image with Keypoints")
    plt.imshow(right_keypoints_rgb)
    plt.axis('off')
    plt.title('Right Image with Keypoints')
    
    # Display final panorama
    plt.figure("Final Panorama")
    plt.imshow(panorama_rgb)
    plt.axis('off')
    plt.title('Final Panorama')
    
    plt.show()

if __name__ == "__main__":
    main()