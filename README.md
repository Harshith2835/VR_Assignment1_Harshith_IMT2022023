# Computer Vision Projects
`q1.py` - Coin Detection & Counting  
`q2.py` - Panorama Stitching

## Folder Structure  
```
VR_Assignment1_Harshith_IMT2022023/
├── input_images/
│   ├── coins.jpg
│   ├── panorama_left.jpg
│   └── panorama_right.jpg
├── output_images/
│   ├── canny_edges.jpg
│   ├── contours_detected.jpg
│   ├── segmented_coins.jpg
│   ├── keypoints_1.jpg
│   ├── keypoints_2.jpg
│   └── panorama_output.jpg
├── q1.py
├── q2.py
└── README.md
```
## Requirements
```bash
pip install opencv-python-headless numpy matplotlib
```

## Part 1: Coin Detection & Counting (q1.py)
### Methodology & Explanation
This implementation detects and counts coins in an image using:
1. Adaptive thresholding and morphological operations to enhance coin regions
2. Canny edge detection with post-processing to identify clean boundaries
3. Contour analysis with circularity filtering to distinguish coins from artifacts
4. Color segmentation to isolate individual coins
### How to Run  
1. Place input image in `input_images/coins.jpg`  
2. Execute:  
```bash
python q1.py
```
### Results & Visualizations 
![Input](./input_images/coins.jpg)
*Original image*

![Coin Detection Pipeline](./output_images/contours_detected.jpg)
*Original image with detected coin contours*

![Edge Detection](./output_images/canny_edges.jpg)  
*Canny edge detection output showing coin boundaries*

![Segmentation Results](./output_images/segmented_coins.jpg)  
*Color-coded segmentation of different coins*

![Boundary Visualization](./output_images/coin_boundaries.jpg)  
*Isolated coin boundaries without internal details*

## Part 2: Panorama Stitching (q2.py)
### Methodology & Explanation
This implementation stitches two overlapping images using:
1. SIFT feature detection for scale-invariant keypoints
2. BFMatcher with Lowe's ratio test for robust feature matching
3. RANSAC algorithm for homography matrix estimation
4. Perspective warping and mask blending for seamless stitching

### How to Run  
1. Place images in `input_images/` as:  
   - `panorama_left.jpg`  
   - `panorama_right.jpg`  
2. Execute:  
```bash
python q2.py
```

### Results & Visualizations
![Left Image Features](./output_images/left_image_keypoints.jpg)  
*Left input image with detected SIFT keypoints*

![Right Image Features](./output_images/right_image_keypoints.jpg)  
*Right input image with detected SIFT keypoints*

![Final Panorama](./output_images/final_panorama.jpg)  
*Stitched panorama result with seamless blending*

## Key Algorithms
| Component          | Techniques Used                     |
|--------------------|-------------------------------------|
| Coin Detection     | Adaptive Thresholding, Morphological Ops, Contour Analysis |
| Feature Matching   | SIFT, BFMatcher, Lowe's Ratio Test  |
| Image Alignment    | RANSAC, Homography Estimation       |
| Blending           | Perspective Warping, Mask Blending  |


**Part 1 Workflow:**  
Input → Grayscale Conversion → Edge Detection → Contour Filtering → Segmentation → Counting

**Part 2 Workflow:**  
Image Pair → Feature Detection → Feature Matching → Homography Estimation → Warping → Blending → Panorama


## Results & Observations  

### Coin Detection  
- Successfully detected 4 coins in sample image  
- Edge detection preserved circular boundaries  
- Morphological ops removed small artifacts  

### Image Stitching  
- SIFT found 5000+ keypoints per image  
- RANSAC eliminated outlier matches  
- Final panorama shows seamless transition 

