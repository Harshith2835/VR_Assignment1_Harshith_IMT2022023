```markdown
# Computer Vision Projects
`q1.py` - Coin Detection & Counting  
`q2.py` - Panorama Stitching

## Requirements
```bash
# Create virtual environment (recommended)
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
.\cv_env\Scripts\activate   # Windows

# Install dependencies
pip install opencv-python-headless numpy matplotlib
```

## Part 1: Coin Detection & Counting (q1.py)
### Methodology
1. Preprocessing: Resize + Grayscale conversion
2. Edge Detection: Gaussian blur → Adaptive threshold → Morphological operations → Canny edge
3. Contour Analysis: Filter by circularity (4πA/P²) & area
4. Segmentation: Extract individual coins with mask operations

### How to Run
1. Directory structure:
```
.
├── input_images/
│   └── coins.jpg      # Input image
├── output_images/     # Auto-created
```

2. Execute:
```bash
python q1.py
```

3. Outputs:
- Canny edge detection result
- Contour boundaries visualization 
- Color-segmented coins
- Individual coin ROIs

![Coin Detection Demo](https://via.placeholder.com/600x200.png?text=Coin+Detection+Visualization)

## Part 2: Panorama Stitching (q2.py)
### Methodology
1. Feature Detection: SIFT keypoints
2. Feature Matching: BFMatcher with ratio test
3. Homography: RANSAC for robust matrix estimation
4. Warping: Perspective transformation
5. Blending: Mask-based image composition

### How to Run
1. Directory structure:
```
.
├── input_images/
│   ├── panaroma_left.jpg
│   └── panaroma_right.jpg
├── output_images/     # Auto-created
```

2. Execute:
```bash
python q2.py
```

3. Outputs:
- Keypoints visualization for both images
- Final stitched panorama
- Matching keypoints count

![Panorama Demo](https://via.placeholder.com/600x200.png?text=Panorama+Stitching+Process)

## Key Algorithms
| Component          | Techniques Used                     |
|--------------------|-------------------------------------|
| Coin Detection     | Adaptive Thresholding, Morphological Ops, Contour Analysis |
| Feature Matching   | SIFT, BFMatcher, Lowe's Ratio Test  |
| Image Alignment    | RANSAC, Homography Estimation       |
| Blending           | Perspective Warping, Mask Blending  |

## Expected Output Structure
```bash
output_images/
├── canny_edges.jpg         # q1 output
├── coin_boundaries.jpg     # q1 output
├── contours_detected.jpg   # q1 output
├── segmented_coins.jpg     # q1 output
├── left_image_keypoints.jpg # q2 output
├── right_image_keypoints.jpg # q2 output
└── final_panorama.jpg      # q2 output
```

## Troubleshooting
1. Image Path Errors:
   - Verify images are in `input_images/`
   - Check filenames match exactly
2. Package Issues:
   - Ensure using OpenCV 4.x+
   - Use `opencv-python-headless` if GUI issues occur
3. Stitching Failures:
   - Ensure 30-50% image overlap
   - Try swapping image order

---

*Note: Actual images will appear in matplotlib windows during execution. Outputs are saved in both PNG format and displayed interactively.*
```

To use this README:

1. Save as `README.md` in your project root
2. Replace placeholder URLs with actual screenshots
3. Adjust filenames/paths if different from default structure

The markdown includes:
- Installation instructions
- Per-project methodology
- Visual workflow diagrams
- Directory structure requirements
- Command-line execution guide
- Output expectations
- Troubleshooting common issues
