import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def setup_output_directory():
    # Create output directory if it doesn't exist
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_preprocess_image(image_path, resize=True):
    # Load image and convert to different color spaces
    image = cv2.imread(image_path)
    if resize:
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for processing
    return image, rgb_image, gray_image

def detect_edges(gray_image):
    # Apply stronger blur to reduce internal coin details
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 3)
    
    # Run adaptive thresholding to better separate coins from background
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # Apply morphological operations to close gaps and remove small details
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Now apply Canny with parameters that focus on strong edges only
    edges = cv2.Canny(morph, 50, 150)
    
    # Apply dilation to connect any broken edges
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated_edges

def find_coin_contours(edge_image):
    # Find contours from the edge image
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and circularity
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity: 4*pi*area/perimeter^2
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter based on area and circularity (coins are typically circular)
        if area > 200 and circularity > 0.7:
            filtered_contours.append(contour)
    
    return filtered_contours

def segment_coins(original_image, contours):
    # Initialize output variables
    segmented_coins = []
    masked_image = np.zeros_like(original_image)
    
    for i, contour in enumerate(contours):
        # Create a mask for this contour
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask to original image to extract this coin
        coin_segment = cv2.bitwise_and(original_image, original_image, mask=mask)
        
        # Add this segment to the full masked image
        masked_image = cv2.add(masked_image, coin_segment)
        
        # Extract the ROI for the individual coin
        x, y, w, h = cv2.boundingRect(contour)
        coin_roi = coin_segment[y:y+h, x:x+w]
        
        # Only add non-empty ROIs
        if coin_roi.size > 0 and np.sum(coin_roi) > 0:
            segmented_coins.append(coin_roi)
    
    return masked_image, segmented_coins

def create_colored_segments(gray_image, contours):
    # Create a blank colored image for segmentation visualization
    height, width = gray_image.shape
    segmented_viz = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate distinct colors for each coin
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    # Fill each contour with a different color
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        segmented_viz[mask == 255] = color
    
    return segmented_viz

def create_boundary_visualization(shape, contours, color=(255, 165, 0)):
    """Create an image showing only the boundaries of coins without internal details"""
    # Create a blank image (black background)
    height, width = shape[:2]
    boundary_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw only the contours (outlines)
    cv2.drawContours(boundary_image, contours, -1, color, 2)
    
    return boundary_image

def visualize_results(original, edges, contours, output_dir):
    # Convert edges to color (orange) for better visualization
    orange_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    orange_edges[edges > 0] = [255, 165, 0]  # Orange color for edges
    
    # Create contour visualization (green contours on original image)
    contour_image = original.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Create segmentation visualization (filled colored regions)
    segmented_viz = create_colored_segments(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), contours)
    
    # Create boundaries-only visualization (orange outlines on black background)
    boundary_image = create_boundary_visualization(original.shape, contours, color=(255, 165, 0))
    
    # Display: Original, Canny edges, and coin boundaries
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Canny Edges')
    plt.imshow(orange_edges)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Coin Boundaries Only')
    plt.imshow(boundary_image)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Display: Contour detection and segmentation
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Detected Coins (Contours)')
    plt.imshow(contour_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Coins")
    plt.imshow(segmented_viz)
    plt.axis("off")
    
    plt.tight_layout()
    cv2.imwrite(os.path.join(output_dir, "canny_edges.jpg"), 
               cv2.cvtColor(orange_edges, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "contours_detected.jpg"), 
               cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "coin_boundaries.jpg"), 
               cv2.cvtColor(boundary_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "segmented_coins.jpg"), 
               cv2.cvtColor(segmented_viz, cv2.COLOR_RGB2BGR))
    
    plt.show()

def detect_and_count_coins(image_path):
    # Set up directory for saving output images
    output_dir = setup_output_directory()
    
    # Load and preprocess image
    original_cv, original_rgb, gray = load_and_preprocess_image(image_path)
    
    # Detect edges in the image
    edges = detect_edges(gray)
    
    # Find contours with filtering for coin-like shapes
    contours = find_coin_contours(edges)
    
    # Count total coins detected
    coin_count = len(contours)
    print(f"Total Coins Detected: {coin_count}")
    
    # Visualize and save results
    visualize_results(original_rgb, edges, contours, output_dir)
    
    # Extract individual coin segments (for future use)
    _, individual_coins = segment_coins(original_rgb, contours)
    
    return coin_count, contours, individual_coins

if __name__ == "__main__":
    image_path = "./input_images/coins.jpg"  # Update with your image path
    coin_count, contours, segments = detect_and_count_coins(image_path)