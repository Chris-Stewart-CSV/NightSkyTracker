import cv2
import numpy as np

def track_and_find_outliers(prev_image, next_image):
    """
    Detects features in the previous image, tracks them in the next image,
    and uses RANSAC to identify non-stationary features (outliers).

    Args:
        prev_image (np.ndarray): The first frame (black with white dots).
        next_image (np.ndarray): The second frame.

    Returns:
        tuple: (outlier_points, H_matrix) 
               where outlier_points are the coordinates of the moving dots in the next_image,
               and H_matrix is the Homography defining the stationary motion.
    """

    # --- 1. Feature Detection (Shi-Tomasi/Good Features to Track) ---
    # Find strong corners/dots in the previous image.
    prev_points = cv2.goodFeaturesToTrack(
        prev_image,
        maxCorners=1000,          # Maximum number of dots to track
        qualityLevel=0.01,        # Minimum quality of the detected corners
        minDistance=10            # Minimum distance between corners
    )

    # If no features are found, return empty lists
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected for tracking.")
        return [], None

    # --- 2. Feature Tracking (Lucas-Kanade Optical Flow) ---
    # Track the points from the previous image to the next image.
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prev_image, 
        next_image, 
        prev_points, 
        None, 
        winSize=(15, 15), 
        maxLevel=2, 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Filter out failed tracking points
    prev_tracked_points = prev_points[status == 1]
    next_tracked_points = next_points[status == 1]

    # --- 3. Motion Model Estimation and Outlier Isolation (RANSAC) ---
    
    # We use a Homography (H) matrix since the dot field is planar.
    # RANSAC is applied to find the best H that describes the most points (inliers).
    # The 'cv2.RANSAC' flag is what performs the robust outlier filtering.
    H, mask = cv2.findHomography(
        prev_tracked_points, 
        next_tracked_points, 
        cv2.RANSAC, 
        ransacReprojThreshold=5.0  # Tolerance for a point to be considered an inlier
    )

    # 'mask' is a boolean array: 1 = Inlier (stationary), 0 = Outlier (moving)
    if mask is None:
        print("RANSAC failed to find a reliable model.")
        return [], H

    # Isolate the Outliers (the features that did NOT fit the stationary motion)
    outlier_mask = (mask == 0) # Invert the mask to select the outliers
    
    # Get the coordinates of the moving dots in the *next* frame
    outlier_points = next_tracked_points[outlier_mask.flatten()]
    
    # Optional: Get the coordinates of the stationary dots (inliers)
    # inlier_points = next_tracked_points[mask.flatten() == 1]

    return outlier_points, H

# --- Example Usage (Requires Image Loading) ---
if __name__ == '__main__':
    # NOTE: In a real program, you would loop this for a series of images (video frames)
    
    # Load your images. They should ideally be grayscale (single channel).
    # Replace these with your actual file paths
    try:
        image1 = cv2.imread('image_frame_1.jpg', cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread('image_frame_2.jpg', cv2.IMREAD_GRAYSCALE)
    except:
        print("\n*** ERROR: Could not load dummy images. Please replace placeholders with your image paths. ***")
        exit()

    if image1 is not None and image2 is not None:
        print(f"Processing images of size: {image1.shape}")
        
        moving_dots, homography_matrix = track_and_find_outliers(image1, image2)

        print("\n--- Results ---")
        if homography_matrix is not None:
            # The Homography matrix (H) describes the collective movement of the stationary dots.
            print("Stationary Motion (Homography Matrix H):\n", homography_matrix)
        
        # The coordinates of the dots you want to track
        print(f"\nFound {len(moving_dots)} non-stationary (outlier) dots.")
        if len(moving_dots) > 0:
            print("Coordinates of Moving Dots (in image 2):\n", moving_dots[:5]) # Print first 5
            
            # Draw the moving dots on the image for visual verification
            display_image = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR) # Convert to color for drawing
            for dot in moving_dots:
                x, y = dot.ravel().astype(int)
                # Draw a red circle around the moving dot
                cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1) 

            # Visualization for user    
            cv2.imshow('Moving Dots Detected (Red)', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
