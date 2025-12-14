import cv2
import numpy as np
import csv
import glob # For easily getting a list of sequential files

def process_image_sequence(image_paths, output_csv_path='moving_dots_image_data.csv'):
    """
    Processes a list of images sequentially to track features, identify outliers, 
    and guarantees saving of collected data to CSV on exit.
    
    Args:
        image_paths (list): A sorted list of image file paths.
        output_csv_path (str): Path to save the output CSV file.
    """
    
    if not image_paths:
        print("Error: No image paths provided.")
        return

    # --- DATA STORAGE INITIALIZATION ---
    moving_dot_log = [] 
    
    # --- INITIALIZATION (First Image) ---
    prev_frame_bgr = cv2.imread(image_paths[0])
    if prev_frame_bgr is None:
        print(f"Error: Could not read the first image: {image_paths[0]}")
        return

    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Define parameters for detection (using refined values for stability)
    QUALITY_LEVEL = 0.05
    MIN_DISTANCE = 15
    RANSAC_THRESHOLD = 2.0
    WINDOW_SIZE = (21, 21)
    
    # Feature Detection (Shi-Tomasi)
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=3
    )
    
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected in the first image to establish a motion model.")
        return

    print(f"Successfully detected {len(prev_points)} initial features in {image_paths[0]}")
    
    # --- ADJUSTABLE WINDOW FIX ---
    window_name = 'Image Sequence Tracker (Red = Moving, Green = Stationary)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- Start Image Processing Loop ---
    # Start loop from the second image (index 1)
    try:
        for i, next_image_path in enumerate(image_paths[1:], start=2):
            next_frame_bgr = cv2.imread(next_image_path)
            if next_frame_bgr is None:
                print(f"Warning: Could not read image {next_image_path}. Skipping.")
                continue

            next_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # --- Feature Tracking (Lucas-Kanade Optical Flow) ---
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_points, None, 
                winSize=WINDOW_SIZE, maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            status_mask = status.flatten() == 1
            prev_tracked_points = prev_points[status_mask]
            next_tracked_points = next_points[status_mask]

            if len(prev_tracked_points) < 4:
                print(f"Image {i}: Not enough tracked points. Re-detecting features.")
                # Re-detection uses the same quality parameters
                prev_points = cv2.goodFeaturesToTrack(next_gray, maxCorners=1000, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=3)
                prev_gray = next_gray
                if prev_points is not None:
                    print(f"Re-detected {len(prev_points)} features.")
                continue

            # --- Motion Model Estimation and Outlier Isolation (RANSAC) ---
            H, mask = cv2.findHomography(
                prev_tracked_points, next_tracked_points, cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
            )

            if mask is not None:
                inlier_mask = mask.flatten() == 1
                outlier_mask = mask.flatten() == 0
                
                outlier_points = next_tracked_points[outlier_mask]
                inlier_points = next_tracked_points[inlier_mask]
                
                # --- LOG THE MOVING DOTS DATA ---
                for dot in outlier_points:
                    x, y = dot.ravel()
                    moving_dot_log.append({
                        'frame_id': i,
                        'x_coord': x,
                        'y_coord': y
                    })

                # --- Visualization ---
                for dot in inlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 3, (0, 255, 0), -1) # Green = Stationary

                for dot in outlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 5, (0, 0, 255), -1) # Red = Moving
                    
                print(f"Image {i}: Found {len(inlier_points)} stationary (Green) and {len(outlier_points)} moving (Red) dots.")

            cv2.imshow(window_name, next_frame_bgr)

            # Update 'prev' variables for the next iteration:
            prev_gray = next_gray.copy() 
            if mask is not None:
                prev_points = next_tracked_points[inlier_mask]
            else:
                prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=3)

            # Wait 1ms or press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e: 
        print(f"\n--- Processing Interrupted by Error --- \n{e}")
        
    finally: # GUARANTEED EXECUTION: Cleanup and Data Saving
        cv2.destroyAllWindows()
        print("\nImage processing and display windows closed.")
        
        # --- Save the collected data ---
        if moving_dot_log:
            keys = moving_dot_log[0].keys()
            try:
                with open(output_csv_path, 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(moving_dot_log)
                print(f"✅ Data saved successfully to: {output_csv_path}")
            except IOError:
                print(f"❌ Error: Could not write data to {output_csv_path}. Check file permissions.")
        else:
            print("No moving dots were detected or logged to save.")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # 🚨 1. SET THE DIRECTORY CONTAINING YOUR SEQUENTIAL IMAGES
    image_directory = 'your/path/to/images/folder/here'
    # Use a pattern to match your files (e.g., 'frame_001.png', 'frame_002.png', etc.)
    image_pattern = image_directory + 'frame_*.png' 
    
    # Get and sort the list of files to ensure correct processing order
    all_images = sorted(glob.glob(image_pattern))

    # 🚨 2. SET YOUR OUTPUT CSV PATH
    output_data_file = 'your/path/to/images/folder/here/moving_dot_tracks.csv'

    # --- EXECUTION ---
    print(f"Found {len(all_images)} images to process.")

    process_image_sequence(all_images, output_data_file)
