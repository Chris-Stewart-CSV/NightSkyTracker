import cv2
import numpy as np
import csv # <--- NEW: Import the CSV module

def track_video_outliers(video_path, output_csv_path='moving_dots_data.csv'): # <--- CHANGED: Added CSV path argument
    """
    Tracks white dots in a video, identifies outliers, and logs their coordinates.
    """
    
    # --- DATA STORAGE INITIALIZATION ---
    # NEW: List to hold all the data points from every frame
    moving_dot_log = [] 
    
    # --- 1. Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Read the first frame
    ret, prev_frame_bgr = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Convert to grayscale for faster processing
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Feature Detection (Initialize Points) ---
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3
    )
    
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected in the first frame to establish a motion model.")
        cap.release()
        return

    print(f"Successfully detected {len(prev_points)} initial features.")
    
    # --- 💡 Resizable Window Fix ---
    cv2.namedWindow('Moving Dot Tracker (Red = Moving, Green = Stationary)', cv2.WINDOW_NORMAL)

    # --- 3. Start Frame Processing Loop ---
    frame_count = 1
    while(cap.isOpened()):
        ret, next_frame_bgr = cap.read()
        frame_count += 1
        
        if not ret:
            print("End of video or read error.")
            break

        next_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- 4. Feature Tracking (Lucas-Kanade Optical Flow) ---
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, prev_points, None, 
            winSize=(15, 15), maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        status_mask = status.flatten() == 1
        prev_tracked_points = prev_points[status_mask]
        next_tracked_points = next_points[status_mask]

        if len(prev_tracked_points) < 4:
             print(f"Frame {frame_count}: Not enough tracked points. Re-detecting features.")
             # ... (re-detection logic remains the same) ...
             continue

        # --- 5. Motion Model Estimation and Outlier Isolation (RANSAC) ---
        H, mask = cv2.findHomography(
            prev_tracked_points, next_tracked_points, cv2.RANSAC, ransacReprojThreshold=5.0 
        )

        if mask is not None:
            inlier_mask = mask.flatten() == 1
            outlier_mask = mask.flatten() == 0
            
            outlier_points = next_tracked_points[outlier_mask]
            inlier_points = next_tracked_points[inlier_mask]
            
            # --- NEW: LOG THE MOVING DOTS DATA ---
            # Loop through each moving dot found in the current frame
            for dot in outlier_points:
                x, y = dot.ravel()
                moving_dot_log.append({
                    'frame_id': frame_count,
                    'x_coord': x,
                    'y_coord': y
                })

            # --- 6. Visualization and Output ---
            # ... (Drawing logic remains the same) ...
            for dot in inlier_points:
                x, y = dot.ravel().astype(int)
                cv2.circle(next_frame_bgr, (x, y), 3, (0, 255, 0), -1) 

            for dot in outlier_points:
                x, y = dot.ravel().astype(int)
                cv2.circle(next_frame_bgr, (x, y), 5, (0, 0, 255), -1) 
                
            print(f"Frame {frame_count}: Found {len(inlier_points)} stationary dots (Green) and {len(outlier_points)} moving dots (Red).")

        # Display the result
        cv2.imshow('Moving Dot Tracker (Red = Moving, Green = Stationary)', next_frame_bgr)

        # Update 'prev' variables for the next iteration:
        prev_gray = next_gray.copy() 
        
        if mask is not None:
            prev_points = next_tracked_points[inlier_mask]
        else:
            prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup and FILE WRITING ---
    cap.release()
    cv2.destroyAllWindows()
    
    # NEW: Save the results to the specified CSV file
    if moving_dot_log:
        keys = moving_dot_log[0].keys()
        with open(output_csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(moving_dot_log)
        print(f"\n✅ Data saved successfully to: {output_csv_path}")
    else:
        print("\nNo moving dots were detected to save.")


if __name__ == '__main__':
    video_file = 'your/path/to/video_file/here/starrynight.mp4' 
    output_data_file = 'your/path/to/ouput_data/here/moving_dot_tracks.csv' # Set your desired output path
    
    track_video_outliers(video_file, output_data_file)
