import cv2
import numpy as np

def track_video_outliers(video_path):
    """
    Tracks white dots in a video, identifies the dominant motion, and highlights 
    dots that do not conform to that motion (outliers).
    """
    
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
    # Find strong corners/dots in the initial frame using Shi-Tomasi
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000,          # Max dots to track
        qualityLevel=0.01,        # Quality of corners
        minDistance=10,           # Minimum distance between dots
        blockSize=3
    )
    
    # Ensure enough initial features are detected
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected in the first frame to establish a motion model.")
        cap.release()
        return

    print(f"Successfully detected {len(prev_points)} initial features.")

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
        # Calculate optical flow to find the new position of the previous points
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, 
            next_gray, 
            prev_points, 
            None, 
            winSize=(15, 15), 
            maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
		
		# Flatten the status array for correct boolean indexing
        status_mask = status.flatten() == 1

        # Filter points that were successfully tracked
        prev_tracked_points = prev_points[status_mask]
        next_tracked_points = next_points[status_mask]

        if len(prev_tracked_points) < 4:
             # Need at least 4 points to estimate a Homography
             print(f"Frame {frame_count}: Not enough tracked points. Re-detecting features.")
             # Re-detect features in the current frame to continue tracking
             prev_points = cv2.goodFeaturesToTrack(next_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)
             prev_gray = next_gray
             continue

        # --- 5. Motion Model Estimation and Outlier Isolation (RANSAC) ---
        H, mask = cv2.findHomography(
            prev_tracked_points, 
            next_tracked_points, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0 
        )

        if mask is not None:
            # Mask: 1 = Inlier (Stationary), 0 = Outlier (Moving)
            outlier_mask = (mask == 0) # Invert the mask to select the outliers
            outlier_points = next_tracked_points[outlier_mask.flatten()]
            
            # --- 6. Visualization and Output ---
            
            # Draw Inliers (Stationary Dots) in Green
            inlier_points = next_tracked_points[mask.flatten() == 1]
            for dot in inlier_points:
                x, y = dot.ravel().astype(int)
                cv2.circle(next_frame_bgr, (x, y), 3, (0, 255, 0), -1) # Green circle

            # Draw Outliers (Moving Dots) in Red
            for dot in outlier_points:
                x, y = dot.ravel().astype(int)
                cv2.circle(next_frame_bgr, (x, y), 5, (0, 0, 255), -1) # Red circle
                
            # Print status update
            print(f"Frame {frame_count}: Found {len(inlier_points)} stationary dots (Green) and {len(outlier_points)} moving dots (Red).")

        # Display the result
        cv2.imshow('Moving Dot Tracker (Red = Moving, Green = Stationary)', next_frame_bgr)

        # Update 'prev' variables for the next iteration:
        # The new 'prev_gray' is the current 'next_gray'
        prev_gray = next_gray.copy() 
        # The new 'prev_points' are the points that were successfully tracked AND were INLIERS (stationary)
        # This keeps the model stable and prevents the moving points from polluting the next tracking iteration.
        prev_points = next_tracked_points[mask.flatten() == 1]
        
        # Press 'q' to exit the loop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 🚨 REPLACE 'path/to/your/video.mp4' with the actual path to your video file.
    video_file = 'your/path/to/video_file/here/starrynight.mp4' 

    track_video_outliers(video_file)
