import cv2
import numpy as np
import csv

def process_video_frames(video_path):
    """
    Tracks dots in the video, identifies and logs moving (outlier) dots 
    with refined parameters for better stability.
    Returns the list of logged moving dots.
    """
    
    # --- DATA STORAGE INITIALIZATION ---
    moving_dot_log = [] 
    
    # --- 1. Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return moving_dot_log

    # Read the first frame
    ret, prev_frame_bgr = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return moving_dot_log
    
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Feature Detection (INITIAL PARAMETERS) ---
    # Refinement 1: Increased qualityLevel and minDistance to filter weak/noisy features
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000,          
        qualityLevel=0.05,        # Adjusted from 0.01
        minDistance=15,           # Adjusted from 10
        blockSize=3
    )
    
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected in the first frame to establish a motion model.")
        cap.release()
        return moving_dot_log

    print(f"Successfully detected {len(prev_points)} initial features.")
    
    # --- Resizable Window Fix ---
    cv2.namedWindow('Moving Dot Tracker (Red = Moving, Green = Stationary)', cv2.WINDOW_NORMAL)

    # --- 3. Start Frame Processing Loop ---
    frame_count = 1
    
    try: 
        while(cap.isOpened()):
            ret, next_frame_bgr = cap.read()
            frame_count += 1
            
            if not ret:
                print("End of video or read error.")
                break

            next_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # --- 4. Feature Tracking (Lucas-Kanade Optical Flow) ---
            # Refinement 3: Increased winSize for smoother tracking of large/noisy dots
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_points, None, 
                winSize=(21, 21), # Adjusted from (15, 15)
                maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # CRITICAL FIX: Flatten status mask for correct indexing
            status_mask = status.flatten() == 1
            prev_tracked_points = prev_points[status_mask]
            next_tracked_points = next_points[status_mask]

            if len(prev_tracked_points) < 4:
                print(f"Frame {frame_count}: Not enough tracked points. Re-detecting features.")
                # Re-detection must use the same refined parameters
                prev_points = cv2.goodFeaturesToTrack(
                    next_gray, maxCorners=1000, qualityLevel=0.05, minDistance=15, blockSize=3
                )
                prev_gray = next_gray
                if prev_points is not None:
                    print(f"Re-detected {len(prev_points)} features.")
                continue

            # --- 5. Motion Model Estimation and Outlier Isolation (RANSAC) ---
            # Refinement 1: Decreased RANSAC threshold for stricter stationary definition
            H, mask = cv2.findHomography(
                prev_tracked_points, 
                next_tracked_points, 
                cv2.RANSAC, 
                ransacReprojThreshold=2.0 # Adjusted from 5.0
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
                        'frame_id': frame_count,
                        'x_coord': x,
                        'y_coord': y
                    })

                # --- 6. Visualization ---
                for dot in inlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 3, (0, 255, 0), -1) # Green = Stationary

                for dot in outlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 5, (0, 0, 255), -1) # Red = Moving
                    
                print(f"Frame {frame_count}: Found {len(inlier_points)} stationary dots (Green) and {len(outlier_points)} moving dots (Red).")

            cv2.imshow('Moving Dot Tracker (Red = Moving, Green = Stationary)', next_frame_bgr)

            # Update 'prev' variables for the next iteration:
            prev_gray = next_gray.copy() 
            
            if mask is not None:
                prev_points = next_tracked_points[inlier_mask]
            else:
                # Fallback: re-detect if RANSAC failed
                prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.05, minDistance=15, blockSize=3)

            # Press 'q' to exit the loop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e: 
        print(f"\n--- Processing Interrupted by Error --- \n{e}")
        
    finally: # CODE HERE ALWAYS EXECUTES (Guaranteed Cleanup and Data Return)
        # --- 7. Cleanup ---
        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo processing and display windows closed.")
        
    return moving_dot_log 


def save_dot_data(data_log, output_csv_path):
    """Saves the collected data to a CSV file."""
    if data_log:
        keys = data_log[0].keys()
        try:
            with open(output_csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(data_log)
            print(f"✅ Data saved successfully to: {output_csv_path}")
        except IOError:
            print(f"❌ Error: Could not write data to {output_csv_path}. Check file permissions.")
    else:
        print("No moving dots were detected or logged to save.")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # 🚨 1. SET YOUR VIDEO FILE PATH
    video_file = 'your/path/to/video_file/here/starrynight.mp4' 
    
    # 🚨 2. SET YOUR OUTPUT CSV PATH
    output_data_file = 'your/path/to/ouput_data/here/moving_dot_tracks.csv'

    # --- EXECUTION ---
    print(f"Starting processing of: {video_file}")
    
    # 1. Process the video (data collected, even on interrupt)
    collected_data = process_video_frames(video_file)
    
    # 2. Save the data collected up to the point of termination (guaranteed to run)
    save_dot_data(collected_data, output_data_file)
