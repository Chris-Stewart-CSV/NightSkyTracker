import cv2
import numpy as np
import csv
import atexit # NEW: Optional import for more robust shutdown handling

# ... (The track_video_outliers function definition remains the same as before, 
#      but we will rename it and move the file saving logic outside.) ...


def process_video_frames(video_path):
    """
    Tracks dots in the video and returns the list of logged moving dots.
    The file saving logic is moved out of this function.
    """
    
    # --- DATA STORAGE INITIALIZATION ---
    moving_dot_log = [] 
    
    # --- 1. Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return moving_dot_log

    # Read the first frame... (remaining initialization logic as before)
    # ...
    ret, prev_frame_bgr = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return moving_dot_log
    
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3
    )
    
    if prev_points is None or len(prev_points) < 4:
        print("Not enough features detected in the first frame to establish a motion model.")
        cap.release()
        return moving_dot_log

    print(f"Successfully detected {len(prev_points)} initial features.")
    
    cv2.namedWindow('Moving Dot Tracker (Red = Moving, Green = Stationary)', cv2.WINDOW_NORMAL)

    # --- 3. Start Frame Processing Loop ---
    frame_count = 1
    
    try: # <--- NEW: Start the try block
        while(cap.isOpened()):
            ret, next_frame_bgr = cap.read()
            frame_count += 1
            
            if not ret:
                print("End of video or read error.")
                break

            next_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # ... (The rest of the feature tracking, RANSAC, and outlier isolation logic) ...

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
                prev_points = cv2.goodFeaturesToTrack(next_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)
                prev_gray = next_gray
                if prev_points is not None:
                    print(f"Re-detected {len(prev_points)} features.")
                continue

            H, mask = cv2.findHomography(
                prev_tracked_points, next_tracked_points, cv2.RANSAC, ransacReprojThreshold=5.0 
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

                # --- 6. Visualization and Output ---
                for dot in inlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 3, (0, 255, 0), -1) 

                for dot in outlier_points:
                    x, y = dot.ravel().astype(int)
                    cv2.circle(next_frame_bgr, (x, y), 5, (0, 0, 255), -1) 
                    
                print(f"Frame {frame_count}: Found {len(inlier_points)} stationary dots (Green) and {len(outlier_points)} moving dots (Red).")

            cv2.imshow('Moving Dot Tracker (Red = Moving, Green = Stationary)', next_frame_bgr)

            # Update 'prev' variables for the next iteration:
            prev_gray = next_gray.copy() 
            
            if mask is not None:
                prev_points = next_tracked_points[inlier_mask]
            else:
                prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)

            # Press 'q' to exit the loop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e: # Catches any unexpected error during processing
        print(f"\n--- Processing Interrupted by Error --- \n{e}")
        
    finally: # <--- CRUCIAL: Code here always executes, even on 'break' or error
        # --- 7. Cleanup ---
        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo processing and display windows closed.")
        
    return moving_dot_log # Return the data collected thus far


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
	video_file = 'your/path/to/video_file/here/starrynight.mp4' 
    output_data_file = 'your/path/to/ouput_data/here/moving_dot_tracks.csv'

    # 1. Process the video, trapping any interruptions gracefully
    collected_data = process_video_frames(video_file)
    
    # 2. Save the data collected up to the point of termination
    save_dot_data(collected_data, output_data_file)
