import cv2
from ultralytics import YOLO
import numpy as np

import matplotlib.pyplot as plt

def detect_court_homography():
    frame_points = np.array([[461,219], 
                             [93,471], 
                             [1271,588],
                             [1276,271]], dtype=np.float32)

    court_points = np.array([
    [0, 50],
    [0,0],
    [19, 17],     
    [19, 33],        
    ], dtype=np.float32)
    H, status = cv2.findHomography(frame_points, court_points)
    print("Homography Matrix H:")
    print(H)
    print("\nStatus:", status)
    
    # Test: Transform the original frame points and see if we get court points back
    print("\n" + "="*60)
    print("Testing Homography Transformation:")
    print("="*60)
    
    for i, point in enumerate(frame_points):
        # Reshape for perspectiveTransform (needs shape (1, 1, 2))
        test_point = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(test_point, H)
        
        expected = court_points[i]
        actual = transformed[0][0]
        print(f"\nPoint {i+1}:")
        print(f"  Frame coords:    ({point[0]:.0f}, {point[1]:.0f})")
        print(f"  Expected court:  ({expected[0]:.1f}, {expected[1]:.1f})")
        print(f"  Actual court:    ({actual[0]:.1f}, {actual[1]:.1f})")
        print(f"  Error:           ({abs(expected[0]-actual[0]):.3f}, {abs(expected[1]-actual[1]):.3f})")
        
    return H

def visualize_frame_points():
    """Draw the frame points on the actual video frame to verify their positions"""
    cap = cv2.VideoCapture("jabarismith.mp4")
    ret, frame = cap.read()
    cap.release()
    
    frame_points = np.array([
        [1014, 428],
        [112, 1048],
        [1338, 862],
        [1524, 660]
    ], dtype=np.int32)
    
    # Draw the points
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different color for each
    labels = ["P1: [1014,428]", "P2: [112,1048]", "P3: [1338,862]", "P4: [1524,660]"]
    
    for i, (pt, color, label) in enumerate(zip(frame_points, colors, labels)):
        cv2.circle(frame, tuple(pt), 15, color, -1)
        cv2.putText(frame, label, (pt[0]+20, pt[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw lines connecting them
    cv2.polylines(frame, [frame_points], isClosed=True, color=(255, 255, 255), thickness=3)
    
    # Save the image
    cv2.imwrite("frame_points_check.jpg", frame)
    print("✓ Saved frame_points_check.jpg - check if the points are where you expect!")
    
    return frame



def filter_crowd_detections(x1, y1, x2, y2, H):
    """Check if a detection is on the court or in the crowd"""
    # Use bottom-center of bounding box (player's feet)
    foot_x = (x1 + x2) / 2
    foot_y = y2
    
    # Transform to court coordinates
    foot_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(foot_point, H)
    
    court_x, court_y = transformed[0][0]
    
    # Check if within court bounds (0-50 ft wide, 0-19 ft deep)
    if 0 <= court_x <= 19 and 0 <= court_y <= 50:
        return True  # On court
    else:
        return False  # In crowd

def detect_basketball_objects(video_path, H, output_path='output_detection.mp4'):
    """
    Detect players, ball, and hoop in basketball footage using YOLO
    
    Args:
        video_path: Path to input video file
        output_path: Path for output annotated video
    """
    
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print("Press 'q' to quit early\n")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False, conf = 0.50, imgsz= 1280, iou=0.7)
        
        # Get detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if cls == 0:  # Person
                    if not filter_crowd_detections(x1, y1, x2, y2, H):
                        continue
                    
                    color = (0, 255, 0)
                    label = f"Player {conf:.2f}"
                    
                elif cls == 32:  # Sports ball
                    color = (0, 165, 255)
                    label = f"Ball {conf:.2f}"
                else:
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Processing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    



    
if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "jabarismith.mp4"  # Change this to your video path
    OUTPUT_VIDEO = "output_detection.mp4"
    
    print("="*60)
    print("Basketball Shot Detector - Proof of Concept")
    print("="*60)
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}\n")
    
    # Run detection
    H = detect_court_homography()
    detect_basketball_objects(INPUT_VIDEO, H,OUTPUT_VIDEO)
    # Add this to your main
    #
    #accurate_points = get_accurate_frame_points("jabarismith.mp4")


    
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. Review output_detection.mp4 to see detection quality")
    print("2. Check if players and ball are detected consistently")
    print("3. Note any issues (missed detections, false positives)")
    print("="*60)