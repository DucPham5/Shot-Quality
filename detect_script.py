import cv2
from ultralytics import YOLO
import numpy as np

def detect_basketball_objects(video_path, output_path='output_detection.mp4'):
    """
    Detect players, ball, and hoop in basketball footage using YOLO
    
    Args:
        video_path: Path to input video file
        output_path: Path for output annotated video
    """
    
    # Load YOLOv8 model (using pretrained COCO model)
    # COCO classes include: person (0), sports ball (32)
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # 'n' is nano (fastest), can use 's', 'm', 'l', 'x' for more accuracy
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
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
        results = model(frame, verbose=False, conf = 0.15, imgsz= 1280, iou=0.7)
        
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
                
                # Filter for relevant objects
                # 0 = person, 32 = sports ball
                if cls == 0:  # Person (player)
                    color = (0, 255, 0)  # Green for players
                    label = f"Player {conf:.2f}"
                elif cls == 32:  # Sports ball
                    color = (0, 165, 255)  # Orange for ball
                    label = f"Ball {conf:.2f}"
                else:
                    continue  # Skip other objects
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
        
        # Optional: Display frame (comment out for faster processing)
        # cv2.imshow('Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Cleanup
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
    detect_basketball_objects(INPUT_VIDEO, OUTPUT_VIDEO)
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("1. Review output_detection.mp4 to see detection quality")
    print("2. Check if players and ball are detected consistently")
    print("3. Note any issues (missed detections, false positives)")
    print("="*60)