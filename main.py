from ultralytics import YOLO 
import time
import cv2
import torch

if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    
    start_time = time.time()
    # Load both models
    player_model = YOLO("models/player_detector.pt")
    ball_model = YOLO("models/balldetectionv2.pt")

    # Open video
    cap = cv2.VideoCapture("jabarismith.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    out = cv2.VideoWriter(
        'output_combined.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # ========================================
        # Step 1: Run both models
        # ========================================
        player_results = player_model.track(frame, conf=0.3, tracker="botsort.yaml", persist=True, verbose=False, imgsz=1088, device='mps', agnostic_nms=True)
        ball_results = ball_model.track(frame, conf=0.1, persist=True, verbose=False, imgsz = 1088, device='mps', agnostic_nms=True)
        
        # ========================================
        # Step 2: Draw players
        # ========================================
        if player_results[0].boxes.id is not None:
            for box, track_id in zip(player_results[0].boxes.xyxy, player_results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                tid = int(track_id)
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(frame, f"#{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ========================================
        # Step 3: Draw ball
        # ========================================
        if len(ball_results[0].boxes.xyxy) > 0:
            for box, cls in zip(ball_results[0].boxes.xyxy, ball_results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                class_id = int(cls)
                class_name = ball_model.names[class_id]
                
                if class_name == 'ball':
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, center, 12, (0, 165, 255), -1)
                    cv2.circle(frame, center, 15, (255, 255, 255), 2)
                
                elif class_name == 'hoop':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Hoop", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # ========================================
        # Step 4: Write to output
        # ========================================
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Processed frame {frame_num}")

    cap.release()
    out.release()
    total_time = time.time() - start_time

    print(f"\n✓ Done! Check output_combined.mp4")
    print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Total frames: {frame_num}")
    print(f"Avg speed: {frame_num/total_time:.2f} frames/sec")
    
  