import cv2
from ultralytics import YOLO
import numpy as np

import matplotlib.pyplot as plt
def get_accurate_frame_points(video_path):
    """Interactive tool to click on frame and get accurate pixel coordinates"""
    
    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error loading video")
        return
    
    # Store clicked points
    points = []
    labels = ["Top-Left", "Bottom-Left", "Bottom-Right", "Top-Right"]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"{labels[len(points)-1]}: [{x}, {y}]")
            
            # Draw the point
            cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame_copy, f"P{len(points)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If we have multiple points, draw lines
            if len(points) > 1:
                cv2.line(frame_copy, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2)
            
            # Close the trapezoid when we have 4 points
            if len(points) == 4:
                cv2.line(frame_copy, tuple(points[-1]), tuple(points[0]), (255, 0, 0), 2)
                print("\n✓ All 4 points selected!")
                print("\nCopy this into your code:")
                print("frame_points = np.array([")
                for i, pt in enumerate(points):
                    print(f"    {pt},  # {labels[i]}")
                print("], dtype=np.float32)")
            
            cv2.imshow('Select Points', frame_copy)
    
    # Create window
    frame_copy = frame.copy()
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', mouse_callback)
    
    print("="*60)
    print("Click on the 4 corners of the court trapezoid in this order:")
    print("1. Top-Left (baseline at left sideline)")
    print("2. Bottom-Left (baseline at right sideline)")  
    print("3. Bottom-Right (free-throw/paint at right edge)")
    print("4. Top-Right (free-throw/paint at left edge)")
    print("\nPress 'q' to quit, 'r' to reset")
    print("="*60)
    
    while True:
        cv2.imshow('Select Points', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            frame_copy = frame.copy()
            print("Reset! Click again.")
    
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32) if len(points) == 4 else None


if __name__ == "__main__":
    accurate_points = get_accurate_frame_points("jabarismith.mp4")