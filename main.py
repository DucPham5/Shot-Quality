from ultralytics import YOLO 
if __name__ == "__main__":
    INPUT_VIDEO = "jabarismith.mp4"
    OUTPUT_VIDEO = "output_detection_pose.mp4"
    model = YOLO("models/player_detector.pt")
    results = model.predict(INPUT_VIDEO, save = True)
    for box in results[0].boxes:
        print(box)
  