from ultralytics import YOLO 
if __name__ == "__main__":
    INPUT_VIDEO = "jabarismith.mp4"
    OUTPUT_VIDEO = "output_detection_pose.mp4"
    model = YOLO("models/player_detector.pt")
    results = model.track(INPUT_VIDEO, save = True, show_labels=True, show_conf = False)
    for box in results[0].boxes:
        print(box)
  