from ultralytics import YOLO

model = YOLO("models/balldetector030926.pt")

results = model.track("input_videos/video_2.mp4", save=True, conf = 0.05)
print(results)
print("+=========")
for boxes in results[0].boxes:
    print(boxes)