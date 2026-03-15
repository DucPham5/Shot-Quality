from ultralytics import YOLO

model = YOLO("models/balldetectorv3.pt")

results = model.track("input_videos/jabarismith.mp4", save=True, conf = 0.05)
print(results)
print("+=========")
for boxes in results[0].boxes:
    print(boxes)