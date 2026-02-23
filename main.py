from ultralytics import YOLO
import supervision as sv
import time
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path

from utils.ballTracker import BallTracker


def apply_nms(detections, iou_threshold=0.5):
    """Collapse duplicate detections of the same ball into one box."""
    if len(detections) == 0:
        return detections

    boxes = torch.tensor(detections.xyxy, dtype=torch.float32)
    scores = torch.tensor(detections.confidence, dtype=torch.float32)

    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return detections[keep.numpy()]


if __name__ == "__main__":

    start_time = time.time()

    # Load both models
    player_model = YOLO("models/player_detector.pt")
    ball_model = YOLO("models/balldetectionv2.pt")
    print(ball_model.names)
    print(player_model.names)

    # Ball model class IDs
    name_to_id = {v: k for k, v in ball_model.names.items()}
    BALL_CLASS_ID = name_to_id['ball']
    HOOP_CLASS_ID = name_to_id.get('hoop', None)

    # Filter hoop out of player model — ball model handles hoops
    player_name_to_id = {v: k for k, v in player_model.names.items()}
    PLAYER_HOOP_CLASS_ID = player_name_to_id.get('Hoop', None)

    # Open video first so we have fps for the tracker
    cap = cv2.VideoCapture("video_1.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps")

    # Ball tracker
    ball_tracker = BallTracker(
        max_lost=fps * 3,
        max_distance=250
    )

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

        # ── Player detection + tracking ───────────────────────
        player_results = player_model.track(
            frame, conf=0.3, tracker="botsort.yaml",
            persist=True, verbose=False, agnostic_nms=True
        )

        # ── Ball detection ────────────────────────────────────
        ball_results = ball_model.predict(frame, conf=0.05, verbose=False, agnostic_nms=True)
        ball_detections = sv.Detections.from_ultralytics(ball_results[0])
        ball_only = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
        ball_only = apply_nms(ball_only, iou_threshold=0.5)

        center, tid = ball_tracker.update(ball_only)

        # ── Draw players ──────────────────────────────────────
        if player_results[0].boxes.id is not None:
            for box, track_id, cls in zip(
                player_results[0].boxes.xyxy,
                player_results[0].boxes.id,
                player_results[0].boxes.cls
            ):
                # Skip hoops from player model
                if PLAYER_HOOP_CLASS_ID is not None and int(cls) == PLAYER_HOOP_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                tid_p = int(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{tid_p}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Draw ball ─────────────────────────────────────────
        if center is not None:
            is_predicted = ball_tracker.lost_frames > 0
            color = (80, 80, 180) if is_predicted else (0, 165, 255)
            label = f"ball #{tid} (predicted)" if is_predicted else f"ball #{tid}"
            cv2.circle(frame, center, 12, color, -1)
            cv2.circle(frame, center, 15, (255, 255, 255), 2)
            cv2.putText(frame, label, (center[0] - 20, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── Draw hoops ────────────────────────────────────────
        if HOOP_CLASS_ID is not None:
            hoop_only = ball_detections[ball_detections.class_id == HOOP_CLASS_ID]
            for box in hoop_only.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Hoop", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)

        if frame_num % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_num} | {frame_num / elapsed:.1f} fps")

    cap.release()
    out.release()
    total_time = time.time() - start_time

    print(f"\n✓ Done! Check output_combined.mp4")
    print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Total frames: {frame_num}")
    print(f"Avg speed: {frame_num / total_time:.2f} frames/sec")