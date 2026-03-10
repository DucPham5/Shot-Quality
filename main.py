from ultralytics import YOLO
import supervision as sv
import time
import cv2
import numpy as np

from utils.ballTracker import BallTracker


if __name__ == "__main__":

    start_time = time.time()

    # ── Load models ───────────────────────────────────────────
    player_model = YOLO("models/player_detector.pt")
    ball_tracker = BallTracker(
        model_path="models/balldetectorv3.pt",
        conf=0.6,
        batch_size=20
    )

    # ── Open video ────────────────────────────────────────────
    cap = cv2.VideoCapture("input_videos/video_2.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps")

    # ── Read all frames ───────────────────────────────────────
    print("Reading all frames...")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Total frames: {len(frames)}")

    # ── Offline ball detection + interpolation ────────────────
    ball_positions = ball_tracker.get_ball_positions(frames)

    # ── Setup output video ────────────────────────────────────
    out = cv2.VideoWriter(
        'output_combined.mp4',
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (width, height)
    )

    # ── Render frames with player tracking + ball positions ───
    print("Rendering output video...")
    for frame_num, frame in enumerate(frames):

        # Player detection + tracking (still frame by frame)
        player_results = player_model.track(
            frame, conf=0.3, tracker="botsort.yaml",
            persist=True, verbose=False, agnostic_nms=True
        )

        # ── Draw players ──────────────────────────────────────
        player_name_to_id = {v: k for k, v in player_model.names.items()}
        PLAYER_HOOP_CLASS_ID = player_name_to_id.get('Hoop')

        if player_results[0].boxes.id is not None:
            for box, track_id, cls in zip(
                player_results[0].boxes.xyxy,
                player_results[0].boxes.id,
                player_results[0].boxes.cls
            ):
                if PLAYER_HOOP_CLASS_ID is not None and int(cls) == PLAYER_HOOP_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                tid_p = int(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"#{tid_p}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Draw ball ─────────────────────────────────────────
        ball_pos = ball_positions[frame_num]
        if ball_pos:
            center = ball_pos['center']
            cv2.circle(frame, center, 12, (0, 165, 255), -1)
            cv2.circle(frame, center, 15, (255, 255, 255), 2)
            cv2.putText(frame, "ball", (center[0] - 20, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        out.write(frame)

        if frame_num % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_num}/{len(frames)} | {frame_num / elapsed:.1f} fps")

    out.release()
    total_time = time.time() - start_time

    print(f"\n✓ Done! Check output_combined.mp4")
    print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Total frames: {len(frames)}")
    print(f"Avg speed: {len(frames) / total_time:.2f} frames/sec")
