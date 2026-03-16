from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import torch
import torchvision




def apply_nms(detections, iou_threshold=0.5):
    """Collapse duplicate detections of the same ball into one box."""
    if len(detections) == 0:
        return detections
    boxes = torch.tensor(detections.xyxy, dtype=torch.float32)
    scores = torch.tensor(detections.confidence, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return detections[keep.numpy()]


class BallTracker:
    """
    Offline ball tracker — processes all frames at once, then interpolates gaps.
    Much more accurate than frame-by-frame prediction for pre-recorded video.
    """

    def __init__(self, model_path, conf=0.6, batch_size=20):
        self.model = YOLO(model_path)
        self.conf = conf
        self.batch_size = batch_size
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Get class IDs from model
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.BALL_CLASS_ID = name_to_id.get('ball')
        self.HOOP_CLASS_ID = name_to_id.get('hoop')

    def detect_frames(self, frames):
        """Run ball detection on all frames in batches."""
        all_detections = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            results = self.model.predict(batch, conf=self.conf, verbose=False, agnostic_nms=True, device=self.device)
            all_detections += results
            if i % 100 == 0:
                print(f"  Ball detection: {i}/{len(frames)} frames")
        return all_detections

    def get_ball_positions(self, frames):
        """
        Detect ball in all frames, filter false positives, interpolate gaps.
        Returns a list of dicts: [{'center': (x, y), 'bbox': [x1,y1,x2,y2]} or {} per frame]
        """
        print("Detecting ball in all frames...")
        raw_detections = self.detect_frames(frames)

        # Extract best detection per frame
        ball_positions = []
        for detection in raw_detections:
            sv_det = sv.Detections.from_ultralytics(detection)
            ball_only = sv_det[sv_det.class_id == self.BALL_CLASS_ID] if self.BALL_CLASS_ID is not None else sv_det
            ball_only = apply_nms(ball_only)

            if len(ball_only) == 0:
                ball_positions.append({})
                continue

            best_idx = np.argmax(ball_only.confidence)
            box = ball_only.xyxy[best_idx]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            ball_positions.append({'center': center, 'bbox': box.tolist()})

        print("Filtering false positives...")
        ball_positions = self.remove_wrong_detections(ball_positions)

        """TESTING DEBUGGING"""
        for i, pos in enumerate(ball_positions):
            if pos:
                print(f"Frame {i}: {pos['center']}")

        print("Interpolating gaps...")
        ball_positions = self.interpolate_positions(ball_positions)

        return ball_positions

    def get_hoop_positions(self, frames):
        """Extract hoop detections for all frames."""
        if self.HOOP_CLASS_ID is None:
            return [{}] * len(frames)

        raw_detections = self.detect_frames(frames)
        hoop_positions = []
        for detection in raw_detections:
            sv_det = sv.Detections.from_ultralytics(detection)
            hoop_only = sv_det[sv_det.class_id == self.HOOP_CLASS_ID]
            if len(hoop_only) == 0:
                hoop_positions.append({})
                continue
            hoop_positions.append({'boxes': hoop_only.xyxy.tolist()})
        return hoop_positions

    def remove_wrong_detections(self, ball_positions, max_distance_per_frame=15):
        """
        Filter detections where the ball jumps too far between frames.
        Rejects a detection only if it fails both the backward AND forward check.
        """
        valid_indices = [i for i, p in enumerate(ball_positions) if p.get('bbox') is not None]

        for idx, i in enumerate(valid_indices):
            current = ball_positions[i].get('bbox')
            if current is None:
                continue

            failed_prev = False
            failed_next = False

            # Check against nearest surviving previous detection
            if idx > 0:
                prev_bbox = None
                prev_i = None
                for look_back in range(idx - 1, -1, -1):
                    candidate = ball_positions[valid_indices[look_back]].get('bbox')
                    if candidate is not None:
                        prev_bbox = candidate
                        prev_i = valid_indices[look_back]
                        break

                if prev_bbox is not None:
                    frame_gap = i - prev_i
                    allowed = max_distance_per_frame * frame_gap
                    dist = np.linalg.norm(np.array(current[:2]) - np.array(prev_bbox[:2]))
                    if dist > allowed:
                        failed_prev = True

            # Check against nearest surviving next detection
            if idx < len(valid_indices) - 1:
                next_bbox = None
                next_i = None
                for look_ahead in range(idx + 1, len(valid_indices)):
                    candidate = ball_positions[valid_indices[look_ahead]].get('bbox')
                    if candidate is not None:
                        next_bbox = candidate
                        next_i = valid_indices[look_ahead]
                        break

                if next_bbox is not None:
                    frame_gap = next_i - i
                    allowed = max_distance_per_frame * frame_gap
                    dist = np.linalg.norm(np.array(current[:2]) - np.array(next_bbox[:2]))
                    if dist > allowed:
                        failed_next = True

            # Only reject if fails both directions
            if failed_prev and failed_next:
                ball_positions[i] = {}

        return ball_positions

    def interpolate_positions(self, ball_positions):
        """
        Fill gaps in ball positions by interpolating between known detections.
        This is the key advantage over frame-by-frame prediction — we know
        where the ball ends up, so we can draw a smooth path through occlusions.
        """
        bboxes = [p.get('bbox', []) for p in ball_positions]
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values linearly, backfill any leading NaNs
        df = df.interpolate()
        df = df.bfill()

        # Rebuild positions list
        result = []
        for row in df.to_numpy().tolist():
            x1, y1, x2, y2 = row
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            result.append({'center': center, 'bbox': row})

        return result