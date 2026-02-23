from collections import deque
import numpy as np

class BallTracker:
    def __init__(self, max_lost=30, max_distance=200):
        self.track_id = 1
        self.center = None
        self.lost_frames = 0
        self.max_lost = max_lost          # frames to keep track alive with no detection
        self.max_distance = max_distance  # max pixels ball can move between frames
        self.history = deque(maxlen=8)    # for velocity prediction
        self.active = False

    def update(self, detections):
        """
        detections: sv.Detections filtered to ball class only, after NMS
        returns: (center, track_id) or (predicted_center, track_id) or (None, None)
        """
        if len(detections) == 0:
            if self.active:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost:
                    self.active = False
                    self.center = None
                    return None, None
                # predict position from velocity
                return self._predict(), self.track_id
            return None, None

        # Pick best detection (highest confidence)
        best_idx = np.argmax(detections.confidence)
        box = detections.xyxy[best_idx]
        new_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        if not self.active:
            # Start new track
            self.active = True
            self.center = new_center
            self.lost_frames = 0
            self.history.clear()
            self.history.append(new_center)
            return tuple(new_center.astype(int)), self.track_id

        # Check if detection is close enough to be the same ball
        dist = np.linalg.norm(new_center - self._predict())
        if dist < self.max_distance:
            # Same ball — update track
            self.center = new_center
            self.lost_frames = 0
            self.history.append(new_center)
            return tuple(new_center.astype(int)), self.track_id
        else:
            # Too far — treat as new ball, new ID
            self.track_id += 1
            self.center = new_center
            self.lost_frames = 0
            self.history.clear()
            self.history.append(new_center)
            return tuple(new_center.astype(int)), self.track_id

    def _predict(self):
        if self.center is None:
            return np.array([0, 0])
        if len(self.history) >= 2:
            pts = np.array(self.history)
            velocity = np.mean(np.diff(pts, axis=0), axis=0)
            predicted = self.center + velocity * (self.lost_frames + 1)
        else:
            predicted = self.center
        return tuple(predicted.astype(int))  # ← ensure ints