# tracker.py
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import deque


class BrightObjectTracker:
    def __init__(self,
                 max_distance=80,
                 min_area=80,
                 max_missed_frames=12,
                 brightness_threshold=90):

        self.max_distance = max_distance
        self.min_area = min_area
        self.max_missed_frames = max_missed_frames
        self.brightness_threshold = brightness_threshold

        self.next_track_id = 0
        self.tracks = {}  # id → track dict

    def _detect_bright_objects(self, frame):
        _, thresh = cv2.threshold(frame, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append((cx, cy, area))
        return detections

    def _cost_matrix(self, track_centroids, detections):
        if len(track_centroids) == 0 or len(detections) == 0:
            return np.empty((0, 0))
        cost = np.zeros((len(track_centroids), len(detections)))
        for i, t in enumerate(track_centroids):
            for j, d in enumerate(detections):
                cost[i, j] = np.linalg.norm(np.array(t) - np.array(d[:2]))
        return cost

    def update(self, frame):
        detections = self._detect_bright_objects(frame)

        active_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in active_ids]

        cost = self._cost_matrix(track_centroids, detections)

        matched_tracks = set()
        matched_dets = set()

        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.max_distance:
                    tid = active_ids[r]
                    cx, cy, _ = detections[c]
                    self.tracks[tid]['centroid'] = (cx, cy)
                    self.tracks[tid]['missed'] = 0
                    self.tracks[tid]['age'] += 1
                    self.tracks[tid]['history'].append((cx, cy))
                    if len(self.tracks[tid]['history']) > 50:
                        self.tracks[tid]['history'].popleft()
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Unmatched existing tracks
        for tid in active_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['missed'] += 1

        # New detections → new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                cx, cy, _ = det
                self.tracks[self.next_track_id] = {
                    'centroid': (cx, cy),
                    'age': 1,
                    'missed': 0,
                    'history': deque([(cx, cy)], maxlen=50)
                }
                self.next_track_id += 1

        # Remove lost tracks
        dead = [tid for tid in self.tracks if self.tracks[tid]['missed'] > self.max_missed_frames]
        for tid in dead:
            del self.tracks[tid]

        # Return only currently visible tracks
        visible = []
        for tid, tr in self.tracks.items():
            if tr['missed'] == 0:
                visible.append({
                    'id': tid,
                    'x': tr['centroid'][0],
                    'y': tr['centroid'][1],
                    'age': tr['age'],
                    'history': list(tr['history'])
                })
        return visible