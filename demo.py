# demo.py
import numpy as np
import cv2
from tracker import BrightObjectTracker

# Load the generated data
data = np.load("bright_objects_sequence.npz")
frames = data["frames"]  # (150, 512, 512)

tracker = BrightObjectTracker(
    max_distance=80,
    min_area=80,
    max_missed_frames=12,
    brightness_threshold=90
)

print("Starting live tracking demo (press Q to quit)...")

for i, frame in enumerate(frames):
    tracks = tracker.update(frame)

    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for tr in tracks:
        x, y = int(tr['x']), int(tr['y'])
        tid = tr['id']

        # Draw trail
        for j in range(1, len(tr['history'])):
            pt1 = (int(tr['history'][j-1][0]), int(tr['history'][j-1][1]))
            pt2 = (int(tr['history'][j][0]), int(tr['history'][j][1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        # Draw current position + ID
        cv2.circle(vis, (x, y), 12, (0, 255, 255), -1)
        cv2.putText(vis, f"ID {tid}", (x + 15, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(vis, f"Frame {i+1}/{len(frames)}  |  Tracks: {len(tracks)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Bright Object Tracker - Live Demo", vis)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Demo finished.")