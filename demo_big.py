# demo_big.py
import numpy as np
import cv2
from tracker import BrightObjectTracker

# Load the dataset
data = np.load("bright_objects_4096x4096_10targets.npz")
frames = data["frames"]

# Tracker (same settings that worked before)
tracker = BrightObjectTracker(
    max_distance=150,
    min_area=8,
    max_missed_frames=15,
    brightness_threshold=90
)

print("Starting tracker")
print("Press 'q' to quit\n")

# Create the window ONCE and set its size
cv2.namedWindow("4096x4096 Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("4096x4096 Tracker", 1024, 1024)

scale = 1024 / 4096  # 4× downscale factor

for i, frame in enumerate(frames):
    tracks = tracker.update(frame)

    # Downscale full 4096×4096 frame to 1024×1024
    vis = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_AREA)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw each track
    for tr in tracks:
        x = int(tr['x'] * scale)
        y = int(tr['y'] * scale)

        # Short green trail
        for j in range(1, len(tr['history'])):
            x1 = int(tr['history'][j-1][0] * scale)
            y1 = int(tr['history'][j-1][1] * scale)
            x2 = int(tr['history'][j][0] * scale)
            y2 = int(tr['history'][j][1] * scale)
            cv2.line(vis, (x1, y1), (x2, y2), (80, 255, 80), 3)

        # Clean ring (so you can see the real shape underneath)
        ring_radius = 18 + tr['age'] // 10
        cv2.circle(vis, (x, y), ring_radius,     (0, 255, 255), 4)   # yellow outer ring
        cv2.circle(vis, (x, y), ring_radius - 6, (0, 0, 0),     4)   # black inner → transparent
        #cv2.circle(vis, (x, y), 3,               (0, 100, 255), -1)  # tiny red center dot

        # ID label
        cv2.putText(vis, f"ID {tr['id']}", (x + 22, y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

    # Frame info
    cv2.putText(vis, f"Frame {i+1}/{len(frames)}  |  Tracks: {len(tracks)}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    # Show in the ONE window
    cv2.imshow("4096x4096 Tracker", vis)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Clean exit
cv2.destroyAllWindows()
print("Demo finished!")