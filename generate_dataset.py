# generate_big_dataset.py  (4096x4096, 10 targets, 7 robust shapes, NO ERRORS)
import numpy as np
import imageio
import cv2

print("Generating 4096×4096 dataset with 10 varied bright targets...")

H, W = 4096, 4096
n_frames = 300
frames = np.zeros((n_frames, H, W), dtype=np.uint8)

# 10 targets
np.random.seed(42)
targets = []
for i in range(10):
    targets.append({
        'x': np.random.uniform(300, W-300),
        'y': np.random.uniform(300, H-300),
        'vx': np.random.uniform(-5, 5),
        'vy': np.random.uniform(-5, 5),
        'size_base': np.random.randint(4, 9),
        'bright_base': np.random.randint(200, 255),
        'shape_id': i % 7          # 7 different reliable shapes
    })

def draw_shape(temp, cx, cy, size, bright, shape_id):
    color = int(bright)

    if shape_id == 0:           # Circle
        cv2.circle(temp, (cx, cy), size, color, -1)

    elif shape_id == 1:         # Vertical ellipse
        cv2.ellipse(temp, (cx, cy), (size, size*2), 0, 0, 360, color, -1)

    elif shape_id == 2:         # Horizontal ellipse
        cv2.ellipse(temp, (cx, cy), (size*2, size), 0, 0, 360, color, -1)

    elif shape_id == 3:         # Square
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size
        cv2.rectangle(temp, (x1, y1), (x2, y2), color, -1)

    elif shape_id == 4:         # Triangle (pointing up)
        pts = np.array([[cx, cy-size*2], [cx-size*2, cy+size], [cx+size*2, cy+size]], np.int32)
        cv2.fillPoly(temp, [pts], color)

    elif shape_id == 5:         # Diamond (rotated square)
        pts = np.array([[cx, cy-size*2], [cx-size*2, cy], [cx, cy+size*2], [cx+size*2, cy]], np.int32)
        cv2.fillPoly(temp, [pts], color)

    elif shape_id == 6:         # Cross / plus sign
        cv2.rectangle(temp, (cx-size*3, cy-size), (cx+size*3, cy+size), color, -1)
        cv2.rectangle(temp, (cx-size, cy-size*3), (cx+size, cy+size*3), color, -1)

for t in range(n_frames):
    if t % 50 == 0:
        print(f"  → Frame {t}/{n_frames}")

    frame = np.random.randint(0, 30, (H, W), dtype=np.uint8)

    for obj in targets:
        # Update position + bounce
        obj['x'] += obj['vx']
        obj['y'] += obj['vy']
        if obj['x'] < 200 or obj['x'] > W-200: obj['vx'] *= -1
        if obj['y'] < 200 or obj['y'] > H-200: obj['vy'] *= -1
        obj['x'] = np.clip(obj['x'], 200, W-200)
        obj['y'] = np.clip(obj['y'], 200, H-200)

        cx, cy = int(obj['x']), int(obj['y'])

        # Dynamic size (3–10 px)
        size = int(obj['size_base'] + 3 * np.sin(t * 0.12 + obj['shape_id']))
        size = np.clip(size, 2, 10)

        # Flickering brightness
        bright = int(obj['bright_base'] + 25 * np.sin(t * 0.18 + obj['shape_id']))
        bright = np.clip(bright, 180, 255)

        # Draw on temp layer
        temp = np.zeros_like(frame)
        draw_shape(temp, cx, cy, size, bright, obj['shape_id'])

        # Soft fuzzy edges
        temp = cv2.GaussianBlur(temp, (5, 5), 1.3)

        # Composite
        mask = temp > 50
        frame[mask] = np.maximum(frame[mask], temp[mask])

    # Final noise
    noise = np.random.randint(-25, 26, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    frames[t] = frame

# Save full-res data
np.savez_compressed("bright_objects_4096x4096_10targets.npz", frames=frames)
print("\nSaved bright_objects_4096x4096_10targets.npz (~340 MB)")

# Downscaled preview
print("Creating preview video (1024×1024)...")
preview = [cv2.resize(f, (1024, 1024), interpolation=cv2.INTER_AREA) for f in frames[::3]]
imageio.mimsave("preview_10_targets.mp4", preview, fps=30, quality=9)
print("Saved preview_10_targets.mp4")
print("\nAll done! Now run: python demo_big.py")