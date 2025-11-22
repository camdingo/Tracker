# Bright Object Tracker – 4096×4096 Multi-Target Demo

A complete, real-time bright object tracker in Python, designed for large-scale (4096×4096) synthetic sensor data with small, flickering, and varied-shape targets.

Perfect for testing tracking algorithms under realistic conditions: noise, occlusion-like overlaps, shape variation, and fast motion.

demo preview = preview_10_targets.mp4

## Features
- Tracks **10 tiny (3–10 pixel)** moving bright objects  
- Frame size: **4096 × 4096 pixels**  
- Objects have **7 different shapes** (circle, ellipse, square, triangle, diamond, cross, etc.)  
- Realistic effects: flicker, blur, noise, wall bouncing  
- Stable track IDs using the Hungarian algorithm  
- Clean ring overlays so you can see the real object shape underneath  

## Files in this project

- `tracker.py`                  → Core tracking class  
- `generate_big_dataset.py`     → Creates the 300-frame test dataset  
- `demo_big.py`                 → Live demo with big 1024×1024 window  
- `bright_objects_4096x4096_10targets.npz` → Full-res data (~340 MB)  
- `preview_10_targets.mp4`      → Quick video preview of raw data  

## How to run (one-time setup)

```bash
# Activate your environment
source bright_tracker_env/bin/activate

# Generate the dataset (only once)
python generate_big_dataset.py

# Run the live tracker demo
python demo_big.py
```

You’ll see a big 1024×1024 window with 10 brightly colored rings tracking tiny flickering objects of different shapes — all with rock-solid ID stability.

Press q to quit.

## Requirements

```pip install numpy opencv-python scipy imageio imageio-ffmpeg```

For GUI in WSL, use VcXsrv with "Disable access control" enabled.
Customization Ideas

Change max_distance, min_area, or brightness_threshold in demo_big.py to test robustness
Increase number of targets or frame size
Add Kalman filter prediction (easy extension)
Export tracks to CSV for analysis
