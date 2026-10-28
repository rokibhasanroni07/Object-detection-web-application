# VisionAI — Object Detection Studio

A responsive browser-based object detection app built from the original COCO-SSD / TensorFlow.js project.

## AI library versions
- **TensorFlow.js:** 4.22.0 (latest stable release found in the official release list during this update)
- **COCO-SSD:** 2.2.3 (latest published version used by this browser build)
- **Model base:** `mobilenet_v2` — the higher-accuracy COCO-SSD option compared with `lite_mobilenet_v2`, at the cost of more processing time.

Versions are pinned in `index.html` so the app does not silently change when a CDN package is updated. The library versions and the model architecture are different things: COCO-SSD is the package, while `mobilenet_v2` is its model base.

## Features
- Responsive dark dashboard with light/dark theme toggle
- Image upload and drag-and-drop (JPG, PNG, WEBP; 15 MB limit)
- Real-time webcam detection with Start, Pause/Resume and Stop
- Colored bounding boxes, class labels and confidence percentages
- Adjustable confidence threshold
- Object count, unique classes, average confidence and inference time
- Approximate frame-to-frame object tracking with temporary IDs
- Export annotated canvas as PNG
- Local detection history, class frequency analytics and CSV export
- Input validation, model/camera error messages and capped canvas size for performance
- Local browser history; image processing happens in the browser

## Run it
Because camera access is generally restricted to secure contexts, run the folder through `localhost` or HTTPS rather than opening `index.html` directly.

### Option A: Python
From this folder:
```bash
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

### Option B: VS Code
Install the Live Server extension, right-click `index.html`, and select **Open with Live Server**.

## Requirements and notes
- Internet access is required on first load for TensorFlow.js, COCO-SSD and model assets served by their CDNs.
- Allow camera permission when prompted.
- COCO-SSD detects the categories it was trained on; it is not a custom-object model. The `mobilenet_v2` base is more accuracy-oriented and may run slower than `lite_mobilenet_v2`, especially on low-powered devices.
- Tracking uses approximate bounding-box proximity and class matching. It is not a dedicated multi-object tracking algorithm and IDs can change when objects overlap or move quickly.
- History is saved in localStorage on this browser/device. It does not upload images or results to a server.
- `Sample.zip` and the original notebook are retained from the uploaded archive.

## Files
- `index.html` — app layout
- `style.css` — responsive styling and theme
- `script.js` — model, detection, webcam, tracking, exports and history
