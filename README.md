# GridLock Traffic Intelligence

GridLock is a NiceGUI-based traffic violation intelligence dashboard for uploaded videos, images, and live camera-style analysis. It is designed as a single Python app with a modern dark UI, smooth preview playback, live rule tuning, automatic evidence creation, plate OCR, analytics, and a Leaflet hotspot map.

## What It Does

- Accepts traffic videos or images for analysis.
- Shows a preview pane and a live processing pane side by side.
- Keeps the processing view updated with the latest annotated frame.
- Lets you change key traffic rules while a job is running.
- Detects and records violations such as:
  - Helmet non-compliance
  - Triple riding
  - Illegal parking
  - Wrong-side driving
  - Red-light and stop-line violations
  - Vehicle and rider detection
- Extracts number plates with OCR when the optional CV stack is available.
- Stores evidence frames, metadata, and violation history in SQLite.
- Tracks repeat offenders and recent violations.
- Shows live charts, tables, trends, and camera-wise analytics.
- Renders a hotspot map with Leaflet and dark Carto tiles.

## Project Style

This project is intentionally kept as a one-tier application:

- UI, upload handling, processing control, and analytics all live in one codebase.
- The dashboard stays easy to run locally.
- Large files are handled through the same app flow, without a separate frontend-backend split.

## How It Works

Traffic media -> preprocessing -> AI detection -> rule classification -> plate OCR -> evidence generation -> database storage -> analytics dashboard

## Project Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Application entry point. Starts NiceGUI and auto-selects a free port if `8501` is already in use. |
| `config/settings.py` | Global constants, data folders, default cameras, and thresholds. |
| `dashboard/page.py` | Main dashboard layout, upload workflow, live preview, controls, charts, tables, and analysis panels. |
| `dashboard/helpers.py` | Upload helpers, date/time parsing, state seeding, and option building for analysis jobs. |
| `dashboard/widgets.py` | Chart builders, data table helper, and hotspot map rendering with Leaflet. |
| `dashboard/theme.py` | App title, fonts, CSS, and dark theme styling. |
| `services/video_processor.py` | Frame processing, preprocessing, annotation, violation detection, and live preview generation. |
| `services/job_manager.py` | Background job lifecycle, live job updates, cancellation, and progress tracking. |
| `services/detector.py` | Detection model wrapper. |
| `services/ocr_service.py` | Number plate OCR helper. |
| `services/violation_engine.py` | Rule-based violation classification logic. |
| `services/preprocessing.py` | Image and frame enhancement helpers. |
| `services/db_service.py` | SQLite schema and query layer for cameras, evidence, violations, and reports. |
| `services/settings_store.py` | Saves and loads dashboard settings. |
| `services/runtime.py` | Shared runtime access for the job manager and settings store. |
| `services/evaluation.py` | Benchmark loading and summary helpers. |
| `data/` | Runtime storage for the database, evidence, uploads, parking coordinates, and saved settings. |
| `requirements.txt` | Base runtime dependencies. |
| `requirements-cv.txt` | Optional computer vision stack for full detection and OCR. |
| `Dockerfile` | Container build for deployment. |
| `docker-compose.yml` | Local container orchestration helper. |
| `verify_environment.py` | Sanity check for local setup. |
| `sitecustomize.py` | Local Python startup tweaks used by the project environment. |
| `yolov8n.pt` | Default YOLO model weight file used by the detector. |

## Core Features

### Upload and Preview

- Upload an image or video from the first screen.
- See the uploaded media in a clean preview panel.
- Watch a live annotated processing panel next to it.
- Keep playback and analysis visually aligned for smoother review.

### Live Rule Controls

- Camera ID
- Location name
- Latitude and longitude
- Date and start time
- Signal state
- Traffic-way and allowed-direction settings
- Left and right lane direction
- Parking checks
- Wrong-side checks
- Signal checks
- Preprocessing profile
- Frame skip
- Detection confidence
- Parking dwell time
- Wrong-side motion threshold
- Triple-riding overlap
- Helmet skin ratio
- Stop-line Y position
- Frame width and height

These settings are editable from the dashboard so the analysis can be tuned without restarting the app.

### Analytics and Reporting

- Total violations today
- Violations by category
- Camera-wise violation totals
- Recent violations table
- Trend charts
- Repeat offender list
- Evidence gallery
- Hotspot map

## Leaflet Map

The hotspot map uses Leaflet with dark Carto tiles.

- Camera hotspots are shown with colored markers.
- The map works without any external map API key.
- The map keeps the dashboard portable for local runs and deployments.

## Installation

### Base app

```bash
pip install -r requirements.txt
```

### Full CV stack

```bash
pip install -r requirements-cv.txt
```

The optional CV stack enables OpenCV, YOLO, and OCR features for full detection.

## Run

```bash
py app.py
```

Open the URL printed in the terminal.

If port `8501` is already in use, the app now checks the next free port instead of crashing.

For deployment platforms such as Render, the app also respects the `PORT` environment variable.

## Data And Output

The app stores runtime data in `data/`:

- `data/traffic_dashboard.db` for SQLite records
- `data/evidence/` for annotated evidence frames
- `data/uploads/` for uploaded files
- `data/parking_coords/` for generated or saved parking zone data
- `data/dashboard_settings.json` for saved dashboard settings

This keeps the project output easy to inspect and easy to version.

## Troubleshooting

- If the page does not load, confirm that the optional CV stack is installed.
- If `8501` is busy, the app will automatically use the next available port.
- If you want a fixed port, set `PORT` before launching the app.

## Deployment

The project can be deployed with Docker or on platforms like Render.

Typical deployment settings:

- Start command: `python app.py`
- Environment variables:
  - `PORT` if your host requires a specific port

## Notes

- The dashboard is tuned for smooth uploads, live processing, and modern dark visuals.
- Evidence and analytics are stored locally so the current output can be kept in GitHub if you choose to track it.
- Raw uploads can still be kept out of Git history if they are too large.
- When the selected scope has no real violations yet, the dashboard shows synthetic demo data so the UI stays populated.
