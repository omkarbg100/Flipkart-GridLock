# GridLock Traffic Violation Dashboard

GridLock is a single-tier Python dashboard for traffic-violation monitoring, video upload analysis, evidence review, analytics, and repeat-offender tracking.

The UI is built with NiceGUI, uses one Python app, and keeps the workflow upload-first with no sidebar.

## What It Covers

- Upload-first video analysis
- Inline video preview and playback
- Traffic rule controls directly below the player
- Violation analytics and trend charts
- Evidence browser with stored frames
- Repeat-offender alerts
- Camera hotspot summaries
- Saved dashboard settings
- Background jobs for uploads and webcam runs
- Optional computer-vision processing for detection, tracking, and OCR

## Project Layout

- `app.py` is the active NiceGUI dashboard entrypoint
- `services/` contains the database, settings, job manager, and video pipeline
- `data/` stores the SQLite database, evidence images, uploads, and saved dashboard settings
- The repo is intentionally single-tier, so there is no separate frontend service to run

## Install

Install the dashboard stack:

```bash
pip install -r requirements.txt
```

If you want the full video-analysis stack, install the optional CV extras too:

```bash
pip install -r requirements-cv.txt
```

## Run

Start the dashboard:

```bash
python app.py
```

On Windows, you can also use:

```bash
py app.py
```

Open the app at:

```text
http://127.0.0.1:8501
```

## Optional CV Stack

The dashboard can launch without the heavy detection packages, but upload and webcam analysis stay disabled until the CV extras are installed.

The optional stack includes:

- `opencv-python-headless`
- `numpy`
- `ultralytics`
- `easyocr`
- `torch`

## Docker

Build and run the one-tier dashboard:

```bash
docker compose up --build
```

## Validation

Run the environment check script to confirm the dashboard stack and optional analysis stack:

```bash
python verify_environment.py
```

## Notes

- Evidence images are written to `data/evidence/`
- Uploaded videos are cached in `data/uploads/`
- SQLite data lives in `data/traffic_dashboard.db`
- Dashboard settings are saved in `data/dashboard_settings.json`
- The app stays responsive during large-file processing by using background jobs
