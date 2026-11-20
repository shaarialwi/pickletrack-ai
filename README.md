# PickleTrack AI 🏓

Video-based pickleball analytics MVP. Upload footage, calibrate court corners, automatically track people on the court, view player movement heatmaps, manually review rally outcomes by ball-landing zone and calculate missed-return rates.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads the `yolo11n.pt` YOLO weights automatically (internet required). CPU runs slowly; a GPU is preferable for longer videos.

## Deploy

1. Upload `app.py`, `analysis.py`, `requirements.txt`, `.streamlit/config.toml` to a GitHub repository.
2. Connect the repo to Streamlit Community Cloud (main file: `app.py`) or another Python-capable hosting provider.
3. For larger videos, select a host with enough CPU/RAM, storage and request duration. Do **not** deploy to GitHub Pages alone: that is static hosting and cannot execute Python/YOLO.
4. Protect footage and access if uploading non-public match recordings. Uploaded file bytes may be written to the host's temp disk for processing.

## Recommended camera

- Stationary high/wide view of the whole pickleball court, with all four outer corners visible.
- Avoid frequent zoom, pans and cuts; these invalidate perspective calibration.
- Good light, minimal occlusion; preferably one court fills most of frame.
- Near/far corners are defined from the camera view, not absolute compass direction.

## Interpretation and limitations

- Automatic feature: person detection/tracking and court-projected movement footprint/estimated movement distance.
- Review feature: user records rally timestamp, player, ball end zone, and outcome.
- Missing return % is `(missed returns / playable opportunities) * 100`, per player and court zone. Out/non-playable balls excluded.
- This does **not** yet provide reliable automatic ball landing detection, racket-hit detection, or automatic miss classification. A normal generic person detector cannot infer those events.
- Tracker IDs are not guaranteed permanent: occlusion, scene cuts and close players may swap IDs. Validate before attributing to named players.
- Distance is approximate, sensitive to perspective calibration and frame sampling. Heatmaps indicate detections, not time weighted movement.
- Annotation sessions are ephemeral (in-memory session state). Download CSV before leaving the application.
- Before production use: authenticated accounts, persistent database, video retention policy, formal QA, a trained small-object pickleball detector, shot/landing recognition, and reviewer override/audit trail.

## Licence note

Review Ultralytics licensing (AGPL-3.0 / enterprise) before proprietary or commercial deployment. Dependencies have their own licences.
