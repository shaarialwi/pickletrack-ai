"""Video tracking and perspective transformation for PickleTrack AI."""
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

COURT_W, COURT_L = 6.10, 13.41  # metres, doubles court

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Unable to open video. Convert to MP4 (H.264) and retry.')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {'fps': fps, 'frames': count, 'width': w, 'height': h, 'seconds': count/fps if fps else 0}

def preview_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Could not read first video frame.')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def homography(corners):
    """Corners must be ordered near-left, near-right, far-right, far-left."""
    src = np.array(corners, dtype=np.float32)
    if src.shape != (4, 2) or abs(cv2.contourArea(src)) < 100:
        raise ValueError('Court corners do not form a valid quadrilateral.')
    dest = np.array([[0, COURT_L], [COURT_W, COURT_L], [COURT_W, 0], [0, 0]], np.float32)
    return cv2.getPerspectiveTransform(src, dest)

def transform_point(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x,y]]], np.float32), H)[0,0]
    return float(p[0]), float(p[1])

def track_people(path, corners, every_n=3, conf=0.3, max_seconds=180, progress=None):
    """Tracks players by YOLO track IDs; identifies all people, not just active players.
    IDs can change during occlusion and must be checked before interpretation.
    """
    H = homography(corners)
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open video.')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(max_seconds*fps))
    rows=[]; idx=0; processed=0
    try:
        while idx < total:
            ok, frame = cap.read()
            if not ok: break
            if idx % every_n == 0:
                results = model.track(frame, persist=True, verbose=False, classes=[0], conf=conf,
                                      tracker='bytetrack.yaml')
                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    coords = boxes.xyxy.cpu().numpy()
                    for tid, (x1,y1,x2,y2) in zip(ids, coords):
                        x = (x1+x2)/2; y = y2
                        cx, cy = transform_point(H, x, y)
                        # People outside calibrated doubles court excluded
                        if 0 <= cx <= COURT_W and 0 <= cy <= COURT_L:
                            rows.append({'time_s':round(idx/fps,2), 'frame':idx, 'track_id':int(tid),
                                         'court_x_m':round(cx,3), 'court_y_m':round(cy,3),
                                         'pixel_x':round(float(x),1),'pixel_y':round(float(y),1)})
                processed += 1
                if progress and processed % 10 == 0:
                    progress(min(1.0,idx/max(total,1)))
            idx += 1
    finally:
        cap.release()
    if progress: progress(1.0)
    return pd.DataFrame(rows, columns=['time_s','frame','track_id','court_x_m','court_y_m','pixel_x','pixel_y'])

def total_distance(df):
    if len(df)<2: return 0.
    df=df.sort_values('time_s').copy()
    delta=df[['court_x_m','court_y_m']].diff()
    gaps=df['time_s'].diff()
    step=np.sqrt((delta**2).sum(axis=1))
    # ignore jumps > 3m between samples; ID switches/camera artifacts
    return float(step.where((gaps <= 1.5)&(step <= 3.0), 0).sum())
