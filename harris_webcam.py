"""Display Harris corners from a webcam. Press Q or Escape to quit."""

import argparse
import sys

import cv2
import numpy as np


def detect_corners(frame, threshold=0.01):
    """Return (x, y) corner locations after local maximum suppression."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    response = cv2.cornerHarris(np.float32(gray) / 255.0,
                               blockSize=3, ksize=3, k=0.04)
    peak = float(response.max())
    if peak <= 0:
        return np.empty((0, 2), dtype=np.int32)
    maxima = cv2.dilate(response, np.ones((7, 7), dtype=np.uint8))
    # A relative threshold alone amplifies sensor noise in dark/flat frames.
    mask = ((response == maxima) &
            (response > max(threshold * peak, 1e-5))).astype(np.uint8)
    # Collapse plateaus to one marker per connected region.
    _, _, _, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return np.rint(centers[1:]).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Fraction of maximum Harris response, between 0 and 1')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be between 0 and 1 (exclusive)')

    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    window = 'Harris corners | Q / Esc to quit'
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {args.camera}. Check camera permissions '
                               'and try --camera 1 if needed.')
        # Keep the camera's native capture mode; resize only for display.
        cv2.namedWindow(window)
        cv2.createTrackbar('Threshold x10000', window,
                           max(1, round(args.threshold * 10000)), 10000, lambda _: None)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Could not read a frame from the webcam.')
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
            raw = frame.copy()
            brightness = float(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).mean())
            threshold = max(1, cv2.getTrackbarPos('Threshold x10000', window)) / 10000
            corners = detect_corners(frame, threshold)
            for x, y in corners:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Corners: {len(corners)}  Threshold: {threshold:.4f}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(raw, f'Raw camera {args.camera} | brightness {brightness:.1f}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            if brightness < 15:
                cv2.putText(raw, 'Dark input: check lens, lighting, or camera index',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imshow(window, np.hstack((raw, frame)))
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
