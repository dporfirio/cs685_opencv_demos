"""Webcam semantic segmentation with OpenCV ENet. Q / Escape exits."""

import argparse
from pathlib import Path
import sys
import time

import cv2 as cv
import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def segment(net, frame):
    """Return one predicted category ID per pixel in the original frame."""
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (512, 256),
                               swapRB=True, crop=False)
    net.setInput(blob)
    scores = net.forward()
    ids = np.argmax(scores[0], axis=0).astype(np.uint8)
    return cv.resize(ids, (frame.shape[1], frame.shape[0]),
                     interpolation=cv.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default: 0)')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay opacity, 0 to 1')
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')

    model = DIRECTORY / 'Enet-model-best.net'
    labels = DIRECTORY / 'enet-classes.txt'
    if not model.is_file() or not labels.is_file():
        raise RuntimeError('Keep Enet-model-best.net and enet-classes.txt beside this script.')
    names = labels.read_text().splitlines()
    net = cv.dnn.readNetFromTorch(str(model))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    colors = np.random.default_rng(42).integers(40, 230, (len(names), 3), dtype=np.uint8)
    colors[0] = 0

    backend = cv.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv.CAP_ANY
    cap = cv.VideoCapture(args.camera, backend)
    window = 'Semantic segmentation | Q / Esc to quit'
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {args.camera}. Check camera permissions '
                               'or try --camera 1.')
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Could not read a webcam frame.')
            frame = cv.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
            start = time.perf_counter()
            ids = segment(net, frame)
            elapsed = time.perf_counter() - start
            overlay = cv.addWeighted(frame, 1 - args.alpha, colors[ids], args.alpha, 0)
            # Leave unlabeled pixels uncolored.
            overlay[ids == 0] = frame[ids == 0]
            raw = frame.copy()
            cv.putText(raw, f'Raw camera {args.camera}', (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
            if float(frame.mean()) < 15:
                cv.putText(raw, 'Dark input: check lens, lighting, camera index', (10, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
            cv.putText(overlay, f'ENet | inference {elapsed * 1000:.0f} ms', (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
            legend = np.zeros((frame.shape[0], 200, 3), dtype=np.uint8)
            row_height = min(22, (frame.shape[0] - 12) // len(names))
            for i, name in enumerate(names):
                y = 6 + i * row_height
                cv.rectangle(legend, (8, y), (25, y + 15), tuple(map(int, colors[i])), -1)
                cv.putText(legend, name, (33, y + 13), cv.FONT_HERSHEY_SIMPLEX,
                           0.4, (230, 230, 230), 1, cv.LINE_AA)
            cv.imshow(window, np.hstack((raw, overlay, legend)))
            if cv.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, cv.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
