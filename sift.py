from pathlib import Path

import cv2 as cv


def main():
    sift = cv.SIFT_create()
    directory = Path(__file__).resolve().parent
    for name in ("image", "image_small"):
        source = directory / f"{name}.jpg"
        destination = directory / f"{name}_sift.jpg"
        img = cv.imread(str(source))
        if img is None:
            raise SystemExit(f"Could not read {source}")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keypoints = sift.detect(gray, None)
        out = cv.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        if not cv.imwrite(str(destination), out):
            raise SystemExit(f"Could not write {destination}")
        print(f"Saved {destination.name} ({len(keypoints)} keypoints)")


if __name__ == "__main__":
    main()
