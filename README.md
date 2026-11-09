# Harris corner detection with a webcam

## Webcam semantic segmentation

```sh
python3 segmentation_webcam.py
```

Shows the raw feed, a colored semantic overlay, and the category legend.
Press Q, Escape, or Ctrl+C to stop. Use `--camera 1` for another camera and
`--alpha 0.6` for a stronger overlay. Requires the same OpenCV/NumPy dependencies.
Keep `Enet-model-best.net` and `enet-classes.txt` beside the script.

ENet predicts 20 categories including unlabeled, road, building, sky, person,
and car. It is trained for outdoor road scenes, so indoor results may be poor.
Frames are processed locally and are not saved. The model was downloaded from
the URL in OpenCV's official model configuration, with SHA1 verified as
`b4123a73bf464b9ebe9cfc4ab9c2d5c72b161315`.

Model/preprocessing source:
https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml

## Harris demo

Run with Python 3 and a desktop display:

```sh
python3 -m pip install -r requirements.txt
python3 harris_webcam.py
```

The left panel shows the raw camera feed and its average brightness; the right
panel shows detected corners as red circles. A dark-input message means the
camera itself is supplying dark frames: check the lens/lighting and try another
camera index. On macOS the script uses the AVFoundation capture backend and
keeps the native camera mode. A minimum response rejects weak sensor noise.

Adjust the threshold slider to control sensitivity;
lower values detect more corners. Press **Q**, **Escape**, or close the window to exit.
The slider value is divided by 10000 to obtain the response threshold fraction.

To select another webcam or set the initial threshold:

```sh
python3 harris_webcam.py --camera 1 --threshold 0.02
```

On macOS, allow camera access for the app running Python in **System Settings →
Privacy & Security → Camera**. Close other apps using the camera if capture fails.

The detector converts frames to grayscale, lightly blurs them, computes
`cv2.cornerHarris`, thresholds positive responses relative to the frame maximum,
and suppresses nearby nonmaximum responses before drawing markers. Frames are
processed locally and are not saved.

API reference: https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html
