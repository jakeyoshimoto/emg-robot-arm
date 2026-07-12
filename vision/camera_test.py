"""
Sanity-checks the webcam. Opens it, shows the live feed, prints
resolution and measured FPS. Run this first if anything vision-related
misbehaves.

Usage:
    python vision/camera_test.py
    python vision/camera_test.py --index 1 --width 1280 --height 720

Press 'q' or close the window to exit.
"""

import argparse
import time

import cv2

# Defaults, override via CLI args.
CAMERA_INDEX = 0  # 0 is usually the first/only camera plugged in
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480
WINDOW_NAME = "camera_test"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=CAMERA_INDEX,
                         help="camera device index (default: %(default)s)")
    parser.add_argument("--width", type=int, default=REQUESTED_WIDTH,
                         help="requested capture width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=REQUESTED_HEIGHT,
                         help="requested capture height (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {args.index}. "
            "Check that it's connected and not in use by another app."
        )

    # Some cameras ignore the requested size, so check what we actually got.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested resolution: {args.width}x{args.height}")
    print(f"Actual resolution:    {actual_width}x{actual_height}")

    # FPS = frames counted over a short window, divided by elapsed time.
    frame_count = 0
    fps_report_interval = 2.0  # seconds between FPS prints
    window_start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            frame_count += 1
            elapsed = time.time() - window_start_time
            if elapsed >= fps_report_interval:
                fps = frame_count / elapsed
                print(f"Measured FPS: {fps:.1f}")
                frame_count = 0
                window_start_time = time.time()

            cv2.imshow(WINDOW_NAME, frame)

            # waitKey also redraws the window, so it has to run every loop.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # covers clicking the window's 'X' button
    finally:
        cap.release()  # always release, even if the loop broke via an exception
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
