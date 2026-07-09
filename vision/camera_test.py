"""
Quick sanity check to confirm the camera works before building anything
more complicated on top of it. Opens the webcam, shows what it sees in a
window, prints resolution and frame rate. If this doesn't run cleanly,
nothing else in this folder will either.

Usage:
    python vision/camera_test.py
    python vision/camera_test.py --index 1 --width 1280 --height 720

Press 'q' or close the window to exit.
"""

import argparse
import time

import cv2

# Starting values if nothing is passed on the command line.
CAMERA_INDEX = 0        # which camera to use. 0 is usually the first/only one plugged in
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480
WINDOW_NAME = "camera_test"


def parse_args():
    # Reads command-line args like --index 1 into args.index, args.width,
    # args.height. Falls back to the defaults above if nothing is passed.
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

    # cap is the open connection to the camera. Read frames from it,
    # close it when done, same as a file.
    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {args.index}. "
            "Check that it's connected and not in use by another app."
        )

    # Only a request, some cameras ignore it, hence checking actual after.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested resolution: {args.width}x{args.height}")
    print(f"Actual resolution:    {actual_width}x{actual_height}")

    # FPS measured by counting frames over a short window, then dividing.
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

            # waitKey also lets the window redraw, so it has to run even
            # when key presses aren't the concern.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Covers clicking the 'X' button instead of pressing 'q'.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Runs no matter how the loop ends, so the camera doesn't stay
        # locked for other programs.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
