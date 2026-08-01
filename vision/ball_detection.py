"""
Finds a single, known-colored ball in the live camera feed and reports
its pixel position and size, frame by frame. Plain HSV color filtering
plus a roundness check, no ML involved.

Usage:
    python vision/ball_detection.py
    python vision/ball_detection.py --index 1
    python vision/ball_detection.py --tune   # show HSV trackbars for live tuning

Controls:
    'q' - quit
    (with --tune) drag the trackbars in the "tuning" window until only the
    ball is white in the "mask" window, then copy the printed HSV values
    into HSV_LOWER / HSV_UPPER below. --tune only exposes one band; if the
    color needs the red-style wraparound second band, set HSV_LOWER2 /
    HSV_UPPER2 by hand afterward.

Per-frame output:
    Prints "x=<px> y=<px> radius=<px>" when the ball is detected, once per
    frame, to stdout. Also returned as (x, y, radius) or None from
    detect_ball() for programmatic use by later pipeline stages.
"""

import argparse

import cv2
import numpy as np

# HSV holds up better than RGB under uneven lighting, since brightness
# changes mostly just move V, leaving H alone. HSV_LOWER/HSV_UPPER define
# a box in H/S/V space; a pixel is kept if it falls inside those bounds.
# A second band (HSV_LOWER2/HSV_UPPER2) is only needed for colors like red
# that wrap around past hue 179; blue sits mid-range so one band covers it.

HSV_LOWER = (95, 80, 55)
HSV_UPPER = (130, 255, 255)
HSV_LOWER2 = None
HSV_UPPER2 = None

CAMERA_INDEX = 0
MIN_RADIUS_PX = 8          # blobs smaller than this get ignored as noise, not the ball
MIN_CONTOUR_AREA_PX = 150  # a second, area-based noise filter, belt-and-braces with the radius one
MIN_CIRCULARITY = 0.55     # how round a blob has to be to count as "the ball". see detect_ball()

# Balls with non-target-colored markings (e.g. basketball-style seam
# lines) leave thin gaps in the mask that can fragment one ball into
# several smaller blobs, each too small/irregular to pass the checks
# above. This closes gaps up to roughly this many pixels wide before
# contours are found, so those fragments merge back into one blob.
STRIPE_CLOSE_KERNEL_PX = 15

WINDOW_NAME = "ball_detection"
TUNE_WINDOW_NAME = "tuning"
MASK_WINDOW_NAME = "mask"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--tune", action="store_true",
                         help="show HSV trackbars for interactive threshold tuning")
    return parser.parse_args()


def _nothing(_value):
    # OpenCV's trackbar API requires a callback, but slider positions are
    # only read once per frame in read_trackbars(), so there's nothing to
    # do here.
    pass


def create_trackbars():
    # Six sliders: low/high for each of H, S, V. Lets the color range be
    # tuned live instead of editing constants and restarting the script.
    cv2.namedWindow(TUNE_WINDOW_NAME)
    cv2.createTrackbar("H min", TUNE_WINDOW_NAME, HSV_LOWER[0], 179, _nothing)
    cv2.createTrackbar("S min", TUNE_WINDOW_NAME, HSV_LOWER[1], 255, _nothing)
    cv2.createTrackbar("V min", TUNE_WINDOW_NAME, HSV_LOWER[2], 255, _nothing)
    cv2.createTrackbar("H max", TUNE_WINDOW_NAME, HSV_UPPER[0], 179, _nothing)
    cv2.createTrackbar("S max", TUNE_WINDOW_NAME, HSV_UPPER[1], 255, _nothing)
    cv2.createTrackbar("V max", TUNE_WINDOW_NAME, HSV_UPPER[2], 255, _nothing)


def read_trackbars():
    # Packages the current slider positions into the same (H, S, V)
    # lower/upper format as the constants above, so tuning mode and
    # normal mode share the same detection code.
    lower = np.array([
        cv2.getTrackbarPos("H min", TUNE_WINDOW_NAME),
        cv2.getTrackbarPos("S min", TUNE_WINDOW_NAME),
        cv2.getTrackbarPos("V min", TUNE_WINDOW_NAME),
    ])
    upper = np.array([
        cv2.getTrackbarPos("H max", TUNE_WINDOW_NAME),
        cv2.getTrackbarPos("S max", TUNE_WINDOW_NAME),
        cv2.getTrackbarPos("V max", TUNE_WINDOW_NAME),
    ])
    return lower, upper


def build_mask(frame_bgr, lower, upper, lower2=None, upper2=None):
    # White wherever a pixel's color falls inside the HSV range, black
    # elsewhere, i.e. everywhere the ball's color appears to show up.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if lower2 is not None:
        # Second band for red's wraparound hue; OR the two masks together.
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bridge gaps left by any non-target-colored markings on the ball
    # (e.g. seam lines) so they don't fragment it into multiple blobs -
    # see STRIPE_CLOSE_KERNEL_PX above.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (STRIPE_CLOSE_KERNEL_PX, STRIPE_CLOSE_KERNEL_PX)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def detect_ball(mask):
    # Picks whichever white blob in the mask looks most like the ball.
    # Returns its (x, y, radius) in pixels, or None if nothing qualifies.
    #
    # Circularity = blob area / area of its smallest enclosing circle. A
    # perfect circle scores close to 1.0. Only blobs above
    # MIN_CIRCULARITY count, and the largest of those wins.

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA_PX:
            continue  # too small to be the ball, treated as noise

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < MIN_RADIUS_PX:
            continue

        circularity = area / (np.pi * radius * radius)
        if circularity < MIN_CIRCULARITY:
            continue  # not round enough, probably not the ball

        if area > best_area:
            best_area = area
            best = (x, y, radius)

    return best


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {args.index}.")

    if args.tune:
        create_trackbars()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            if args.tune:
                # Tuning mode uses the sliders instead of the fixed
                # constants, and only the one band.
                lower, upper = read_trackbars()
                mask = build_mask(frame, lower, upper)
            else:
                lower, upper = np.array(HSV_LOWER), np.array(HSV_UPPER)
                lower2 = np.array(HSV_LOWER2) if HSV_LOWER2 is not None else None
                upper2 = np.array(HSV_UPPER2) if HSV_UPPER2 is not None else None
                mask = build_mask(frame, lower, upper, lower2, upper2)
            result = detect_ball(mask)

            display = frame.copy()
            if result is not None:
                x, y, radius = result
                # Green circle around the ball, red dot at its center.
                cv2.circle(display, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(display, (int(x), int(y)), 3, (0, 0, 255), -1)
                print(f"x={x:.1f} y={y:.1f} radius={radius:.1f}")

            cv2.imshow(WINDOW_NAME, display)
            if args.tune:
                cv2.imshow(MASK_WINDOW_NAME, mask)  # raw mask for the current HSV range

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Read the final slider positions (if any) before tearing down the
        # window they live on - destroyAllWindows() first would leave
        # read_trackbars() nothing to read from and crash on exit.
        if args.tune:
            lower, upper = read_trackbars()

        cap.release()
        cv2.destroyAllWindows()

        if args.tune:
            print(f"\nFinal HSV range, lower={tuple(lower)} upper={tuple(upper)}")
            print("Copy these into HSV_LOWER / HSV_UPPER at the top of this file.")


if __name__ == "__main__":
    main()
