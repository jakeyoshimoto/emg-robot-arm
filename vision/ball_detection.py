"""
Finds a single, known-colored ball in the live camera feed and works out
its pixel position and size, frame by frame. Simple color filtering plus
some basic shape checking.

How it works:
Convert each frame to HSV color space. Keep only the pixels that fall
inside a chosen color range (mask). Find the blob(s) of
leftover pixels. Pick whichever blob looks most like a round ball.

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

# Using HSV RGB because lighting changes move the V number around while
# leaving H roughly alone. Filtering by hue holds up much better across
# a room with uneven lighting than filtering by raw color would.

# HSV_LOWER/HSV_UPPER define a box in that H/S/V space. Any pixel whose
# H, S, and V all fall inside those bounds gets kept.

HSV_LOWER = (0, 170, 60)
HSV_UPPER = (10, 255, 255)
HSV_LOWER2 = (170, 170, 60)
HSV_UPPER2 = (179, 255, 255)

CAMERA_INDEX = 0
MIN_RADIUS_PX = 8          # blobs smaller than this get ignored as noise, not the ball
MIN_CONTOUR_AREA_PX = 150  # a second, area-based noise filter, belt-and-braces with the radius one
MIN_CIRCULARITY = 0.55     # how round a blob has to be to count as "the ball". see detect_ball()

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
    # OpenCV's trackbar function requires a callback to run whenever the
    # slider moves, but nothing needs to happen instantly. Current slider
    # positions get read once per frame in read_trackbars() instead. So
    # this callback intentionally does nothing.
    pass


def create_trackbars():
    # Opens a small window with six sliders, low/high for each of H, S, V.
    # Lets the color range be adjusted live instead of editing the
    # constants above and restarting the script repeatedly.
    cv2.namedWindow(TUNE_WINDOW_NAME)
    cv2.createTrackbar("H min", TUNE_WINDOW_NAME, HSV_LOWER[0], 179, _nothing)
    cv2.createTrackbar("S min", TUNE_WINDOW_NAME, HSV_LOWER[1], 255, _nothing)
    cv2.createTrackbar("V min", TUNE_WINDOW_NAME, HSV_LOWER[2], 255, _nothing)
    cv2.createTrackbar("H max", TUNE_WINDOW_NAME, HSV_UPPER[0], 179, _nothing)
    cv2.createTrackbar("S max", TUNE_WINDOW_NAME, HSV_UPPER[1], 255, _nothing)
    cv2.createTrackbar("V max", TUNE_WINDOW_NAME, HSV_UPPER[2], 255, _nothing)


def read_trackbars():
    # Reads wherever the six sliders currently sit and packages them up
    # into the same (H, S, V) lower/upper format the rest of the script
    # expects, so tuning mode and normal mode can share the same code path.
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
    # A mask is a black-and-white image the same size as the
    # camera frame. White wherever a pixel's color fell inside the chosen
    # HSV range, black everywhere else. Marks everywhere the ball's color
    # appears to be showing up.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if lower2 is not None:
        # For red, a second band catches the other half of the red
        # hues that wrapped around past 179. bitwise_or merges the two
        # black-and-white masks together, a pixel shows up white if it
        # matched either band.
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def detect_ball(mask):
    # Looks at every separate white blob in the mask and decides which
    # one actually the ball. Returns its (x, y, radius) in pixels, 
    # or None if nothing qualifies.

    # For every blob, area is measured, and the smallest possible circle
    # that fully contains it also gets fit (minEnclosingCircle). Dividing
    # the two, blob area vs that circle's area, gives a score. A blob 
    # shaped like a circle fills almost all of its enclosing circle, scoring 
    # close to 1.0. Only blobs above MIN_CIRCULARITY count, and the largest 
    # one gets picked.
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # findContours traces the outline of every separate white blob in the
    # mask and hands them back as a list. contours is that list, one
    # entry per blob.

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
                # In tuning mode, use the sliders instead of the fixed constants,
                # and only the one band.
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
                # Draws a green circle around the detected ball and a
                # small red dot at its center, purely as a visual
                # confirmation that the detection looks right.
                cv2.circle(display, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(display, (int(x), int(y)), 3, (0, 0, 255), -1)
                print(f"x={x:.1f} y={y:.1f} radius={radius:.1f}")

            cv2.imshow(WINDOW_NAME, display)
            if args.tune:
                # Shows the raw black-and-white mask based off 
                # the current HSV range.
                cv2.imshow(MASK_WINDOW_NAME, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        if args.tune:
            # Prints the final slider positions.
            lower, upper = read_trackbars()
            print(f"\nFinal HSV range, lower={tuple(lower)} upper={tuple(upper)}")
            print("Copy these into HSV_LOWER / HSV_UPPER at the top of this file.")


if __name__ == "__main__":
    main()
