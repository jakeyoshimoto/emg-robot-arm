"""
Calibrates the camera: works out its camera matrix (focal length,
optical center) and lens distortion coefficients, using OpenCV's
standard checkerboard method. Needed later to turn a pixel position
into a real-world position/angle.

Show the camera a checkerboard from a bunch of angles and distances.
OpenCV finds the inner corners in each captured frame and compares them
against the known flat layout to solve for the lens math.

Usage:
    python vision/camera_calibration.py
    python vision/camera_calibration.py --index 1 --cols 9 --rows 6 --square-size 25.0

Controls during capture:
    'c' - capture the current frame (only works when the board is detected)
    'q' - stop capturing early and run calibration on what's been collected

Board size note:
    --cols/--rows are the number of INNER corners (where black squares
    meet), not the number of squares. A standard "9x6" chessboard has
    9x6 inner corners, i.e. 10x7 squares.
"""

import argparse
import os

import cv2
import numpy as np

# Defaults, override via CLI args.
CAMERA_INDEX = 0
BOARD_COLS = 9           # inner corners, horizontal
BOARD_ROWS = 6           # inner corners, vertical
SQUARE_SIZE_MM = 25.0    # real-world size of one checkerboard square
NUM_FRAMES_TARGET = 20   # how many good captures to collect before calibrating
REPROJECTION_ERROR_THRESHOLD = 0.5  # px; flag calibration if worse than this

# Built relative to this file's own location, so it works no matter which
# folder the script is run from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "hardware", "camera_calibration.yaml")

WINDOW_NAME = "camera_calibration"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--cols", type=int, default=BOARD_COLS,
                         help="inner corners per row (default: %(default)s)")
    parser.add_argument("--rows", type=int, default=BOARD_ROWS,
                         help="inner corners per column (default: %(default)s)")
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE_MM,
                         help="checkerboard square size in mm (default: %(default)s)")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES_TARGET,
                         help="number of good captures to collect (default: %(default)s)")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                         help="output YAML path (default: %(default)s)")
    return parser.parse_args()


def build_object_points(cols, rows, square_size):
    # Ground-truth 3D corner positions, board flat with top-left corner
    # at (0, 0, 0), scaled to real millimeters. Every captured frame gets
    # compared against this same ideal layout.
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def main():
    args = parse_args()
    board_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_size)

    # Stop corner refinement after 30 iterations or 0.001px precision,
    # whichever comes first.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = []   # 3D points, one array per captured frame
    image_points = []    # 2D points, one array per captured frame

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {args.index}.")

    frame_size = None  # (width, height), set from the first frame

    try:
        while len(object_points) < args.frames:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])

            # Corner-finding works on grayscale, so convert first.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size, None)

            display = frame.copy()  # draw on a copy so the raw frame stays clean
            if found:
                cv2.drawChessboardCorners(display, board_size, corners, found)

            status = f"captured {len(object_points)}/{args.frames}"
            hint = "'c' to capture, 'q' to stop" if found else "board not detected"
            cv2.putText(display, f"{status} | {hint}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and found:
                # Refine the approximate corners to sub-pixel accuracy.
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                object_points.append(objp)
                image_points.append(refined)
                print(f"Captured frame {len(object_points)}/{args.frames}")
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(object_points) < 4:
        # Too few frames/angles gives an untrustworthy result.
        print(
            f"Only {len(object_points)} frame(s) captured; need at least a "
            "handful (10+ recommended) from varied angles for a stable "
            "calibration. Not saving a result."
        )
        return

    # Solves for the camera matrix and distortion coefficients that best
    # explain object_points vs image_points across all captured frames.
    # reprojection_error is how far off, in pixels, the solved model's
    # predicted corners are from where the corners actually were.
    reprojection_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, frame_size, None, None
    )

    print(f"\nUsed {len(object_points)} frames.")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
    print(f"Reprojection error: {reprojection_error:.4f} px")

    if reprojection_error > REPROJECTION_ERROR_THRESHOLD:
        print(
            f"WARNING: reprojection error {reprojection_error:.4f} px exceeds "
            f"the {REPROJECTION_ERROR_THRESHOLD} px quality threshold. "
            "Calibration will still be saved, but consider recapturing with "
            "more frames / more varied angles and distances before trusting it."
        )

    save_calibration(args.output, camera_matrix, dist_coeffs, frame_size,
                      reprojection_error, len(object_points))


def save_calibration(output_path, camera_matrix, dist_coeffs, frame_size,
                      reprojection_error, num_frames):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # OpenCV's own YAML format, so any other OpenCV script can load it
    # back in with a couple of lines.
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", frame_size[0])
    fs.write("image_height", frame_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("reprojection_error", reprojection_error)
    fs.write("num_frames_used", num_frames)
    fs.release()

    print(f"Saved calibration to {output_path}")


if __name__ == "__main__":
    main()
