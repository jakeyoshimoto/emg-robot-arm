"""
Finds the ball via vision/ball_detection.py and drives the arm's base
axis (axis 0, horizontal) and axis 2 (vertical - axis 1 is skipped, not
used for this) until the ball is centered in frame on both axes. This is
a first, deliberately narrow step toward a full pick-up sequence: it only
aims, doesn't reach or grab yet.

How a pixel offset becomes a step count:
    Each axis's pixel offset from its frame-center axis (horizontal
    offset from the vertical center line for axis 0, vertical offset
    from the horizontal center line for axis 2) is multiplied by that
    axis's hand-tunable gain (STEPS_PER_PIXEL / Y_STEPS_PER_PIXEL, not
    derived from camera calibration, since none has been run yet - see
    vision/camera_calibration.py) and direction sign (BASE_DIRECTION /
    Y_DIRECTION). Both axes sit behind the same 30:1 cycloidal gearbox
    (see GEAR_RATIO in arm/src/main.cpp). Y_DIRECTION hasn't been
    confirmed against real hardware yet (BASE_DIRECTION was, the hard
    way) - expect to flip it once tested.

Usage:
    python integration/ball_pickup.py --list-ports
    python integration/ball_pickup.py --port COM4
    python integration/ball_pickup.py --dry-run   # detect + print, don't move the arm

Shows a live camera view with the detected ball circled and reference
lines through the frame's horizontal and vertical center. Press SPACE to
start a centering run: on every camera frame it re-measures the ball's
offset on both axes and sends fresh corrections (or just prints them,
under --dry-run) proportional to those offsets - so the commanded move
shrinks, and the arm naturally slows down, on each axis independently as
it gets closer to centered, rather than one big jump that has to be
corrected afterward. It stops issuing moves once the ball holds within
CENTER_TOLERANCE_PX of center on both axes for STABLE_FRAMES_REQUIRED
frames in a row (debounced so one lucky/noisy frame doesn't end it
early), or after MAX_CENTERING_SECONDS if it never gets there. Only if it
actually settled (not on a timeout) does it then estimate distance to the
ball from its apparent radius (similar-triangles, see compute_distance_cm())
and print it, before going back to waiting for the next SPACE press.
Press 'q' any time
(including mid-run) or close the window to quit.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import serial
from serial.tools import list_ports

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from vision import ball_detection as bd  # noqa: E402  (needs REPO_ROOT on sys.path first)

BAUD_RATE = 115200

BASE_AXIS = 0              # axis 0 = base rotation (horizontal), see AXIS_PINS in arm/src/main.cpp
Y_AXIS = 2                 # axis 2 = vertical tilt; axis 1 is skipped, not used for this
STEPS_PER_MOTOR_REV = 200  # matches STEPS_PER_MOTOR_REV in arm/src/main.cpp
BASE_GEAR_RATIO = 30       # matches GEAR_RATIO[0] in arm/src/main.cpp
Y_GEAR_RATIO = 30          # matches GEAR_RATIO[2] in arm/src/main.cpp - same 30:1 as the base axis

# Hand-tunable proportional gains: motor steps per pixel of offset from
# frame center, one per axis since they may need different tuning.
# STEPS_PER_PIXEL was bumped up from an original 2.0 (too weak a turn on
# real hardware); Y_STEPS_PER_PIXEL just starts at that same value since
# it hasn't been tuned against real hardware yet either.
STEPS_PER_PIXEL = 6.0
Y_STEPS_PER_PIXEL = 6.0

# Flip to -1 if a positive offset (ball right of frame center / below
# frame center) ends up moving that axis the wrong way on real hardware.
# BASE_DIRECTION was confirmed on real hardware (+1 was backwards, hence
# -1 here); Y_DIRECTION is still an unverified guess - expect to flip it
# once tested.
BASE_DIRECTION = -1
Y_DIRECTION = 1

# The firmware clamps every axis to +/-1 output/joint rotation from
# wherever it powered on, as a stand-in for limit switches/homing - see
# maxAxisSteps() in arm/src/main.cpp. Both these axes sit behind a 30:1
# gearbox, so that's 6000 motor steps, not 200. Mirrored here so we can
# warn instead of silently sending a move that gets truncated.
MAX_BASE_AXIS_STEPS = STEPS_PER_MOTOR_REV * BASE_GEAR_RATIO
MAX_Y_AXIS_STEPS = STEPS_PER_MOTOR_REV * Y_GEAR_RATIO

# How close to frame-center counts as "centered" and stops the run. Ball
# radius is usually 15-40px at typical distances, so this is intentionally
# a bit loose rather than chasing exact pixel alignment.
CENTER_TOLERANCE_PX = 15

# Consecutive under-tolerance frames required before declaring "centered"
# and stopping, so a single noisy/lucky frame doesn't end the run early.
STABLE_FRAMES_REQUIRED = 3

# Minimum time between sent correction commands during a centering run.
# Short enough to feel continuous (re-targets several times a second as
# new frames come in) without flooding the serial link/motor with a new
# command on literally every camera frame.
CORRECTION_INTERVAL_SECONDS = 0.15

# Safety cap on how long a single centering run is allowed to keep
# correcting, in case it never converges (persistent detection noise, an
# overshooting gain, oscillation, etc).
MAX_CENTERING_SECONDS = 20.0

# Dynamic speed control, using the firmware's existing "speed m<axis> <v>"
# command. A proportional step count alone doesn't actually slow the arm
# down as it approaches: AccelStepper accelerates up to whatever max speed
# is set and cruises there for as long as the target keeps getting
# extended (which it does every tick while offset is still large), so it
# was cruising at the firmware's DEFAULT_MAX_SPEED (500 steps/sec, see
# arm/src/main.cpp) right up until nearly the end. Instead, explicitly cap
# and taper the speed itself: far from center it moves at
# MAX_SPEED_STEPS_PER_SEC (already well under the firmware default), and
# within SPEED_TAPER_RANGE_PX of center that tapers linearly down to
# MIN_SPEED_STEPS_PER_SEC.
FIRMWARE_DEFAULT_MAX_SPEED = 500  # matches DEFAULT_MAX_SPEED in arm/src/main.cpp, restored on exit
MAX_SPEED_STEPS_PER_SEC = 150
MIN_SPEED_STEPS_PER_SEC = 20
SPEED_TAPER_RANGE_PX = 150  # offset at/beyond which full MAX_SPEED_STEPS_PER_SEC applies

# Set True if the camera is physically mounted upside down - every frame
# gets rotated 180 degrees before detection/display. This also mirrors
# left/right in pixel space (a 180-degree rotation flips both axes), so
# BASE_DIRECTION was tuned against the unflipped image and may need
# re-checking on real hardware now that this is on.
CAMERA_UPSIDE_DOWN = True

# Distance estimation via similar triangles: distance = (real_radius *
# focal_length) / apparent_radius. BALL_RADIUS_CM is measured; FOCAL_LENGTH_PX
# came from a single-point calibration (not a full checkerboard calibration -
# no lens distortion correction), by placing the ball at a known 20in
# (50.8cm) from the camera and reading its apparent radius there (36.86px
# upside-down-corrected frame -> 36.86 * 50.8 / 3.5 = 534.94px). Re-derive
# this the same way if the camera or lens setup changes.
BALL_RADIUS_CM = 3.5
FOCAL_LENGTH_PX = 534.94

WINDOW_NAME = "ball_pickup"


class ArmLink:
    def __init__(self, port, baud_rate=BAUD_RATE):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # ESP32 resets when the serial port opens; wait for firmware to boot

    def move_axis(self, axis, steps):
        self.ser.write(f"m{axis} {steps}\n".encode("ascii"))

    def set_speed(self, axis, steps_per_sec):
        self.ser.write(f"speed m{axis} {steps_per_sec}\n".encode("ascii"))

    def stop_all(self):
        self.ser.write(b"stop\n")

    def close(self):
        self.ser.close()


def capture_and_detect(cap):
    ok, frame = cap.read()
    if not ok:
        return None, None
    if CAMERA_UPSIDE_DOWN:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    lower2 = np.array(bd.HSV_LOWER2) if bd.HSV_LOWER2 is not None else None
    upper2 = np.array(bd.HSV_UPPER2) if bd.HSV_UPPER2 is not None else None
    mask = bd.build_mask(frame, np.array(bd.HSV_LOWER), np.array(bd.HSV_UPPER), lower2, upper2)
    return frame, bd.detect_ball(mask)


def pixel_offset(ball_x_px, frame_width_px):
    # Positive => ball is right of center.
    return ball_x_px - (frame_width_px / 2.0)


def compute_move(offset_px, gain, direction):
    return int(round(offset_px * gain * direction))


def compute_speed(offset_px):
    fraction = min(abs(offset_px) / SPEED_TAPER_RANGE_PX, 1.0)
    return int(round(MIN_SPEED_STEPS_PER_SEC + (MAX_SPEED_STEPS_PER_SEC - MIN_SPEED_STEPS_PER_SEC) * fraction))


def compute_distance_cm(apparent_radius_px):
    return (BALL_RADIUS_CM * FOCAL_LENGTH_PX) / apparent_radius_px


def report_distance(cap):
    # Only meant to be called once a centering run has actually settled
    # (not on timeout/quit) - measuring mid-motion would read a radius
    # that's still changing along with everything else.
    frame, result = capture_and_detect(cap)
    if frame is None or result is None:
        print("Couldn't measure distance: no ball detected right after centering.")
        return
    radius = result[2]
    distance_cm = compute_distance_cm(radius)
    print(f"Distance to ball: {distance_cm:.1f} cm (radius={radius:.1f}px)")


def draw_overlay(frame, result, status_line, help_line):
    # Green circle + red center dot on the ball (same style as
    # vision/ball_detection.py), reference lines through the frame's
    # horizontal and vertical center to make both axes' offsets visible,
    # plus status/help text.
    display = frame.copy()
    h, w = display.shape[:2]
    cv2.line(display, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.line(display, (0, h // 2), (w, h // 2), (255, 255, 0), 1)

    if result is not None:
        x, y, radius = result
        cv2.circle(display, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(display, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.putText(display, status_line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, help_line, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return display


def check_quit():
    key = cv2.waitKey(1) & 0xFF
    return key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def run_centering(cap, arm, dry_run):
    # Runs at camera frame rate: every frame re-measures the ball's offset
    # on both axes and, at most every CORRECTION_INTERVAL_SECONDS, sends
    # fresh proportional corrections from wherever each axis actually is
    # right now (not a pre-planned trajectory), along with a speed cap
    # that tapers down near center (see compute_speed()) so each axis
    # actually slows down on approach instead of cruising at full speed
    # and stopping abruptly. Stops once STABLE_FRAMES_REQUIRED consecutive
    # frames read within CENTER_TOLERANCE_PX on both axes, or after
    # MAX_CENTERING_SECONDS if it never settles. Returns (quit, centered):
    # quit is True if the user quit mid-run (caller should stop entirely,
    # not go back to idle); centered is True only if it actually settled
    # (not on timeout/frame-failure/quit) - callers that need to know the
    # ball is genuinely still and centered (e.g. before measuring distance)
    # should gate on that, not just on quit being False.
    try:
        return _run_centering(cap, arm, dry_run)
    finally:
        # Restore the firmware's normal max speed on both axes so anything
        # after this run (another centering run, manual jogging over
        # serial, etc.) isn't stuck at the slow taper speed from the end
        # of this one.
        if not dry_run:
            arm.set_speed(BASE_AXIS, FIRMWARE_DEFAULT_MAX_SPEED)
            arm.set_speed(Y_AXIS, FIRMWARE_DEFAULT_MAX_SPEED)


def _send_correction(arm, dry_run, axis, axis_name, offset, gain, direction, max_steps):
    steps = compute_move(offset, gain, direction)
    speed = compute_speed(offset)
    if abs(steps) > max_steps:
        print(f"Warning: {steps} steps exceeds the firmware's +/-{max_steps}-step safety cap "
              "(1 output rev from power-on) and will be clamped on the arm side.")
    if dry_run:
        print(f"{axis_name}_offset={offset:+.1f}px -> speed={speed} steps/s, m{axis} {steps} "
              "(--dry-run, not sent)")
    else:
        arm.set_speed(axis, speed)
        arm.move_axis(axis, steps)
        print(f"{axis_name}_offset={offset:+.1f}px -> speed={speed} steps/s, m{axis} {steps}")


def _run_centering(cap, arm, dry_run):
    stable_count = 0
    last_send_time = 0.0
    start_time = time.time()

    while True:
        if time.time() - start_time > MAX_CENTERING_SECONDS:
            print(f"Centering timed out after {MAX_CENTERING_SECONDS:.0f}s without settling; stopping.")
            return False, False

        frame, result = capture_and_detect(cap)
        if frame is None:
            print("Frame grab failed; stopping centering run.")
            return False, False

        if result is None:
            stable_count = 0
            cv2.imshow(WINDOW_NAME, draw_overlay(frame, None, "no ball detected - waiting", "q: abort"))
            if check_quit():
                print("Quit.")
                return True, False
            continue

        x, y = result[0], result[1]
        x_offset = pixel_offset(x, frame.shape[1])
        y_offset = pixel_offset(y, frame.shape[0])
        x_centered = abs(x_offset) <= CENTER_TOLERANCE_PX
        y_centered = abs(y_offset) <= CENTER_TOLERANCE_PX

        if x_centered and y_centered:
            # Actively brake rather than just assuming the motors already
            # stopped: they may still be coasting toward whatever (larger)
            # target was in flight from just before the offset crossed
            # into tolerance, and if left alone can keep drifting past
            # center well after this loop declares victory and moves on -
            # which is exactly the overshoot this was producing before.
            if not dry_run:
                arm.stop_all()
            stable_count += 1
            status = (f"centered (x={x_offset:+.0f}px y={y_offset:+.0f}px, "
                      f"holding {stable_count}/{STABLE_FRAMES_REQUIRED})")
            cv2.imshow(WINDOW_NAME, draw_overlay(frame, result, status, "q: abort"))
            if stable_count >= STABLE_FRAMES_REQUIRED:
                print(f"Centered: x_offset={x_offset:+.1f}px y_offset={y_offset:+.1f}px.")
                return False, True
        else:
            stable_count = 0
            preview_x_steps = compute_move(x_offset, STEPS_PER_PIXEL, BASE_DIRECTION)
            preview_y_steps = compute_move(y_offset, Y_STEPS_PER_PIXEL, Y_DIRECTION)
            status = (f"turning: x={x_offset:+.0f}px({preview_x_steps}) "
                      f"y={y_offset:+.0f}px({preview_y_steps})")
            cv2.imshow(WINDOW_NAME, draw_overlay(frame, result, status, "q: abort"))

            now = time.time()
            if now - last_send_time >= CORRECTION_INTERVAL_SECONDS:
                if not x_centered:
                    _send_correction(arm, dry_run, BASE_AXIS, "x", x_offset,
                                      STEPS_PER_PIXEL, BASE_DIRECTION, MAX_BASE_AXIS_STEPS)
                if not y_centered:
                    _send_correction(arm, dry_run, Y_AXIS, "y", y_offset,
                                      Y_STEPS_PER_PIXEL, Y_DIRECTION, MAX_Y_AXIS_STEPS)
                last_send_time = now

        if check_quit():
            print("Quit.")
            return True, False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=str, default=None, help="serial port for the arm, e.g. COM4")
    parser.add_argument("--list-ports", action="store_true", help="list available serial ports and exit")
    parser.add_argument("--index", type=int, default=bd.CAMERA_INDEX, help="camera index")
    parser.add_argument("--dry-run", action="store_true",
                         help="detect the ball and print each computed move, but don't send anything")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_ports:
        for p in list_ports.comports():
            print(f"{p.device}  {p.description}")
        return

    if not args.dry_run and not args.port:
        raise SystemExit("Specify --port (use --list-ports to see available ports), or pass --dry-run.")

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {args.index}.")

    # Connect up front (not just-in-time on SPACE) so there's no extra
    # ESP32-reset delay between deciding to start and it actually moving.
    arm = None if args.dry_run else ArmLink(args.port)

    try:
        while True:
            frame, result = capture_and_detect(cap)
            if frame is None:
                print("Frame grab failed; stopping.")
                break

            if result is not None:
                x_offset = pixel_offset(result[0], frame.shape[1])
                y_offset = pixel_offset(result[1], frame.shape[0])
                status = f"ball x={result[0]:.0f} y={result[1]:.0f}  offset=({x_offset:+.0f},{y_offset:+.0f})px"
            else:
                status = "no ball detected"
            cv2.imshow(WINDOW_NAME, draw_overlay(frame, result, status, "SPACE: start centering   q: quit"))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit.")
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord(" "):
                quit_requested, centered = run_centering(cap, arm, args.dry_run)
                if centered:
                    report_distance(cap)
                if quit_requested:
                    break  # user quit mid-run
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    main()
