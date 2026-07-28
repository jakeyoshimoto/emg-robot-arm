"""
Drives the hand's servos from live hand tracking: each finger's
extended/curled state (from vision/hand_tracking.py) maps to one PCA9685
servo channel, sent as the `s<channel> <angle>` serial command that
arm/src/main.cpp already parses. No firmware changes needed.

Reuses vision/hand_tracking.py's camera + MediaPipe pipeline instead of
duplicating it; only the per-frame action changes (send a servo command
instead of just printing the finger states).

Usage:
    python integration/hand_control.py --list-ports
    python integration/hand_control.py --port COM4
    python integration/hand_control.py --port COM4 --index 1

Press 'q' or close the window to exit.
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import serial
from serial.tools import list_ports

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from vision import hand_tracking as ht  # noqa: E402  (needs REPO_ROOT on sys.path first)

BAUD_RATE = 115200

# Per-finger servo calibration: PCA9685 channel (matching NUM_SERVOS=6
# in arm/src/main.cpp) plus the angle sent for curled vs extended. Each
# servo is mounted at its own orientation, so curled/extended need
# calibrating individually per finger - edit these once real angles are
# known.
FINGER_SERVOS = {
    "index": {"channel": 2, "curled": 0, "extended": 180},
    "middle": {"channel": 3, "curled": 0, "extended": 180},
    "ring": {"channel": 4, "curled": 0, "extended": 180},
    "pinky": {"channel": 5, "curled": 180, "extended": 0},
}

# The thumb drives two servos off the same classify_finger_states()
# "thumb" signal (splitting that into separate lower-joint/curl vision
# signals was tried and proved too noisy, see CLAUDE.md), sent one after
# the other rather than simultaneously so the joints don't fight each
# other: closing curls the lower joint (0) first, then the thumb (1);
# opening reverses that order. Angles are calibrated the same way as
# FINGER_SERVOS above - edit once real angles are known.
THUMB_LOWER_SERVO = {"channel": 0, "curled": 50, "extended": 180}
THUMB_SERVO = {"channel": 1, "curled": 0, "extended": 180}
THUMB_SEQUENCE_DELAY = 0.15  # seconds between the lower-joint and thumb servo moves


class ArmLink:
    # Wraps the serial connection to the arm board and only sends a hand
    # servo command when a finger's state actually changes, instead of
    # every frame, to avoid saturating the link and jittering the servos.
    def __init__(self, port, baud_rate=BAUD_RATE):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # ESP32 resets when the serial port opens; wait for firmware to boot
        self._last_angle = {}  # channel -> last angle sent

    def send_finger_states(self, finger_states):
        if "thumb" in finger_states:
            self._send_thumb(bool(finger_states["thumb"]))
        for name, servo in FINGER_SERVOS.items():
            if name not in finger_states:
                continue
            angle = servo["extended"] if finger_states[name] else servo["curled"]
            self._send(servo["channel"], angle)

    def _send_thumb(self, extended):
        lower_angle = THUMB_LOWER_SERVO["extended"] if extended else THUMB_LOWER_SERVO["curled"]
        thumb_angle = THUMB_SERVO["extended"] if extended else THUMB_SERVO["curled"]
        if (
            self._last_angle.get(THUMB_LOWER_SERVO["channel"]) == lower_angle
            and self._last_angle.get(THUMB_SERVO["channel"]) == thumb_angle
        ):
            return
        if extended:
            self._send(THUMB_SERVO["channel"], thumb_angle)
            time.sleep(THUMB_SEQUENCE_DELAY)
            self._send(THUMB_LOWER_SERVO["channel"], lower_angle)
        else:
            self._send(THUMB_LOWER_SERVO["channel"], lower_angle)
            time.sleep(THUMB_SEQUENCE_DELAY)
            self._send(THUMB_SERVO["channel"], thumb_angle)

    def _send(self, channel, angle):
        if self._last_angle.get(channel) == angle:
            return
        self.ser.write(f"s{channel} {angle}\n".encode("ascii"))
        self._last_angle[channel] = angle

    def close(self):
        self.ser.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=str, default=None, help="serial port for the arm, e.g. COM4")
    parser.add_argument("--list-ports", action="store_true", help="list available serial ports and exit")
    parser.add_argument("--index", type=int, default=ht.CAMERA_INDEX, help="camera index")
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--model", type=str, default=ht.MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_ports:
        for p in list_ports.comports():
            print(f"{p.device}  {p.description}")
        return

    if not args.port:
        raise SystemExit("Specify --port (use --list-ports to see available ports).")

    ht.ensure_model(args.model)
    landmarker = ht.build_landmarker(args.model, args.max_hands)
    arm = ArmLink(args.port)

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {args.index}.")

    start_time = time.time()
    last_timestamp_ms = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            # MediaPipe expects RGB, OpenCV gives BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # detect_for_video needs a strictly increasing timestamp per call.
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            display = frame.copy()
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                finger_states = ht.classify_finger_states(hand_landmarks)
                state = ht.classify_hand_state(finger_states)
                ht.draw_hand(display, hand_landmarks, state)
                if i == 0:  # only the first detected hand drives the arm
                    arm.send_finger_states(finger_states)

            cv2.imshow(ht.WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(ht.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        arm.close()


if __name__ == "__main__":
    main()
