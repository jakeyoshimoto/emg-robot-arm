"""
Tracks a hand in the live camera feed and classifies it as open or
closed, frame by frame. Uses MediaPipe's hand landmark model
rather than color thresholding, since a hand isn't one fixed color and
its shape changes a lot pose to pose.

For each of the four non-thumb fingers: extended if the fingertip is
farther from the wrist than its knuckle is, curled if closer. The thumb
folds across the palm instead of toward the wrist, so it's checked
against the pinky's base knuckle instead. Overall state is "open" if 2+
of the four non-thumb fingers are extended; the thumb is reported
per-finger but doesn't count toward that total.

The thumb has two independent signals, since it has two joints that
move separately: "thumb_lower" is the base joint's rotation/opposition
across the palm, measured as the angle at the CMC joint between the MCP
and pinky-knuckle landmarks (not the tip, so curling the last joint
doesn't affect it); "thumb" is just the bend at the last joint, measured
in 3D (using MediaPipe's relative z) so it stays flat when the base
rotates instead of picking up 2D foreshortening as a "bend". Both are
angle-based with a wide swing between straight and curled, which is
more resistant to landmark jitter than a raw distance comparison.

First run downloads the hand landmark model file (a few MB) from Google
and caches it in vision/models/. Needs internet access once.

Usage:
    python vision/hand_tracking.py
    python vision/hand_tracking.py --index 1
    python vision/hand_tracking.py --max-hands 2

Press 'q' or close the window to exit.

Per-frame output:
    Prints "hand N: open (thumb_lower:1 thumb:0 index:0 middle:1 ring:0
    pinky:0) thumb_lower_angle=142 thumb_angle=61" for each detected
    hand - the angle suffixes are the raw joint angles behind
    thumb_lower/thumb, useful for tuning the two THUMB_*_STRAIGHT_ANGLE_DEG
    thresholds below against your own camera/hand.
"""

import argparse
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

CAMERA_INDEX = 0
MAX_HANDS = 1
EXTENDED_FINGERS_FOR_OPEN = 2  # out of 4, thumb excluded, see module docstring

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "vision", "models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

WINDOW_NAME = "hand_tracking"

# Landmark indices per MediaPipe's hand layout, (knuckle, tip) per finger.
FINGER_JOINTS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
WRIST = 0

# Thumb doesn't fold toward the wrist like the other fingers, so it's
# checked against the pinky's base knuckle instead (see module docstring).
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
PINKY_MCP = 17

# Below these angles (degrees), the corresponding thumb joint is
# considered curled rather than straight. 180 = perfectly straight.
# Both are starting points - tune against the printed raw angles
# (see module docstring) for your own hand/camera setup.
THUMB_LOWER_STRAIGHT_ANGLE_DEG = 90
THUMB_STRAIGHT_ANGLE_DEG = 160

FINGER_DISPLAY_ORDER = ["thumb_lower", "thumb", "index", "middle", "ring", "pinky"]

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--max-hands", type=int, default=MAX_HANDS,
                         help="max hands to track at once (default: %(default)s)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                         help="path to hand_landmarker.task (default: %(default)s)")
    return parser.parse_args()


def ensure_model(model_path):
    # Downloads the model once and caches it locally.
    if os.path.exists(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Hand landmark model not found, downloading to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the hand landmark model: {exc}. "
            f"Download it manually from {MODEL_URL} and save it to {model_path}."
        ) from exc
    print("Done.")


def build_landmarker(model_path, max_hands):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=max_hands,
    )
    return HandLandmarker.create_from_options(options)


def _joint_angle_deg(landmarks, a_idx, b_idx, c_idx):
    # Angle at b_idx, in 3D, between rays toward a_idx and c_idx. Uses
    # MediaPipe's relative z (not just x, y) so the angle reflects the
    # joint's actual bend, not 2D foreshortening when the joint rotates
    # out of the image plane. 180 = the three points are collinear
    # (joint is straight).
    a, b, c = landmarks[a_idx], landmarks[b_idx], landmarks[c_idx]
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    mag1 = math.sqrt(sum(v * v for v in v1))
    mag2 = math.sqrt(sum(v * v for v in v2))
    if mag1 == 0 or mag2 == 0:
        return 180.0
    dot = sum(p * q for p, q in zip(v1, v2))
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def classify_finger_states(landmarks):
    # landmarks is one hand's 21 points, normalized (0-1) image
    # coordinates. Returns {finger_name: 1 | 0} (1 = extended) for the
    # four non-thumb fingers, plus "thumb_lower" and "thumb" (see module
    # docstring for what each thumb signal means), plus the raw angles
    # behind those two ("thumb_lower_angle_deg", "thumb_angle_deg") for
    # tuning - not 1/0, ignore these two if you just want finger state.
    def dist(a_idx, b_idx):
        a, b = landmarks[a_idx], landmarks[b_idx]
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    finger_states = {}
    for name, (knuckle_idx, tip_idx) in FINGER_JOINTS.items():
        extended = dist(tip_idx, WRIST) > dist(knuckle_idx, WRIST)
        finger_states[name] = 1 if extended else 0

    # Angle at the CMC joint between the MCP and pinky-knuckle rays -
    # not the tip, so curling the last joint ("thumb", below) can't
    # move this signal. Shrinks as the thumb sweeps toward opposition.
    thumb_lower_angle = _joint_angle_deg(landmarks, THUMB_MCP, THUMB_CMC, PINKY_MCP)
    finger_states["thumb_lower"] = 1 if thumb_lower_angle >= THUMB_LOWER_STRAIGHT_ANGLE_DEG else 0
    finger_states["thumb_lower_angle_deg"] = thumb_lower_angle

    thumb_angle = _joint_angle_deg(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP)
    finger_states["thumb"] = 1 if thumb_angle >= THUMB_STRAIGHT_ANGLE_DEG else 0
    finger_states["thumb_angle_deg"] = thumb_angle
    return finger_states


def classify_hand_state(finger_states):
    # Non-thumb fingers only, see module docstring.
    extended_count = sum(finger_states[name] for name in FINGER_JOINTS)
    return "open" if extended_count >= EXTENDED_FINGERS_FOR_OPEN else "closed"


def draw_hand(display, landmarks, state):
    # Landmarks are normalized (0-1), scale to actual pixel positions
    # before drawing.
    height, width = display.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for connection in HAND_CONNECTIONS:
        cv2.line(display, points[connection.start], points[connection.end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(display, point, 4, (0, 0, 255), -1)

    label_pos = (points[WRIST][0] - 20, points[WRIST][1] + 40)
    cv2.putText(display, state, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main():
    args = parse_args()
    ensure_model(args.model)
    landmarker = build_landmarker(args.model, args.max_hands)

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
                finger_states = classify_finger_states(hand_landmarks)
                state = classify_hand_state(finger_states)
                draw_hand(display, hand_landmarks, state)
                fingers_str = " ".join(
                    f"{name}:{finger_states[name]}" for name in FINGER_DISPLAY_ORDER
                )
                print(
                    f"hand {i}: {state} ({fingers_str}) "
                    f"thumb_lower_angle={finger_states['thumb_lower_angle_deg']:.0f} "
                    f"thumb_angle={finger_states['thumb_angle_deg']:.0f}"
                )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
