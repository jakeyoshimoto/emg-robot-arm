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

First run downloads the hand landmark model file (a few MB) from Google
and caches it in vision/models/. Needs internet access once.

Usage:
    python vision/hand_tracking.py
    python vision/hand_tracking.py --index 1
    python vision/hand_tracking.py --max-hands 2

Press 'q' or close the window to exit.

Per-frame output:
    Prints "hand N: open (thumb:1 index:0 middle:1 ring:0 pinky:0)" for
    each detected hand, where 1 = extended and 0 = curled.
"""

import argparse
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
THUMB_MCP = 2
THUMB_TIP = 4
PINKY_MCP = 17

FINGER_DISPLAY_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

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


def classify_finger_states(landmarks):
    # landmarks is one hand's 21 points, normalized (0-1) image
    # coordinates. Returns {finger_name: 1 | 0} (1 = extended) for all
    # five fingers, thumb included.
    def dist(a_idx, b_idx):
        a, b = landmarks[a_idx], landmarks[b_idx]
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    finger_states = {}
    for name, (knuckle_idx, tip_idx) in FINGER_JOINTS.items():
        extended = dist(tip_idx, WRIST) > dist(knuckle_idx, WRIST)
        finger_states[name] = 1 if extended else 0

    thumb_extended = dist(THUMB_TIP, PINKY_MCP) > dist(THUMB_MCP, PINKY_MCP)
    finger_states["thumb"] = 1 if thumb_extended else 0
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
                print(f"hand {i}: {state} ({fingers_str})")

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
