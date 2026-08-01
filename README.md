# emg-robot-arm

A project to build a 6-DOF robotic arm that uses a tripod-mounted
camera to visually locate a ball on a table and autonomously moves to
grab it. Earlier plans to control the gripper via EMG (muscle-signal)
gestures have been dropped in favor of this camera-only approach.

```
emg-robot-arm/
├── README.md
├── requirements.txt
├── arm/           # PlatformIO firmware for the arm's stepper motors
├── CAD/           # CAD/STL files for the arm and gripper
├── vision/        # camera calibration, ball detection, hand tracking
├── integration/   # wires vision output to arm serial commands
├── docs/          # gantt chart / deadlines
└── datasheets/    # reference PDFs
```