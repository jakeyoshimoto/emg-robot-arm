# emg-robot-arm

A 30-day project to build an EMG-controlled robotic hand mounted on a
7-DOF robotic arm, the hand reads muscle signals to control grip, and
the arm autonomously locates and moves objects.

```
emg-robot-arm/
├── README.md
├── requirements.txt
├── hand/          # EMG acquisition, gesture classifier, servo control
├── arm/           # PlatformIO firmware for the arm's stepper motors
├── vision/        # camera calibration, object detection, path planning
├── integration/   # unified control loop, end-to-end tests
├── hardware/      # CAD files, wiring diagrams, bom.csv
├── docs/          # gantt chart / deadlines
└── scripts/       # setup/utility scripts
```