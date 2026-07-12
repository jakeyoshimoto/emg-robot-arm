// Stepper + servo bring-up firmware for the 7-DOF arm.
// Drives NEMA17/A4988 stepper axes and one gripper servo over serial.
// No limit switches or homing yet.
//
// Serial commands, 115200 baud:
//   <axis> <steps>    move one axis a relative number of steps, e.g. 0 200
//   speed <axis> <v>  set that axis's max speed in steps/sec
//   stop              stop all axes immediately
//   servo <angle>     move the gripper servo to an absolute angle, 0-180
//   ?                 show this again

#include <Arduino.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>

constexpr int NUM_AXES = 7;

// Servo configuration (only one for now)
constexpr int SERVO_PIN = 14;
constexpr int SERVO_MIN_ANGLE = 0;
constexpr int SERVO_MAX_ANGLE = 180;

Servo gripperServo;

struct AxisPins {
  uint8_t step;
  uint8_t dir;
};

// Stepper configuration
// One STEP/DIR pair per joint
const AxisPins AXIS_PINS[NUM_AXES] = {
  {1, 2},
  {21, 47},
  {48, 40},
  {39, 38},
  {37, 36},
  {35, 41},
  {42, 45},
};

constexpr float DEFAULT_MAX_SPEED = 500.0;     // steps/sec, conservative for first bring-up
constexpr float DEFAULT_ACCELERATION = 250.0;  // steps/sec^2

AccelStepper axes[NUM_AXES] = {
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[0].step, AXIS_PINS[0].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[1].step, AXIS_PINS[1].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[2].step, AXIS_PINS[2].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[3].step, AXIS_PINS[3].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[4].step, AXIS_PINS[4].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[5].step, AXIS_PINS[5].dir),
  AccelStepper(AccelStepper::DRIVER, AXIS_PINS[6].step, AXIS_PINS[6].dir),
};

void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  <axis> <steps>    move one axis a relative number of steps, e.g. 0 200");
  Serial.println("  speed <axis> <v>  set that axis's max speed in steps/sec");
  Serial.println("  stop              stop all axes immediately");
  Serial.println("  servo <angle>     move the gripper servo to an absolute angle, 0-180");
  Serial.println("  ?                 show this again");
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }

  if (line == "?") {
    printHelp();
    return;
  }

  if (line == "stop") {
    for (int i = 0; i < NUM_AXES; i++) {
      axes[i].stop();
    }
    Serial.println("Stopped all axes.");
    return;
  }

  if (line.startsWith("speed ")) {
    int axis, value;
    if (sscanf(line.c_str(), "speed %d %d", &axis, &value) == 2 &&
        axis >= 0 && axis < NUM_AXES) {
      axes[axis].setMaxSpeed((float)value);
      Serial.printf("Axis %d max speed set to %d steps/sec.\n", axis, value);
    } else {
      Serial.println("Usage: speed <axis> <steps/sec>");
    }
    return;
  }

  if (line.startsWith("servo ")) {
    int angle;
    if (sscanf(line.c_str(), "servo %d", &angle) == 1 &&
        angle >= SERVO_MIN_ANGLE && angle <= SERVO_MAX_ANGLE) {
      gripperServo.write(angle);
      Serial.printf("Servo moving to %d degrees.\n", angle);
    } else {
      Serial.printf("Usage: servo <angle %d-%d>\n", SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
    }
    return;
  }

  int axis, steps;
  if (sscanf(line.c_str(), "%d %d", &axis, &steps) == 2) {
    if (axis >= 0 && axis < NUM_AXES) {
      axes[axis].move(steps);
      Serial.printf("Axis %d moving %d steps.\n", axis, steps);
    } else {
      Serial.printf("Axis must be 0-%d.\n", NUM_AXES - 1);
    }
    return;
  }

  Serial.println("Unrecognized command, type ? for help.");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    // wait for USB serial to come up on boards where this matters
  }

  for (int i = 0; i < NUM_AXES; i++) {
    axes[i].setMaxSpeed(DEFAULT_MAX_SPEED);
    axes[i].setAcceleration(DEFAULT_ACCELERATION);
  }

  gripperServo.setPeriodHertz(50);
  gripperServo.attach(SERVO_PIN, 500, 2400);

  Serial.println("Stepper bring-up firmware ready.");
  printHelp();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }

  for (int i = 0; i < NUM_AXES; i++) {
    axes[i].run();
  }
}
