// Stepper motor bring-up firmware for the 7-DOF arm.
//
// Drives up to NUM_AXES NEMA17 steppers through A4988 drivers via
// STEP/DIR pins. No limit switches or homing yet, that's follow-up
// work. This is just for confirming each axis actually turns once
// motors, drivers, and power are wired up, driven over a serial
// console so there's no dependency on the rest of the arm code.
//
// Wiring, per axis:
//   A4988 STEP -> the STEP pin below
//   A4988 DIR  -> the DIR pin below
//   A4988 EN   -> tied to GND directly, driver always enabled. Fine
//                 for bring-up. Revisit with a shared enable pin once
//                 the arm needs to sit idle without holding torque.
//   A4988 VMOT/GND -> motor power supply, set to the stepper's rated
//                 voltage, check its datasheet, don't just run it at
//                 the supply's max.
//   A4988 VDD/GND  -> 3.3V logic + GND from the ESP32 board.
//
// Pin choice note: GPIO4-18 on this board are shared with the onboard
// camera header. Since this board isn't running a camera, vision runs
// separately on a laptop over USB, those pins are free to use here.
// If that ever changes, these assignments need to change too.
//
// Serial commands, 115200 baud:
//   <axis> <steps>    move one axis a relative number of steps, e.g. 0 200
//   speed <axis> <v>  set that axis's max speed in steps/sec
//   stop              stop all axes immediately
//   ?                 show this again

#include <Arduino.h>
#include <AccelStepper.h>

constexpr int NUM_AXES = 7;

struct AxisPins {
  uint8_t step;
  uint8_t dir;
};

// One STEP/DIR pair per joint. See wiring note above for why GPIO4-18
// are safe to use on this board.
const AxisPins AXIS_PINS[NUM_AXES] = {
  {4, 5},
  {6, 7},
  {15, 16},
  {17, 18},
  {8, 9},
  {10, 11},
  {12, 13},
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
