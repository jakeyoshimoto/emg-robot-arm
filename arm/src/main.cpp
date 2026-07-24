// Stepper + servo bring-up firmware for the 7-DOF arm.
// Drives NEMA17/A4988 stepper axes and 5 gripper/hand servos over serial.
// Servos run through a PCA9685 I2C PWM driver board. Also reads a
// single-channel analog EMG sensor on GPIO5 for bring-up testing. No
// limit switches or homing yet.
//
// Serial commands, 115200 baud:
//   m<axis> <steps>      move one motor a relative number of steps, e.g. m0 200
//   speed m<axis> <v>    set that motor's max speed in steps/sec
//   stop                 stop all motors immediately
//   s<ch> <angle>        move one servo channel to an absolute angle, 0-180
//   emg                  toggle streaming raw EMG readings ("emg <0-4095>")
//   ?                    show this again

#include <Arduino.h>
#include <AccelStepper.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

constexpr int NUM_AXES = 6;

struct AxisPins {
  uint8_t step;
  uint8_t dir;
};

// Stepper configuration - one STEP/DIR pair per joint
const AxisPins AXIS_PINS[NUM_AXES] = {
  {1, 2},
  {42, 41},
  {40, 39},
  {38, 37},
  {36, 35},
  {48, 47},
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
};

// Servo configuration - PCA9685 over I2C, default address 0x40
constexpr int I2C_SDA_PIN = 21;
constexpr int I2C_SCL_PIN = 45;  // strapping pin, only sampled at boot
constexpr int NUM_SERVOS = 6;
constexpr int SERVO_MIN_ANGLE = 0;
constexpr int SERVO_MAX_ANGLE = 180;
constexpr int SERVO_MIN_PULSE_US = 500;
constexpr int SERVO_MAX_PULSE_US = 2400;
constexpr float PWM_FREQ_HZ = 50.0;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// EMG sensor (Advancer Technologies Muscle Sensor v3 or clone) - needs
// a dual +Vs/-Vs supply (e.g. two 9V batteries, center tap = GND), and
// its SIG output swings 0-Vs, so it goes through an external resistor
// divider (SIG -> 20k -> node -> 10k -> GND, node -> GPIO5) to land
// within the ADC's 0-3.3V range before reaching this pin. GPIO5 is
// ADC1_CH4, free and not shared with any stepper/I2C pin above.
constexpr int EMG_PIN = 5;
constexpr unsigned long EMG_SAMPLE_INTERVAL_MS = 20;  // ~50 Hz

bool emgStreaming = false;
unsigned long lastEmgSampleMs = 0;

uint16_t angleToTicks(int angle) {
  long pulseUs = map(angle, 0, 180, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);
  return (uint16_t)((pulseUs * 4096L) / (long)(1000000.0 / PWM_FREQ_HZ));
}

void printHelp() {
  Serial.println();
  Serial.println("Commands:");
  Serial.println("  m<axis> <steps>    move one motor a relative number of steps, e.g. m0 200");
  Serial.println("  speed m<axis> <v>  set that motor's max speed in steps/sec");
  Serial.println("  stop               stop all motors immediately");
  Serial.println("  s<ch> <angle>      move one servo channel to an absolute angle, 0-180");
  Serial.println("  emg                toggle streaming raw EMG readings (\"emg <0-4095>\")");
  Serial.println("  ?                  show this again");
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
    Serial.println("Stopped all motors.");
    return;
  }

  if (line.startsWith("speed ")) {
    int axis, value;
    if (sscanf(line.c_str(), "speed m%d %d", &axis, &value) == 2 &&
        axis >= 0 && axis < NUM_AXES) {
      axes[axis].setMaxSpeed((float)value);
      Serial.printf("Motor %d max speed set to %d steps/sec.\n", axis, value);
    } else {
      Serial.println("Usage: speed m<axis> <steps/sec>");
    }
    return;
  }

  if (line.startsWith("m")) {
    int axis, steps;
    if (sscanf(line.c_str(), "m%d %d", &axis, &steps) == 2 &&
        axis >= 0 && axis < NUM_AXES) {
      axes[axis].move(steps);
      Serial.printf("Motor %d moving %d steps.\n", axis, steps);
    } else {
      Serial.printf("Usage: m<axis 0-%d> <steps>\n", NUM_AXES - 1);
    }
    return;
  }

  if (line == "emg") {
    emgStreaming = !emgStreaming;
    Serial.println(emgStreaming ? "EMG streaming on." : "EMG streaming off.");
    return;
  }

  if (line.startsWith("s")) {
    int channel, angle;
    if (sscanf(line.c_str(), "s%d %d", &channel, &angle) == 2 &&
        channel >= 0 && channel < NUM_SERVOS &&
        angle >= SERVO_MIN_ANGLE && angle <= SERVO_MAX_ANGLE) {
      pwm.setPWM(channel, 0, angleToTicks(angle));
      Serial.printf("Servo %d moving to %d degrees.\n", channel, angle);
    } else {
      Serial.printf("Usage: s<channel 0-%d> <angle %d-%d>\n",
                     NUM_SERVOS - 1, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
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

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ_HZ);

  pinMode(EMG_PIN, INPUT);
  analogReadResolution(12);                     // 0-4095 across 0-3.3V
  analogSetPinAttenuation(EMG_PIN, ADC_11db);    // full 0-3.3V input range

  Serial.println("Stepper + servo bring-up firmware ready.");
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

  if (emgStreaming && millis() - lastEmgSampleMs >= EMG_SAMPLE_INTERVAL_MS) {
    lastEmgSampleMs = millis();
    Serial.printf("emg %d\n", analogRead(EMG_PIN));
  }
}
