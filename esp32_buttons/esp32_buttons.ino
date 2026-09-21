#include <Adafruit_NeoPixel.h>
#include <string.h>

// ESP32-C3 SuperMini wiring from the existing installation.
#define NEO_LEFT_PIN   0
#define NEO_RIGHT_PIN  3
#define BTN_LEFT_PIN   1
#define BTN_RIGHT_PIN 10
#define PIXELS_PER_SIDE 2

Adafruit_NeoPixel neoLeft(PIXELS_PER_SIDE, NEO_LEFT_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel neoRight(PIXELS_PER_SIDE, NEO_RIGHT_PIN, NEO_GRB + NEO_KHZ800);

const unsigned long DEBOUNCE_MS = 50;
const unsigned long STARTUP_GUARD_MS = 700;
const unsigned long BLINK_MS = 500;
const unsigned long ACK_TIMEOUT_MS = 3000;
const unsigned long INSPECTION_TIMEOUT_MS = 60000;
const unsigned long RESULT_MS = 2500;

enum Phase { IDLE, WAITING, RESULT };

struct Side {
  uint8_t buttonPin;
  Adafruit_NeoPixel *pixels;
  const char *request;
  char letter;
  Phase phase;
  bool lastRaw;
  bool stable;
  bool armed;
  bool acknowledged;
  bool blinkOn;
  unsigned long rawChangedAt;
  unsigned long phaseStartedAt;
  unsigned long blinkChangedAt;
};

Side leftSide = {BTN_LEFT_PIN, &neoLeft, "LeftCheck", 'L', IDLE,
                 HIGH, HIGH, false, false, false, 0, 0, 0};
Side rightSide = {BTN_RIGHT_PIN, &neoRight, "RightCheck", 'R', IDLE,
                  HIGH, HIGH, false, false, false, 0, 0, 0};

char commandBuffer[32];
size_t commandLength = 0;
bool commandOverflow = false;
unsigned long startedAt = 0;

void setPixels(Side &side, uint8_t red, uint8_t green, uint8_t blue) {
  uint32_t color = side.pixels->Color(red, green, blue);
  for (uint16_t index = 0; index < side.pixels->numPixels(); ++index) {
    side.pixels->setPixelColor(index, color);
  }
  side.pixels->show();
}

void showResult(Side &side, uint8_t red, uint8_t green, uint8_t blue) {
  side.phase = RESULT;
  side.phaseStartedAt = millis();
  setPixels(side, red, green, blue);
}

void startRequest(Side &side) {
  Serial.println(side.request);
  side.phase = WAITING;
  side.phaseStartedAt = millis();
  side.blinkChangedAt = side.phaseStartedAt;
  side.acknowledged = false;
  side.blinkOn = true;
  setPixels(side, 255, 80, 0);  // Orange: request sent, waiting.
}

void updateButton(Side &side, unsigned long now) {
  bool raw = digitalRead(side.buttonPin);
  if (raw != side.lastRaw) {
    side.lastRaw = raw;
    side.rawChangedAt = now;
  }
  if (raw != side.stable && now - side.rawChangedAt >= DEBOUNCE_MS) {
    side.stable = raw;
    if (raw == HIGH) {
      side.armed = now - startedAt >= STARTUP_GUARD_MS;
    } else if (side.armed && side.phase == IDLE &&
               now - startedAt >= STARTUP_GUARD_MS) {
      side.armed = false;
      startRequest(side);
    } else {
      side.armed = false;
    }
  }
  if (side.stable == HIGH && !side.armed &&
      now - startedAt >= STARTUP_GUARD_MS) {
    side.armed = true;
  }
}

void updateLight(Side &side, unsigned long now) {
  if (side.phase == WAITING) {
    unsigned long limit = side.acknowledged
        ? INSPECTION_TIMEOUT_MS : ACK_TIMEOUT_MS;
    if (now - side.phaseStartedAt >= limit) {
      showResult(side, 255, 0, 0);  // Red: no response or operation timeout.
      return;
    }
    if (now - side.blinkChangedAt >= BLINK_MS) {
      side.blinkChangedAt = now;
      side.blinkOn = !side.blinkOn;
      if (side.blinkOn) setPixels(side, 255, 80, 0);
      else setPixels(side, 0, 0, 0);
    }
  } else if (side.phase == RESULT && now - side.phaseStartedAt >= RESULT_MS) {
    side.phase = IDLE;
    setPixels(side, 0, 0, 0);
  }
}

void handleCommand(const char *command) {
  if (command[0] != 'L' && command[0] != 'R') return;
  Side &side = command[0] == 'L' ? leftSide : rightSide;
  if (side.phase != WAITING || command[1] != '_') return;

  const char *action = command + 2;
  if (strcmp(action, "ACK") == 0) {
    side.acknowledged = true;
    side.phaseStartedAt = millis();
  } else if (strcmp(action, "DONE") == 0) {
    showResult(side, 0, 0, 255);  // Blue: completed; no PASS/FAIL decision.
  } else if (strcmp(action, "PASS") == 0) {
    showResult(side, 0, 255, 0);  // Reserved for a real PASS result.
  } else if (strcmp(action, "BUSY") == 0) {
    showResult(side, 150, 0, 150);  // Purple: another inspection is active.
  } else if (strcmp(action, "FAIL") == 0 ||
             strcmp(action, "ERROR") == 0 ||
             strcmp(action, "NOT_READY") == 0) {
    showResult(side, 255, 0, 0);
  }
}

void readCommands() {
  while (Serial.available() > 0) {
    char character = static_cast<char>(Serial.read());
    if (character == '\r') continue;
    if (character == '\n') {
      if (!commandOverflow && commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        handleCommand(commandBuffer);
      }
      commandLength = 0;
      commandOverflow = false;
    } else if (!commandOverflow) {
      if (commandLength < sizeof(commandBuffer) - 1) {
        commandBuffer[commandLength++] = character;
      } else {
        commandOverflow = true;
      }
    }
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(BTN_LEFT_PIN, INPUT_PULLUP);   // Pressed connects to GND.
  pinMode(BTN_RIGHT_PIN, INPUT_PULLUP);

  neoLeft.begin();
  neoRight.begin();
  neoLeft.setBrightness(120);
  neoRight.setBrightness(120);
  setPixels(leftSide, 0, 0, 0);
  setPixels(rightSide, 0, 0, 0);

  startedAt = millis();
  leftSide.lastRaw = leftSide.stable = digitalRead(BTN_LEFT_PIN);
  rightSide.lastRaw = rightSide.stable = digitalRead(BTN_RIGHT_PIN);
  leftSide.rawChangedAt = rightSide.rawChangedAt = startedAt;
}

void loop() {
  unsigned long now = millis();
  updateButton(leftSide, now);
  updateButton(rightSide, now);
  readCommands();
  updateLight(leftSide, now);
  updateLight(rightSide, now);
}
