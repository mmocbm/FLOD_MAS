# ESP32-C3 SuperMini inspection buttons

Upload `esp32_buttons.ino` with the ESP32-C3 board package and the
`Adafruit_NeoPixel` Arduino library. The existing pin assignments are preserved:

| Side | Button GPIO | NeoPixel data GPIO |
| --- | ---: | ---: |
| Left | 1 | 0 |
| Right | 10 | 3 |

Buttons connect each GPIO to GND when pressed. The sketch enables the internal
pull-ups. `PIXELS_PER_SIDE` is 2 to match the previous sketch; change it to 1
if each button actually contains one LED.

The ESP32 sends `LeftCheck` or `RightCheck` at 9600 baud after a stable press.
Python replies with line-terminated commands for the same side:

| Reply | Light | Meaning |
| --- | --- | --- |
| `L_ACK` / `R_ACK` | Blinking orange | Inspection accepted and running |
| `L_DONE` / `R_DONE` | Blue | Inspection finished; no PASS/FAIL judgement |
| `L_NOT_READY` / `R_NOT_READY` | Red | Camera or calibration not ready |
| `L_BUSY` / `R_BUSY` | Purple | Another inspection is running |
| `L_ERROR` / `R_ERROR` | Red | Inspection failed |

`L_PASS`, `R_PASS`, `L_FAIL`, and `R_FAIL` are reserved in the sketch for a
future real quality decision. The current Python inspection does not calculate
a reliable product PASS/FAIL result, so it deliberately does not send those.
The old `L2`/`R2` messages meant FAIL in the previous sketch and were sent
prematurely as acknowledgements; they are no longer used.

The sketch requires a stable button press for 50 ms and a release before
another request. It does not block with `delay()` or `readStringUntil()`.
If false presses persist, watch the terminal for unsolicited `LeftCheck` or
`RightCheck` and inspect button wiring, common ground, and LED power. For long
button wires, stronger external pull-ups and cable routing away from LED power
wires may be needed.
