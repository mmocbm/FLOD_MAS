class Checkpoint:
    """Tracks hand appear -> leave, confirmed by a quiet period.

    States:
      IDLE       nothing seen yet, waiting for a hand
      ARMED      a hand is in view (the checkpoint is primed)
      WAITING    the hand left; confirming that it stays gone for the hold time
      TRIGGERED  the checkpoint fired and the flash is showing
    """

    IDLE = "IDLE"
    ARMED = "ARMED"
    WAITING = "WAITING"
    TRIGGERED = "TRIGGERED"

    def __init__(self, confirm_seconds: float = 1.0, flash_seconds: float = 1.5,
                 min_present_seconds: float = 0.0):
        self.confirm_seconds = max(0.05, confirm_seconds)
        self.flash_seconds = max(0.1, flash_seconds)
        self.min_present_seconds = max(0.0, min_present_seconds)

        self.state = self.IDLE
        self.triggers = 0
        self.cancelled = 0
        self.progress = 0.0        # 0..1 through the confirmation wait
        self.remaining = 0.0       # seconds left in the confirmation wait
        self.remaining_ratio = 0.0 # 1..0 through the trigger flash
        self._t_present = 0.0
        self._t_absent = 0.0
        self._t_trigger = 0.0

    def reset_counts(self):
        self.triggers = 0
        self.cancelled = 0

    def update(self, present: bool, now: float) -> bool:
        """Advance one step. Returns True only on the frame the checkpoint fires."""
        fired = False

        if self.state == self.IDLE:
            if present:
                self.state = self.ARMED
                self._t_present = now

        elif self.state == self.ARMED:
            if not present:
                if now - self._t_present >= self.min_present_seconds:
                    self.state = self.WAITING
                    self._t_absent = now
                else:
                    self.state = self.IDLE  # barely glimpsed; treat as noise

        elif self.state == self.WAITING:
            if present:
                # The hand came back inside the wait window - the disappearance
                # was noise. Cancel and restart the cycle from ARMED.
                self.state = self.ARMED
                self._t_present = now
                self.cancelled += 1
            elif now - self._t_absent >= self.confirm_seconds:
                self.state = self.TRIGGERED
                self._t_trigger = now
                self.triggers += 1
                fired = True

        elif self.state == self.TRIGGERED:
            if now - self._t_trigger >= self.flash_seconds:
                if present:
                    self.state = self.ARMED
                    self._t_present = now
                else:
                    self.state = self.IDLE

        # live values for the on-screen bar, computed from the clock only
        if self.state == self.WAITING:
            elapsed = now - self._t_absent
            self.progress = min(1.0, elapsed / self.confirm_seconds)
            self.remaining = max(0.0, self.confirm_seconds - elapsed)
        else:
            self.progress = 0.0
            self.remaining = 0.0

        if self.state == self.TRIGGERED:
            left = max(0.0, self.flash_seconds - (now - self._t_trigger))
            self.remaining_ratio = left / self.flash_seconds
        else:
            self.remaining_ratio = 0.0

        return fired


# --------------------------------------------------------------------------
# presence debounce (same idea as hand_tracker's inline version)
# --------------------------------------------------------------------------
class Debouncer:
    """Holds a boolean steady until `hold` fresh samples agree on the new value."""

    def __init__(self, hold: int):
        self.hold = max(1, hold)
        self.value = False
        self._pending = False
        self._count = 0

    def update(self, raw: bool) -> bool:
        if raw == self.value:
            self._count = 0
            self._pending = self.value
        else:
            if raw == self._pending:
                self._count += 1
            else:
                self._pending = raw
                self._count = 1
            if self._count >= self.hold:
                self.value = self._pending
                self._count = 0
        return self.value



