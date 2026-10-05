"""Latest-frame hand monitor with fabric gating and stale-result protection.

    submit(raw_frame, sequence, undistorter, profile) is camera-independent.
    poll() returns events; the caller alone owns Tk widgets.
"""
from concurrent.futures import Future
from datetime import datetime
import queue
import threading
import time
from .checkpoint import Checkpoint, Debouncer
from .models import HandPresence, FabricGate


class AutoController(threading.Thread):
    def __init__(self, config, pipeline_factory, hand_factory=HandPresence, fabric_factory=FabricGate):
        super().__init__(name='automatic-hand-monitor', daemon=True)
        self.config = config
        self.settings = config['auto_trigger']
        self.pipeline_factory = pipeline_factory
        self.hand_factory, self.fabric_factory = hand_factory, fabric_factory
        self.events = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopped = threading.Event()
        self.active = True
        self.generation = 0
        self.epoch = 0
        self.latest = None
        self.future = None
        self.pending_job = None
        self.pipeline = None
        self.gate_busy = False
        self.job_generation = -1
        self.last_status = None

    def submit(self, frame, sequence, undistorter, profile):
        with self.lock:
            if self.active:
                self.latest = (frame, sequence, undistorter, profile, self.epoch)
        self.wake.set()

    def set_active(self, active):
        with self.lock:
            if self.active != active:
                self.active = active
                self.generation += 1
                self.epoch += 1
                self.latest = None
        self.wake.set()

    def stop(self):
        self.stopped.set()
        self.wake.set()

    def poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            yield event

    def _status(self, text):
        if text != self.last_status:
            self.last_status = text
            self.events.put(('status', self.generation, text))

    def _reset_checkpoint(self):
        self.checkpoint = Checkpoint(self.settings['hand_absence_seconds'], 0.1,
                                     self.settings['min_hand_present_seconds'])
        self.presence = Debouncer(self.settings['debounce_frames'])
        self.was_present = False

    def _complete(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
            if self.job_generation == self.generation and self.active:
                self.events.put(('result', self.generation, result))
            else:
                self._status('Previous capture saved; checking the latest fabric arrangement')
        except Exception as error:
            # Errors are always surfaced, even if that capture is now obsolete.
            self.events.put(('error', self.generation, str(error)))
        if self.pending_job is not None:
            job = self.pending_job
            if (self.active and not self.stopped.is_set()
                    and job['generation'] == self.generation and job['epoch'] == self.epoch):
                self.pending_job = None
                self._start_job(job)
            else:
                self._discard_pending('cancelled')

    def _discard_pending(self, status):
        job, self.pending_job = self.pending_job, None
        if job is not None and job['path'] is not None:
            try:
                self.pipeline.store.finish(job['path'], {'status': status, 'capture': job['metadata']})
            except OSError as error:
                self.events.put(('error', self.generation, str(error)))

    def _start_job(self, job):
        self._status('Segmenting and measuring glue strips…')
        self.future = Future()
        self.job_generation = job['generation']

        def process(future):
            try:
                future.set_result(self.pipeline.run(
                    job['original'], job['corrected'], job['metadata'],
                    job['profile']['width'], job['profile']['tolerance'], job['path']))
            except Exception as error:
                future.set_exception(error)
            self.wake.set()
        threading.Thread(target=process, args=(self.future,),
                         name='automatic-inspection', daemon=True).start()

    def run(self):
        hand = None
        try:
            self._status('Loading hand and fabric models…')
            hand = self.hand_factory(self.settings)
            if self.stopped.is_set():
                return
            gate = self.fabric_factory(self.settings)
            if self.stopped.is_set():
                return
            self.pipeline = pipeline = self.pipeline_factory()
            self._reset_checkpoint()
            last_seq, epoch = None, self.epoch
            self._status('Automatic inspection ready — waiting for a hand')
            while not self.stopped.is_set():
                self._complete()
                self.wake.wait(0.05)
                self.wake.clear()
                with self.lock:
                    item, self.latest = self.latest, None
                    active, current_epoch = self.active, self.epoch
                if epoch != current_epoch:
                    self._discard_pending('cancelled')
                    epoch = current_epoch
                    last_seq = None
                    self._reset_checkpoint()
                if not active or item is None:
                    continue
                raw, sequence, undistorter, profile, item_epoch = item
                if item_epoch != epoch or sequence == last_seq:
                    continue
                last_seq = sequence
                capture_path = None
                try:
                    raw_present = hand.present(raw)
                    if item_epoch != self.epoch or not self.active:
                        continue
                    # Invalidate old work immediately on a new visible hand,
                    # before debounce; noise must never publish an obsolete result.
                    if raw_present and not self.was_present:
                        self.generation += 1
                        self._discard_pending('superseded')
                    self.was_present = raw_present
                    present = self.presence.update(raw_present)
                    fired = self.checkpoint.update(present, time.monotonic())
                    if not fired:
                        if present:
                            self._status('Hand present — waiting for hands to leave')
                        elif self.checkpoint.state == Checkpoint.WAITING:
                            self._status('Hands clear — confirming absence…')
                        continue
                    self._status('Checking fabric…')
                    self.gate_busy = True
                    trigger_generation = self.generation
                    # The hand result, fabric verdict, saved raw image and
                    # undistortion all refer to this exact immutable snapshot.
                    original = raw.copy()
                    metadata = {'triggered_at': datetime.now().astimezone().isoformat(),
                                'frame_sequence': sequence, 'fabric': None,
                                'size': profile['size'], 'target_width_mm': profile['width'],
                                'tolerance_mm': profile['tolerance']}
                    corrected = None
                    if self.config['capture_storage']['save_rejected_triggers']:
                        corrected = undistorter.undistort(original) if undistorter else None
                        capture_path = pipeline.store.begin(original, corrected, metadata)
                    verdict = gate.classify(original)
                    metadata['fabric'] = verdict
                    if capture_path is not None:
                        pipeline.store.update_metadata(capture_path, metadata)
                    if item_epoch != self.epoch or not self.active:
                        if capture_path is not None:
                            pipeline.store.finish(capture_path, {'status': 'cancelled', 'capture': metadata})
                        continue
                    if verdict['empty']:
                        self.generation += 1
                        self._discard_pending('cancelled')
                        self.events.put(('clear', self.generation, 'No fabric — ready for the next hand cycle'))
                    if not verdict['accepted']:
                        if capture_path is not None:
                            pipeline.store.finish(capture_path, {'status': 'rejected', 'capture': metadata})
                        self._status(f"Fabric check: {verdict['label']} ({verdict['confidence']:.0%})")
                        continue
                    if undistorter is None:
                        raise RuntimeError('Camera calibration is required before automatic measurement')
                    if corrected is None:
                        corrected = undistorter.undistort(original)
                    job = {'original': original, 'corrected': corrected, 'metadata': metadata,
                           'profile': profile, 'path': capture_path, 'generation': trigger_generation,
                           'epoch': item_epoch}
                    if self.future is not None:
                        self._discard_pending('superseded')
                        self.pending_job = job
                        self._status('New fabric arrangement captured — processing it next')
                    else:
                        self._start_job(job)
                except Exception as error:
                    if capture_path is not None:
                        try:
                            pipeline.store.finish(capture_path, {'status': 'error', 'error': str(error)})
                        except OSError:
                            pass
                    self._reset_checkpoint()
                    self.events.put(('error', self.generation, str(error)))
                finally:
                    self.gate_busy = False
        except Exception as error:
            self.events.put(('error', self.generation, f'Automatic inspection unavailable: {error}'))
        finally:
            self._discard_pending('cancelled')
            if hand is not None:
                hand.close()
