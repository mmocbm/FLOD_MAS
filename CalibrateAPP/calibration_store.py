"""Replace a calibration's related JSON files, rolling back failed saves."""
import json
import os
from pathlib import Path
import tempfile


def replace_calibration_files(updates):
    """Stage writes first; None removes obsolete data. Restore on write failure.

    Callers pause inspection while committing. This provides rollback for ordinary
    I/O failures; separate filesystem files cannot be replaced atomically as a group.
    """
    originals = {}
    staged = {}
    changed = []
    try:
        for filename, body in updates.items():
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            originals[path] = path.read_bytes() if path.exists() else None
            if body is None:
                staged[path] = None
                continue
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=path.parent, suffix='.tmp', delete=False) as stream:
                staged[path] = Path(stream.name)
                json.dump(body, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        for path, temporary in staged.items():
            if temporary is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(temporary, path)
            changed.append(path)
    except Exception:
        for path in reversed(changed):
            original = originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
                    rollback = Path(stream.name)
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.replace(rollback, path)
                finally:
                    rollback.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            if temporary is not None:
                temporary.unlink(missing_ok=True)
