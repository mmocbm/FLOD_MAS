#!/usr/bin/env bash
# Start the Pipeline Lab. The SAM choices need inference-sdk, which does not
# install on Python 3.14, so an environment with Python 3.12 is preferred.
cd "$(dirname "$0")"
for python in "$HOME/flod_experiments/gpu/bin/python" .venv/bin/python python3; do
    if "$python" -c "import cv2, PIL, skimage, onnxruntime" >/dev/null 2>&1; then
        exec "$python" tools/pipeline_lab.py "$@"
    fi
done
echo "No Python environment with opencv, Pillow, scikit-image and onnxruntime was found." >&2
exit 1
