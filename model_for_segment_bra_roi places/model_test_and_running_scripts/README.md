# Crop-part (paper strip) model: running, testing, ONNX

The trained model that finds the crop part (the bare paper strip beside each glue bead, like
`WIN_20261006_21_43_37_Pro - Edited.jpg`) in a photo, with scripts to run it, test it and convert it.
Training lives in `../train_segemnt_crop_part`.

## Files

| File | What it is |
|---|---|
| `models/strip_unet_resnet34.pt` | PyTorch checkpoint (weights + training settings), 98 MB |
| `models/strip_unet_resnet34.onnx` | Same model as ONNX (opset 17), 98 MB |
| `strip_model.py` | Loading (.pt or .onnx) and inference: pre-processing, mirrored second pass, clean-up |
| `run_model.py` | Run on photos/folders: strip masks, overlays, JSON |
| `test_model.py` | Accuracy against hand masks; or compare the .onnx and .pt models |
| `convert_to_onnx.py` | Export the .pt to ONNX and check both give the same result |
| `requirements.txt` | Packages (the ONNX model needs only numpy, opencv-python, onnxruntime) |

## Run

    python run_model.py --input ../real_captures --output results/real_captures
    python run_model.py --input "../latest_captures/Machine4/captures/*.jpg" --output results/m4
    python run_model.py --model models/strip_unet_resnet34.pt --input photo.jpg --output out   # PyTorch
    python run_model.py --input photo.jpg --output out --device cpu --no-flip                  # fastest on CPU

Outputs per photo: `<stem>_strip_mask.png` (full size, 255 = strip), `<stem>_strip_overlay.jpg`
(quarter size, outline red), `<stem>_strips.json` (each separate strip region: area, bounding box,
centre; time taken). `--save-probability` adds the probability map.

Speed (RTX 4060 laptop, 4608x3456 photo, including resizing and clean-up): ONNX on GPU about
0.35 s, ONNX on CPU about 1.1 s; `--no-flip` halves the model time.

## Model

- U-Net with a ResNet-34 encoder (ImageNet pre-trained), one output: strip probability per pixel.
- Input: the photo scaled to 1152 px wide, padded to a multiple of 32, RGB normalised with the
  ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225). ONNX input `image`
  1x3xHxW float32 (any H, W that are multiples of 32); output `logits` 1x1xHxW,
  probability = sigmoid(logits). `strip_model.py` does all of this.
- Post-processing: mean of a normal and a mirrored pass, resized to the photo, threshold 0.5,
  regions under 0.2% of the image dropped, holes filled.
- Trained on 72 hand-masked, undistorted frames (Machine1 59, Machine4 13) for 120 epochs.

## Accuracy

Measured with the same training recipe on 14 frames held out from training (5 arrangements;
Machine1 dark, pink, white and Machine4), at full resolution:

| | IoU | Strips found | Outline error |
|---|---|---|---|
| Model | 0.990 (worst frame 0.987) | 48/48 | 1.3 px mean, 2.9 px p95 (about 0.17 mm on Machine1) |
| Previous rule-based areas | 0.59 | 31/48 | 48 px mean |

On a webcam photo (a camera not in the training data) the painted strip in
`WIN_20261006_21_43_37_Pro - Edited.jpg` was fully covered (IoU 0.91); the outline lies about
10 px outside the paint. `test_model.py` on frames the model was trained on scores higher than
on new frames; use new hand-masked frames for a fair test.

ONNX vs PyTorch: on CPU identical masks (IoU 1.000000, largest probability difference about
3e-6); on GPU, where the .pt model runs in half precision, mask agreement IoU 0.99997 or better
and the largest probability difference 0.006 (`test_model.py --compare-backends`).
GPU time per photo: .pt about 0.13 s, .onnx about 0.15 s (model and resizing).

## Limits

- Touching strips come out as one region (the hand masks also join them); splitting them needs
  the bead traces (one bead per strip).
- The model expects panels on the board; empty boards (ArUco markers only) are out of scope and
  give small false regions.
