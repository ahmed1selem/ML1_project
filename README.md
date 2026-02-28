# Hand Gesture Classification Project

This project trains a hand-gesture classifier from landmark data and runs live webcam inference to generate a labeled output video.

## Project Structure

- `hand_gesture_classification.ipynb`: training and model selection notebook.
- `live_inference.py`: live webcam inference script (writes annotated video).
- `helper.py`: model/camera/preprocessing helper utilities.
- `hand_landmarks_data.csv`: training dataset.
- `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`: saved training artifacts.
- `output.mp4`: sample generated output video.

## Requirements

Use Python 3.9+ and install:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost opencv-python mediapipe protobuf
```

## Training

1. Open and run `hand_gesture_classification.ipynb`.
2. Confirm these files are generated/updated:
   - `best_model.pkl`
   - `scaler.pkl`
   - `label_encoder.pkl`

## Live Inference

Run:

```bash
python live_inference.py --camera 0 --output live_frame_by_frame_output.mp4
```

Useful options:

```bash
python live_inference.py --help
```

Examples:

```bash
python live_inference.py --max-seconds 10 --no-preview
python live_inference.py --camera 1 --smooth-window 5 --output output.mp4
```

## Important Note

`live_inference.py` currently imports from `camera_inference_helper`, while this repo currently contains `helper.py`.

If you run inference as-is, make sure one of these is true:

1. You have a file named `camera_inference_helper.py` with the same helper functions.
2. You update the import in `live_inference.py` to use `helper`.

## Troubleshooting

- `Missing required file: best_model.pkl`:
  Run the notebook first to generate model artifacts.
- `Could not open camera.`:
  Check camera index (`--camera 0`, `--camera 1`, ...), and close other apps using the camera.
- Poor prediction stability:
  Increase `--smooth-window` to reduce per-frame jitter.
