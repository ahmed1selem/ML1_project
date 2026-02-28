# Hand Gesture Classification

This project trains a hand-gesture classifier from hand-landmark data and runs live webcam inference.

## Project Files

- `hand_gesture_classification.ipynb`: model training and evaluation notebook.
- `hand_landmarks_data.csv`: labeled landmark dataset used for training.
- `best_model.pkl`: trained classifier artifact.
- `scaler.pkl`: scaler used for x/y feature standardization.
- `label_encoder.pkl`: label encoder for class mapping.
- `helper.py`: shared preprocessing and model-loading utilities.
- `live_inference.py`: webcam inference script that predicts and overlays gesture labels.

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn xgboost opencv-python mediapipe protobuf
```

## Training

1. Open `hand_gesture_classification.ipynb`.
2. Run all cells.
3. Confirm these files are generated/updated:
   - `best_model.pkl`
   - `scaler.pkl`
   - `label_encoder.pkl`

## Live Inference

Run with defaults:

```bash
python live_inference.py
```

Useful options:

```bash
python live_inference.py --help
python live_inference.py --output output.mp4 --camera 0 --smooth-window 3
python live_inference.py --no-preview --max-seconds 30
```
