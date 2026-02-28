import pickle
from pathlib import Path

import cv2
import numpy as np

# Mediapipe/protobuf compatibility for some local environments.
from google.protobuf import message_factory, symbol_database

if not hasattr(message_factory.MessageFactory, "GetPrototype"):
    def _factory_get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)

    message_factory.MessageFactory.GetPrototype = _factory_get_prototype

if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
    def _db_get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)

    symbol_database.SymbolDatabase.GetPrototype = _db_get_prototype


LANDMARK_COUNT = 21
MIDDLE_FINGERTIP_IDX = 12  # 0-based index
TIPS = [4, 8, 12, 16, 20]
TIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
MCPS = [1, 5, 9, 13, 17]


def load_pipeline(model_path="best_model.pkl", scaler_path="scaler.pkl", label_encoder_path="label_encoder.pkl"):
    paths = [Path(model_path), Path(scaler_path), Path(label_encoder_path)]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(paths[0], "rb") as f:
        model = pickle.load(f)
    with open(paths[1], "rb") as f:
        scaler = pickle.load(f)
    with open(paths[2], "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


def base_feature_columns():
    # Training order: x1,y1,z1, x2,y2,z2, ..., x21,y21,z21
    return [f"{axis}{i}" for i in range(1, LANDMARK_COUNT + 1) for axis in ("x", "y", "z")]


def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def normalize_landmarks(hand_landmarks, width, height):
    """
    Build 63 values and apply training normalization:
    1) wrist-centering
    2) x/y scaling by wrist->middle fingertip distance
    3) z remains recentered only
    """
    coords = np.array(
        [[lm.x * width, lm.y * height, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = coords[0].copy()
    coords -= wrist

    scale = np.sqrt(coords[MIDDLE_FINGERTIP_IDX, 0] ** 2 + coords[MIDDLE_FINGERTIP_IDX, 1] ** 2)
    if scale < 1e-6:
        scale = 1e-6
    coords[:, 0] /= scale
    coords[:, 1] /= scale

    return coords.reshape(-1)


def add_engineered_features(sample_63, feature_cols):
    arr = sample_63.reshape(1, -1)

    def get_lm(lm_idx):
        cx = feature_cols.index(f"x{lm_idx + 1}")
        cy = feature_cols.index(f"y{lm_idx + 1}")
        cz = feature_cols.index(f"z{lm_idx + 1}")
        return arr[:, cx], arr[:, cy], arr[:, cz]

    def dist_3d(a, b):
        ax, ay, az = get_lm(a)
        bx, by, bz = get_lm(b)
        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    feats = {}

    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            feats[f"dist_{TIP_NAMES[i]}_{TIP_NAMES[j]}"] = dist_3d(TIPS[i], TIPS[j])

    for tip, name in zip(TIPS, TIP_NAMES):
        feats[f"len_{name}"] = dist_3d(0, tip)

    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            a = feats[f"len_{TIP_NAMES[i]}"]
            b = feats[f"len_{TIP_NAMES[j]}"]
            feats[f"ratio_{TIP_NAMES[i]}_{TIP_NAMES[j]}"] = a / (b + 1e-8)

    for tip, mcp, name in zip(TIPS, MCPS, TIP_NAMES):
        tip_mcp = dist_3d(tip, mcp)
        feats[f"curl_{name}"] = tip_mcp / (feats[f"len_{name}"] + 1e-8)

    engineered = np.column_stack(list(feats.values()))
    return np.hstack([arr, engineered]).reshape(-1)


def build_model_input(sample_63, scaler, model, feature_cols):
    # Scaler was fit on x/y only
    x_idx = [feature_cols.index(f"x{i}") for i in range(1, LANDMARK_COUNT + 1)]
    y_idx = [feature_cols.index(f"y{i}") for i in range(1, LANDMARK_COUNT + 1)]
    xy_idx = sorted(x_idx + y_idx)

    sample_scaled = sample_63.copy()
    sample_scaled[xy_idx] = scaler.transform(sample_63.reshape(1, -1)[:, xy_idx])[0]

    sample_engineered = add_engineered_features(sample_scaled, feature_cols)
    expected = getattr(model, "n_features_in_", None)

    if expected is None:
        return sample_engineered.reshape(1, -1)
    if expected == sample_scaled.shape[0]:
        return sample_scaled.reshape(1, -1)
    if expected == sample_engineered.shape[0]:
        return sample_engineered.reshape(1, -1)

    raise ValueError(
        f"Model expects {expected} features, but got {sample_scaled.shape[0]} (base) "
        f"or {sample_engineered.shape[0]} (engineered)."
    )
