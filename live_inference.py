import argparse
import time
from collections import Counter, deque

import cv2

from helper import (
    base_feature_columns,
    build_model_input,
    load_pipeline,
    normalize_landmarks,
    open_camera,
)
import mediapipe as mp


def create_gesture_video(
    model_path="best_model.pkl",
    scaler_path="scaler.pkl",
    label_encoder_path="label_encoder.pkl",
    camera_index=0,
    output_path="live_frame_by_frame_output.mp4",
    min_det=0.5,
    min_track=0.5,
    smooth_window=3,
    max_seconds=0,
    preview=True,
):
    model, scaler, label_encoder = load_pipeline(model_path, scaler_path, label_encoder_path)
    feature_cols = base_feature_columns()

    cap = open_camera(camera_index)
    if cap is None:
        raise RuntimeError("Could not open camera.")

    mp_hands = mp.solutions.hands
    draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )

    writer = None
    buffer = deque(maxlen=max(1, int(smooth_window)))
    start = time.time()
    fps_t0 = time.time()
    fps = 0.0

    print("Video creation started. Press 'q' to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label_text = "No hand"
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                h, w = frame.shape[:2]
                sample_63 = normalize_landmarks(hand, w, h)
                model_input = build_model_input(sample_63, scaler, model, feature_cols)
                pred = model.predict(model_input)[0]
                label = label_encoder.inverse_transform([pred])[0]
                buffer.append(label)
                label_text = f"Gesture: {Counter(buffer).most_common(1)[0][0]}"

            now = time.time()
            dt = max(1e-6, now - fps_t0)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            fps_t0 = now

            cv2.putText(frame, label_text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if writer is None:
                h_out, w_out = frame.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (w_out, h_out),
                )
            writer.write(frame)

            if preview:
                cv2.imshow("Live Gesture Video Creator", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_seconds > 0 and (time.time() - start) >= max_seconds:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        hands.close()
        cv2.destroyAllWindows()

    print(f"Saved video: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Create labeled gesture video from webcam using trained model.")
    p.add_argument("--model", type=str, default="best_model.pkl")
    p.add_argument("--scaler", type=str, default="scaler.pkl")
    p.add_argument("--label-encoder", type=str, default="label_encoder.pkl")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--output", type=str, default="live_frame_by_frame_output.mp4")
    p.add_argument("--min-det", type=float, default=0.5)
    p.add_argument("--min-track", type=float, default=0.5)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--max-seconds", type=int, default=0, help="Stop automatically after N seconds (0 = no limit).")
    p.add_argument("--no-preview", action="store_true", help="Run without cv2.imshow window.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_gesture_video(
        model_path=args.model,
        scaler_path=args.scaler,
        label_encoder_path=args.label_encoder,
        camera_index=args.camera,
        output_path=args.output,
        min_det=args.min_det,
        min_track=args.min_track,
        smooth_window=args.smooth_window,
        max_seconds=args.max_seconds,
        preview=not args.no_preview,
    )
