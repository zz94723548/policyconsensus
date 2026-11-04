from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def _draw_weight_panel(
    frame: np.ndarray,
    weights: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    """Draw normalized policy weights on an RGB video frame."""
    height, width = frame.shape[:2]
    panel_width = min(max(300, width // 2), width - 16)
    row_height = 27
    panel_height = min(42 + row_height * len(weights), height - 16)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height),
                  (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)

    cv2.putText(
        frame, "Policy weights (softmax)", (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1,
        cv2.LINE_AA,
    )

    label_width = min(125, panel_width // 3)
    bar_left = 20 + label_width
    bar_right = 8 + panel_width - 58
    bar_width = max(20, bar_right - bar_left)
    colors = (
        (75, 190, 255), (45, 135, 230),
        (90, 220, 120), (45, 165, 85),
        (220, 130, 255), (170, 80, 220),
    )

    for idx, weight in enumerate(weights):
        y = 58 + idx * row_height
        if y + 12 > 8 + panel_height:
            break
        label = labels[idx] if idx < len(labels) else f"Expert {idx + 1}"
        color = colors[idx % len(colors)]
        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_left, y - 11), (bar_right, y + 2),
                      (75, 75, 75), thickness=-1)
        filled_right = bar_left + int(round(bar_width * float(weight)))
        cv2.rectangle(frame, (bar_left, y - 11), (filled_right, y + 2),
                      color, thickness=-1)
        cv2.putText(frame, f"{weight:.3f}", (bar_right + 7, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return frame


def annotate_policy_weights(
    video_path: str,
    decision_weights: Sequence[np.ndarray],
    labels: Sequence[str],
    frames_per_decision: int,
) -> str:
    """Create a copy of a rollout video with time-aligned gating weights."""
    if not video_path or not decision_weights:
        return video_path

    source_path = Path(video_path)
    output_path = source_path.with_name(f"{source_path.stem}_weights{source_path.suffix}")
    weights = [
        _softmax(np.asarray(step_weights).reshape(-1))
        for step_weights in decision_weights
    ]

    with av.open(str(source_path), mode="r") as input_container:
        input_stream = input_container.streams.video[0]
        frame_rate = input_stream.average_rate or 10
        with av.open(str(output_path), mode="w") as output_container:
            output_stream = output_container.add_stream("h264", rate=frame_rate)
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": "22", "profile": "high"}

            for frame_idx, video_frame in enumerate(input_container.decode(input_stream)):
                decision_idx = min(
                    frame_idx // max(1, frames_per_decision),
                    len(weights) - 1,
                )
                image = video_frame.to_ndarray(format="rgb24")
                image = _draw_weight_panel(image, weights[decision_idx], labels)
                annotated_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in output_stream.encode(annotated_frame):
                    output_container.mux(packet)

            for packet in output_stream.encode():
                output_container.mux(packet)

    return str(output_path)
