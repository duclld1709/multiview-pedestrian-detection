
"""
Run MVDet inference on 300 consecutive dataset frames.

For each frame:
    - 7 camera input images
    - predicted occupancy heatmap

The script composes them into ONE dashboard frame and finally encodes
all dashboard frames into a 30 FPS MP4 in the current working directory.

Usage:
    python infer_300_to_video.py \
        --data-root /path/to/Data \
        --dataset wildtrack \
        --checkpoint /path/to/MultiviewDetector.pth \
        --arch resnet18 \
        --start-frame 0 \
        --num-frames 300 \
        --fps 30 \
        --output mvdet_300frames.mp4

Notes:
    - This runs inference sequentially over the requested frames.
    - It does NOT run Streamlit.
    - The final video is written to the current directory unless --output
      specifies another path.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from demo_infer import resolve_dataset_root, build_model_and_load_ckpt
from multiview_detector.datasets import Wildtrack, MultiviewX, frameDataset
from multiview_detector.utils.image_utils import img_color_denormalize


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NUM_CAMERAS = 7
DEFAULT_NUM_FRAMES = 300
DEFAULT_FPS = 30

# Final video resolution.
VIDEO_W = 1920
VIDEO_H = 1080


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------

def fig_to_bgr(fig, width, height):
    """Convert a matplotlib figure to an OpenCV BGR image."""
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return cv2.resize(
        bgr,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def make_camera_grid(camera_images, frame_no):
    """
    Create a 4x2 camera grid.

    The first 7 cells contain the 7 cameras.
    The last cell is a title/status panel.
    """
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 7),
        facecolor="#101010",
    )

    axes = axes.ravel()

    for cam in range(NUM_CAMERAS):
        ax = axes[cam]
        img = camera_images[cam]

        ax.imshow(img)
        ax.set_title(
            f"CAM {cam + 1}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

    # 8th panel
    ax = axes[7]
    ax.set_facecolor("#101010")
    ax.text(
        0.5,
        0.58,
        "MVDet",
        ha="center",
        va="center",
        color="white",
        fontsize=28,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.40,
        f"FRAME {frame_no}",
        ha="center",
        va="center",
        color="#cccccc",
        fontsize=16,
        transform=ax.transAxes,
    )
    ax.axis("off")

    fig.tight_layout(
        pad=0.8,
        w_pad=0.5,
        h_pad=0.8,
    )

    image = fig_to_bgr(
        fig,
        VIDEO_W,
        VIDEO_H // 2,
    )

    plt.close(fig)

    return image


def make_occupancy_map(map_res):
    """
    Render predicted occupancy heatmap.
    """
    fig, ax = plt.subplots(
        figsize=(8, 5),
        facecolor="#101010",
    )

    occupancy = np.asarray(map_res)

    im = ax.imshow(
        occupancy,
        cmap="jet",
    )

    ax.set_title(
        "PREDICTED OCCUPANCY",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.03,
    )

    cbar.ax.tick_params(
        colors="white",
    )

    fig.tight_layout(
        pad=0.5,
    )

    image = fig_to_bgr(
        fig,
        VIDEO_W // 2,
        VIDEO_H // 2,
    )

    plt.close(fig)

    return image


def add_header(frame, frame_no, total_frames, fps):
    """Add a simple dashboard header."""
    cv2.rectangle(
        frame,
        (0, 0),
        (VIDEO_W, 65),
        (15, 23, 42),
        -1,
    )

    cv2.putText(
        frame,
        "MVDet - MULTIVIEW SURVEILLANCE",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    text = f"FRAME {frame_no + 1}/{total_frames}   |   {fps} FPS"

    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    cv2.putText(
        frame,
        text,
        (VIDEO_W - tw - 28, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def compose_dashboard(camera_grid, occupancy, frame_no, total_frames, fps):
    """
    Final 1920x1080 dashboard:

        ┌──────────────────────────────────────────┐
        │                 HEADER                   │
        ├──────────────────────────────────────────┤
        │                                            │
        │          7 CAMERA INPUTS                  │
        │                                            │
        ├───────────────────────┬───────────────────┤
        │                       │                   │
        │   CAMERA GRID         │ OCCUPANCY HEATMAP │
        │                       │                   │
        └───────────────────────┴───────────────────┘
    """
    frame = np.zeros(
        (VIDEO_H, VIDEO_W, 3),
        dtype=np.uint8,
    )

    header_h = 65
    top_h = VIDEO_H // 2 - header_h
    bottom_h = VIDEO_H - header_h - top_h

    camera_grid = cv2.resize(
        camera_grid,
        (VIDEO_W, top_h),
        interpolation=cv2.INTER_AREA,
    )

    occupancy = cv2.resize(
        occupancy,
        (VIDEO_W // 2, bottom_h),
        interpolation=cv2.INTER_AREA,
    )

    # Put camera section.
    frame[
        header_h:header_h + top_h,
        0:VIDEO_W,
    ] = camera_grid

    # Put occupancy section.
    frame[
        header_h + top_h:VIDEO_H,
        0:VIDEO_W // 2,
    ] = occupancy

    # Right-bottom status panel.
    x0 = VIDEO_W // 2
    y0 = header_h + top_h

    frame[
        y0:VIDEO_H,
        x0:VIDEO_W,
    ] = (18, 18, 18)

    cv2.putText(
        frame,
        "OCCUPANCY OUTPUT",
        (x0 + 35, y0 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Source frame : {frame_no}",
        (x0 + 35, y0 + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Playback FPS : {fps}",
        (x0 + 35, y0 + 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    add_header(
        frame,
        frame_no,
        total_frames,
        fps,
    )

    return frame


# ---------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------

def load_pipeline(data_root, dataset_name, checkpoint, arch, device_name):
    dataset_dir = resolve_dataset_root(
        data_root,
        dataset_name,
    )

    if dataset_name == "wildtrack":
        base = Wildtrack(dataset_dir)
    else:
        base = MultiviewX(dataset_dir)

    normalize = T.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    denormalize = img_color_denormalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )

    test_trans = T.Compose([
        T.Resize([720, 1280]),
        T.ToTensor(),
        normalize,
    ])

    test_set = frameDataset(
        base,
        train=False,
        transform=test_trans,
        grid_reduce=4,
    )

    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Bạn chọn CUDA nhưng torch.cuda.is_available() == False."
        )

    model = build_model_and_load_ckpt(
        test_set,
        arch,
        checkpoint,
        device,
    )

    model.eval()

    return base, test_set, model, denormalize, device


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=["wildtrack", "multiviewx"],
        default="wildtrack",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--arch",
        choices=["resnet18", "vgg11"],
        default="resnet18",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
    )

    parser.add_argument(
        "--output",
        default="mvdet_300frames.mp4",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    args = parser.parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames phải > 0.")

    if args.fps <= 0:
        raise ValueError("--fps phải > 0.")

    output_path = Path(args.output)

    # If a bare filename is provided, this is the current directory.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MVDet 300-frame inference -> MP4")
    print("=" * 70)
    print(f"Dataset       : {args.dataset}")
    print(f"Start frame   : {args.start_frame}")
    print(f"Num frames    : {args.num_frames}")
    print(f"FPS           : {args.fps}")
    print(f"Output        : {output_path.resolve()}")
    print("=" * 70)

    base, test_set, model, denormalize, device = load_pipeline(
        args.data_root,
        args.dataset,
        args.checkpoint,
        args.arch,
        args.device,
    )

    total_available = len(test_set)

    if args.start_frame < 0:
        raise ValueError("--start-frame phải >= 0.")

    if args.start_frame >= total_available:
        raise ValueError(
            f"start-frame={args.start_frame} nhưng dataset chỉ có "
            f"{total_available} frames."
        )

    end_frame = min(
        args.start_frame + args.num_frames,
        total_available,
    )

    actual_num_frames = end_frame - args.start_frame

    print(f"Available frames: {total_available}")
    print(
        f"Processing frames: "
        f"{args.start_frame} -> {end_frame - 1} "
        f"({actual_num_frames} frames)"
    )
    print(f"Device: {device}")
    print()

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        args.fps,
        (VIDEO_W, VIDEO_H),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Không thể mở VideoWriter: {output_path}"
        )

    inference_start = torch.cuda.Event(
        enable_timing=True
    ) if device == "cuda" else None

    inference_end = torch.cuda.Event(
        enable_timing=True
    ) if device == "cuda" else None

    import time
    wall_start = time.perf_counter()

    try:
        with torch.inference_mode():

            for n, idx in enumerate(
                range(args.start_frame, end_frame),
                start=1,
            ):
                frame_start = time.perf_counter()

                print(
                    f"[{n:03d}/{actual_num_frames}] "
                    f"Infer frame {idx}...",
                    end=" ",
                    flush=True,
                )

                imgs, map_gt, imgs_gt, frame_no = test_set[idx]

                imgs_b = imgs.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                if inference_start is not None:
                    inference_start.record()

                map_res, imgs_res = model(imgs_b)

                if inference_end is not None:
                    inference_end.record()
                    torch.cuda.synchronize()

                # -------------------------------------------------
                # Convert camera tensors back to display images.
                # -------------------------------------------------
                camera_images = (
                    denormalize(
                        imgs.unsqueeze(0)
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )

                if camera_images.shape[0] != NUM_CAMERAS:
                    raise RuntimeError(
                        f"Expected {NUM_CAMERAS} cameras, "
                        f"got {camera_images.shape[0]}."
                    )

                # -------------------------------------------------
                # Occupancy map.
                # -------------------------------------------------
                occupancy = (
                    map_res
                    .squeeze()
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------------
                # Render dashboard frame.
                # -------------------------------------------------
                camera_grid = make_camera_grid(
                    camera_images,
                    frame_no,
                )

                occupancy_image = make_occupancy_map(
                    occupancy,
                )

                dashboard = compose_dashboard(
                    camera_grid,
                    occupancy_image,
                    frame_no,
                    actual_num_frames,
                    args.fps,
                )

                writer.write(dashboard)

                elapsed = time.perf_counter() - frame_start

                if inference_start is not None:
                    gpu_ms = inference_start.elapsed_time(
                        inference_end
                    )
                    print(
                        f"GPU={gpu_ms:.1f} ms | "
                        f"total={elapsed:.2f} s"
                    )
                else:
                    print(
                        f"total={elapsed:.2f} s"
                    )

                # Explicitly release frame GPU tensor references.
                del imgs_b
                del map_res
                del imgs_res

    finally:
        writer.release()

    total_time = time.perf_counter() - wall_start

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Frames written : {actual_num_frames}")
    print(f"Video FPS      : {args.fps}")
    print(f"Video duration : {actual_num_frames / args.fps:.2f} sec")
    print(f"Total runtime  : {total_time:.2f} sec")
    print(f"Output         : {output_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
