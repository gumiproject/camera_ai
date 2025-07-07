#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import logging
import shutil
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import matplotlib; matplotlib.use('Agg')
from yt_dlp import YoutubeDL
import mediapipe as mp

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def choose_local_file() -> str:
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="로컬 파일을 선택하세요",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def normalize_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ''
    vid = None
    if 'youtu.be' in host:
        vid = parsed.path.lstrip('/')
    elif 'youtube.com' in host:
        qs = parse_qs(parsed.query)
        if 'v' in qs:
            vid = qs['v'][0]
    return f'https://www.youtube.com/watch?v={vid}' if vid else url

def download_youtube_video(url: str, tmp_prefix: str, max_h: int) -> str:
    logger.info("▶ 다운로드 시작: %s", url)
    has_ffmpeg = shutil.which("ffmpeg") is not None
    opts = {
        'format': f'bestvideo[height<={max_h}]+bestaudio/best' if has_ffmpeg else f'best[height<={max_h}][ext=mp4]',
        'merge_output_format': 'mp4' if has_ffmpeg else None,
        'outtmpl': tmp_prefix + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("❌ 다운로드 실패: %s", e)
        sys.exit(1)

    mp4 = tmp_prefix + '.mp4'
    if not os.path.isfile(mp4) or os.path.getsize(mp4) == 0:
        logger.error("❌ 잘못된 mp4: %s", mp4)
        sys.exit(1)

    logger.info("✅ 다운로드 완료: %s", mp4)
    return mp4

def process_with_mediapipe(inp: str, outp: str, progress_callback=None):
    logger.info("▶ MediaPipe 세그멘테이션 시작")
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(inp)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(outp, fourcc, fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        mask = res.segmentation_mask
        m = (mask > 0.5).astype(np.uint8) if mask is not None else np.zeros((h, w), dtype=np.uint8)
        comp = np.where(m[..., None], frame, np.zeros_like(frame))
        writer.write(comp)

        if progress_callback:
            percent = int((idx / total_frames) * 100)
            progress_callback(percent)

        if idx % 100 == 0:
            logger.info("  • %d frames processed", idx)

    cap.release()
    writer.release()
    seg.close()
    logger.info("✅ MediaPipe 완료: %s", outp)

def process_with_yoloseg(inp: str, outp: str, model_name='yolov8n-seg.pt', target_indices=None, progress_callback=None):
    logger.info("▶ YOLOv8-seg 모델 로드: %s", model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_name).to(device)

    cap = cv2.VideoCapture(inp)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(outp, fourcc, fps, (w, h))

    if target_indices is None:
        target_indices = [0]  # 기본: 사람

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        results = model(frame)[0]
        masks = []
        if results.masks and results.masks.data is not None:
            for cls, m in zip(results.boxes.cls, results.masks.data):
                if int(cls) in target_indices:
                    masks.append(m.cpu().numpy())

        if masks:
            combined = np.clip(sum(masks), 0, 1).astype(np.uint8)
            combined_up = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
            comp = np.where(combined_up[..., None], frame, np.zeros_like(frame))
        else:
            comp = np.zeros_like(frame)

        writer.write(comp)

        if progress_callback:
            percent = int((idx / total_frames) * 100)
            progress_callback(percent)

        if idx % 50 == 0:
            logger.info("  • %d frames processed", idx)

    cap.release()
    writer.release()
    logger.info("✅ YOLO 완료: %s", outp)
