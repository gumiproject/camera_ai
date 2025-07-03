#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prerequisites:

# GPU 환경용 (CUDA 11.x 빌드)
pip install yt_dlp ultralytics opencv-python numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 전용 환경용
pip install yt_dlp ultralytics opencv-python numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

또는, Ultraytics(YOLOv8) 없이 MediaPipe만 쓰려면:
pip install yt_dlp mediapipe opencv-python numpy matplotlib
"""

import os
import sys
import tempfile
import logging
import shutil
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import matplotlib; matplotlib.use('Agg')  # suppress tkagg messages
from yt_dlp import YoutubeDL

# Try to import YOLOv8-seg; if unavailable, skip
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

import mediapipe as mp


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


def download_youtube_video(url: str, tmp_prefix: str, max_height: int) -> str:
    logger.info("▶ 다운로드 시작: %s", url)
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        opts = {
            'format': f'bestvideo[height<={max_height}]+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
        }
    else:
        logger.warning("⚠️ ffmpeg 미설치: progressive mp4만 다운로드")
        opts = {
            'format': f'best[height<={max_height}][ext=mp4]',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
        }

    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("❌ 다운로드 실패: %s", e)
        sys.exit(1)

    mp4_path = tmp_prefix + '.mp4'
    if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
        logger.error("❌ 잘못된 mp4: %s", mp4_path)
        sys.exit(1)

    logger.info("✅ 다운로드 완료: %s", mp4_path)
    return mp4_path


def process_with_mediapipe(input_path: str, output_path: str):
    logger.info("▶ MediaPipe 세그멘테이션 시작")
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(input_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = seg.process(rgb)
        mask    = res.segmentation_mask
        if mask is None:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = (mask > 0.5).astype(np.uint8)

        bg   = np.zeros_like(frame)
        comp = np.where(mask[..., None], frame, bg)
        out.write(comp)

        if frame_idx % 100 == 0:
            logger.info("  • %d 프레임 처리 중…", frame_idx)

    cap.release()
    out.release()
    seg.close()
    logger.info("✅ MediaPipe 처리 완료: %s", output_path)


def process_with_yoloseg(input_path: str, output_path: str, model_name='yolov8n-seg.pt'):
    logger.info("▶ YOLOv8-seg 모델 로드: %s", model_name)
    # CPU/GPU 자동 선택
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_name).to(device)

    cap = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame)[0]

        masks = []
        if results.masks and results.masks.data is not None:
            for cls, m in zip(results.boxes.cls, results.masks.data):
                if int(cls) == 0:
                    masks.append(m.cpu().numpy())

        if masks:
            combined = np.clip(sum(masks), 0, 1).astype(np.uint8)
            combined_up = cv2.resize(
                combined, (width, height),
                interpolation=cv2.INTER_NEAREST
            )
            bg   = np.zeros_like(frame)
            comp = np.where(combined_up[..., None], frame, bg)
        else:
            comp = np.zeros_like(frame)

        out.write(comp)
        if frame_idx % 50 == 0:
            logger.info("  • %d 프레임 처리 중…", frame_idx)

    cap.release()
    out.release()
    logger.info("✅ YOLO 처리 완료: %s", output_path)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    raw = input('유튜브 링크: ').strip()
    if not raw.startswith(('http://','https://')):
        raw = 'https://' + raw
    url = normalize_youtube_url(raw)
    logger.info("▶ 정규화 URL: %s", url)

    output = input('저장 파일명 (기본 output.mp4): ').strip() or 'output.mp4'
    try:
        mh = int(input('최대 해상도 높이 (예:2160,1440,1080; 기본1080): ').strip() or '1080')
    except ValueError:
        mh = 1080
    logger.info("▶ 해상도 제한: %d", mh)

    tmp_pref   = tempfile.NamedTemporaryFile(delete=False).name
    video_file = download_youtube_video(url, tmp_pref, mh)

    # 사용자 선택: YOLO 또는 MediaPipe
    if YOLO_AVAILABLE:
        choice = input('세그멘테이션 방법을 선택하세요 (1: YOLOv8-seg, 2: MediaPipe) [기본 1]: ').strip() or '1'
    else:
        logger.warning("⚠️ YOLOv8-seg 라이브러리 미설치: MediaPipe만 사용 가능합니다.")
        choice = '2'

    if choice == '1' and YOLO_AVAILABLE:
        logger.info("▶ 선택됨: YOLOv8-seg")
        process_with_yoloseg(video_file, output)
    else:
        logger.info("▶ 선택됨: MediaPipe")
        process_with_mediapipe(video_file, output)

    # 임시 파일 정리
    for ext in ('.mp4', '.m4a', '.webm', ''):
        try:
            os.remove(tmp_pref + ext)
        except OSError:
            pass
