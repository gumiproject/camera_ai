#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GPU있으면 pip install yt-dlp mediapipe opencv-python numpy matplotlib ultralytics 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU면 pip install yt-dlp mediapipe opencv-python numpy matplotlib ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


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

# 파일 다이얼로그
from tkinter import Tk, filedialog

# YOLOv8-seg 사용 가능 여부
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
    """파일 탐색기를 띄워 사용자가 파일을 선택하게 한 뒤, 경로를 반환."""
    root = Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    file_path = filedialog.askopenfilename(
        title="로컬 파일을 선택하세요",
        filetypes=[("MP4 files","*.mp4"), ("All files","*.*")]
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
    if has_ffmpeg:
        opts = {
            'format': f'bestvideo[height<={max_h}]+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
        }
    else:
        logger.warning("⚠️ ffmpeg 미설치: progressive mp4만 다운로드")
        opts = {
            'format': f'best[height<={max_h}][ext=mp4]',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
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


def process_with_mediapipe(inp: str, outp: str):
    logger.info("▶ MediaPipe 세그멘테이션 시작")
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(inp)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(outp, fourcc, fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = seg.process(rgb)
        mask = res.segmentation_mask
        if mask is None:
            m = np.zeros((h, w), dtype=np.uint8)
        else:
            m = (mask > 0.5).astype(np.uint8)
        comp = np.where(m[..., None], frame, np.zeros_like(frame))
        writer.write(comp)
        if idx % 100 == 0:
            logger.info("  • %d frames processed", idx)

    cap.release()
    writer.release()
    seg.close()
    logger.info("✅ MediaPipe 완료: %s", outp)


def process_with_yoloseg(inp: str, outp: str, model_name='yolov8n-seg.pt'):
    logger.info("▶ YOLOv8-seg 모델 로드: %s", model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = YOLO(model_name).to(device)

    cap    = cv2.VideoCapture(inp)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(outp, fourcc, fps, (w, h))

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
                if int(cls) == 0:
                    masks.append(m.cpu().numpy())

        if masks:
            combined   = np.clip(sum(masks), 0, 1).astype(np.uint8)
            combined_up = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
            comp       = np.where(combined_up[..., None], frame, np.zeros_like(frame))
        else:
            comp = np.zeros_like(frame)

        writer.write(comp)
        if idx % 50 == 0:
            logger.info("  • %d frames processed", idx)

    cap.release()
    writer.release()
    logger.info("✅ YOLO 완료: %s", outp)


if __name__ == '__main__':
    # 로깅
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    """

    # 입력 모드 선택
    mode = input('처리할 영상 선택: 1) YouTube 링크  2) 로컬 파일 [기본 1]: ').strip() or '1'
    if mode == '2':
        inp = choose_local_file()
        if not inp:
            logger.error("❌ 파일을 선택하지 않았습니다.")
            sys.exit(1)
    else:
        raw = input('YouTube 링크를 입력하세요: ').strip()
        if not raw.startswith(('http://','https://')):
            raw = 'https://' + raw
        url = normalize_youtube_url(raw)
        mh  = input('최대 해상도 높이(pixels) 입력(기본1080): ').strip() or '1080'
        try:
            mh = int(mh)
        except ValueError:
            mh = 1080
        tmp = tempfile.NamedTemporaryFile(delete=False).name
        inp = download_youtube_video(url, tmp, mh)

    outp = input('결과 저장 파일명 (기본 output.mp4): ').strip() or 'output.mp4'

    # 세그멘테이션 방법 선택
    if YOLO_AVAILABLE:
        choice = input('방법 선택: 1) YOLOv8-seg  2) MediaPipe [기본 1]: ').strip() or '1'
    else:
        logger.warning("⚠️ YOLOv8-seg 미설치: MediaPipe만 사용")
        choice = '2'

    if choice == '1' and YOLO_AVAILABLE:
        process_with_yoloseg(inp, outp)
    else:
        process_with_mediapipe(inp, outp)

    # 임시 파일 정리
    if mode != '2':
        for ext in ('.mp4','.m4a','.webm',''):
            try:
                os.remove(tmp + ext)
            except OSError:
                pass

    logger.info("모두 완료되었습니다.")
