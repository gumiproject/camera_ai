#pip install yt-dlp mediapipe opencv-python numpy matplotlib
# Python 3.7 이상 – 3.10 이하 권장 (특히 3.8~3.10에서 안정적으로 동작합니다)


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
import mediapipe as mp
from yt_dlp import YoutubeDL


def normalize_youtube_url(url: str) -> str:
    """유튜브 URL에서 video ID만 뽑아 https://www.youtube.com/watch?v=<ID> 형태로 반환"""
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


def download_youtube_video(url: str, tmp_prefix: str) -> str:
    """
    yt_dlp로 비디오 다운로드.
    - FFmpeg 있으면 bestvideo+bestaudio 병합(mp4)
    - 없으면 best[ext=mp4] 단일 progressive 스트림
    반환: 생성된 .mp4 경로
    """
    logger.info("▶ 다운로드 시작: %s", url)

    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        # video+audio separate → mp4 병합
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'merge_output_format': 'mp4',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
    else:
        logger.warning("⚠️ ffmpeg가 감지되지 않아 progressive mp4 스트림만 다운로드합니다.")
        # audio+video 포함된 가장 화질 좋은 mp4 단일 스트림
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("❌ 다운로드 실패: %s", e)
        sys.exit(1)

    mp4_path = tmp_prefix + '.mp4'
    if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
        logger.error("❌ 다운로드된 mp4 파일이 없거나 손상되었습니다: %s", mp4_path)
        sys.exit(1)

    logger.info("✅ 다운로드 완료: %s", mp4_path)
    return mp4_path


def process_video(input_path: str, output_path: str):
    """Selfie Segmentation으로 사람만 남기고 배경을 검은색으로 바꿔 저장"""
    logger.info("▶ 배경 제거 처리 시작")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("❌ 파일 열기 실패: %s", input_path)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("▶ 프레임 처리 완료 (총 %d 프레임)", idx)
                break
            idx += 1

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = seg.process(rgb).segmentation_mask
            if mask is None:
                mask = np.zeros((height, width), dtype=np.float32)

            bg       = np.zeros_like(frame)
            composed = np.where(mask[..., None] > 0.5, frame, bg)
            out.write(composed)

            if idx % 100 == 0:
                logger.info("  • %d 프레임 처리 중…", idx)

    cap.release()
    out.release()
    logger.info("✅ 최종 파일 저장됨: %s", output_path)


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # 1) URL 입력 및 정규화
    raw = input('유튜브 영상 링크를 입력하세요: ').strip()
    if not raw.startswith(('http://', 'https://')):
        raw = 'https://' + raw
    url = normalize_youtube_url(raw)
    logger.info("▶ 사용 URL (정규화 후): %s", url)

    # 2) 출력 파일명
    output = input('저장할 파일명을 입력하세요 (기본: output.mp4): ').strip() or 'output.mp4'

    # 3) 임시 prefix
    tmp_prefix = tempfile.NamedTemporaryFile(delete=False).name

    # 4) 다운로드 & 배경 제거
    video_file = download_youtube_video(url, tmp_prefix)
    process_video(video_file, output)

    # 5) 임시 파일들(.mp4, .m4a 등) 삭제
    for ext in ('.mp4', '.m4a', '.webm', ''):
        try:
            os.remove(tmp_prefix + ext)
        except OSError:
            pass
