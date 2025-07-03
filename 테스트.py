#pip install yt-dlp mediapipe opencv-python numpy matplotlib
# Python 3.7 이상 – 3.10 이하 권장 (특히 3.8~3.10에서 안정적으로 동작합니다)
# 개발 환경 3.10.8, Windows 11


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, tempfile, logging, shutil
from urllib.parse import urlparse, parse_qs

import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')  # tkagg 메시지 억제
import mediapipe as mp
from yt_dlp import YoutubeDL


def normalize_youtube_url(url: str) -> str:  # 유튜브 URL에서 video ID만 뽑아 정규화
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


def download_youtube_video(url: str, tmp_prefix: str, max_height: int) -> str:  # 영상 다운로드
    logger.info("▶ 다운로드 시작: %s", url)
    has_ffmpeg = shutil.which("ffmpeg") is not None

    if has_ffmpeg:
        ydl_opts = {
            'format': f'bestvideo[height<={max_height}]+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
        }
    else:
        logger.warning("⚠️ ffmpeg 미설치: progressive mp4 스트림만 다운로드합니다.")
        ydl_opts = {
            'format': f'best[height<={max_height}][ext=mp4]',
            'outtmpl': tmp_prefix + '.%(ext)s',
            'quiet': True, 'no_warnings': True,
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


def process_video(input_path: str, output_path: str):  # 배경 제거 처리
    logger.info("▶ 배경 제거 처리 시작")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("❌ 파일 열기 실패: %s", input_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("▶ 프레임 처리 완료 (총 %d 프레임)", idx)
                break
            idx += 1

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = seg.process(rgb)
            # segmentation_mask가 None일 때만 대체 배열 생성
            if results.segmentation_mask is None:
                mask = np.zeros((h, w), dtype=np.float32)
            else:
                mask = results.segmentation_mask

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

    raw = input('유튜브 영상 링크를 입력하세요: ').strip()
    if not raw.startswith(('http://', 'https://')):
        raw = 'https://' + raw
    url = normalize_youtube_url(raw)
    logger.info("▶ 사용 URL (정규화 후): %s", url)

    output = input('저장할 파일명을 입력하세요 (기본: output.mp4): ').strip() or 'output.mp4'

    try:
        mh = int(input('최대 해상도 높이(pixels) 입력(예:2160,1440,1080; 기본1080): ').strip() or '1080')
    except ValueError:
        mh = 1080
    logger.info("▶ 선택된 최대 해상도: %d", mh)

    tmp_prefix = tempfile.NamedTemporaryFile(delete=False).name
    video_file = download_youtube_video(url, tmp_prefix, mh)
    process_video(video_file, output)

    # 임시 파일(.mp4, .m4a, .webm 등) 정리
    for ext in ('.mp4', '.m4a', '.webm', ''):
        try: os.remove(tmp_prefix + ext)
        except: pass
