#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
필수 패키지
──────────
pip install flask yt-dlp mediapipe opencv-python-headless numpy ultralytics gunicorn
# GPU PC
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU PC
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
"""

import os, tempfile, logging, shutil
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from yt_dlp import YoutubeDL
import cv2, numpy as np, mediapipe as mp

# YOLO가 설치돼 있으면 사용
try:
    import torch
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

app = Flask(__name__)
app.secret_key = "please-change-this"
TMP_DIR = tempfile.gettempdir()

# ── YouTube URL 정규화 ─────────────────────────────────────────────
def clean_url(raw: str) -> str:
    p = urlparse(raw)
    host = (p.hostname or "").lower()
    vid = None
    if 'youtube.com' in host and p.path.startswith('/shorts/'):
        vid = p.path.split('/')[2]
    elif 'youtu.be' in host:
        vid = p.path.lstrip('/')
    elif 'youtube.com' in host:
        vid = parse_qs(p.query).get('v', [None])[0]
    if not vid:
        raise ValueError("잘못된 YouTube 링크입니다.")
    return f"https://www.youtube.com/watch?v={vid}"

# ── yt-dlp 다운로드 ────────────────────────────────────────────────
def download_yt(url: str, out_path: str, max_h: int = 1080):
    has_ffmpeg = shutil.which("ffmpeg") is not None
    opt = {
        "format": (f"bestvideo[height<={max_h}]+bestaudio/best"
                   if has_ffmpeg else
                   f"best[height<={max_h}][ext=mp4]"),
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(opt) as ydl:
        ydl.download([url])

# ── MediaPipe 세그멘테이션 ────────────────────────────────────────
def seg_mediapipe(inp: str, outp: str):
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(inp)
    fps,w,h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    while True:
        ok,f = cap.read();  # noqa
        if not ok: break
        m = seg.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).segmentation_mask
        mask = (m>0.5).astype(np.uint8) if m is not None else np.zeros((h,w),np.uint8)
        out.write(np.where(mask[...,None], f, 0))
    cap.release(); out.release(); seg.close()

# ── YOLOv8-seg 세그멘테이션 ───────────────────────────────────────
def seg_yolo(inp: str, outp: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = YOLO('yolov8n-seg.pt').to(device)
    cap = cv2.VideoCapture(inp)
    fps,w,h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    while True:
        ok,f = cap.read();  # noqa
        if not ok: break
        res = model(f)[0]
        mk = [m.cpu().numpy() for c,m in zip(res.boxes.cls, res.masks.data) if int(c)==0] if res.masks else []
        if mk:
            up = cv2.resize(np.clip(sum(mk),0,1).astype(np.uint8),(w,h),cv2.INTER_NEAREST)
            f  = np.where(up[...,None], f, 0)
        else:
            f = np.zeros_like(f)
        out.write(f)
    cap.release(); out.release()

# ── Flask 라우트 ─────────────────────────────────────────────────
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        yt = request.form.get('youtube_url','').strip()
        up = request.files.get('video_file')
        if not yt and (not up or up.filename==''):
            flash("URL 또는 파일을 입력해 주세요", "danger")
            return redirect(url_for('index'))

        # 임시 파일 경로
        src = os.path.join(TMP_DIR, next(tempfile._get_candidate_names())+'.mp4')
        dst = src.replace('.mp4','_out.mp4')

        try:
            if yt:
                yt_clean = clean_url(yt)
                download_yt(yt_clean, src)
            else:
                up.save(src)
        except Exception as e:
            flash(f"다운로드/업로드 오류: {e}", "danger")
            return redirect(url_for('index'))

        method = request.form.get('method','mp')
        try:
            if method=='yolo' and YOLO_OK:
                seg_yolo(src, dst)
            else:
                seg_mediapipe(src, dst)
        except Exception as e:
            flash(f"처리 오류: {e}", "danger")
            return redirect(url_for('index'))

        return send_file(dst, as_attachment=True, download_name='result.mp4')

    # 'yolo_ok'를 'use_yolo'로 변경하여 템플릿과 일치시킵니다.
    return render_template('index.html', use_yolo=YOLO_OK)

# ── 로컬 개발용 실행 ─────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    app.run(host='0.0.0.0', port=8000)
