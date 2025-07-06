#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, tempfile, logging, shutil, re, uuid, threading
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
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

# 작업 상태 및 스레드 동기화를 위한 글로벌 변수
TASKS = {}
task_lock = threading.Lock()

# ── YouTube URL 정규화 및 ID 추출 ──────────────────────────────────
def get_yt_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|shorts\/|live\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ── yt-dlp 다운로드 ────────────────────────────────────────────────
def download_yt(url: str, out_path: str, max_h: int = 1080):
    video_id = get_yt_video_id(url)
    if not video_id:
        raise ValueError("유효하지 않은 YouTube URL입니다.")
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
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
        ydl.download([clean_url])

# ── 백그라운드 작업을 위한 함수 (진행도 업데이트 포함) ───────────────
def _process_video(task_id: str, processing_func, src_path: str, dst_path: str):
    """실제 비디오 처리 로직을 감싸고 진행도를 업데이트하는 함수"""
    try:
        # app 컨텍스트 내에서 실행하여 스레드 관련 문제 방지
        with app.app_context():
            processing_func(task_id, src_path, dst_path)
        with task_lock:
            TASKS[task_id]['status'] = 'complete'
            TASKS[task_id]['progress'] = 100
    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}")
        with task_lock:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['error'] = str(e)

# ── MediaPipe 세그멘테이션 (진행도 보고 기능 추가) ───────────────────
def seg_mediapipe(task_id: str, inp: str, outp: str):
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(inp)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with task_lock:
        TASKS[task_id]['status'] = 'processing'
    
    fps,w,h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    
    frame_num = 0
    while True:
        ok,f = cap.read();
        if not ok: break
        frame_num += 1
        m = seg.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).segmentation_mask
        mask = (m>0.5).astype(np.uint8) if m is not None else np.zeros((h,w),np.uint8)
        out.write(np.where(mask[...,None], f, 0))
        if total_frames > 0:
            progress = int((frame_num / total_frames) * 100)
            with task_lock:
                TASKS[task_id]['progress'] = progress

    cap.release(); out.release(); seg.close()

# ── YOLOv8-seg 세그멘테이션 (진행도 보고 기능 추가) ──────────────────
def seg_yolo(task_id: str, inp: str, outp: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = YOLO('yolov8n-seg.pt').to(device)
    cap = cv2.VideoCapture(inp)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with task_lock:
        TASKS[task_id]['status'] = 'processing'
    
    fps,w,h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    
    frame_num = 0
    while True:
        ok,f = cap.read();
        if not ok: break
        frame_num += 1
        res = model(f, verbose=False)[0]
        mk = [m.cpu().numpy() for c,m in zip(res.boxes.cls, res.masks.data) if int(c)==0] if res.masks else []
        if mk:
            up = cv2.resize(np.clip(sum(mk),0,1).astype(np.uint8),(w,h),cv2.INTER_NEAREST)
            f  = np.where(up[...,None], f, 0)
        else:
            f = np.zeros_like(f)
        out.write(f)
        if total_frames > 0:
            progress = int((frame_num / total_frames) * 100)
            with task_lock:
                TASKS[task_id]['progress'] = progress
    
    cap.release(); out.release()

# ── Flask 라우트 ───────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_keywords = ['mobi', 'iphone', 'ipad', 'android', 'ipod', 'windows phone']
    is_mobile = any(keyword in user_agent for keyword in mobile_keywords)

    if is_mobile:
        return render_template('mobile.html', yolo_ok=YOLO_OK)
    else:
        return render_template('index.html', yolo_ok=YOLO_OK)

@app.route('/submit', methods=['POST'])
def submit():
    yt = request.form.get('youtube_url','').strip()
    up = request.files.get('video_file')
    if not yt and (not up or up.filename==''):
        return jsonify({"error": "URL 또는 파일을 입력해 주세요"}), 400

    task_id = str(uuid.uuid4())
    src = os.path.join(TMP_DIR, f"{task_id}.mp4")
    dst = os.path.join(TMP_DIR, f"{task_id}_out.mp4")

    with task_lock:
        TASKS[task_id] = {'status': 'pending', 'progress': 0, 'src_path': src, 'dst_path': dst}

    try:
        if yt:
            with task_lock:
                TASKS[task_id]['status'] = 'downloading'
            download_yt(yt, src)
        else:
            up.save(src)
    except Exception as e:
        logging.error(f"Download/Upload error for task {task_id}: {e}")
        with task_lock:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['error'] = f"다운로드/업로드 오류: {e}"
        return jsonify({"error": TASKS[task_id]['error']}), 500

    method = request.form.get('method','mp')
    process_func = seg_yolo if (method=='yolo' and YOLO_OK) else seg_mediapipe
    
    thread = threading.Thread(target=_process_video, args=(task_id, process_func, src, dst))
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/progress/<task_id>', methods=['GET'])
def progress(task_id):
    with task_lock:
        task = TASKS.get(task_id)
    if not task:
        return jsonify({"error": "작업을 찾을 수 없습니다."}), 404
    return jsonify(task)

@app.route('/download/<task_id>', methods=['GET'])
def download(task_id):
    with task_lock:
        task = TASKS.get(task_id)

    if not task or task.get('status') != 'complete':
        # flash()와 redirect()는 컨텍스트 문제를 일으킬 수 있으므로 직접 사용하지 않습니다.
        # 프론트엔드에서 이미 오류를 처리하고 있으므로 여기서는 단순히 메인으로 보냅니다.
        logging.warning(f"Download attempt for incomplete/failed task: {task_id}")
        return redirect(url_for('index'))
    
    dst_path = task.get('dst_path')
    if not os.path.exists(dst_path):
        logging.error(f"Result file not found for task: {task_id}")
        return redirect(url_for('index'))
        
    return send_file(dst_path, as_attachment=True, download_name='result.mp4')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    app.run(host='0.0.0.0', port=8000, threaded=True)