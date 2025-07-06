#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
필수 패키지
──────────
pip install flask yt-dlp mediapipe opencv-python-headless numpy ultralytics gunicorn
"""

import os, tempfile, logging, shutil, uuid
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, render_template, send_file, jsonify, url_for
from yt_dlp import YoutubeDL
import cv2, numpy as np, mediapipe as mp
from threading import Thread, Lock

# YOLO가 설치돼 있으면 사용
try:
    import torch
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

app = Flask(__name__)
app.secret_key = "please-change-this-for-production"
TMP_DIR = tempfile.gettempdir()

# --- 진행 상황 저장을 위한 전역 변수 ---
tasks = {}
tasks_lock = Lock()
# ------------------------------------

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

def download_yt(url: str, out_path: str, task_id: str):
    with tasks_lock:
        tasks[task_id]['status'] = '다운로드 중...'

    has_ffmpeg = shutil.which("ffmpeg") is not None
    opt = {
        "format": (f"bestvideo[height<=1080]+bestaudio/best"
                   if has_ffmpeg else
                   f"best[height<=1080][ext=mp4]"),
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(opt) as ydl:
        ydl.download([url])

# --- 진행률 업데이트 기능이 추가된 처리 함수들 ---
def update_progress(task_id: str, status: str, progress: int):
    with tasks_lock:
        tasks[task_id]['status'] = status
        tasks[task_id]['progress'] = progress

def seg_mediapipe(inp: str, outp: str, task_id: str):
    try:
        seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        cap = cv2.VideoCapture(inp)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps, w, h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        
        frame_count = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            
            m = seg.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).segmentation_mask
            mask = (m > 0.5).astype(np.uint8) if m is not None else np.zeros((h, w), np.uint8)
            out.write(np.where(mask[..., None], f, 0))
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            update_progress(task_id, '처리 중...', progress)

        cap.release(); out.release(); seg.close()
        update_progress(task_id, '완료', 100)
    except Exception as e:
        update_progress(task_id, f'오류: {e}', -1)


def seg_yolo(inp: str, outp: str, task_id: str):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('yolov8n-seg.pt').to(device)
        cap = cv2.VideoCapture(inp)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps, w, h = cap.get(5) or 30.0, int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_count = 0
        while True:
            ok, f = cap.read()
            if not ok: break

            res = model(f)[0]
            mk = [m.cpu().numpy() for c, m in zip(res.boxes.cls, res.masks.data) if int(c) == 0] if res.masks else []
            if mk:
                up = cv2.resize(np.clip(sum(mk), 0, 1).astype(np.uint8), (w, h), cv2.INTER_NEAREST)
                f = np.where(up[..., None], f, 0)
            else:
                f = np.zeros_like(f)
            out.write(f)
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            update_progress(task_id, '처리 중...', progress)

        cap.release(); out.release()
        update_progress(task_id, '완료', 100)
    except Exception as e:
        update_progress(task_id, f'오류: {e}', -1)

# --- 실제 작업을 수행하는 함수 (백그라운드 스레드에서 실행됨) ---
def process_video_task(task_id, method, yt_url, has_file, temp_src_path, temp_dst_path):
    try:
        if yt_url:
            yt_clean = clean_url(yt_url)
            download_yt(yt_clean, temp_src_path, task_id)
        elif not has_file:
             raise ValueError("파일이 제공되지 않았습니다.")
        
        update_progress(task_id, '처리 대기 중...', 0)

        if method == 'yolo' and YOLO_OK:
            seg_yolo(temp_src_path, temp_dst_path, task_id)
        else:
            seg_mediapipe(temp_src_path, temp_dst_path, task_id)
    except Exception as e:
        logging.error(f"작업 오류 [ID: {task_id}]: {e}")
        update_progress(task_id, f'오류: {e}', -1)

# --- Flask 라우트 (API 엔드포인트) ---
@app.route('/')
def index():
    return render_template('index.html', use_yolo=YOLO_OK)

@app.route('/process', methods=['POST'])
def process():
    task_id = str(uuid.uuid4())
    yt = request.form.get('youtube_url', '').strip()
    up = request.files.get('video_file')

    if not yt and (not up or up.filename == ''):
        return jsonify({'error': 'URL 또는 파일을 입력해 주세요'}), 400

    src = os.path.join(TMP_DIR, f"{task_id}.mp4")
    dst = os.path.join(TMP_DIR, f"{task_id}_out.mp4")

    if up and up.filename != '':
        up.save(src)

    with tasks_lock:
        tasks[task_id] = {'status': '대기 중', 'progress': 0, 'output_path': dst}

    # 백그라운드 스레드에서 비디오 처리 시작
    thread = Thread(target=process_video_task, args=(task_id, request.form.get('method', 'mp'), yt, (up and up.filename != ''), src, dst))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    return jsonify(task if task else {'status': '알 수 없는 작업 ID', 'progress': -1})

@app.route('/download/<task_id>')
def download(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if task and task['status'] == '완료':
        return send_file(task['output_path'], as_attachment=True, download_name='result.mp4')
    return "파일을 찾을 수 없거나 작업이 완료되지 않았습니다.", 404

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    app.run(host='0.0.0.0', port=8000)

