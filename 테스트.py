print("룰과 매너를 지켜 즐겁게 듀얼!")
print("듀얼 개시!")

#pip install pytube mediapipe opencv-python
import os
import tempfile
import argparse

import cv2
import numpy as np
from pytube import YouTube
import mediapipe as mp


import os
import tempfile

import cv2
import numpy as np
from pytube import YouTube
import mediapipe as mp


def download_youtube_video(url, output_path):  # YouTube URL에서 MP4 영상을 다운로드
    yt = YouTube(url)
    # progressive 스트림 중 해상도 높은 것 선택
    stream = (yt.streams
                .filter(progressive=True, file_extension='mp4')
                .order_by('resolution').desc()
                .first())
    stream.download(output_path=os.path.dirname(output_path),
                    filename=os.path.basename(output_path))


def process_video(input_path, output_path):  # 입력 영상에서 사람만 분리해 새 영상으로 저장
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mp_seg = mp.solutions.selfie_segmentation
    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = seg.process(rgb)
            mask = results.segmentation_mask  # 사람일 확률 맵

            condition = mask > 0.5
            bg = np.zeros(frame.shape, dtype=np.uint8)
            composed = np.where(condition[..., None], frame, bg)

            out.write(composed)

    cap.release()
    out.release()


if __name__ == '__main__':
    # 링크 입력받기
    url = input('유튜브 영상 링크를 입력하세요: ').strip()
    # 출력 파일명 입력받기 (기본값 지정)
    output = input('저장할 파일명을 입력하세요 (기본: output.mp4): ').strip()
    if not output:
        output = 'output.mp4'

    # 임시 파일에 다운로드
    tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    print(f'▶ 영상을 다운로드 중… ({tmp_file})')
    download_youtube_video(url, tmp_file)

    print('▶ 배경 제거 처리 중…')
    process_video(tmp_file, output)

    os.remove(tmp_file)
    print(f'✅ 완료! 결과물이 "{output}"에 저장되었습니다.')
