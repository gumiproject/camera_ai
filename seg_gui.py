import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
import threading
import tempfile
import os
import shutil
import logging

from 테스트 import (
    process_with_yoloseg,
    process_with_mediapipe,
    YOLO_AVAILABLE,
    normalize_youtube_url,
    download_youtube_video
)

# logger 정의 추가
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation GUI")
        self.root.geometry("450x320")

        self.input_path = ""
        self.temp_file_prefix = None
        self.output_path = tk.StringVar(value="output.mp4")

        # 유튜브 입력
        tk.Label(root, text="1. 유튜브 링크 (또는 비워두기)").pack(pady=5)
        self.entry_url = tk.Entry(root, width=50)
        self.entry_url.pack()

        # 또는 로컬 파일 선택
        tk.Label(root, text="2. 또는 로컬 영상 선택").pack(pady=5)
        self.btn_choose = tk.Button(root, text="파일 선택", command=self.choose_file)
        self.btn_choose.pack()

        # 출력 파일명
        tk.Label(root, text="3. 결과 저장 파일명").pack(pady=5)
        self.entry_out = tk.Entry(root, textvariable=self.output_path)
        self.entry_out.pack()

        # 방법 선택
        tk.Label(root, text="4. 세그멘테이션 방법").pack(pady=5)
        self.combo_method = Combobox(root, values=["YOLOv8", "MediaPipe"])
        self.combo_method.current(0 if YOLO_AVAILABLE else 1)
        self.combo_method.pack()

        # 실행 버튼
        self.btn_run = tk.Button(root, text="처리 시작", command=self.run_thread)
        self.btn_run.pack(pady=10)

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if path:
            self.input_path = path
            self.btn_choose.config(text=os.path.basename(path))

    def run_thread(self):
        thread = threading.Thread(target=self.run_process)
        thread.start()

    def run_process(self):
        url = self.entry_url.get().strip()
        outp = self.output_path.get().strip() or "output.mp4"
        method = self.combo_method.get()

        # 유튜브 다운로드
        if url:
            try:
                url = normalize_youtube_url(url)
                self.temp_file_prefix = tempfile.NamedTemporaryFile(delete=False).name
                inp = download_youtube_video(url, self.temp_file_prefix, 1080)
            except Exception as e:
                messagebox.showerror("유튜브 다운로드 오류", str(e))
                return
        elif self.input_path:
            inp = self.input_path
        else:
            messagebox.showerror("입력 오류", "유튜브 링크나 로컬 파일 중 하나를 선택하세요.")
            return

        try:
            if method == "YOLOv8" and YOLO_AVAILABLE:
                process_with_yoloseg(inp, outp)
            else:
                process_with_mediapipe(inp, outp)
            messagebox.showinfo("완료", f"처리 완료!\n저장 파일: {outp}")
        except Exception as e:
            messagebox.showerror("처리 오류", str(e))
        finally:
            # 임시파일 정리
            if self.temp_file_prefix:
                for ext in ('.mp4', '.m4a', '.webm', ''):
                    try:
                        os.remove(self.temp_file_prefix + ext)
                    except OSError:
                        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
