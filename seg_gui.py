import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox, Progressbar
import threading
import tempfile
import os
from PIL import Image, ImageTk
import urllib.request
import io
import re

from 테스트 import (
    process_with_yoloseg,
    process_with_mediapipe,
    YOLO_AVAILABLE,
    normalize_youtube_url,
    download_youtube_video
)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation GUI")
        self.root.geometry("480x700")
        self.root.resizable(False, False)
        self.root.configure(bg="white")

        self.input_path = ""
        self.temp_file_prefix = None
        self.output_path = tk.StringVar(value="output.mp4")

        container = tk.Frame(root, bg="white")
        container.pack(fill='both', expand=True, padx=10, pady=10)

        tk.Label(container, text="Segmentation 프로그램", font=("맑은 고딕", 16, "bold"), bg="white").pack(pady=(0, 10))

        tk.Label(container, text="1. 유튜브 링크 (또는 비워두기)", anchor='w', bg="white").pack(fill='x')
        self.entry_url = tk.Entry(container, bg="white")
        self.entry_url.pack(fill='x')
        self.entry_url.bind("<Return>", self.update_thumbnail)
        self.entry_url.bind("<KeyRelease>", self.check_url_clear)

        self.thumbnail_label = tk.Label(container, bg="white")
        self.thumbnail_label.pack(pady=5)

        tk.Label(container, text="2. 또는 로컬 영상 선택", anchor='w', bg="white").pack(fill='x', pady=(10, 0))
        file_frame = tk.Frame(container, bg="white")
        file_frame.pack(fill='x')
        file_frame.columnconfigure(0, weight=7)
        file_frame.columnconfigure(1, weight=3)
        self.btn_choose = tk.Button(file_frame, text="파일 선택", bg="#2196F3", fg="white", command=self.choose_file)
        self.btn_choose.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.btn_clear = tk.Button(file_frame, text="선택 취소", bg="#f44336", fg="white", command=self.clear_file)
        self.btn_clear.grid(row=0, column=1, sticky='ew')

        tk.Label(container, text="3. 결과 저장 파일명", anchor='w', bg="white").pack(fill='x', pady=(10, 0))
        self.entry_out = tk.Entry(container, textvariable=self.output_path, bg="white")
        self.entry_out.pack(fill='x')

        tk.Label(container, text="4. 세그멘테이션 방식", anchor='w', bg="white").pack(fill='x', pady=(10, 0))
        self.combo_method = Combobox(container, values=["YOLOv8", "MediaPipe"])
        self.combo_method.current(0 if YOLO_AVAILABLE else 1)
        self.combo_method.pack(fill='x')

        # 추출 대상: 사람/동물
        self.extract_target = tk.StringVar(value='person')
        frame_target = tk.Frame(container, bg="white")
        frame_target.pack(fill='x', pady=(10, 0))

        tk.Label(frame_target, text="추출 대상", bg="white").pack(side='left')
        tk.Radiobutton(frame_target, text="사람", variable=self.extract_target, value='person', bg="white", command=self.toggle_animal_options).pack(side='left')
        tk.Radiobutton(frame_target, text="동물", variable=self.extract_target, value='animal', bg="white", command=self.toggle_animal_options).pack(side='left')

        # 동물 선택 체크박스
        self.animal_vars = {}
        self.animal_classes = {
            "강아지": 16, "고양이": 15, "말": 17, "새": 14,
            "코끼리": 20, "양": 19, "소": 21, "곰": 23, "얼룩말": 24, "기린": 25
        }

        frame_animals = tk.LabelFrame(container, text="분리할 동물을 선택하세요", bg="white")
        frame_animals.pack(fill='x', padx=5, pady=(5, 10))
        self.frame_animals = frame_animals

        row, col = 0, 0
        for name in self.animal_classes:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(frame_animals, text=name, variable=var, bg="white")
            cb.grid(row=row, column=col, sticky='w', padx=5)
            self.animal_vars[name] = var
            col += 1
            if col >= 4:
                row += 1
                col = 0

        self.toggle_animal_options()

        self.progress = Progressbar(container, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(5, 0))
        self.progress['value'] = 0

        self.btn_run = tk.Button(container, text="처리 시작", bg="#4CAF50", fg="white", command=self.run_thread)
        self.btn_run.pack(pady=15, fill='x')

    def toggle_animal_options(self):
        if self.extract_target.get() == 'animal':
            for child in self.frame_animals.winfo_children():
                child.configure(state='normal')
        else:
            for child in self.frame_animals.winfo_children():
                child.configure(state='disabled')

    def update_thumbnail(self, event=None):
        url = self.entry_url.get().strip()

        match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if not match:
            self.thumbnail_label.config(image='', text="올바른 유튜브 URL이 아닙니다.")
            return

        video_id = match.group(1)
        thumb_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"

        try:
            with urllib.request.urlopen(thumb_url) as u:
                raw_data = u.read()
            im = Image.open(io.BytesIO(raw_data)).resize((320, 180))
            self.thumbnail_img = ImageTk.PhotoImage(im)
            self.thumbnail_label.config(image=self.thumbnail_img, text="")
        except Exception:
            self.thumbnail_label.config(image='', text="썸네일 로딩 실패")

    def check_url_clear(self, event=None):
        url = self.entry_url.get().strip()
        if url:
            self.btn_choose.config(state='disabled')
            self.btn_clear.config(state='disabled')
        else:
            self.btn_choose.config(state='normal')
            self.btn_clear.config(state='normal')

    def update_progress(self, percent):
        self.progress['value'] = percent
        self.root.update_idletasks()

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if path:
            self.input_path = path
            self.btn_choose.config(text=os.path.basename(path))
            self.entry_url.config(state='disabled')

    def clear_file(self):
        self.input_path = ""
        self.btn_choose.config(text="파일 선택")
        self.entry_url.config(state='normal')

    def run_thread(self):
        self.btn_run.config(state='disabled')
        self.progress['value'] = 0
        self.progress.pack(fill='x', pady=(5, 0))
        threading.Thread(target=self.run_process).start()

    def run_process(self):
        url = self.entry_url.get().strip()
        outp = self.output_path.get().strip() or "output.mp4"
        method = self.combo_method.get()

        if url:
            try:
                url = normalize_youtube_url(url)
                self.temp_file_prefix = tempfile.NamedTemporaryFile(delete=False).name
                inp = download_youtube_video(url, self.temp_file_prefix, 1080)
            except Exception as e:
                self.progress.pack_forget()
                messagebox.showerror("유튜브 다운로드 오류", str(e))
                self.btn_run.config(state='normal')
                return
        elif self.input_path:
            inp = self.input_path
        else:
            self.progress.pack_forget()
            messagebox.showerror("입력 오류", "유튜브 링크나 로컬 파일 중 하나를 선택하세요.")
            self.btn_run.config(state='normal')
            return

        try:
            if method == "YOLOv8" and YOLO_AVAILABLE:
                if self.extract_target.get() == 'person':
                    target_indices = [0]
                else:
                    target_indices = [self.animal_classes[k] for k, v in self.animal_vars.items() if v.get()]
                    if not target_indices:
                        messagebox.showerror("입력 오류", "분리할 동물을 하나 이상 선택하세요.")
                        self.btn_run.config(state='normal')
                        return

                process_with_yoloseg(inp, outp, target_indices=target_indices, progress_callback=self.update_progress)
            else:
                process_with_mediapipe(inp, outp, progress_callback=self.update_progress)

            self.progress['value'] = 100
            messagebox.showinfo("완료", f"처리 완료!\n저장 파일: {outp}")
        except Exception as e:
            messagebox.showerror("처리 오류", str(e))
        finally:
            self.progress.pack_forget()
            self.btn_run.config(state='normal')
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