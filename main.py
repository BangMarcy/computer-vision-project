import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import threading
import easyocr
#Khởi tạo giao diện
class LicensePlateDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hệ thống nhận diện biển số v1.2 - AUTO LIVE")
        self.root.geometry("1200x800")

        # --- CẤU HÌNH MODEL ---
        self.model_path = "tan1.pt" # Đảm bảo file model đúng tên
        self.model = None
        self.reader = None
        
        self.video_running = False
        self.cap = None
        self.current_crops = []   # Giữ tham chiếu ảnh

        self.setup_ui()

        # Tải model ngầm
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        try:
            self.update_status("Đang tải YOLO model...", "orange")
            self.model = YOLO(self.model_path)
            print("--- YOLO Model loaded! ---")

            self.update_status("Đang tải EasyOCR...", "orange")
            # Thử load GPU, nếu không được thì về CPU
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
                print("--- EasyOCR (GPU) loaded! ---")
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("--- EasyOCR (CPU) loaded! ---")

            self.update_status("Hệ thống sẵn sàng!", "green")
            self.set_buttons_state(normal=True)
        except Exception as e:
            self.update_status(f"Lỗi khởi tạo: {e}", "red")
            messagebox.showerror("Lỗi", f"Không thể tải model: {e}")

    def setup_ui(self):
        main_layout = ttk.Frame(self.root)
        main_layout.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- TOP: Buttons ---
        btn_frame = ttk.Frame(main_layout)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.img_btn = ttk.Button(btn_frame, text="📂 Chọn Ảnh", command=self.select_image, state="disabled")
        self.img_btn.pack(side=tk.LEFT, padx=5)
        self.video_btn = ttk.Button(btn_frame, text="🎥 Chọn Video", command=self.select_video, state="disabled")
        self.video_btn.pack(side=tk.LEFT, padx=5)
        self.cam_btn = ttk.Button(btn_frame, text="📹 Camera Live", command=self.start_camera, state="disabled")
        self.cam_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Dừng", command=self.stop_video, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # --- MIDDLE: Content ---
        content = ttk.PanedWindow(main_layout, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, pady=5)

        # LEFT: Main Display
        left_frame = ttk.LabelFrame(content, text="Màn hình chính")
        content.add(left_frame, weight=3)
        self.img_canvas = tk.Canvas(left_frame, bg="#e1e1e1")
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        # RIGHT: Results
        right_frame = ttk.LabelFrame(content, text="Kết quả (Live)", width=350)
        content.add(right_frame, weight=1)

        self.canvas_crops = tk.Canvas(right_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.canvas_crops.yview)
        self.crop_container = ttk.Frame(self.canvas_crops)
        self.crop_container.bind("<Configure>", lambda e: self.canvas_crops.configure(scrollregion=self.canvas_crops.bbox("all")))
        self.canvas_crops.create_window((0, 0), window=self.crop_container, anchor="nw")
        self.canvas_crops.configure(yscrollcommand=scrollbar.set)
        self.canvas_crops.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- BOTTOM: Status Bar ---
        self.status_var = tk.StringVar(value="Đang khởi tạo...")
        self.status_bar = tk.Label(main_layout, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#d9d9d9")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # === CÁC HÀM XỬ LÝ CHÍNH ===

    def perform_ocr_advanced(self, crop_img):
        """Hàm OCR: Phóng to -> Gray -> Bilateral -> Read -> SỬA LỖI NHẦM LẪN"""
        if self.reader is None: return "..."
        try:
            #  Tiền xử lý ảnh (Làm rõ ảnh trước khi đọc)
            img_big = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            # Chỉ cho phép đọc các ký tự này (giúp giảm sai sót đọc nhầm ký tự lạ)
            allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
            results = self.reader.readtext(gray, detail=0, allowlist=allowlist, decoder='beamsearch', adjust_contrast=0.5)
            
            #  Làm sạch chuỗi kết quả
            text = "".join(results).upper().replace(" ", "").replace("-", "").replace(".", "")
            
            #  GỌI HÀM SỬA LỖI
            if len(text) > 3:
                text = self.heuristic_correction(text)

            return text if len(text) > 3 else None
        except Exception:
            return None
        
    def heuristic_correction(self, text):
        # Dictionary chuyển Chữ -> Số (Dùng cho vị trí nên là Số)
        dict_char_to_int = {
            'B': '8', 'G': '6', 'D': '0', 'O': '0', 'Q': '0', 
            'I': '1', 'S': '5', 'Z': '2', 'A': '4' 
        }
        
        # Dictionary chuyển Số -> Chữ (Dùng cho vị trí nên là Chữ)
        dict_int_to_char = {
            '8': 'B', '6': 'G', '0': 'D', '5': 'S', '2': 'Z', '1': 'I', '4': 'A'
        }

        text_list = list(text)
        length = len(text_list)

        for i in range(length):
            char = text_list[i]

            # QUY TẮC 1: 2 ký tự đầu tiên (Mã tỉnh) -> BẮT BUỘC LÀ SỐ
            if i in [0, 1]:
                if char in dict_char_to_int:
                    text_list[i] = dict_char_to_int[char]

            # QUY TẮC 2: Ký tự thứ 3 (Series) -> BẮT BUỘC LÀ CHỮ
            elif i == 2:
                if char in dict_int_to_char:
                    text_list[i] = dict_int_to_char[char]

            # QUY TẮC 3: Các ký tự còn lại (Thường là số)
            # Lưu ý: Xe máy có thể có chữ ở vị trí thứ 4 (VD: 59-T1), nhưng đa phần OCR sẽ đọc cả cụm
            # Ở đây ta ưu tiên đổi về Số cho phần đuôi
            elif i > 2:
                if char in dict_char_to_int:
                    text_list[i] = dict_char_to_int[char]

        return "".join(text_list)

    def process_static_image(self, img_bgr):
        """Xử lý ảnh tĩnh (chọn file)"""
        try:
            self.update_status("Đang nhận diện...", "orange")
            self.clear_right_panel()

            # YOLO Detect
            results = self.model(img_bgr)
            result = results[0]
            self.display_image_on_canvas(result.plot()) # Vẽ box

            # Crop & OCR
            boxes = result.boxes.cpu().numpy()
            if len(boxes) == 0:
                ttk.Label(self.crop_container, text="Không tìm thấy biển số").pack(pady=10)
            else:
                for i, box in enumerate(boxes):
                    r = box.xyxy[0].astype(int)
                    crop = img_bgr[r[1]:r[3], r[0]:r[2]]
                    if crop.size == 0: continue
                    text = self.perform_ocr_advanced(crop)
                    if text:
                        self.display_crop_result(crop, i+1, text)
                    else:
                        self.display_crop_result(crop, i+1, "Không rõ")

            self.update_status(f"Hoàn thành: {len(boxes)} biển số", "green")
        except Exception as e:
            self.update_status("Lỗi xử lý ảnh", "red")
            messagebox.showerror("Lỗi", str(e))

    # === UPDATE UI ===
    def display_image_on_canvas(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 100: cw, ch = 800, 600
        
        ratio = min(cw/w, ch/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        self.photo = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def display_crop_result(self, crop_img, idx, text):
        """Hiển thị ảnh crop và text lên panel phải"""
        # Tạo một khung nhỏ chứa ảnh cắt và text kết quả
        frame = ttk.Frame(self.crop_container, relief="ridge", borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Resize ảnh nhỏ lại để vừa thanh bên phải 
        h, w = crop_img.shape[:2]
        ratio = 150 / w if w > 0 else 1
        new_h = int(h * ratio)
        crop_rgb = cv2.cvtColor(cv2.resize(crop_img, (150, new_h)), cv2.COLOR_BGR2RGB)
        
        photo = ImageTk.PhotoImage(Image.fromarray(crop_rgb))
        self.current_crops.append(photo)
        
        # Thêm ảnh và text vào khung
        ttk.Label(frame, image=photo).pack(pady=5)
        lbl = ttk.Label(frame, text=f"#{idx}: {text}", font=("Arial", 14, "bold"), foreground="blue")
        lbl.pack(pady=(0, 5))

    def update_status(self, text, color="black"):
        self.status_var.set(text)
        self.status_bar.config(fg=color)
        self.root.update_idletasks()

    def clear_right_panel(self):
        for widget in self.crop_container.winfo_children(): widget.destroy()
        self.current_crops.clear()

    def set_buttons_state(self, normal=True):
        state = "normal" if normal else "disabled"
        self.img_btn.config(state=state)
        self.video_btn.config(state=state)
        self.cam_btn.config(state=state)

    # === SỰ KIỆN NÚT BẤM ===
    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path: threading.Thread(target=lambda: self.process_static_image(cv2.imread(path))).start()

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")])
        if path: self.start_video(path)

    def start_camera(self):
        self.start_video(0)

    def start_video(self, source):
        if self.video_running: self.stop_video()
        self.video_running = True
        self.stop_btn.config(state="normal")
        self.set_buttons_state(normal=False)
        
        self.clear_right_panel()
        ttk.Label(self.crop_container, text="Đang nhận dạng tự động...", font=("Arial", 10, "italic")).pack(pady=10)
        
        threading.Thread(target=self.video_loop, args=(source,), daemon=True).start()

    def video_loop(self, source):
        """Vòng lặp video với Auto OCR""" #Optical Character Recognition( nhận dạng ký tự quang học)
        try:
            self.cap = cv2.VideoCapture(source) 
            frame_count = 0
            process_interval = 30  # Chạy OCR mỗi 30 frame (khoảng 1 giây) để giảm lag

            while self.video_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame_count += 1
                
                # 1. YOLO Detection (Chạy mỗi frame để vẽ box)
                results = self.model(frame, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()

                # Hiển thị video lên giao diện (dùng after để an toàn luồng)
                self.root.after(0, lambda f=annotated_frame: self.display_image_on_canvas(f))

                # 2. AUTO OCR (Chạy theo chu kỳ interval)
                if frame_count % process_interval == 0:
                    self.auto_ocr_process(results, frame)

        finally:
            self.stop_video()

    def auto_ocr_process(self, results, original_frame):
        """Hàm phụ trợ để xử lý OCR trong luồng video"""
        boxes = results[0].boxes.cpu().numpy()
        
        # Chuẩn bị dữ liệu cập nhật UI
        found_plates = []
        for box in boxes:
            r = box.xyxy[0].astype(int)
            crop = original_frame[r[1]:r[3], r[0]:r[2]]
            if crop.size == 0: continue
            
            text = self.perform_ocr_advanced(crop)
            if text:
                found_plates.append((crop, text))

        # Cập nhật UI (phải dùng root.after vì đang ở thread khác)
        if found_plates:
            self.root.after(0, lambda: self.update_panel_live(found_plates))

    def update_panel_live(self, plates_data):
        """Cập nhật panel phải khi có kết quả mới từ video"""
        self.clear_right_panel() # Xóa kết quả cũ để hiển thị cái mới nhất
        for i, (crop, text) in enumerate(plates_data):
            self.display_crop_result(crop, i+1, text)

    def stop_video(self):
        self.video_running = False
        if self.cap: self.cap.release()
        self.root.after(0, lambda: [
            self.stop_btn.config(state="disabled"),
            self.set_buttons_state(normal=True),
            self.update_status("Đã dừng video", "black")
        ])

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    LicensePlateDetector().run()