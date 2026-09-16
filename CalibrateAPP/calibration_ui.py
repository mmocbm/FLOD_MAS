"""
ChArUco Camera Calibration UI
- Two camera selection buttons (0 and 1)
- Calibration files get suffix based on selected camera:
  camera_calibration_0.json / camera_extrinsics_0.json
  camera_calibration_1.json / camera_extrinsics_1.json
- Attempts to use full resolution (2560×1440) for saved images
"""

import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
import random
import os

# ==================== CONFIGURATION ====================

WINDOW_WIDTH  = 1024
WINDOW_HEIGHT = 600

# Preview size (display only)
PREVIEW_WIDTH  = 720
PREVIEW_HEIGHT = 540

# Desired capture resolution
TARGET_WIDTH  = 2560
TARGET_HEIGHT = 1440

# ChArUco board parameters
SQUARE_LENGTH = 15   # mm
MARKER_LENGTH = 11   # mm
SQUARES_X = 11
SQUARES_Y = 17
DICT_TYPE = cv2.aruco.DICT_5X5_1000

NUM_CAPTURES = 20
MIN_CORNERS  = 10

TEMP_DIR = "temp_calibration_images"

# ==================== APPLICATION ====================

class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChArUco Camera Calibration")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(960, 540)

        self.cap = None
        self.camera_running = False
        self.camera_index = -1
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.actual_width = 0
        self.actual_height = 0

        self.captured_images = []
        self.is_calibrating = False

        Path(TEMP_DIR).mkdir(exist_ok=True)
        for f in Path(TEMP_DIR).glob("*.jpg"):
            f.unlink()

        self.init_detector()
        self.create_ui()
        self.update_frame()

    def init_detector(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        self.board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH / 1000.0,
            MARKER_LENGTH / 1000.0,
            aruco_dict
        )
        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params, params)

    def create_ui(self):
        canvas = tk.Canvas(self.root, highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        h_scroll = ttk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=canvas.xview)

        scroll_frame = ttk.Frame(canvas, padding="10")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.root.bind_all("<MouseWheel>", _on_mousewheel)

        main_frame = scroll_frame

        ttk.Label(main_frame, text="ChArUco Camera Calibration",
                  font=('Helvetica', 18, 'bold')).grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)

        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        content_frame.columnconfigure(0, weight=7)
        content_frame.columnconfigure(1, weight=3)
        content_frame.rowconfigure(0, weight=1)

        # Preview area
        video_container = ttk.Frame(content_frame, width=PREVIEW_WIDTH + 20, height=PREVIEW_HEIGHT + 40)
        video_container.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        video_container.grid_propagate(False)

        video_frame = ttk.LabelFrame(video_container, text=" Camera Feed (Preview) ", padding=8)
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_frame,
                                    text="Select camera and start",
                                    bg='black', fg='#cccccc',
                                    font=('Arial', 16), justify="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Controls
        control_frame = ttk.Frame(content_frame)
        control_frame.grid(row=0, column=1, sticky="ns", padx=(8,0))

        cam_frame = ttk.LabelFrame(control_frame, text="Select Camera", padding=12)
        cam_frame.pack(fill=tk.X, pady=(0,12))

        cam_buttons_frame = ttk.Frame(cam_frame)
        cam_buttons_frame.pack(fill=tk.X, pady=8)

        self.btn_cam0 = tk.Button(cam_buttons_frame, text="CAMERA 0", command=lambda: self.select_camera(0),
                                  font=('Arial', 16, 'bold'), bg='#4CAF50', fg='white', width=12, height=2)
        self.btn_cam0.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.btn_cam1 = tk.Button(cam_buttons_frame, text="CAMERA 1", command=lambda: self.select_camera(1),
                                  font=('Arial', 16, 'bold'), bg='#2196F3', fg='white', width=12, height=2)
        self.btn_cam1.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.start_btn = tk.Button(cam_frame, text="START SELECTED CAMERA", command=self.start_camera,
                                   font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white')
        self.start_btn.pack(fill=tk.X, pady=8)

        self.stop_btn = tk.Button(cam_frame, text="STOP CAMERA", command=self.stop_camera,
                                  font=('Arial', 12, 'bold'), bg='#f44336', fg='white', state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=4)

        capture_frame = ttk.LabelFrame(control_frame, text="Capture", padding=10)
        capture_frame.pack(fill=tk.X, pady=(0,10))

        self.capture_info = ttk.Label(capture_frame, text=f"Captured: 0 / {NUM_CAPTURES}",
                                      font=('Arial', 11, 'bold'), foreground='#1976D2')
        self.capture_info.pack(pady=6)

        self.progress = ttk.Progressbar(capture_frame, maximum=NUM_CAPTURES, length=220)
        self.progress.pack(fill=tk.X, pady=6)

        self.capture_btn = tk.Button(capture_frame, text="CAPTURE NOW", command=self.manual_capture,
                                     font=('Arial', 12, 'bold'), bg='#2196F3', fg='white', state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, pady=6)

        ttk.Label(capture_frame, text="Move board to different positions & angles",
                  font=('Arial', 9, 'italic')).pack(pady=4)

        status_frame = ttk.LabelFrame(control_frame, text="Status Log", padding=8)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=12, width=40,
                                                     font=('Consolas', 10), wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        self.log("Select CAMERA 0 or CAMERA 1")
        self.log("Full resolution target: 2560 × 1440")
        self.log("Click START after selecting camera")

    def select_camera(self, index):
        self.camera_index = index
        self.btn_cam0["relief"] = tk.SUNKEN if index == 0 else tk.RAISED
        self.btn_cam1["relief"] = tk.SUNKEN if index == 1 else tk.RAISED
        self.log(f"→ Selected camera: {index}")
        self.start_btn.config(state=tk.NORMAL)

    def log(self, msg):
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)

    def start_camera(self):
        if self.camera_index < 0:
            messagebox.showwarning("No camera selected", "Please select CAMERA 0 or CAMERA 1 first")
            return

        self.start_btn.config(state=tk.DISABLED, text="STARTING...")
        self.log(f"\nOpening camera {self.camera_index} at 2560×1440...")
        threading.Thread(target=self._start_camera_thread, daemon=True).start()

    def _start_camera_thread(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if w < 2000:
                self.log("2560×1440 not supported → trying 1920×1080...")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.actual_width = w
            self.actual_height = h

            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")

            self.camera_running = True
            self.root.after(0, self._camera_started_ok)

        except Exception as e:
            self.root.after(0, self._camera_started_fail, str(e))

    def _camera_started_ok(self):
        self.start_btn.config(state=tk.DISABLED, text="START CAMERA")
        self.stop_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        self.log(f"✓ Camera {self.camera_index} started @ {self.actual_width} × {self.actual_height}")

    def _camera_started_fail(self, err):
        self.start_btn.config(state=tk.NORMAL, text="START CAMERA")
        messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_index}:\n{err}")
        self.log(f"✗ Failed: {err}")

    def stop_camera(self):
        self.camera_running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.video_label.config(image='', text="Camera stopped")
        self.log("Camera stopped")

    def update_frame(self):
        if self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()

                display = frame.copy()
                gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)

                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                    color = (0, 255, 0) if len(ids) >= MIN_CORNERS else (255, 165, 0)
                    cv2.putText(display, f"Corners: {len(ids)}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS)

                new_img = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), (0, 0, 0))
                offset = ((PREVIEW_WIDTH - img.width) // 2, (PREVIEW_HEIGHT - img.height) // 2)
                new_img.paste(img, offset)

                imgtk = ImageTk.PhotoImage(new_img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

        self.root.after(30, self.update_frame)

    def manual_capture(self):
        if self.current_frame is None:
            messagebox.showwarning("No frame", "No camera frame available")
            return

        with self.frame_lock:
            frame = self.current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.charuco_detector.detectBoard(gray)

        if ids is None or len(ids) < MIN_CORNERS:
            cnt = len(ids) if ids is not None else 0
            messagebox.showwarning("Too few corners", f"Need ≥ {MIN_CORNERS}\nDetected: {cnt}")
            self.log(f"✗ Capture rejected – only {cnt} corners")
            return

        count = len(self.captured_images)
        fname = f"{TEMP_DIR}/calib_{count:03d}.jpg"
        cv2.imwrite(fname, frame)

        self.captured_images.append(fname)

        count += 1
        self.capture_info.config(text=f"Captured: {count} / {NUM_CAPTURES}")
        self.progress['value'] = count
        self.log(f"✓ Captured {count}/{NUM_CAPTURES}  ({len(ids)} corners)")
        self.log(f"   Saved @ {frame.shape[1]} × {frame.shape[0]}")

        if count >= NUM_CAPTURES:
            self.capture_btn.config(state=tk.DISABLED, text="COMPLETE")
            self.log("\n" + "═"*60)
            self.log("All captures done → starting calibration")
            self.log("═"*60)
            self.stop_btn.config(state=tk.DISABLED)
            threading.Thread(target=self.auto_calibrate, daemon=True).start()

    def auto_calibrate(self):
        if self.is_calibrating:
            return
        self.is_calibrating = True

        try:
            suffix = f"_{self.camera_index}"
            calib_file = f"Files/camera_calibration{suffix}.json"
            extrinsics_file = f"Files/camera_extrinsics{suffix}.json"

            self.root.after(0, self.log, "\n" + "═"*60)
            self.root.after(0, self.log, f"INTRINSIC CALIBRATION (Camera {self.camera_index})")
            self.root.after(0, self.log, "═"*60)

            all_corners = []
            all_ids = []
            image_size = None

            for i, path in enumerate(self.captured_images):
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)

                if corners is not None and len(corners) >= MIN_CORNERS:
                    all_corners.append(corners)
                    all_ids.append(ids)
                    if image_size is None:
                        image_size = gray.shape[::-1]
                    self.root.after(0, self.log, f" {i+1:2d}: {len(corners)} corners ✓")
                else:
                    self.root.after(0, self.log, f" {i+1:2d}: skipped")

            if len(all_corners) < 8:
                raise RuntimeError(f"Only {len(all_corners)} valid images (need ≥ 8)")

            self.root.after(0, self.log, f"\nCalibrating with {len(all_corners)} images...")

            ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, self.board, image_size, None, None
            )

            calib_data = {
                'camera_matrix': mtx.tolist(),
                'dist_coeffs': dist.tolist(),
                'rms_error': float(ret),
                'image_size': list(image_size)
            }

            with open(calib_file, 'w') as f:
                json.dump(calib_data, f, indent=2)

            self.camera_matrix = mtx
            self.dist_coeffs = dist

            self.root.after(0, self.log, f"RMS error: {ret:.4f} pixels")
            self.root.after(0, self.log, f"Saved intrinsics: {calib_file}")

            # Extrinsics
            self.root.after(0, self.log, "\n" + "═"*60)
            self.root.after(0, self.log, f"EXTRINSIC CALIBRATION (Camera {self.camera_index})")
            self.root.after(0, self.log, "═"*60)

            random.shuffle(self.captured_images)
            ref_path = ref_corners = ref_ids = None

            for path in self.captured_images:
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
                if corners is not None and len(corners) >= MIN_CORNERS:
                    ref_path = path
                    ref_corners = corners
                    ref_ids = ids
                    break

            if ref_path is None:
                raise RuntimeError("No suitable reference image found")

            self.root.after(0, self.log, f"Reference image: {os.path.basename(ref_path)}")

            obj_pts = self.board.getChessboardCorners()[ref_ids.flatten()]
            img_pts = ref_corners.reshape(-1, 2)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                raise RuntimeError("solvePnP failed")

            extrin_data = {
                'rvec': rvec.reshape(-1).tolist(),
                'tvec': tvec.reshape(-1).tolist()
            }

            with open(extrinsics_file, 'w') as f:
                json.dump(extrin_data, f, indent=2)

            self.root.after(0, self.log, f"Extrinsics saved: {extrinsics_file}")
            self.root.after(0, self.log, f"Board distance ≈ {np.linalg.norm(tvec):.3f} m")

            self.root.after(0, messagebox.showinfo, "Success",
                            f"Calibration finished for camera {self.camera_index}!\n\n"
                            f"Files saved:\n"
                            f"• {calib_file}\n"
                            f"• {extrinsics_file}")

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Calibration Error", str(e))
            self.root.after(0, self.log, f"✗ {str(e)}")

        finally:
            self.is_calibrating = False
            self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL, text="CAPTURE NOW"))

    def on_closing(self):
        self.camera_running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()