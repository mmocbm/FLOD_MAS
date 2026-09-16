import cv2
import numpy as np
import json
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os

# ================= CONFIG =================

CAMERA_INDEX = 0
UI_WIDTH = 1024
UI_HEIGHT = 600
PREVIEW_HEIGHT = UI_HEIGHT - 120

# ChArUco board parameters
SQUARE_LENGTH = 15  # mm
MARKER_LENGTH = 11  # mm
SQUARES_X = 11
SQUARES_Y = 17

CALIB_FILE = "Files/camera_calibration_0.json"
EXTRINSICS_FILE = "Files/camera_extrinsics_0.json"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================


def resize_keep_aspect(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def image_point_to_plane(img_point, K, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    plane_normal = R[:, 2]
    d = -plane_normal.dot(tvec.flatten())

    pts = np.array(img_point, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, None)
    x, y = und[0, 0]

    ray = np.array([x, y, 1.0])
    s = -d / plane_normal.dot(ray)
    P = s * ray

    obj_xy = R[:, :2].T.dot(P - tvec.flatten())
    return np.array([obj_xy[0], obj_xy[1], 0.0])


def measure_distance(p1, p2, K, rvec, tvec):
    P1 = image_point_to_plane(p1, K, rvec, tvec)
    P2 = image_point_to_plane(p2, K, rvec, tvec)
    return np.linalg.norm(P1 - P2) * 1000


def ground_truth(id1, id2):
    cols = SQUARES_X - 1
    r1, c1 = divmod(id1, cols)
    r2, c2 = divmod(id2, cols)
    return np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2) * SQUARE_LENGTH


class CharucoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChArUco Accuracy Checker")
        self.root.geometry(f"{UI_WIDTH}x{UI_HEIGHT}")
        self.root.resizable(False, False)

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack()

        self.capture_btn = tk.Button(
            root, text="CAPTURE",
            font=("Arial", 18),
            bg="#2ecc71",
            fg="white",
            height=2,
            command=self.capture
        )
        self.capture_btn.pack(fill=tk.X)

        self.status = tk.Label(root, text="Camera Ready",
                               font=("Arial", 14))
        self.status.pack()

        self.load_calibration()
        self.init_charuco()
        self.open_camera()

        self.running = True
        self.update_frame()

    # ---------- Initialization ----------

    def load_calibration(self):
        with open(CALIB_FILE) as f:
            calib = json.load(f)
        with open(EXTRINSICS_FILE) as f:
            extr = json.load(f)

        self.K = np.array(calib["camera_matrix"])
        self.dist = np.array(calib["dist_coeffs"])
        self.rvec = np.array(extr["rvec"]).reshape(3, 1)
        self.tvec = np.array(extr["tvec"]).reshape(3, 1)

    def init_charuco(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_1000
        )
        self.board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH / 1000,
            MARKER_LENGTH / 1000,
            aruco_dict
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def open_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        # Force maximum resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened")

    # ---------- Live Preview ----------

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.live_frame = frame.copy()
            preview = resize_keep_aspect(
                frame, UI_WIDTH, PREVIEW_HEIGHT
            )

            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.root.after(10, self.update_frame)

    # ---------- Capture + Accuracy ----------

    def capture(self):
        self.status.config(text="Processing...", fg="blue")
        threading.Thread(target=self.process).start()

    def process(self):
        img = self.live_frame.copy()
        und = cv2.undistort(img, self.K, self.dist)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        corners, ids, _, _ = self.detector.detectBoard(gray)

        if ids is None or len(ids) < 10:
            self.status.config(
                text="❌ ChArUco not detected",
                fg="red"
            )
            return

        ids = ids.flatten()
        errors = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                gt = ground_truth(ids[i], ids[j])
                if gt < 50:
                    continue

                d = measure_distance(
                    corners[i][0],
                    corners[j][0],
                    self.K, self.rvec, self.tvec
                )
                errors.append(abs(d - gt))

        mean_err = np.mean(errors)
        rmse = np.sqrt(np.mean(np.square(errors)))

        if mean_err < 0.5:
            result, color = "GOOD", "green"
        elif mean_err < 1.5:
            result, color = "ACCEPTABLE", "orange"
        else:
            result, color = "POOR", "red"

        cv2.aruco.drawDetectedCornersCharuco(
            und, corners, ids
        )

        out = os.path.join(
            OUTPUT_DIR, f"accuracy_{int(time.time())}.jpg"
        )
        cv2.imwrite(out, und)

        self.status.config(
            text=f"{result} | Mean: {mean_err:.2f} mm | RMSE: {rmse:.2f} mm",
            fg=color
        )

    # ---------- Cleanup ----------

    def close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CharucoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
