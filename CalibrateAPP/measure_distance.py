"""
Distance Measurement Tool
Interactive GUI for selecting points on undistorted images and measuring real-world distances.
"""

import cv2
try:
    from .runtime_config import CONFIG, project_path
except ImportError:
    from runtime_config import CONFIG, project_path
import numpy as np
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Board parameters (must match calibration)
SQUARE_LENGTH = CONFIG['board']['square_length_mm']
MARKER_LENGTH = CONFIG['board']['marker_length_mm']
SQUARES_X = CONFIG['board']['squares_x']
SQUARES_Y = CONFIG['board']['squares_y']

# Calibration files
CALIB_FILE = r"C:\Users\Obhash\Desktop\Factory_Day_14\2\Files\camera_calibration_0.json"
EXTRINSICS_FILE = r"C:\Users\Obhash\Desktop\Factory_Day_14\2\Files\camera_extrinsics_0.json"

class MeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChArUco Distance Measurement Tool")
        
        # Load calibration data
        self.load_calibration()
        
        # State variables
        self.original_image = None
        self.undistorted_image = None
        self.display_image = None
        self.photo_image = None
        self.zoom_level = 1.0
        self.points = []  # Selected points
        self.measurements = []  # List of measurement results
        
        # UI setup
        self.setup_ui()
        
    def load_calibration(self):
        """Load camera calibration data"""
        try:
            # Load intrinsics
            with open(CALIB_FILE, 'r') as f:
                calib_data = json.load(f)
            
            self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
            self.dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)
            
            # Load extrinsics
            with open(EXTRINSICS_FILE, 'r') as f:
                ext_data = json.load(f)
            
            self.rvec = np.array(ext_data['rvec'], dtype=np.float64).reshape(3, 1)
            self.tvec = np.array(ext_data['tvec'], dtype=np.float64).reshape(3, 1)
            
            print("✓ Calibration data loaded successfully")
            
        except FileNotFoundError as e:
            messagebox.showerror("Calibration Error", 
                               f"Calibration file not found!\n\n{str(e)}\n\n"
                               "Please run calibration scripts first.")
            self.root.destroy()
            
    def setup_ui(self):
        """Create GUI layout"""
        
        # Top control panel
        control_frame = tk.Frame(self.root, bg='#2c3e50', pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Load image button
        btn_load = tk.Button(control_frame, text="📁 Load Image", 
                            command=self.load_image, 
                            bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                            padx=20, pady=5)
        btn_load.pack(side=tk.LEFT, padx=10)
        
        # Zoom controls
        tk.Label(control_frame, text="Zoom:", bg='#2c3e50', fg='white', 
                font=('Arial', 11)).pack(side=tk.LEFT, padx=(20,5))
        
        btn_zoom_in = tk.Button(control_frame, text="🔍+", 
                               command=self.zoom_in,
                               bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                               padx=10, pady=5)
        btn_zoom_in.pack(side=tk.LEFT, padx=2)
        
        btn_zoom_out = tk.Button(control_frame, text="🔍−", 
                                command=self.zoom_out,
                                bg='#e67e22', fg='white', font=('Arial', 11, 'bold'),
                                padx=10, pady=5)
        btn_zoom_out.pack(side=tk.LEFT, padx=2)
        
        btn_zoom_reset = tk.Button(control_frame, text="⟲ Reset", 
                                  command=self.zoom_reset,
                                  bg='#95a5a6', fg='white', font=('Arial', 11),
                                  padx=10, pady=5)
        btn_zoom_reset.pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = tk.Label(control_frame, text="100%", 
                                   bg='#2c3e50', fg='white', 
                                   font=('Arial', 11, 'bold'))
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        
        # Clear points button
        btn_clear = tk.Button(control_frame, text="🗑 Clear Points", 
                            command=self.clear_points,
                            bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                            padx=15, pady=5)
        btn_clear.pack(side=tk.LEFT, padx=10)
        
        # Main container with scrollbars
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars for image display
        self.canvas = tk.Canvas(main_container, bg='#34495e', cursor="crosshair")
        
        scrollbar_y = tk.Scrollbar(main_container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = tk.Scrollbar(main_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Right panel for measurements
        right_panel = tk.Frame(self.root, bg='#ecf0f1', width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Title
        tk.Label(right_panel, text="Measurements", 
                bg='#ecf0f1', font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Info labels
        self.info_label = tk.Label(right_panel, text="Load an image to start", 
                                   bg='#ecf0f1', font=('Arial', 10),
                                   justify=tk.LEFT, wraplength=280)
        self.info_label.pack(pady=5, padx=10)
        
        # Measurements list
        list_frame = tk.Frame(right_panel, bg='#ecf0f1')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.measurements_listbox = tk.Listbox(list_frame, 
                                               yscrollcommand=scrollbar.set,
                                               font=('Courier', 10),
                                               bg='white')
        self.measurements_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.measurements_listbox.yview)
        
        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Load an image\n"
            "2. Click 2 points to measure\n"
            "3. Use zoom for precision\n"
            "4. Results appear on right"
        )
        tk.Label(right_panel, text=instructions, 
                bg='#ecf0f1', font=('Arial', 9), 
                justify=tk.LEFT).pack(side=tk.BOTTOM, pady=10, padx=10)
        
    def load_image(self):
        """Load and undistort image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Load original image
        self.original_image = cv2.imread(file_path)
        
        if self.original_image is None:
            messagebox.showerror("Error", "Failed to load image!")
            return
        
        # Undistort image
        self.undistorted_image = cv2.undistort(self.original_image, 
                                              self.camera_matrix, 
                                              self.dist_coeffs)
        
        # Reset state
        self.points = []
        self.zoom_level = 1.0
        
        # Display
        self.update_display()
        
        self.info_label.config(
            text=f"Image loaded: {os.path.basename(file_path)}\n"
                 f"Size: {self.undistorted_image.shape[1]}x{self.undistorted_image.shape[0]}\n"
                 f"Click 2 points to measure"
        )
        
    def update_display(self):
        """Update canvas with current image and zoom"""
        if self.undistorted_image is None:
            return
        
        # Create display image with points drawn
        img = self.undistorted_image.copy()
        
        # Draw selected points
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), (x+12, y-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Draw line if 2 points
        if len(self.points) == 2:
            cv2.line(img, self.points[0], self.points[1], (255, 0, 0), 3)
        
        # Apply zoom
        h, w = img.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        
    def zoom_in(self):
        """Zoom in"""
        if self.undistorted_image is not None:
            self.zoom_level = min(self.zoom_level * 1.25, 5.0)
            self.update_display()
    
    def zoom_out(self):
        """Zoom out"""
        if self.undistorted_image is not None:
            self.zoom_level = max(self.zoom_level / 1.25, 0.2)
            self.update_display()
    
    def zoom_reset(self):
        """Reset zoom to 100%"""
        if self.undistorted_image is not None:
            self.zoom_level = 1.0
            self.update_display()
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas"""
        if self.undistorted_image is None:
            return
        
        if len(self.points) >= 2:
            messagebox.showinfo("Info", "2 points already selected. Clear to start new measurement.")
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Account for zoom
        img_x = int(canvas_x / self.zoom_level)
        img_y = int(canvas_y / self.zoom_level)
        
        # Check bounds
        h, w = self.undistorted_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            self.points.append((img_x, img_y))
            print(f"Point {len(self.points)}: ({img_x}, {img_y})")
            
            self.update_display()
            
            # If 2 points selected, calculate distance
            if len(self.points) == 2:
                self.calculate_distance()
    
    def calculate_distance(self):
        """Calculate and display distance between two points"""
        if len(self.points) != 2:
            return
        
        try:
            # Project points to 3D plane
            p1_3d = self.image_point_to_plane(self.points[0])
            p2_3d = self.image_point_to_plane(self.points[1])
            
            # Calculate distance
            distance_m = np.linalg.norm(p1_3d - p2_3d)
            distance_mm = distance_m * 1000
            distance_cm = distance_mm / 10
            
            # Add to measurements list
            result = f"P1:{self.points[0]} P2:{self.points[1]} | {distance_mm:.2f}mm ({distance_cm:.2f}cm)"
            self.measurements.append(result)
            self.measurements_listbox.insert(tk.END, result)
            
            # Update info
            self.info_label.config(
                text=f"Measurement #{len(self.measurements)}:\n"
                     f"Distance: {distance_mm:.2f} mm\n"
                     f"         = {distance_cm:.2f} cm\n"
                     f"         = {distance_m:.4f} m"
            )
            
            # Draw result on image
            img_with_result = self.undistorted_image.copy()
            
            for i, (x, y) in enumerate(self.points):
                cv2.circle(img_with_result, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(img_with_result, str(i+1), (x+12, y-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            cv2.line(img_with_result, self.points[0], self.points[1], (255, 0, 0), 3)
            
            mid_x = (self.points[0][0] + self.points[1][0]) // 2
            mid_y = (self.points[0][1] + self.points[1][1]) // 2
            
            text = f"{distance_mm:.1f} mm"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(img_with_result,
                         (mid_x - text_size[0]//2 - 10, mid_y - text_size[1] - 20),
                         (mid_x + text_size[0]//2 + 10, mid_y + 10),
                         (0, 0, 0), -1)
            cv2.putText(img_with_result, text, (mid_x - text_size[0]//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            self.undistorted_image = img_with_result
            self.update_display()
            
            print(f"Distance: {distance_mm:.2f} mm = {distance_cm:.2f} cm")
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Failed to calculate distance:\n{str(e)}")
    
    def image_point_to_plane(self, img_point):
        """Project image point to 3D measurement plane"""
        # No distortion since image is already undistorted
        no_distortion = np.zeros((5, 1), dtype=np.float64)
        
        # Get rotation matrix and plane normal
        R, _ = cv2.Rodrigues(self.rvec)
        plane_normal = R[:, 2]
        d = -plane_normal.dot(self.tvec.flatten())
        
        # Undistort to normalized coordinates
        pts = np.array(img_point, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, no_distortion, P=None)
        x, y = undistorted[0, 0, 0], undistorted[0, 0, 1]
        
        # Ray from camera
        ray_direction = np.array([x, y, 1.0])
        
        # Intersect with plane
        denominator = plane_normal.dot(ray_direction)
        if abs(denominator) < 1e-9:
            raise ValueError("Ray parallel to measurement plane")
        
        s = -d / denominator
        point_camera = s * ray_direction
        
        # Transform to board coordinates
        obj_xy = R[:, :2].T.dot(point_camera - self.tvec.flatten())
        
        return np.array([obj_xy[0], obj_xy[1], 0.0])
    
    def clear_points(self):
        """Clear selected points"""
        self.points = []
        
        # Reload undistorted image without annotations
        if self.original_image is not None:
            self.undistorted_image = cv2.undistort(self.original_image, 
                                                  self.camera_matrix, 
                                                  self.dist_coeffs)
            self.update_display()
            self.info_label.config(text="Points cleared. Click 2 new points.")

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = MeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
