import cv2
import numpy as np
from measure.makeUndistored import ImageUndistorter


class CameraHandler:
    """Handles single camera: capture, undistortion, and frame management."""
    
    def __init__(self, camera_index, calib_file_path):
        self.camera_index = camera_index
        self.calib_file_path = calib_file_path
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Initialize undistorter
        self.undistorter = ImageUndistorter(calib_file_path)
        
        # Frame storage
        self.current_frame_raw = None
        self.current_frame_undistorted = None
        
    def read_frame(self):
        """Capture frame and create undistorted version."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_raw = frame.copy()
            self.current_frame_undistorted = self.undistorter.undistort(frame, crop=False)
            return True
        return False
    
    def get_raw_frame(self):
        """Get original captured frame (for prediction)."""
        return self.current_frame_raw
    
    def get_undistorted_frame(self):
        """Get undistorted frame (for display and measurement)."""
        return self.current_frame_undistorted
    
    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()

    def get_raw_frame_with_ret(self):
        """Returns a tuple (success, raw_frame) by capturing a new frame."""
        success = self.read_frame()
        return success, self.get_raw_frame()
