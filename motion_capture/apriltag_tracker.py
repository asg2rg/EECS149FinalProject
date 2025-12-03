import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
import math
import time
import threading
from collections import deque
from datetime import datetime
import os

# ==================== CONFIGURATION ====================
# Camera ID (0 = built-in webcam, 1 = USB camera)
CAMERA_ID = 1

# Camera resolution - Significantly reduced for maximum speed
# Try (640, 480), (320, 240), or (424, 240) for different speed/accuracy tradeoffs
IMAGE_RES = (640, 480)  # Lower resolution = faster processing (was 1280x720)

# Path to camera calibration file
CALIBRATION_FILE = 'camera_calibration.yaml'

# AprilTag family (options: 'tag36h11', 'tag25h9', 'tag16h5', 'tagCircle21h7', 'tagStandard41h12')
# tag16h5 is MUCH faster than tag36h11 but slightly less robust
TAG_FAMILY = 'tag16h5'

# AprilTag size in meters (physical size of the tag)
TAG_SIZE = 0.023  # 23mm = 0.023 meters

# Detection quality threshold (minimum decision margin)
# Higher values = fewer false positives but may miss some valid detections
# Typical range: 20-100. Lower for permissive, higher for strict.
MIN_DECISION_MARGIN = 50.0  # Reject detections with decision_margin below this

# Display settings
ENABLE_VISUAL_OUTPUT = False  # Disabled for maximum performance
WINDOW_NAME = 'AprilTag Tracker'
PERFORMANCE_WINDOW = 'Performance Monitor'

# Target FPS
TARGET_FPS = 60
# =======================================================


def load_camera_calibration(calibration_file):
    """
    Load camera calibration parameters from YAML file.
    
    Args:
        calibration_file (str): Path to calibration YAML file
        
    Returns:
        tuple: (camera_matrix, dist_coeffs) as numpy arrays
    """
    try:
        with open(calibration_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['distortion_coefficients'])
        
        print("Camera calibration loaded successfully!")
        print(f"Focal length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
        print(f"Principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
        
        return camera_matrix, dist_coeffs
        
    except FileNotFoundError:
        print(f"Error: Calibration file '{calibration_file}' not found!")
        print("Please run camera_calibration.py first.")
        exit(1)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        exit(1)


def try_open_camera(cam_id, backend):
    """
    Try to open camera with specified backend.
    
    Args:
        cam_id (int): Camera ID
        backend: OpenCV backend (e.g., cv2.CAP_DSHOW)
        
    Returns:
        cv2.VideoCapture or None
    """
    cap = cv2.VideoCapture(cam_id, backend)
    if not cap.isOpened():
        return None
    return cap


def initialize_camera(camera_id, resolution):
    """
    Initialize camera with fallback to different backends.
    Optimized for high-speed capture with reduced motion blur.
    
    Args:
        camera_id (int): Camera ID
        resolution (tuple): Desired resolution (width, height)
        
    Returns:
        cv2.VideoCapture: Opened camera capture object
    """
    print(f"Opening camera {camera_id}...")
    
    # Try DSHOW first - more reliable on Windows
    cap = try_open_camera(camera_id, cv2.CAP_DSHOW)
    if cap is None:
        cap = try_open_camera(camera_id, cv2.CAP_MSMF)
        if cap is None:
            cap = cv2.VideoCapture(camera_id)
    
    if cap is None or not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        exit(1)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Set FPS to maximum (60 fps)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # Disable MJPEG compression for faster processing
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Reduce exposure for less motion blur (lower values = faster shutter)
    # Try to set exposure manually (negative values are typically used)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Balanced exposure (was -8, going back to -7)
    
    # Increase brightness and gain to compensate for lower exposure
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.65)
    cap.set(cv2.CAP_PROP_GAIN, 60)  # Slightly increased gain for better contrast
    cap.set(cv2.CAP_PROP_CONTRAST, 1.2)  # Increase contrast for better edge detection
    
    # Disable auto white balance for consistency and speed
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    # Set buffer size to 1 to always get latest frame (critical for low latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    backend_name = cap.getBackendName()
    
    print(f"\n=== CAMERA DIAGNOSTICS ===")
    print(f"Backend: {backend_name}")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"FPS (reported): {actual_fps}")
    print(f"Exposure: {actual_exposure}")
    print(f"FOURCC: {actual_fourcc}")
    print(f"Buffer size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    print(f"=========================\n")
    
    # Warning if FPS is below target
    if actual_fps < TARGET_FPS:
        print(f"WARNING: Camera limited to {actual_fps} FPS (hardware limitation)")
        print(f"Target was {TARGET_FPS} FPS - this is likely a camera hardware limit.\n")
    
    return cap


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Uses the XYZ convention (roll-pitch-yaw).
    
    Args:
        R (numpy.ndarray): 3x3 rotation matrix
        
    Returns:
        tuple: (roll, pitch, yaw) in degrees
    """
    # Check for gimbal lock
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0
    
    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg


def estimate_pose(detection, camera_matrix, tag_size):
    """
    Estimate 3D pose of detected AprilTag.
    
    Args:
        detection: AprilTag detection object
        camera_matrix (numpy.ndarray): Camera intrinsic matrix
        tag_size (float): Physical size of tag in meters
        
    Returns:
        tuple: (position, orientation) where position is (x, y, z) in meters
               and orientation is (roll, pitch, yaw) in degrees
    """
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Get pose estimation from detection
    # The detection object contains pose_R (rotation) and pose_t (translation)
    pose_t = detection.pose_t  # Translation vector (x, y, z)
    pose_R = detection.pose_R  # Rotation matrix (3x3)
    
    # Position in meters
    x = pose_t[0, 0]
    y = pose_t[1, 0]
    z = pose_t[2, 0]
    
    # Convert rotation matrix to Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_R)
    
    return (x, y, z), (roll, pitch, yaw)


def draw_tag_info(frame, detection, position, orientation):
    """
    Draw tag outline, axes, and pose information on the frame.
    
    Args:
        frame (numpy.ndarray): Video frame
        detection: AprilTag detection object
        position (tuple): (x, y, z) position in meters
        orientation (tuple): (roll, pitch, yaw) in degrees
    """
    # Draw tag outline
    corners = detection.corners.astype(int)
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # Draw tag ID at center
    center = tuple(detection.center.astype(int))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.putText(frame, f"ID: {detection.tag_id}", 
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display pose information on frame
    x, y, z = position
    roll, pitch, yaw = orientation
    
    text_y = 30
    line_height = 25
    
    cv2.putText(frame, f"Tag ID: {detection.tag_id}", 
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    text_y += line_height
    cv2.putText(frame, f"Position (m): X={x:.3f} Y={y:.3f} Z={z:.3f}", 
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    text_y += line_height
    cv2.putText(frame, f"Orientation (deg): R={roll:.1f} P={pitch:.1f} Y={yaw:.1f}", 
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_pose_info(detection, position, orientation):
    """
    Print pose information to terminal.
    
    Args:
        detection: AprilTag detection object
        position (tuple): (x, y, z) position in meters
        orientation (tuple): (roll, pitch, yaw) in degrees
    """
    x, y, z = position
    roll, pitch, yaw = orientation
    
    print(f"\n{'='*70}")
    print(f"Tag ID: {detection.tag_id}")
    print(f"Quality (decision_margin): {detection.decision_margin:.1f}")
    print(f"{'='*70}")
    print(f"Position (meters):")
    print(f"  X: {x:7.3f} m  (left/right relative to camera)")
    print(f"  Y: {y:7.3f} m  (up/down relative to camera)")
    print(f"  Z: {z:7.3f} m  (distance from camera)")
    print(f"\nOrientation (degrees):")
    print(f"  Roll:  {roll:7.1f}°  (rotation around Z-axis)")
    print(f"  Pitch: {pitch:7.1f}°  (rotation around Y-axis)")
    print(f"  Yaw:   {yaw:7.1f}°  (rotation around X-axis)")
    print(f"{'='*70}")


class MeasurementLogger:
    """
    Logger to store all measurements during tracking session.
    Saves to Excel file after session ends.
    """
    def __init__(self):
        self.measurements = []
        self.start_time = time.time()
        
    def log_measurement(self, tag_id, position, orientation, timestamp_offset):
        """
        Log a single measurement.
        
        Args:
            tag_id: AprilTag ID
            position: (x, y, z) tuple in meters
            orientation: (roll, pitch, yaw) tuple in degrees
            timestamp_offset: Time offset from start in seconds
        """
        x, y, z = position
        roll, pitch, yaw = orientation
        
        self.measurements.append({
            'timestamp': timestamp_offset,
            'tag_id': tag_id,
            'x_m': x,
            'y_m': y,
            'z_m': z,
            'roll_deg': roll,
            'pitch_deg': pitch,
            'yaw_deg': yaw
        })
    
    def save_to_excel(self, output_dir='reports'):
        """
        Save all measurements to Excel file.
        
        Args:
            output_dir: Directory to save the report (relative to project root)
        """
        if not self.measurements:
            print("No measurements to save.")
            return None
        
        try:
            import pandas as pd
            
            # Create reports directory if it doesn't exist
            report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
            os.makedirs(report_path, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apriltag_measurements_{timestamp_str}.xlsx"
            filepath = os.path.join(report_path, filename)
            
            # Convert measurements to DataFrame
            df = pd.DataFrame(self.measurements)
            
            # Create Excel writer with formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Measurements', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': [
                        'Total Measurements',
                        'Unique Tags Detected',
                        'Duration (seconds)',
                        'Average Rate (measurements/sec)',
                        'Start Time',
                        'End Time'
                    ],
                    'Value': [
                        len(self.measurements),
                        len(df['tag_id'].unique()),
                        f"{df['timestamp'].max():.2f}",
                        f"{len(self.measurements) / df['timestamp'].max():.2f}",
                        datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.fromtimestamp(self.start_time + df['timestamp'].max()).strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"\n{'='*70}")
            print(f"Measurements saved to: {filepath}")
            print(f"Total measurements: {len(self.measurements)}")
            print(f"{'='*70}")
            
            return filepath
            
        except ImportError:
            print("\n⚠ Warning: pandas not installed. Installing...")
            print("Run: pip install pandas openpyxl")
            return self.save_to_csv_fallback(output_dir)
        except Exception as e:
            print(f"\n⚠ Error saving to Excel: {e}")
            return self.save_to_csv_fallback(output_dir)
    
    def save_to_csv_fallback(self, output_dir='reports'):
        """
        Fallback: Save to CSV if Excel writing fails.
        """
        try:
            import csv
            
            # Create reports directory
            report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
            os.makedirs(report_path, exist_ok=True)
            
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apriltag_measurements_{timestamp_str}.csv"
            filepath = os.path.join(report_path, filename)
            
            # Write CSV
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'tag_id', 'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.measurements)
            
            print(f"\n{'='*70}")
            print(f"Measurements saved to CSV: {filepath}")
            print(f"Total measurements: {len(self.measurements)}")
            print(f"{'='*70}")
            
            return filepath
            
        except Exception as e:
            print(f"\n⚠ Error saving to CSV: {e}")
            return None


class PipelineProfiler:
    """
    Profile each stage of the detection pipeline to identify bottlenecks.
    """
    def __init__(self, window_size=30):
        self.timings = {
            'frame_capture': deque(maxlen=window_size),
            'preprocessing': deque(maxlen=window_size),
            'detection': deque(maxlen=window_size),
            'pose_estimation': deque(maxlen=window_size),
            'total_loop': deque(maxlen=window_size)
        }
        self.sample_count = 0
        
    def record_timing(self, stage, duration_ms):
        """Record timing for a specific stage."""
        self.timings[stage].append(duration_ms)
        
    def get_avg_timing(self, stage):
        """Get average timing for a stage."""
        if not self.timings[stage]:
            return 0.0
        return sum(self.timings[stage]) / len(self.timings[stage])
    
    def get_bottleneck_analysis(self):
        """Identify the biggest bottleneck in the pipeline."""
        avgs = {stage: self.get_avg_timing(stage) for stage in self.timings.keys() if stage != 'total_loop'}
        total = sum(avgs.values())
        
        if total == 0:
            return None
            
        bottleneck = max(avgs.items(), key=lambda x: x[1])
        percentages = {stage: (time_ms / total * 100) for stage, time_ms in avgs.items()}
        
        return {
            'bottleneck_stage': bottleneck[0],
            'bottleneck_time_ms': bottleneck[1],
            'percentages': percentages,
            'total_time_ms': total
        }
    
    def print_profile_report(self):
        """Print detailed profiling report."""
        analysis = self.get_bottleneck_analysis()
        if not analysis:
            return
            
        print("\n" + "="*70)
        print("PIPELINE PROFILING REPORT")
        print("="*70)
        print(f"\nAverage Timings (based on last {len(self.timings['total_loop'])} frames):")
        print(f"  Frame Capture:    {self.get_avg_timing('frame_capture'):6.2f} ms ({analysis['percentages']['frame_capture']:5.1f}%)")
        print(f"  Preprocessing:    {self.get_avg_timing('preprocessing'):6.2f} ms ({analysis['percentages']['preprocessing']:5.1f}%)")
        print(f"  Tag Detection:    {self.get_avg_timing('detection'):6.2f} ms ({analysis['percentages']['detection']:5.1f}%)")
        print(f"  Pose Estimation:  {self.get_avg_timing('pose_estimation'):6.2f} ms ({analysis['percentages']['pose_estimation']:5.1f}%)")
        print(f"  {'─'*40}")
        print(f"  Total Pipeline:   {analysis['total_time_ms']:6.2f} ms")
        print(f"  Total Loop Time:  {self.get_avg_timing('total_loop'):6.2f} ms")
        print(f"\n  Theoretical Max FPS: {1000/self.get_avg_timing('total_loop'):.1f}")
        print(f"\n🔍 BOTTLENECK: {analysis['bottleneck_stage'].upper()} ({analysis['percentages'][analysis['bottleneck_stage']]:.1f}% of processing time)")
        
        # Recommendations
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        bottleneck = analysis['bottleneck_stage']
        if bottleneck == 'detection':
            print("  • Detection is the bottleneck (~{:.0f}% of time)".format(analysis['percentages']['detection']))
            print("  • Try: Increase quad_decimate (faster but less accurate)")
            print("  • Try: Use smaller resolution (e.g., 320x240)")
            print("  • Try: Switch to tag25h9 or tag16h5 (faster families)")
        elif bottleneck == 'frame_capture':
            print("  • Camera capture is slow (hardware limit)")
            print("  • Current camera limited to ~30 FPS")
            print("  • Consider high-speed USB camera (60+ FPS)")
        elif bottleneck == 'preprocessing':
            print("  • Preprocessing taking too long")
            print("  • Try: Remove Gaussian blur")
            print("  • Try: Use cv2.INTER_NEAREST for faster resizing")
        elif bottleneck == 'pose_estimation':
            print("  • Pose calculation is slow")
            print("  • This is typically fast - may indicate many detections")
        
        print("="*70)


class MeasurementLogger:
    """
    Logger to store all measurements during tracking session.
    Saves to Excel file after session ends.
    """
    def __init__(self):
        self.measurements = []
        self.start_time = time.time()
        
    def log_measurement(self, tag_id, position, orientation, timestamp_offset):
        """
        Log a single measurement.
        
        Args:
            tag_id: AprilTag ID
            position: (x, y, z) tuple in meters
            orientation: (roll, pitch, yaw) tuple in degrees
            timestamp_offset: Time offset from start in seconds
        """
        x, y, z = position
        roll, pitch, yaw = orientation
        
        self.measurements.append({
            'timestamp': timestamp_offset,
            'tag_id': tag_id,
            'x_m': x,
            'y_m': y,
            'z_m': z,
            'roll_deg': roll,
            'pitch_deg': pitch,
            'yaw_deg': yaw
        })
    
    def save_to_excel(self, output_dir='reports'):
        """
        Save all measurements to Excel file.
        
        Args:
            output_dir: Directory to save the report (relative to project root)
        """
        if not self.measurements:
            print("No measurements to save.")
            return None
        
        try:
            import pandas as pd
            
            # Create reports directory if it doesn't exist
            report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
            os.makedirs(report_path, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apriltag_measurements_{timestamp_str}.xlsx"
            filepath = os.path.join(report_path, filename)
            
            # Convert measurements to DataFrame
            df = pd.DataFrame(self.measurements)
            
            # Create Excel writer with formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Measurements', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': [
                        'Total Measurements',
                        'Unique Tags Detected',
                        'Duration (seconds)',
                        'Average Rate (measurements/sec)',
                        'Start Time',
                        'End Time'
                    ],
                    'Value': [
                        len(self.measurements),
                        len(df['tag_id'].unique()),
                        f"{df['timestamp'].max():.2f}",
                        f"{len(self.measurements) / df['timestamp'].max():.2f}",
                        datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.fromtimestamp(self.start_time + df['timestamp'].max()).strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"\n{'='*70}")
            print(f"Measurements saved to: {filepath}")
            print(f"Total measurements: {len(self.measurements)}")
            print(f"{'='*70}")
            
            return filepath
            
        except ImportError:
            print("\n⚠ Warning: pandas not installed. Installing...")
            print("Run: pip install pandas openpyxl")
            return self.save_to_csv_fallback(output_dir)
        except Exception as e:
            print(f"\n⚠ Error saving to Excel: {e}")
            return self.save_to_csv_fallback(output_dir)
    
    def save_to_csv_fallback(self, output_dir='reports'):
        """
        Fallback: Save to CSV if Excel writing fails.
        """
        try:
            import csv
            
            # Create reports directory
            report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
            os.makedirs(report_path, exist_ok=True)
            
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apriltag_measurements_{timestamp_str}.csv"
            filepath = os.path.join(report_path, filename)
            
            # Write CSV
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'tag_id', 'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.measurements)
            
            print(f"\n{'='*70}")
            print(f"Measurements saved to CSV: {filepath}")
            print(f"Total measurements: {len(self.measurements)}")
            print(f"{'='*70}")
            
            return filepath
            
        except Exception as e:
            print(f"\n⚠ Error saving to CSV: {e}")
            return None


class PerformanceMonitor:
    """
    Monitor and display tracking performance metrics.
    """
    def __init__(self, window_size=60):
        self.frame_times = deque(maxlen=window_size)
        self.detection_counts_per_frame = deque(maxlen=10)  # Track detections in each frame
        self.frame_timestamps = deque(maxlen=10)  # Track when each frame was processed
        self.last_time = time.time()
        self.frame_count = 0
        self.total_detections = 0
        
    def update_frame(self, num_detections=0):
        """Update frame timing and detection count."""
        current_time = time.time()
        if self.frame_count > 0:
            self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        self.frame_count += 1
        
        # Track detections for this frame
        self.detection_counts_per_frame.append(num_detections)
        self.frame_timestamps.append(current_time)
        self.total_detections += num_detections
        
    def get_fps(self):
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_detection_rate(self):
        """Calculate current detection rate (actual detections per second)."""
        if len(self.frame_timestamps) < 2:
            return 0.0
        
        # Calculate time span of our window
        time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if time_span <= 0:
            return 0.0
        
        # Sum total detections in window
        total_detections_in_window = sum(self.detection_counts_per_frame)
        
        # Calculate rate
        return total_detections_in_window / time_span
    
    def get_avg_detections_per_frame(self):
        """Calculate average number of detections per frame."""
        if len(self.detection_counts_per_frame) < 1:
            return 0.0
        return sum(self.detection_counts_per_frame) / len(self.detection_counts_per_frame)
    
    def get_avg_frame_time(self):
        """Get average frame processing time in ms."""
        if len(self.frame_times) < 1:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000
    
    def create_performance_display(self, width=500, height=400, profiler=None):
        """
        Create a performance display window with profiling data.
        
        Args:
            profiler: PipelineProfiler instance for detailed breakdown
            
        Returns:
            numpy.ndarray: Performance display image
        """
        # Create black background
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate metrics
        detection_rate = self.get_detection_rate()
        fps = self.get_fps()
        avg_time = self.get_avg_frame_time()
        
        y_pos = 30
        
        # Draw title
        cv2.putText(display, "PERFORMANCE MONITOR", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        
        # Draw detection rate (actual AprilTag detections per second)
        rate_color = (0, 255, 0) if detection_rate >= 50 else (0, 165, 255) if detection_rate >= 20 else (0, 0, 255)
        cv2.putText(display, f"Detections/sec: {detection_rate:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, rate_color, 2)
        y_pos += 40
        
        # Draw current detections per frame
        avg_detections = self.get_avg_detections_per_frame()
        det_color = (0, 255, 0) if avg_detections >= 1.0 else (0, 165, 255) if avg_detections >= 0.5 else (0, 0, 255)
        cv2.putText(display, f"Tags/frame: {avg_detections:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_color, 2)
        y_pos += 35
        
        # Draw frame rate
        cv2.putText(display, f"Frame Rate: {fps:.1f} FPS", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        # Draw frame time
        cv2.putText(display, f"Frame Time: {avg_time:.1f} ms", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30
        
        # Draw separator
        cv2.line(display, (20, y_pos), (width-20, y_pos), (100, 100, 100), 1)
        y_pos += 25
        
        # Add profiling breakdown if available
        if profiler:
            cv2.putText(display, "PIPELINE BREAKDOWN:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_pos += 25
            
            # Get timing data
            capture_time = profiler.get_avg_timing('frame_capture')
            preprocess_time = profiler.get_avg_timing('preprocessing')
            detection_time = profiler.get_avg_timing('detection')
            pose_time = profiler.get_avg_timing('pose_estimation')
            total_time = capture_time + preprocess_time + detection_time + pose_time
            
            if total_time > 0:
                # Calculate percentages
                capture_pct = (capture_time / total_time) * 100
                preprocess_pct = (preprocess_time / total_time) * 100
                detection_pct = (detection_time / total_time) * 100
                pose_pct = (pose_time / total_time) * 100
                
                # Find bottleneck
                timings = {
                    'Capture': capture_time,
                    'Preprocess': preprocess_time,
                    'Detection': detection_time,
                    'Pose Est': pose_time
                }
                bottleneck = max(timings.items(), key=lambda x: x[1])[0]
                
                # Draw each stage with color coding
                stages = [
                    ('Capture', capture_time, capture_pct),
                    ('Preprocess', preprocess_time, preprocess_pct),
                    ('Detection', detection_time, detection_pct),
                    ('Pose Est', pose_time, pose_pct)
                ]
                
                for stage_name, stage_time, stage_pct in stages:
                    # Highlight bottleneck in red
                    color = (0, 0, 255) if stage_name == bottleneck else (150, 150, 150)
                    
                    cv2.putText(display, f"  {stage_name}:", (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(display, f"{stage_time:5.1f}ms ({stage_pct:4.1f}%)", (180, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 22
                
                # Draw separator
                cv2.line(display, (20, y_pos), (width-20, y_pos), (100, 100, 100), 1)
                y_pos += 20
                
                # Highlight bottleneck
                cv2.putText(display, f"Bottleneck: {bottleneck}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                y_pos += 25
        
        # Draw separator
        cv2.line(display, (20, y_pos), (width-20, y_pos), (100, 100, 100), 1)
        y_pos += 25
        
        # Draw total detections
        cv2.putText(display, f"Total Detections: {self.total_detections}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        
        # Draw total frames
        cv2.putText(display, f"Total Frames: {self.frame_count}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        y_pos += 30
        
        # Draw status based on detection rate
        if detection_rate >= TARGET_FPS * 0.9:
            status = "EXCELLENT"
            status_color = (0, 255, 0)
        elif detection_rate >= TARGET_FPS * 0.7:
            status = "GOOD"
            status_color = (0, 255, 255)
        elif detection_rate >= 30:
            status = "MODERATE"
            status_color = (0, 165, 255)
        else:
            status = "LOW"
            status_color = (0, 0, 255)
            
        cv2.putText(display, f"Status: {status}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return display


def main():
    """
    Main function to run AprilTag tracking.
    Optimized for maximum speed and performance.
    """
    print("AprilTag Tracker Starting...")
    print(f"Tag size: {TAG_SIZE*1000}mm")
    print(f"Tag family: {TAG_FAMILY}")
    print(f"Visual output: {'ENABLED' if ENABLE_VISUAL_OUTPUT else 'DISABLED (for max performance)'}")
    print(f"Target FPS: {TARGET_FPS}")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration(CALIBRATION_FILE)
    
    # Initialize camera
    cap = initialize_camera(CAMERA_ID, IMAGE_RES)
    
    # Extract camera parameters for detector
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Initialize AprilTag detector with balanced speed/accuracy for max detection rate
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,  # Optimal for most CPUs (more threads can cause overhead)
        quad_decimate=1.5,  # Reduced from 3.0 for better detection (1.0 = no decimation, slower but more accurate)
        quad_sigma=0.0,  # No blur for speed
        refine_edges=1,  # Re-enabled for better detection (slight speed cost)
        decode_sharpening=0.25,  # Re-enabled for better tag recognition
        debug=0
    )
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(window_size=60)
    
    # Initialize measurement logger
    measurement_logger = MeasurementLogger()
    session_start_time = time.time()
    
    # Initialize pipeline profiler
    profiler = PipelineProfiler(window_size=30)
    enable_profiling = True  # Set to False to disable profiling overhead
    
    print("\nAprilTag detector initialized!")
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Position tag in view to see tracking data")
    print("  - Performance window shows real-time metrics & profiling\n")
    
    frame_count = 0
    
    try:
        while True:
            # Start loop timing
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            ret, frame = cap.read()
            if enable_profiling:
                profiler.record_timing('frame_capture', (time.time() - capture_start) * 1000)
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Preprocessing
            preprocess_start = time.time()
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply slight Gaussian blur to reduce noise and improve detection
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            if enable_profiling:
                profiler.record_timing('preprocessing', (time.time() - preprocess_start) * 1000)
            
            # Detect AprilTags
            detection_start = time.time()
            raw_detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[fx, fy, cx, cy],
                tag_size=TAG_SIZE
            )
            
            # Filter detections by quality (decision_margin)
            detections = [d for d in raw_detections if d.decision_margin >= MIN_DECISION_MARGIN]
            
            if enable_profiling:
                profiler.record_timing('detection', (time.time() - detection_start) * 1000)
            
            # Process each detected tag
            num_detections = len(detections) if detections else 0
            if detections:
                pose_start = time.time()
                
                for detection in detections:
                    
                    # Estimate pose
                    position, orientation = estimate_pose(detection, camera_matrix, TAG_SIZE)
                    
                    # Log measurement (in memory, no disk I/O during tracking)
                    timestamp_offset = time.time() - session_start_time
                    measurement_logger.log_measurement(
                        detection.tag_id,
                        position,
                        orientation,
                        timestamp_offset
                    )
                    
                    # Draw on frame (only if visual output is enabled)
                    if ENABLE_VISUAL_OUTPUT:
                        draw_tag_info(frame, detection, position, orientation)
                    
                    # Print to terminal for every detection
                    print_pose_info(detection, position, orientation)
                
                if enable_profiling:
                    profiler.record_timing('pose_estimation', (time.time() - pose_start) * 1000)
            
            # Update frame performance metrics with detection count
            perf_monitor.update_frame(num_detections)
            
            # Show visual output (only if enabled)
            if ENABLE_VISUAL_OUTPUT:
                # Display FPS on main frame
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(WINDOW_NAME, frame)
            
            # Always show performance monitor window with profiling data
            perf_display = perf_monitor.create_performance_display(profiler=profiler if enable_profiling else None)
            cv2.imshow(PERFORMANCE_WINDOW, perf_display)
            
            # Record total loop time
            if enable_profiling:
                profiler.record_timing('total_loop', (time.time() - loop_start) * 1000)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting...")
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Print final statistics
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Total frames processed: {perf_monitor.frame_count}")
        print(f"Total measurements: {perf_monitor.detection_count}")
        print(f"Average frame rate: {perf_monitor.get_fps():.1f} FPS")
        print(f"Average measurement rate: {perf_monitor.get_detection_rate():.1f} measurements/sec")
        print(f"Average frame time: {perf_monitor.get_avg_frame_time():.1f} ms")
        print("="*70)
        
        # Print profiling report
        if enable_profiling:
            profiler.print_profile_report()
        
        # Save measurements to Excel
        print("\nSaving measurements to Excel...")
        measurement_logger.save_to_excel()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    main()
