import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
import math

# ==================== CONFIGURATION ====================
# Camera ID (0 = built-in webcam, 1 = USB camera)
CAMERA_ID = 1

# Camera resolution
IMAGE_RES = (1920, 1080)

# Path to camera calibration file
CALIBRATION_FILE = 'motion_capture/camera_calibration.yaml'

# AprilTag family (options: 'tag36h11', 'tag25h9', 'tag16h5', 'tagCircle21h7', 'tagStandard41h12')
TAG_FAMILY = 'tag36h11'

# AprilTag size in meters (physical size of the tag)
TAG_SIZE = 0.023  # 23mm = 0.023 meters

# Display window name
WINDOW_NAME = 'AprilTag Tracker'
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
    
    Args:
        camera_id (int): Camera ID
        resolution (tuple): Desired resolution (width, height)
        
    Returns:
        cv2.VideoCapture: Opened camera capture object
    """
    print(f"Opening camera {camera_id}...")
    
    # Try different backends
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
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
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


def main():
    """
    Main function to run AprilTag tracking.
    """
    print("AprilTag Tracker Starting...")
    print(f"Tag size: {TAG_SIZE*1000}mm")
    print(f"Tag family: {TAG_FAMILY}")
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration(CALIBRATION_FILE)
    
    # Initialize camera
    cap = initialize_camera(CAMERA_ID, IMAGE_RES)
    
    # Extract camera parameters for detector
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Initialize AprilTag detector
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    print("\nAprilTag detector initialized!")
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Position tag in view to see tracking data\n")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags
            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[fx, fy, cx, cy],
                tag_size=TAG_SIZE
            )
            
            # Process each detected tag
            if detections:
                for detection in detections:
                    # Estimate pose
                    position, orientation = estimate_pose(detection, camera_matrix, TAG_SIZE)
                    
                    # Draw on frame
                    draw_tag_info(frame, detection, position, orientation)
                    
                    # Print to terminal (only every 10 frames to avoid spam)
                    if frame_count % 10 == 0:
                        print_pose_info(detection, position, orientation)
            else:
                # No tags detected
                cv2.putText(frame, "No AprilTags detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display FPS
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting...")
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    main()
