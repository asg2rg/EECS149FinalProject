import cv2
import numpy as np
import os
import yaml
import glob

# ==================== CONFIGURATION ====================
# Directory containing calibration images
CALIBRATION_IMAGES_DIR = 'calibration_images'

# Chessboard pattern size (number of INNER corners)
CHESSBOARD_SIZE = (13, 9)  # (columns, rows)

# Physical size of each square on the chessboard (in millimeters)
SQUARE_SIZE = 25.0  # Adjust this to match your actual chessboard

# Output file for calibration results
OUTPUT_FILE = 'camera_calibration.yaml'

# Supported image file extensions
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

# Termination criteria for corner refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# =======================================================


def load_calibration_images(directory):
    """
    Load all image files from the specified directory.
    
    Args:
        directory (str): Path to the directory containing calibration images
        
    Returns:
        list: List of image file paths
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
    
    image_files.sort()  # Ensure consistent ordering
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in directory: {directory}")
    
    print(f"Found {len(image_files)} images in {directory}")
    return image_files


def prepare_object_points(chessboard_size, square_size):
    """
    Prepare 3D object points for the chessboard pattern.
    
    Args:
        chessboard_size (tuple): Number of inner corners (columns, rows)
        square_size (float): Physical size of each square
        
    Returns:
        numpy.ndarray: 3D coordinates of chessboard corners
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_chessboard_corners(image_path, chessboard_size):
    """
    Detect chessboard corners in an image with sub-pixel refinement.
    
    Args:
        image_path (str): Path to the image file
        chessboard_size (tuple): Number of inner corners (columns, rows)
        
    Returns:
        tuple: (success, corners, gray_image) or (False, None, None)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return False, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Refine corner locations to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        return True, corners_refined, gray
    
    return False, None, None


def collect_calibration_data(image_files, chessboard_size, square_size):
    """
    Process all calibration images and collect object points and image points.
    
    Args:
        image_files (list): List of image file paths
        chessboard_size (tuple): Number of inner corners (columns, rows)
        square_size (float): Physical size of each square
        
    Returns:
        tuple: (object_points, image_points, image_size, valid_count)
    """
    # Prepare object points
    objp = prepare_object_points(chessboard_size, square_size)
    
    # Arrays to store object points and image points
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    
    image_size = None
    valid_count = 0
    
    print("\nProcessing images...")
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}", end='')
        
        ret, corners, gray = detect_chessboard_corners(image_path, chessboard_size)
        
        if ret:
            object_points.append(objp)
            image_points.append(corners)
            valid_count += 1
            
            # Store image size (assume all images are the same size)
            if image_size is None:
                image_size = gray.shape[::-1]
            
            print(" ✓")
        else:
            print(" ✗ (chessboard not detected)")
    
    print(f"\nSuccessfully detected chessboard in {valid_count}/{len(image_files)} images")
    
    return object_points, image_points, image_size, valid_count


def calibrate_camera(object_points, image_points, image_size):
    """
    Perform camera calibration using collected points.
    
    Args:
        object_points (list): List of 3D object points
        image_points (list): List of 2D image points
        image_size (tuple): Image dimensions (width, height)
        
    Returns:
        tuple: (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    print("\nPerforming camera calibration...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, 
        image_points, 
        image_size, 
        None, 
        None
    )
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calculate_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Calculate the mean reprojection error across all calibration images.
    
    Args:
        object_points (list): List of 3D object points
        image_points (list): List of 2D image points
        rvecs (list): Rotation vectors
        tvecs (list): Translation vectors
        camera_matrix (numpy.ndarray): Camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        
    Returns:
        float: Mean reprojection error
    """
    total_error = 0
    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(
            object_points[i], 
            rvecs[i], 
            tvecs[i], 
            camera_matrix, 
            dist_coeffs
        )
        error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    
    mean_error = total_error / len(object_points)
    return mean_error


def save_calibration_results(output_file, camera_matrix, dist_coeffs, image_size, rms_error):
    """
    Save calibration results to a YAML file.
    
    Args:
        output_file (str): Output file path
        camera_matrix (numpy.ndarray): Camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        image_size (tuple): Image dimensions (width, height)
        rms_error (float): RMS reprojection error
    """
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'image_width': int(image_size[0]),
        'image_height': int(image_size[1]),
        'rms_reprojection_error': float(rms_error)
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"\nCalibration results saved to: {output_file}")


def print_calibration_results(rms_error, camera_matrix, dist_coeffs):
    """
    Print calibration results to console.
    
    Args:
        rms_error (float): RMS reprojection error
        camera_matrix (numpy.ndarray): Camera intrinsic matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
    """
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    
    print(f"\nRMS Reprojection Error: {rms_error:.4f} pixels")
    
    print("\nCamera Intrinsic Matrix:")
    print(camera_matrix)
    
    print("\nDistortion Coefficients:")
    print(f"k1={dist_coeffs[0][0]:.6f}, k2={dist_coeffs[0][1]:.6f}, "
          f"p1={dist_coeffs[0][2]:.6f}, p2={dist_coeffs[0][3]:.6f}, "
          f"k3={dist_coeffs[0][4]:.6f}")
    
    # Extract focal lengths and principal point
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\nFocal Length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal Point: cx={cx:.2f}, cy={cy:.2f}")
    print("="*60)


def main():
    """
    Main function to perform camera calibration.
    """
    try:
        # Load calibration images
        image_files = load_calibration_images(CALIBRATION_IMAGES_DIR)
        
        # Collect calibration data from images
        object_points, image_points, image_size, valid_count = collect_calibration_data(
            image_files, 
            CHESSBOARD_SIZE, 
            SQUARE_SIZE
        )
        
        # Check if enough valid images were found
        if valid_count < 3:
            raise ValueError(
                f"Insufficient valid images for calibration. "
                f"Found {valid_count}, but at least 3 are required."
            )
        
        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            object_points, 
            image_points, 
            image_size
        )
        
        # Calculate mean reprojection error
        mean_error = calculate_reprojection_error(
            object_points, 
            image_points, 
            rvecs, 
            tvecs, 
            camera_matrix, 
            dist_coeffs
        )
        
        # Print results
        print_calibration_results(ret, camera_matrix, dist_coeffs)
        
        # Save calibration results
        save_calibration_results(
            OUTPUT_FILE, 
            camera_matrix, 
            dist_coeffs, 
            image_size, 
            ret
        )
        
        print("\nCalibration completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
