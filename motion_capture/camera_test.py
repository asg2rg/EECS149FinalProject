"""
Camera Capability Test Script
Tests various resolutions and settings to find maximum FPS
"""
import cv2
import time

CAMERA_ID = 1

def test_configuration(backend, backend_name, resolution, fps_target):
    """Test a specific camera configuration."""
    print(f"\nTesting {backend_name} with {resolution[0]}x{resolution[1]} @ {fps_target} FPS...")
    
    try:
        cap = cv2.VideoCapture(CAMERA_ID, backend)
        if not cap.isOpened():
            print(f"  ❌ Failed to open camera with {backend_name}")
            return None
        
        # Set configuration
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual values
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Measure real FPS
        frame_count = 0
        start_time = time.time()
        test_duration = 3  # seconds
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
        
        elapsed = time.time() - start_time
        measured_fps = frame_count / elapsed
        
        cap.release()
        
        print(f"  ✓ Actual: {actual_width}x{actual_height}")
        print(f"  ✓ Reported FPS: {actual_fps:.1f}")
        print(f"  ✓ Measured FPS: {measured_fps:.1f}")
        
        return {
            'backend': backend_name,
            'resolution': (actual_width, actual_height),
            'reported_fps': actual_fps,
            'measured_fps': measured_fps
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("=" * 60)
    print("CAMERA CAPABILITY TEST")
    print("=" * 60)
    
    # Test configurations
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
    ]
    
    resolutions = [
        (640, 480),
        (320, 240),
        (800, 600),
        (1280, 720),
    ]
    
    fps_targets = [30, 60, 120]
    
    best_config = None
    best_fps = 0
    
    for backend, backend_name in backends:
        print(f"\n{'='*60}")
        print(f"Testing Backend: {backend_name}")
        print(f"{'='*60}")
        
        for resolution in resolutions:
            for fps_target in fps_targets:
                result = test_configuration(backend, backend_name, resolution, fps_target)
                
                if result and result['measured_fps'] > best_fps:
                    best_fps = result['measured_fps']
                    best_config = result
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION FOUND:")
    print("=" * 60)
    if best_config:
        print(f"Backend: {best_config['backend']}")
        print(f"Resolution: {best_config['resolution'][0]}x{best_config['resolution'][1]}")
        print(f"Reported FPS: {best_config['reported_fps']:.1f}")
        print(f"Measured FPS: {best_config['measured_fps']:.1f}")
    else:
        print("No successful configuration found!")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    if best_fps < 40:
        print("⚠ Your camera is hardware-limited to ~30 FPS")
        print("  This is a common limitation of USB webcams.")
        print("  To achieve 60+ FPS, you would need:")
        print("  - A high-speed USB camera (USB 3.0 with 60+ FPS capability)")
        print("  - Industrial camera with high frame rate")
        print("  - Different camera model designed for high-speed capture")
    else:
        print("✓ Your camera supports higher frame rates!")
        print(f"  Use the configuration above to achieve {best_fps:.1f} FPS")

if __name__ == "__main__":
    main()
