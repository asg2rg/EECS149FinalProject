#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AprilTag GUI Tracker (Python 3)

- Detects tag, sends (x,y,z,spx,spy) over UDP to controller (UDP_PORT)
- Listens for controller ACKs on ACK_PORT and overlays status on the video
"""

import cv2
import numpy as np
import yaml
import socket
import time
from pupil_apriltags import Detector

# ------------- CONFIG -------------
UDP_IP    = "127.0.0.1"   
UDP_PORT  = 5005
ACK_PORT  = 5006          

CAMERA_ID = 1
IMAGE_RES = (1920, 1080)
CALIBRATION_FILE = "camera_calibration.yaml"

TAG_SIZE   = 0.023
TAG_FAMILY = "tag36h11"
WINDOW     = "AprilTag GUI Tracker"

sp_x = 320
sp_y = 240
BOX_SIZE  = 60
MATCH_TOL = 40
# ----------------------------------

def load_camera_calibration(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"])

def mouse_callback(event, x, y, flags, param):
    global sp_x, sp_y
    if event == cv2.EVENT_LBUTTONDOWN:
        sp_x, sp_y = x, y
        print("New waypoint:", sp_x, sp_y)

# add near the top, after load_camera_calibration(...)
def px_to_meters(x_px, y_px, Z, K):
    """
    Convert image pixel (x_px, y_px) at depth Z (m) into camera-frame meters.
    K: 3x3 camera matrix
    Returns (X_m, Y_m) in meters, using OpenCV camera coords (x right, y down, z forward).
    """
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    X = (x_px - cx) * (Z / fx)
    Y = (y_px - cy) * (Z / fy)
    return X, Y


class CameraGUI:
    def __init__(self):
        # TX socket (to controller)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # RX socket for ACKs
        self.ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack_sock.bind(("0.0.0.0", ACK_PORT))
        self.ack_sock.setblocking(False)
        self.last_ack = ""
        self.last_ack_ts = 0.0

        # camera + calib
        self.camera_matrix, self.dist = load_camera_calibration(CALIBRATION_FILE)
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # detector
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=2.0,
            refine_edges=1,
        )

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, mouse_callback)

    def run(self):
        global sp_x, sp_y
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame[frame < 80] = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[
                    self.camera_matrix[0, 0],
                    self.camera_matrix[1, 1],
                    self.camera_matrix[0, 2],
                    self.camera_matrix[1, 2],
                ],
                tag_size=TAG_SIZE,
            )

            # draw target box
            cv2.rectangle(
                frame,
                (sp_x - BOX_SIZE, sp_y - BOX_SIZE),
                (sp_x + BOX_SIZE, sp_y + BOX_SIZE),
                (0, 255, 0), 2
            )
            match_status = False

            if len(detections) > 0:
                det = detections[0]
                # x = int(det.center[0])
                # y = int(det.center[1])
                # z = float(det.pose_t[2, 0])  # camera depth to tag

                # corners = det.corners.astype(int)
                # for i in range(4):
                #     cv2.line(frame, tuple(corners[i]),
                #              tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                # cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

                # if abs(x - sp_x) < MATCH_TOL and abs(y - sp_y) < MATCH_TOL:
                #     match_status = True

                # # send UDP to controller
                # msg = "{},{},{},{},{}".format(x, y, z, sp_x, sp_y).encode()
                # self.tx.sendto(msg, (UDP_IP, UDP_PORT))
                # print("[GUI] Sent UDP:", msg.decode())

                # cv2.putText(frame, "Sent: " + msg.decode(), (10, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Pixels (image coords) fomr Chat
                x_px = float(det.center[0])
                y_px = float(det.center[1])

                # Depth in meters (camera -> tag along optical axis)
                z_m = float(det.pose_t[2, 0])
                # Optional: guard against bad/zero depth
                if z_m <= 0:
                    z_m = 1e-6

                # Convert pixel center -> meters in camera frame
                x_m, y_m = px_to_meters(x_px, y_px, z_m, self.camera_matrix)

                # Convert pixel waypoint -> meters in camera frame at same depth
                spX_m, spY_m = px_to_meters(sp_x, sp_y, z_m, self.camera_matrix)

                # If your controller expects Y-up (math convention), flip sign:
                # y_m  = -y_m
                # spY_m = -spY_m

                # --- send UDP in meters ---
                msg = f"{x_m:.6f},{y_m:.6f},{z_m:.6f},{spX_m:.6f},{spY_m:.6f}".encode()
                self.tx.sendto(msg, (UDP_IP, UDP_PORT))
                print("[GUI] Sent UDP (meters):", msg.decode())

                cv2.putText(frame, "Sent(m): " + msg.decode(), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                cv2.putText(frame, "NO TAG", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if match_status:
                cv2.putText(frame, "MATCH!", (sp_x - 50, sp_y - BOX_SIZE - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.rectangle(
                    frame,
                    (sp_x - BOX_SIZE, sp_y - BOX_SIZE),
                    (sp_x + BOX_SIZE, sp_y + BOX_SIZE),
                    (0, 255, 0), 4 
                )

            # poll ACKs (non-blocking)
            try:
                ack, _ = self.ack_sock.recvfrom(4096)
                self.last_ack = ack.decode(errors="ignore")
                self.last_ack_ts = time.time()
            except BlockingIOError:
                pass
            except Exception:
                pass

            # overlay last ACK + age
            age_ms = int((time.time() - self.last_ack_ts) * 1000) if self.last_ack_ts else 9999
            cv2.putText(frame, "ACK age: {} ms".format(age_ms), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "ACK: {}".format(self.last_ack[:90]), (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CameraGUI().run()
