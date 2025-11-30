#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, socket, time, signal
import cflib.crtp
from cflib.crazyflie import Crazyflie

# Your existing PID classes
from pid import PID, PID_RP

# Limits (same spirit as reference)
CAP    = 15.0
TH_CAP = 55000

URI_DEFAULT = "radio://0/10/250K"
UDP_PORT    = 5005

class UDPTracker(object):
    """Minimal UDP front-end that mimics the Kinect API."""
    def __init__(self, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.01)

        self._x = None; self._y = None; self._z = None
        self._spx = 320.0; self._spy = 240.0; self._spz = 0.40
        self._last_rx = 0.0

    def find_position(self):
        """Return (x,y,depth) or (None, None, None) if no fresh packet."""
        try:
            data, _ = self._sock.recvfrom(1024)
            xs, ys, zs, spxs, spys = data.split(",")
            self._x = float(xs); self._y = float(ys); self._z = float(zs)
            self._spx = float(spxs); self._spy = float(spys)
            self._last_rx = time.time()
        except socket.timeout:
            pass
        except Exception as e:
            sys.stdout.write("[UDP] parse error: %s\n" % str(e))

        if self._x is None or self._y is None or self._z is None:
            return (None, None, None)
        if (time.time() - self._last_rx) > 0.5:
            return (None, None, None)
        return (self._x, self._y, self._z)

    def show(self, texts): pass
    def show_xy_sp(self, spx, spy): pass

    def get_spx(self): return self._spx
    def get_spy(self): return self._spy
    def get_spz(self): return self._spz

class OverheadPilot(object):
    """Crazyflie Overhead-Camera Pilot (minimal change from reference)."""
    def __init__(self):
        cflib.crtp.init_drivers()
        self._cf = Crazyflie()

        # Keep reference PID style: roll<-x, pitch<-y, thrust<-z
        self.r_pid = PID_RP(P=0.05, D=1.0,  I=0.00025, set_point=0.0)
        self.p_pid = PID_RP(P=0.10, D=1.0,  I=0.00025, set_point=0.0)
        self.t_pid = PID   (P=30.0, D=500.0, I=40.0,   set_point=0.0)

        self.sp_x = 320.0; self.sp_y = 240.0; self.sp_z = 0.40
        self.tracker = UDPTracker(UDP_PORT)

        signal.signal(signal.SIGINT, signal.SIG_DFL)

        self._cf.connected.add_callback(self._on_connected)
        self._cf.connection_failed.add_callback(self._on_conn_failed)
        self._cf.disconnected.add_callback(self._on_disconnected)
        self._connected = False

    def connect_crazyflie(self, link_uri):
        self._cf.open_link(link_uri)

    def _on_connected(self, link_uri):
        sys.stdout.write("Connected to %s\n" % link_uri); self._connected = True

    def _on_conn_failed(self, link_uri, msg):
        sys.stdout.write("Connection failed on %s: %s\n" % (link_uri, msg)); sys.exit(-1)

    def _on_disconnected(self, link_uri):
        sys.stdout.write("Disconnected from %s\n" % link_uri); self._connected = False
        
    def _p2t(self, percentage):
        """Convert a percentage to raw thrust"""
        return int(65000 * (percentage / 100.0))

    def set_sp_callback(self, x, y):
        self.sp_x = x; self.sp_y = y

    def control(self, dry=False):
        safety = 10
        period = 1.0 / 50.0  # 50 Hz

        while True:
            (x, y, depth) = self.tracker.find_position()
            if x is not None and y is not None and depth is not None:
                safety = 10
                # refresh setpoints from UDP
                self.sp_x = self.tracker.get_spx()
                self.sp_y = self.tracker.get_spy()
                self.sp_z = self.tracker.get_spz()

                # Keep the reference structure:
                roll   = self.r_pid.update(self.sp_x - x)
                pitch  = self.p_pid.update(self.sp_y - y)
                thrust = self.t_pid.update(self.sp_z - depth)

                roll_sp   = -roll
                pitch_sp  = -pitch
                thrust_sp = thrust + 38000

                if roll_sp >  CAP: roll_sp =  CAP
                if roll_sp < -CAP: roll_sp = -CAP
                if pitch_sp >  CAP: pitch_sp =  CAP
                if pitch_sp < -CAP: pitch_sp = -CAP
                if thrust_sp > TH_CAP: thrust_sp = TH_CAP
                if thrust_sp < 0:     thrust_sp = 0

                # print self.t_pid.error
                sys.stdout.write("[CMD] roll=%.2f pitch=%.2f thrust=%d | ex=%.1f ey=%.1f ez=%.3f\n" %
                                 (roll_sp, pitch_sp, int(thrust_sp),
                                  (self.sp_x - x), (self.sp_y - y), (self.sp_z - depth)))

                if not dry:
                    self._cf.commander.send_setpoint(roll_sp, pitch_sp, 0, int(thrust_sp))
            else:
                safety -= 1

            if safety < 0 and not dry:
                self._cf.commander.send_setpoint(0, 0, 0, 0)

            time.sleep(period)

def main():
    import argparse
    parser = argparse.ArgumentParser(prog="overhead_minimal_controller")
    parser.add_argument("-u", "--uri", dest="uri", type=str,
                        default=URI_DEFAULT,
                        help="Crazyflie URI (default: %s)" % URI_DEFAULT)
    parser.add_argument("-y", "--dry", dest="dry", action="store_true",
                        help="Do not send commands to Crazyflie")
    parser.add_argument("-d", "--debug", action="store_true", dest="debug",
                        help="Enable debug output")
    args, _ = parser.parse_known_args()

    if args.debug: logging.basicConfig(level=logging.DEBUG)
    else:          logging.basicConfig(level=logging.INFO)

    pilot = OverheadPilot()
    if not args.dry:
        pilot.connect_crazyflie(link_uri=args.uri)
    pilot.control(args.dry)

if __name__ == "__main__":
    main()
