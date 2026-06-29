"""Camera calibration for fisheye lenses using ChArUco boards."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template_string, send_from_directory

# ChArUco Configuration
ARUCO_DICT: cv2.aruco.Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SQUARES_X: int = 11
SQUARES_Y: int = 8
SQUARE_LENGTH: float = 0.010    # in meters
MARKER_LENGTH: float = 0.00733  # in meters

CHARUCO_BOARD: cv2.aruco.CharucoBoard = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)

ARUCO_PARAMS: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()
# ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.05
# ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# ARUCO_PARAMS.cornerRefinementWinSize = 5
# ARUCO_PARAMS.cornerRefinementMaxIterations = 30
# ARUCO_PARAMS.cornerRefinementMinAccuracy = 0.1


app: Flask = Flask(__name__)
log: logging.Logger = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

frame_lock: threading.Lock = threading.Lock()
state_lock: threading.Lock = threading.Lock()

current_frame: Optional[np.ndarray] = None
jpeg_buffer: Optional[bytes] = None
image_size: Optional[Tuple[int, int]] = None

captured_corners: List[np.ndarray] = []
captured_ids: List[np.ndarray] = []

calibrated_K: Optional[List[List[float]]] = None
calibrated_D: Optional[List[List[float]]] = None
calibrated_RMS: Optional[float] = None
per_frame_errors: Optional[List[float]] = None

MIN_CAPTURES: int = 15
MAX_CAPTURES: int = 30
auto_mode_enabled: bool = False
is_calibrating: bool = False


class ObservabilitySampler:
    """Tracks geometric diversity across SE(3) space to ensure observability."""

    def __init__(self) -> None:
        self.grid_coverage: Set[str] = set()
        self.scale_coverage: Set[str] = set()
        self.roll_coverage: Set[str] = set()
        
        self.last_guidance_msg: str = "INITIALIZATION. Move the board into the frame to begin."
        self.max_velocity_px: float = 20.0
        self.last_centroid: Optional[np.ndarray] = None
        self.last_capture_time: float = 0.0

    @property
    def is_coverage_optimal(self) -> bool:
        """Evaluates if the essential geometric bins are saturated."""
        return (
            len(self.grid_coverage) >= 9 and 
            len(self.scale_coverage) >= 3 and 
            len(self.roll_coverage) >= 1
        )

    def get_xy_zone(self, cx: float, cy: float, width: int, height: int) -> str:
        """Determines the planar quadrant of the board centroid."""
        x_zone = "LEFT" if cx < width / 3 else "RIGHT" if cx > 2 * width / 3 else "CENTER"
        y_zone = "TOP" if cy < height / 3 else "BOTTOM" if cy > 2 * height / 3 else "MIDDLE"
        if x_zone == "CENTER" and y_zone == "MIDDLE":
            return "CENTER"
        return f"{y_zone}-{x_zone}"

    def get_scale_zone(self, area: float, total_area: float) -> str:
        """Determines the depth bin based on bounding box area ratio."""
        ratio = area / total_area
        if ratio < 0.05: return "VERY_FAR"
        if ratio < 0.10: return "FAR"
        if ratio > 0.90: return "CLOSE"
        return "MEDIUM"

    def estimate_in_plane_rot(self, corners: np.ndarray) -> str:
        """Estimates roll rotation (diamond vs flat) of the pattern."""
        rect = cv2.minAreaRect(corners)
        angle = rect[2]
        if angle < -80 or angle > -10: return "ROLL_0"
        if -55 < angle < -35: return "ROLL_45"
        return "ROLL_OTHER"

    def compute_next_best_view(self, current_captures: int) -> str:
        """Discrete trajectory planner based on the residual state space."""
        if self.is_coverage_optimal:
            if current_captures < MIN_CAPTURES:
                return f"[FINAL PHASE] Coverage optimal. Random complementary acquisition..."
            return "[READY] Metrics saturated. Optimization triggered..."

        all_zones = {"CENTER", "TOP-LEFT", "TOP-CENTER", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-CENTER", "BOTTOM-RIGHT", "MIDDLE-LEFT", "MIDDLE-RIGHT"}
        missing_zones = all_zones - self.grid_coverage

        if len(self.grid_coverage) == 0:
            return "[Phase 1/3 - Principal Point] Place the board flat in the CENTER of the frame."
        if missing_zones:
            target = list(missing_zones)[0]
            return f"[Phase 1/3 - Principal Point] Move the board slowly to the {target}."
            
        all_scales = {"VERY_FAR", "FAR", "CLOSE", "MEDIUM"}
        missing_scales = all_scales - self.scale_coverage
        if missing_scales:
            target = list(missing_scales)[0]
            if target in ["FAR", "VERY_FAR"]: return "[Phase 2/3 - Focal Length] Move the board as FAR away as possible."
            if target == "CLOSE": return "[Phase 2/3 - Focal Length] Move the board CLOSE (while keeping it sharp)."
            
        if "ROLL_45" not in self.roll_coverage:
            return "[Phase 3/3 - Distortion] Rotate the board 45 degrees in-plane."

        return "[FINAL PHASE] Random complementary acquisition."

    def evaluate_frame(self, frame_w: int, frame_h: int, corners: np.ndarray, current_captures: int) -> Tuple[bool, str]:
        """Evaluates if the current frame provides novel geometric constraints."""
        centroid = np.mean(corners, axis=0)[0]
        hull = cv2.convexHull(corners)
        area = cv2.contourArea(hull)
        total_area = float(frame_w * frame_h)
        
        zone = self.get_xy_zone(centroid[0], centroid[1], frame_w, frame_h)
        scale = self.get_scale_zone(area, total_area)
        roll = self.estimate_in_plane_rot(corners)
        
        self.last_guidance_msg = self.compute_next_best_view(current_captures)

        if self.last_centroid is not None:
            velocity = float(np.linalg.norm(centroid - self.last_centroid))
            self.last_centroid = centroid
            if velocity > self.max_velocity_px:
                return False, "MOVING..."
        else:
            self.last_centroid = centroid
            return False, "STABILIZING..."

        is_novel = False
        if zone not in self.grid_coverage:
            self.grid_coverage.add(zone)
            is_novel = True
        elif scale not in self.scale_coverage:
            self.scale_coverage.add(scale)
            is_novel = True
        elif roll not in self.roll_coverage and roll != "ROLL_0":
            self.roll_coverage.add(roll)
            is_novel = True
        elif self.is_coverage_optimal and current_captures < MIN_CAPTURES:
            if np.random.random() < 0.4:
                is_novel = True

        t_now = time.time()
        if is_novel and (t_now - self.last_capture_time) > 1.2:
            self.last_capture_time = t_now
            return True, "GOOD POSITION ACQUIRED!"
            
        return False, "REDUNDANT VIEW (This kind of pose is already sampled)"


sampler = ObservabilitySampler()


def evaluate_rms(rms: float) -> str:
    """Provides qualitative feedback on the final calibration root-mean-square error."""
    if rms <= 0.4: return "Excellent calibration!"
    if rms <= 0.8: return "Acceptable calibration."
    return "Poor calibration... Consider re-runing the procedure."


def compute_per_frame_errors(
    obj_points: List[np.ndarray], 
    img_points: List[np.ndarray], 
    rvecs: Any, 
    tvecs: Any, 
    K: np.ndarray, 
    D: np.ndarray
) -> List[float]:
    """Calculates the mean pixel reprojection error for each individual keyframe."""
    errors: List[float] = []
    for i in range(len(obj_points)):
        img_proj, _ = cv2.fisheye.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        err = np.linalg.norm(img_points[i].reshape(-1, 2) - img_proj.reshape(-1, 2), axis=1)
        errors.append(float(np.mean(err)))
    return errors


def run_calibration_task() -> None:
    """Background task to run the Levenberg-Marquardt fisheye optimization."""
    global calibrated_K, calibrated_D, calibrated_RMS, per_frame_errors
    global is_calibrating, auto_mode_enabled

    is_calibrating = True
    print("[*] Running Levenberg-Marquardt Optimization...")

    with state_lock:
        if len(captured_corners) < 10:
            is_calibrating = False
            return
            
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        board_obj_pts = np.array(CHARUCO_BOARD.getChessboardCorners(), dtype=np.float32)

        for i in range(len(captured_ids)):
            flat_ids = captured_ids[i].flatten()
            objp = board_obj_pts[flat_ids]
            obj_points.append(objp.reshape(1, -1, 3).astype(np.float32))
            img_points.append(captured_corners[i].reshape(1, -1, 2).astype(np.float32))

    K_in = np.zeros((3, 3), dtype=np.float64)
    D_in = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        if image_size is None: 
            raise ValueError("Image dimensions are undefined.")

        rms_val, K_out, D_out, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points, img_points, image_size, K_in, D_in, None, None, flags=flags
        )

        K_final = np.array(K_out, dtype=np.float64)
        D_final = np.array(D_out, dtype=np.float64)

        frame_errors = compute_per_frame_errors(obj_points, img_points, rvecs, tvecs, K_final, D_final)

        with state_lock:
            calibrated_K = K_final.tolist()
            calibrated_D = D_final.tolist()
            calibrated_RMS = float(rms_val)
            per_frame_errors = frame_errors
            auto_mode_enabled = False

        print(f"[SUCCESS] Calibration Complete. RMS = {rms_val:.4f} ({evaluate_rms(float(rms_val))})")

    except Exception as e:
        print(f"[!] Levenberg-Marquardt Optimizer failed: {e}")
        with state_lock: auto_mode_enabled = False
    finally:
        is_calibrating = False


def camera_thread() -> None:
    """Main CV loop handling CSI/USB frame grabbing and ArUco detection."""
    global current_frame, jpeg_buffer, image_size

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, -1)
        fw, fh = frame.shape[1], frame.shape[0]
        if image_size is None: 
            image_size = (fw, fh)

        display_frame = frame.copy()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
        
        status_text = "BOARD NOT DETECTED"
        color = (0, 0, 255)

        with state_lock:
            current_captures = len(captured_corners)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            ret_corn, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, frame, CHARUCO_BOARD
            )

            if ret_corn >= 4: 
                cv2.aruco.drawDetectedCornersCharuco(display_frame, ch_corners, ch_ids, (0, 255, 0))
                
                with state_lock:
                    if auto_mode_enabled and not is_calibrating and current_captures < MAX_CAPTURES:
                        accept, reason = sampler.evaluate_frame(fw, fh, ch_corners, current_captures)
                        if accept:
                            captured_corners.append(ch_corners)
                            captured_ids.append(ch_ids)
                            current_captures += 1
                            display_frame = cv2.bitwise_not(display_frame) 
                            status_text = reason
                            color = (0, 255, 0)
                        else:
                            status_text = reason
                            color = (0, 165, 255)

        with state_lock:
            ready_to_calibrate = (
                auto_mode_enabled
                and not is_calibrating
                and calibrated_RMS is None
                and (
                    (sampler.is_coverage_optimal and current_captures >= MIN_CAPTURES)
                    or current_captures >= MAX_CAPTURES
                )
            )

        if ready_to_calibrate:
            threading.Thread(target=run_calibration_task, daemon=True).start()

        cv2.putText(display_frame, f"Captures: {current_captures} / {MAX_CAPTURES} (Min: {MIN_CAPTURES})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if auto_mode_enabled:
            cv2.putText(display_frame, f"STATE: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.line(display_frame, (fw//3, 0), (fw//3, fh), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (2*fw//3, 0), (2*fw//3, fh), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (0, fh//3), (fw, fh//3), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(display_frame, (0, 2*fh//3), (fw, 2*fh//3), (255,255,255), 1, cv2.LINE_AA)

        if is_calibrating:
            cv2.putText(display_frame, "OPTIMIZING (Levenberg-Marquardt)...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        if calibrated_RMS is not None:
            cv2.putText(display_frame, f"FINAL RMS: {calibrated_RMS:.3f} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with frame_lock:
            current_frame = frame.copy()
            ret_enc, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret_enc: 
                jpeg_buffer = buf.tobytes()

threading.Thread(target=camera_thread, daemon=True).start()

# HTTP Routes
@app.route("/video_feed") # type: ignore[untyped-decorator]
def video_feed() -> Response:
    """Streams MJPEG frames to the web UI."""
    def generate() -> Any:
        while True:
            with frame_lock: 
                buf = jpeg_buffer
            if buf: 
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status", methods=["GET"]) # type: ignore[untyped-decorator]
def get_status() -> Response:
    """Returns real-time JSON state for the frontend polling."""
    with state_lock:
        return jsonify({
            "is_auto": auto_mode_enabled,
            "captures": len(captured_corners),
            "target": MIN_CAPTURES,
            "guidance": sampler.last_guidance_msg if auto_mode_enabled else "System idle. Press START.",
            "rms": calibrated_RMS
        })


@app.route("/api/toggle_auto", methods=["POST"]) # type: ignore[untyped-decorator]
def toggle_auto() -> Tuple[Response, int]:
    """Toggles the autonomous sample collection state."""
    global auto_mode_enabled, calibrated_RMS, per_frame_errors
    with state_lock:
        auto_mode_enabled = not auto_mode_enabled
        if auto_mode_enabled and calibrated_RMS is not None:
            captured_corners.clear()
            captured_ids.clear()
            sampler.grid_coverage.clear()
            sampler.scale_coverage.clear()
            sampler.roll_coverage.clear()
            calibrated_RMS = None
            per_frame_errors = None
            
    msg = "Calibration STARTED." if auto_mode_enabled else "Calibration STOPPED."
    return jsonify({"status": "success", "message": msg}), 200


@app.route("/api/save", methods=["POST"]) # type: ignore[untyped-decorator]
def save() -> Tuple[Response, int]:
    """Exports camera metrics and intrinsics."""
    with state_lock:
        if calibrated_K is None:
            return jsonify({
                "status": "error", 
                "message": "Configuration not generated. Reach capture target first."
            }), 400

        config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
        try:
            with open(config_path, "r") as f: 
                config: Dict[str, Any] = yaml.safe_load(f) or {}
        except FileNotFoundError: 
            config = {}

        config["camera"] = {
            "resolution": list(image_size) if image_size else [640, 480],
            "camera_matrix": calibrated_K,
            "dist_coeffs": calibrated_D,
            "rms_error": calibrated_RMS,
            "per_frame_errors": per_frame_errors,
            "model": "fisheye",
        }

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f: 
            yaml.safe_dump(config, f, default_flow_style=False)

    return jsonify({"status": "success", "message": f"Parameters exported to {config_path}."}), 200


@app.route("/api/reset", methods=["POST"]) # type: ignore[untyped-decorator]
def reset() -> Tuple[Response, int]:
    """Purges all memory buffers and resets matrices."""
    global calibrated_K, calibrated_D, calibrated_RMS, per_frame_errors, auto_mode_enabled
    with state_lock:
        captured_corners.clear()
        captured_ids.clear()
        sampler.grid_coverage.clear()
        sampler.scale_coverage.clear()
        sampler.roll_coverage.clear()
        calibrated_K = None
        calibrated_D = None
        calibrated_RMS = None
        per_frame_errors = None
        auto_mode_enabled = False
    return jsonify({"status": "success", "message": "Reset complete."}), 200


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chromapi Camera Calibration</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico?v=1">
    <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico?v=1">
    <style>
        body { font-family: monospace; background: #121212; color: #eee; text-align: center; margin: 20px;}
        img { border: 2px solid #333; border-radius: 4px; max-width: 100%;}
        .hud-panel { background: #1e1e1e; padding: 15px; margin: 15px auto; max-width: 800px; border-left: 4px solid #b605fc; text-align: left; }
        button { padding: 12px 24px; margin: 5px; font-size: 14px; background: #222; color: white; border: 1px solid #444; border-radius: 20px; cursor: pointer; text-transform: uppercase; transition: 0.2s;}
        button:hover { background: #333; border-color: #b605fc; }
        .btn-auto { background: #b605fc; color: #fff; font-weight: bold; border-color: #b605fc; }
        .btn-auto:hover { background: #8a03b8; }
        #guidance-box { margin-top: 10px; font-size: 1.2em; font-weight: bold; color: #ffaa00; background: #2d2d2d; padding: 10px; border-radius: 4px;}
        #action-logs { margin-top: 10px; font-size: 0.9em; color: #aaa; }
        .tip-fade { animation: fadeIn 0.5s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>
    <h2>Chromapi Camera Calibration</h2>
    
    <div class="hud-panel">
        <p style="margin:0 0 10px 0; color:#ccc;"><strong>Instructions:</strong> Grab your ChAruco board and follow the indications below after pressing START.</p>
        <div id="guidance-box" class="tip-fade">Ready? Press START to initialize the capture sequence.</div>
    </div>

    <img src="/video_feed" alt="Camera Feed"><br>
    
    <div style="margin-top: 20px;">
        <button class="btn-auto" onclick="req('/api/toggle_auto')">START / STOP Calibration</button>
        <button onclick="req('/api/save')">Save Camera Configuration</button>
        <button onclick="req('/api/reset')" style="border-color:#d90429; color:#ff4444;">Reset Calibration</button>
    </div>
    
    <div id="action-logs"></div>
    
    <script>
        const idleTips = [
            "CHROMA'TIP: Slightly tilt the board forward and backward while moving it.",
            "CHROMA'TIP: Bright, diffuse lighting ensures best results.",
            "CHROMA'TIP: Fisheye lenses distort most at the edges. Focus on the periphery.",
            "CHROMA'TIP: Keep the board perfectly still during the acquisition flash to avoid blur.",
            "CHROMA'TIP: Don't forget to press SAVE CAMERA CONFIGURATION when done!",
            "Ready? Press START to launch calibration."
        ];
        let tipIndex = 0;

        function req(endpoint) {
            fetch(endpoint, {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                const logDiv = document.getElementById('action-logs');
                logDiv.innerHTML = data.status === 'error' 
                    ? '<span style="color:#ff4444;">[ERR] ' + data.message + '</span>' 
                    : '<span style="color:#00ffcc;">[SYS] ' + data.message + '</span>';
            });
        }

        setInterval(() => {
            fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                const gBox = document.getElementById('guidance-box');
                
                if(gBox.dataset.lastText !== data.guidance && data.is_auto) {
                    gBox.classList.remove('tip-fade');
                    void gBox.offsetWidth; 
                    gBox.classList.add('tip-fade');
                    gBox.dataset.lastText = data.guidance;
                }

                if (data.rms !== null) {
                    gBox.innerHTML = "<span style='color:#00ffcc;'>CALIBRATION SUCCESSFUL (RMS: " + data.rms.toFixed(3) + ") - Press SAVE CAMERA CALIBRATION to save the results.</span>";
                } else if (data.is_auto) {
                    gBox.innerText = data.guidance + " (" + data.captures + " captures)";
                } else {
                    gBox.innerText = idleTips[tipIndex];
                }
            });
        }, 500);

        setInterval(() => {
            tipIndex = (tipIndex + 1) % idleTips.length;
        }, 4000);
    </script>
</body>
</html>
"""

@app.route("/") # type: ignore[untyped-decorator]
def index() -> str:
    """Renders the single-page HTML client."""
    rendering: str = render_template_string(HTML_TEMPLATE)
    return rendering


@app.route("/favicon.ico") # type: ignore[untyped-decorator]
def serve_favicon() -> Any:
    """Resolves and serves the favicon.ico dynamically from assets directory."""
    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "assets"
    
    if not (assets_dir / "favicon.ico").exists():
        print(f"[!] ERROR: Target file 'favicon.ico' does not exist in {assets_dir}!")
        return "", 404
        
    return send_from_directory(
        directory=str(assets_dir), 
        path="favicon.ico", 
        mimetype="image/vnd.microsoft.icon"
    )


if __name__ == "__main__":
    try:
        from waitress import serve
        print("[*] WSGI Waitress production server running on 0.0.0.0:5000...")
        serve(app, host="0.0.0.0", port=5000, _quiet=True)
    except ImportError:
        print("[!] Waitress missing. Falling back to default Werkzeug server.")
        app.run(host="0.0.0.0", port=5000, debug=False)