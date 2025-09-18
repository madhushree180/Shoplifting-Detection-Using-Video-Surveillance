from flask import Flask, request, jsonify, Response, render_template_string, redirect, url_for, session, send_from_directory
from flask_cors import CORS
import cv2
import os
import time
import threading
import json
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import numpy as np
import imutils
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Import your existing detection classes
from ultralytics import YOLO

# Enhanced parameters
try:
    from config.enhanced_parameters import (
        WIDTH, BEHAVIOR_CLASSES, BEHAVIOR_THRESHOLDS, BEHAVIOR_COLORS,
        HIGH_VALUE_ZONES, LOITER_TIME_THRESHOLD, TRACKING_MAX_DISAPPEARED,
        TEMPORAL_WINDOW_SIZE, quit_key, frame_name
    )
except ImportError:
    # Default values if config file is not available
    WIDTH = 640
    BEHAVIOR_CLASSES = ['normal', 'suspicious', 'concealing', 'shoplifting']
    BEHAVIOR_THRESHOLDS = {'suspicious': 0.7, 'concealing': 0.8, 'shoplifting': 0.9}
    BEHAVIOR_COLORS = {'normal': (0, 255, 0), 'suspicious': (0, 255, 255), 'concealing': (0, 165, 255), 'shoplifting': (0, 0, 255)}
    HIGH_VALUE_ZONES = {
        'electronics': ([100, 100, 300, 300], 10.0),
        'jewelry': ([400, 150, 600, 350], 15.0)
    }
    LOITER_TIME_THRESHOLD = 30.0
    TRACKING_MAX_DISAPPEARED = 30
    TEMPORAL_WINDOW_SIZE = 10
    quit_key = 'q'
    frame_name = 'Detection'

# Authentication credentials
VALID_EMAIL = "madhu25@gmail.com"
VALID_PASSWORD = "madhu123"

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_jsonify(data, **kwargs):
    """Safe JSON serialization that handles NumPy types"""
    try:
        # Convert the data to JSON-serializable format
        json_str = json.dumps(data, cls=NumpyEncoder, **kwargs)
        parsed_data = json.loads(json_str)
        return jsonify(parsed_data)
    except Exception as e:
        logging.error(f"JSON serialization error: {str(e)}")
        # Return a safe fallback
        return jsonify({'error': 'Data serialization failed', 'success': False})

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

@dataclass
class PersonTrack:
    """Represents a tracked person with behavioral history"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    last_seen: int
    behaviors: List[str]
    behavior_history: deque
    zone_entry_time: Optional[float] = None
    zone_name: Optional[str] = None
    items_detected: List[str] = None
    alert_level: str = "normal"  # normal, warning, critical
    
    def __post_init__(self):
        if self.items_detected is None:
            self.items_detected = []

class PersonTracker:
    """Enhanced person tracking with behavioral analysis"""
    
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.persons = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, bbox, behaviors, confidences):
        """Register a new person"""
        center = self._get_center(bbox)
        person = PersonTrack(
            id=self.next_id,
            bbox=bbox,
            center=center,
            last_seen=0,
            behaviors=behaviors,
            behavior_history=deque(maxlen=TEMPORAL_WINDOW_SIZE)
        )
        # Convert NumPy types to Python types for JSON serialization
        safe_confidences = [float(c) if hasattr(c, 'item') else c for c in confidences]
        person.behavior_history.append((behaviors, safe_confidences, time.time()))
        
        self.persons[self.next_id] = person
        self.next_id += 1
        return person
    
    def update(self, detections):
        """Update tracking with new detections"""
        if len(detections) == 0:
            # Mark all existing persons as disappeared
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            return list(self.persons.values())
        
        # Extract detection data
        bboxes = []
        behaviors_list = []
        confidences_list = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            conf = detection[4]
            cls = int(detection[5])
            
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            behaviors_list.append([BEHAVIOR_CLASSES[cls]])
            # Convert to native Python float
            confidences_list.append([float(conf)])
        
        # Simple distance-based matching
        if len(self.persons) == 0:
            for i, bbox in enumerate(bboxes):
                self.register(bbox, behaviors_list[i], confidences_list[i])
        else:
            self._match_detections(bboxes, behaviors_list, confidences_list)
        
        return list(self.persons.values())
    
    def _match_detections(self, bboxes, behaviors_list, confidences_list):
        """Match detections to existing tracks"""
        person_centers = [self._get_center(p.bbox) for p in self.persons.values()]
        person_ids = list(self.persons.keys())
        
        detection_centers = [self._get_center(bbox) for bbox in bboxes]
        
        # Compute distance matrix
        if len(person_centers) > 0 and len(detection_centers) > 0:
            distances = np.linalg.norm(
                np.array(person_centers)[:, np.newaxis] - np.array(detection_centers), 
                axis=2
            )
            
            # Simple greedy matching
            used_detection_idxs = set()
            used_person_idxs = set()
            
            # Find best matches
            for _ in range(min(len(person_ids), len(bboxes))):
                if distances.size == 0:
                    break
                    
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                person_idx, detection_idx = min_idx
                
                if person_idx in used_person_idxs or detection_idx in used_detection_idxs:
                    distances[min_idx] = np.inf
                    continue
                
                if distances[min_idx] < 100:  # Distance threshold
                    person_id = person_ids[person_idx]
                    self._update_person(
                        person_id, 
                        bboxes[detection_idx], 
                        behaviors_list[detection_idx],
                        confidences_list[detection_idx]
                    )
                    used_person_idxs.add(person_idx)
                    used_detection_idxs.add(detection_idx)
                
                distances[min_idx] = np.inf
            
            # Register new detections
            for i, bbox in enumerate(bboxes):
                if i not in used_detection_idxs:
                    self.register(bbox, behaviors_list[i], confidences_list[i])
            
            # Mark unmatched persons as disappeared
            for i, person_id in enumerate(person_ids):
                if i not in used_person_idxs:
                    self.disappeared[person_id] = self.disappeared.get(person_id, 0) + 1
    
    def _update_person(self, person_id, bbox, behaviors, confidences):
        """Update existing person track"""
        person = self.persons[person_id]
        person.bbox = bbox
        person.center = self._get_center(bbox)
        person.behaviors = behaviors
        # Convert NumPy types to Python types
        safe_confidences = [float(c) if hasattr(c, 'item') else c for c in confidences]
        person.behavior_history.append((behaviors, safe_confidences, time.time()))
        person.last_seen = 0
        
        # Reset disappeared counter
        if person_id in self.disappeared:
            del self.disappeared[person_id]
    
    def _get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def deregister(self, person_id):
        """Remove person from tracking"""
        if person_id in self.persons:
            del self.persons[person_id]
        if person_id in self.disappeared:
            del self.disappeared[person_id]

class ZoneMonitor:
    """Monitor high-value zones for loitering behavior"""
    
    def __init__(self, zones=None):
        self.zones = zones or HIGH_VALUE_ZONES
        self.zone_entries = {}
    
    def check_zones(self, persons):
        """Check if persons are in high-value zones"""
        alerts = []
        
        for person in persons:
            cx, cy = person.center
            current_time = time.time()
            
            # Check which zone person is in
            in_zone = None
            for zone_name, (coords, threshold) in self.zones.items():
                x1, y1, x2, y2 = coords
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    in_zone = zone_name
                    break
            
            # Handle zone entry/exit
            if in_zone:
                if person.zone_name != in_zone:
                    # Entered new zone
                    person.zone_entry_time = current_time
                    person.zone_name = in_zone
                elif person.zone_entry_time:
                    # Check for loitering
                    time_in_zone = current_time - person.zone_entry_time
                    threshold = self.zones[in_zone][1]
                    
                    if time_in_zone > threshold:
                        person.alert_level = "warning"
                        alerts.append({
                            'person_id': int(person.id),
                            'type': 'loiter_high_value_zone',
                            'zone': in_zone,
                            'duration': float(time_in_zone),
                            'severity': 'warning'
                        })
            else:
                # Exited zone
                person.zone_entry_time = None
                person.zone_name = None
        
        return alerts

class BehaviorAnalyzer:
    """Analyze behavioral patterns and generate alerts"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'suspicious': {'threshold': 0.7, 'severity': 'warning'},
            'concealing': {'threshold': 0.8, 'severity': 'critical'},
            'shoplifting': {'threshold': 0.9, 'severity': 'critical'}
        }
    
    def analyze_behaviors(self, persons):
        """Analyze person behaviors and generate alerts"""
        alerts = []
        
        for person in persons:
            if not person.behavior_history:
                continue
                
            # Get recent behaviors
            recent_behaviors, recent_confs, _ = person.behavior_history[-1]
            
            # Check each behavior against thresholds
            for behavior, conf in zip(recent_behaviors, recent_confs):
                if behavior in self.suspicious_patterns:
                    pattern = self.suspicious_patterns[behavior]
                    
                    if conf > pattern['threshold']:
                        person.alert_level = pattern['severity']
                        
                        alerts.append({
                            'person_id': int(person.id),
                            'type': behavior,
                            'confidence': float(conf),
                            'severity': pattern['severity'],
                            'timestamp': float(time.time())
                        })
        
        return alerts

class AnomalyCapture:
    """Handle anomaly detection and image capture"""
    
    def __init__(self, output_dir="anomalies"):
        self.output_dir = output_dir
        self.ensure_directory()
        self.captured_anomalies = []
    
    def ensure_directory(self):
        """Ensure anomaly directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created anomaly directory: {self.output_dir}")
    
    def is_anomaly(self, alerts):
        """Determine if current alerts constitute an anomaly"""
        if not alerts:
            return False
        
        # Check for critical alerts or multiple warnings
        critical_alerts = [alert for alert in alerts if alert.get('severity') == 'critical']
        warning_alerts = [alert for alert in alerts if alert.get('severity') == 'warning']
        
        # Anomaly conditions:
        # 1. Any critical alert (shoplifting, concealing)
        # 2. Multiple warning alerts simultaneously
        return len(critical_alerts) > 0 or len(warning_alerts) >= 2
    
    def capture_anomaly(self, frame, alerts):
        """Capture anomaly frame with metadata"""
        if not self.is_anomaly(alerts):
            return None
        
        timestamp = datetime.now()
        filename = f"anomaly_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the frame
        success = cv2.imwrite(filepath, frame)
        
        if success:
            # Convert alerts to ensure JSON serialization
            safe_alerts = convert_numpy_types(alerts)
            
            # Create metadata
            anomaly_data = {
                'filename': filename,
                'filepath': filepath,
                'timestamp': timestamp.isoformat(),
                'alerts': safe_alerts,
                'alert_count': len(alerts),
                'critical_alerts': len([a for a in alerts if a.get('severity') == 'critical']),
                'warning_alerts': len([a for a in alerts if a.get('severity') == 'warning'])
            }
            
            self.captured_anomalies.append(anomaly_data)
            
            logging.info(f"Anomaly captured: {filename}")
            logging.info(f"Alerts: {[alert['type'] for alert in alerts]}")
            
            return anomaly_data
        else:
            logging.error(f"Failed to save anomaly image: {filepath}")
            return None
    
    def get_anomaly_list(self):
        """Get list of captured anomalies with safe JSON serialization"""
        return convert_numpy_types(self.captured_anomalies)
    
    def save_anomaly_log(self):
        """Save anomaly log to JSON file"""
        if self.captured_anomalies:
            log_file = os.path.join(self.output_dir, 'anomaly_log.json')
            safe_anomalies = convert_numpy_types(self.captured_anomalies)
            with open(log_file, 'w') as f:
                json.dump(safe_anomalies, f, indent=2, cls=NumpyEncoder)
            logging.info(f"Anomaly log saved: {log_file}")

class FlaskShopliftingDetector:
    """Flask-adapted shoplifting detection system with anomaly capture"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.current_video_path = None
        self.cap = None
        
        # Initialize components
        self.tracker = PersonTracker(max_disappeared=TRACKING_MAX_DISAPPEARED)
        self.zone_monitor = ZoneMonitor()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.anomaly_capture = AnomalyCapture()
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Alert system
        self.alerts = deque(maxlen=100)
        self.frame_count = 0
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def set_video(self, video_path):
        """Set the video file for processing"""
        self.current_video_path = video_path
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        logging.info(f"Video set: {video_path}")
    
    def process_frame(self, frame):
        """Process single frame with anomaly detection"""
        # Resize frame
        frame = imutils.resize(frame, width=WIDTH)
        
        # Run inference
        results = self.model.predict(frame, verbose=False)
        
        current_alerts = []
        
        if len(results[0].boxes) > 0:
            # Extract detection data
            boxes_data = results[0].boxes.data.cpu().numpy()
            detections = []
            
            for detection in boxes_data:
                x1, y1, x2, y2, conf, cls = detection
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
            
            # Update tracking
            persons = self.tracker.update(detections)
            
            # Analyze zones and behaviors
            zone_alerts = self.zone_monitor.check_zones(persons)
            behavior_alerts = self.behavior_analyzer.analyze_behaviors(persons)
            
            # Combine alerts and ensure JSON serialization
            current_alerts = convert_numpy_types(zone_alerts + behavior_alerts)
            self.alerts.extend(current_alerts)
            
            # Check for anomalies and capture if needed
            anomaly_data = self.anomaly_capture.capture_anomaly(frame, current_alerts)
            
            # Draw annotations
            frame = self.draw_annotations(frame, persons, current_alerts, anomaly_data)
        
        return frame, len(self.alerts), current_alerts
    
    def draw_annotations(self, frame, persons, alerts, anomaly_data=None):
        """Draw enhanced annotations on frame"""
        # Draw high-value zones
        for zone_name, (coords, _) in HIGH_VALUE_ZONES.items():
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(frame, f"HV: {zone_name}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw person tracks and behaviors
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            
            # Determine color based on alert level
            if person.alert_level == "critical":
                color = (0, 0, 255)  # Red
                thickness = 3
            elif person.alert_level == "warning":
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw person ID
            cv2.putText(frame, f"ID: {person.id}", (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw behaviors
            if person.behaviors:
                behavior_text = ", ".join(person.behaviors)
                cv2.putText(frame, behavior_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center point
            cx, cy = person.center
            cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Draw zone info if in zone
            if person.zone_name:
                zone_text = f"Zone: {person.zone_name}"
                cv2.putText(frame, zone_text, (x1, y2+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw alert summary
        critical_alerts = sum(1 for alert in alerts if alert.get('severity') == 'critical')
        warning_alerts = sum(1 for alert in alerts if alert.get('severity') == 'warning')
        
        alert_text = f"Alerts - Critical: {critical_alerts}, Warning: {warning_alerts}"
        cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw anomaly capture notification
        if anomaly_data:
            cv2.putText(frame, "ANOMALY CAPTURED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(frame, f"Saved: {anomaly_data['filename']}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw recent alerts
        y_offset = 110 if anomaly_data else 60
        for alert in alerts[-3:]:  # Show last 3 alerts
            alert_text = f"ID {alert['person_id']}: {alert['type']} ({alert.get('confidence', 0):.2f})"
            color = (0, 0, 255) if alert.get('severity') == 'critical' else (0, 165, 255)
            cv2.putText(frame, alert_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 20
        
        return frame
    
    def processing_loop(self):
        """Main processing loop for threaded execution"""
        logging.info("Starting video processing thread...")
        
        while self.is_processing and self.cap is not None:
            ret, frame = self.cap.read()
            
            if not ret:
                # End of video, restart from beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            try:
                # Process frame
                processed_frame, alert_count, current_alerts = self.process_frame(frame)
                
                # Store current frame thread-safely
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                self.frame_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logging.error(f"Error in processing loop: {str(e)}")
                break
        
        logging.info("Video processing thread stopped")
    
    def start_processing(self):
        """Start video processing in background thread"""
        if self.is_processing:
            return False, "Processing is already running"
        
        if self.current_video_path is None:
            return False, "No video file set"
        
        # Reset state
        self.frame_count = 0
        self.alerts.clear()
        self.tracker = PersonTracker(max_disappeared=TRACKING_MAX_DISAPPEARED)
        
        # Start processing
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True, "Processing started successfully"
    
    def stop_processing(self):
        """Stop video processing"""
        if not self.is_processing:
            return False, "Processing is not running"
        
        self.is_processing = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        return True, "Processing stopped successfully"
    
    def get_current_frame(self):
        """Get current processed frame thread-safely"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_status(self):
        """Get current processing status"""
        return {
            'is_processing': self.is_processing,
            'frame_count': self.frame_count,
            'total_alerts': len(self.alerts),
            'has_video': self.current_video_path is not None,
            'anomalies_captured': len(self.anomaly_capture.captured_anomalies)
        }
    
    def get_anomalies(self):
        """Get captured anomalies with safe JSON serialization"""
        return self.anomaly_capture.get_anomaly_list()
    
    def save_alerts_log(self):
        """Save alerts to JSON file"""
        if self.alerts:
            safe_alerts = convert_numpy_types(list(self.alerts))
            alerts_data = {
                'timestamp': time.time(),
                'total_alerts': len(self.alerts),
                'alerts': safe_alerts
            }
            
            with open('alerts_log.json', 'w') as f:
                json.dump(alerts_data, f, indent=2, cls=NumpyEncoder)
        
        # Save anomaly log
        self.anomaly_capture.save_anomaly_log()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_processing()
        if self.cap is not None:
            self.cap.release()
        self.save_alerts_log()

# Flask application
app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
MODEL_PATH = "config/shoplifting_weights.pt"  # Update this path
detector = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_detector():
    """Initialize the detector with error handling"""
    global detector
    try:
        detector = FlaskShopliftingDetector(MODEL_PATH)
        logging.info("Detector initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize detector: {str(e)}")
        return False

def login_required(f):
    """Decorator to require login for routes"""
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/', methods=['GET'])
def login():
    """Login page"""
    login_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shoplifting Detection System - Login</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .login-container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
            }
            .login-header {
                text-align: center;
                margin-bottom: 30px;
            }
            .login-header h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .login-header p {
                color: #666;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                color: #333;
                font-weight: bold;
            }
            .form-group input {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                box-sizing: border-box;
            }
            .form-group input:focus {
                border-color: #667eea;
                outline: none;
            }
            .login-btn {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
            }
            .login-btn:hover {
                transform: translateY(-2px);
            }
            .error-message {
                background: #ffebee;
                color: #c62828;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
            }
            .security-note {
                margin-top: 20px;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 5px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="login-header">
                <h1>🛡️ Security Login</h1>
                <p>Shoplifting Detection System</p>
            </div>
            
            {% if error %}
            <div class="error-message">
                {{ error }}
            </div>
            {% endif %}
            
            <form method="POST" action="/authenticate">
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input type="email" id="email" name="email" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                
                <button type="submit" class="login-btn">Login to System</button>
            </form>
            
            <div class="security-note">
                <strong>Security Notice:</strong> This system is for authorized personnel only. 
                All access attempts are logged and monitored.
            </div>
        </div>
    </body>
    </html>
    '''
    
    error_message = request.args.get('error')
    return render_template_string(login_template, error=error_message)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handle login authentication"""
    email = request.form.get('email')
    password = request.form.get('password')
    
    if email == VALID_EMAIL and password == VALID_PASSWORD:
        session['logged_in'] = True
        session['user_email'] = email
        logging.info(f"Successful login for {email}")
        return redirect(url_for('dashboard'))
    else:
        logging.warning(f"Failed login attempt for {email}")
        return redirect(url_for('login', error="Invalid Credentials"))

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard after login"""
    dashboard_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shoplifting Detection System - Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 0;
                background: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
            }
            .user-info {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            .logout-btn {
                background: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                padding: 8px 15px;
                border-radius: 5px;
                text-decoration: none;
                transition: background 0.3s;
            }
            .logout-btn:hover {
                background: rgba(255,255,255,0.3);
            }
            .container { 
                max-width: 1200px; 
                margin: 20px auto; 
                padding: 0 20px;
            }
            .card {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                padding: 20px;
            }
            .upload-section, .control-section { 
                margin: 20px 0; 
            }
            .upload-section h3, .control-section h3 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            button { 
                padding: 12px 20px; 
                margin: 10px 5px; 
                font-size: 16px; 
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .start-btn { 
                background-color: #28a745; 
                color: white; 
            }
            .start-btn:hover {
                background-color: #218838;
                transform: translateY(-2px);
            }
            .stop-btn { 
                background-color: #dc3545; 
                color: white; 
            }
            .stop-btn:hover {
                background-color: #c82333;
                transform: translateY(-2px);
            }
            .info-btn {
                background-color: #17a2b8;
                color: white;
            }
            .info-btn:hover {
                background-color: #138496;
                transform: translateY(-2px);
            }
            .upload-btn {
                background-color: #667eea;
                color: white;
            }
            .upload-btn:hover {
                background-color: #5a6fd8;
                transform: translateY(-2px);
            }
            .status { 
                padding: 15px; 
                margin: 10px 0; 
                background-color: #e3f2fd; 
                border-left: 4px solid #2196f3;
                border-radius: 5px;
                font-family: monospace;
            }
            .video-section {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .video-feed {
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            }
            .video-feed img { 
                width: 100%; 
                height: auto;
                display: block;
            }
            .anomaly-panel {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .anomaly-panel h3 {
                margin-top: 0;
                color: #dc3545;
                border-bottom: 2px solid #dc3545;
                padding-bottom: 10px;
            }
            .anomaly-list {
                max-height: 400px;
                overflow-y: auto;
            }
            .anomaly-item {
                background: #fff5f5;
                border: 1px solid #ffcdd2;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                font-size: 12px;
            }
            .anomaly-item .timestamp {
                font-weight: bold;
                color: #d32f2f;
            }
            .anomaly-item .alerts {
                margin-top: 5px;
                color: #666;
            }
            .file-input-wrapper {
                position: relative;
                display: inline-block;
                cursor: pointer;
                margin-right: 10px;
            }
            .file-input-wrapper input[type=file] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                padding: 12px 20px;
                background-color: #667eea;
                color: white;
                border-radius: 5px;
                display: inline-block;
                cursor: pointer;
                transition: all 0.3s;
            }
            .file-input-label:hover {
                background-color: #5a6fd8;
                transform: translateY(-2px);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                margin-top: 5px;
            }
            @media (max-width: 768px) {
                .video-section {
                    grid-template-columns: 1fr;
                }
                .stats-grid {
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ Shoplifting Detection System</h1>
            <div class="user-info">
                <span>Welcome, {{ session.user_email }}</span>
                <a href="/logout" class="logout-btn">Logout</a>
            </div>
        </div>
        
        <div class="container">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="frameCount">0</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="alertCount">0</div>
                    <div class="stat-label">Total Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="anomalyCount">0</div>
                    <div class="stat-label">Anomalies Captured</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="processingStatus">Stopped</div>
                    <div class="stat-label">System Status</div>
                </div>
            </div>
            
            <div class="card">
                <div class="upload-section">
                    <h3>📹 Upload Video</h3>
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="file-input-wrapper">
                            <input type="file" id="videoFile" name="video" accept=".mp4,.avi,.mov,.mkv,.webm">
                            <label for="videoFile" class="file-input-label">Choose Video File</label>
                        </div>
                        <button type="submit" class="upload-btn">Upload Video</button>
                    </form>
                </div>
                
                <div class="control-section">
                    <h3>🎮 Detection Control</h3>
                    <button class="start-btn" onclick="startDetection()">🚀 Start Detection</button>
                    <button class="stop-btn" onclick="stopDetection()">⏹️ Stop Detection</button>
                    <button class="info-btn" onclick="getStatus()">📊 Get Status</button>
                    <button class="info-btn" onclick="viewAnomalies()">🚨 View Anomalies</button>
                </div>
            </div>
            
            <div id="status" class="status">
                Status: Ready - Upload a video and start detection
            </div>
            
            <div class="video-section">
                <div class="card video-feed">
                    <h3>🔴 Live Detection Feed</h3>
                    <img id="videoFeed" src="/video_feed" style="width: 100%;" onerror="this.style.display='none'">
                </div>
                
                <div class="anomaly-panel">
                    <h3>🚨 Recent Anomalies</h3>
                    <div id="anomalyList" class="anomaly-list">
                        <p style="text-align: center; color: #999;">No anomalies detected yet</p>
                    </div>
                    <button class="info-btn" onclick="downloadAnomalies()" style="width: 100%; margin-top: 10px;">
                        📥 Download Anomaly Report
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData();
                const fileField = document.querySelector('input[type="file"]');
                
                if (!fileField.files[0]) {
                    alert('Please select a video file first');
                    return;
                }
                
                formData.append('video', fileField.files[0]);
                
                document.getElementById('status').textContent = 'Uploading video...';
                
                fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.message;
                    if (data.success) {
                        document.getElementById('videoFeed').style.display = 'block';
                    }
                })
                .catch(error => {
                    document.getElementById('status').textContent = 'Upload failed: ' + error;
                });
            });
            
            function startDetection() {
                fetch('/start_detection', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.message;
                    if (data.success) {
                        document.getElementById('processingStatus').textContent = 'Running';
                    }
                });
            }
            
            function stopDetection() {
                fetch('/stop_detection', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.message;
                    if (data.success) {
                        document.getElementById('processingStatus').textContent = 'Stopped';
                    }
                });
            }
            
            function getStatus() {
                fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('frameCount').textContent = data.frame_count || 0;
                    document.getElementById('alertCount').textContent = data.total_alerts || 0;
                    document.getElementById('anomalyCount').textContent = data.anomalies_captured || 0;
                    document.getElementById('processingStatus').textContent = data.is_processing ? 'Running' : 'Stopped';
                    
                    document.getElementById('status').innerHTML = `
                        Processing: ${data.is_processing}<br>
                        Frame Count: ${data.frame_count}<br>
                        Total Alerts: ${data.total_alerts}<br>
                        Anomalies Captured: ${data.anomalies_captured}<br>
                        Has Video: ${data.has_video}
                    `;
                });
            }
            
            function viewAnomalies() {
                fetch('/anomalies')
                .then(response => response.json())
                .then(data => {
                    const anomalyList = document.getElementById('anomalyList');
                    if (data.anomalies && data.anomalies.length > 0) {
                        anomalyList.innerHTML = data.anomalies.slice(-10).reverse().map(anomaly => `
                            <div class="anomaly-item">
                                <div class="timestamp">${new Date(anomaly.timestamp).toLocaleString()}</div>
                                <div class="alerts">
                                    Critical: ${anomaly.critical_alerts}, Warning: ${anomaly.warning_alerts}<br>
                                    File: ${anomaly.filename}
                                </div>
                            </div>
                        `).join('');
                    } else {
                        anomalyList.innerHTML = '<p style="text-align: center; color: #999;">No anomalies detected yet</p>';
                    }
                });
            }
            
            function downloadAnomalies() {
                window.open('/download_anomalies', '_blank');
            }
            
            // Auto-refresh status and anomalies every 3 seconds
            setInterval(() => {
                getStatus();
                viewAnomalies();
            }, 3000);
            
            // Initial load
            getStatus();
            viewAnomalies();
        </script>
    </body>
    </html>
    '''
    return render_template_string(dashboard_template, session=session)

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    """Upload and set video for processing"""
    try:
        if 'video' not in request.files:
            return safe_jsonify({'success': False, 'message': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return safe_jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(file_path)
            
            if detector is None:
                if not initialize_detector():
                    return safe_jsonify({'success': False, 'message': 'Failed to initialize detector'}), 500
            
            detector.set_video(file_path)
            
            return safe_jsonify({
                'success': True, 
                'message': f'Video uploaded successfully: {filename}',
                'filename': filename
            })
        else:
            return safe_jsonify({'success': False, 'message': 'Invalid file type'}), 400
            
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return safe_jsonify({'success': False, 'message': f'Upload failed: {str(e)}'}), 500

@app.route('/start_detection', methods=['POST'])
@login_required
def start_detection():
    """Start detection processing"""
    try:
        if detector is None:
            return safe_jsonify({'success': False, 'message': 'Detector not initialized'}), 500
        
        success, message = detector.start_processing()
        return safe_jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logging.error(f"Start detection error: {str(e)}")
        return safe_jsonify({'success': False, 'message': f'Failed to start detection: {str(e)}'}), 500

@app.route('/stop_detection', methods=['POST'])
@login_required
def stop_detection():
    """Stop detection processing"""
    try:
        if detector is None:
            return safe_jsonify({'success': False, 'message': 'Detector not initialized'}), 500
        
        success, message = detector.stop_processing()
        return safe_jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logging.error(f"Stop detection error: {str(e)}")
        return safe_jsonify({'success': False, 'message': f'Failed to stop detection: {str(e)}'}), 500

@app.route('/status')
@login_required
def get_status():
    """Get current processing status"""
    try:
        if detector is None:
            return safe_jsonify({
                'is_processing': False,
                'frame_count': 0,
                'total_alerts': 0,
                'has_video': False,
                'anomalies_captured': 0,
                'message': 'Detector not initialized'
            })
        
        status = detector.get_status()
        return safe_jsonify(status)
        
    except Exception as e:
        logging.error(f"Status error: {str(e)}")
        return safe_jsonify({'error': f'Failed to get status: {str(e)}'}), 500

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                if detector is None or not detector.is_processing:
                    # Send a blank frame when not processing
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "No video processing", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    frame = blank_frame
                else:
                    frame = detector.get_current_frame()
                    if frame is None:
                        # Send a blank frame if no current frame
                        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank_frame, "Loading...", (250, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        frame = blank_frame
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logging.error(f"Video feed error: {str(e)}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anomalies')
@login_required
def get_anomalies():
    """Get captured anomalies"""
    try:
        if detector is None:
            return safe_jsonify({'anomalies': [], 'total': 0})
        
        anomalies = detector.get_anomalies()
        
        return safe_jsonify({
            'anomalies': anomalies,
            'total': len(anomalies),
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Get anomalies error: {str(e)}")
        return safe_jsonify({'success': False, 'error': f'Failed to get anomalies: {str(e)}'}), 500

@app.route('/anomaly_images/<filename>')
@login_required
def serve_anomaly_image(filename):
    """Serve anomaly images"""
    try:
        return send_from_directory('anomalies', filename)
    except Exception as e:
        logging.error(f"Serve image error: {str(e)}")
        return safe_jsonify({'error': 'Image not found'}), 404

@app.route('/download_anomalies')
@login_required
def download_anomalies():
    """Download anomaly report"""
    try:
        if detector is None:
            return safe_jsonify({'success': False, 'message': 'Detector not initialized'}), 500
        
        detector.anomaly_capture.save_anomaly_log()
        
        log_file = os.path.join('anomalies', 'anomaly_log.json')
        if os.path.exists(log_file):
            return send_from_directory('anomalies', 'anomaly_log.json', as_attachment=True)
        else:
            return safe_jsonify({'success': False, 'message': 'No anomaly log found'}), 404
            
    except Exception as e:
        logging.error(f"Download anomalies error: {str(e)}")
        return safe_jsonify({'success': False, 'error': f'Failed to download: {str(e)}'}), 500

@app.route('/alerts')
@login_required
def get_alerts():
    """Get current alerts"""
    try:
        if detector is None:
            return safe_jsonify({'alerts': [], 'total': 0})
        
        # Get recent alerts (last 20)
        recent_alerts = list(detector.alerts)[-20:] if detector.alerts else []
        safe_alerts = convert_numpy_types(recent_alerts)
        
        return safe_jsonify({
            'alerts': safe_alerts,
            'total': len(detector.alerts),
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Get alerts error: {str(e)}")
        return safe_jsonify({'success': False, 'error': f'Failed to get alerts: {str(e)}'}), 500

@app.route('/reset')
@login_required
def reset_system():
    """Reset the detection system"""
    try:
        if detector is not None:
            detector.cleanup()
        
        return safe_jsonify({'success': True, 'message': 'System reset successfully'})
        
    except Exception as e:
        logging.error(f"Reset error: {str(e)}")
        return safe_jsonify({'success': False, 'error': f'Failed to reset: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return safe_jsonify({'success': False, 'message': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return safe_jsonify({'success': False, 'message': 'Internal server error occurred.'}), 500

def cleanup_old_uploads():
    """Clean up old uploaded files (older than 24 hours)"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        current_time = time.time()
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 86400:  # 24 hours in seconds
                    os.remove(file_path)
                    logging.info(f"Removed old file: {filename}")
                    
    except Exception as e:
        logging.error(f"Cleanup error: {str(e)}")

def periodic_cleanup():
    """Run periodic cleanup in background"""
    while True:
        time.sleep(3600)  # Run every hour
        cleanup_old_uploads()

if __name__ == '__main__':
    # Initialize detector on startup
    if not initialize_detector():
        logging.error("Failed to initialize detector. Check your model path.")
        exit(1)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    try:
        # Run Flask app
        app.run(
            host='0.0.0.0',  # Listen on all interfaces
            port=5000,
            debug=False,  # Set to False in production
            threaded=True
        )
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        # Cleanup on exit
        if detector is not None:
            detector.cleanup()
        logging.info("Application stopped")