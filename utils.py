"""
Utility functions for setting up and testing the enhanced shoplifting detection system
"""
import cv2
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from config.enhanced_parameters import HIGH_VALUE_ZONES, BEHAVIOR_CLASSES, BEHAVIOR_COLORS

class ZoneSetupTool:
    """Interactive tool to set up high-value zones"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.zones = {}
        self.current_zone = None
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_zone:
                self.zones[self.current_zone] = [
                    self.start_point[0], self.start_point[1], x, y
                ]
                print(f"Zone '{self.current_zone}' set: {self.zones[self.current_zone]}")
                self.drawing = False
    
    def setup_zones(self):
        """Interactive zone setup"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read video")
            return
        
        cv2.namedWindow('Zone Setup')
        cv2.setMouseCallback('Zone Setup', self.mouse_callback)
        
        zone_names = ['electronics', 'cosmetics', 'liquor', 'pharmacy', 'jewelry']
        zone_idx = 0
        
        print("Zone Setup Instructions:")
        print("- Click and drag to define zones")
        print("- Press SPACE to move to next zone")
        print("- Press 'q' to finish")
        
        while zone_idx < len(zone_names):
            frame_copy = frame.copy()
            self.current_zone = zone_names[zone_idx]
            
            # Draw existing zones
            for zone_name, coords in self.zones.items():
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_copy, zone_name, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show current zone being set
            cv2.putText(frame_copy, f"Setting: {self.current_zone}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Zone Setup', frame_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to next zone
                if self.current_zone in self.zones:
                    zone_idx += 1
                else:
                    print(f"Please set zone '{self.current_zone}' first")
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.zones
    
    def save_zones(self, output_path):
        """Save zones to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.zones, f, indent=4)
        print(f"Zones saved to {output_path}")

class CalibrationTool:
    """Tool for calibrating detection parameters"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.parameters = {
            'motion_threshold': 30,
            'min_contour_area': 500,
            'dwell_time_threshold': 3.0,
            'loitering_threshold': 10.0,
            'concealment_confidence': 0.6
        }
    
    def interactive_calibration(self):
        """Interactive parameter calibration"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Create trackbars
        cv2.namedWindow('Calibration')
        cv2.createTrackbar('Motion Threshold', 'Calibration', 30, 100, lambda x: None)
        cv2.createTrackbar('Min Area', 'Calibration', 500, 2000, lambda x: None)
        cv2.createTrackbar('Dwell Time (s)', 'Calibration', 30, 100, lambda x: None)
        cv2.createTrackbar('Loitering (s)', 'Calibration', 100, 300, lambda x: None)
        
        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
        print("Calibration Instructions:")
        print("- Adjust trackbars to tune parameters")
        print("- Press 's' to save current parameters")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            # Get trackbar values
            motion_thresh = cv2.getTrackbarPos('Motion Threshold', 'Calibration')
            min_area = cv2.getTrackbarPos('Min Area', 'Calibration')
            dwell_time = cv2.getTrackbarPos('Dwell Time (s)', 'Calibration') / 10.0
            loitering_time = cv2.getTrackbarPos('Loitering (s)', 'Calibration') / 10.0
            
            # Apply background subtraction
            fg_mask = background_subtractor.apply(frame)
            
            # Apply motion threshold
            _, thresh_mask = cv2.threshold(fg_mask, motion_thresh, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area and draw
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display parameters
            param_text = [
                f"Motion Thresh: {motion_thresh}",
                f"Min Area: {min_area}",
                f"Dwell Time: {dwell_time:.1f}s",
                f"Loitering: {loitering_time:.1f}s"
            ]
            
            for i, text in enumerate(param_text):
                cv2.putText(frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', frame)
            cv2.imshow('Motion Mask', thresh_mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.parameters.update({
                    'motion_threshold': motion_thresh,
                    'min_contour_area': min_area,
                    'dwell_time_threshold': dwell_time,
                    'loitering_threshold': loitering_time
                })
                print("Parameters saved!")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.parameters

class TestDataGenerator:
    """Generate test scenarios for system validation"""
    
    def __init__(self):
        self.scenarios = []
    
    def generate_normal_behavior(self, duration=30):
        """Generate normal shopping behavior data"""
        scenario = {
            'type': 'normal',
            'duration': duration,
            'behaviors': ['walking', 'browsing', 'item_selection'],
            'zone_visits': ['electronics', 'cosmetics'],
            'dwell_times': [2.5, 4.0],
            'expected_alert': False
        }
        self.scenarios.append(scenario)
        return scenario
    
    def generate_suspicious_behavior(self, behavior_type='concealment'):
        """Generate suspicious behavior scenario"""
        behavior_configs = {
            'concealment': {
                'duration': 45,
                'behaviors': ['concealment', 'looking_around', 'rapid_movement'],
                'zone_visits': ['electronics'],
                'dwell_times': [15.0],
                'expected_alert': True,
                'alert_type': 'concealment'
            },
            'loitering': {
                'duration': 60,
                'behaviors': ['loitering', 'looking_around'],
                'zone_visits': ['jewelry', 'cosmetics'],
                'dwell_times': [25.0, 20.0],
                'expected_alert': True,
                'alert_type': 'loitering'
            },
            'unusual_path': {
                'duration': 40,
                'behaviors': ['unusual_path', 'rapid_movement'],
                'zone_visits': ['electronics', 'pharmacy', 'liquor'],
                'dwell_times': [3.0, 2.0, 1.5],
                'expected_alert': True,
                'alert_type': 'unusual_behavior'
            }
        }
        
        scenario = behavior_configs.get(behavior_type, behavior_configs['concealment'])
        scenario['type'] = 'suspicious'
        self.scenarios.append(scenario)
        return scenario
    
    def export_test_suite(self, output_path):
        """Export test scenarios to JSON"""
        test_suite = {
            'created': datetime.now().isoformat(),
            'scenarios': self.scenarios,
            'validation_metrics': [
                'detection_accuracy',
                'false_positive_rate',
                'response_time',
                'zone_coverage'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(test_suite, f, indent=4)
        
        print(f"Test suite exported to {output_path}")

class PerformanceAnalyzer:
    """Analyze system performance and generate reports"""
    
    def __init__(self):
        self.metrics = {
            'detection_rate': [],
            'false_positives': [],
            'response_times': [],
            'zone_coverage': {}
        }
    
    def analyze_detection_results(self, results_file):
        """Analyze detection results from log file"""
        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found")
            return
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Calculate metrics
        total_detections = len(results.get('detections', []))
        true_positives = sum(1 for d in results.get('detections', []) 
                           if d.get('validated', False))
        
        detection_rate = true_positives / max(total_detections, 1)
        false_positive_rate = (total_detections - true_positives) / max(total_detections, 1)
        
        self.metrics['detection_rate'].append(detection_rate)
        self.metrics['false_positives'].append(false_positive_rate)
        
        print(f"Detection Rate: {detection_rate:.2%}")
        print(f"False Positive Rate: {false_positive_rate:.2%}")
        
        return {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'total_detections': total_detections,
            'true_positives': true_positives
        }
    
    def generate_performance_report(self, output_path):
        """Generate comprehensive performance report"""
        if not self.metrics['detection_rate']:
            print("No metrics data available")
            return
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Detection rate over time
        ax1.plot(self.metrics['detection_rate'], marker='o')
        ax1.set_title('Detection Rate Over Time')
        ax1.set_ylabel('Detection Rate')
        ax1.grid(True)
        
        # False positive rate
        ax2.plot(self.metrics['false_positives'], marker='s', color='red')
        ax2.set_title('False Positive Rate')
        ax2.set_ylabel('False Positive Rate')
        ax2.grid(True)
        
        # Response times histogram
        if self.metrics['response_times']:
            ax3.hist(self.metrics['response_times'], bins=20, alpha=0.7)
            ax3.set_title('Response Time Distribution')
            ax3.set_xlabel('Response Time (seconds)')
            ax3.set_ylabel('Frequency')
        
        # Zone coverage
        if self.metrics['zone_coverage']:
            zones = list(self.metrics['zone_coverage'].keys())
            coverage = list(self.metrics['zone_coverage'].values())
            ax4.bar(zones, coverage)
            ax4.set_title('Zone Coverage')
            ax4.set_ylabel('Coverage %')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path.replace('.json', '_performance.png'), dpi=300, bbox_inches='tight')
        
        # Generate text report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'avg_detection_rate': np.mean(self.metrics['detection_rate']),
                'avg_false_positive_rate': np.mean(self.metrics['false_positives']),
                'total_tests': len(self.metrics['detection_rate'])
            },
            'detailed_metrics': self.metrics,
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Performance report saved to {output_path}")
        print(f"Performance charts saved to {output_path.replace('.json', '_performance.png')}")
        
        return report
    
    def _generate_recommendations(self):
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if self.metrics['detection_rate']:
            avg_detection = np.mean(self.metrics['detection_rate'])
            if avg_detection < 0.8:
                recommendations.append("Consider lowering detection thresholds to improve sensitivity")
            
            avg_false_positive = np.mean(self.metrics['false_positives'])
            if avg_false_positive > 0.2:
                recommendations.append("Consider raising thresholds or improving behavior classification")
        
        if self.metrics['response_times']:
            avg_response = np.mean(self.metrics['response_times'])
            if avg_response > 2.0:
                recommendations.append("Optimize processing pipeline to reduce response time")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations

def setup_system_interactive():
    """Interactive system setup wizard"""
    print("=== Enhanced Shoplifting Detection System Setup ===")
    print()
    
    # Get video path
    video_path = input("Enter path to calibration video: ").strip()
    if not os.path.exists(video_path):
        print("Video file not found!")
        return
    
    # Setup zones
    print("\n1. Setting up high-value zones...")
    zone_tool = ZoneSetupTool(video_path)
    zones = zone_tool.setup_zones()
    
    if zones:
        zones_file = input("Save zones to file (default: zones.json): ").strip() or "zones.json"
        zone_tool.save_zones(zones_file)
    
    # Calibrate parameters
    print("\n2. Calibrating detection parameters...")
    calib_tool = CalibrationTool(video_path)
    parameters = calib_tool.interactive_calibration()
    
    params_file = input("Save parameters to file (default: parameters.json): ").strip() or "parameters.json"
    with open(params_file, 'w') as f:
        json.dump(parameters, f, indent=4)
    
    # Generate test scenarios
    print("\n3. Generating test scenarios...")
    test_gen = TestDataGenerator()
    
    # Generate various scenarios
    test_gen.generate_normal_behavior()
    test_gen.generate_suspicious_behavior('concealment')
    test_gen.generate_suspicious_behavior('loitering')
    test_gen.generate_suspicious_behavior('unusual_path')
    
    test_file = input("Save test suite to file (default: test_suite.json): ").strip() or "test_suite.json"
    test_gen.export_test_suite(test_file)
    
    print("\n=== Setup Complete ===")
    print(f"Zones saved to: {zones_file}")
    print(f"Parameters saved to: {params_file}")
    print(f"Test suite saved to: {test_file}")
    print("\nYou can now run the detection system with these configurations.")

if __name__ == "__main__":
    setup_system_interactive()