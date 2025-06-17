import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import defaultdict
from datetime import datetime

class TrafficAnalyzer:
    def __init__(self, weights_path, video_path, conf_threshold=0.4):
        self.model = YOLO(weights_path)
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.conf_threshold = conf_threshold
        self.video_path = video_path
        
        # Initialize tracking variables
        self.track_history = defaultdict(list)  # Store track history for each vehicle
        self.up_count = 0                       # Vehicles moving up
        self.down_count = 0                     # Vehicles moving down
        self.speeds = defaultdict(list)         # Store speed measurements
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        # Counting line parameters
        self.line_y = 390  # y-coordinate of the counting line
        self.crossed_ids = set()  # Keep track of vehicles that have crossed the line
        
        # Speed calculation parameters
        self.pixels_to_meters = 0.1  # Conversion factor (adjust based on your video)
        
        # Traffic density parameters
        self.density_window = 30  # Number of frames to consider for density calculation
        self.vehicle_positions = []  # Store recent vehicle positions
        self.density_thresholds = {
            'low': 3,    # Less than 3 vehicles
            'medium': 6  # Between 3 and 6 vehicles
        }
        
        # Vehicle type analysis
        self.vehicle_types = defaultdict(int)
        self.vehicle_sizes = {
            'small': (0, 1000),    # Area in pixels
            'medium': (1000, 3000),
            'large': (3000, float('inf'))
        }
        
        # Time-based analysis
        self.start_time = datetime.now()
        self.peak_hours = defaultdict(int)
        
    def calculate_speed(self, track_id, current_pos):
        """Calculate vehicle speed based on position history"""
        if len(self.track_history[track_id]) < 2:
            return None
            
        prev_pos = self.track_history[track_id][-2]
        current_pos = self.track_history[track_id][-1]
        
        # Calculate distance in pixels
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        # Convert to meters and calculate speed (m/s)
        distance_meters = distance * self.pixels_to_meters
        time_diff = 1/self.fps if self.fps > 0 else 1/30  # Time between frames
        speed = distance_meters / time_diff
        
        # Convert to km/h
        speed_kmh = speed * 3.6
        
        return speed_kmh

    def analyze_traffic_density(self, current_vehicles):
        """Analyze traffic density and return status"""
        # Update vehicle positions
        self.vehicle_positions.append(len(current_vehicles))
        if len(self.vehicle_positions) > self.density_window:
            self.vehicle_positions.pop(0)
        
        # Calculate average density
        avg_density = sum(self.vehicle_positions) / len(self.vehicle_positions)
        
        # Determine traffic status
        if avg_density < self.density_thresholds['low']:
            status = "LOW"
            color = (255, 0, 0)  # Blue
        elif avg_density < self.density_thresholds['medium']:
            status = "MEDIUM"
            color = (0, 165, 255)  # Orange
        else:
            status = "HIGH"
            color = (0, 0, 255)  # Red
            
        return status, color

    def classify_vehicle(self, x1, y1, x2, y2):
        """Classify vehicle based on size"""
        area = (x2 - x1) * (y2 - y1)
        for size, (min_area, max_area) in self.vehicle_sizes.items():
            if min_area <= area <= max_area:
                return size
        return 'unknown'

    def process_frame(self, frame):
        """Process a single frame for vehicle detection and tracking"""
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.conf_threshold)[0]
        
        # Prepare detections for DeepSORT
        dets = []
        current_vehicles = []
        
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            dets.append(([x1, y1, x2-x1, y2-y1], conf.item(), 'vehicle'))
            current_vehicles.append((x1, y1, x2, y2))
        
        # Analyze traffic density
        density_status, density_color = self.analyze_traffic_density(current_vehicles)
        
        # Update tracks
        tracks = self.tracker.update_tracks(dets, frame=frame)
        
        # Process each track
        for t in tracks:
            if not t.is_confirmed():
                continue
                
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Update track history
            self.track_history[tid].append((cx, cy))
            if len(self.track_history[tid]) > 30:  # Keep last 30 positions
                self.track_history[tid].pop(0)
            
            # Calculate speed
            speed = self.calculate_speed(tid, (cx, cy))
            if speed is not None:
                self.speeds[tid].append(speed)
            
            # Classify vehicle
            vehicle_type = self.classify_vehicle(x1, y1, x2, y2)
            self.vehicle_types[vehicle_type] += 1
            
            # Check if vehicle crossed the line
            if tid not in self.crossed_ids:
                if len(self.track_history[tid]) >= 2:
                    prev_y = self.track_history[tid][-2][1]
                    if prev_y < self.line_y <= cy:  # Vehicle moving down
                        self.down_count += 1
                        self.crossed_ids.add(tid)
                    elif prev_y > self.line_y >= cy:  # Vehicle moving up
                        self.up_count += 1
                        self.crossed_ids.add(tid)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Draw speed above vehicle
            if speed is not None:
                speed_text = f"{speed:.1f} km/h"
                # Get text size for centering
                text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = cx - text_size[0] // 2
                text_y = y1 - 5
                cv2.putText(frame, speed_text, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw thin track path
            for i in range(1, len(self.track_history[tid])):
                cv2.line(frame, 
                        self.track_history[tid][i-1],
                        self.track_history[tid][i],
                        (0, 255, 0), 1)
        
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 0, 255), 2)
        
        # Display vehicle counts
        cv2.putText(frame, f"Up: {self.up_count}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Down: {self.down_count}", 
                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Live Count: {len(current_vehicles)}", 
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        
        # Display traffic density status
        cv2.putText(frame, f"Traffic: {density_status}", 
                  (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, density_color, 2)
        
        # Update peak hours
        current_hour = datetime.now().hour
        self.peak_hours[current_hour] += len(current_vehicles)
        
        return frame

    def run_analysis(self):
        """Run the traffic analysis on the video"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = 'traffic_analysis.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"Processing video and saving to: {output_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = time.time()
                self.fps = 30 / (current_time - self.prev_time)
                self.prev_time = current_time
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", 
                      (10, processed_frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Show frame
            cv2.imshow('Traffic Analysis', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nVideo saved successfully to: {output_path}")
        
        # Print final statistics
        print("\nTraffic Analysis Results:")
        print("------------------------")
        print(f"Vehicles Moving Up: {self.up_count}")
        print(f"Vehicles Moving Down: {self.down_count}")
        print(f"Total Vehicles: {self.up_count + self.down_count}")
        
        # Print vehicle type distribution
        print("\nVehicle Type Distribution:")
        print("-------------------------")
        for vtype, count in self.vehicle_types.items():
            print(f"{vtype.capitalize()}: {count}")
        
        # Print peak hours
        print("\nPeak Hours Analysis:")
        print("-------------------")
        peak_hour = max(self.peak_hours.items(), key=lambda x: x[1])
        print(f"Peak Hour: {peak_hour[0]}:00 with {peak_hour[1]} vehicles")
        
        # Calculate average speeds
        print("\nAverage Speeds:")
        print("--------------")
        for tid, speeds in self.speeds.items():
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                print(f"Vehicle {tid}: {avg_speed:.1f} km/h")

if __name__ == "__main__":
    # Initialize and run traffic analysis
    analyzer = TrafficAnalyzer(
        weights_path="best.pt",
        video_path="traffic.mp4",
        conf_threshold=0.4
    )
    analyzer.run_analysis()
