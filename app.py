from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json
import threading
import time
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from collections import deque
import yt_dlp
import tempfile
import shutil
from io import BytesIO

app = Flask(__name__)

# Global variables for progress tracking
progress_data = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'result': None,
    'error': None
}

class HeartRateAnalyzer:
    def __init__(self):
        self.fs = 30  # Sampling frequency in Hz
        self.f_low = 0.5  # Low cutoff frequency
        self.f_high = 2  # High cutoff frequency
        self.selected_channel = 1  # BGR = Green
        
    def bandpass_filter(self, data, lowcut=0.5, highcut=2, fs=30, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
    
    def get_peaks(self, array):
        peak_locations = find_peaks(array)[0]
        peak_values = [array[x] for x in peak_locations]
        return (peak_locations, peak_values)
    
    def cv2_downscale_frame(self, frame, levels=3):
        downscaled_frame = frame
        for _ in range(levels):
            downscaled_frame = cv2.pyrDown(downscaled_frame)
        return downscaled_frame
    
    def _initialize_tracker(self, first_frame, bbox):
        """Initialize tracker with simplified approach for better compatibility."""
        print(f"DEBUG: Trying MIL with bbox: {bbox}, frame: {first_frame.shape}")
        
        try:
            # Try creating MIL tracker - most compatible
            tracker = cv2.TrackerMIL_create()
            if tracker is None:
                print("✗ MIL tracker creation failed")
                return None
                
            # Ensure frame is grayscale for tracker (some trackers prefer this)
            if len(first_frame.shape) == 3:
                gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                # Convert back to 3-channel for consistency
                init_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            else:
                init_frame = first_frame.copy()
                
            # Validate bbox bounds
            x, y, w, h = [int(v) for v in bbox]
            frame_h, frame_w = init_frame.shape[:2]
            
            if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h or w <= 0 or h <= 0:
                print(f"✗ Invalid bbox bounds: ({x}, {y}, {w}, {h}) for frame {frame_w}x{frame_h}")
                return None
                
            # Try to initialize tracker
            success = tracker.init(init_frame, bbox)
            if success:
                print(f"✓ MIL tracker initialized successfully")
                return tracker
            else:
                print("✗ MIL tracker init returned False")
                return None
                
        except Exception as e:
            print(f"✗ MIL tracker failed with exception: {e}")
            return None
    
    def _extract_consistent_region(self, frame, crop, target_width, target_height, tracked_bbox=None, rotation=0):
        """Extract region with consistent dimensions, supporting rotation."""
        frame_height, frame_width = frame.shape[:2]
        
        if tracked_bbox:
            # Use tracking result and allow scale change while preserving aspect ratio
            if len(tracked_bbox) >= 5:  # Rotated rectangle: (center_x, center_y, width, height, angle)
                center_x, center_y, w_t, h_t, angle = tracked_bbox[:5]
                center_x, center_y, w_t, h_t = [int(v) for v in [center_x, center_y, w_t, h_t]]
                rotation = float(angle)
            else:  # Regular rectangle: (x, y, width, height)
                x_t, y_t, w_t, h_t = [int(v) for v in tracked_bbox]
                center_x = x_t + w_t // 2
                center_y = y_t + h_t // 2

            # Compute scale relative to original target dimensions
            scale_w = w_t / target_width
            scale_h = h_t / target_height
            scale = max(scale_w, scale_h)
            region_w = int(target_width * scale)
            region_h = int(target_height * scale)

            # Maintain original aspect ratio
            aspect = target_width / target_height
            if abs((region_w / region_h) - aspect) > 0.01:
                region_h = int(region_w / aspect)

            # Extract rotated region if rotation is significant
            if abs(rotation) > 5:  # Only apply rotation if > 5 degrees
                try:
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation, 1.0)
                    
                    # Calculate rotated bounding box
                    cos_angle = abs(rotation_matrix[0, 0])
                    sin_angle = abs(rotation_matrix[0, 1])
                    new_w = int((region_h * sin_angle) + (region_w * cos_angle))
                    new_h = int((region_h * cos_angle) + (region_w * sin_angle))
                    
                    # Ensure minimum size
                    new_w = max(new_w, target_width)
                    new_h = max(new_h, target_height)
                    
                    # Adjust translation to keep center
                    rotation_matrix[0, 2] += (new_w / 2) - center_x
                    rotation_matrix[1, 2] += (new_h / 2) - center_y
                    
                    # Apply rotation to entire frame
                    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (new_w, new_h))
                    
                    # Extract centered region with bounds checking
                    start_x = max(0, (new_w - region_w) // 2)
                    start_y = max(0, (new_h - region_h) // 2)
                    end_x = min(new_w, start_x + region_w)
                    end_y = min(new_h, start_y + region_h)
                    
                    # Ensure we have valid bounds
                    if end_x > start_x and end_y > start_y:
                        cropped_region = rotated_frame[start_y:end_y, start_x:end_x]
                    else:
                        # Fallback to regular extraction
                        raise ValueError("Invalid rotation bounds")
                        
                    # Store rotated rectangle info for overlay
                    bbox_info = {
                        'center_x': center_x, 'center_y': center_y,
                        'width': region_w, 'height': region_h, 'angle': rotation
                    }
                except Exception as e:
                    print(f"DEBUG: Rotation extraction failed: {e}, using regular rectangle")
                    # Fall back to regular rectangle extraction
                    x = center_x - region_w // 2
                    y = center_y - region_h // 2
                    x = max(0, min(x, frame_width - region_w))
                    y = max(0, min(y, frame_height - region_h))
                    cropped_region = frame[y:y+region_h, x:x+region_w]
                    bbox_info = (x, y, region_w, region_h)
            else:
                # Regular rectangle extraction (no rotation)
                x = center_x - region_w // 2
                y = center_y - region_h // 2
                # Clamp
                x = max(0, min(x, frame_width - region_w))
                y = max(0, min(y, frame_height - region_h))
                # Crop region
                cropped_region = frame[y:y+region_h, x:x+region_w]
                bbox_info = (x, y, region_w, region_h)
                
            # Final fallback and size validation
            if cropped_region.size == 0 or cropped_region.shape[0] == 0 or cropped_region.shape[1] == 0:
                # Fallback to static crop
                x, y = crop['x'], crop['y']
                x = max(0, min(x, frame_width - target_width))
                y = max(0, min(y, frame_height - target_height))
                cropped_region = frame[y:y+target_height, x:x+target_width]
                bbox_info = (x, y, target_width, target_height)
                
            # CRITICAL: Always resize to exact target dimensions for consistent array shapes
            if cropped_region.shape[:2] != (target_height, target_width):
                cropped_region = cv2.resize(cropped_region, (target_width, target_height))
            
            return cropped_region, bbox_info
        else:
            # Use static crop position
            x, y = crop['x'], crop['y']
            x = max(0, min(x, frame_width - target_width))
            y = max(0, min(y, frame_height - target_height))
            cropped_region = frame[y:y+target_height, x:x+target_width]
            if cropped_region.shape[:2] != (target_height, target_width):
                cropped_region = cv2.resize(cropped_region, (target_width, target_height))
            return cropped_region, (x, y, target_width, target_height)
    
    def _template_match_update(self, frame_gray):
        """Template matching fallback - REMOVED to prevent jerkiness"""
        return None
    
    def read_from_video(self, filepath, start_time=None, end_time=None, crop=None, enable_tracking=True):
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame ranges
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        array = []
        frame_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Store target dimensions for consistent frame sizing
        target_height, target_width = None, None
        
        # Store tracking coordinates for overlay video
        self.tracking_coordinates = []
        
        # Initialize tracker variable - will be set based on conditions
        tracker = None
        
        # Initialize tracker if crop is specified and tracking is enabled
        if crop and enable_tracking:
            # Set target dimensions from crop
            target_width, target_height = crop['width'], crop['height']
            
            # Read first frame to initialize tracker
            ret, first_frame = cap.read()
            if ret:
                # Validate bounding box first
                bbox = (crop['x'], crop['y'], crop['width'], crop['height'])
                frame_h, frame_w = first_frame.shape[:2]
                if (bbox[0] < 0 or bbox[1] < 0 or 
                    bbox[0] + bbox[2] > frame_w or bbox[1] + bbox[3] > frame_h or
                    bbox[2] <= 0 or bbox[3] <= 0):
                    print(f"DEBUG: Invalid bounding box - out of frame bounds")
                    tracker = None
                    progress_data['message'] = f'Tracking failed - Invalid crop region'
                else:
                    # Try to initialize tracker with fallback options (includes init)
                    tracker = self._initialize_tracker(first_frame, bbox)
                    if tracker:
                        progress_data['message'] = f'Dynamic tracking initialized - Following selected region'
                    else:
                        progress_data['message'] = f'Tracking unavailable - Using static crop (install opencv-contrib-python for tracking)'
                
                # Process first frame with consistent dimensions
                cropped_frame, tracked_bbox = self._extract_consistent_region(first_frame, crop, target_width, target_height)
                array.append(cropped_frame)
                frame_count += 1
                
                # Store initial tracking coordinates
                self.tracking_coordinates.append({
                    'x': crop['x'], 
                    'y': crop['y'], 
                    'width': crop['width'], 
                    'height': crop['height']
                })
                
                # Remove template matching setup
                
                print(f"DEBUG: First frame processed, tracker initialized: {tracker is not None}")
                
                frame_count = 0
        elif crop:
            # Static crop mode (tracking disabled)
            target_width, target_height = crop['width'], crop['height']
        
        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame based on configuration
            if crop:
                # Set target dimensions from crop if not already set
                if target_width is None:
                    target_width, target_height = crop['width'], crop['height']
                    
                # Debug tracker state at start of main loop
                if frame_count == 1:  # Only log on first frame of main loop
                    print(f"DEBUG: Main loop - tracker available: {tracker is not None}")
                    if tracker:
                        print(f"DEBUG: Main loop - tracker type: {type(tracker)}")
                    
                if tracker:
                    # Update tracker
                    success, bbox = tracker.update(frame)
                    frame_count += 1
                    
                    if success and bbox is not None:
                        # Estimate rotation if we have a previous frame
                        rotation = 0
                        if hasattr(self, '_prev_frame') and self._prev_frame is not None:
                            rotation = self._estimate_rotation(self._prev_frame, frame, bbox)
                            # Smooth rotation to prevent jitter
                            if hasattr(self, '_prev_rotation'):
                                alpha_rot = 0.8  # Higher smoothing for rotation
                                rotation = alpha_rot * self._prev_rotation + (1 - alpha_rot) * rotation
                            self._prev_rotation = rotation
                        
                        # Smooth tracking coordinates to reduce jitter
                        x_new, y_new, w_new, h_new = [int(v) for v in bbox]
                        
                        if hasattr(self, '_last_smooth_coords'):
                            # Exponential smoothing with alpha=0.7
                            alpha = 0.7
                            x_smooth = int(alpha * self._last_smooth_coords[0] + (1-alpha) * x_new)
                            y_smooth = int(alpha * self._last_smooth_coords[1] + (1-alpha) * y_new) 
                            w_smooth = int(alpha * self._last_smooth_coords[2] + (1-alpha) * w_new)
                            h_smooth = int(alpha * self._last_smooth_coords[3] + (1-alpha) * h_new)
                        else:
                            x_smooth, y_smooth, w_smooth, h_smooth = x_new, y_new, w_new, h_new
                            
                        # Update smoothed coordinates
                        self._last_smooth_coords = (x_smooth, y_smooth, w_smooth, h_smooth)
                        
                        # Create extended bbox with rotation if significant
                        if abs(rotation) > 5:  # Only include rotation if > 5 degrees
                            center_x = x_smooth + w_smooth // 2
                            center_y = y_smooth + h_smooth // 2
                            smooth_bbox = (center_x, center_y, w_smooth, h_smooth, rotation)
                        else:
                            smooth_bbox = (x_smooth, y_smooth, w_smooth, h_smooth)
                        
                        processed_frame, final_bbox = self._extract_consistent_region(frame, crop, target_width, target_height, smooth_bbox)
                        
                        # Store actual tracking coordinates for overlay (handle both regular and rotated)
                        if isinstance(final_bbox, dict):  # Rotated rectangle
                            self.tracking_coordinates.append({
                                'type': 'rotated',
                                'center_x': final_bbox['center_x'],
                                'center_y': final_bbox['center_y'], 
                                'width': final_bbox['width'],
                                'height': final_bbox['height'],
                                'angle': final_bbox['angle']
                            })
                        else:  # Regular rectangle
                            self.tracking_coordinates.append({
                                'type': 'regular',
                                'x': final_bbox[0], 
                                'y': final_bbox[1], 
                                'width': final_bbox[2], 
                                'height': final_bbox[3]
                            })
                        
                        # Store current frame for next rotation estimation
                        self._prev_frame = frame.copy()
                        
                        if frame_count < 5:  # Only log first few frames
                            if abs(rotation) > 1:
                                print(f"DEBUG: Frame {frame_count}, tracker success, rotation: {rotation:.1f}°")
                            else:
                                print(f"DEBUG: Frame {frame_count}, tracker success, no rotation")
                    else:
                        # Tracker lost - fall back to static crop  
                        processed_frame, _ = self._extract_consistent_region(frame, crop, target_width, target_height)
                        
                        # Store static coordinates
                        self.tracking_coordinates.append({
                            'type': 'regular',
                            'x': crop['x'], 
                            'y': crop['y'], 
                            'width': crop['width'], 
                            'height': crop['height']
                        })
                        
                        if frame_count < 5:
                            print(f"DEBUG: Frame {frame_count}, tracker LOST - using static crop")
                else:
                    # No tracker available - use static crop only
                    processed_frame, _ = self._extract_consistent_region(frame, crop, target_width, target_height)
                    
                    # Store static coordinates
                    self.tracking_coordinates.append({
                        'type': 'regular',
                        'x': crop['x'], 
                        'y': crop['y'], 
                        'width': crop['width'], 
                        'height': crop['height']
                    })
                    
                    if frame_count < 5:  # Only log first few frames
                        print(f"DEBUG: Frame {frame_count}, using STATIC crop")
                
                # CRITICAL: Final shape validation before adding to array
                if processed_frame.shape[:2] != (target_height, target_width):
                    print(f"DEBUG: Frame shape mismatch detected: {processed_frame.shape[:2]} != ({target_height}, {target_width}), fixing...")
                    # OpenCV resize expects (width, height), numpy shape is (height, width)
                    processed_frame = cv2.resize(processed_frame, (target_width, target_height))
                    print(f"DEBUG: Frame resized to: {processed_frame.shape[:2]}")
                
                # Double-check the shape is now correct
                if processed_frame.shape[:2] != (target_height, target_width):
                    print(f"ERROR: Frame still has wrong shape after resize: {processed_frame.shape[:2]}")
                    # Force correct shape with exact dimensions
                    processed_frame = cv2.resize(processed_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                
                array.append(processed_frame)
            
            else:
                # No crop - ensure consistent full frame dimensions
                if target_width is None:
                    target_height, target_width = frame.shape[:2]
                    print(f"Setting target dimensions for full frame: {target_width}x{target_height}")
                
                # Ensure all frames have the same dimensions
                if frame.shape[:2] != (target_height, target_width):
                    print(f"Frame size mismatch: expected {(target_height, target_width)}, got {frame.shape[:2]}. Resizing...")
                    frame = cv2.resize(frame, (target_width, target_height))
                
                array.append(frame)
                
                # Store full frame coordinates for overlay video
                self.tracking_coordinates.append({
                    'type': 'regular',
                    'x': 0, 
                    'y': 0, 
                    'width': target_width, 
                    'height': target_height
                })
            
            frame_count += 1
            
            progress = (frame_count / (end_frame - start_frame)) * 30
            progress_data['progress'] = progress
            if tracker:
                progress_data['message'] = f'Dynamic tracking: {frame_count}/{end_frame - start_frame}'
            else:
                progress_data['message'] = f'Reading video frames: {frame_count}/{end_frame - start_frame}'
            
        cap.release()
        return np.array(array)
    
    def analyze_video(self, video_path, start_time=None, end_time=None, crop=None, enable_tracking=True):
        try:
            progress_data['status'] = 'processing'
            progress_data['progress'] = 0
            progress_data['message'] = 'Starting video analysis...'
            
            # Read video with constraints
            face = self.read_from_video(video_path, start_time, end_time, crop, enable_tracking)
            
            if face.size == 0:
                raise ValueError("No frames to analyze")
            
            # Smart downscaling based on crop usage
            progress_data['message'] = 'Optimizing frame resolution...'
            
            # If using crop, minimal or no downscaling needed
            if crop:
                # Light downscaling only (1 level = half resolution)
                downscaled_faces = []
                for i, frame in enumerate(face):
                    downscaled_frame = self.cv2_downscale_frame(frame, levels=1)
                    downscaled_faces.append(downscaled_frame)
                    progress = 30 + (i / len(face)) * 20  # 30-50% progress
                    progress_data['progress'] = progress
                    progress_data['message'] = f'Light optimization: {i+1}/{len(face)}'
                face = np.array(downscaled_faces)
            else:
                # Full frame needs more downscaling for performance
                downscaled_faces = []
                for i, frame in enumerate(face):
                    downscaled_frame = self.cv2_downscale_frame(frame, levels=2)  # Reduced from 3
                    downscaled_faces.append(downscaled_frame)
                    progress = 30 + (i / len(face)) * 20  # 30-50% progress
                    progress_data['progress'] = progress
                    progress_data['message'] = f'Optimizing resolution: {i+1}/{len(face)}'
                face = np.array(downscaled_faces)
            
            # Apply bandpass filter and create magnified video
            progress_data['message'] = 'Applying bandpass filter...'
            filtered_data = np.empty((face.shape[0], face.shape[1], face.shape[2]))
            magnified_frames = []
            
            for i in range(face.shape[1]):
                for j in range(face.shape[2]):
                    one_pixel = face[:, i, j, self.selected_channel]
                    filtered_data[:, i, j] = self.bandpass_filter(one_pixel, self.f_low, self.f_high, self.fs, order=5)
                
                progress = 50 + (i / face.shape[1]) * 25  # 50-75% progress
                progress_data['progress'] = progress
                progress_data['message'] = f'Filtering pixels: {i+1}/{face.shape[1]}'
            
            # Create magnified frames
            progress_data['message'] = 'Creating magnified video...'
            for frame_idx in range(face.shape[0]):
                # Apply amplification only to the green channel (self.selected_channel)
                amplified_frame = face[frame_idx].copy().astype(np.float64)
                amplified_frame[:, :, self.selected_channel] += (filtered_data[frame_idx] * 50)  # Amplification factor
                amplified_frame = np.clip(amplified_frame, 0, 255).astype(np.uint8)
                magnified_frames.append(amplified_frame)
                
                progress = 75 + (frame_idx / face.shape[0]) * 5  # 75-80% progress
                progress_data['progress'] = progress
            
            # Calculate heart rate
            progress_data['message'] = 'Calculating heart rate...'
            progress_data['progress'] = 80
            
            average_filtered_signal = np.mean(filtered_data, axis=(1, 2))
            peak_locations, peak_values = self.get_peaks(average_filtered_signal)
            
            # Heart Rate Calculations
            if len(peak_locations) > 1:
                number_of_frames = peak_locations[-1] - peak_locations[0]
                number_of_heartbeats = len(peak_locations)
                bpm = round(number_of_heartbeats * 60 / (number_of_frames / self.fs))
            else:
                bpm = 0
            
            # Generate plots
            progress_data['message'] = 'Generating visualization...'
            progress_data['progress'] = 90
            
            plt.style.use('dark_background')
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Create time axis (adjust for start time offset)
            time_offset = start_time if start_time else 0
            time_axis = (np.arange(len(average_filtered_signal)) / self.fs) + time_offset
            
            # Plot 1: Signal with peaks
            ax1.plot(time_axis, average_filtered_signal, 'g', label='Green Channel Signal', linewidth=2)
            ax1.scatter((peak_locations / self.fs) + time_offset, peak_values, color='red', marker='x', s=100, label='Detected Peaks')
            
            title = f'Heart Rate Analysis - Average {bpm} BPM'
            if crop:
                title += f' (Cropped Region)'
            ax1.set_title(title, fontsize=16, color='white')
            ax1.set_xlabel('Time (seconds)', fontsize=12, color='white')
            ax1.set_ylabel('Color Intensity', fontsize=12, color='white')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: BPM over time (rolling 6-beat calculation)
            if len(peak_locations) >= 6:
                bpm_times = []
                bpm_values = []
                
                for i in range(5, len(peak_locations)):
                    # Calculate BPM from last 6 beats
                    recent_peaks = peak_locations[i-5:i+1]
                    if len(recent_peaks) >= 6:
                        # Time span of 5 intervals (6 beats)
                        time_span = (recent_peaks[-1] - recent_peaks[0]) / self.fs
                        # 5 intervals = 5 heartbeats in time_span seconds
                        instant_bpm = (5 * 60) / time_span
                        
                        bpm_times.append((recent_peaks[-1] / self.fs) + time_offset)  # Time of latest peak
                        bpm_values.append(instant_bpm)
                
                if bpm_times:
                    ax2.plot(bpm_times, bpm_values, 'cyan', linewidth=3, label='Real-time BPM')
                    ax2.scatter(bpm_times, bpm_values, color='red', s=30, alpha=0.7)
                    
                    # Add average line
                    ax2.axhline(y=bpm, color='yellow', linestyle='--', linewidth=2, label=f'Average: {bpm} BPM')
                    
                    # Set y-axis limits for 60-160 BPM range
                    ax2.set_ylim(60, 160)
                    ax2.set_xlim(min(time_axis), max(time_axis))
                    
                    # Add target zones
                    ax2.axhspan(60, 100, alpha=0.1, color='green', label='Resting (60-100)')
                    ax2.axhspan(100, 140, alpha=0.1, color='orange', label='Elevated (100-140)')
                    ax2.axhspan(140, 160, alpha=0.1, color='red', label='High (140-160)')
                else:
                    ax2.text(0.5, 0.5, 'Not enough peaks for BPM calculation', 
                            transform=ax2.transAxes, ha='center', va='center', 
                            fontsize=14, color='white')
            else:
                ax2.text(0.5, 0.5, 'Not enough peaks detected', 
                        transform=ax2.transAxes, ha='center', va='center', 
                        fontsize=14, color='white')
            
            ax2.set_title('Heart Rate Over Time (Rolling 6-Beat Calculation)', fontsize=16, color='white')
            ax2.set_xlabel('Time (seconds)', fontsize=12, color='white')
            ax2.set_ylabel('Heart Rate (BPM)', fontsize=12, color='white')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # Format results
            analysis_range = "Full Video"
            if start_time is not None or end_time is not None:
                start_str = self.format_time(start_time) if start_time else "0:00"
                end_str = self.format_time(end_time) if end_time else "End"
                analysis_range = f"{start_str} - {end_str}"
            
            crop_region = "Full Frame"
            if crop:
                crop_region = f"{crop['width']}x{crop['height']} region"
            
            # Create video files
            progress_data['message'] = 'Creating video files...'
            progress_data['progress'] = 95
            
            # Save magnified video
            magnified_video_path = self.create_magnified_video(magnified_frames, video_path)
            
            # Create overlay video with heart rate and inset magnified view
            overlay_video_path = self.create_overlay_video(video_path, magnified_video_path, peak_locations, start_time, end_time, crop, enable_tracking)
            
            progress_data['progress'] = 100
            progress_data['message'] = 'Analysis complete!'
            progress_data['status'] = 'completed'
            progress_data['result'] = {
                'bpm': bpm,
                'plot_url': plot_url,
                'peaks_count': len(peak_locations),
                'video_duration': len(average_filtered_signal) / self.fs,
                'analysis_range': analysis_range,
                'crop_region': crop_region,
                'magnified_video': magnified_video_path,
                'overlay_video': overlay_video_path
            }
            
            return True
            
        except Exception as e:
            progress_data['status'] = 'error'
            progress_data['error'] = str(e)
            return False
    
    def format_time(self, seconds):
        if seconds is None:
            return None
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def create_magnified_video(self, magnified_frames, original_video_path):
        """Create a video file from magnified frames"""
        try:
            temp_output = 'static/videos/magnified_temp.avi'
            final_output = 'static/videos/magnified_output.mp4'
            os.makedirs('static/videos', exist_ok=True)
            
            if magnified_frames:
                height, width = magnified_frames[0].shape[:2]
                
                # Use MJPG codec first (most reliable)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(temp_output, fourcc, float(self.fs), (width, height))
                
                if not out.isOpened():
                    print("Could not open video writer")
                    return None
                
                for frame in magnified_frames:
                    # Ensure frame is in correct format
                    if len(frame.shape) == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    out.write(frame_bgr)
                
                out.release()
                print(f"Temporary magnified video created: {temp_output}")
                
                # Try to convert to web-compatible MP4 using ffmpeg
                try:
                    import subprocess
                    cmd = [
                        'ffmpeg', '-y', '-i', temp_output,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', final_output
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        os.remove(temp_output)  # Clean up temp file
                        print(f"Converted magnified video created: {final_output}")
                        return final_output
                    else:
                        print(f"FFmpeg conversion failed: {result.stderr}")
                        # Fallback to original file
                        os.rename(temp_output, final_output)
                        return final_output
                except (FileNotFoundError, subprocess.CalledProcessError) as e:
                    print(f"FFmpeg not available: {e}")
                    # Fallback to original file
                    os.rename(temp_output, final_output)
                    return final_output
                    
        except Exception as e:
            print(f"Error creating magnified video: {e}")
            return None
        
        return None
    
    def create_overlay_video(self, original_video_path, magnified_video_path, peak_locations, start_time=None, end_time=None, crop=None, enable_tracking=True):
        """Create a video with heart rate overlay"""
        try:
            temp_output = 'static/videos/overlay_temp.avi'
            final_output = 'static/videos/overlay_output.mp4'
            os.makedirs('static/videos', exist_ok=True)
            
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_time * fps) if start_time else 0
            end_frame = int(end_time * fps) if end_time else total_frames
            
            # Use MJPG codec for temporary file
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("Could not open overlay video writer")
                return None
            
            # Calculate BPM for each frame  
            frame_bpms = self.calculate_frame_bpms(peak_locations, end_frame - start_frame, self.fs)

            # Prepare magnified video capture for PIP
            mag_cap = None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = 0
            
            # Debug: Print tracking coordinates info
            if hasattr(self, 'tracking_coordinates'):
                print(f"Overlay video: Found {len(self.tracking_coordinates)} tracking coordinate entries")
                if len(self.tracking_coordinates) > 0:
                    print(f"First coordinates: {self.tracking_coordinates[0]}")
                    if len(self.tracking_coordinates) > 1:
                        print(f"Second coordinates: {self.tracking_coordinates[1]}")
                    if len(self.tracking_coordinates) > 10:
                        print(f"10th coordinates: {self.tracking_coordinates[10]}")
                    if len(self.tracking_coordinates) > 50:
                        print(f"50th coordinates: {self.tracking_coordinates[50]}")
                        
                    # Check if coordinates are actually changing
                    unique_coords = set()
                    for i in range(min(100, len(self.tracking_coordinates))):
                        if self.tracking_coordinates[i] is not None:
                            coord_tuple = (self.tracking_coordinates[i]['x'], self.tracking_coordinates[i]['y'])
                            unique_coords.add(coord_tuple)
                    print(f"Number of unique coordinate positions in first 100 frames: {len(unique_coords)}")
            else:
                print("Overlay video: No tracking_coordinates found")
            
            while cap.isOpened() and frame_idx < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current BPM for this frame
                current_bpm = frame_bpms[frame_idx] if frame_idx < len(frame_bpms) else 0
                
                # Add heart rate overlay
                overlay_text = "HR: -- BPM" if current_bpm < 60 else f"HR: {current_bpm:.0f} BPM"
                
                # Create overlay background
                (text_width, text_height), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(frame, (10, height - 60), (text_width + 20, height - 10), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, height - 60), (text_width + 20, height - 10), (0, 255, 255), 2)
                
                # Add text
                cv2.putText(frame, overlay_text, (15, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Draw tracking rectangle
                if frame_idx < len(self.tracking_coordinates):
                    coord = self.tracking_coordinates[frame_idx]
                    
                    if coord.get('type') == 'rotated':
                        # Draw rotated rectangle
                        center_x = coord['center_x']
                        center_y = coord['center_y'] 
                        width = coord['width']
                        height = coord['height']
                        angle = coord['angle']
                        
                        # Create rotated rectangle
                        rect = ((center_x, center_y), (width, height), angle)
                        box_points = cv2.boxPoints(rect)
                        box_points = np.int0(box_points)
                        
                        # Draw rotated rectangle with green lines
                        cv2.polylines(frame, [box_points], True, (0, 255, 0), 2)
                    else:
                        # Draw regular rectangle
                        x, y, w, h = coord['x'], coord['y'], coord['width'], coord['height']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # Fallback - draw at original crop position (no text)
                    x, y, w, h = crop['x'], crop['y'], crop['width'], crop['height']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # ---- Magnified video PIP ----
                if magnified_video_path:
                    if mag_cap is None:
                        mag_cap = cv2.VideoCapture(magnified_video_path)
                    ret_mag, mag_frame = mag_cap.read()
                    if not ret_mag:
                        # restart to loop frames if PIP shorter
                        mag_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_mag, mag_frame = mag_cap.read()
                    if ret_mag:
                        m_h, m_w = mag_frame.shape[:2]
                        max_w = int(width * 0.33)
                        max_h = int(height * 0.33)
                        scale = min(max_w / m_w, max_h / m_h)
                        new_w = max(1, int(m_w * scale))
                        new_h = max(1, int(m_h * scale))
                        mag_resized = cv2.resize(mag_frame, (new_w, new_h))
                        # Draw border
                        cv2.rectangle(frame, (8, 8), (8 + new_w + 4, 8 + new_h + 4), (0, 255, 255), 2)
                        frame[10:10+new_h, 10:10+new_w] = mag_resized
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            print(f"Temporary overlay video created: {temp_output}")
            
            # Try to convert to web-compatible MP4 with synchronized audio using ffmpeg
            try:
                import subprocess
                
                # Build FFmpeg command with audio synchronization
                cmd = ['ffmpeg', '-y', '-i', temp_output]
                
                # Add audio input with time offset if custom range is used
                if start_time is not None:
                    cmd.extend(['-ss', str(start_time), '-i', original_video_path])
                else:
                    cmd.extend(['-i', original_video_path])
                
                # Add duration limit if end time is specified
                if start_time is not None and end_time is not None:
                    duration = end_time - start_time
                    cmd.extend(['-t', str(duration)])
                
                # Add encoding and mapping parameters
                cmd.extend([
                    '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p',
                    '-map', '0:v:0', '-map', '1:a:0', '-shortest',
                    '-movflags', '+faststart', final_output
                ])
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    os.remove(temp_output)  # Clean up temp file
                    print(f"Converted overlay video created: {final_output}")
                    return final_output
                else:
                    print(f"FFmpeg conversion failed: {result.stderr}")
                    # Fallback to original file
                    os.rename(temp_output, final_output)
                    return final_output
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"FFmpeg not available: {e}")
                # Fallback to original file
                os.rename(temp_output, final_output)
                return final_output
            
        except Exception as e:
            print(f"Error creating overlay video: {e}")
            return None
    
    def calculate_frame_bpms(self, peak_locations, total_frames, sampling_rate):
        """Calculate BPM for each frame using rolling 6-beat calculation"""
        frame_bpms = []
        
        if len(peak_locations) < 6:
            return [0] * total_frames
        
        for frame_idx in range(total_frames):
            current_sample = frame_idx
            
            # Find recent peaks (within reasonable sample window)
            recent_peaks = []
            sample_window = sampling_rate * 15  # 15 second window in samples
            
            for peak in peak_locations:
                if abs(peak - current_sample) <= sample_window:
                    recent_peaks.append(peak)
            
            if len(recent_peaks) >= 6:
                # Use the 6 most recent peaks
                recent_peaks = sorted(recent_peaks)[-6:]
                time_span = (recent_peaks[-1] - recent_peaks[0]) / sampling_rate
                if time_span > 0:
                    bpm = (5 * 60) / time_span  # 5 intervals between 6 peaks
                    frame_bpms.append(max(40, min(200, bpm)))  # Clamp to reasonable range
                else:
                    frame_bpms.append(0)
            else:
                frame_bpms.append(0)
        
        return frame_bpms

    def _estimate_rotation(self, prev_frame, curr_frame, bbox):
        """Estimate rotation angle using optical flow on corner points."""
        try:
            x, y, w, h = [int(v) for v in bbox]
            
            # Define corner points of the rectangle
            corners = np.float32([
                [x, y],           # Top-left
                [x + w, y],       # Top-right  
                [x + w, y + h],   # Bottom-right
                [x, y + h]        # Bottom-left
            ]).reshape(-1, 1, 2)
            
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
            
            # Calculate optical flow for corner points
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            new_corners, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, **lk_params)
            
            # Only use successfully tracked corners
            good_corners = corners[status == 1]
            good_new_corners = new_corners[status == 1]
            
            if len(good_corners) >= 3:  # Need at least 3 points for rotation
                # Calculate centroid of old and new corners
                old_center = np.mean(good_corners, axis=0)
                new_center = np.mean(good_new_corners, axis=0)
                
                # Calculate rotation using first two good corners
                if len(good_corners) >= 2:
                    # Vector from center to first corner (old)
                    old_vec = good_corners[0] - old_center
                    new_vec = good_new_corners[0] - new_center
                    
                    # Calculate angle between vectors
                    old_angle = np.arctan2(old_vec[1], old_vec[0])
                    new_angle = np.arctan2(new_vec[1], new_vec[0])
                    
                    rotation_radians = new_angle - old_angle
                    rotation_degrees = np.degrees(rotation_radians)
                    
                    # Normalize angle to [-180, 180]
                    while rotation_degrees > 180:
                        rotation_degrees -= 360
                    while rotation_degrees < -180:
                        rotation_degrees += 360
                        
                    return rotation_degrees
                    
        except Exception as e:
            print(f"DEBUG: Rotation estimation failed: {e}")
            
        return 0  # No rotation detected

def download_youtube_video(url, output_path):
    try:
        progress_data['status'] = 'downloading'
        progress_data['progress'] = 0
        progress_data['message'] = 'Downloading YouTube video...'
        
        ydl_opts = {
            'format': 'mp4[height<=720]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return True
    except Exception as e:
        progress_data['status'] = 'error'
        progress_data['error'] = f'Error downloading video: {str(e)}'
        return False

def process_video_thread(youtube_url, timestamps=None, crop=None, enable_tracking=True):
    temp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(temp_dir, 'video.%(ext)s')
        
        # Download video
        if not download_youtube_video(youtube_url, video_path):
            return
        
        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith('video.')]
        if not downloaded_files:
            progress_data['status'] = 'error'
            progress_data['error'] = 'No video file found after download'
            return
        
        actual_video_path = os.path.join(temp_dir, downloaded_files[0])
        
        # Analyze video with constraints
        analyzer = HeartRateAnalyzer()
        start_time = timestamps.get('start') if timestamps else None
        end_time = timestamps.get('end') if timestamps else None
        analyzer.analyze_video(actual_video_path, start_time, end_time, crop, enable_tracking)
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    youtube_url = data.get('url')
    timestamps = data.get('timestamps')
    crop = data.get('crop')
    enable_tracking = data.get('enable_tracking', True)
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Reset progress
    progress_data['status'] = 'starting'
    progress_data['progress'] = 0
    progress_data['message'] = 'Initializing...'
    progress_data['result'] = None
    progress_data['error'] = None
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_video_thread, args=(youtube_url, timestamps, crop, enable_tracking))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

@app.route('/progress')
def get_progress():
    return jsonify(progress_data)

@app.route('/download/<path:filename>')
def download_video(filename):
    return send_from_directory('static/videos', filename, as_attachment=True)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    from flask import Response
    try:
        return send_from_directory('static/videos', filename, mimetype='video/mp4')
    except Exception as e:
        print(f"Error serving video {filename}: {e}")
        return Response("Video not found", status=404)

@app.route('/preview', methods=['POST'])
def preview():
    data = request.json
    youtube_url = data.get('url')
    timestamps = data.get('timestamps', {})
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    temp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(temp_dir, 'video.%(ext)s')
        
        # Download video
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Get highest quality MP4, fallback to best available
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Find downloaded file
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith('video.')]
        if not downloaded_files:
            return jsonify({'error': 'No video file found'}), 400
        
        actual_video_path = os.path.join(temp_dir, downloaded_files[0])
        
        # Extract frame from custom time range or beginning
        cap = cv2.VideoCapture(actual_video_path)
        
        # If custom start time is specified, seek to that position
        start_time = timestamps.get('start') if timestamps else None
        if start_time:
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Could not read video frame'}), 400
        
        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({'preview_frame': frame_base64})
        
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 