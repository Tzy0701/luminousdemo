import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
import random
from datetime import datetime
import json
from pathlib import Path
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


print("=" * 80)
print("HYBRID POINCARÉ INDEX - STRUCTURAL CORRECTION (Ridge-Aligned Cores)")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# MINIMAL PREPROCESSING - Images are already clean
# ============================================================================

class MinimalPreprocessor:
    """Minimal preprocessing - just normalization"""
    
    @staticmethod
    def preprocess(image_path):
        """Minimal preprocessing for clean images"""
        try:
            # Read image (Unicode support)
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
        except Exception as e:
            return None
        
        # Resize if too large
        max_dim = 600
        h, w = image.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Light CLAHE for contrast enhancement only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Very light noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        return denoised


# ============================================================================
# FINGERPRINT SEGMENTATION - Eliminate Background Detections
# ============================================================================

class RidgeCounter:
    """Ridge counting between core and delta using Bresenham's algorithm and sub-pixel analysis"""
    
    @staticmethod
    def bresenham_line(y0, x0, y1, x1):
        """
        Bresenham's line algorithm for precise pixel-level line generation
        
        Returns list of (y, x) coordinates along the line
        """
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x, y = x0, y0
        
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        if dx > dy:
            error = dx / 2
            while x != x1:
                points.append((y, x))
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != y1:
                points.append((y, x))
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
        
        points.append((y1, x1))
        return points
    
    @staticmethod
    def sample_intensity_along_line(image, line_points, use_interpolation=True):
        """
        Sample intensity values along the line with optional sub-pixel interpolation
        
        Args:
            image: Fingerprint image
            line_points: List of (y, x) coordinates from Bresenham
            use_interpolation: Use bilinear interpolation for sub-pixel accuracy
            
        Returns:
            Array of intensity values along the line
        """
        intensities = []
        
        for y, x in line_points:
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                if use_interpolation and y < image.shape[0] - 1 and x < image.shape[1] - 1:
                    # Bilinear interpolation for sub-pixel accuracy
                    y_frac = y - int(y)
                    x_frac = x - int(x)
                    
                    y_int = int(y)
                    x_int = int(x)
                    
                    # Get 2x2 neighborhood
                    v00 = float(image[y_int, x_int])
                    v01 = float(image[y_int, x_int + 1])
                    v10 = float(image[y_int + 1, x_int])
                    v11 = float(image[y_int + 1, x_int + 1])
                    
                    # Bilinear interpolation
                    v0 = v00 * (1 - x_frac) + v01 * x_frac
                    v1 = v10 * (1 - x_frac) + v11 * x_frac
                    intensity = v0 * (1 - y_frac) + v1 * y_frac
                    
                    intensities.append(intensity)
                else:
                    # Simple pixel sampling
                    intensities.append(float(image[int(y), int(x)]))
        
        return np.array(intensities)
    
    @staticmethod
    def calculate_neighborhood_contrast(inverted, peak_idx, window_size=5):
        """
        Calculate local contrast around a specific peak for adaptive prominence
        
        Args:
            inverted: Inverted intensity profile
            peak_idx: Index of the peak
            window_size: Size of neighborhood window (default: 5 pixels)
            
        Returns:
            local_std: Standard deviation in the neighborhood
        """
        half_win = window_size // 2
        start = max(0, peak_idx - half_win)
        end = min(len(inverted), peak_idx + half_win + 1)
        
        neighborhood = inverted[start:end]
        local_std = np.std(neighborhood) if len(neighborhood) > 1 else 0
        
        return local_std
    
    @staticmethod
    def refine_peak_center_derivative(inverted, peak_idx, search_radius=2):
        """
        🎯 Precise peak center using first derivative zero-crossing
        
        A true ridge center is where the first derivative = 0 (peak)
        This ensures we center on the exact ridge, not its shoulder
        
        Args:
            inverted: Inverted intensity profile
            peak_idx: Initial peak position
            search_radius: Search radius for refinement
            
        Returns:
            refined_idx: Precise peak center (can be sub-pixel via interpolation)
        """
        # Calculate first derivative
        first_deriv = np.gradient(inverted)
        
        # Search for zero-crossing near the peak
        start = max(1, peak_idx - search_radius)
        end = min(len(first_deriv) - 1, peak_idx + search_radius + 1)
        
        # Find the point closest to zero derivative (true peak)
        local_region = first_deriv[start:end]
        
        if len(local_region) == 0:
            return peak_idx
        
        # Find minimum absolute derivative (closest to zero)
        min_deriv_idx = np.argmin(np.abs(local_region))
        refined_center = start + min_deriv_idx
        
        # Sub-pixel interpolation if derivative crosses zero
        if min_deriv_idx > 0 and min_deriv_idx < len(local_region) - 1:
            d_left = local_region[min_deriv_idx - 1]
            d_center = local_region[min_deriv_idx]
            d_right = local_region[min_deriv_idx + 1]
            
            # If there's a sign change, interpolate for sub-pixel accuracy
            if (d_left * d_right) < 0:  # Derivative crosses zero
                # Linear interpolation to find exact zero-crossing
                if abs(d_left - d_right) > 1e-6:
                    offset = -d_center / (d_right - d_left)
                    offset = np.clip(offset, -0.5, 0.5)
                    refined_center += offset
        
        return refined_center
    
    @staticmethod
    def count_ridges_on_profile(intensity_profile, min_ridge_spacing=2):
        """
        ENHANCED ridge counting with dynamic thresholding and structural verification
        
        Improvements:
        1. Adaptive thresholds based on local contrast
        2. Lower base thresholds to catch faint ridges
        3. Structural verification to prevent duplicates
        4. Ridge width constraint for validation
        
        Args:
            intensity_profile: Array of intensity values along the line
            min_ridge_spacing: Minimum distance between ridges (pixels, default=2 for high-res)
            
        Returns:
            ridge_count: Number of ridges
            peak_positions: Positions of detected ridges
        """
        if len(intensity_profile) < 10:
            return 0, []
        
        # Smooth the profile to reduce noise
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(intensity_profile, sigma=1.5)
        
        # Normalize to [0, 1]
        smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-10)
        
        # 🔍 STEP 1: Calculate local contrast for dynamic thresholding
        # Use sliding window to calculate local standard deviation
        window_size = len(smoothed) // 4  # 1/4 of the line length
        window_size = max(10, min(window_size, 50))  # Clamp to reasonable range
        
        # Calculate local std deviation
        local_std = np.std(smoothed)
        
        # Dynamic prominence based on local contrast
        # More sensitive to faint ridges while maintaining quality
        dynamic_prominence = max(0.04, 0.5 * local_std)  # Lowered from 0.08 to 0.04 for dense ridges
        dynamic_height = max(0.18, 0.6 * local_std)      # Lowered from 0.25 to 0.18 for faint ridges
        
        # Cap the thresholds to avoid being too strict
        dynamic_prominence = min(dynamic_prominence, 0.12)
        dynamic_height = min(dynamic_height, 0.35)
        
        # Invert profile (ridges are dark, become peaks after inversion)
        inverted = 1.0 - smoothed
        
        # 🔍 STEP 2: Find peaks with ADAPTIVE spacing
        # Start with base spacing, will be adjusted if peaks are detected
        peaks_initial, properties = find_peaks(
            inverted,
            distance=min_ridge_spacing,      # Initial distance
            prominence=dynamic_prominence,    # Adaptive: 0.04-0.12 based on contrast
            height=dynamic_height             # Adaptive: 0.18-0.35 based on contrast
        )
        
        # 🎚️ DYNAMIC ADJUSTMENT: If initial peaks suggest dense ridges, re-run with tighter spacing
        if len(peaks_initial) > 2:
            # Calculate actual spacing from detected peaks
            detected_spacings = np.diff(peaks_initial)
            avg_detected_spacing = np.median(detected_spacings)
            
            # If ridges are very close (< 3 pixels apart on average), use tighter spacing
            if avg_detected_spacing < 3.0:
                # Dense region - reduce distance to 1.5 pixels
                adjusted_distance = max(1.5, avg_detected_spacing * 0.6)
                
                # Re-run peak detection with adjusted spacing
                peaks, properties = find_peaks(
                    inverted,
                    distance=adjusted_distance,      # Dynamic: 1.5-2.5 based on density
                    prominence=dynamic_prominence,
                    height=dynamic_height
                )
            else:
                # Normal spacing - use initial results
                peaks = peaks_initial
        else:
            peaks = peaks_initial
        
        if len(peaks) == 0:
            return 0, []
        
        # 🔍 STEP 3: Per-peak neighborhood contrast analysis and derivative refinement
        refined_peaks = []
        peak_prominences_adjusted = []
        
        for i, peak_idx in enumerate(peaks):
            # Refine peak center using first derivative zero-crossing
            refined_center = RidgeCounter.refine_peak_center_derivative(inverted, peak_idx, search_radius=2)
            
            # Calculate neighborhood contrast for adaptive prominence
            local_contrast = RidgeCounter.calculate_neighborhood_contrast(inverted, peak_idx, window_size=5)
            
            # Adaptive prominence threshold based on local contrast
            # High contrast area → higher threshold; Low contrast → lower threshold
            local_prominence_threshold = max(0.03, 0.4 * local_contrast)
            local_prominence_threshold = min(local_prominence_threshold, 0.15)
            
            # Check if this peak meets its local prominence requirement
            peak_prominence = properties['prominences'][i]
            
            if peak_prominence >= local_prominence_threshold:
                refined_peaks.append(int(peak_idx))  # Keep integer for logic
                peak_prominences_adjusted.append(peak_prominence)
        
        if len(refined_peaks) == 0:
            return 0, []
        
        # Update peaks list with refined, contrast-validated peaks
        peaks = np.array(refined_peaks)
        
        # 🛡️ STEP 4: Conditional structural verification with peak-strength awareness
        # Filter out peaks that are too close together and likely the same ridge
        
        # Calculate average spacing between all detected peaks
        if len(peaks) > 1:
            peak_spacings = np.diff(peaks)
            avg_spacing = np.mean(peak_spacings)
        else:
            avg_spacing = min_ridge_spacing
        
        # Ridge width constraint: Base requirement is 50% of avg spacing
        base_structural_distance = max(min_ridge_spacing * 0.7, avg_spacing * 0.50)
        
        validated_peaks = []
        validated_prominences = []
        last_peak_idx = -1000  # Initialize far away
        
        for i, peak_idx in enumerate(peaks):
            # Get prominence for this peak
            peak_prominence = peak_prominences_adjusted[i] if i < len(peak_prominences_adjusted) else 0
            
            # 🌀 CONDITIONAL STRUCTURAL DISTANCE
            # If peak is strong, allow it to be closer to previous peak
            distance_to_last = peak_idx - last_peak_idx
            
            # Strong peaks get relaxed distance requirement
            # If prominence > 1.5x dynamic_prominence, can be closer (80% of avg spacing)
            if peak_prominence > dynamic_prominence * 1.5:
                min_distance_required = max(min_ridge_spacing * 0.6, avg_spacing * 0.40)
            else:
                min_distance_required = base_structural_distance
            
            # Check distance from last accepted peak
            if distance_to_last >= min_distance_required:
                # Additional check: verify this is a true valley (ridge)
                # Look at the depth between this peak and the last one
                
                if len(validated_peaks) > 0:
                    # Calculate valley depth between last peak and current peak
                    start = validated_peaks[-1]
                    end = peak_idx
                    
                    # Find minimum in the region between peaks
                    region = inverted[start:end+1]
                    if len(region) > 0:
                        # Get actual valley depth (prominence)
                        valley_depth = peak_prominence
                        
                        # Calculate relative drop: ratio of peak height to valley depth
                        peak_height = inverted[peak_idx]
                        valley_min = np.min(region)
                        relative_drop = (peak_height - valley_min) / (peak_height + 1e-10)
                        
                        # Accept if valley is significant (relaxed criteria)
                        # Use BOTH absolute and relative criteria
                        absolute_check = valley_depth >= dynamic_prominence * 0.50  # Very relaxed
                        relative_check = relative_drop >= 0.10  # At least 10% drop into valley
                        
                        if absolute_check or relative_check:
                            validated_peaks.append(peak_idx)
                            validated_prominences.append(peak_prominence)
                            last_peak_idx = peak_idx
                    else:
                        validated_peaks.append(peak_idx)
                        validated_prominences.append(peak_prominence)
                        last_peak_idx = peak_idx
                else:
                    # First peak, always accept
                    validated_peaks.append(peak_idx)
                    validated_prominences.append(peak_prominence)
                    last_peak_idx = peak_idx
        
        # 🛡️ STEP 5: Ridge width constraint - verify each ridge has proper width
        # Final validation: check that spacing is consistent
        final_peaks = []
        
        for peak_idx in validated_peaks:
            # Find local minima around this peak to verify it's a true ridge
            search_radius = int(min_ridge_spacing * 1.5)
            
            # Left side
            left_start = max(0, peak_idx - search_radius)
            left_region = inverted[left_start:peak_idx]
            
            # Right side  
            right_end = min(len(inverted), peak_idx + search_radius + 1)
            right_region = inverted[peak_idx+1:right_end]
            
            # Check if there are valleys on both sides (characteristic of a ridge)
            # Relaxed to 0.87 for dense ridges (shallower valleys acceptable)
            has_left_valley = len(left_region) > 0 and np.min(left_region) < inverted[peak_idx] * 0.87
            has_right_valley = len(right_region) > 0 and np.min(right_region) < inverted[peak_idx] * 0.87
            
            # Accept peak if it has valleys on at least one side (edge ridges may only have one)
            if has_left_valley or has_right_valley:
                final_peaks.append(peak_idx)
        
        ridge_count = len(final_peaks)
        peaks_array = np.array(final_peaks)
        
        return ridge_count, peaks_array
    
    @staticmethod
    def get_line_angle(y1, x1, y2, x2):
        """
        Calculate the angle of the line in degrees (0-180)
        
        Returns:
            angle: Angle in degrees from horizontal (0° = horizontal, 90° = vertical)
        """
        dy = y2 - y1
        dx = x2 - x1
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = np.arctan2(abs(dy), abs(dx))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def create_perpendicular_sampling_line(image, y1, x1, y2, x2, offset_range=10):
        """
        For steep lines (close to vertical), create multiple parallel sampling lines
        and combine results to avoid missing ridges
        
        Args:
            image: Fingerprint image
            y1, x1, y2, x2: Start and end points
            offset_range: How far to offset perpendicular lines (pixels)
            
        Returns:
            combined_ridge_count: Average ridge count from multiple samples
        """
        # Calculate perpendicular direction
        dy = y2 - y1
        dx = x2 - x1
        length = np.sqrt(dx*dx + dy*dy)
        
        # Perpendicular vector (rotated 90 degrees)
        perp_dx = -dy / (length + 1e-10)
        perp_dy = dx / (length + 1e-10)
        
        # Sample at multiple offsets
        offsets = [-offset_range, -offset_range//2, 0, offset_range//2, offset_range]
        ridge_counts = []
        
        for offset in offsets:
            # Offset start and end points perpendicular to the line
            new_y1 = int(y1 + perp_dy * offset)
            new_x1 = int(x1 + perp_dx * offset)
            new_y2 = int(y2 + perp_dy * offset)
            new_x2 = int(x2 + perp_dx * offset)
            
            # Check bounds
            if (0 <= new_y1 < image.shape[0] and 0 <= new_x1 < image.shape[1] and
                0 <= new_y2 < image.shape[0] and 0 <= new_x2 < image.shape[1]):
                
                try:
                    line_points = RidgeCounter.bresenham_line(new_y1, new_x1, new_y2, new_x2)
                    intensity_profile = RidgeCounter.sample_intensity_along_line(
                        image, line_points, use_interpolation=True
                    )
                    
                    if len(intensity_profile) > 10:
                        ridge_count, _ = RidgeCounter.count_ridges_on_profile(
                            intensity_profile, min_ridge_spacing=2.0
                        )
                        ridge_counts.append(ridge_count)
                except:
                    continue
        
        if ridge_counts:
            # Use maximum count (most perpendicular sampling gives best result)
            return max(ridge_counts)
        
        return 0
    
    @staticmethod
    def count_ridges_between_points(image, y1, x1, y2, x2, visualization=False):
        """
        Complete ridge counting pipeline between two points (e.g., core and delta)
        WITH ANGLE-AWARE SAMPLING for steep/vertical lines
        
        Args:
            image: Fingerprint image
            y1, x1: Starting point (e.g., core)
            y2, x2: Ending point (e.g., delta)
            visualization: Return additional data for visualization
            
        Returns:
            ridge_count: Number of ridges
            line_points: Points along the counting line (for visualization)
            intensity_profile: Intensity values along line (for visualization)
            ridge_positions: Positions of detected ridges (for visualization)
        """
        # Step 0: Check line angle
        angle = RidgeCounter.get_line_angle(y1, x1, y2, x2)
        
        # If line is steep (> 70° from horizontal, close to vertical)
        # Use perpendicular sampling strategy for more accurate ridge counting
        is_steep = angle > 70
        
        if is_steep:
            # For steep lines, use multiple parallel samples and take the maximum
            ridge_count = RidgeCounter.create_perpendicular_sampling_line(
                image, y1, x1, y2, x2, offset_range=8
            )
            
            if visualization:
                # Still provide visualization data from the original line
                line_points = RidgeCounter.bresenham_line(int(y1), int(x1), int(y2), int(x2))
                intensity_profile = RidgeCounter.sample_intensity_along_line(
                    image, line_points, use_interpolation=True
                )
                # Create ridge positions based on count (approximate)
                ridge_positions = np.linspace(0, len(intensity_profile)-1, ridge_count).astype(int)
                return ridge_count, line_points, intensity_profile, ridge_positions
            else:
                return ridge_count
        else:
            # Normal angle - use standard direct sampling
            # Step 1: Generate precise line using Bresenham's algorithm
            line_points = RidgeCounter.bresenham_line(
                int(y1), int(x1), int(y2), int(x2)
            )
            
            # Step 2: Sample intensity with sub-pixel interpolation
            intensity_profile = RidgeCounter.sample_intensity_along_line(
                image, line_points, use_interpolation=True
            )
            
            # Step 3: Count ridges with dynamic spacing and adaptive thresholds
            # min_ridge_spacing=2.0, adjusted dynamically for dense regions
            ridge_count, ridge_positions = RidgeCounter.count_ridges_on_profile(
                intensity_profile, min_ridge_spacing=2.0
            )
            
            if visualization:
                return ridge_count, line_points, intensity_profile, ridge_positions
            else:
                return ridge_count


class FingerprintSegmentation:
    """Segment fingerprint from background to eliminate false detections in white areas"""
    
    @staticmethod
    def create_fingerprint_mask(image, block_size=10, erosion_iterations=2):
        """
        Create binary mask separating fingerprint (foreground) from background (white areas)
        
        ENHANCED: Now includes erosion to eliminate boundary detections
        
        Uses local coherence and variance to identify valid fingerprint regions
        """
        h, w = image.shape
        rows = h // block_size
        cols = w // block_size
        
        # Calculate gradients
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Block-wise mask
        mask = np.zeros((rows, cols), dtype=bool)
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * block_size
                y_end = min(y_start + block_size, h)
                x_start = j * block_size
                x_end = min(x_start + block_size, w)
                
                block = image[y_start:y_end, x_start:x_end]
                block_gx = gx[y_start:y_end, x_start:x_end]
                block_gy = gy[y_start:y_end, x_start:x_end]
                
                # Metric 1: Grayscale variance (fingerprint has texture, white area is uniform)
                variance = np.var(block)
                
                # Metric 2: Mean intensity (white areas have high intensity ~255)
                mean_intensity = np.mean(block)
                
                # Metric 3: Gradient strength (fingerprint has ridges, white area has no gradients)
                gradient_strength = np.mean(np.abs(block_gx) + np.abs(block_gy))
                
                # Valid fingerprint block criteria (slightly relaxed thresholds)
                is_fingerprint = (
                    variance > 150 and           # Has texture
                    mean_intensity < 230 and     # Not white background
                    gradient_strength > 4.0      # Has ridge patterns
                )
                
                mask[i, j] = is_fingerprint
        
        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Close small holes
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small isolated regions
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # ✨ NEW: Erosion to exclude boundary regions
        # This creates a buffer zone and eliminates edge detections
        kernel_erode = np.ones((3, 3), dtype=np.uint8)
        mask_eroded = cv2.erode(mask_opened, kernel_erode, iterations=erosion_iterations)
        
        # Convert back to boolean
        mask_final = (mask_eroded > 127)
        
        return mask_final
    
    @staticmethod
    def calculate_distance_to_boundary(fingerprint_mask, i_block, j_block):
        """
        Calculate minimum distance from a block to the fingerprint boundary
        
        Used to filter out detections too close to edges
        """
        rows, cols = fingerprint_mask.shape
        
        if not fingerprint_mask[i_block, j_block]:
            return 0
        
        # Use distance transform to find distance to nearest background pixel
        # Invert mask so background is 1
        inverted_mask = (~fingerprint_mask).astype(np.uint8)
        
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(
            (~inverted_mask).astype(np.uint8), 
            cv2.DIST_L2, 
            3
        )
        
        # Get distance at this block
        distance = dist_transform[i_block, j_block]
        
        return distance


# ============================================================================
# ACCURATE ORIENTATION FIELD ESTIMATION
# ============================================================================

class AccurateOrientationEstimator:
    """High-accuracy orientation field estimation"""
    
    @staticmethod
    def estimate_orientation_accurate(image, block_size=10):
        """
        Accurate orientation field estimation
        
        Key improvements:
        1. Larger Sobel kernel (5x5) for better gradient estimation
        2. Overlapping blocks for smoother transitions
        3. Better averaging and smoothing
        4. Reliability weighting
        """
        h, w = image.shape
        
        # Use larger kernel for more accurate gradients
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
        # Calculate orientation using gradient squared method
        Gxx = gx * gx
        Gyy = gy * gy
        Gxy = gx * gy
        
        # Apply Gaussian smoothing to gradient components
        # This reduces noise while preserving structure
        sigma = 3.0
        Gxx_smooth = gaussian_filter(Gxx, sigma=sigma)
        Gyy_smooth = gaussian_filter(Gyy, sigma=sigma)
        Gxy_smooth = gaussian_filter(Gxy, sigma=sigma)
        
        # Block-wise processing
        rows = h // block_size
        cols = w // block_size
        orientation = np.zeros((rows, cols))
        coherence = np.zeros((rows, cols))
        gradient_strength = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * block_size
                y_end = min(y_start + block_size, h)
                x_start = j * block_size
                x_end = min(x_start + block_size, w)
                
                # Extract smoothed gradient components
                block_gxx = Gxx_smooth[y_start:y_end, x_start:x_end]
                block_gyy = Gyy_smooth[y_start:y_end, x_start:x_end]
                block_gxy = Gxy_smooth[y_start:y_end, x_start:x_end]
                
                # Sum over block
                sum_gxx = np.sum(block_gxx)
                sum_gyy = np.sum(block_gyy)
                sum_gxy = np.sum(block_gxy)
                
                # Orientation (divide by 2 because we're using squared gradients)
                orientation[i, j] = 0.5 * np.arctan2(2 * sum_gxy, sum_gxx - sum_gyy)
                
                # Coherence (reliability measure)
                numerator = np.sqrt((sum_gxx - sum_gyy) ** 2 + 4 * sum_gxy ** 2)
                denominator = sum_gxx + sum_gyy + 1e-10
                coherence[i, j] = numerator / denominator
                
                # Gradient strength (for quality assessment)
                gradient_strength[i, j] = sum_gxx + sum_gyy
        
        # Apply minimal smoothing to orientation field
        # Use vector averaging to handle angle wrapping
        orientation_x = np.cos(2 * orientation)
        orientation_y = np.sin(2 * orientation)
        
        # Very light smoothing
        smoothed_x = gaussian_filter(orientation_x, sigma=0.7)
        smoothed_y = gaussian_filter(orientation_y, sigma=0.7)
        
        orientation_smoothed = 0.5 * np.arctan2(smoothed_y, smoothed_x)
        
        # Smooth coherence lightly
        coherence_smoothed = gaussian_filter(coherence, sigma=0.7)
        
        # ✨ NEW: Generate quality map for advanced filtering
        # Quality based on coherence and gradient strength
        normalized_strength = gradient_strength / (np.max(gradient_strength) + 1e-10)
        quality_map = coherence_smoothed * 0.6 + normalized_strength * 0.4
        
        return orientation_smoothed, coherence_smoothed, gradient_strength, quality_map


# ============================================================================
# IMAGE QUALITY ASSESSMENT (for API / batch pipelines)
# ============================================================================

class ImageQualityAssessor:
    """
    Lightweight quality assessment for a fingerprint image.

    Returns a dict shaped for `api.py`:
      {
        "status": "OK" | "WARN" | "BAD",
        "reasons": [{"code": str, "message": str}, ...],
        "metrics": {...}
      }
    """

    def __init__(
        self,
        block_size=10,
        min_foreground_ratio=0.06,
        warn_quality=0.35,
        bad_quality=0.25,
    ):
        self.block_size = int(block_size)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.warn_quality = float(warn_quality)
        self.bad_quality = float(bad_quality)

    def assess(self, image_path: str) -> dict:
        image = MinimalPreprocessor.preprocess(image_path)
        if image is None:
            return {
                "status": "BAD",
                "reasons": [{"code": "load_failed", "message": "Failed to read image."}],
                "metrics": {},
            }

        # Block-level foreground mask (same grid as orientation/coherence/quality_map)
        try:
            fingerprint_mask = FingerprintSegmentation.create_fingerprint_mask(
                image, block_size=self.block_size, erosion_iterations=1
            )
        except Exception:
            fingerprint_mask = None

        orientation, coherence, gradient_strength, quality_map = AccurateOrientationEstimator.estimate_orientation_accurate(
            image, block_size=self.block_size
        )

        if fingerprint_mask is None or fingerprint_mask.shape != quality_map.shape:
            mask = np.ones_like(quality_map, dtype=bool)
        else:
            mask = fingerprint_mask.astype(bool)

        fg_ratio = float(mask.mean()) if mask.size else 0.0
        if mask.any():
            mean_quality = float(np.mean(quality_map[mask]))
            mean_coherence = float(np.mean(coherence[mask]))
            max_strength = float(np.max(gradient_strength) + 1e-10)
            mean_strength_norm = float(np.mean(gradient_strength[mask]) / max_strength)
        else:
            mean_quality = 0.0
            mean_coherence = 0.0
            mean_strength_norm = 0.0

        reasons = []
        status = "OK"

        if fg_ratio < self.min_foreground_ratio:
            status = "BAD"
            reasons.append(
                {
                    "code": "low_foreground",
                    "message": f"Fingerprint area too small (foreground_ratio={fg_ratio:.3f}).",
                }
            )

        if mean_quality < self.bad_quality:
            status = "BAD"
            reasons.append(
                {
                    "code": "low_quality",
                    "message": f"Low ridge quality (mean_quality={mean_quality:.3f}).",
                }
            )
        elif mean_quality < self.warn_quality and status != "BAD":
            status = "WARN"
            reasons.append(
                {
                    "code": "borderline_quality",
                    "message": f"Borderline ridge quality (mean_quality={mean_quality:.3f}).",
                }
            )

        return {
            "status": status,
            "reasons": reasons,
            "metrics": {
                "foreground_ratio": fg_ratio,
                "mean_quality": mean_quality,
                "mean_coherence": mean_coherence,
                "mean_strength_norm": mean_strength_norm,
                "block_size": self.block_size,
            },
        }


# ============================================================================
# ACCURATE POINCARÉ INDEX DETECTION WITH HYBRID VALIDATION
# ============================================================================

class AccuratePoincareDetector:
    """High-accuracy Poincaré Index detection with hybrid validation"""
    
    @staticmethod
    def create_complex_filter_kernel(window_size=9, singularity_type='core'):
        """
        Create complex filter kernel tuned for singular points
        
        Core: spiral (+180°)
        Delta: triradius (-180°)
        """
        half = window_size // 2
        y, x = np.mgrid[-half:half+1, -half:half+1]
        angle = np.arctan2(y, x + 1e-6)
        radius = np.sqrt(x**2 + y**2) + 1e-6
        sigma = window_size / 3.0
        gaussian_envelope = np.exp(-(radius**2) / (2 * sigma**2))
        
        if singularity_type == 'core':
            kernel = gaussian_envelope * np.exp(1j * angle * 2)  # spiral
        else:
            kernel = gaussian_envelope * np.exp(-1j * angle * 2)  # tri-radii
        
        # Normalize kernel
        norm = np.sqrt(np.sum(np.abs(kernel)**2)) + 1e-10
        kernel = kernel / norm
        return kernel
    
    @staticmethod
    def complex_filter_response(tensor_field, y_min, y_max, x_min, x_max, singularity_type='core', window_size=9):
        """
        Apply complex filter within specified region and return peak location
        """
        half = window_size // 2
        kernel = AccuratePoincareDetector.create_complex_filter_kernel(window_size, singularity_type)
        
        best_response = 0
        best_y = y_min + (y_max - y_min) // 2
        best_x = x_min + (x_max - x_min) // 2
        
        for y in range(y_min + half, y_max - half):
            for x in range(x_min + half, x_max - half):
                patch = tensor_field[y-half:y+half+1, x-half:x+half+1]
                response = np.abs(np.sum(patch * np.conj(kernel)))
                if response > best_response:
                    best_response = response
                    best_y = y
                    best_x = x
        
        return best_y, best_x, best_response
    
    @staticmethod
    def orientation_gradient_validation(orientation, i, j, radius=2):
        """
        Validate singular point by checking orientation gradient magnitude
        """
        rows, cols = orientation.shape
        y_min = max(1, i - radius)
        y_max = min(rows - 1, i + radius + 1)
        x_min = max(1, j - radius)
        x_max = min(cols - 1, j + radius + 1)
        
        region = orientation[y_min:y_max, x_min:x_max]
        if region.size == 0:
            return 0.0
        
        grad_y, grad_x = np.gradient(region)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        score = np.mean(grad_mag)
        # Normalize score to [0,1] using heuristic scale
        score = np.clip(score / 1.5, 0, 1)
        return score
    
    @staticmethod
    def calculate_poincare_index_accurate(orientation, i, j):
        """
        Accurate Poincaré Index calculation
        
        Uses 3x3 window with proper angle unwrapping
        """
        rows, cols = orientation.shape
        
        if i < 1 or i >= rows - 1 or j < 1 or j >= cols - 1:
            return 0.0
        
        # 3x3 window, clockwise from top-left
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),
            (1, 1), (1, 0), (1, -1),
            (0, -1)
        ]
        
        # Extract angles
        angles = []
        for dr, dc in offsets:
            angles.append(orientation[i + dr, j + dc])
        
        # Close the loop
        angles.append(angles[0])
        
        # Calculate Poincaré Index
        poincare_sum = 0.0
        for k in range(len(angles) - 1):
            delta = angles[k + 1] - angles[k]
            
            # Unwrap angle (orientation is modulo π)
            if delta > np.pi / 2:
                delta -= np.pi
            elif delta < -np.pi / 2:
                delta += np.pi
            
            poincare_sum += delta
        
        return poincare_sum
    
    @staticmethod
    def calculate_modified_poincare_index(orientation, i, j, window_size=2):
        """
        Modified Poincaré Index using larger window for more robust detection
        
        This uses a 5x5 window (when window_size=2) instead of just 3x3
        """
        rows, cols = orientation.shape
        
        if i < window_size or i >= rows - window_size or j < window_size or j >= cols - window_size:
            return 0.0
        
        # Sample points in a larger window (square pattern)
        sample_points = []
        
        # Top row
        for dj in range(-window_size, window_size + 1):
            sample_points.append((-window_size, dj))
        
        # Right column (skip top corner already added)
        for di in range(-window_size + 1, window_size + 1):
            sample_points.append((di, window_size))
        
        # Bottom row (skip right corner already added)
        for dj in range(window_size - 1, -window_size - 1, -1):
            sample_points.append((window_size, dj))
        
        # Left column (skip bottom and top corners)
        for di in range(window_size - 1, -window_size, -1):
            sample_points.append((di, -window_size))
        
        # Extract angles along the path
        angles = []
        for dr, dc in sample_points:
            angles.append(orientation[i + dr, j + dc])
        
        # Close the loop
        angles.append(angles[0])
        
        # Calculate Poincaré Index
        poincare_sum = 0.0
        for k in range(len(angles) - 1):
            delta = angles[k + 1] - angles[k]
            
            # Unwrap angle
            if delta > np.pi / 2:
                delta -= np.pi
            elif delta < -np.pi / 2:
                delta += np.pi
            
            poincare_sum += delta
        
        return poincare_sum
    
    @staticmethod
    def structural_validation_core(orientation, coherence, i, j, radius=3):
        """
        Structural validation for CORE points
        
        Core points should show concentric circular pattern:
        - Orientations should rotate around the center
        - High local coherence
        - Smooth orientation changes
        """
        rows, cols = orientation.shape
        
        if i < radius or i >= rows - radius or j < radius or j >= cols - radius:
            return 0.0
        
        # Check 1: Orientation consistency in concentric circles
        circle_scores = []
        
        for r in range(1, radius + 1):
            # Sample points in a circle
            num_samples = max(8, r * 8)
            angles_sampled = []
            coherences = []
            
            for k in range(num_samples):
                angle = 2 * np.pi * k / num_samples
                di = int(round(r * np.sin(angle)))
                dj = int(round(r * np.cos(angle)))
                
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    angles_sampled.append(orientation[ni, nj])
                    coherences.append(coherence[ni, nj])
            
            if len(angles_sampled) > 4:
                # Check smoothness of orientation changes
                deltas = []
                for k in range(len(angles_sampled)):
                    delta = angles_sampled[(k + 1) % len(angles_sampled)] - angles_sampled[k]
                    if delta > np.pi / 2:
                        delta -= np.pi
                    elif delta < -np.pi / 2:
                        delta += np.pi
                    deltas.append(abs(delta))
                
                # Score: prefer smooth changes
                avg_delta = np.mean(deltas)
                smoothness_score = np.exp(-avg_delta * 2)
                
                # Score: high coherence
                avg_coherence = np.mean(coherences)
                
                # Combined score
                circle_scores.append(smoothness_score * 0.6 + avg_coherence * 0.4)
        
        # Check 2: Central coherence should be high
        central_coherence = coherence[i, j]
        
        # Combined structural score
        if circle_scores:
            avg_circle_score = np.mean(circle_scores)
            structural_score = avg_circle_score * 0.7 + central_coherence * 0.3
        else:
            structural_score = central_coherence
        
        return structural_score
    
    @staticmethod
    def structural_validation_delta(orientation, coherence, i, j, radius=3):
        """
        Structural validation for DELTA points
        
        Delta points typically show triangular/Y-shaped pattern:
        - Three ridge flows converging
        - Local orientation changes more abrupt than core
        - Different pattern than core
        """
        rows, cols = orientation.shape
        
        if i < radius or i >= rows - radius or j < radius or j >= cols - radius:
            return 0.0
        
        # Check for three-way split pattern
        # Sample orientations in 6 directions (60-degree intervals)
        directions = []
        coherences = []
        
        for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
            di = int(round(radius * np.sin(angle)))
            dj = int(round(radius * np.cos(angle)))
            
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                directions.append(orientation[ni, nj])
                coherences.append(coherence[ni, nj])
        
        if len(directions) < 4:
            return 0.0
        
        # Check for three dominant directions (characteristic of delta)
        # Calculate variance in orientations
        orientation_variance = np.var(directions)
        
        # Delta should have moderate variance (not too uniform, not too chaotic)
        variance_score = 1.0 - abs(orientation_variance - 0.3) / 0.5
        variance_score = max(0, min(1, variance_score))
        
        # Average coherence
        avg_coherence = np.mean(coherences)
        
        # Combined structural score
        structural_score = variance_score * 0.5 + avg_coherence * 0.5
        
        return structural_score
    
    @staticmethod
    def reliability_coherence_check(coherence, gradient_strength, i, j, radius=2):
        """
        Advanced reliability check using coherence and gradient strength
        
        Returns a reliability score [0, 1]
        """
        rows, cols = coherence.shape
        
        if i < radius or i >= rows - radius or j < radius or j >= cols - radius:
            return 0.0
        
        # Extract neighborhood
        neighborhood_coherence = coherence[i-radius:i+radius+1, j-radius:j+radius+1]
        neighborhood_strength = gradient_strength[i-radius:i+radius+1, j-radius:j+radius+1]
        
        # Metrics
        avg_coherence = np.mean(neighborhood_coherence)
        min_coherence = np.min(neighborhood_coherence)
        
        avg_strength = np.mean(neighborhood_strength)
        max_strength_global = np.max(gradient_strength) + 1e-10
        normalized_strength = avg_strength / max_strength_global
        
        # Reliability score
        # High average coherence, not too low minimum coherence, reasonable gradient
        reliability = (
            avg_coherence * 0.5 +
            min_coherence * 0.2 +
            normalized_strength * 0.3
        )
        
        return min(1.0, reliability)
    
    @staticmethod
    def calculate_confidence(orientation, coherence, gradient_strength, i, j, pi_value, gradient_score=None):
        """
        Calculate confidence score for a detected singular point
        
        Higher confidence = more likely to be true singularity
        """
        rows, cols = orientation.shape
        
        # Factor 1: PI value close to ±π is better
        pi_quality = 1.0 - abs(abs(pi_value) - np.pi) / np.pi
        
        # Factor 2: Local coherence
        local_coherence = coherence[i, j]
        
        # Factor 3: Gradient strength in neighborhood
        y_min = max(0, i - 2)
        y_max = min(rows, i + 3)
        x_min = max(0, j - 2)
        x_max = min(cols, j + 3)
        neighborhood_strength = np.mean(gradient_strength[y_min:y_max, x_min:x_max])
        
        # Normalize strength
        max_strength = np.max(gradient_strength) + 1e-10
        strength_score = neighborhood_strength / max_strength
        
        # Gradient score (optional)
        if gradient_score is None:
            # Estimate local gradient score if not provided
            gradient_score = AccuratePoincareDetector.orientation_gradient_validation(orientation, i, j)
        
        # Combined confidence (now includes gradient structure)
        confidence = (
            pi_quality * 0.4 +
            local_coherence * 0.25 +
            strength_score * 0.15 +
            gradient_score * 0.2
        )
        
        return confidence
    
    @staticmethod
    def detect_singular_points_accurate(orientation, coherence, gradient_strength,
                                        coherence_threshold=0.45,  # Original threshold
                                        min_pi_value=2.6,
                                        max_pi_value=3.6,
                                        confidence_threshold=0.40,  # Original threshold
                                        border_margin=3,
                                        use_hybrid_validation=True,
                                        fingerprint_mask=None,
                                        quality_map=None,
                                        quality_threshold=0.35):  # Original threshold
        """
        HYBRID APPROACH: Accurate singular point detection with multiple validation layers
        
        Step 1: Use modified Poincaré Index to generate candidates
        Step 2: Apply reliability/coherence checks
        Step 3: Use structural rules to validate
        Step 4: Refine final locations
        
        NEW: Uses fingerprint_mask + quality_map to eliminate background and low-quality detections
        """
        rows, cols = orientation.shape
        
        cores_candidates = []
        deltas_candidates = []
        
        # STEP 1: Generate candidates using BOTH standard and modified PI
        for i in range(border_margin, rows - border_margin):
            for j in range(border_margin, cols - border_margin):
                # BACKGROUND ELIMINATION: Skip if outside fingerprint region
                if fingerprint_mask is not None and not fingerprint_mask[i, j]:
                    continue
                
                # ✨ QUALITY MAP FILTERING: Skip low-quality regions
                if quality_map is not None and quality_map[i, j] < quality_threshold:
                    continue
                
                # Basic coherence check
                if coherence[i, j] < coherence_threshold:
                    continue
                
                # Calculate STANDARD Poincaré Index (3x3 window)
                pi_standard = AccuratePoincareDetector.calculate_poincare_index_accurate(
                    orientation, i, j
                )
                
                # Calculate MODIFIED Poincaré Index (5x5 window) for robustness
                pi_modified = AccuratePoincareDetector.calculate_modified_poincare_index(
                    orientation, i, j, window_size=2
                )
                
                # Use average of both for more robust detection
                pi_combined = (pi_standard + pi_modified) / 2.0
                pi_abs = abs(pi_combined)
                
                # Check if in valid range
                if min_pi_value <= pi_abs <= max_pi_value:
                    
                    if use_hybrid_validation:
                        # STEP 2: Reliability/Coherence Check
                        reliability_score = AccuratePoincareDetector.reliability_coherence_check(
                            coherence, gradient_strength, i, j, radius=2
                        )
                        
                        # STEP 3: Structural Validation
                        if pi_combined > 0:  # Core candidate
                            structural_score = AccuratePoincareDetector.structural_validation_core(
                                orientation, coherence, i, j, radius=3
                            )
                        else:  # Delta candidate
                            structural_score = AccuratePoincareDetector.structural_validation_delta(
                                orientation, coherence, i, j, radius=3
                            )
                        
                        # Gradient validation
                        gradient_score = AccuratePoincareDetector.orientation_gradient_validation(
                            orientation, i, j, radius=2
                        )
                        
                        # Calculate base confidence
                        confidence_base = AccuratePoincareDetector.calculate_confidence(
                            orientation, coherence, gradient_strength, i, j, pi_combined, gradient_score=gradient_score
                        )
                        
                        # HYBRID CONFIDENCE: Combine all validation scores
                        confidence_hybrid = (
                            confidence_base * 0.4 +
                            reliability_score * 0.3 +
                            structural_score * 0.3
                        )
                        
                        confidence = confidence_hybrid
                    else:
                        # Use only base confidence
                        confidence = AccuratePoincareDetector.calculate_confidence(
                            orientation, coherence, gradient_strength, i, j, pi_combined
                        )
                    
                    # Keep high-confidence detections
                    if confidence >= confidence_threshold:
                        if pi_combined > 0:
                            cores_candidates.append((i, j, pi_combined, confidence))
                        else:
                            deltas_candidates.append((i, j, pi_combined, confidence))
        
        return cores_candidates, deltas_candidates
    
    @staticmethod
    def apply_complex_gabor_filter(image, y_center, x_center, window_size=9, singularity_type='core'):
        """
        Apply complex Gabor filter for sub-pixel accuracy - MATHEMATICAL CENTER
        
        This finds (x_M, y_M), the center based on the mathematical model
        
        Args:
            image: Original fingerprint image
            y_center, x_center: Candidate center location
            window_size: Size of analysis window (e.g., 9×9)
            singularity_type: 'core' or 'delta'
            
        Returns:
            (refined_y, refined_x): Mathematical center coordinates
        """
        h, w = image.shape
        half_win = window_size // 2
        
        # Extract window around candidate
        y_min = max(0, y_center - half_win)
        y_max = min(h, y_center + half_win + 1)
        x_min = max(0, x_center - half_win)
        x_max = min(w, x_center + half_win + 1)
        
        window = image[y_min:y_max, x_min:x_max].astype(np.float32)
        
        if window.size == 0:
            return y_center, x_center
        
        # Create complex Gabor filter bank for singularity detection
        responses = []
        
        # For core: circular/spiral patterns
        # For delta: Y-shaped patterns
        num_orientations = 8
        
        for i in range(num_orientations):
            theta = np.pi * i / num_orientations
            
            # Gabor parameters tuned for singularities
            if singularity_type == 'core':
                # Core: circular pattern, medium frequency
                kernel = cv2.getGaborKernel(
                    ksize=(window_size, window_size),
                    sigma=2.5,
                    theta=theta,
                    lambd=6.0,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
            else:  # delta
                # Delta: sharper features
                kernel = cv2.getGaborKernel(
                    ksize=(window_size, window_size),
                    sigma=2.0,
                    theta=theta,
                    lambd=5.0,
                    gamma=0.7,
                    psi=0,
                    ktype=cv2.CV_32F
                )
            
            # Apply filter
            filtered = cv2.filter2D(window, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))
        
        # Combine responses (maximum response across orientations)
        combined_response = np.max(responses, axis=0)
        
        # Find peak response
        max_response_idx = np.unravel_index(np.argmax(combined_response), combined_response.shape)
        
        # Convert back to image coordinates
        refined_y = y_min + max_response_idx[0]
        refined_x = x_min + max_response_idx[1]
        
        return refined_y, refined_x
    
    @staticmethod
    def trace_innermost_ridge(image, y_center, x_center, search_radius=30):
        """
        Trace the innermost ridge that forms the core loop
        
        Returns a list of (y, x) coordinates along the traced ridge
        """
        h, w = image.shape
        
        # Define search window
        y_min = max(0, y_center - search_radius)
        y_max = min(h, y_center + search_radius)
        x_min = max(0, x_center - search_radius)
        x_max = min(w, x_center + search_radius)
        
        patch = image[y_min:y_max, x_min:x_max].copy()
        
        if patch.size == 0:
            return []
        
        # Enhance ridges using morphological operations
        # Binarize to separate ridges from valleys
        _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert so ridges are white
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Simple skeleton extraction using morphological operations
        # (alternative to ximgproc.thinning which may not be available)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(binary)
        temp = binary.copy()
        
        # Iterative thinning (simplified Zhang-Suen)
        for _ in range(10):
            eroded = cv2.erode(temp, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(temp, opened))
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        
        # Find contours in skeleton
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return []
        
        # Find the contour closest to the center (innermost ridge)
        center_patch_y = y_center - y_min
        center_patch_x = x_center - x_min
        
        min_dist = float('inf')
        innermost_contour = None
        
        for contour in contours:
            if len(contour) < 10:  # Skip tiny fragments
                continue
            
            # Calculate minimum distance from contour to center
            for point in contour:
                px, py = point[0]
                dist = np.sqrt((py - center_patch_y)**2 + (px - center_patch_x)**2)
                if dist < min_dist:
                    min_dist = dist
                    innermost_contour = contour
        
        if innermost_contour is None:
            return []
        
        # Convert contour points back to image coordinates
        ridge_points = []
        for point in innermost_contour:
            px, py = point[0]
            ridge_points.append((y_min + py, x_min + px))
        
        return ridge_points
    
    @staticmethod
    def calculate_curvature_along_ridge(ridge_points):
        """
        Calculate curvature at each point along the traced ridge
        
        Returns array of curvature values and corresponding points
        """
        if len(ridge_points) < 5:
            return [], []
        
        curvatures = []
        valid_points = []
        
        # Use sliding window to calculate local curvature
        window = 3  # Points on each side
        
        for i in range(window, len(ridge_points) - window):
            # Get local neighborhood
            prev_points = ridge_points[i-window:i]
            curr_point = ridge_points[i]
            next_points = ridge_points[i+1:i+window+1]
            
            # Fit circle to these points to estimate curvature
            points = prev_points + [curr_point] + next_points
            ys = np.array([p[0] for p in points])
            xs = np.array([p[1] for p in points])
            
            # Calculate local curvature using change in direction
            if len(xs) >= 3:
                # Vector from prev to curr
                v1_y = ys[window] - ys[0]
                v1_x = xs[window] - xs[0]
                
                # Vector from curr to next
                v2_y = ys[-1] - ys[window]
                v2_x = xs[-1] - xs[window]
                
                # Normalize
                norm1 = np.sqrt(v1_y**2 + v1_x**2) + 1e-10
                norm2 = np.sqrt(v2_y**2 + v2_x**2) + 1e-10
                
                v1_y, v1_x = v1_y / norm1, v1_x / norm1
                v2_y, v2_x = v2_y / norm2, v2_x / norm2
                
                # Curvature ~ angle change
                dot_product = v1_y * v2_y + v1_x * v2_x
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle_change = np.arccos(dot_product)
                
                curvatures.append(angle_change)
                valid_points.append(curr_point)
        
        return curvatures, valid_points
    
    @staticmethod
    def find_structural_core_center(image, x_M, y_M, search_radius=30):
        """
        STRUCTURAL CORRECTION: Find (x_T, y_T) - the true structural center
        
        This is the FINAL step that aligns the core with the physical ridge structure
        
        Steps:
        1. Trace the innermost ridge forming the core loop
        2. Calculate curvature along that ridge
        3. Find the point of maximum curvature
        4. Return (x_T, y_T) as the structural center
        
        Args:
            image: Original fingerprint image
            x_M, y_M: Mathematical center from complex filtering
            search_radius: Radius to search for ridge structure
            
        Returns:
            (y_T, x_T): Structural center aligned with ridge curvature
        """
        # Step 1: Trace innermost ridge
        ridge_points = AccuratePoincareDetector.trace_innermost_ridge(
            image, int(y_M), int(x_M), search_radius=search_radius
        )
        
        if len(ridge_points) < 10:
            # Fallback: if tracing fails, return mathematical center
            return y_M, x_M
        
        # Step 2: Calculate curvature along ridge
        curvatures, valid_points = AccuratePoincareDetector.calculate_curvature_along_ridge(ridge_points)
        
        if len(curvatures) == 0:
            return y_M, x_M
        
        # Step 3: Find point of maximum curvature
        max_curvature_idx = np.argmax(curvatures)
        y_T, x_T = valid_points[max_curvature_idx]
        
        # Step 4: Verify this point is reasonably close to mathematical center
        # (to avoid jumping to a completely different region)
        distance = np.sqrt((y_T - y_M)**2 + (x_T - x_M)**2)
        
        if distance > search_radius * 0.5:
            # Too far - likely noise, use weighted blend
            blend_factor = 0.3  # Move 30% toward structural center
            y_T = y_M + blend_factor * (y_T - y_M)
            x_T = x_M + blend_factor * (x_T - x_M)
        
        return y_T, x_T
    
    @staticmethod
    def weighted_averaging_pi_maxima(pi_map, y_center, x_center, radius=2):
        """
        Weighted averaging of neighboring PI maxima
        
        Instead of single pixel with highest PI, use weighted average of
        high-PI pixels in vicinity to smooth out fluctuations
        """
        h, w = pi_map.shape
        
        # Extract neighborhood
        y_min = max(0, y_center - radius)
        y_max = min(h, y_center + radius + 1)
        x_min = max(0, x_center - radius)
        x_max = min(w, x_center + radius + 1)
        
        neighborhood = np.abs(pi_map[y_min:y_max, x_min:x_max])
        
        if neighborhood.size == 0:
            return y_center, x_center
        
        # Find strong PI pixels (above 80% of local maximum)
        max_pi = np.max(neighborhood)
        threshold = max_pi * 0.8
        
        # Weight pixels by their PI strength
        weights = neighborhood * (neighborhood > threshold)
        
        if np.sum(weights) == 0:
            return y_center, x_center
        
        # Calculate weighted centroid
        y_indices, x_indices = np.indices(neighborhood.shape)
        
        weighted_y = np.sum(y_indices * weights) / np.sum(weights)
        weighted_x = np.sum(x_indices * weights) / np.sum(weights)
        
        # Convert back to image coordinates
        refined_y = int(y_min + weighted_y)
        refined_x = int(x_min + weighted_x)
        
        return refined_y, refined_x
    
    @staticmethod
    def pixel_level_refinement(orientation, i_block, j_block, block_size, image, pi_sign):
        """
        ENHANCED pixel-level refinement with complex filtering and weighted averaging
        
        Multi-stage refinement:
        1. Initial pixel-level PI search
        2. Weighted averaging of PI maxima
        3. Complex Gabor filter for sub-pixel accuracy
        """
        h, w = image.shape
        
        # Convert block coordinates to pixel coordinates
        y_center_pixel = int((i_block + 0.5) * block_size)
        x_center_pixel = int((j_block + 0.5) * block_size)
        
        # STAGE 1: High-resolution PI search
        search_radius = block_size // 2
        y_min = max(3, y_center_pixel - search_radius)
        y_max = min(h - 3, y_center_pixel + search_radius)
        x_min = max(3, x_center_pixel - search_radius)
        x_max = min(w - 3, x_center_pixel + search_radius)
        
        pixel_window = 2  # Smaller window for finer detail
        
        # Calculate gradients for search region
        region = image[max(0, y_min-pixel_window):min(h, y_max+pixel_window+1), 
                      max(0, x_min-pixel_window):min(w, x_max+pixel_window+1)]
        
        if region.size == 0:
            return y_center_pixel, x_center_pixel
        
        gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate pixel-level orientation
        h_region, w_region = region.shape
        orientation_pixel = np.zeros((h_region, w_region))
        pi_map = np.zeros((h_region, w_region))
        
        for i in range(pixel_window, h_region - pixel_window):
            for j in range(pixel_window, w_region - pixel_window):
                # 3×3 pixel window for orientation
                local_gx = gx[i-pixel_window:i+pixel_window+1, j-pixel_window:j+pixel_window+1]
                local_gy = gy[i-pixel_window:i+pixel_window+1, j-pixel_window:j+pixel_window+1]
                
                Gxx = np.sum(local_gx * local_gx)
                Gyy = np.sum(local_gy * local_gy)
                Gxy = np.sum(local_gx * local_gy)
                
                orientation_pixel[i, j] = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
                
                # Calculate PI at this pixel
                if i >= 1 and i < h_region - 1 and j >= 1 and j < w_region - 1:
                    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
                    angles = [orientation_pixel[i+dy, j+dx] for dy, dx in offsets]
                    angles.append(angles[0])
                    
                    pi = 0.0
                    for k in range(len(angles) - 1):
                        delta = angles[k + 1] - angles[k]
                        if delta > np.pi / 2:
                            delta -= np.pi
                        elif delta < -np.pi / 2:
                            delta += np.pi
                        pi += delta
                    
                    if np.sign(pi) == pi_sign:
                        pi_map[i, j] = abs(pi)
        
        # Find candidate pixel with strongest PI
        if np.max(pi_map) < 2.0:
            return y_center_pixel, x_center_pixel
        
        max_idx = np.unravel_index(np.argmax(pi_map), pi_map.shape)
        candidate_y = y_min + max_idx[0] - pixel_window
        candidate_x = x_min + max_idx[1] - pixel_window
        
        # STAGE 2: Weighted averaging of neighboring PI maxima
        # This smooths out minor fluctuations
        refined_y_weighted, refined_x_weighted = AccuratePoincareDetector.weighted_averaging_pi_maxima(
            pi_map, max_idx[0], max_idx[1], radius=2
        )
        
        # Convert to image coordinates
        candidate_y_weighted = y_min + refined_y_weighted - pixel_window
        candidate_x_weighted = x_min + refined_x_weighted - pixel_window
        
        # Build complex tensor field for refined search region
        # Use gradient magnitude as coherence proxy
        coherence_region = np.hypot(gx, gy)
        coherence_region = coherence_region / (np.max(coherence_region) + 1e-10)
        tensor_field = coherence_region * np.exp(1j * 2 * orientation_pixel)
        
        # Convert candidate coordinates to region index
        cand_y_region = candidate_y_weighted - (y_min - pixel_window)
        cand_x_region = candidate_x_weighted - (x_min - pixel_window)
        
        # Define search bounds for complex filter (ensure within region)
        half_window = 5
        y_region_min = max(half_window, int(cand_y_region) - half_window)
        y_region_max = min(tensor_field.shape[0] - half_window - 1, int(cand_y_region) + half_window)
        x_region_min = max(half_window, int(cand_x_region) - half_window)
        x_region_max = min(tensor_field.shape[1] - half_window - 1, int(cand_x_region) + half_window)
        
        # STAGE 3: Complex Gabor filter for sub-pixel accuracy
        # This gives us the MATHEMATICAL CENTER (x_M, y_M)
        singularity_type = 'core' if pi_sign > 0 else 'delta'
        final_y_region, final_x_region, _ = AccuratePoincareDetector.complex_filter_response(
            tensor_field,
            y_region_min, y_region_max,
            x_region_min, x_region_max,
            singularity_type=singularity_type,
            window_size=9
        )
        
        # Convert back to image coordinates
        y_M = (y_min - pixel_window) + final_y_region
        x_M = (x_min - pixel_window) + final_x_region
        
        # STAGE 4: STRUCTURAL CORRECTION (Core-specific)
        # Find the TRUE STRUCTURAL CENTER (x_T, y_T) by tracing actual ridge curvature
        if pi_sign > 0:  # Core only
            y_T, x_T = AccuratePoincareDetector.find_structural_core_center(
                image, x_M, y_M, search_radius=25
            )
            return y_T, x_T
        else:  # Delta - mathematical center is sufficient
            return y_M, x_M
    
    @staticmethod
    def filter_by_confidence_and_distance(points, min_distance=7, max_points=5):
        """
        Filter points by confidence and spatial separation
        
        Keep highest confidence points that are spatially separated
        """
        if len(points) == 0:
            return []
        
        # Sort by confidence (highest first)
        points = sorted(points, key=lambda p: p[3], reverse=True)
        
        filtered = []
        for point in points:
            i, j, pi, conf = point
            
            # Check distance from already selected points
            too_close = False
            for fi, fj, _, _ in filtered:
                distance = np.sqrt((i - fi) ** 2 + (j - fj) ** 2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(point)
                
                # Limit number of points
                if len(filtered) >= max_points:
                    break
        
        return filtered
    
    @staticmethod
    def geometric_validation(cores, deltas, image_shape):
        """
        Validate detections using geometric constraints
        
        - Core typically in upper region
        - Delta typically in lower region  
        - Reasonable spatial arrangement
        """
        rows, cols = image_shape
        
        validated_cores = []
        validated_deltas = []
        
        # Cores: prefer upper-middle region but don't be too strict
        for i, j, pi, conf in cores:
            y_norm = i / rows
            x_norm = j / cols
            
            # Prefer y < 0.7 (upper 70%) and central x (0.2 to 0.8)
            positional_score = 1.0
            if y_norm > 0.7:
                positional_score *= 0.7  # Penalize but don't eliminate
            if x_norm < 0.15 or x_norm > 0.85:
                positional_score *= 0.8  # Penalize edge positions
            
            # Adjust confidence with positional score
            adjusted_conf = conf * positional_score
            
            # Keep if still above threshold (original: 0.6)
            if adjusted_conf >= 0.6:
                validated_cores.append((i, j, pi, adjusted_conf))
        
        # Deltas: prefer middle-lower region
        for i, j, pi, conf in deltas:
            y_norm = i / rows
            x_norm = j / cols
            
            positional_score = 1.0
            if y_norm < 0.2:
                positional_score *= 0.7  # Penalize upper region
            if x_norm < 0.1 or x_norm > 0.9:
                positional_score *= 0.8
            
            adjusted_conf = conf * positional_score
            
            # Original threshold: 0.6
            if adjusted_conf >= 0.6:
                validated_deltas.append((i, j, pi, adjusted_conf))
        
        # Re-sort by adjusted confidence
        validated_cores = sorted(validated_cores, key=lambda p: p[3], reverse=True)
        validated_deltas = sorted(validated_deltas, key=lambda p: p[3], reverse=True)
        
        return validated_cores, validated_deltas


# ============================================================================
# COMPLETE ACCURATE DETECTOR
# ============================================================================

class ImprovedFingerprintDetector:
    """Complete accuracy-focused detector"""
    
    def __init__(self, block_size=10):
        self.block_size = block_size
        self.preprocessor = MinimalPreprocessor()
        self.orientation_estimator = AccurateOrientationEstimator()
        self.poincare_detector = AccuratePoincareDetector()
    
    def detect(self, image_path):
        """Complete accurate detection pipeline with background elimination and pixel-level refinement"""
        try:
            # Minimal preprocessing
            image = self.preprocessor.preprocess(image_path)
            if image is None:
                return {'success': False, 'error': 'Failed to load image'}
            
            # IMPROVEMENT 1: Create fingerprint mask to eliminate background detections
            segmentation = FingerprintSegmentation()
            fingerprint_mask = segmentation.create_fingerprint_mask(image, self.block_size)
            
            # Accurate orientation field estimation with quality map
            orientation, coherence, gradient_strength, quality_map = self.orientation_estimator.estimate_orientation_accurate(
                image, self.block_size
            )
            
            # HYBRID singular point detection with multiple validation layers
            # Pass mask AND quality map to restrict detection to high-quality regions
            cores, deltas = self.poincare_detector.detect_singular_points_accurate(
                orientation, coherence, gradient_strength,
                use_hybrid_validation=True,
                fingerprint_mask=fingerprint_mask,
                quality_map=quality_map
            )
            
            # Geometric validation
            cores, deltas = self.poincare_detector.geometric_validation(
                cores, deltas, orientation.shape
            )
            
            # Filter by confidence and distance
            cores = self.poincare_detector.filter_by_confidence_and_distance(cores, min_distance=7, max_points=5)
            deltas = self.poincare_detector.filter_by_confidence_and_distance(deltas, min_distance=7, max_points=5)
            
            # ✨ BOUNDARY CHECK: Eliminate detections too close to fingerprint edge
            cores_filtered = []
            deltas_filtered = []
            
            min_boundary_distance = 2.5  # blocks (not too strict)
            
            for i, j, pi, conf in cores:
                dist_to_boundary = segmentation.calculate_distance_to_boundary(fingerprint_mask, i, j)
                if dist_to_boundary >= min_boundary_distance:
                    cores_filtered.append((i, j, pi, conf))
            
            for i, j, pi, conf in deltas:
                dist_to_boundary = segmentation.calculate_distance_to_boundary(fingerprint_mask, i, j)
                if dist_to_boundary >= min_boundary_distance:
                    deltas_filtered.append((i, j, pi, conf))
            
            # IMPROVEMENT 2: Enhanced pixel-level refinement with complex filtering
            refined_cores = []
            for i, j, pi, conf in cores_filtered:
                y_pixel, x_pixel = self.poincare_detector.pixel_level_refinement(
                    orientation, i, j, self.block_size, image, pi_sign=1
                )
                refined_cores.append((y_pixel, x_pixel, pi, conf))
            
            refined_deltas = []
            for i, j, pi, conf in deltas_filtered:
                y_pixel, x_pixel = self.poincare_detector.pixel_level_refinement(
                    orientation, i, j, self.block_size, image, pi_sign=-1
                )
                refined_deltas.append((y_pixel, x_pixel, pi, conf))
            
            # RIDGE COUNTING between core and delta
            ridge_counts = []
            ridge_count_details = []
            
            if len(refined_cores) > 0 and len(refined_deltas) > 0:
                # Count ridges between each core-delta pair
                for core_y, core_x, core_pi, core_conf in refined_cores:
                    for delta_y, delta_x, delta_pi, delta_conf in refined_deltas:
                        # Perform ridge counting
                        ridge_count, line_points, intensity_profile, ridge_positions = \
                            RidgeCounter.count_ridges_between_points(
                                image, core_y, core_x, delta_y, delta_x, visualization=True
                            )
                        
                        ridge_counts.append(ridge_count)
                        ridge_count_details.append({
                            'core': (core_y, core_x),
                            'delta': (delta_y, delta_x),
                            'ridge_count': ridge_count,
                            'line_points': line_points,
                            'intensity_profile': intensity_profile,
                            'ridge_positions': ridge_positions
                        })
            
            return {
                'success': True,
                'cores': refined_cores,
                'deltas': refined_deltas,
                'num_cores': len(refined_cores),
                'num_deltas': len(refined_deltas),
                'ridge_counts': ridge_counts,
                'ridge_count_details': ridge_count_details,
                'image': image,
                'orientation': orientation,
                'coherence': coherence,
                'fingerprint_mask': fingerprint_mask,
                'quality_map': quality_map
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ============================================================================
# VISUALIZATION WITH CONFIDENCE
# ============================================================================

def visualize_detection(result, image_path, output_path, block_size=10):
    """Enhanced visualization showing confidence and pixel-level accuracy"""
    
    if not result['success']:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot 1: Detection results
    ax1 = axes[0]
    ax1.imshow(result['image'], cmap='gray')
    ax1.set_title(f"Structural Correction (Ridge-Aligned Cores)\n{Path(image_path).name}", fontsize=10)
    ax1.axis('off')
    
    # Draw cores with confidence (now using pixel-level coordinates)
    for y_pixel, x_pixel, pi, conf in result['cores']:
        # Use pixel coordinates directly
        
        # Circle size based on confidence
        radius = 12 + conf * 8
        circle = Circle((x_pixel, y_pixel), radius=radius, color='red', fill=False, linewidth=2, alpha=0.7 + conf * 0.3)
        ax1.add_patch(circle)
        
        label_text = f'CORE\n{conf:.2f}'
        ax1.text(x_pixel, y_pixel - 25, label_text, color='red', fontsize=8, 
                ha='center', weight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw deltas with confidence (now using pixel-level coordinates)
    for y_pixel, x_pixel, pi, conf in result['deltas']:
        # Use pixel coordinates directly
        
        size = 12 + conf * 8
        ax1.plot(x_pixel, y_pixel, 'b^', markersize=size, markeredgewidth=2, fillstyle='none', alpha=0.7 + conf * 0.3)
        
        label_text = f'DELTA\n{conf:.2f}'
        ax1.text(x_pixel, y_pixel - 25, label_text, color='blue', fontsize=8, 
                ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw ridge counting lines and ridge positions
    has_ridge_counts = len(result.get('ridge_count_details', [])) > 0
    if has_ridge_counts:
        for detail in result['ridge_count_details']:
            core_y, core_x = detail['core']
            delta_y, delta_x = detail['delta']
            ridge_count = detail['ridge_count']
            
            # Draw counting line (Bresenham's path)
            ax1.plot([core_x, delta_x], [core_y, delta_y], 'g-', 
                    linewidth=2.5, alpha=0.7, zorder=2)
            
            # Ridge positions visualization removed to avoid blocking view
            # (Subpixel interpolation is still used in the counting algorithm)
            # line_points = detail['line_points']
            # ridge_positions = detail['ridge_positions']
            # for ridge_idx in ridge_positions:
            #     if ridge_idx < len(line_points):
            #         ry, rx = line_points[ridge_idx]
            #         ax1.plot(rx, ry, 'o', color='yellow', markersize=6, 
            #                 markeredgecolor='orange', markeredgewidth=1.5, zorder=3)
            
            # Add ridge count label well beside the line to avoid blocking view
            mid_x = (core_x + delta_x) / 2
            mid_y = (core_y + delta_y) / 2
            
            # Calculate perpendicular offset to move label far to the side
            dx = delta_x - core_x
            dy = delta_y - core_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Perpendicular direction (rotate 90 degrees)
                perp_x = -dy / length
                perp_y = dx / length
                
                # Offset label by 40 pixels to the side (increased from 20)
                offset = 40
                label_x = mid_x + perp_x * offset
                label_y = mid_y + perp_y * offset
            else:
                label_x = mid_x + 40
                label_y = mid_y
            
            ax1.text(label_x, label_y, f'RC: {ridge_count}', color='white', fontsize=9, 
                     weight='bold', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='green', 
                              edgecolor='white', linewidth=1.5, alpha=0.85), zorder=4)
    
    # Add ridge count to stats if available
    stats_text = f"Cores: {result['num_cores']} | Deltas: {result['num_deltas']}"
    if has_ridge_counts and len(result['ridge_counts']) > 0:
        avg_rc = np.mean(result['ridge_counts'])
        stats_text += f" | Avg Ridge Count: {avg_rc:.1f}"
    
    ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes,
            ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Plot 2: Improved orientation field
    ax2 = axes[1]
    ax2.imshow(result['image'], cmap='gray', alpha=0.6)
    ax2.set_title("High-Quality Ridge Orientation Field", fontsize=10)
    ax2.axis('off')
    
    orientation = result['orientation']
    coherence = result['coherence']
    rows, cols = orientation.shape
    
    # Draw orientation field with quality-based opacity
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            if coherence[i, j] > 0.3:
                y = (i + 0.5) * block_size
                x = (j + 0.5) * block_size
                
                angle = orientation[i, j]
                length = 12
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                
                # Line opacity based on coherence
                alpha = 0.4 + coherence[i, j] * 0.6
                linewidth = 1.0 if coherence[i, j] > 0.6 else 0.7
                
                ax2.plot([x - dx, x + dx], [y - dy, y + dy], 
                        'g-', linewidth=linewidth, alpha=alpha)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def collect_all_images(dataset_path):
    """Collect all images"""
    all_images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    return all_images


def evaluate_detection(result):
    """Evaluate detection success"""
    if not result['success']:
        return False
    
    num_cores = result['num_cores']
    num_deltas = result['num_deltas']
    
    if num_cores == 0 and num_deltas == 0:
        return False
    
    if num_cores > 5 or num_deltas > 5:
        return False
    
    return True


def run_batch_evaluation(dataset_path, output_dir, num_samples=100):
    """Run batch evaluation"""
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION - STRUCTURAL CORRECTION METHOD")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCollecting images...")
    all_images = collect_all_images(dataset_path)
    print(f"[OK] Found {len(all_images)} total images")
    
    if len(all_images) > num_samples:
        sampled_images = random.sample(all_images, num_samples)
    else:
        sampled_images = all_images
    
    print(f"\n[INFO] Testing on {len(sampled_images)} random images")
    
    detector = ImprovedFingerprintDetector(block_size=10)
    
    results = []
    successful_detections = 0
    failed_detections = 0
    
    print("\nProcessing images...")
    print("-" * 80)
    
    for idx, image_path in enumerate(sampled_images, 1):
        print(f"[{idx}/{len(sampled_images)}] Processing: {Path(image_path).name}...", end=' ')
        
        result = detector.detect(image_path)
        is_successful = evaluate_detection(result)
        
        result_info = {
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'success': result['success'],
            'detection_successful': is_successful,
            'num_cores': result.get('num_cores', 0),
            'num_deltas': result.get('num_deltas', 0),
            'error': result.get('error', None)
        }
        
        # Add confidence scores and ridge counts
        if is_successful:
            cores_conf = [c[3] for c in result.get('cores', [])]
            deltas_conf = [d[3] for d in result.get('deltas', [])]
            result_info['avg_core_confidence'] = np.mean(cores_conf) if cores_conf else 0
            result_info['avg_delta_confidence'] = np.mean(deltas_conf) if deltas_conf else 0
            
            # Add ridge counts if available
            if 'ridge_counts' in result:
                result_info['ridge_counts'] = result['ridge_counts']
        
        results.append(result_info)
        
        if is_successful:
            successful_detections += 1
            print("[OK] SUCCESS")
            
            output_path = os.path.join(output_dir, f"improved_detection_{idx:03d}.png")
            visualize_detection(result, image_path, output_path, block_size=10)
        else:
            failed_detections += 1
            print("[X] FAILED")
    
    print("-" * 80)
    
    success_rate = (successful_detections / len(sampled_images)) * 100
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - STRUCTURAL CORRECTION METHOD")
    print("=" * 80)
    print(f"\nTotal Images Tested: {len(sampled_images)}")
    print(f"Successful Detections: {successful_detections}")
    print(f"Failed Detections: {failed_detections}")
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    
    if success_rate >= 90:
        print("\n[SUCCESS] Goal achieved! Detection rate >= 90%")
    else:
        print(f"\n[INFO] Detection rate is {success_rate:.2f}%")
    
    # Calculate average confidence
    successful_results = [r for r in results if r['detection_successful']]
    if successful_results:
        avg_confidences = []
        for r in successful_results:
            if 'avg_core_confidence' in r:
                avg_confidences.append((r['avg_core_confidence'] + r.get('avg_delta_confidence', 0)) / 2)
        
        if avg_confidences:
            overall_confidence = np.mean(avg_confidences)
            print(f"Average Detection Confidence: {overall_confidence:.3f}")
    
    # Ridge counting statistics
    all_ridge_counts = []
    for r in results:
        if r['detection_successful'] and 'ridge_counts' in r:
            all_ridge_counts.extend(r['ridge_counts'])
    
    if all_ridge_counts:
        avg_ridge_count = np.mean(all_ridge_counts)
        min_ridge_count = np.min(all_ridge_counts)
        max_ridge_count = np.max(all_ridge_counts)
        std_ridge_count = np.std(all_ridge_counts)
        
        print(f"\n--- RIDGE COUNTING STATISTICS ---")
        print(f"Total Core-Delta Pairs Analyzed: {len(all_ridge_counts)}")
        print(f"Average Ridge Count: {avg_ridge_count:.2f} ± {std_ridge_count:.2f}")
        print(f"Ridge Count Range: {min_ridge_count:.0f} - {max_ridge_count:.0f}")
    
    results_file = os.path.join(output_dir, 'improved_evaluation_results.json')
    summary = {
        'method': 'Structural Correction Poincaré Index (Ridge-Aligned Cores) + Ridge Counting',
        'improvements': [
            'Larger Sobel kernel (5x5) for accurate gradients',
            'Gaussian smoothing of gradient components',
            'Minimal orientation smoothing (sigma=0.7)',
            'Modified PI with 5x5 window for robustness',
            'Reliability/coherence multi-point check',
            'Structural validation (concentric patterns for core, Y-pattern for delta)',
            'Hybrid confidence scoring combining all validations',
            'Geometric validation',
            '✨ Fingerprint segmentation with erosion (eliminates background + boundary)',
            '✨ Enhanced pixel-level refinement with complex Gabor filtering',
            '✨ Weighted averaging of PI maxima for stability',
            '✨ Quality map thresholding (high-quality regions only)',
            '✨ Distance-to-boundary check (eliminates edge detections)',
            '🎯 STRUCTURAL CORE CORRECTION: Ridge tracing + curvature analysis',
            '🎯 Mathematical center (x_M, y_M) → Structural center (x_T, y_T)',
            '🎯 Highest precision core localization via physical ridge alignment',
            '📏 RIDGE COUNTING: Bresenham line + sub-pixel interpolation',
            '📏 Dynamic thresholding based on local contrast (adaptive prominence)',
            '📏 🎚️ ADAPTIVE SPACING: Auto-detects dense regions and adjusts distance (1.5-2.5px)',
            '📏 🔍 PER-PEAK neighborhood contrast analysis (5px window)',
            '📏 🎯 First derivative zero-crossing for precise peak centers',
            '📏 🌀 CONDITIONAL structural distance (strong peaks can be closer)',
            '📏 Relative drop ratio for valley validation (10% minimum)',
            '📏 Sensitive thresholds (prominence 0.03-0.15 per peak, height 0.18-0.35)',
            '📏 Multi-level validation (absolute + relative criteria)',
            '📏 Optimized for dense ridges with peak-by-peak precision',
            '📏 Automated ridge count between all core-delta pairs'
        ],
        'dataset_path': dataset_path,
        'num_samples': len(sampled_images),
        'successful_detections': successful_detections,
        'failed_detections': failed_detections,
        'success_rate': success_rate,
        'goal_achieved': success_rate >= 90,
        'ridge_counting': {
            'total_pairs_analyzed': len(all_ridge_counts) if all_ridge_counts else 0,
            'average_ridge_count': float(np.mean(all_ridge_counts)) if all_ridge_counts else 0,
            'min_ridge_count': float(np.min(all_ridge_counts)) if all_ridge_counts else 0,
            'max_ridge_count': float(np.max(all_ridge_counts)) if all_ridge_counts else 0,
            'std_ridge_count': float(np.std(all_ridge_counts)) if all_ridge_counts else 0
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Results saved: {results_file}")
    
    return summary


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    DATASET_PATH = "Finger Print Type"
    OUTPUT_DIR = "results/poincare_detection_improved"
    NUM_SAMPLES = 100
    
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset not found: {DATASET_PATH}")
        exit(1)
    
    summary = run_batch_evaluation(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES
    )
    
    print("\n" + "=" * 80)
    print("STRUCTURAL CORRECTION EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    
    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
