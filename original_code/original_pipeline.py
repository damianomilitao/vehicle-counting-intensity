# =============================================================================
# Vehicle Flow Analysis on Urban Roads Using Image Processing
# Author: Felipe Schott Rodrigues
# Institution: Universidade do Estado do Rio de Janeiro (UERJ)
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.signal import find_peaks
import os

# =============================================================================
# 1. VIDEO PREPROCESSING
# =============================================================================

def preprocess_video(video_path, target_resolution=(1280, 720)):
    """
    Reads a video file, resizes each frame to the target resolution,
    converts to grayscale, and stacks frames into a 3D numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames = []
    width, height = target_resolution
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    video_array = np.array(frames)
    return video_array


def save_video_array(video_array, output_path):
    """Saves the 3D video array to a .npy file for fast access."""
    np.save(output_path, video_array)
    print(f"Video array saved to {output_path}")


def load_video_array(input_path):
    """Loads a 3D video array from a .npy file."""
    return np.load(input_path)


# =============================================================================
# 2. INTERACTIVE REGION OF INTEREST (ROI) SELECTION
# =============================================================================

def select_roi_interactive(video_array):
    """Displays the first frame and allows the user to draw a polygon ROI."""
    first_frame = video_array[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(first_frame, cmap='gray')
    ax.set_title("Draw a polygon on the road (click points, then close the window)")
    points = []
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    if len(points) < 3:
        raise ValueError("At least 3 points are required to define a polygon.")
    polygon_path = Path(points)
    return polygon_path


def apply_roi_mask(frame, roi_path):
    """Applies a polygonal mask to a frame, returning only the pixels inside the ROI."""
    height, width = frame.shape
    y_grid, x_grid = np.mgrid[:height, :width]
    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    mask = roi_path.contains_points(points).reshape(height, width)
    return frame[mask]


# =============================================================================
# 3. TIME-SERIES EXTRACTION AND PEAK DETECTION
# =============================================================================

def extract_intensity_time_series(video_array, roi_path):
    """Extracts the average intensity time series from the ROI across all frames."""
    intensity_series = []
    for frame in video_array:
        roi_pixels = apply_roi_mask(frame, roi_path)
        intensity_series.append(np.mean(roi_pixels))
    return np.array(intensity_series)


def compute_absolute_variation(intensity_series):
    """Computes the absolute variation signal D(t) = |I(t) - I_med|."""
    median_intensity = np.median(intensity_series)
    variation = np.abs(intensity_series - median_intensity)
    return variation


def detect_peaks(variation_signal, height_threshold=10, min_distance=30):
    """Detects peaks in the absolute variation signal that exceed a threshold."""
    peaks, properties = find_peaks(
        variation_signal,
        height=height_threshold,
        distance=min_distance
    )
    return peaks, properties


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_variation_signal(variation_signal, peaks, height_threshold):
    """Plots the absolute variation signal with detected peaks highlighted."""
    plt.figure(figsize=(12, 6))
    plt.plot(variation_signal, label='Activity in Zone', color='black')
    plt.plot(peaks, variation_signal[peaks], 'x', color='red', label='Vehicle (Peak)')
    plt.axhline(y=height_threshold, color='red', linestyle='--', label='Tolerance Line')
    plt.xlabel('Frames')
    plt.ylabel('Intensity Variation')
    plt.title(f'Detected Vehicles: {len(peaks)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. RELATIVE ERROR QUANTIFICATION
# =============================================================================

def compute_relative_error(measured_count, real_count):
    """Computes the relative percentage error between measured and real counts."""
    if real_count == 0:
        return float('inf')
    error = (np.abs(measured_count - real_count) / real_count) * 100
    return error


def run_full_pipeline(video_path, target_resolution=(1280, 720),
                      height_threshold=10, min_distance=30,
                      real_count=0):
    """Runs the complete vehicle flow analysis pipeline."""
    print("Preprocessing video...")
    video_array = preprocess_video(video_path, target_resolution)
    print("Please draw a polygon on the road...")
    roi_path = select_roi_interactive(video_array)
    print("Extracting intensity time series...")
    intensity_series = extract_intensity_time_series(video_array, roi_path)
    variation_signal = compute_absolute_variation(intensity_series)
    print("Detecting peaks...")
    peaks, properties = detect_peaks(variation_signal, height_threshold, min_distance)
    vehicle_count = len(peaks)
    print(f"Detected vehicles: {vehicle_count}")
    relative_error = compute_relative_error(vehicle_count, real_count)
    print(f"Relative error: {relative_error:.2f}%")
    plot_variation_signal(variation_signal, peaks, height_threshold)
    return {
        'video_array': video_array,
        'intensity_series': intensity_series,
        'variation_signal': variation_signal,
        'peaks': peaks,
        'properties': properties,
        'vehicle_count': vehicle_count,
        'relative_error': relative_error
    }


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    real_count = 247  # Reference count from manual inspection (Video B)
    results = run_full_pipeline(
        video_path=video_path,
        target_resolution=(1280, 720),
        height_threshold=10,
        min_distance=30,
        real_count=real_count
    )
    print("\n=== FINAL RESULTS ===")
    print(f"Vehicle count: {results['vehicle_count']}")
    print(f"Relative error: {results['relative_error']:.2f}%")
