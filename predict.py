
import glob
import matplotlib.pyplot
import os
import signal
import shutil
import sys
import time

from collections import deque
from math import sqrt
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.50     # Prediction confidence threshold to accept results
DIST_THRESHOLD = 0.05           # Distance threshold for checking proximity
MAX_ALERT_HISTORY = 5           # Track the last 5 files for distance checking
MAX_HEATMAP_HISTORY = 30        # Keep the most recent 30 data points
DETECTION_INTERVAL = 30         # Time in seconds to generate heatmap and archive results

alert_history = deque(maxlen=MAX_ALERT_HISTORY)
heatmap_history = deque(maxlen=MAX_HEATMAP_HISTORY)

model = YOLO("custom.pt")

process_dir = 'test'
results_dir = 'runs/detect/predict/'

def signal_handler(sig, frame):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    print("\nExitting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def calculate_distance(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def parse_label_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming each line is of format: class_id x_center y_center width height confidence
            parts = line.split()
            if len(parts) >= 6 and float(parts[5]) >= CONFIDENCE_THRESHOLD:
                # ignore class_id
                data.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
    return data

def update_alert_history(new_data):
    alert_history.extend(new_data)

def update_heatmap_history(new_data):
    heatmap_history.extend(new_data)

def generate_distance_alerts(current_data):
    for curr_point in current_data:
        match_curr_point = True
        for i, previous_file in enumerate(alert_history):
            if i >= MAX_ALERT_HISTORY:
                break

            match_curr_file = False
            for prev_point in previous_file:
                dist = calculate_distance(prev_point[:2], curr_point[:2])
                if dist <= DIST_THRESHOLD:
                    match_curr_file = True
                    break
            if not match_curr_file:
                match_curr_point = False
                break

        if match_curr_point:
            print(f"TODO: Generate Proximity Alert for Point {curr_point[:2]}")

def generate_heatmap():
    if len(heatmap_history) == 0:
        return
    
    x_coords = [item[0] for item in heatmap_history]
    y_coords = [item[1] for item in heatmap_history]

    matplotlib.pyplot.scatter(x_coords, y_coords, c='blue', marker='o')
    matplotlib.pyplot.title(f"Fish Heatmap @ {time.time()}")
    matplotlib.pyplot.xlabel("X Coordinate")
    matplotlib.pyplot.ylabel("Y Coordinate")
    matplotlib.pyplot.savefig(f"heatmap_{time.time()}.png")
    matplotlib.pyplot.close()

def process_yolo_predictions(image_path):
    model.predict(source=image_path, save=False, save_txt=True, save_conf=True)

    try:
        os.remove(image_path)
    except Exception as e:
        print(f"ERROR: could not delete {image_path} - {e}")

def process_alert_results():
    label_files = glob.glob(os.path.join(results_dir, 'labels', '*.txt'))
    if not label_files:
        return
    
    latest_label_file = max(label_files, key=os.path.getmtime)
    current_data = parse_label_file(latest_label_file)
    generate_distance_alerts(current_data)
    update_alert_history(current_data)

def process_heatmap_results():
    label_files = glob.glob(os.path.join(results_dir, 'labels', '*.txt'))
    if not label_files:
        return
    
    label_files.sort(key=os.path.getmtime, reverse=True)
    latest_label_files = label_files[:30]

    for label_file in latest_label_files:
        current_data = parse_label_file(label_file)
        update_heatmap_history(current_data)

    generate_heatmap()

    for label_file in label_files:
        if os.path.isfile(label_file):
            try:
                os.remove(label_file)
            except Exception as e:
                print(f"ERROR: could not delete {label_file} - {e}")

def yolo_processing():
    last_processed_time = time.time()
    while True:
        current_time = time.time()

        images = [f for f in os.listdir(process_dir) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        if images:
            image_path = os.path.join(process_dir, images[0])
            process_yolo_predictions(image_path)
            process_alert_results()

            if current_time - last_processed_time >= DETECTION_INTERVAL:
                process_heatmap_results()
                last_processed_time = current_time

        time.sleep(1)

def main():
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    yolo_processing()


if __name__ == "__main__":
    main()
