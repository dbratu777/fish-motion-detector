
import datetime
import glob
import json
import matplotlib.pyplot
import os
import seaborn
import shutil
import signal
import subprocess
import sys
import time

from collections import deque
from math import sqrt
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.60     # Prediction confidence threshold to accept results
DIST_THRESHOLD = 0.05           # Distance threshold for checking proximity
MAX_ALERT_HISTORY = 60          # Track the last 60 files for distance checking
MAX_HEATMAP_HISTORY = 120       # Keep the most recent 120 data points
DETECTION_INTERVAL = 60         # Time in seconds to generate heatmap and archive results

alert_history = deque(maxlen=MAX_ALERT_HISTORY)
heatmap_history = deque(maxlen=MAX_HEATMAP_HISTORY)

model = YOLO("custom.pt")

process_dir = 'datasets/test'
results_dir = 'runs/detect/predict/'

Base = declarative_base()


# ALERT INFO:
# Types: 0 = Temp, 1 = pH, 2 = ORP, 3 = Fish Health
class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    type = Column(Integer, nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(String(200), nullable=True)
    timestamp = Column(
        DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    read = Column(Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'read': self.read
        }


def signal_handler(sig, frame):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    print("\nExiting...")
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
                data.append([float(parts[1]), float(parts[2])])
    return data


def update_alert_history(new_data):
    alert_history.append(new_data)


def update_heatmap_history(new_data):
    heatmap_history.append(new_data)


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
            alert = Alert(type=3,
                          title="Fish Health",
                          description=f"The fish found at point '({curr_point[:2]})' is acting abnormally.",
                          timestamp=datetime.datetime.now(datetime.timezone.utc))
            alert_dict = alert.to_dict()
            alert_json = json.dumps(alert_dict)

            dp_client_path = os.path.join(
                '..', 'fish-websockets', 'dp_client.py')
            subprocess.run(["python", dp_client_path, "alert", alert_json])


def generate_heatmap():
    if len(heatmap_history) == 0:
        return

    x_coords = [item[0] for file in heatmap_history for item in file]
    y_coords = [item[1] for file in heatmap_history for item in file]

    heatmap_name = f"heatmap_{time.time()}.png"
    seaborn.kdeplot(
        x=x_coords,
        y=y_coords,
        cmap="mako",
        fill=True,
        thresh=0.05,
        bw_adjust=0.25
    )
    matplotlib.pyplot.title(f"Fish Heatmap @ {time.time()}")
    matplotlib.pyplot.xlabel("X Coordinate")
    matplotlib.pyplot.ylabel("Y Coordinate")
    matplotlib.pyplot.savefig(heatmap_name)
    matplotlib.pyplot.close()

    heatmap_path = os.path.join(os.getcwd(), heatmap_name)
    dp_client_path = os.path.join('..', 'fish-websockets', 'dp_client.py')
    subprocess.run(["python", dp_client_path, "heatmap", heatmap_path], check=False)

    try:
        os.remove(heatmap_path)
    except Exception as e:
        print(f"ERROR: could not delete {heatmap_path} - {e}")


def process_yolo_predictions(image_path):
    try:
        model.predict(source=image_path, save=False,
                      save_txt=True, save_conf=True)
    except Exception as e:
        print(f"WARNING: could not process {image_path} - {e}")

    try:
        os.remove(image_path)
    except Exception as e:
        print(f"WARNING: could not delete {image_path} - {e}")


def process_alert_results():
    label_files = glob.glob(os.path.join(results_dir, 'labels', '*.txt'))
    if not label_files:
        return

    latest_label_file = max(label_files, key=os.path.getmtime)
    current_data = parse_label_file(latest_label_file)
    if len(alert_history) >= MAX_ALERT_HISTORY:
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

        images = [f for f in os.listdir(process_dir) if f.lower().endswith(
            ('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
        if images:
            image_path = os.path.join(process_dir, images[0])
            while True:
                last_modified = os.path.getmtime(image_path)
                age = time.time() - last_modified
                if age > 1:
                    break
                time.sleep(0.1)

            process_yolo_predictions(image_path)
            process_alert_results()

            if current_time - last_processed_time >= DETECTION_INTERVAL:
                process_heatmap_results()
                last_processed_time = current_time

        time.sleep(0.1)


def main():
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    yolo_processing()


if __name__ == "__main__":
    main()
