import os
import shutil
from ultralytics import YOLO

model = YOLO("custom.pt")

process_dir = 'test'
for filename in os.listdir(process_dir):
    image_path = os.path.join(process_dir, filename)
    if os.path.isfile(image_path):
        model.predict(source=image_path, save=True, save_txt=True, save_conf=True)

results_dir = 'runs/detect/predict'
zip_file = f"{results_dir.rstrip(os.sep)}"
shutil.make_archive(zip_file, 'zip', results_dir)
shutil.rmtree(results_dir)