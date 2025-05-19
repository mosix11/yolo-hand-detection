from ultralytics import YOLO
import requests
from tqdm import tqdm
import os
from pathlib import Path
import torch
import yaml

def download_file_fast(url, output_path):
    """Downloads a file with a progress bar and handles retries."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=16384,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))
        
        
        
yolov11_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt'
yolov11_path = Path('yolov11l.pt')
if not yolov11_path.exists():
    download_file_fast(yolov11_url, yolov11_path)


with open('train_conf.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

model = YOLO(yolov11_path)
results = model.train(
    data="data/hand_data.yaml",
    imgsz=640,
    epochs=100,
    batch=0.90,
    patience=6,
    pretrained=True,
    save=True,
    device='cuda',
    **cfg
)
print(results)