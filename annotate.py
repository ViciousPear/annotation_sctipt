import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy

class AutoAnnotator:
    def __init__(self, model_path, images_folder, output_folder, conf_threshold=0.2, device="cpu"):
        self.model_path = model_path
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.device = device
        self.conf_threshold = conf_threshold
        self.output_folder.mkdir(parents=True, exist_ok=True)

        print("Загрузка модели...")
        self.model = YOLO(model_path)

    def get_images(self):
        extensions = ['*.png', '*.jpg', '*.jpeg']
        images = []

        for ext in extensions:
            images.extend(self.images_folder.glob(ext))

        return images
    
    def save_yolo_annotation(self, results, img_path):
        txt_path = self.output_folder / f"{img_path.stem}.txt"

        with open(txt_path, 'w') as f:
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                h, w = result.orig_shape[:2]

                for box, conf, cls in zip (boxes, confs, classes):
                    x_center = (box[0] + box[2]) / 2 / w
                    y_center = (box[1] + box[3]) / 2 / h
                    width = (box[2] - box[0]) / w
                    height = (box[3] - box[1]) / h

                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def process_batch(self, images, batch_size=32):
        for i in range(0, len(images), batch_size):
            batch = images [i:i+batch_size]
            batch_paths = [str(img) for img in batch]

            results = self.model(batch_paths, verbose=False, device=self.device)

            for img_path, result in zip(batch, results):
                self.save_yolo_annotation([result], img_path)

    def run(self):
        images = self.get_images()

        if not images:
            print(f" В папке {self.images_folder} нет изображений")
            return False
        
        print(f"Найдено {len(images)} изображений")
        print("Запуск аннотации...")

        for img_path in tqdm(images, desc="Инференс"):
            results = self.model(str(img_path), verbose=False, device=self.device)
            self.save_yolo_annotation(results, img_path)

        print(f'Аннотации сохранены в {self.output_folder}')
        return True
        

if __name__== "__main__":
    annotator = AutoAnnotator(
        model_path='runs/restudying_neuro_v6.5s/weights/best.pt',
        images_folder="images", 
        output_folder="output/annotations"
    )

    annotator.run()
