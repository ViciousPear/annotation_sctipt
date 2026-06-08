#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from annotate import AutoAnnotator
from prepare_for_cvat import CVATUploader

# ========== КОНФИГУРАЦИЯ ==========
CONFIG = {
    # Пути
    "model_path": "runs/restudying_neuro_v6.5s/weights/best.pt",
    "images_folder": "images",
    "annotations_folder": "output/annotations",
    "output_zip": "cvat_upload.zip",
    
    # Классы (В ТОМ ЖЕ ПОРЯДКЕ, что и в модели)
    "labels": [
        "automatics",
        "transformators",
        "WH_Counters",
        "other_rect",
        "bracing",
        "other_circle",
        "other_device",
        "documentation_list",
        "text",
        "diff_auto",
        "uzo",
        "other_switch"
    ],
    
    # Параметры аннотации
    "device": "cpu",  
    "conf_threshold": 0.25,  # порог уверенности
}
# ==================================

def main():
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
    
    # Шаг 1: Автоматическая аннотация
    print("\n ШАГ 1: Автоматическая аннотация")
    
    annotator = AutoAnnotator(
        model_path=CONFIG["model_path"],
        images_folder=CONFIG["images_folder"],
        output_folder=CONFIG["annotations_folder"],
        device=CONFIG["device"],
        conf_threshold=CONFIG["conf_threshold"]
    )
    
    if not annotator.run():
        print("Ошибка при аннотации. Проверьте настройки.")
        return
    
    # Шаг 2: Подготовка для CVAT
    print("\n\nШАГ 2: Подготовка для CVAT")
    
    uploader = CVATUploader(
        images_folder=CONFIG["images_folder"],
        annotations_folder=CONFIG["annotations_folder"],
        labels=CONFIG["labels"],
        output_zip=CONFIG["output_zip"]
    )
    
    if uploader.create_zip():
        uploader.print_instructions()
    else:
        print("Ошибка при создании архива")

if __name__ == "__main__":
    main()