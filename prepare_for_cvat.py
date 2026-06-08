import os
import shutil
import zipfile
from pathlib import Path

class CVATUploader:
    def __init__(self, annotations_folder, images_folder, labels, output_zip="labels.zip"):
        """
        Args:
            annotations_folder: Папка, где лежат .txt файлы с аннотациями.
            images_folder: Папка, где лежат оригинальные изображения (для определения формата).
            labels: Список названий классов (в порядке их id).
            output_zip: Имя выходного ZIP-архива.
        """
        self.annotations_folder = Path(annotations_folder)
        self.images_folder = Path(images_folder)
        self.labels = labels
        self.output_zip = output_zip

    def get_image_format(self, txt_file):
        """Определяет формат изображения по наличию файла .jpg или .png"""
        jpg_path = self.images_folder / f"{txt_file.stem}.jpg"
        png_path = self.images_folder / f"{txt_file.stem}.png"
        
        if jpg_path.exists():
            return "jpg"
        elif png_path.exists():
            return "png"
        else:
            return None

    def create_zip(self):
        """Создает ZIP-архив аннотаций в формате, требуемом CVAT."""
        txt_files = list(self.annotations_folder.glob("*.txt"))

        if not txt_files:
            print(f" Не найдено .txt файлов в папке {self.annotations_folder}")
            return False

        print(f" Найдено {len(txt_files)} файлов аннотаций.")

        # Создаем временную директорию для сборки архива
        temp_dir = Path("temp_cvat_import")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        # 1. Создаем папку для аннотаций
        annotations_dir = temp_dir / "obj_train_data"
        annotations_dir.mkdir()

        # 2. Копируем все .txt файлы
        for txt_file in txt_files:
            shutil.copy2(txt_file, annotations_dir / txt_file.name)
        print(f" Скопировано {len(txt_files)} .txt файлов")

        # 3. Создаем файл obj.names
        names_path = temp_dir / "obj.names"
        with open(names_path, 'w') as f:
            f.write("\n".join(self.labels))
        # Удаляем последнюю пустую строку, если она есть
        with open(names_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            if f.read() == b'\n':
                f.seek(-1, os.SEEK_END)
                f.truncate()
        print(f" Создан {names_path.name} ({len(self.labels)} классов)")

        # 4. Создаем файл train.txt с ПРАВИЛЬНЫМИ путями (с data/)
        train_path = temp_dir / "train.txt"
        train_lines = []  # Сохраняем строки для вывода
        with open(train_path, 'w') as f:
            for txt_file in txt_files:
                # Определяем формат изображения
                img_format = self.get_image_format(txt_file)
                if img_format is None:
                    print(f"⚠️ Не найден файл изображения для {txt_file.name}")
                    continue
                image_name = f"{txt_file.stem}.{img_format}"
                # ВАЖНО: путь с data/ в начале
                line = f"data/obj_train_data/{image_name}"
                train_lines.append(line)
                f.write(line + "\n")
        # Удаляем последнюю пустую строку в train.txt
        with open(train_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            if f.read() == b'\n':
                f.seek(-1, os.SEEK_END)
                f.truncate()
        print(f" Создан {train_path.name} ({len(train_lines)} изображений)")

        # 5. Создаем файл obj.data (с путями, начинающимися с data/)
        data_path = temp_dir / "obj.data"
        data_content = f"""classes = {len(self.labels)}
train = data/train.txt
names = data/obj.names
backup = backup/"""
        with open(data_path, 'w') as f:
            f.write(data_content)
        print(f" Создан {data_path.name}")

        # Показываем содержимое train.txt ДО удаления временной папки
        print("\n Содержимое train.txt (первые 3 строки):")
        for line in train_lines[:3]:
            print(f"   {line}")

        # 6. Создаем ZIP-архив
        with zipfile.ZipFile(self.output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, arcname=file_path.relative_to(temp_dir))

        # 7. Очищаем временную папку
        shutil.rmtree(temp_dir)

        print(f"\n🎉 Архив '{self.output_zip}' успешно создан!")
        
        # Выводим структуру архива
        print("\n СТРУКТУРА АРХИВА:")
        print(f"{self.output_zip}")
        print("├── obj_train_data/")
        print("│   └── (файлы .txt)")
        print("├── train.txt")
        print("├── obj.names")
        print("└── obj.data")
        
        return True

    def print_instructions(self):
        """Выводит инструкцию по использованию архива."""
        print("\n" + "="*60)
        print("📤 ИНСТРУКЦИЯ ПО ЗАГРУЗКЕ В CVAT")
        print("="*60)
        print("1. Войдите в CVAT.")
        print("2. Создайте задачу (Task).")
        print("3. При создании задачи загрузите ваши ИЗОБРАЖЕНИЯ (архив с ними или папку).")
        print(f"4. Убедитесь, что в задаче созданы ВСЕ классы (лейблы) в ТОМ ЖЕ ПОРЯДКЕ:")
        for i, label in enumerate(self.labels):
            print(f"   {i}: {label}")
        print(f"5. Откройте созданную задачу.")
        print(f"6. Нажмите Actions -> Upload annotations.")
        print(f"7. Выберите формат YOLO 1.1.")
        print(f"8. Загрузите файл '{self.output_zip}'.")
        print("\n✨ АННОТАЦИИ ПОЯВЯТСЯ НА ИЗОБРАЖЕНИЯХ!")


if __name__ == "__main__":
    labels = [
        "automatics", "transformators", "WH_Counters",
        "other_rect", "bracing", "other_circle",
        "other_device", "documentation_list", "text",
        "diff_auto", "uzo", "other_switch"
    ]
    
    uploader = CVATUploader(
        annotations_folder="output/annotations",
        images_folder="images",
        labels=labels,
        output_zip="cvat_upload.zip"
    )
    
    if uploader.create_zip():
        uploader.print_instructions()