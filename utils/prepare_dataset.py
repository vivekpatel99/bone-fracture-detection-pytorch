import os
from pathlib import Path

import numpy as np
import tqdm
from tqdm.notebook import tqdm


class PrepareDataset:
    def __init__(self, image_dir: Path, label_dir: Path) -> None:
        """
        Args:
            image_dir (str): Path to the directory containing the images.
            label_dir (str): Path to the directory containing the labels.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir

    def get_dataset(self) -> tuple[list[str], list[int], list[np.ndarray]]:
        """Loads and parses YOLOv8 labels.

        Args:
            None

        Returns:
            tuple[list[str], list[int], list[np.ndarray]]: A tuple containing:
                - A list of image paths.
                - A list of class ids.
                - A list of bounding boxes, where each bounding box is an array of 8 floats.
        """
        image_paths = []
        class_ids = []
        bboxes = []
        for file_name in tqdm(self.image_dir.iterdir()):
            if file_name.suffix.endswith((".jpg", ".png")):
                imaga_file_path = file_name
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                if not label_file_path.exists():
                    print(f"Label file not found for image: {imaga_file_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                # NOTE: for the time being avoid using ragged batches, that's why these line commented out
                # image_bboxes = []
                # image_classes = []

                # 0 0.458736435546875 0.3806510419921875 0.3614540244140625 0.389472591796875 0.36892872265625 0.48237111328125 0.457112263671875 0.471341630859375 0.458736435546875 0.3806510419921875
                for line in lines:
                    try:
                        values = np.array([value for value in line.split()], dtype=np.float32)
                        class_id = int(values[0])
                        coords = values[1:]  # Coords are already normalized YOLO format

                        # Reshape coords to (1, 4) if it's a single box
                        if coords.size == 4:
                            coords = coords.reshape(1, 4)  # Ensure it's a 2D array even for one box

                        # image_bboxes.append(coords)
                        # image_classes.append(class_id)
                        # image_paths.append(str(imaga_file_path))

                        bboxes.append(coords)
                        class_ids.append(class_id)
                        image_paths.append(str(imaga_file_path))
                    except Exception as e:
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    
                # image_paths.append(imaga_file_path)
                # class_ids.append(image_classes)
                # bboxes.append(np.concatenate(image_bboxes, axis=0))  # Concatenate boxes for the image

        return image_paths, class_ids, bboxes
