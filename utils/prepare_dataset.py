import os

import numpy as np
import tqdm
from tqdm.notebook import tqdm


class PrepareDataset:
    def __init__(self, image_dir: str, label_dir: str) -> None:
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
        for file_name in tqdm(os.listdir(self.image_dir)):
            if file_name.endswith((".jpg", ".png")):
                imaga_file_path = os.path.join(self.image_dir, file_name)
                label_file_path = os.path.join(self.label_dir, file_name.replace(
                    ".jpg", ".txt").replace(".png", ".txt"))

                if not os.path.exists(label_file_path):
                    print(f"Label file not found for image: {imaga_file_path}")
                    continue

                with open(label_file_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    # 0 0.458736435546875 0.3806510419921875 0.3614540244140625 0.389472591796875 0.36892872265625 0.48237111328125 0.457112263671875 0.471341630859375 0.458736435546875 0.3806510419921875
                    try:
                        values = [value for value in line.split()]
                        class_id, *coords = map(float, values)
                        image_paths.append(imaga_file_path)
                        class_ids.append(int(class_id))
                        bboxes.append(np.array(coords))

                    except Exception as e:
                        # TODO: Handle specific exceptions
                        print(f"[ERROR] - {e}")
                        continue

        return image_paths, class_ids, bboxes
