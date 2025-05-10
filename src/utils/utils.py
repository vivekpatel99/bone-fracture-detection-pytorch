import collections
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from torchvision import io
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src import utils
from src.utils import rich_utils

log = utils.get_pylogger(__name__)


def create_dataset_dir(dataset_split_dir: Path, dataset_df: pd.DataFrame) -> None:
    dataset_split_dir.mkdir(parents=True, exist_ok=True)
    for dir_name in dataset_df.columns:
        os.makedirs(dataset_split_dir / dir_name, exist_ok=True)
        for file_path in dataset_df[dir_name]:
            file_name = file_path.name
            new_file_path = dataset_split_dir / dir_name / file_name
            if not new_file_path.exists():
                os.symlink(file_path, new_file_path)


def remove_dir(dir_path: Path) -> None:
    if dir_path.exists():
        log.info(f"Removing directory: {dir_path}")
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error: {e.strerror}")
    else:
        log.info(f"Directory does not exist: {dir_path}")


def save_images_from_datasets(img_save_path: Path, dataset: ImageFolder):
    class_image_counters = collections.defaultdict(int)

    img_save_path.mkdir(parents=True, exist_ok=True)
    for image_data_tuple in tqdm(iterable=dataset, desc="Saving images", unit="image"):
        img = image_data_tuple[0]
        label = dataset.classes[image_data_tuple[1]]
        log.debug(f"Label: {label}, Image size: {img.size()[-2:]}")

        current_count = class_image_counters[label]

        img_path = img_save_path / label
        log.debug(f"Image path: {img_path}")
        img_path.mkdir(parents=True, exist_ok=True)
        io.write_jpeg(input=img, filename=str(img_path / f"{label}_{current_count}.jpeg"), quality=100)
        class_image_counters[label] += 1


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
