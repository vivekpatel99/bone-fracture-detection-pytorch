import pyrootutils
import pytorch_lightning as pl

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402
from src.cloud_storage.aws_storage import S3Bucket  # noqa: E402
from src.entity.aws_secrets import AwsSecrets  # noqa: E402

log = utils.get_pylogger(__name__)


class S3Classifier:
    """
    This class is used to save and retrieve our model from s3 bucket and to do prediction
    """

    def __init__(
        self,
        bucket_name: str,
        cloud_model_key: str,
        cloud_model_save_path: str,
    ) -> None:
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.cloud_model_key = cloud_model_key
        self.cloud_model_save_path = cloud_model_save_path
        self.s3 = S3Bucket()
        self.loaded_model: pl.LightningModule = None

    def is_model_present(self, model_name: str) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.

        Args:
            model_path (str): Key path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            return self.s3.s3_key_path_available(key=model_name)
        except Exception as e:
            log.error(e)
            return False

    def fetch_model_weights(self) -> str | bool:
        """
        Loads a serialized model from the specified S3 bucket.

        Returns:
            object: The deserialized model object.
        """
        try:
            return self.s3.fetch_model_weights(cloud_model_key=self.cloud_model_key, model_save_path=self.cloud_model_save_path)
        except Exception as e:
            log.error(e)
            return False

    def upload_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            log.info("Saving model to S3 bucket")
            self.s3.upload_file(from_file, to_filename=self.cloud_model_key, bucket_name=self.bucket_name, remove=remove)
        except Exception as e:
            log.error(e)
        log.info("Model saved to S3 bucket")


if __name__ == "__main__":
    secrets = AwsSecrets()
    bucket_name = secrets.bucket_name
    s3_classifier = S3Classifier(bucket_name=bucket_name, cloud_cloud_model_save_path="results")
    file_path = "logs/train/runs/2025-05-14_12-56-57/checkpoints/epoch_001.ckpt"
    # s3_classifier.save_model(from_file=file_path)
    result = s3_classifier.is_model_present(model_path=file_path)
    print(result)
    model_weights = s3_classifier.fetch_model_weights()
