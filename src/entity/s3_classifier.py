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
        bucket_name,
        model_path,
    ):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = S3Bucket()
        self.loaded_model: pl.LightningModule = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.

        Args:
            model_path (str): Key path of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Exception as e:
            print(e)
            return False

    def _load_model(self) -> pl.LightningModule:
        """
        Loads a serialized model from the specified S3 bucket.

        Returns:
            object: The deserialized model object.
        """
        try:
            model_weigthts = self.s3.load_model_weights(model_name=self.model_path, bucket_name=self.bucket_name)
            print(model_weigthts)
        except Exception as e:
            print(e)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            log.info("Saving model to S3 bucket")
            self.s3.upload_file(from_file, to_filename=self.model_path, bucket_name=self.bucket_name, remove=remove)
        except Exception as e:
            print(e)
        log.info("Model saved to S3 bucket")

    # def predict(self, dataframe: DataFrame):
    #     """
    #     :param dataframe:
    #     :return:
    #     """
    #     try:
    #         if self.loaded_model is None:
    #             self.loaded_model = self._load_model()
    #         return self.loaded_model.predict(dataframe=dataframe)
    #     except Exception as e:
    #         print(e)


if __name__ == "__main__":
    secrets = AwsSecrets()
    bucket_name = secrets.bucket_name
    s3_classifier = S3Classifier(bucket_name=bucket_name, model_path="epoch_001.ckpt")
    file_path = "logs/train/runs/2025-05-14_12-56-57/checkpoints/epoch_001.ckpt"
    s3_classifier.save_model(from_file=file_path)
