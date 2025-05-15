import os
import secrets
import sys
from io import BufferedWriter, BytesIO

from src import utils

# from mypy_boto3_s3.service_resource import Bucket
from src.connections.aws_connection import S3Client
from src.entity.aws_secrets import AwsSecrets

log = utils.get_pylogger(__name__)


class S3Bucket:
    def __init__(
        self,
    ):
        self.secrets = AwsSecrets()
        self.bucket_name = self.secrets.bucket_name
        self.s3_client = S3Client(aws_scerets=self.secrets)
        self.s3_resource = self.s3_client.s3_resource
        self.s3_client = self.s3_client.s3_client
        self.s3_key = ""

    def get_bucket(self):
        """
        Retrieves the S3 bucket object based on the provided bucket name.

        Args:
            bucket_name (str): The name of the S3 bucket.

        Returns:
            Bucket: S3 bucket object.
        """
        log.info("Entered the get_bucket method of SimpleStorageService class")
        try:
            bucket = self.s3_resource.Bucket(self.bucket_name)
            log.info("Exited the get_bucket method of SimpleStorageService class")
            return bucket
        except Exception as e:
            print(e)
            raise

    def s3_key_path_available(self, key: str) -> bool:
        """
        Checks if a specified S3 key path (file path) is available in the specified bucket.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            bucket = self.get_bucket()
            file_object = list(bucket.objects.filter(Prefix=key))
            return len(file_object) > 0
        except Exception as e:
            log.error(e)
            raise

    def read_stream_object(self, model_key: str) -> str | BytesIO:
        """
        Reads the specified S3 object with optional decoding and formatting.

        Args:
            make_readable (bool): Whether to convert content to StringIO for DataFrame usage.

        Returns:
            Union[StringIO, str]: The content of the object, as a StringIO or decoded string.
        """
        try:
            s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=model_key)
            model_stream = s3_object["Body"]
            model_buffer = BytesIO(model_stream.read())
            return model_buffer
        except Exception as e:
            log.error(e)
            raise

    def fetch_object(self, save_file_path: str, s3_object_key: str):
        with open(save_file_path, "wb") as f:
            self.s3_client.download_fileobj(self.bucket_name, s3_object_key, f)

    def fetch_model_weights(self, cloud_model_key: str, model_save_path: str) -> str:
        """
        Loads a serialized model from the specified S3 bucket.

        Args:
            model_name (str): Name of the model file in the bucket.
            model_dir (str): Directory path within the bucket.

        Returns:
            object: The deserialized model object.
        """
        try:
            self.fetch_object(save_file_path=model_save_path, s3_object_key=cloud_model_key)

            log.info(f"Production model weights fetched from S3 bucket and saved at {model_save_path}")
            return model_save_path
        except Exception as e:
            log.error(e)
            raise

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """
        Uploads a local file to the specified S3 bucket with an optional file deletion.

        Args:
            from_filename (str): Path of the local file.
            to_filename (str): Target file path in the bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool): If True, deletes the local file after upload.
        """
        log.info("Entered the upload_file method of SimpleStorageService class")
        try:
            log.info(f"Uploading file: {from_filename} to bucket: {bucket_name}")
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)
            # Delete local file if remove=True
            if remove:
                log.info(f"Removing file: {from_filename}")
                os.remove(from_filename)

            log.info("Exited the upload_file method of SimpleStorageService class")
        except Exception as e:
            log.error(e)
            raise


if __name__ == "__main__":
    from src.entity.aws_secrets import AwsSecrets

    bucket_name = AwsSecrets().bucket_name
    bucket = S3Bucket(bucket_name)
