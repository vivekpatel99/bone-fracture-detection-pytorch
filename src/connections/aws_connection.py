import boto3

from src.entity.aws_secrets import AwsSecrets


class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, aws_scerets: AwsSecrets):
        """
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket
        and raise exception when environment variable is not set
        """
        self.aws_scerets = aws_scerets
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = self.aws_scerets.access_key
            __secret_access_key = self.aws_scerets.secret_key
            if __access_key_id is None:
                raise Exception("Environment variable AWS_ACCESS_KEY_ID_ENV_KEY is not set")
            if __secret_access_key is None:
                raise Exception("Environment variable AWS_SECRET_ACCESS_KEY_ENV_KEY is not set")

            S3Client.s3_resource = boto3.resource(
                "s3",
                region_name=self.aws_scerets.region_name,
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
            )
            S3Client.s3_client = boto3.client(
                "s3",
                region_name=self.aws_scerets.region_name,
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
