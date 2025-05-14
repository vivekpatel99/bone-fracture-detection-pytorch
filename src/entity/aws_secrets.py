import os
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AwsSecrets(BaseSettings):
    # default=... keeps the field required for Pydantic, but satisfies Pylance
    access_key: str = Field(default=..., validation_alias="AWS_ACCESS_KEY_ID")  # os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key: str = Field(default=..., validation_alias="AWS_SECRET_ACCESS_KEY")
    bucket_name: str = Field(default=..., validation_alias="AWS_S3_BUCKET_NAME")
    region_name: str = Field(default=..., validation_alias="AWS_S3_REGION_NAME")

    # aws_access_key_id: str = ""
    # aws_secret_access_key: str = ""
    # aws_s3_bucket_name: str = ""
    # aws_s3_region_name: str = ""


class AwsEcrSecrets(BaseSettings):
    aws_ecr_name: str = Field(default=..., validation_alias="AWS_ECR_NAME")
    aws_ecr_uri: str = Field(default=..., validation_alias="AWS_ECR_URI")
    aws_ecr_machinne: str = Field(default=..., validation_alias="AWS_ECR_MACHINNE")


if __name__ == "__main__":
    print(AwsSecrets().model_dump())
