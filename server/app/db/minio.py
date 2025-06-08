import os

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.utils.logger import logger


class MinioClient:
    def __init__(self):
        # 初始化 MinIO 客户端
        self.client = Minio(
            endpoint=settings.MINIO_SERVER_URL,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )

    def create_bucket(self, bucket_name: str):
        """创建存储桶"""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.success(f"Bucket '{bucket_name}' created successfully")
                return
            logger.info(f"Bucket '{bucket_name}' already exists")
        except S3Error as e:
            logger.error(f"Error creating bucket: {e}")

    def upload_file(self, bucket_name: str, object_name: str, file_path: str):
        """上传文件"""
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            logger.success(
                f"File '{file_path}' uploaded successfully to '{bucket_name}/{object_name}'"
            )
        except S3Error as e:
            logger.error(f"Error uploading file: {e}")

    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        """下载文件"""
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            logger.success(f"File downloaded successfully to {file_path}")
        except S3Error as e:
            logger.error(f"Error downloading file: {e}")

    def list_objects(self, bucket_name: str):
        """列出存储桶中的所有对象"""
        try:
            objects = self.client.list_objects(bucket_name)
            for obj in objects:
                logger.info(
                    f"Object: {obj.object_name}, Size: {obj.size} bytes"
                )
        except S3Error as e:
            logger.error(f"Error listing objects: {e}")


def test():
    # 创建 MinIO 客户端实例
    minio_client = MinioClient()

    # 测试基本操作
    bucket_name = "test-bucket"

    # 创建存储桶
    minio_client.create_bucket(bucket_name)

    # 上传文件
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write("Hello MinIO!")

    minio_client.upload_file(bucket_name, "test.txt", test_file)

    # 列出存储桶中的对象
    minio_client.list_objects(bucket_name)

    # 下载文件
    minio_client.download_file(bucket_name, "test.txt", "downloaded_test.txt")

    # 清理测试文件
    os.remove(test_file)
    os.remove("downloaded_test.txt")


if __name__ == "__main__":
    test()
