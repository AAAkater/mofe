from io import BytesIO

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.utils.logger import logger


class MinioClient:
    def __init__(self, bucket: str):
        self.client = Minio(
            endpoint=settings.MINIO_SERVER_URL,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )
        self.bucket_name: str = bucket
        self.create_bucket()

    def create_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.success(
                    f"Bucket '{self.bucket_name}' created successfully"
                )
                return
            logger.info(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as e:
            logger.error(f"Error creating bucket: {e}")

    def upload_file(self, object_name: str, file_data: bytes):
        """
        Uploads a file to the specified bucket.

        Args:
            object_name (str): The name of the object to be uploaded.
            file_data (bytes): The file data in bytes to be uploaded.

        Returns:
            None

        Raises:
            Any exceptions raised by the underlying client's put_object method.

        Notes:
            Logs a success message upon successful upload with the bucket name and object name.
        """
        file_stream = BytesIO(file_data)
        self.client.put_object(
            self.bucket_name,
            object_name,
            file_stream,
            length=len(file_data),
        )
        logger.success(
            f"File uploaded successfully to '{self.bucket_name}/{object_name}'"
        )

    def download_file(self, object_name: str) -> bytes:
        """
        Downloads a file from the specified bucket and returns its contents as bytes.

        Args:
            object_name (str): The name of the object (file) to download from the bucket.

        Returns:
            bytes: The contents of the downloaded file.

        Raises:
            Any exceptions raised by the underlying client's `get_object` method will propagate.

        Example:
            >>> file_data = download_file("example.txt")
            >>> print(
            ...     len(file_data)
            ... )  # Prints the size of the downloaded file in bytes.

        Note:
            This method logs a success message upon successful download using the module's logger.
        """
        data = self.client.get_object(self.bucket_name, object_name)
        file_data = data.read()
        logger.success(
            f"File downloaded successfully from {self.bucket_name}/{object_name}"
        )
        return file_data

    def list_specific_objects(self, object_names: list[str]) -> list[dict]:
        """
        Get information for specific objects by their names using batch listing.

        Args:
            object_names (list[str]): List of object names to get information for.

        Returns:
            list[dict]: List of dictionaries containing object information.
                       Each dict contains 'name', 'size', and 'last_modified'.
        """
        try:
            # Create a set of object names for faster lookup
            name_set = set(object_names)
            results = []

            # Get all objects in one request
            objects = self.client.list_objects(
                self.bucket_name,
                recursive=True,  # Include all objects in subdirectories
            )

            # Filter objects that match our requested names
            for obj in objects:
                if obj.object_name in name_set:
                    results.append(
                        {
                            "name": obj.object_name,
                            "size": obj.size,
                            "last_modified": obj.last_modified,
                        }
                    )

            return results
        except S3Error as e:
            logger.error(f"Error listing specific objects: {e}")
            return []


minio_client = MinioClient(settings.MINIO_BUCKET_NAME)
