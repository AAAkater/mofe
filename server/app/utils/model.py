# from lama.model import Lama
from io import BytesIO


def image_to_image(file_content: bytes):
    _ = BytesIO(file_content)

    return file_content
