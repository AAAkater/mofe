import io

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import LongestMaxSize, Normalize, PadIfNeeded, ToTensorV2
from PIL import Image
from unet.model import UNet


def load_model(model_path: str):
    """加载模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


def get_transform():
    """获取图像预处理变换"""
    return A.Compose(
        [
            LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
            PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def process_single_image(image_bytes: bytes, model, device: str) -> bytes:
    """处理单张图片

    Args:
        image_bytes: 输入图片的字节数据
        model: 加载好的模型
        device: 设备类型 ('cuda' or 'cpu')

    Returns:
        bytes: 处理后图片的字节数据
    """
    # 将字节数据转换为PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # 转换为RGB模式
    image = image.convert("RGB")
    # 转换为numpy数组
    image_np = np.array(image)

    # 获取预处理transform
    transform = get_transform()

    # 应用预处理
    processed = transform(image=image_np)
    input_tensor = processed["image"]

    # 添加batch维度并移到指定设备
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 后处理
    output_np = output_tensor.squeeze(0).cpu().numpy()
    output_np = output_np.transpose(1, 2, 0)
    output_np = (output_np * 0.5 + 0.5) * 255
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    # 转换回PIL Image
    output_image = Image.fromarray(output_np)

    # 转换为字节数据
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


# 使用示例：
if __name__ == "__main__":
    # 加载模型
    model_path = "./checkpoints/unet_final.pth"
    model, device = load_model(model_path)

    image_path = "./bbbb.png"
    # 读取测试图片
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 处理图片
    output_bytes = process_single_image(image_bytes, model, device)

    # 保存结果
    with open("output.png", "wb") as f:
        f.write(output_bytes)
