import cv2
import numpy as np


def create_corrupted_image(
    original_path, mask_path, output_path="corrupted.png"
):
    """
    创建破损图片
    Args:
        original_path: 原始图片路径
        mask_path: 掩码图片路径（白色区域保留，黑色区域擦除）
        output_path: 输出图片路径，默认为'corrupted.png'
    """
    # 加载数据
    original_img = cv2.imread(original_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 单通道掩码

    resized_mask = cv2.resize(
        mask,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    # 二值化（可选，确保严格0/255）
    _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    # 调整掩码尺寸

    # 合成破损图片 - 直接使用mask_float，不再使用(1 - mask_float)
    mask_float = (binary_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    corrupted_img = original_img * mask_float  # 修改这里，直接使用mask_float

    # 保存结果
    cv2.imwrite(
        output_path,
        cv2.cvtColor(corrupted_img.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    return corrupted_img


if __name__ == "__main__":
    # 使用示例
    org_path = "./cuit图标.jpg"
    mask_path = "./irregular_mask/disocclusion_img_mask/00124.png"
    create_corrupted_image(org_path, mask_path, "./corrupted.png")
