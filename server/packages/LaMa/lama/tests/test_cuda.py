import torch


def test_cuda():
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # 获取当前设备信息
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_count = torch.cuda.device_count()

        print(f"Current device: {current_device}")
        print(f"Device name: {device_name}")
        print(f"Number of CUDA devices: {device_count}")
    else:
        print(
            "CUDA is not available. Please check your PyTorch installation and GPU drivers."
        )


if __name__ == "__main__":
    test_cuda()
