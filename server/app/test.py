import torch


def test_cuda():
    # 检查CUDA是否可用
    print(f"CUDA is available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        # 显示CUDA设备信息
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

        # 创建一个测试张量并移动到GPU
        x = torch.rand(5, 3)
        print("\nCPU Tensor:")
        print(x)

        x = x.cuda()
        print("\nGPU Tensor:")
        print(x)
    else:
        print("CUDA is not available. Running on CPU only.")


def main():
    print("Hello from backend!")
    test_cuda()


if __name__ == "__main__":
    main()
