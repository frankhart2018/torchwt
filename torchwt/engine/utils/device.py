import torch


def get_device():
    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        print(f"Using GPU: {torch.cuda.get_device_name()}, using this!")

    return torch.device("cuda:0" if is_cuda_available else "cpu")