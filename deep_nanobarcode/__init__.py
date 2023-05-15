import torch


is_cuda_available = torch.cuda.is_available()

print(f"Is cuda available: {is_cuda_available}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

print(f"Cuda device name: {torch.cuda.get_device_name(0)}")