import torch
print(torch.cuda.get_device_name(0))
print(torch.zeros(1, device="cuda"))