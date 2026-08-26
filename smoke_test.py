# import torch
# print(torch.cuda.get_device_name(0))
# print(torch.zeros(1, device="cuda"))

raise RuntimeError(
    "CUDA unknown error - this may be due to an incorrectly set up environment, "
    "e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. "
    "Setting the available devices to be zero."
)
