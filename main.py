import torch
# [ADDED] Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED hang in smooth_output's
# F.conv2d call. The system cuDNN (9.2.0) conflicts with PyTorch's bundled version (9.1.0).
# This has no effect on model accuracy — it falls back to PyTorch's native convolution.
torch.backends.cudnn.enabled = False

from cfg import args
from trainer import build_trainer_from_cfg
from dataset import get_dataloader

def main():
    train_loader, val_loader, test_loader = get_dataloader(args.train_ratio, args.val_ratio)
    solver = build_trainer_from_cfg(train_loader, val_loader, test_loader)
    solver.run()

if __name__ == '__main__':
    main()