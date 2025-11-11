import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else '无'}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else '无'}")

import numpy as np
print("NumPy版本:", np.__version__)

#check python version
import sys
print(f"Python版本: {sys.version}")

import pytorch_lightning as pl
print(pl.__version__)