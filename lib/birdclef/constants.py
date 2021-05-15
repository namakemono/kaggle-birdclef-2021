import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 397
SAMPLING_RATE = 32_000 # サンプリリングレート

