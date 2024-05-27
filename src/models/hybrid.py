import torch.nn as nn

try:
    from torch.cuda.amp import autocast
except ModuleNotFoundError:
    def autocast():
        return lambda f: f
import time
class Hybrid(nn.Module):
    def __init__(self, gcnn, group_transformer):
        super().__init__()
        self.gcnn = gcnn
        self.group_transformer = group_transformer
    
    @autocast() # required for mixed-precision when training on multiple GPUs.
    def forward(self, x):
        a = time.time()
        out = self.gcnn(x)  # B, C, H, W
        b = time.time()
        a = time.time()
        out = self.group_transformer(out)
        b = time.time()
        return out