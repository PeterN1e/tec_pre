import torch
a = torch.randn(3,3,3)
b = torch.randn(1,3,3)
c= (a,b).unsqueeze