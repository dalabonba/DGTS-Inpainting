import torch
print(torch.cuda.is_available())
x = torch.rand(5,3)
print('PyTorch 版本:', torch.__version__)
print('PyTorch 綁定的 CUDA 版本:', torch.version.cuda)
if torch.cuda.is_available():
    print('實際運行的 CUDA 驅動版本:', torch.version.cuda)
    print('GPU 型號:', torch.cuda.get_device_name(0))
else:
    print('未偵測到可用的 GPU')
