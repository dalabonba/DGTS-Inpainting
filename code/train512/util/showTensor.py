import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def imshow_tensor(tensor, title=""):
    """
    將 tensor 轉換為圖片並顯示
    假設 tensor 的 shape 為 [B, C, H, W] 且 C=1 (灰階)
    """
    if tensor.is_cuda:  # 確保 tensor 在 CPU
        tensor = tensor.cpu()
    
    if tensor.dim() == 4:  # shape 為 [B, C, H, W]
        tensor = tensor[:4]  # 只取前 4 張圖，避免過多
        grid = make_grid(tensor, nrow=4, normalize=True)
        np_img = grid.numpy().transpose((1, 2, 0))  # 轉換為 numpy
    elif tensor.dim() == 3:  # shape 為 [C, H, W]
        np_img = tensor.numpy().transpose((1, 2, 0))
    elif tensor.dim() == 2:  # shape 為 [H, W] (已 squeeze)
        np_img = tensor.numpy()
    else:
        raise ValueError("不支援的 tensor 形狀")

    plt.figure(figsize=(6, 6))
    plt.imshow(np_img.squeeze(), cmap='gray')  # 確保是灰階圖
    plt.title(title)
    plt.axis('off')
    plt.show()
