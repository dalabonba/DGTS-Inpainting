import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 假设build_backbone()函数在之前的代码中已定义
from backbone import build_backbone, NestedTensor

def visualize_backbone_outputs(image_path, output_path):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建backbone
    backbone = build_backbone().to(device)
    backbone.eval()  # 设置为评估模式

    # 圖像預處理(複製自dataset_loader.py)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 创建一个虚拟的mask（全1矩阵）
    mask = torch.ones((1, 224, 224), dtype=torch.bool).to(device)
    
    # 将图像和mask包装成NestedTensor
    nested_tensor = NestedTensor(img_tensor, mask)
    
    # 通过backbone处理图像
    with torch.no_grad():
        outputs, _ = backbone(nested_tensor)
    
    # 可视化每一层的输出
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Backbone Outputs", fontsize=16)
    
    for idx, (name, output) in enumerate(zip(['layer5', 'layer6', 'layer7', 'layer8'], outputs)):
        # 获取特征图
        feature_map = output.tensors.squeeze().cpu()
        
        # 计算特征图的平均值across channels
        mean_feature_map = feature_map.mean(dim=0)
        
        # 绘制特征图
        ax = axs[idx // 2, idx % 2]
        im = ax.imshow(mean_feature_map, cmap='viridis')
        ax.set_title(f'{name} Output')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path)
    plt.close(fig)  # 关闭图形以释放内存
    
    print(f"Visualization saved to {output_path}")

# 使用示例
# visualize_backbone_outputs('path_to_your_image.jpg', 'output_visualization.png')

def process_image_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 处理文件夹中的所有图像

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_backbone_output.png")
            
            visualize_backbone_outputs(input_path, output_path)
            print(f"Processed {filename}")

# 使用示例
# process_image_folder('path_to_input_folder', 'path_to_output_folder')
process_image_folder('data/places2/train', 'logs/backboneOutput')