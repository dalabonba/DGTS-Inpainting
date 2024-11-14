import cv2
import numpy as np

def crop_mask(original_image, mask):
    """
    將原始圖片根據遮罩圖進行裁剪，只保留非遮罩區域
    
    Parameters:
    original_image: numpy.ndarray - 原始圖片
    mask: numpy.ndarray - 遮罩圖（0表示要保留的區域，1表示要移除的區域）
    
    Returns:
    numpy.ndarray: 裁剪後的圖片
    """

    # 確保遮罩和原圖大小相同
    if original_image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # 將遮罩轉換為布爾數組（0變成True，1變成False）
    binary_mask = (mask == 0)
    
    # 如果原圖是彩色圖片，需要擴展遮罩維度以匹配通道數
    if len(original_image.shape) == 3:
        binary_mask = np.expand_dims(binary_mask, axis=-1)
        binary_mask = np.repeat(binary_mask, original_image.shape[2], axis=-1)
    
    # 應用遮罩
    masked_image = np.where(binary_mask, original_image, 0)
    
    return masked_image

def compute_brightness_ratio(source_img, target_img):
    # 計算兩張圖片的平均亮度
    source_mean = np.mean(source_img)
    target_mean = np.mean(target_img)
    
    # 計算亮度調整係數
    brightness_ratio = target_mean / source_mean
    # print(f"暗圖亮度是亮圖的{brightness_ratio}倍")

    return brightness_ratio

def adjust_depth_map_brightness(source_img_path, target_img_path, mask_path, output_path = ""):
    """
    調整源深度圖的亮度以匹配目標深度圖
    會先進行裁切將兩張深度圖的遮罩區域切掉
    再用切掉遮罩區域的圖片計算亮度調整係數
    再調整亮度
    
    Parameters:
    source_img_path (str): 需要被調整的深度圖 B 的路徑
    target_img_path (str): 作為參考的深度圖 A 的路徑
    mask_path (str): 遮罩圖的路徑
    [option] output_path (str): 輸出結果的路徑
    """
    # 讀取圖片
    source_img = cv2.imread(source_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if source_img is None or target_img is None:
        raise ValueError("無法讀取圖片")
    
    # 裁切
    croped_source = crop_mask(source_img, mask)
    croped_target = crop_mask(target_img, mask)

    # 計算亮度調整係數
    brightness_ratio = compute_brightness_ratio(croped_source, croped_target)

    # 調整亮度
    adjusted_img = cv2.multiply(source_img, brightness_ratio)
    
    # 確保像素值在有效範圍內 (0-255)
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    
    # 儲存結果
    if output_path != "":
        cv2.imwrite(output_path, adjusted_img)
    
    return adjusted_img


# 使用示例
if __name__ == "__main__":
    # 替換為實際的檔案路徑
    source_path = "C:/Users/upup5/Downloads/my_model_test03/data0025_f.jpg"  # 較亮的深度圖 B
    target_path = "C:/Users/upup5/Desktop/research/2_DGTS-Inpainting/data/teeth_seem_inlay/test/data0025.png"  # 較暗的深度圖 A
    mask_path = "C:/Users/upup5/Desktop/research/2_DGTS-Inpainting/data/mask_teeth_seem_inlay/data0025.png" # 遮罩圖
    output_path = "adjusted_depth_map.png"  # 輸出結果
    
    try:
        adjusted_img = adjust_depth_map_brightness(source_path, target_path, mask_path, output_path)
        print(f"調整完成！結果已儲存至 {output_path}")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")