from pathlib import Path

def get_all_filenames(folder_path):
    # 確認路徑是否存在
    path = Path(folder_path)
    if not path.exists():
        print(f"資料夾路徑 {folder_path} 不存在。")
        return []

    # 獲取資料夾中的所有檔案名稱，返回其中裝著Path物件的list
    filenames = [file for file in path.glob('*') if file.is_file()]
    
    return filenames


# 使用範例
if __name__ == "__main__": # 只有直接執行此檔案時為True
    folder_path = r'C:\Users\upup5\Desktop\research\DGTS-Inpainting\data\teeth_seem_inlay\test'
    filenames = get_all_filenames(folder_path)

    print(f"路徑 ({folder_path}) 中共有{len(filenames)}個檔案")
    # 列出所有檔案名稱(含後綴)
    for filename in filenames:
        print(filename.name)