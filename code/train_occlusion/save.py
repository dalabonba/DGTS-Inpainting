import torch
import os

class ModelCheckpoint:
    def __init__(self, save_path, monitor=['avg_lossa'], mode='min', keep_top_k=3):
        """
        初始化模型檢查點管理器
        
        參數:
        - save_path: 模型儲存路徑
        - monitor: 用於追蹤模型性能的指標列表
        - mode: 指標的評估模式 ('min' 或 'max')
        - keep_top_k: 保留最佳的 K 個模型
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.keep_top_k = keep_top_k
        self.best_models = {metric: [] for metric in monitor}
        
        # 確保儲存路徑存在
        os.makedirs(save_path, exist_ok=True)
    
    def step(self, model, metrics, epoch):
        """
        根據性能指標決定是否儲存模型
        
        參數:
        - model: 要儲存的神經網路模型
        - metrics: 包含性能指標的字典
        - epoch: 當前訓練的 epoch 數
        """
        for metric in self.monitor:
            current_metric = metrics.get(metric)
            
            if current_metric is None:
                print(f"Warning: Metric {metric} not found in metrics dictionary.")
                continue
            
            # 決定是否儲存模型的條件
            should_save = (
                len(self.best_models[metric]) < self.keep_top_k or 
                (self.mode == 'min' and current_metric < self.best_models[metric][-1]['metric']) or
                (self.mode == 'max' and current_metric > self.best_models[metric][-1]['metric'])
            )
            
            if should_save:
                # 準備模型檢查點
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metric': current_metric,
                    'all_metrics': metrics
                }
                
                # 建立檔案名稱
                filename = f"{metric}_{current_metric:.4f}_epoch{epoch}.pth"
                filepath = os.path.join(self.save_path, filename)
                
                # 儲存模型
                torch.save(checkpoint, filepath)
                
                # 更新最佳模型列表
                self.best_models[metric].append(checkpoint)
                self.best_models[metric].sort(
                    key=lambda x: x['metric'], 
                    reverse=(self.mode == 'max')
                )
                
                # 如果超過保留數量，刪除最差的模型
                if len(self.best_models[metric]) > self.keep_top_k:
                    old_model = self.best_models[metric].pop()
                    old_filepath = os.path.join(
                        self.save_path, 
                        f"{metric}_{old_model['metric']:.4f}_epoch{old_model['epoch']}.pth"
                    )
                    os.remove(old_filepath)
                
                print(f"Saved best model for {metric}: {filename}")