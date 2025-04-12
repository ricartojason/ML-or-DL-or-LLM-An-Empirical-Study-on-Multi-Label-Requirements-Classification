#encoding:utf-8
import torch
import numpy as np
import os
import json
import time
from pathlib import Path
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,logger,n_gpu,test_metrics,log_dir=None,model_name=None):
        self.model = model
        self.logger = logger
        self.test_metrics = test_metrics
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.log_dir = log_dir
        self.model_name = model_name or "model"

    def test_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.test_metrics:
            metric.reset()

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        self.test_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                # 这里不需要显式应用sigmoid，因为评估指标中会应用
                # logits = logits.sigmoid()
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        print("\n------------- test result --------------")
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        if self.test_metrics:
            for metric in self.test_metrics:
                if metric.name() == 'multilabel_report':
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'test_{metric.name()}'] = value
                else:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'test_{metric.name()}'] = value
        logs = dict(self.result)
        show_info = '\nTest:'
        for key, value in logs.items():
            if isinstance(value, dict):
                show_info += f' {key}: {value} -'
            else:
                show_info += f' {key}: {value:.4f} -'
        if show_info.endswith('-'):
            show_info = show_info[:-1]
        self.logger.info(show_info)
        
        # 保存测试指标到日志文件
        self._save_test_metrics_to_file(logs)

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return logs
        
    def _save_test_metrics_to_file(self, logs):
        """保存测试指标到日志文件"""
        try:
            if self.log_dir is None:
                # 如果没有指定日志目录，尝试使用默认目录
                current_dir = Path(os.getcwd())
                self.log_dir = current_dir / "logs" / "test_results"
            else:
                self.log_dir = Path(self.log_dir)
                
            # 创建日志目录
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建日志文件名：时间戳_模型名_test_results.json
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            log_file = self.log_dir / f"{timestamp}_{self.model_name}_test_results.json"
            
            # To ensure detailed metrics for each class, we need to extract directly from test metrics objects
            detailed_metrics = {}
            multilabel_report_data = None
            
            # Collect detailed reports from test metrics
            for metric in self.test_metrics:
                if metric.name() == 'multilabel_report':
                    # Get report data directly from object
                    multilabel_report_data = metric.get_detailed_report()
            
            # 处理logs字典，确保所有内容都可以序列化为JSON
            serializable_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                    serializable_logs[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_logs[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    serializable_logs[key] = value.tolist()
                else:
                    serializable_logs[key] = str(value)
            
            # 将结果写入JSON文件
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_logs, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"Test metrics saved to: {log_file}")
            
            # 写入文本报告
            txt_file = self.log_dir / f"{timestamp}_{self.model_name}_test_results.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Test Time: {timestamp}\n")
                f.write(f"Model Name: {self.model_name}\n")
                f.write("-" * 50 + "\n")
                f.write("Test Metrics Summary:\n")
                
                # 写入非字典类型的指标
                for key, value in logs.items():
                    if not isinstance(value, dict):
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                # 写入详细的类别报告
                f.write("\n" + "=" * 50 + "\n")
                f.write("Detailed Class Reports:\n")
                f.write("=" * 50 + "\n\n")
                
                # 写入MultiLabelReport的详细数据
                f.write("Multi-Label Classification Report:\n")
                f.write("-" * 40 + "\n")
                
                # 写入表头
                f.write(f"{'Label':20} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Support':10}\n")
                f.write("-" * 60 + "\n")
                
                # 确保我们有数据可写
                if multilabel_report_data:
                    # 写入每个标签的指标
                    for label, metrics in multilabel_report_data.items():
                        if label not in ['macro avg', 'weighted avg', 'accuracy']:
                            precision = metrics.get('precision', 0)
                            recall = metrics.get('recall', 0)
                            f1 = metrics.get('f1-score', 0)
                            support = metrics.get('support', 0)
                            f.write(f"{label:20} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}\n")
                    
                    # 写入平均指标
                    f.write("-" * 60 + "\n")
                    for avg in ['macro avg', 'weighted avg']:
                        if avg in multilabel_report_data:
                            metrics = multilabel_report_data[avg]
                            precision = metrics.get('precision', 0)
                            recall = metrics.get('recall', 0)
                            f1 = metrics.get('f1-score', 0)
                            support = metrics.get('support', 0)
                            f.write(f"{avg:20} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}\n")
                else:
                    f.write("无法获取多标签分类详细报告数据\n")
                
                f.write("\n\n")
                
                # Write raw test metrics
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("Raw Test Metrics (JSON format):\n")
                f.write("=" * 50 + "\n\n")
                f.write(json.dumps(serializable_logs, indent=4, ensure_ascii=False))
            
            self.logger.info(f"Human-readable test metrics saved to: {txt_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving test metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())








