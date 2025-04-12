# encoding:utf-8
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ..common.tools import load_json
from ..common.tools import save_json
plt.switch_backend('agg')


class TrainingMonitor():
    def __init__(self, file_dir, arch, add_test=False):
        '''
        :param startAt: 重新开始训练的epoch点
        '''
        if isinstance(file_dir, Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")

    def reset(self,start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():

                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            # 跳过字典类型的值
            if isinstance(v, dict):
                continue
                
            # 尝试从self.H字典中获取键k对应的值，并将其存储在变量l中
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, np.float):
                try:
                    v = round(float(v), 4)
                except (TypeError, ValueError):
                    # 如果无法转换为float，则跳过此指标
                    continue
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            save_json(data = self.H,file_path=self.json_path)

        # 保存train图像
        if "loss" in self.H and len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if "loss" in self.H and len(self.H["loss"]) > 1:
            # 指标变化
            # 曲线
            # 需要成对出现
            # keys = [key for key, _ in self.H.items() if '_' not in key]
            keys = [key for key, _ in self.H.items() if
                    key in ['auc','precision', 'recall', 'hamming_score', 'hamming_loss', 'f1']]
            for key in keys:
                # 确保对应的验证指标存在
                if f"valid_{key}" not in self.H:
                    continue
                    
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                
                # 防止长度不匹配错误，始终计算min_len
                min_len = min(len(self.H[key]), len(self.H[f"valid_{key}"]))
                N = N[:min_len]
                
                train_data = self.H[key][:min_len]
                valid_data = self.H[f"valid_{key}"][:min_len]
                
                plt.plot(N, train_data, label=f"train_{key}")
                plt.plot(N, valid_data, label=f"valid_{key}")
                
                if self.add_test and f"test_{key}" in self.H:
                    # 确保测试数据长度也匹配
                    test_len = min(min_len, len(self.H[f"test_{key}"]))
                    test_data = self.H[f"test_{key}"][:test_len]
                    plt.plot(N[:test_len], test_data, label=f"test_{key}")
                    
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {min_len}]")
                plt.savefig(str(self.paths[key]))
                plt.close()
