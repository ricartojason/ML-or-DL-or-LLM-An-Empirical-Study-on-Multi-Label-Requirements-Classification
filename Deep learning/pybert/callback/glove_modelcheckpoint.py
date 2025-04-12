import os
import torch
import numpy as np
from pathlib import Path
from ..common.tools import logger

class GloveModelCheckpoint(object):
    """
    在每个epoch后保存模型，支持以下功能：
    1. 保存最佳模型
    2. 每隔epochs保存模型
    """
    def __init__(self, 
                 checkpoint_dir,
                 monitor,
                 arch,
                 mode='min',
                 epoch_freq=1,
                 best=None,
                 save_best_only=True,
                 model_name="pytorch_model.bin"):
        """
        Args:
            checkpoint_dir: 模型保存根路径
            monitor: 需要监控的指标
            arch: 模型架构名称，用于构建子目录
            mode: 模式，'min'或'max'或'auto'
            epoch_freq: 每隔几个epoch保存一次模型
            best: 当前最佳值，用于恢复训练
            save_best_only: 是否只保存最佳模型
            model_name: 模型文件名，默认为pytorch_model.bin
        """
        # 基本参数
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only
        self.model_name = model_name
        
        # 确保checkpoint_dir是Path对象
        self.checkpoint_dir = Path(checkpoint_dir) if not isinstance(checkpoint_dir, Path) else checkpoint_dir
        
        # 构建模型保存路径: checkpoint_dir/arch
        self.model_dir = self.checkpoint_dir / arch
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置监控模式
        if mode not in ['min', 'max', 'auto']:
            logger.warning(f'ModelCheckpoint mode {mode} 未知，将设为auto模式')
            mode = 'auto'
            
        if mode == 'auto':
            # 根据monitor自动推断模式
            if 'acc' in self.monitor or self.monitor.startswith('f'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        elif mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        else:  # mode == 'max'
            self.monitor_op = np.greater
            self.best = -np.Inf
            
        # 如果提供了best值，覆盖默认值
        if best is not None:
            self.best = best
            
        # 用于记录上一个最佳值
        self.best_prev = self.best
        
        # 用于存储模型
        self.model = None
        
    def epoch_step(self, state, current=None):
        """
        普通epoch步骤，用于一般情况
        
        Args:
            state: 包含模型状态的字典
            current: 当前指标值，如果为None则从state中获取
        """
        # 记录模型
        if self.model is None:
            self.model = state['model']
            
        # 获取当前指标值
        if current is None:
            current = state[self.monitor]
            
        # 判断是否需要保存
        if self.monitor_op(current, self.best):
            # 更新最佳值
            self.best_prev = self.best
            self.best = current
            state.update({'best': self.best})
            
            # 保存模型
            self._save_model(state, is_best=True, current=current)
        elif state['epoch'] % self.epoch_freq == 0 and not self.save_best_only:
            # 保存常规checkpoint
            self._save_model(state, is_best=False, current=current)
            
    def bert_epoch_step(self, current, state):
        """
        专用于BERT模型的epoch步骤
        
        Args:
            current: 当前指标值
            state: 包含模型状态的字典
        """
        # 记录模型
        if self.model is None:
            self.model = state['model']
            
        # 判断是否需要保存
        if self.monitor_op(current, self.best):
            # 更新最佳值
            self.best_prev = self.best
            self.best = current
            state.update({'best': self.best})
            
            # 保存模型
            self._save_model(state, is_best=True, current=current)
        elif state['epoch'] % self.epoch_freq == 0 and not self.save_best_only:
            # 保存常规checkpoint
            self._save_model(state, is_best=False, current=current)
    
    def _save_model(self, state, is_best=True, current=None):
        """
        保存模型的内部方法
        
        Args:
            state: 包含模型状态的字典
            is_best: 是否是最佳模型
            current: 当前指标值
        """
        model_to_save = state['model'].module if hasattr(state['model'], 'module') else state['model']
        
        if is_best or self.save_best_only:
            # 保存最佳模型
            model_path = self.model_dir / self.model_name
            torch.save(model_to_save.state_dict(), str(model_path))
            
            # 保存checkpoint信息
            checkpoint_info = {
                'epoch': state['epoch'],
                'best': self.best,
                'monitor': self.monitor
            }
            info_path = self.model_dir / 'checkpoint_info.bin'
            torch.save(checkpoint_info, info_path)
            
            # 记录日志
            if is_best:
                # 完全按照用户要求的新格式
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best_prev:.5f} to {current:.5f}")
                logger.info(f"Best model saved to: {model_path}")
        else:
            # 保存常规epoch checkpoint
            epoch_dir = self.model_dir / f"checkpoint-epoch-{state['epoch']}"
            epoch_dir.mkdir(exist_ok=True, parents=True)
            
            # 保存模型
            model_path = epoch_dir / self.model_name
            torch.save(model_to_save.state_dict(), str(model_path))
            
            # 保存checkpoint信息
            checkpoint_info = {
                'epoch': state['epoch'],
                'monitor_value': state.get(self.monitor, 'unknown'),
                'monitor': self.monitor
            }
            info_path = epoch_dir / 'checkpoint_info.bin'
            torch.save(checkpoint_info, info_path)
            
            # 记录日志
            logger.info(f"Epoch {state['epoch']}: model saved to {model_path}") 