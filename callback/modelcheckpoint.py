from pathlib import Path
import numpy as np
import torch
from ..common.tools import logger
import os

class ModelCheckpoint(object):
    """Save the model after every epoch.
    # Arguments
        checkpoint_dir: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
    """
    def __init__(self, checkpoint_dir,
                 monitor,
                 arch,
                 mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True,
                 model_name = None):
        if isinstance(checkpoint_dir,Path):
            checkpoint_dir = checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.is_dir()
        checkpoint_dir.mkdir(exist_ok=True)
        self.base_path = checkpoint_dir
        self.arch = arch
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

        if save_best_only:
            self.model_name = model_name or f"BEST_{arch}_MODEL.pth"

    def epoch_step(self, state,current):
        '''
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_path/ self.model_name
                torch.save(state, str(best_path))

        else:
            filename = self.base_path / f"epoch_{state['epoch']}_{state[self.monitor]}_{self.arch}_model.bin"
            if state['epoch'] % self.epoch_freq == 0:
                logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                torch.save(state, str(filename))

    def bert_epoch_step(self, state, current):
        model_to_save = state['model']
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                
                # 保存模型到目录
                save_dir = self.base_path
                save_dir.mkdir(exist_ok=True)
                
                # 1. 保存模型配置
                output_config_file = save_dir / 'config.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                
                # 2. 保存模型权重 - 使用自定义名称
                output_model_file = save_dir / self.model_name
                torch.save(model_to_save.state_dict(), str(output_model_file))
                
                # 3. 保存检查点信息
                state_copy = state.copy()
                state_copy.pop("model")
                torch.save(state_copy, save_dir / 'checkpoint_info.bin')
                
                logger.info(f"Model saved as: {output_model_file}")
        else:
            if state['epoch'] % self.epoch_freq == 0:
                save_path = self.base_path / f"checkpoint-epoch-{state['epoch']}"
                save_path.mkdir(exist_ok=True)
                logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                
                # 1. 保存模型配置
                output_config_file = save_path / 'config.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                
                # 2. 保存模型权重 - 使用自定义名称
                output_model_file = save_path / self.model_name
                torch.save(model_to_save.state_dict(), str(output_model_file))
                
                # 3. 保存检查点信息
                state_copy = state.copy()
                state_copy.pop("model")
                torch.save(state_copy, save_path / 'checkpoint_info.bin')
                
                logger.info(f"Model saved as: {output_model_file}")

    def _save_improved_model(self, state):
        if self.save_best:
            # 1. Save model configuration
            if not isinstance(self.checkpoint_dir, Path):
                checkpoint_dir = Path(self.checkpoint_dir)
            else:
                checkpoint_dir = self.checkpoint_dir
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
            # 2. Save model weights with custom name
            output_model_file = checkpoint_dir / self.monitor_op.best_model_name
            state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            torch.save(state_dict, str(output_model_file))
            
            # 3. Save checkpoint information
            checkpoint_info = {'epoch': state['epoch'], 'best': self.best}
            torch.save(checkpoint_info, checkpoint_dir / 'checkpoint_info.bin')
            
            current = state[self.monitor]
            self.logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
            self.logger.info(f"Model saved as: {output_model_file}")
        else:
            current = state[self.monitor]
            # 1. Save model configuration
            if not isinstance(self.checkpoint_dir, Path):
                checkpoint_dir = Path(self.checkpoint_dir)
            else:
                checkpoint_dir = self.checkpoint_dir
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
            # 2. Save model weights with custom name
            output_model_file = checkpoint_dir / f"{self.monitor_op.mode}_{state['epoch']}_{current:.4f}.bin"
            state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            torch.save(state_dict, str(output_model_file))
            
            # 3. Save checkpoint information
            checkpoint_info = {'epoch': state['epoch'], 'best': self.best}
            torch.save(checkpoint_info, checkpoint_dir / 'checkpoint_info.bin')
            
            self.logger.info(f"\nEpoch {state['epoch']}: {self.monitor} was {current:.5f}")
            self.logger.info(f"Model saved as: {output_model_file}")
