import numpy as np
from ..common.tools import logger

class EarlyStopping(object):
    """
    早停机制
    
    当监控指标在指定的耐心值内没有改善时，停止训练
    
    Args:
        monitor: 要监控的指标
        mode: 'min' 或 'max'，指标是越小越好还是越大越好
        patience: 容忍多少个epoch指标没有改善
        min_delta: 最小改善阈值
        verbose: 是否打印日志
    """
    def __init__(self, monitor='valid_loss', mode='min', patience=3, min_delta=1e-3, verbose=1):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_training = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, val_score):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.val_score_min = score
        elif (self.mode == 'min' and score > self.best_score + self.min_delta) or \
             (self.mode == 'max' and score < self.best_score - self.min_delta):
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_training = True
        else:
            if (self.mode == 'min' and score <= self.best_score) or \
               (self.mode == 'max' and score >= self.best_score):
                self.best_score = score
                self.val_score_min = score
            self.counter = 0
            
    def epoch_step(self, epoch, current):
        """
        在每个epoch后评估模型性能，决定是否应该提前停止训练
        
        Args:
            epoch: 当前epoch
            current: 当前监控指标的值
        """
        score = current
        
        if self.best_score is None:
            self.best_score = score
            self.val_score_min = score
        elif (self.mode == 'min' and score > self.best_score + self.min_delta) or \
             (self.mode == 'max' and score < self.best_score - self.min_delta):
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_training = True
        else:
            if (self.mode == 'min' and score <= self.best_score) or \
               (self.mode == 'max' and score >= self.best_score):
                self.best_score = score
                self.val_score_min = score
            self.counter = 0 