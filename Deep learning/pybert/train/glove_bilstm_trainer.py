import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_

class GloveBiLstmTrainer(object):
    def __init__(self, model, logger, criterion, optimizer, scheduler, epochs,
                 early_stopping=None, batch_metrics=None, epoch_metrics=None,
                 verbose=1, training_monitor=None, model_checkpoint=None,
                 gradient_accumulation_steps=1, n_gpu='0', fp16=False):
        """
        GloVe-BiLSTM模型的训练器
        
        Args:
            model: 模型
            logger: 日志记录器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            epochs: 训练轮数
            early_stopping: 早停对象
            batch_metrics: 批处理指标列表
            epoch_metrics: 轮次指标列表
            verbose: 输出详细程度
            training_monitor: 训练监视器
            model_checkpoint: 模型检查点
            gradient_accumulation_steps: 梯度累积步数
            n_gpu: GPU设置
            fp16: 是否使用混合精度训练
        """
        self.model = model
        self.logger = logger
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics if epoch_metrics else []
        self.batch_metrics = batch_metrics if batch_metrics else []
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.start_epoch = 1
        self.global_step = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_gpu = n_gpu
        self.fp16 = fp16
        
        # 设置设备
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {
            "model": model_save,
            'epoch': epoch,
            'best': best
        }
        return state

    def valid_epoch(self, data):
        pbar = ProgressBar(n_total=len(data), desc="评估中")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask
                )
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        
        # 计算验证损失
        loss = self.criterion(self.outputs, self.targets)
        self.result['valid_loss'] = loss.item()
        
        print("------------- 验证结果 --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                if metric.name() == 'multilabel_report':
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'valid_{metric.name()}'] = value
                else:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'valid_{metric.name()}'] = value
        
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train_epoch(self, train_data):
        # 设置模型为训练模式
        self.model.train()
        pbar = ProgressBar(n_total=len(train_data), desc='训练中')
        tr_loss = AverageMeter()
        self.epoch_reset()
        
        # 遍历训练数据
        for step, batch in enumerate(train_data):
            # batch是元组格式，直接解包
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播 - 对于GloVe模型，我们只需要input_ids和attention_mask
            logits = self.model(
                input_ids=input_ids,
                attention_mask=input_mask
            )
            
            # 计算损失
            loss = self.criterion(logits, label_ids)
            
            # 如果是多GPU，取平均值
            if isinstance(self.n_gpu, str) and self.n_gpu.count(',') > 0:
                loss = loss.mean()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录损失
            tr_loss.update(loss.item(), n=1)
            
            # 更新进度条信息
            if self.verbose >= 1:
                pbar(step=step, info={'loss': loss.item()})
            
            # 收集输出和目标
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        
        print("\n------------- 训练结果 --------------")
        # 计算epoch级别的指标
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        self.result['loss'] = tr_loss.avg
        
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                if metric.name() == 'multilabel_report':
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'{metric.name()}'] = value
                else:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'{metric.name()}'] = value
        
        return self.result

    def train(self, train_data, valid_data):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            valid_data: 验证数据
        """
        self.model.zero_grad()
        seed_everything(42)  # 固定随机种子确保可重现性
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)
            
            # 合并训练和验证日志
            logs = dict(train_log, **valid_log)
            
            # 过滤logs，移除字典类型的值
            filtered_logs = {k: v for k, v in logs.items() if not isinstance(v, dict)}
            
            # 格式化日志项
            log_items = []
            for key, value in logs.items():
                if isinstance(value, dict):
                    continue
                elif isinstance(value, (int, float)):
                    log_items.append(f' {key}: {value:.4f} ')
                else:
                    log_items.append(f' {key}: {value} ')
            
            show_info = f'\nEpoch: {epoch} - ' + "-".join(log_items)
            self.logger.info(show_info)

            # 更新训练监视器
            if self.training_monitor:
                self.training_monitor.epoch_step(filtered_logs) 

            # 保存模型
            if self.model_checkpoint:
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # 早停
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break 