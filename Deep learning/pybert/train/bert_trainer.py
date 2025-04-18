import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import summary
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)  # 设置设备

class BertTrainer(object):
    """
    BERT模型专用训练器，与原始Trainer类功能相同，
    但在调用模型时使用了命名参数以确保与BERT模型兼容
    """
    def __init__(self, args, model, logger, criterion, optimizer, scheduler, early_stopping, epoch_metrics,
                 batch_metrics, verbose=1, training_monitor=None, model_checkpoint=None
                 ):
        self.args = args
        self.model = model
        self.logger = logger
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu=args.n_gpu, model=self.model)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.resume_path:
            self.logger.info(f"\nLoading checkpoint: {args.resume_path}")
            resume_dict = torch.load(args.resume_path / 'checkpoint_info.bin')
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{args.resume_path}' and epoch {self.start_epoch} loaded")

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
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data):
        pbar = ProgressBar(n_total=len(data), desc="Evaluating")
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                # 使用命名参数调用模型
                logits = self.model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask
                ).to(self.device)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            pbar(step=step)
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        loss = self.criterion(target=self.targets, output=self.outputs)
        self.result['valid_loss'] = loss.item()
        print("------------- valid result --------------")
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                if metric.name() == 'class_report' or metric.name() == 'multilabel_report':
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

    def train_epoch(self, data):
        pbar = ProgressBar(n_total=len(data), desc='Training')
        tr_loss = AverageMeter()
        self.epoch_reset()
        for step, batch in enumerate(data):
            self.batch_reset()
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # 使用命名参数调用模型
            logits = self.model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask
            ).to(self.device)
            # smooth_target = self.label_smoothing_crossentropy(logits, label_ids, epsilon=0.1, num_classes=4)
            loss = self.criterion(output=logits, target=label_ids)
            if len(self.args.n_gpu) >= 2:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip)
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            # 更新权重参数
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    if metric.name() == 'class_report' or metric.name() == 'multilabel_report':
                        metric(logits=logits, target=label_ids)
                        self.info[f'{metric.name()}'] = metric.value()
                    else:
                        value = metric(logits=logits, target=label_ids)
                        self.info[metric.name()] = value

            self.info['loss'] = loss.item()
            tr_loss.update(loss.item(), n=1)
            if self.verbose >= 1:
                pbar(step=step, info=self.info)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        print("\n------------- train result --------------")
        # epoch metric
        # cat将list沿着第一个维度拼接在一起
        self.outputs = torch.cat(self.outputs, dim=0).detach().cpu()
        self.targets = torch.cat(self.targets, dim=0).detach().cpu()
        self.result['loss'] = tr_loss.avg
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                if metric.name() == 'class_report' or metric.name() == 'multilabel_report':
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'{metric.name()}'] = value
                else:
                    value = metric(logits=self.outputs, target=self.targets)
                    if value:
                        self.result[f'{metric.name()}'] = value

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, valid_data):
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.train_epoch(train_data)
            valid_log = self.valid_epoch(valid_data)
            '''
            使用 ** 操作符将valid_metrics解包传递给dict()时，
            valid_log中的键值对会覆盖了train_log中的同名项，有多的项会加入字典
            例如：
            train_metrics = {'accuracy': 0.90, 'precision': 0.85}
            valid_metrics = {'accuracy': 0.88, 'recall': 0.78}
            {'accuracy': 0.88, 'precision': 0.85, 'recall': 0.78}
            '''
            logs = dict(train_log, **valid_log)

            # 过滤logs，移除字典类型的值
            filtered_logs = {k: v for k, v in logs.items() if not isinstance(v, dict)}

            # 修改格式化方式，支持不同类型的值
            log_items = []
            for key, value in logs.items():
                # 跳过字典类型的指标值，这些通常由multilabel_report和class_report生成
                if isinstance(value, dict):
                    continue
                # 对于数值类型使用保留4位小数的格式
                elif isinstance(value, (int, float)):
                    log_items.append(f' {key}: {value:.4f} ')
                # 对于其他类型直接转为字符串
                else:
                    log_items.append(f' {key}: {value} ')

            show_info = f'\nEpoch: {epoch} - ' + "-".join(log_items)
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(filtered_logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break 