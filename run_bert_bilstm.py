import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss, BCEWithLoss, MultiLabelCrossEntropy
from pybert.train.bert_trainer import BertTrainer
from pybert.test.predictor import Predictor
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger, save_pickle, load_pickle
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_bilstm_for_multi_label import BertBiLstmForMultiLabel
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.glove_modelcheckpoint import GloveModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, Precision, Recall, HammingScore, HammingLoss, \
    F1Score, ClassReport, Jaccard, Accuracy
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
from torchsummary import summary
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertConfig
import numpy as np
import os
import json
import logging
import shutil
from pybert.callback.early_stopping import EarlyStopping

warnings.filterwarnings("ignore")

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def run_train(args):
    """
    执行BERT+BiLSTM多标签分类模型的训练过程
    
    Args:
        args: 命令行参数
    """
    # --------- 数据处理 ---------
    logger.info("Processing training and validation data...")
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 加载训练数据
    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    train_examples = processor.create_examples(
        lines=train_data,
        example_type='train',
        cached_examples_file=config['data_dir'] / f"cached_train_examples_{args.arch}"
    )
    train_features = processor.create_features(
        examples=train_examples,
        max_seq_len=args.train_max_seq_len,
        cached_features_file=config['data_dir'] / f"cached_train_features_{args.train_max_seq_len}_{args.arch}"
    )
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    
    # 创建训练数据加载器
    train_sampler = SequentialSampler(train_dataset) if args.sorted else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size,
        collate_fn=collate_fn
    )

    # 加载验证数据
    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(
        lines=valid_data,
        example_type='valid',
        cached_examples_file=config['data_dir'] / f"cached_valid_examples_{args.arch}"
    )
    valid_features = processor.create_features(
        examples=valid_examples,
        max_seq_len=args.eval_max_seq_len,
        cached_features_file=config['data_dir'] / f"cached_valid_features_{args.eval_max_seq_len}_{args.arch}"
    )
    valid_dataset = processor.create_dataset(valid_features)
    
    # 创建验证数据加载器
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset, 
        sampler=valid_sampler, 
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn
    )
    logger.info("Data processing completed")

    # --------- 模型初始化 ---------
    logger.info("Initializing model...")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        logger.info(f"Loading model from resume path: {args.resume_path}")
        
        # 如果resume_path是目录，从该目录加载模型
        if args.resume_path.is_dir():
            model_dir = args.resume_path
            model_path = model_dir / 'pytorch_model.bin'
            logger.info(f"Looking for model file at: {model_path}")
            
            if model_path.exists():
                try:
                    model = BertBiLstmForMultiLabel.from_pretrained(
                        model_dir, 
                        num_labels=len(label_list)
                    )
                    logger.info("Model loaded successfully from directory")
                except Exception as e:
                    logger.warning(f"Failed to load model with from_pretrained: {e}")
                    bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                    bert_config.num_labels = len(label_list)
                    model = BertBiLstmForMultiLabel(bert_config)
                    
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    logger.info("Model loaded successfully with manual loading")
            else:
                logger.warning(f"Model file not found at {model_path}, initializing new model")
                model = BertBiLstmForMultiLabel.from_pretrained(
                    config['bert_model_dir'], 
                    num_labels=len(label_list)
                )
        else:
            # 如果resume_path是文件，使用该文件作为model weights
            logger.info(f"Resume path is a file: {args.resume_path}")
            
            if args.resume_path.exists():
                bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                bert_config.num_labels = len(label_list)
                model = BertBiLstmForMultiLabel(bert_config)
                
                state_dict = torch.load(args.resume_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("Model loaded successfully from file")
            else:
                logger.warning(f"Model file not found at {args.resume_path}, initializing new model")
                model = BertBiLstmForMultiLabel.from_pretrained(
                    config['bert_model_dir'], 
                    num_labels=len(label_list)
                )
    else:
        # 从预训练BERT模型初始化
        logger.info(f"Initializing new model from: {config['bert_model_dir']}")
        model = BertBiLstmForMultiLabel.from_pretrained(
            config['bert_model_dir'], 
            num_labels=len(label_list)
        )

    # --------- 优化器和调度器设置 ---------
    # 计算整个训练过程的总步数
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    
    # 设置优化器的参数组
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    # 设置预热步数
    warmup_steps = int(t_total * args.warmup_proportion)
    
    # 初始化优化器和调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    
    # 如果启用了混合精度训练
    if args.fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            logger.info(f"FP16 training enabled with opt_level: {args.fp16_opt_level}")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # --------- 回调设置 ---------
    logger.info("Initializing callbacks...")
    
    # 训练监控器用于可视化
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    
    # 添加早停机制
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        mode=args.mode,
        patience=3,  # 设置耐心值为3
        min_delta=1e-3,
        verbose=1
    )
    
    # 保存检查点
    checkpoint_dir = config['checkpoint_dir']
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存检查点
    model_checkpoint = GloveModelCheckpoint(
        checkpoint_dir=checkpoint_dir, 
        mode=args.mode,
        monitor=args.monitor, 
        arch=args.arch,
        save_best_only=True,
        model_name="pytorch_model.bin"  # 统一使用pytorch_model.bin作为模型文件名
    )

    # --------- 训练过程 ---------
    logger.info("***** Running training *****")
    logger.info(f"Num examples: {len(train_examples)}")
    logger.info(f"Num Epochs: {args.epochs}")
    batch_size_info = args.train_batch_size * args.gradient_accumulation_steps
    if args.local_rank != -1:
        batch_size_info *= torch.distributed.get_world_size()
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation): {batch_size_info}")
    logger.info(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {t_total}")
    logger.info(f"Early stopping patience: 3")

    # Create trainer
    trainer = BertTrainer(
        args=args, 
        model=model, 
        logger=logger, 
        criterion=BCEWithLogLoss(), 
        optimizer=optimizer,
        scheduler=scheduler, 
        early_stopping=early_stopping,
        training_monitor=train_monitor,
        model_checkpoint=model_checkpoint,
        batch_metrics=[
            # Fixed threshold metrics
            AccuracyThresh(thresh=0.5, search_thresh=False),
        ],
        epoch_metrics=[
            # Fixed threshold metrics for evaluation
            AccuracyThresh(thresh=0.5, search_thresh=False),
            Precision(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
            Recall(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
            F1Score(thresh=0.5, normalizate=True, task_type='binary', average='micro', search_thresh=False),
            HammingScore(thresh=0.5, search_thresh=False),
            HammingLoss(thresh=0.5, search_thresh=False),
            AUC(task_type='binary'),
            MultiLabelReport(id2label=id2label, average='micro', logger=logger)
        ],
        verbose=args.verbose
    )
    
    # 训练模型
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)
    
    # Save final model state
    final_model_path = config['checkpoint_dir'] / args.arch / 'best_model.bin'  # 使用相同的文件名
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save model config
    config_path = config['checkpoint_dir'] / args.arch / 'config.json'
    if not config_path.exists():
        bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
        bert_config.num_labels = len(label_list)
        bert_config.save_pretrained(config['checkpoint_dir'] / args.arch)
    
    logger.info("Training completed")


def run_test(args):
    """
    执行BERT+BiLSTM多标签分类模型的测试过程
    
    Args:
        args: 命令行参数
    """
    try:
        # --------- 数据处理 ---------
        logger.info("Processing test data...")
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
        label_list = processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}
        
        # 加载测试数据
        test_data = processor.get_train(config['data_dir'] / f"{args.data_name}.test.pkl")
        test_examples = processor.create_examples(
            lines=test_data,
            example_type='test',
            cached_examples_file=config['data_dir'] / f"cached_test_examples_{args.arch}"
        )
        test_features = processor.create_features(
            examples=test_examples,
            max_seq_len=args.eval_max_seq_len,
            cached_features_file=config['data_dir'] / f"cached_test_features_{args.eval_max_seq_len}_{args.arch}"
        )
        test_dataset = processor.create_dataset(test_features)
        
        # 创建测试数据加载器
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, 
            sampler=test_sampler, 
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn
        )
        logger.info("Test data processing completed")

        try:
            # --------- 模型加载 ---------
            logger.info("Loading model...")
            
            # 确定检查点目录和模型文件 - 使用与GloveModelCheckpoint相同的路径构建逻辑
            checkpoint_dir = args.resume_path if args.resume_path else config['checkpoint_dir']  
            checkpoint_dir = Path(checkpoint_dir)
            
            # 确保使用与GloveModelCheckpoint相同的路径结构
            model_dir = checkpoint_dir / args.arch
            model_path = model_dir / 'pytorch_model.bin'  # 统一使用pytorch_model.bin作为模型文件名
            
            logger.info(f"Model directory: {model_dir}")
            logger.info(f"Model file: {model_path}")
            logger.info(f"Model file exists: {model_path.exists()}")
            
            try:
                # 首先尝试使用transformers标准方法加载
                from transformers import BertConfig
                from pybert.model.bert_bilstm_for_multi_label import BertBiLstmForMultiLabel
                
                # 检查是否存在config.json，若不存在则创建
                config_path = model_dir / "config.json"
                if not config_path.exists():
                    logger.info(f"Config file not found, creating from {config['bert_model_dir']}")
                    bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                    bert_config.num_labels = len(label_list)
                    bert_config.save_pretrained(model_dir)
                
                # 使用transformers的标准加载方式
                logger.info(f"Loading model from checkpoint directory: {model_dir}")
                model = BertBiLstmForMultiLabel.from_pretrained(
                    model_dir,
                    num_labels=len(label_list)
                )
            except Exception as e:
                logger.warning(f"Error loading with transformers method: {e}")
                logger.info("Falling back to manual loading")
                
                # 如果transformers方法失败，回退到手动加载
                bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                bert_config.num_labels = len(label_list)
                model = BertBiLstmForMultiLabel(bert_config)
                
                if model_path.exists():
                    logger.info(f"Loading weights from {model_path}")
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    logger.error(f"Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model.to(device)
            logger.info("Model loaded successfully")

            # --------- 测试过程 ---------
            logger.info("***** Running testing *****")
            logger.info(f"Num examples: {len(test_examples)}")
            logger.info(f"Batch size: {args.eval_batch_size}")
            
            # 创建测试日志目录
            test_log_dir = config['log_dir'] / 'test_result'
            test_log_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Test results will be saved to: {test_log_dir}")
            
            # 确定用于日志记录的模型名称
            model_name_for_log = 'model'
                
            # 创建用于测试的预测器
            predictor = Predictor(
                model=model,
                logger=logger,
                n_gpu=args.n_gpu,
                test_metrics=[
                    # 评估的固定阈值指标
                    AccuracyThresh(thresh=0.5, search_thresh=False),
                    Precision(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
                    Recall(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
                    F1Score(thresh=0.5, normalizate=True, task_type='binary', average='micro', search_thresh=False),
                    HammingScore(thresh=0.5, search_thresh=False),
                    HammingLoss(thresh=0.5, search_thresh=False),
                    Jaccard(average='micro', thresh=0.5, search_thresh=False),
                    AUC(task_type='binary'),
                    MultiLabelReport(id2label=id2label, average='micro', logger=logger, thresh=0.5)
                ],
                log_dir=test_log_dir,
                model_name=f"{model_name_for_log}_thresh0.5"
            )
            
            # 执行预测
            logger.info("Starting prediction...")
            result = predictor.predict(data=test_dataloader)
            logger.info("Model test results:")
            print(result)
            
        except Exception as e:
            logger.error(f"Error occurred during model testing: {str(e)}")
            raise e
    except Exception as e:
        logger.error(f"Error occurred during data processing: {str(e)}")
        raise e


def main():
    parser = ArgumentParser()
    # 模式选项
    parser.add_argument("--arch", default='bert_bilstm', type=str, help="模型架构名称")
    parser.add_argument("--do_data", action='store_true', help="是否处理原始数据")
    parser.add_argument("--do_train", action='store_true', help="是否运行训练")
    parser.add_argument("--do_test", action='store_true', help="是否运行测试")
    parser.add_argument("--save_best", action='store_true', help="是否只保存最佳模型")
    parser.add_argument("--do_lower_case", action='store_true', help="是否将输入文本转为小写")
    
    # 模型加载选项
    parser.add_argument("--resume_path", default=None, type=str, help="从检查点恢复训练的路径")
    parser.add_argument("--model_name", default=None, type=str, help="检查点目录中的模型文件名")
    
    # 数据选项
    parser.add_argument("--data_name", default='bert-bilstm', type=str, help="数据集名称前缀")
    parser.add_argument("--train_max_seq_len", default=64, type=int, help="训练的最大序列长度")
    parser.add_argument("--eval_max_seq_len", default=64, type=int, help="评估的最大序列长度")
    parser.add_argument("--train_batch_size", default=128, type=int, help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="评估批次大小")
    parser.add_argument("--sorted", default=0, type=int, help="按序列长度排序数据")
    
    # 优化选项
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积步数")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="学习率")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="预热比例")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam优化器的epsilon")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="梯度裁剪的最大梯度范数")
    parser.add_argument("--epochs", default=6, type=int, help="训练轮数")
    parser.add_argument("--local_rank", type=int, default=-1, help="用于GPU分布式训练的本地rank")
    parser.add_argument('--seed', type=int, default=42, help="初始化的随机种子")
    
    # FP16选项
    parser.add_argument('--fp16', action='store_true', help="是否使用fp16混合精度训练")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="混合精度优化级别")
    
    # 回调选项
    parser.add_argument("--mode", default='min', type=str, help="优化模式（min或max）")
    parser.add_argument("--monitor", default='valid_loss', type=str, help="用于模型检查点的监控指标")
    parser.add_argument("--early_stopping", default=15, type=int, help="提前停止的耐心值")
    parser.add_argument("--verbose", default=1, type=int, help="详细模式")
    
    # 设备选项
    parser.add_argument("--n_gpu", default='0', type=str, help="要使用的GPU设备ID，例如'0,1,2'")
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置路径和配置
    init_logger(log_file=config['log_dir'] / f'{args.arch}.log')
    seed_everything(args.seed)
    logger.info("***** Arguments *****")
    logger.info(f"  Args: {args}")
    
    # 初始化日志记录器
    if args.do_data:
        from pybert.preprocessing.preprocessor import EnglishPreProcessor
        from pybert.io.task_data import TaskData
        
        # 创建预处理器
        preprocessor = EnglishPreProcessor(min_len=2)
        
        # 创建TaskData实例
        data = TaskData(
                      task_data_dir=config['data_dir'],
                      tokenizer=BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case),
                      data_name=args.data_name,
                      preprocessor=preprocessor)
        
        # 首先读取原始数据
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'], 
                                           preprocessor=preprocessor,
                                           is_train=True)
        
        # 然后进行训练集和验证集的分割
        data.train_val_split(X=sentences, y=targets, 
                            shuffle=True, 
                            stratify=False,
                            train_size=0.8,  # 使用80%的数据作为训练集
                            data_dir=config['data_dir'],
                            data_name=args.data_name)
    
    # 运行训练
    if args.do_train:
        run_train(args)
    
    # 运行测试
    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main() 