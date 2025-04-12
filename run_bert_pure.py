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
from pybert.model.bert_pure_for_multi_label import BertPureForMultiLabel
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
    Execute the training process for Pure BERT model
    
    Args:
        args: Command line arguments
    """
    # --------- Data Processing ---------
    logger.info("Processing training and validation data...")
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # Load training data
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
    
    # Create train data loader
    train_sampler = SequentialSampler(train_dataset) if args.sorted else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=args.train_batch_size,
        collate_fn=collate_fn
    )

    # Load validation data
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
    
    # Create validation data loader
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset, 
        sampler=valid_sampler, 
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn
    )
    logger.info("Data processing completed")

    # 分析标签分布情况 - 只打印分布情况，不计算权重
    train_labels = torch.tensor([example.label for example in train_examples])
    valid_labels = torch.tensor([example.label for example in valid_examples])
    
    logger.info("\n***** Label distribution analysis *****")
    for i, label in enumerate(label_list):
        train_pos = (train_labels[:, i] == 1).sum().item()
        train_neg = len(train_labels) - train_pos
        train_pos_ratio = train_pos / len(train_labels) if train_pos > 0 else 0
        
        valid_pos = (valid_labels[:, i] == 1).sum().item()
        valid_pos_ratio = valid_pos / len(valid_labels) if valid_pos > 0 else 0
        
        logger.info(f"Label {label}: Train {train_pos}/{len(train_labels)} ({train_pos_ratio:.2%}), " 
                   f"Valid {valid_pos}/{len(valid_labels)} ({valid_pos_ratio:.2%})")

    # --------- Model Initialization ---------
    logger.info("Initializing model...")
    if args.resume_path is not None:
        logger.info(f"Loading model from {args.resume_path}")
        model, _ = BertPureForMultiLabel.from_pretrained(
            config['bert_model_dir'],
            num_labels=len(label_list),
            state_dict=torch.load(args.resume_path / 'best_model.bin', map_location=torch.device('cpu')),
            threshold=0.5  # 使用标准阈值0.5
        )
    else:
        logger.info(f"Loading pretrained model from: {config['bert_model_dir']}")
        model, _ = BertPureForMultiLabel.from_pretrained(
            config['bert_model_dir'],
            num_labels=len(label_list),
            threshold=0.5  # 使用标准阈值0.5
        )

    # --------- Optimizer and Scheduler Setup ---------
    # Calculate total steps for the entire training process
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    
    # Set up parameter groups for optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    # Setup warmup steps
    warmup_steps = int(t_total * args.warmup_proportion)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    
    # Setup mixed precision training if enabled
    if args.fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            logger.info(f"FP16 training enabled with opt_level: {args.fp16_opt_level}")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # --------- Callback Setup ---------
    logger.info("Initializing callbacks...")
    
    # Training monitor for visualization
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    
    # Model checkpoint for saving best model
    checkpoint_dir = config['checkpoint_dir']
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 添加早停机制
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        mode=args.mode,
        patience=3,  # 设置耐心值为3
        min_delta=1e-3,
        verbose=1
    )
    
    # 保存检查点
    model_checkpoint = GloveModelCheckpoint(
        checkpoint_dir=checkpoint_dir, 
        mode=args.mode,
        monitor=args.monitor, 
        arch=args.arch,
        save_best_only=True,
        model_name="pytorch_model.bin"
    )

    # --------- Training Process ---------
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
    logger.info(f"Using threshold: 0.5 for evaluation metrics")

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
            AccuracyThresh(thresh=0.5, search_thresh=False),
        ],
        epoch_metrics=[
            AccuracyThresh(thresh=0.5, search_thresh=False),
            Precision(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
            Recall(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
            F1Score(thresh=0.5, normalizate=True, task_type='binary', average='micro', search_thresh=False),
            HammingScore(thresh=0.5, search_thresh=False),
            HammingLoss(thresh=0.5, search_thresh=False),
            AUC(task_type='binary'),
            MultiLabelReport(id2label=id2label, average='micro', logger=logger, thresh=0.5)
        ],
        verbose=args.verbose
    )
    
    # Train the model
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)
    
    # 不需要额外保存模型，模型已经通过model_checkpoint保存了
    logger.info(f"Training completed. Best model saved at: {checkpoint_dir / args.arch / 'pytorch_model.bin'}")
    
    # 确保config.json文件存在
    config_path = checkpoint_dir / args.arch / 'config.json'
    if not config_path.exists():
        logger.info(f"Saving model config to {config_path}")
        bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
        bert_config.num_labels = len(label_list)
        bert_config.save_pretrained(checkpoint_dir / args.arch)
    
    logger.info("Training completed")


def run_test(args):
    """
    Execute the testing process for Pure BERT model
    
    Args:
        args: Command line arguments
    """
    try:
        # --------- Data Processing ---------
        logger.info("Processing test data...")
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
        label_list = processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}
        
        # Load test data
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
        
        # Create test data loader
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, 
            sampler=test_sampler, 
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn
        )
        logger.info("Test data processing completed")

        try:
            # --------- Model Loading ---------
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
                from pybert.model.bert_pure_for_multi_label import BertPureForMultiLabel
                
                # 检查是否存在config.json，若不存在则创建
                config_path = model_dir / "config.json"
                if not config_path.exists():
                    logger.info(f"Config file not found, creating from {config['bert_model_dir']}")
                    bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                    bert_config.num_labels = len(label_list)
                    bert_config.save_pretrained(model_dir)
                
                # 使用transformers的标准加载方式
                logger.info(f"Loading model from checkpoint directory: {model_dir}")
                model_loaded = BertPureForMultiLabel.from_pretrained(
                    model_dir,
                    num_labels=len(label_list)
                )
                # 处理可能返回的元组
                if isinstance(model_loaded, tuple):
                    model = model_loaded[0]  # 第一个元素是模型
                else:
                    model = model_loaded
            except Exception as e:
                logger.warning(f"Error loading with transformers method: {e}")
                logger.info("Falling back to manual loading")
                
                # 如果标准方法失败，回退到手动加载
                bert_config = BertConfig.from_pretrained(config['bert_model_dir'])
                bert_config.num_labels = len(label_list)
                model = BertPureForMultiLabel(bert_config)
                
                if model_path.exists():
                    logger.info(f"Loading weights from {model_path}")
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    logger.error(f"Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model.to(device)
            logger.info("Model loaded successfully")

            # --------- Testing Process ---------
            logger.info("***** Running testing *****")
            logger.info(f"Num examples: {len(test_examples)}")
            logger.info(f"Batch size: {args.eval_batch_size}")
            
            # Create test log directory
            test_log_dir = config['log_dir'] / 'test_result'
            test_log_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Test results will be saved to: {test_log_dir}")
            
            # 尝试加载预先计算的最佳阈值
            optimal_thresholds_path = model_dir / 'optimal_thresholds.json'
            if optimal_thresholds_path.exists():
                logger.info(f"Loading optimal thresholds from {optimal_thresholds_path}")
                with open(optimal_thresholds_path, 'r') as f:
                    saved_thresholds = json.load(f)
                logger.info(f"Loaded thresholds: {saved_thresholds}")
            else:
                logger.info("No pre-computed optimal thresholds found.")
                saved_thresholds = None
            
            # 分析测试集标签分布
            test_labels = [example.label for example in test_examples]
            test_labels = torch.tensor(test_labels)
            logger.info("\n***** Test data label distribution *****")
            for i, label in enumerate(label_list):
                pos_count = (test_labels[:, i] == 1).sum().item()
                percentage = (pos_count / len(test_labels)) * 100
                logger.info(f"Label {label}: {pos_count}/{len(test_labels)} ({percentage:.2f}%)")
            
            # 使用固定阈值0.5进行测试
            logger.info(f"\n***** Testing with threshold 0.5 *****")
            # 使用统一的模型名称
            model_name_for_log = 'model'
            
            # Configure predictor with metrics
            predictor = Predictor(
                model=model,
                logger=logger,
                n_gpu=args.n_gpu,
                test_metrics=[
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
            
            # Run prediction
            result = predictor.predict(data=test_dataloader)
            logger.info(f"Model test results (threshold = 0.5):")
            print(result)
            
            logger.info("Testing completed")
            
        except Exception as e:
            logger.error(f"Error during model testing: {e}")
            raise e
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise e


def main():
    parser = ArgumentParser()
    # Mode options
    parser.add_argument("--arch", default='mnor_bert', type=str, help="Model architecture name")
    parser.add_argument("--do_data", action='store_true', help="Whether to process raw data")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing")
    parser.add_argument("--save_best", action='store_true', help="Whether to save best model only")
    parser.add_argument("--do_lower_case", action='store_true', help="Whether to lower case the input text")
    
    # Model loading options
    parser.add_argument("--resume_path", default=None, type=str, help="Path to resume training from checkpoint")
    parser.add_argument("--model_name", default=None, type=str, help="Model file name in checkpoint directory")
    
    # Data options
    parser.add_argument("--data_name", default='bert_pure', type=str, help="Dataset name prefix")
    parser.add_argument("--train_max_seq_len", default=128, type=int, help="Maximum sequence length for training")
    parser.add_argument("--eval_max_seq_len", default=128, type=int, help="Maximum sequence length for evaluation")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation")
    parser.add_argument("--sorted", default=0, type=int, help="Sort the data by sequence length")
    
    # Optimization options
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--epochs", default=6, type=int, help="Number of epochs")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    
    # FP16 options
    parser.add_argument('--fp16', action='store_true', help="Whether to use fp16 mixed precision training")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="Mixed precision opt level")
    
    # Callback options
    parser.add_argument("--mode", default='min', type=str, help="Optimization mode (min or max)")
    parser.add_argument("--monitor", default='valid_loss', type=str, help="Metric to monitor for model checkpointing")
    parser.add_argument("--early_stopping", default=15, type=int, help="Early stopping patience")
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity mode")
    
    # Device options
    parser.add_argument("--n_gpu", default='0', type=str, help="GPU device ids to use, e.g., '0,1,2'")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up paths and configs
    init_logger(log_file=config['log_dir'] / f'{args.arch}.log')
    seed_everything(args.seed)
    logger.info("***** Arguments *****")
    logger.info(f"  Args: {args}")
    
    # Initialize logger
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
    
    # Run training
    if args.do_train:
        run_train(args)
    
    # Run testing
    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main() 