import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.glove_bilstm_trainer import GloveBiLstmTrainer
from pybert.test.glove_bilstm_predictor import GloveBiLstmPredictor
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.glove_processor import GloveProcessor
from pybert.common.tools import init_logger, logger, load_pickle
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.glove_bilstm_for_multi_label import GloveBiLSTMForMultiLabel
from torch.utils.data import RandomSampler, SequentialSampler
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, Precision, Recall, HammingScore, HammingLoss, \
    F1Score, Jaccard
from pybert.callback.glove_modelcheckpoint import GloveModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.callback.early_stopping import EarlyStopping
import shutil
import json
import os

warnings.filterwarnings("ignore")

# Set device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def run_train(args):
    """
    Execute the training process for GloVe-BiLSTM model
    
    Args:
        args: Command line arguments
    """
    # --------- Data Processing ---------
    logger.info("Processing training and validation data...")
    processor = GloveProcessor(glove_path=config['glove_path'])  # 直接传入glove_path
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # Load training data
    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    train_examples = processor.create_examples(
        lines=train_data,
        example_type='train',
        cached_examples_file=config['data_dir'] / f"cached_train_examples_glove_bilstm"
    )
    train_features = processor.create_features(
        examples=train_examples,
        max_seq_len=args.train_max_seq_len,
        cached_features_file=config['data_dir'] / f"cached_train_features_{args.train_max_seq_len}_glove_bilstm"
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
        cached_examples_file=config['data_dir'] / f"cached_valid_examples_glove_bilstm"
    )
    valid_features = processor.create_features(
        examples=valid_examples,
        max_seq_len=args.eval_max_seq_len,
        cached_features_file=config['data_dir'] / f"cached_valid_features_{args.eval_max_seq_len}_glove_bilstm"
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

    # --------- Model Initialization ---------
    logger.info("Initializing GloVe-BiLSTM model...")
    
    # Create a new GloVe-BiLSTM model
    model = GloveBiLSTMForMultiLabel(
        num_labels=len(label_list),
        vocab_size=processor.vocab_size,
        embedding_dim=config['embedding_dim']
    )
    
    # Load GloVe embeddings
    logger.info("Loading GloVe embeddings...")
    model.load_glove_embeddings(
        glove_path=str(config['glove_path']),
        word_to_idx=processor.vocab
    )
    logger.info("Model initialization completed")

    # --------- Optimizer and Scheduler Setup ---------
    # Calculate total steps for the entire training process
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_total,
        eta_min=args.learning_rate / 100
    )
    
    # --------- Loss Function Setup ---------
    logger.info("Setting up loss function...")
    loss_fn = BCEWithLogLoss()
    
    # --------- Callback Setup ---------
    logger.info("Setting up callbacks...")
    
    # Model checkpoint directory
    checkpoint_dir = config['checkpoint_dir'] / args.arch
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 添加早停机制
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        mode=args.mode,
        patience=3,  # 设置耐心值为3
        min_delta=1e-3,
        verbose=1
    )
    
    # Model checkpoint callback - 使用与其他模型统一的文件名和路径结构
    model_checkpoint = GloveModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        mode=args.mode,
        monitor=args.monitor,
        arch=args.arch,
        save_best_only=True,
        model_name='pytorch_model.bin'  # 统一使用pytorch_model.bin作为模型文件名
    )
    
    # Training monitor callback
    train_monitor = TrainingMonitor(
        file_dir=config['figure_dir'],
        arch=args.arch
    )
    
    # --------- Metrics Setup ---------
    logger.info("Setting up metrics...")
    metrics = [
        AccuracyThresh(thresh=0.5),
        Precision(thresh=0.5, average='micro'),
        Recall(thresh=0.5, average='micro'),
        F1Score(thresh=0.5, average='micro'),
        AUC(task_type='binary', average='micro', search_thresh=False),
        MultiLabelReport(id2label=id2label, average='micro', logger=logger)
    ]
    
    # --------- Trainer Setup ---------
    logger.info("Setting up trainer...")
    logger.info(f"Early stopping patience: 3")
    trainer = GloveBiLstmTrainer(
        model=model,
        logger=logger,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        batch_metrics=[AccuracyThresh(thresh=0.5)],
        epoch_metrics=metrics,
        verbose=args.verbose,
        training_monitor=train_monitor,
        model_checkpoint=model_checkpoint,
        early_stopping=early_stopping,  # 添加早停
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_gpu=args.n_gpu,
        fp16=args.fp16
    )
    
    # --------- Start Training ---------
    logger.info("Starting training...")
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)
    
    # Save final model state
    final_model_path = checkpoint_dir / 'pytorch_model.bin'  # 使用相同的文件名
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save model config
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        model_config = {
            'vocab_size': processor.vocab_size,
            'embedding_dim': config['embedding_dim'],
            'num_labels': len(label_list),
            'hidden_size': 512,  # LSTM隐藏层大小
            'num_layers': 1,     # LSTM层数
            'dropout': 0.1       # dropout率
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f)
    
    logger.info("Training completed")

def run_test(args):
    """
    执行GloVe+BiLSTM多标签分类模型的测试过程
    
    Args:
        args: 命令行参数
    """
    try:
        # --------- 数据处理 ---------
        logger.info("Processing test data...")
        processor = GloveProcessor(glove_path=config['glove_path'])  # 直接传入glove_path
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
            
            # 确定检查点目录和模型文件
            checkpoint_dir = args.resume_path if args.resume_path else config['checkpoint_dir']
            checkpoint_dir = Path(checkpoint_dir)
            model_dir = checkpoint_dir / args.arch
            model_path = model_dir / 'pytorch_model.bin'
            
            logger.info(f"Model directory: {model_dir}")
            logger.info(f"Model file: {model_path}")
            
            # 加载模型
            model = GloveBiLSTMForMultiLabel(
                num_labels=len(label_list),
                vocab_size=processor.vocab_size,
                embedding_dim=config['embedding_dim']
            )
            
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
            
            # 创建用于测试的预测器
            predictor = GloveBiLstmPredictor(
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
                model_name=f"glove_bilstm_thresh0.5"
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
    
    # Mode options
    parser.add_argument("--mode", default='min', type=str, help="Optimization mode (min or max)")
    parser.add_argument("--monitor", default='valid_loss', type=str, help="Metric to monitor for model checkpointing")
    
    # File and location options
    parser.add_argument("--data_name", default='glove_bilstm', type=str, help="Dataset name prefix")
    parser.add_argument("--arch", default='glove_bilstm', type=str, help="Model architecture name")
    parser.add_argument("--do_data", action='store_true', help="Whether to process data")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing")
    parser.add_argument("--save_best", action='store_true', help="Whether to save best model only")
    
    # Model options
    parser.add_argument("--resume_path", default=None, type=str, help="Path to resume training from checkpoint")
    parser.add_argument("--model_name", default=None, type=str, help="Model file name in checkpoint directory")
    
    # Data processing options
    parser.add_argument("--do_lower_case", action='store_true', help="Whether to lowercase the text")
    parser.add_argument("--train_max_seq_len", default=128, type=int, help="Maximum sequence length for training")
    parser.add_argument("--eval_max_seq_len", default=128, type=int, help="Maximum sequence length for evaluation")
    parser.add_argument("--sorted", default=0, type=int, help="Whether to sort data by length (1 or 0)")
    
    # Training options
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay rate")
    parser.add_argument("--evaluate_every", default=5, type=int, help="Evaluate every N steps")
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity level (1 or 0)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--n_gpu", default='0', type=str, help="GPU indices to use (comma-separated string)")
    parser.add_argument("--fp16", action='store_true', help="Whether to use 16-bit (mixed) precision")
    
    args = parser.parse_args()

    # Initialize logger
    init_logger(log_file=config['log_dir'] / f"{args.arch}.log")
    
    # 确保在没有可用GPU的情况下禁用fp16
    if not torch.cuda.is_available():
        args.fp16 = False
        args.n_gpu = ''
        logger.warning("CUDA不可用，将使用CPU进行训练")
    
    # Set random seed
    seed_everything(args.seed)
    
    # Print arguments
    logger.info(f"Arguments: {args}")
    
    # 添加数据处理逻辑
    if args.do_data:
        from pybert.preprocessing.preprocessor import EnglishPreProcessor
        from pybert.io.task_data import TaskData
        
        # 创建预处理器
        preprocessor = EnglishPreProcessor(min_len=2)
        
        # 创建TaskData实例
        data = TaskData(
            task_data_dir=config['data_dir'],
            tokenizer=GloveProcessor(vocab_path=config['glove_vocab_path']),
            data_name=args.data_name,
            preprocessor=preprocessor
        )
        
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
    
    # Execute tasks based on arguments
    if args.do_train:
        run_train(args)
    if args.do_test:
        run_test(args)

if __name__ == '__main__':
    main() 