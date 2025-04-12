import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss, BCEWithLoss, MultiLabelCrossEntropy
from pybert.train.trainer import Trainer
from pybert.test.predictor import Predictor
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger, save_pickle, load_pickle
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, Precision, Recall, HammingScore, HammingLoss, \
    F1Score, ClassReport, Jaccard, Accuracy
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
from torchsummary import summary
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def run_train(args):
    """
    Execute the training process for BERT model
    
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

    # --------- Model Initialization ---------
    logger.info("Initializing model...")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        # Load model from checkpoint if resume_path is provided
        if args.resume_path.is_dir():
            logger.info(f"Loading model from directory: {args.resume_path} with model file: {args.model_name}")
            model = BertForMultiLable.from_pretrained(
                args.resume_path, 
                num_labels=len(label_list),
                model_name=args.model_name
            )
        else:
            logger.info(f"Loading model from specific file: {args.resume_path}")
            model = BertForMultiLable.from_pretrained(
                args.resume_path.parent, 
                num_labels=len(label_list),
                model_name=args.resume_path.name
            )
    else:
        # Load pretrained BERT model
        logger.info(f"Loading pretrained model from: {config['bert_model_dir']}")
        model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))

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
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir=config['checkpoint_dir'], 
        mode=args.mode,
        monitor=args.monitor, 
        arch=args.arch,
        save_best_only=args.save_best,
        model_name=args.model_name
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

    # Create trainer
    trainer = Trainer(
        args=args, 
        model=model, 
        logger=logger, 
        criterion=BCEWithLogLoss(), 
        optimizer=optimizer,
        scheduler=scheduler, 
        early_stopping=None, 
        training_monitor=train_monitor,
        model_checkpoint=model_checkpoint,
        batch_metrics=[
            # Fixed threshold metrics
            AccuracyThresh(thresh=0.5, search_thresh=False),
        ],
        epoch_metrics=[
            # Fixed threshold metrics for evaluation
            AccuracyThresh(thresh=0.5, search_thresh=False),
            AUC(task_type='binary', average='macro', search_thresh=False),
            Precision(task_type='binary', average='macro', thresh=0.5, search_thresh=False),
            Recall(task_type='binary', average='macro', thresh=0.5, search_thresh=False),
            HammingScore(thresh=0.5, search_thresh=False),
            HammingLoss(thresh=0.5, search_thresh=False),
            F1Score(thresh=0.5, normalizate=True, task_type='binary', average='macro', search_thresh=False),
            MultiLabelReport(id2label=id2label, average='macro', logger=logger),
            Jaccard(average='macro', thresh=0.5, search_thresh=False)
        ]
    )
    
    # Start training
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)
    logger.info("Training completed successfully")



# %%
def run_test(args):
    """
    Testing process for BERT model
    """
    from pybert.io.bert_processor import BertProcessor
    from pybert.test.predictor import Predictor
    from pybert.configs.basic_config import config
    from pybert.preprocessing.preprocessor import EnglishPreProcessor
    from pybert.callback.progressbar import ProgressBar
    
    logger.info("Starting test process...")
    
    # 0. Output path and device setup
    from pybert.configs.basic_config import config as base_config
    
    # 重置checkpoint_dir，确保只包含一层arch
    if 'arch' not in config:
        config['arch'] = args.arch
    
    if config['checkpoint_dir'].name == args.arch:
        # 检查点路径已经包含了arch作为最后一个目录名，不需要再添加
        logger.info(f"Using existing checkpoint directory: {config['checkpoint_dir']}")
    else:
        # 需要重置检查点目录，确保只有一层arch
        config['checkpoint_dir'] = base_config['checkpoint_dir'] / args.arch
        config['checkpoint_dir'].mkdir(exist_ok=True)
        logger.info(f"Reset checkpoint directory to: {config['checkpoint_dir']}")

    # 1. Load test data
    test_data_path = config['data_dir'] / f"{args.data_name}.test.pkl"
    test_data = load_pickle(test_data_path)
    logger.info(f"Loaded test data: {test_data_path}, samples: {len(test_data)}")
    
    # 2. Process test data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    
    # 准备缓存文件路径
    cached_examples_file = config['data_dir'] / f"cached_test_examples_{args.arch}"
    cached_features_file = config['data_dir'] / f"cached_test_features_{args.eval_max_seq_len}_{args.arch}"
    
    # 使用原来的方法获取测试数据和标签
    test_data = processor.get_test(test_data)
    label_list = processor.get_labels()
    
    # 准备id2label映射
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 创建examples和features
    test_examples = processor.create_examples(test_data, example_type="test", cached_examples_file=cached_examples_file)
    test_features = processor.create_features(examples=test_examples, max_seq_len=args.eval_max_seq_len, cached_features_file=cached_features_file)
    test_dataset = processor.create_dataset(test_features, is_sorted=False)
    
    # 3. Create test data loader
    from torch.utils.data import DataLoader, SequentialSampler
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, 
        sampler=test_sampler, 
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn
    )
    
    # 4. Load model
    logger.info("Loading model...")
    
    # Determine model loading path
    resume_path = args.resume_path
    if resume_path:
        path = Path(resume_path)
        if path.is_dir():
            resume_path = path
            model_path = resume_path / args.model_name
            logger.info(f"Loading model from directory {resume_path}, using model file: {model_path}")
        else:
            model_path = path
            logger.info(f"Loading model file directly: {model_path}")
    else:
        # 使用检查点目录
        model_path = config['checkpoint_dir'] / args.model_name
        logger.info(f"Loading model from checkpoint directory: {config['checkpoint_dir']}, using model file: {model_path}")
    
    # 检查模型文件是否存在
    if not model_path.exists():
        logger.warning(f"Model file does not exist: {model_path}")
    else:
        logger.info(f"Model file exists: {model_path}")
    
    # Load model configuration and weights
    try:
        from pybert.model.bert_for_multi_label import BertForMultiLable
        
        # 确保使用正确的检查点路径
        checkpoint_dir = config['checkpoint_dir']
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        logger.info(f"Checking if config.json exists: {(checkpoint_dir / 'config.json').exists()}")
        
        model = BertForMultiLable.from_pretrained(
            pretrained_model_name_or_path=checkpoint_dir,
            num_labels=len(label_list),
            id2label=id2label,
            model_name=args.model_name
        )
        
        # Initialize predictor
        try:
            # Load checkpoint info
            if args.resume_path:
                checkpoint_path = Path(args.resume_path)
                if checkpoint_path.is_dir():
                    checkpoint_info_path = checkpoint_path / 'checkpoint_info.bin'
                else:
                    checkpoint_info_path = checkpoint_path.parent / 'checkpoint_info.bin'
            else:
                checkpoint_info_path = config['checkpoint_dir'] / 'checkpoint_info.bin'
                
            if checkpoint_info_path.exists():
                checkpoint_info = torch.load(checkpoint_info_path)
                logger.info(f"Best {args.monitor}: {checkpoint_info.get('best', 'N/A')}, Epoch: {checkpoint_info.get('epoch', 'N/A')}")
            
            # Create log directory for test results
            # Avoid adding arch directory redundantly
            if "test_results" in str(config['log_dir']) and config['log_dir'].name == args.arch:
                # Already the correct test log directory path
                test_log_dir = config['log_dir']
            else:
                # Need to create new test log directory path
                test_log_dir = base_config['log_dir'] / "test_results" / args.arch
            
            test_log_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Test results will be saved to: {test_log_dir}")
            
            # Determine model name for logging
            model_name_for_log = args.model_name
            if args.resume_path:
                resume_path = Path(args.resume_path)
                if not resume_path.is_dir():
                    model_name_for_log = resume_path.name
                
            # Configure predictor with metrics
            predictor = Predictor(
                model=model,
                logger=logger,
                n_gpu=args.n_gpu,
                test_metrics=[
                    AccuracyThresh(thresh=0.5, search_thresh=False),
                    Precision(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
                    Recall(task_type='binary', average='micro', thresh=0.5, search_thresh=False),
                    AccuracyThresh(thresh=0.5, search_thresh=False),
                    HammingScore(thresh=0.5, search_thresh=False),
                    HammingLoss(thresh=0.5, search_thresh=False),
                    Jaccard(average='micro', thresh=0.5, search_thresh=False),
                    F1Score(thresh=0.5, normalizate=True, task_type='binary', average='micro', search_thresh=False),
                    AUC(task_type='binary'),
                    MultiLabelReport(id2label=id2label, average='micro', logger=logger)
                ],
                log_dir=test_log_dir,
                model_name=model_name_for_log
            )
            
            # Run prediction
            logger.info("Starting prediction...")
            result = predictor.predict(data=test_dataloader)
            logger.info("Model test results:")
            print(result)
            
        except Exception as e:
            logger.error(f"Error occurred during model testing: {str(e)}")
            raise e
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise e
        
    logger.info("Testing completed")


# %%
def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='BERT-TEXTCNN1', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--model_name", default='pytorch_model.bin', type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.3, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument("--train_max_seq_len", default=128, type=int)
    parser.add_argument("--eval_max_seq_len", default=128, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-05, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    # Call init_logger function to initialize the logger, recording logs to the specified file
    # The filename is composed of args.arch and the current timestamp
    log_file_path = Path(config['log_dir']) / f'{args.arch}-{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())}.log'
    init_logger(log_file_path)
    # Set checkpoint directory:
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        
        # Read and merge CSV file
        emse_file_path = 'pybert/dataset/emse.csv'
        df = pd.read_csv(emse_file_path)
        
        # Extract sentences and labels
        sentences = df['review'].tolist()
        targets = df[['Usa', 'Sup', 'Dep', 'Per']].values.tolist()
        
        # Analyze label distribution
        logger.info("Dataset label distribution:")
        label_counts = df[['Usa', 'Sup', 'Dep', 'Per']].sum().to_dict()
        total_samples = len(df)
        for label, count in label_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
        
        # Analyze multi-label combinations
        label_combinations = df[['Usa', 'Sup', 'Dep', 'Per']].apply(lambda x: ''.join([str(int(i)) for i in x]), axis=1)
        combination_counts = label_combinations.value_counts()
        logger.info("\nMulti-label combination distribution:")
        for combo, count in combination_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"Combination {combo}: {count} samples ({percentage:.2f}%)")
        
        # First split data into training set (80%) and temporary set (20%)
        train_sentences, temp_sentences, train_targets, temp_targets = train_test_split(
            sentences, targets, test_size=0.2, random_state=args.seed, shuffle=True
        )
        
        # Then split temporary set equally into validation set (10%) and test set (10%)
        valid_sentences, test_sentences, valid_targets, test_targets = train_test_split(
            temp_sentences, temp_targets, test_size=0.5, random_state=args.seed, shuffle=True
        )
        
        logger.info(f"Dataset split completed, Training: {len(train_sentences)}, Validation: {len(valid_sentences)}, Testing: {len(test_sentences)}")
        
        # Create list of data pairs
        train_data = list(zip(train_sentences, train_targets))
        valid_data = list(zip(valid_sentences, valid_targets))
        test_data = list(zip(test_sentences, test_targets))
        
        # Save training, validation and test sets
        train_path = config['data_dir'] / f"{args.data_name}.train.pkl"
        valid_path = config['data_dir'] / f"{args.data_name}.valid.pkl"
        test_path = config['data_dir'] / f"{args.data_name}.test.pkl"
        
        save_pickle(data=train_data, file_path=train_path)
        save_pickle(data=valid_data, file_path=valid_path)
        save_pickle(data=test_data, file_path=test_path)
        
        logger.info(f"Data saved to: {train_path}, {valid_path}, {test_path}")
        
    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
