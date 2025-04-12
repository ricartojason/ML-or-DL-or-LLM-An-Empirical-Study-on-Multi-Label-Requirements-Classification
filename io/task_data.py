import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
import chardet

class TaskData(object):
    def __init__(self, task_data_dir=None, tokenizer=None, data_name=None, preprocessor=None):
        """
        初始化TaskData对象
        
        Args:
            task_data_dir: 数据目录
            tokenizer: 分词器
            data_name: 数据集名称
            preprocessor: 预处理器
        """
        self.task_data_dir = task_data_dir
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.preprocessor = preprocessor
    
    # x表示特征数据，y表示标签数据，valid_size表示验证集的比例，shuffle: 是否随机打乱数据，
    # stratify: 是否根据标签 y 来分层抽样，以保持训练集和验证集中类别的比例与整个数据集相同。
    def train_val_split(self, X, y, train_size=0.8, stratify=False, shuffle=True, save=True,
                        seed=None, data_dir=None, data_name=None):
        """
        将数据分割为训练集、验证集和测试集，比例为8:1:1
        
        Args:
            X: 特征数据
            y: 标签数据
            train_size: 训练集比例，默认0.8
            stratify: 是否分层抽样
            shuffle: 是否打乱数据
            save: 是否保存分割后的数据
            seed: 随机种子
            data_dir: 数据目录，如果未提供则使用初始化时的目录
            data_name: 数据集名称，如果未提供则使用初始化时的名称
        
        Returns:
            train, valid, test: 训练集、验证集和测试集
        """
        # 使用类属性作为默认值
        if data_dir is None:
            data_dir = self.task_data_dir
        if data_name is None:
            data_name = self.data_name
            
        # 验证集和测试集各占10%
        valid_size = 0.1
        test_size = 0.1
            
        # 创建进度条
        pbar = ProgressBar(n_total=len(X), desc='bucket')
        # 记录日志，提示正在将原始数据划分
        logger.info('splitting raw data into train, validation and test sets (8:1:1)')
        
        # 非分层抽样方式
        if not stratify:
            data = []
            for step, (data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            
            # 计算数据的总数量
            N = len(data)
            
            # 如果需要打乱数据
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            
            # 计算验证集和测试集的大小
            train_count = int(N * train_size)
            valid_count = int(N * valid_size)
            test_count = int(N * test_size)
            
            # 划分数据
            train = data[:train_count]
            valid = data[train_count:train_count + valid_count]
            test = data[train_count + valid_count:]
            
            # 再次打乱训练集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        
        # 分层抽样方式
        else:
            # 计算类别的数量
            num_classes = len(list(set(y)))
            train, valid, test = [], [], []
            
            # 初始化每个类别的桶
            bucket = [[] for _ in range(num_classes)]
            
            # 遍历原始数据，按照类别划分到不同的桶中
            for step, (data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            del X, y
            
            # 遍历每个桶，按照比例划分数据
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                
                # 打乱桶内数据
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                
                # 计算每个集合应有的样本数
                test_count = int(N * test_size)
                valid_count = int(N * valid_size)
                
                # 划分数据
                train.extend(bt[:train_count])
                valid.extend(bt[train_count:train_count + valid_count])
                test.extend(bt[train_count + valid_count:])
            
            # 再次打乱各个集合
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
                random.seed(seed)
                random.shuffle(valid)
                random.seed(seed)
                random.shuffle(test)
        
        # 保存划分后的数据
        if save:
            # 定义训练集、验证集和测试集的保存路径
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            test_path = data_dir / f"{data_name}.test.pkl"
            
            # 保存数据
            save_pickle(data=train, file_path=train_path)
            save_pickle(data=valid, file_path=valid_path)
            save_pickle(data=test, file_path=test_path)
            
            logger.info(f"数据划分完成: 训练集({train_size*100}%): {len(train)}样本, 验证集({valid_size*100}%): {len(valid)}样本, 测试集({test_size*100}%): {len(test)}样本")
            logger.info(f"数据已保存到 {data_dir}")
        
        # 返回划分后的数据集
        return train, valid, test

    def read_data(self, raw_data_path, preprocessor=None, is_train=True, aug_data_path=None, is_augament=False):
        '''
        读取原始数据
        
        Args:
            raw_data_path: 原始数据路径
            preprocessor: 预处理器
            is_train: 是否为训练模式
            aug_data_path: 数据增强路径，如果为None则不使用数据增强
            is_augament: 是否进行数据增强
            
        Returns:
            targets, sentences: 标签和文本数据
        '''
        targets, sentences = [], []

        # 检测文件编码
        with open(raw_data_path, 'rb') as f:
            result = chardet.detect(f.read())
        encode = result['encoding']
        logger.info(f'文件编码: {encode}')

        # 读取原始数据
        data = pd.read_csv(raw_data_path)
        for row in data.values:
            if is_train:
                # 取原始数据里第2列到第6列，即标签列
                target = row[2:]
            else:
                target = row[2:]
            sentence = str(row[1])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
                
        # 数据增强（如果启用）
        if is_augament and aug_data_path:
            logger.info(f'使用数据增强: {aug_data_path}')
            TTA_data = pd.read_csv(aug_data_path)
            for row1 in TTA_data.values:
                aug_target = row1[2:]
                aug_sentence = str(row1[1])
                if preprocessor:
                    aug_sentence = preprocessor(aug_sentence)
                if aug_sentence:
                    targets.append(aug_target)
                    sentences.append(aug_sentence)
        
        logger.info(f'读取数据: {len(sentences)}行')
        return targets, sentences
