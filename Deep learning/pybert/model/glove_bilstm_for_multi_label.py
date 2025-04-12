import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path

class GloveBiLSTMForMultiLabel(nn.Module):
    """
    GloVe + BiLSTM model for multi-label classification
    """
    def __init__(self, vocab_size, embedding_dim, num_labels):
        super().__init__()
        
        # 保存配置参数
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        
        # GloVe embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)  # spatial dropout
        
        # 1D卷积层
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=128,  # Conv1D filters = 128
            kernel_size=5      # Conv1D kernel size = 5
        )
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=128,    # 与Conv1D的out_channels相同
            hidden_size=512,   # LSTM units = 512
            bidirectional=True,
            batch_first=True,
            dropout=0.1        # LSTM dropout = 0.1
        )
        
        # 全连接层
        self.dense1 = nn.Linear(1024, 128)  # 1024 = 512*2 (bidirectional)
        self.dense2 = nn.Linear(128, 32)    # dense layer 1&2 units = 128, 32
        
        self.dropout = nn.Dropout(0.1)      # recurrent dropout = 0.1
        
        # 分类层
        self.classifier = nn.Linear(32, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding_dropout(embedded)
        
        # Convolution
        conv_input = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        conv_output = self.conv(conv_input)
        conv_output = F.relu(conv_output)
        
        # BiLSTM
        lstm_input = conv_output.permute(0, 2, 1)  # [batch_size, seq_len, conv_channels]
        lstm_output, _ = self.lstm(lstm_input)
        
        # Global max pooling
        pooled_output = torch.max(lstm_output, dim=1)[0]
        
        # Dense layers with ReLU
        dense1_output = self.dense1(pooled_output)
        dense1_output = F.relu(dense1_output)
        dense1_output = self.dropout(dense1_output)
        
        dense2_output = self.dense2(dense1_output)
        dense2_output = F.relu(dense2_output)
        dense2_output = self.dropout(dense2_output)
        
        # 分类层
        logits = self.classifier(dense2_output)
        
        return logits
    
    def load_glove_embeddings(self, glove_path, word_to_idx):
        """
        Load GloVe embeddings from file and initialize embedding layer
        
        Args:
            glove_path: Path to GloVe embeddings file
            word_to_idx: Dictionary mapping words to indices
        """
        # 如果存在预处理好的pt文件
        if os.path.exists(glove_path + '.pt'):
            try:
                # 直接加载预处理好的嵌入
                embeddings = torch.load(glove_path + '.pt', map_location='cpu')
                if isinstance(embeddings, tuple):  # 如果是元组，取第一个元素
                    embeddings = embeddings[0]
                
                # 检查嵌入维度是否匹配
                if embeddings.size(1) != self.embedding_dim:
                    print(f"嵌入维度不匹配: 预处理的维度为 {embeddings.size(1)}，期望的维度为 {self.embedding_dim}")
                    raise ValueError("嵌入维度不匹配")
                
                # 如果词表大小不匹配，只取前vocab_size个词
                if embeddings.size(0) != self.vocab_size:
                    print(f"词表大小不匹配: 预处理的大小为 {embeddings.size(0)}，期望的大小为 {self.vocab_size}")
                    print("将只使用前{}个词".format(self.vocab_size))
                    embeddings = embeddings[:self.vocab_size]
                
                # 更新嵌入层
                self.embedding.weight.data.copy_(embeddings)
                print(f"成功加载预处理嵌入，形状: {embeddings.shape}")
                return
            except Exception as e:
                print(f"加载预处理嵌入时出错: {str(e)}，将从原始GloVe文件加载")
        
        print(f"从原始GloVe文件加载嵌入: {glove_path}")
        # 初始化嵌入随机值
        embeddings = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        
        # 加载GloVe嵌入
        found_count = 0
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_to_idx:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[word_to_idx[word]] = vector
                    found_count += 1
        
        print(f"在GloVe中找到 {found_count}/{len(word_to_idx)} 个词")
        
        # 更新嵌入层
        self.embedding.weight.data.copy_(torch.FloatTensor(embeddings))
        
        # 保存处理后的嵌入以便future使用
        try:
            torch.save(self.embedding.weight.data, glove_path + '.pt')
            print(f"嵌入已保存至 {glove_path}.pt")
        except Exception as e:
            print(f"保存嵌入时出错: {str(e)}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained model
        
        Args:
            pretrained_model_name_or_path: Path to pre-trained model or model name
            model_args: Additional positional arguments for model initialization
            kwargs: Additional keyword arguments for model initialization
        """
        model_name = kwargs.pop('model_name', 'pytorch_model.bin')
        
        # Try to load model from path
        if os.path.isdir(pretrained_model_name_or_path):
            path = Path(pretrained_model_name_or_path)
            model_file = path / model_name
            
            if model_file.exists():
                # Create config object
                class Config:
                    pass
                config = Config()
                config.vocab_size = kwargs.pop('vocab_size', 30522)  # Default BERT vocab size
                config.embedding_dim = kwargs.pop('embedding_dim', 300)  # GloVe 300d
                config.num_labels = kwargs.pop('num_labels', 4)
                
                # Initialize model
                model = cls(config.vocab_size, config.embedding_dim, config.num_labels, *model_args, **kwargs)
                
                # Load model weights
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict)
                
                return model
        
        # If model couldn't be loaded, create a new one
        class Config:
            pass
        config = Config()
        config.vocab_size = kwargs.pop('vocab_size', 30522)  # Default BERT vocab size
        config.embedding_dim = kwargs.pop('embedding_dim', 300)  # GloVe 300d
        config.num_labels = kwargs.pop('num_labels', 4)
        
        return cls(config.vocab_size, config.embedding_dim, config.num_labels, *model_args, **kwargs) 