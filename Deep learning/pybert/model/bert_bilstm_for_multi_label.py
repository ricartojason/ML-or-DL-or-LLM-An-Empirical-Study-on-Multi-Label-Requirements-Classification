import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class BertBiLstmForMultiLabel(BertPreTrainedModel):
    """
    BERT+BiLSTM多标签分类模型
    使用BERT提取特征，BiLSTM处理序列信息，然后进行多标签分类
    
    参数:
        batch_size: 128
        learning_rate: 0.01
        N-gram window size: 3
        dropout: 0.5
    """
    def __init__(self, config):
        super(BertBiLstmForMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,  # 双向，所以隐藏状态大小减半
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=0.5 if config.hidden_dropout_prob > 0.5 else config.hidden_dropout_prob
        )
        
        # N-gram卷积层 (window size = 3)
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.dense = nn.Linear(config.hidden_size, 128)
        
        # 输出分类层
        self.classifier = nn.Linear(128, config.num_labels)
        
        # 初始化权重
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入的token IDs
            token_type_ids: token类型IDs，用于区分句子对
            attention_mask: 注意力掩码，用于忽略padding
            head_mask: head掩码，用于掩蔽特定的注意力头
        
        Returns:
            logits: 每个标签的预测概率（未经过sigmoid）
        """
        # 获取BERT输出
        outputs = self.bert(
            input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask
        )
        
        # 获取序列输出
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 应用N-gram卷积
        # 先变换维度以适应卷积操作
        conv_input = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        conv_output = self.conv(conv_input)  # [batch_size, hidden_size, seq_len]
        conv_output = conv_output.permute(0, 2, 1)  # [batch_size, seq_len, hidden_size]
        
        # 应用BiLSTM层处理序列
        lstm_output, _ = self.lstm(conv_output)  # [batch_size, seq_len, hidden_size]
        
        # 获取序列的最后一个非填充位置的隐藏状态
        # 使用attention_mask来获取每个序列的实际长度
        if attention_mask is not None:
            # 计算每个序列的实际长度
            seq_lengths = attention_mask.sum(dim=1) - 1  # 减1是因为索引从0开始
            
            # 收集每个序列最后一个token的hidden state
            batch_size = lstm_output.size(0)
            batch_indices = torch.arange(0, batch_size, device=lstm_output.device)
            seq_indices = seq_lengths.long()
            
            # 获取每个序列最后一个非填充位置的隐藏状态
            final_hidden = lstm_output[batch_indices, seq_indices]
        else:
            # 如果没有attention_mask，直接使用最后一个位置的隐藏状态
            final_hidden = lstm_output[:, -1, :]
        
        # 应用dropout
        final_hidden = self.dropout(final_hidden)
        
        # 通过全连接层
        dense_output = F.relu(self.dense(final_hidden))
        
        # 再次应用dropout
        dense_output = self.dropout(dense_output)
        
        # 应用分类器
        logits = self.classifier(dense_output)
        
        return logits

    def unfreeze(self, start_layer, end_layer):
        """
        解冻BERT的特定层，用于微调
        
        Args:
            start_layer: 开始解冻的层索引
            end_layer: 结束解冻的层索引
        """
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # 首先冻结所有BERT层
        set_trainable(self.bert, False)
        
        # 然后解冻指定的层
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练模型加载模型
        
        Args:
            pretrained_model_name_or_path: 预训练模型名称或路径
            model_args: 传递给模型初始化的位置参数
            kwargs: 传递给模型初始化的关键字参数
                - num_labels: 标签数量
                - model_name: 自定义模型文件名，默认为'pytorch_model.bin'
        """
        model_name = kwargs.pop('model_name', 'pytorch_model.bin')
        
        # 首先尝试使用父类的from_pretrained方法
        try:
            # 标准路径或模型名称
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        except Exception as e:
            # 如果是目录，并且存在自定义名称的模型文件
            if os.path.isdir(pretrained_model_name_or_path):
                path = Path(pretrained_model_name_or_path)
                model_file = path / model_name
                config_file = path / 'config.json'
                
                if model_file.exists() and config_file.exists():
                    # 首先加载配置
                    config = cls.config_class.from_pretrained(config_file)
                    
                    # 使用配置初始化模型
                    model = cls(config, **kwargs)
                    
                    # 加载模型权重
                    state_dict = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(state_dict)
                    
                    return model
                    
            # 如果上述都失败，则报错
            raise ValueError(f"Failed to load model, check path and filename: {pretrained_model_name_or_path}, {model_name}, original error: {str(e)}") 