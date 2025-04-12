import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import os
from pathlib import Path

class BertForMultiLable(BertPreTrainedModel):
    # 简单的BERT多标签分类架构（注释掉）
    # def __init__(self, config):
    #     super(BertForMultiLable, self).__init__(config)
    #     self.bert = BertModel(config)
    #     #dropout丢弃比率0.1，防止过拟合
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    #     self.init_weights()
    # 
    # def forward(self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None):
    #     outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
    #     pooled_output = outputs[1]
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     return logits

    # BERT+TextCNN架构
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        self.conv1 = nn.Conv2d(1, 3, (2, config.hidden_size))
        self.conv2 = nn.Conv2d(1, 3, (3, config.hidden_size))
        self.conv3 = nn.Conv2d(1, 3, (4, config.hidden_size))
        self.linear = nn.Linear(9, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])) #(filter_height, filter_width)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)
        # pooled_output = outputs[1]
        # print(pooled_output.shape)
        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # 为卷积操作准备输入，增加一个维度以模拟通道
        out = sequence_output.unsqueeze(1)  # 形状变为 (batch_size, 1, sequence_length, hidden_size)即（一批次有几个句子，通道数，一个句子的序列长度，一个单词的词向量维度）
        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)
        # 展平 out1, out2, out3 到二维张量
        out1_flat = out1.view(out1.size(0), -1)
        out2_flat = out2.view(out2.size(0), -1)
        out3_flat = out3.view(out3.size(0), -1)

        # 拼接展平后的张量
        out = torch.cat([out1_flat, out2_flat, out3_flat], dim=1)
        out = self.dropout(out)
        logits = self.linear(out)
        
        # 对于多标签分类，需要应用sigmoid将输出转换为概率值
        # 注意：我们不在模型中应用sigmoid，而是保持logits输出
        # 这是因为BCEWithLogitsLoss已经包含了sigmoid计算
        # 在预测阶段，predictor会使用sigmoid进行转换
        return logits

    def unfreeze(self,start_layer,end_layer):
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

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        自定义from_pretrained方法，支持model_name参数
        
        参数:
            pretrained_model_name_or_path: 预训练模型名称或路径
                - 如果是目录，将检查该目录中的自定义模型文件
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
                    
                    # 使用配置初始化模型，不传递额外的关键字参数
                    model = cls(config)
                    
                    # 加载模型权重
                    state_dict = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(state_dict)
                    
                    return model
                    
            # 如果上述都失败，则报错
            raise ValueError(f"Failed to load model, check path and filename: {pretrained_model_name_or_path}, {model_name}, original error: {str(e)}")