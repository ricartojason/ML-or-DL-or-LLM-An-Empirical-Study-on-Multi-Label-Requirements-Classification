import torch
import torch.nn as nn
import os
from pathlib import Path
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class BertPureForMultiLabel(BertPreTrainedModel):
    """
    纯BERT多标签分类模型
    使用BERT的[CLS]表示经过一个中间密集层后进行分类
    """
    def __init__(self, config, threshold=0.5):
        super(BertPureForMultiLabel, self).__init__(config)
        self.bert = BertModel(config)
        
        # 注释掉原有的复杂架构
        # # 定义中间密集层，使用20个神经元（可以根据需要改为30）
        self.dense = nn.Linear(config.hidden_size, 20)
        
        # Dropout层，防止过拟合，设置为0.2
        self.dropout = nn.Dropout(0.2)
        
        # 输出分类层
        self.classifier = nn.Linear(20, config.num_labels)
        
        # 新的简化架构，与正常工作的实现一致
        # Dropout层，使用模型配置中的dropout概率
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # # 直接从BERT隐藏状态到标签的线性分类器
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 设置阈值，默认为0.5（与工作实现保持一致）
        self.threshold = threshold
        
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
        
        # 获取[CLS]表示，即第一个token的隐藏状态
        pooled_output = outputs[1]
        
        # 注释掉原有的复杂前向传播
        # # 通过中间密集层
        dense_output = self.dense(pooled_output)
        
        # 应用dropout
        dense_output = self.dropout(dense_output)
        
        # 应用分类器
        logits = self.classifier(dense_output)
        
        # 新的简化前向传播
        # 应用dropout
        # pooled_output = self.dropout(pooled_output)
        
        # # 直接应用分类器
        # logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        """
        预测函数，返回经过sigmoid和阈值处理后的预测结果
        
        Args:
            input_ids: 输入的token IDs
            token_type_ids: token类型IDs，用于区分句子对
            attention_mask: 注意力掩码，用于忽略padding
            head_mask: head掩码，用于掩蔽特定的注意力头
            
        Returns:
            predictions: 二值化的预测结果 (0或1)
            probabilities: sigmoid后的概率值
        """
        logits = self.forward(input_ids, token_type_ids, attention_mask, head_mask)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities

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
                - threshold: 预测阈值，默认为0.5
                - state_dict: 预加载的模型权重
        """
        model_name = kwargs.pop('model_name', 'pytorch_model.bin')
        threshold = kwargs.pop('threshold', 0.5)
        state_dict = kwargs.pop('state_dict', None)
        
        # 首先尝试使用父类的from_pretrained方法
        try:
            # 标准路径或模型名称
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            model.threshold = threshold
            
            # 如果有预加载的权重，加载它们
            if state_dict is not None:
                model.load_state_dict(state_dict)
                
            return model, None
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
                    model = cls(config, threshold=threshold, **kwargs)
                    
                    # 加载模型权重（优先使用传入的权重）
                    if state_dict is not None:
                        model.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(torch.load(model_file, map_location='cpu'))
                    
                    return model, config_file
                    
            # 如果上述都失败，则报错
            raise ValueError(f"Failed to load model, check path and filename: {pretrained_model_name_or_path}, {model_name}, original error: {str(e)}") 