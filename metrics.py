import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

def calculate_metrics(true_labels, pred_labels):
    """
    计算多标签分类的评估指标
    
    Args:
        true_labels: 包含每个样本真实标签的列表的列表
        pred_labels: 包含每个样本预测标签的列表的列表
        
    Returns:
        包含所有评估指标的字典
    """
    # 定义所有可能的标签
    all_labels = ["Per", "Sup", "Rel", "Usa", "Mis"]
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    # 将标签列表转换为二进制向量
    y_true = []
    y_pred = []
    
    # 用于计算命中率
    total_labels = 0
    correct_labels = 0
    
    # 用于计算完全匹配准确率
    correct = 0
    total = 0
    
    for true_label_set, pred_label_set in zip(true_labels, pred_labels):
        true_vec = [0] * len(all_labels)
        pred_vec = [0] * len(all_labels)
        
        for label in true_label_set:
            if label in label_to_idx:
                true_vec[label_to_idx[label]] = 1
                
        for label in pred_label_set:
            if label in label_to_idx:
                pred_vec[label_to_idx[label]] = 1
        
        # 计算命中标签
        correct_in_sample = sum(1 for label in pred_label_set if label in true_label_set)
        total_in_sample = len(true_label_set)
        
        correct_labels += correct_in_sample
        total_labels += total_in_sample
        
        # 计算完全匹配
        if set(true_label_set) == set(pred_label_set):
            correct += 1
        total += 1
        
        y_true.append(true_vec)
        y_pred.append(pred_vec)
    
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算标签命中率
    label_hit_accuracy = correct_labels / total_labels if total_labels > 0 else 0
    
    # 计算完全匹配准确率
    accuracy = correct / total if total > 0 else 0
    
    # 计算多标签指标
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算Hamming Score
    hamming_score = 1 - hamming_loss(y_true, y_pred)
    
    # 计算每个类别的指标
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for i, label in enumerate(all_labels):
        precision_per_class[label] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall_per_class[label] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_per_class[label] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    
    # 打印结果
    print("\n=== 评估结果 ===")
    print(f"完全匹配准确率: {accuracy*100:.2f}%")
    print(f"标签命中率: {label_hit_accuracy*100:.2f}%")
    print(f"Hamming Score: {hamming_score*100:.2f}%")
    print(f"Micro-Precision: {precision_micro*100:.2f}%")
    print(f"Micro-Recall: {recall_micro*100:.2f}%")
    print(f"Micro-F1: {f1_micro*100:.2f}%")
    print(f"Macro-Precision: {precision_macro*100:.2f}%")
    print(f"Macro-Recall: {recall_macro*100:.2f}%")
    print(f"Macro-F1: {f1_macro*100:.2f}%")
    
    print("\n各类别指标:")
    for label in all_labels:
        print(f"{label}: Precision={precision_per_class[label]*100:.2f}%, Recall={recall_per_class[label]*100:.2f}%, F1={f1_per_class[label]*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'label_hit_accuracy': label_hit_accuracy,
        'hamming_score': hamming_score,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    } 