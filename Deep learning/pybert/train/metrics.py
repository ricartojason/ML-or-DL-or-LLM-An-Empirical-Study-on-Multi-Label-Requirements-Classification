r"""Functional interface"""
import torch
from sklearn.preprocessing import label_binarize, MultiLabelBinarizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold

__call__ = ['Accuracy','AUC','F1Score','ClassReport','MultiLabelReport','AccuracyThresh', 'Precision', 'Recall', 'HammingScore', 'HammingLoss','Jaccard']

class Metric:
    """基础评估指标类"""
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def get_predictions(self, logits, thresh=0.5):
        """通用方法：将logits转换为预测值"""
        y_prob = logits.sigmoid().data.cpu().numpy()
        return y_prob, (y_prob > thresh).astype(int)

class Accuracy(Metric):
    """
    计算准确度
    可以使用topK参数设定计算K准确度
    """
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total if self.total > 0 else 0

    def name(self):
        return 'accuracy'

class AccuracyThresh(Metric):
    """
    计算给定阈值下的准确度
    使用固定阈值处理多标签分类
    """
    def __init__(self, thresh=0.5, search_thresh=False):
        super(AccuracyThresh, self).__init__()
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性
        self.reset()

    def __call__(self, logits, target):
        self.y_true = target
        _, self.y_pred = self.get_predictions(logits, self.thresh)
        acc = self.value()
        return acc

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None:
            return 0
        data_size = self.y_pred.shape[0]
        if data_size == 0:
            return 0
        
        # 确保两个张量在同一设备上
        device = self.y_true.device
        # 将NumPy数组转换为张量并移至正确的设备
        y_pred_tensor = torch.tensor(self.y_pred, device=device)
        
        # 计算准确度
        acc = torch.mean((y_pred_tensor == self.y_true.byte()).float(), dim=1).sum().item()
        return acc / data_size

    def name(self):
        return 'accuracy'

class Precision(Metric):
    """计算精确率"""
    def __init__(self, task_type='binary', average='samples', thresh=0.5, search_thresh=False):
        super(Precision, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            _, self.y_pred = self.get_predictions(logits, self.thresh)
            precision = self.value()
            return precision
        elif self.task_type == 'multiclass':
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
            self.y_pred = np.argmax(self.y_prob, 1)
            precision = self.value()
            return precision

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        return precision_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average, zero_division=0)

    def name(self):
        return 'precision'

class Recall(Metric):
    """计算召回率"""
    def __init__(self, task_type='binary', average='samples', thresh=0.5, search_thresh=False):
        super(Recall, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            _, self.y_pred = self.get_predictions(logits, self.thresh)
            recall = self.value()
            return recall
        elif self.task_type == 'multiclass':
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
            self.y_pred = np.argmax(self.y_prob, 1)
            recall = self.value()
            return recall

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        return recall_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average, zero_division=0)

    def name(self):
        return 'recall'

def hamming_score(y_true, y_pred):
    """计算正确的hamming score"""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true和y_pred的形状必须一致")
    
    acc_list = []
    for i in range(y_true.shape[0]):
        # 计算预测正确的位数（包括0和1）
        correct_predictions = np.sum(y_true[i] == y_pred[i])
        total_positions = len(y_true[i])
        acc_list.append(correct_predictions / total_positions)
                
    return np.mean(acc_list)

class HammingScore(Metric):
    """计算Hamming Score"""
    def __init__(self, thresh=0.5, search_thresh=False):
        super(HammingScore, self).__init__()
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        _, self.y_pred = self.get_predictions(logits, self.thresh)
        score = self.value()
        return score

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None:
            return 0
        return hamming_score(y_true=self.y_true, y_pred=self.y_pred)

    def name(self):
        return 'hamming_score'

class HammingLoss(Metric):
    """计算Hamming Loss"""
    def __init__(self, thresh=0.5, search_thresh=False):
        super(HammingLoss, self).__init__()
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        _, self.y_pred = self.get_predictions(logits, self.thresh)
        loss = self.value()
        return loss

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        return hamming_loss(y_true=self.y_true, y_pred=self.y_pred)

    def name(self):
        return 'hamming_loss'

class Jaccard(Metric):
    """计算Jaccard系数"""
    def __init__(self, average='macro', thresh=0.5, search_thresh=False):
        super(Jaccard, self).__init__()
        self.average = average
        self.thresh = thresh
        self.search_thresh = False  # 固定为False以确保一致性

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        _, self.y_pred = self.get_predictions(logits, self.thresh)
        jac = self.value()
        return jac

    def reset(self):
        self.y_pred = None
        self.y_true = None

    def value(self):
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        return jaccard_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average, zero_division=0)

    def name(self):
        return 'jaccard'

class AUC(Metric):
    """
    计算AUC分数
    确保使用概率值而非二值化预测
    """
    def __init__(self, task_type='binary', average='macro', search_thresh=False):
        super(AUC, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']
        self.task_type = task_type
        self.average = average
        self.search_thresh = False  # AUC不需要阈值，始终为False

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        if self.task_type == 'binary':
            # 对于AUC，保存概率值而非二值化预测
            self.y_prob = logits.sigmoid().data.cpu().numpy()
            return self.value()
        elif self.task_type == 'multiclass':
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
            return self.value()

    def reset(self):
        self.y_prob = None
        self.y_true = None

    def value(self):
        if self.y_prob is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        try:
            # AUC计算使用概率值
            return roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        except ValueError:
            # 处理极端情况，例如只有一个类别
            return 0

    def name(self):
        return 'auc'

class F1Score(Metric):
    """计算F1分数"""
    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='macro', search_thresh=False):
        super(F1Score, self).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted', 'None']
        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = False  # 固定为False以确保一致性
        self.average = average

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        
        if self.normalizate:
            if self.task_type == 'binary':
                self.y_prob = logits.sigmoid().data.cpu().numpy()
                self.y_pred = (self.y_prob > self.thresh).astype(int)
            elif self.task_type == 'multiclass':
                self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
                self.y_pred = np.argmax(self.y_prob, 1)
        else:
            self.y_prob = logits.cpu().detach().numpy()
            if self.task_type == 'binary':
                self.y_pred = (self.y_prob > self.thresh).astype(int)
            else:
                self.y_pred = np.argmax(self.y_prob, 1)
                
        f1 = self.value()
        return f1

    def reset(self):
        self.y_pred = None
        self.y_true = None
        self.y_prob = None

    def value(self):
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return 0
        return f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average, zero_division=0)

    def name(self):
        return 'f1'

class ClassReport(Metric):
    """多标签分类报告"""
    def __init__(self, logger, id2label=None):
        super(ClassReport, self).__init__()
        self.logger = logger
        # 允许传递id2label，而不是硬编码
        if id2label is None:
            self.id2label = {
                0: 'Usa',
                1: 'Sup',
                2: 'Dep',
                3: 'Per'
            }
        else:
            self.id2label = id2label
        self.target_names = [self.id2label[i] for i in range(len(self.id2label))]
        self.reset()

    def reset(self):
        self.y_pred = None
        self.y_true = None
        self.report_dict = None

    def __call__(self, logits, target):
        # 对多标签分类，使用阈值0.5处理
        self.y_pred = (logits.sigmoid() > 0.5).int().cpu().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        """计算并返回每个标签的分类报告"""
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return None
            
        # 多标签分类报告
        report_per_class = {}
        for i, label in enumerate(self.target_names):
            # 对每个标签单独计算分类报告
            try:
                report = classification_report(
                    y_true=self.y_true[:, i],
                    y_pred=self.y_pred[:, i],
                    target_names=['非'+label, label],  # 0表示非该标签，1表示该标签
                    output_dict=True,
                    zero_division=0
                )
                report_per_class[label] = report
                
                # 格式化输出
                report_str = (f"\n{label} classification report:\n"
                             f"              precision    recall  f1-score   support\n"
                             f"       非{label}     {report['非'+label]['precision']:.2f}      {report['非'+label]['recall']:.2f}      {report['非'+label]['f1-score']:.2f}     {report['非'+label]['support']}\n"
                             f"        {label}     {report[label]['precision']:.2f}      {report[label]['recall']:.2f}      {report[label]['f1-score']:.2f}     {report[label]['support']}\n"
                             f"    accuracy                          {report['accuracy']:.2f}     {report['macro avg']['support']}\n"
                             f"   macro avg     {report['macro avg']['precision']:.2f}      {report['macro avg']['recall']:.2f}      {report['macro avg']['f1-score']:.2f}     {report['macro avg']['support']}\n"
                             f"weighted avg     {report['weighted avg']['precision']:.2f}      {report['weighted avg']['recall']:.2f}      {report['weighted avg']['f1-score']:.2f}     {report['weighted avg']['support']}\n")
                
                self.logger.info(report_str)
            except Exception as e:
                self.logger.warning(f"Error computing classification report for {label}: {str(e)}")
        
        # 保存所有标签的详细报告
        self.report_dict = report_per_class
                
        # 计算总体指标
        try:
            overall_metrics = {
                'accuracy': accuracy_score(self.y_true.flatten(), self.y_pred.flatten()),
                'precision': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'recall': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'f1': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
            }
            self.logger.info(f"\nOverall classification metrics: {overall_metrics}")
            return overall_metrics
        except Exception as e:
            self.logger.warning(f"Error computing overall metrics: {str(e)}")
            return None

    def get_detailed_report(self):
        """获取详细的分类报告数据"""
        return self.report_dict

    def name(self):
        return "class_report"

class MultiLabelReport(Metric):
    """多标签评估报告，支持自定义阈值（默认0.5）"""
    def __init__(self, id2label, average, logger, thresh=0.5):
        super(MultiLabelReport, self).__init__()
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted', 'None']
        self.id2label = id2label
        self.average = average
        self.logger = logger
        self.thresh = thresh
        self.reset()

    def reset(self):
        self.y_pred = None
        self.y_true = None
        self.y_prob = None
        self.report_dict = None

    def __call__(self, logits, target):
        # 保存概率值和目标值
        self.y_prob = logits.sigmoid().data.cpu().numpy()
        self.y_true = target.cpu().numpy()
        # 使用设定的阈值进行预测
        self.y_pred = (self.y_prob > self.thresh).astype(int)
        # 生成报告字典
        self._generate_report_dict()

    def _generate_report_dict(self):
        """生成详细的报告字典"""
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return
            
        report_dict = {}
        
        # 计算每个标签的指标
        for i, label in self.id2label.items():
            try:
                # 对当前标签计算指标
                report_dict[label] = {
                    'precision': precision_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'recall': recall_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'f1-score': f1_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'support': int(np.sum(self.y_true[:, i]))
                }
            except Exception as e:
                self.logger.warning(f"Error computing metrics for label {label}: {str(e)}")
        
        # 计算宏平均和加权平均
        try:
            report_dict['macro avg'] = {
                'precision': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'recall': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'f1-score': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
                'support': int(np.sum(self.y_true))
            }
            
            report_dict['weighted avg'] = {
                'precision': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
                'f1-score': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
                'support': int(np.sum(self.y_true))
            }
            
            # 添加准确率
            report_dict['accuracy'] = accuracy_score(self.y_true.flatten(), self.y_pred.flatten())
            
        except Exception as e:
            self.logger.warning(f"Error computing average metrics: {str(e)}")
        
        self.report_dict = report_dict

    def value(self):
        """计算并输出每个标签的评估指标"""
        if self.y_pred is None or self.y_true is None or len(self.y_true) == 0:
            return None
            
        # 计算总体指标
        try:
            overall_metrics = {
                'auc': roc_auc_score(self.y_true, self.y_prob, average=self.average),
                'precision': precision_score(self.y_true, self.y_pred, average=self.average, zero_division=0),
                'recall': recall_score(self.y_true, self.y_pred, average=self.average, zero_division=0),
                'f1': f1_score(self.y_true, self.y_pred, average=self.average, zero_division=0),
                'hamming_loss': hamming_loss(self.y_true, self.y_pred),
                'hamming_score': hamming_score(self.y_true, self.y_pred),
                'jaccard': jaccard_score(self.y_true, self.y_pred, average=self.average, zero_division=0)
            }
            self.logger.info(f"\nOverall metrics using thresh={self.thresh}: {overall_metrics}")
        except Exception as e:
            self.logger.warning(f"Error computing overall metrics: {str(e)}")
            
        # 计算每个标签的指标
        for i, label in self.id2label.items():
            try:
                # 对当前标签计算指标
                label_metrics = {
                    'auc': roc_auc_score(self.y_true[:, i], self.y_prob[:, i]),
                    'precision': precision_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'recall': recall_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'f1': f1_score(self.y_true[:, i], self.y_pred[:, i], zero_division=0),
                    'support': np.sum(self.y_true[:, i]),
                    'positive_pred': np.sum(self.y_pred[:, i])
                }
                
                # 构建并输出报告
                log_message = (f"Label: {label} - "
                              f"AUC: {label_metrics['auc']:.4f}, "
                              f"Precision: {label_metrics['precision']:.4f}, "
                              f"Recall: {label_metrics['recall']:.4f}, "
                              f"F1: {label_metrics['f1']:.4f}, "
                              f"Support: {label_metrics['support']}, "
                              f"Predicted positive: {label_metrics['positive_pred']}")
                              
                self.logger.info(log_message)
            except Exception as e:
                self.logger.warning(f"Error computing metrics for label {label}: {str(e)}")
        
        return overall_metrics

    def get_detailed_report(self):
        """获取详细的多标签分类报告数据"""
        if self.report_dict is None:
            self._generate_report_dict()
        return self.report_dict

    def name(self):
        return "multilabel_report"

