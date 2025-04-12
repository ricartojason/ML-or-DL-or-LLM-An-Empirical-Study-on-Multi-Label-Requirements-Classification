from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import BCELoss


__call__ = ['CrossEntropy','BCEWithLogLoss','BCELoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self, pos_weight=None):
        # pos_weight是一个向量，用于为各个类别的正样本设置权重
        # 对于类别不平衡的情况，可以设置较小类别的权重为较大值
        self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss

class MultiLabelCrossEntropy(object):
    def __init__(self):
        pass
    def __call__(self, output, target):
        loss = CrossEntropyLoss(reduction='none')(output,target)
        return loss

class BCEWithLoss(object):
    def __init__(self):
        self.loss_fn = BCELoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss


