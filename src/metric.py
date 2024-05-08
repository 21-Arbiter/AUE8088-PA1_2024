from torchmetrics import Metric
import torch
#HISTORY


# Author : JaeminSong
# Date : 2024-04-30
# Description : MyF1Score

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

# [TODO] Implement this!
from torchmetrics import Metric
import torch

from torchmetrics import Metric
import torch

from torchmetrics import Metric
import torch

# class MyF1Score(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')  # True Positive
#         self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')  # False Positive
#         self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')  # False Negative

#     def update(self, preds, target):
#         preds = torch.argmax(preds, dim=1)  # 최대 확률을 가진 클래스로 예측
#         self.tp += torch.sum((preds == target) & (preds == 1))
#         self.fp += torch.sum((preds != target) & (preds == 1))
#         self.fn += torch.sum((preds != target) & (target == 1))

#     def compute(self):
#         precision = self.tp / (self.tp + self.fp)
#         recall = self.tp / (self.tp + self.fn)
#         return 2 * (precision * recall) / (precision + recall)  # F1 Score 계산

import torch
from torchmetrics import Metric

class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')  # 각 클래스에 대한 True Positive
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')  # 각 클래스에 대한 False Positive
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')  # 각 클래스에 대한 False Negative

    def update(self, preds, target):
        """
        각 클래스에 대한 TP, FP, FN 값을 업데이트합니다.

        Args:
            preds (Tensor): 모델의 출력에서 가장 높은 확률을 가진 클래스 인덱스.
            target (Tensor): 실제 라벨.
        """
        preds = torch.argmax(preds, dim=1)
        for cls in range(self.num_classes):
            true_class = preds == cls
            true_target = target == cls

            self.tp[cls] += torch.sum(true_class & true_target)
            self.fp[cls] += torch.sum(true_class & ~true_target)
            self.fn[cls] += torch.sum(~true_class & true_target)

    def compute(self):
        """
        각 클래스에 대한 F1 스코어를 계산합니다.

        Returns:
            Tensor: 각 클래스의 F1 스코어.
        """
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1_score.mean()
        

# Author : JaeminSong
# Date : 2024-04-26
# Description : def Update

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds,dim=1)
        

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Plz check Shape (Prediction - Target)"

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
