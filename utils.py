
import cv2
import numpy as np
import os
import random
import sklearn.metrics
import sys

from collections import defaultdict, OrderedDict


def get_logger(filename='log', level='INFO', save=False, verbose=True, stream=sys.stderr):
    from logging import getLogger, StreamHandler, FileHandler, Formatter
    levels = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    logger = getLogger(filename)
    logger.setLevel(levels.get(level.upper(), 20))
    if verbose:
        handler1 = StreamHandler(stream=stream)
        handler1.setFormatter(Formatter("%(message)s"))
        handler1.terminator = '\n' 
        logger.addHandler(handler1)

    if save:
        handler2 = FileHandler(filename=filename)
        handler2.setFormatter(Formatter("%(message)s"))
        handler2.terminator = '\n'
        logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed) # cpu
        if torch.cuda.is_available(): # cuda
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # cudnn seed 0 settings are slower and more reproducible,
            # else faster and less reproducible
            cudnn = torch.backends.cudnn
            cudnn.benchmark, cudnn.deterministic = (
                (False, True) if seed == 0 else (True, False)
            )


def compute_metrics(outputs: np.ndarray, 
                    targets: np.ndarray, 
                    threshold: float = 0.1) -> dict:
    
    prefix = 'I-' if outputs.ndim == 1 else 'P-'
    outputs = outputs.ravel()
    targets = targets.ravel()

    metrics = {}
    metrics[f'{prefix}AUROC'] = sklearn.metrics.roc_auc_score(targets, outputs)
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(targets, outputs)
    metrics[f'{prefix}PRAUC'] = sklearn.metrics.auc(recalls, precisions)
    if prefix == 'I-':
        if threshold is None:
            # Note: This threshold may not be within 0 to 1
            f1_scores =  (2 * precisions * recalls) / (precisions + recalls + 1e-10)
            threshold = thresholds[f1_scores.argmax()]

        preds = (outputs > threshold).astype(np.int32)
        metrics[f'F1score'] = sklearn.metrics.f1_score(targets, preds)
        metrics[f'Accuracy'] = sklearn.metrics.accuracy_score(targets, preds)
        metrics['threshold'] = threshold
    return metrics


class AverageMeter:
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count
    

class MetricMonitor:
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(AverageMeter)

    def update(self, metric_name, val):
        self.metrics[metric_name].update(val)

    def update_dict(self, metric_dict):
        for metric_name, metric in metric_dict.items():
            self.metrics[metric_name].update(metric)

    def get_metrics(self):
        metrics = {
            metric_name: metric.avg 
            for metric_name, metric in self.metrics.items()
        }
        return metrics

    def __str__(self):
        return ' | '.join([
            f'{metric_name}: {metric:.{self.float_precision}f}' 
            for (metric_name, metric) in self.get_metrics().items()
        ])
    

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.is_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.best_score - score < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.is_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ModelCheckpoint:
    def __init__(self, min_delta=0.001):
        self.best_score = None
        self.best_epoch = 0
        self.min_delta = min_delta
        self.state_dict = None

    def step(self, score, epoch, model):
        if self.best_score is None or score - self.best_score > self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.state_dict = self.get_state_dict(model)

    def get_state_dict(self, model):
        state_dict = {'model': OrderedDict(), 'metadata': {}}
        for k, v in model.state_dict().items():
            state_dict['model'][k] = v.cpu()

        state_dict['metadata']['max'] = model.max
        state_dict['metadata']['min'] = model.min
        return state_dict
    

def compute_anomaly_map(patch_score, sigma=4, image_size=None):
    '''
    pathch_scores: (H, W)
    '''
    kernel_size = 2 * sigma * 4 + 1

    if image_size is not None:
        patch_score = cv2.resize(patch_score, image_size)
    blur = cv2.GaussianBlur(patch_score, (kernel_size, kernel_size), sigma)
    return blur


def image_tonumpy(img, 
                  mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225],
                  scale=255.0):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = img.clip(0, 1) * scale
    return img.astype(np.uint8)


def normalize(x, min, max):
    return (x - min) / (max - min)

