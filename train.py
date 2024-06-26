
import argparse
import cv2
import numpy as np
import os
import torch

from pathlib import Path
from tqdm import tqdm

from dataset import MVTecADDataset
from modules import (
    ComputeLoss,
    Optimizers,
    SimpleNet,
    save_model,
)
from utils import (
    MetricMonitor,
    ModelCheckpoint,
    compute_metrics,
    get_optimal_threshold,
    plot_evaluate_results,
    seed_everything,
)


def parse_opt():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        default='mvtecad_dataset',
                        help='dataset root')
    parser.add_argument('--category',
                        type=str,
                        default='bottle',
                        help='category')
    parser.add_argument('--model-path',
                        type=str,
                        default=f'./models/model.pth',
                        help='model path')
    parser.add_argument('--img-size', '--imgsz', '--img-sz', '--image-size',
                        nargs='+',
                        type=int,
                        default=[256, 256],
                        help='inference size h,w')
    parser.add_argument('--bs', '--batch-size',
                        type=int,
                        default=4,
                        help='batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=160,
                        help='number of epochs')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='global random seed')
    
    parser.add_argument('--model-name',
                        type=str,
                        default='resnet18',
                        help='model name')
    parser.add_argument('--layers', '--model-layers',
                        nargs='+',
                        type=int,
                        default=[2, 3],
                        help='model layers')
    parser.add_argument('--operation',
                        type=str,
                        default='Conv2d',
                        help='operations. Options: '
                        'Linear, Conv1d, Conv2d')
    parser.add_argument('--generator-method',
                        type=str,
                        default='normalstd',
                        help='generator method. Options: '
                        'uniform, uniformstd, normalstd, constantstd')
    parser.add_argument('--generator-learnable',
                        type=str2bool,
                        default=True,
                        help='generator learnable')
    parser.add_argument('--sigma',
                        default=4.0,
                        type=float,
                        help='anomaly map sigma')
    parser.add_argument('--temperature', '--loss-temperature',
                        default=0.5,
                        type=float,
                        help='temperature')
    opt = parser.parse_args()
    return opt


def evaluate(model,
             dl,
             device,
             epoch=None,
             draw=False,
             track=False,
             threshold=0.5,
             **kwargs):
    metric_monitor = MetricMonitor()
    model.eval()

    if draw:
        category = kwargs.get('category', 'bottle')
        parent = os.path.join('run', category, f'{epoch}')
        if not os.path.exists(parent):
            os.makedirs(parent)

    if track:
        model.reset_minmax()
        for xs, *_ in dl:
            xs = xs.to(device)
            model.track_minmax(xs)

    scores = []
    labels = []
    maps = []
    masks_gt = []
    print(f'score range: {model.max:.4f}, {model.min:.4f}')  # debug
    for i, (xs, ys, paths, masks) in enumerate(dl, 1):
        xs = xs.to(device)
        labels.extend(ys.numpy().tolist())

        anomaly_maps, image_scores = model(xs)
        scores.extend(image_scores.cpu().numpy().tolist())

        if kwargs.get('comput_pixelwise', False):
            masks_gt.append(masks.numpy())
            maps.append(anomaly_maps.cpu().numpy())

        if draw:
            for img, path, anomaly_map, mask in zip(xs, paths, anomaly_maps, masks):
                plot_evaluate_results(img, anomaly_map, mask, path, parent=parent)

    scores = np.array(scores)
    metrics = compute_metrics(scores, np.array(labels), threshold)
    metric_monitor.update_dict(metrics)
    model.threshold = metrics['threshold']

    # calculate pixel level metrics (XXX: it takes time to compute these metrics)
    if kwargs.get('comput_pixelwise', False):
        masks_gt = np.concatenate(masks_gt, axis=0)
        maps = np.concatenate(maps, axis=0)
        pixel_level_metrics = compute_metrics(maps, masks_gt)
        metric_monitor.update_dict(pixel_level_metrics)

    print(metric_monitor)
    return metric_monitor


def train_one_epoch(model, criterion, optimizers, dl, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()

    stream = tqdm(dl)
    for i, (xs, *_) in enumerate(stream, 1):
        # prepare data
        xs = xs.to(device)
        optimizers.zero_grad()

        # forward
        feats = model.backbone(xs)
        embedding = model.adaptor(feats)
        loss, p_ture, p_fake = criterion(embedding,
                                         model.generator,
                                         model.discriminator)

        # update metric
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('p_true', p_ture.item())
        metric_monitor.update('p_fake', p_fake.item())
        for name, param in model.generator.named_parameters():
            name = name[name.rfind('.')+1:]
            metric_monitor.update(name, param.item())

        # update model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizers.step()

        # update stream
        stream.set_description(
            f'Epoch: {epoch}. Train.      {metric_monitor}'
        )
    return metric_monitor


def train(opt=parse_opt()):
    # Callback: before_fit
    seed_everything(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset and dataloader
    train_ds = MVTecADDataset(opt.root, opt.category, 'train', opt.img_size)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=opt.bs,
                                           num_workers=4,
                                           shuffle=True)
    test_ds = MVTecADDataset(opt.root, opt.category, 'test', opt.img_size)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=1,
                                          num_workers=4,
                                          shuffle=False)
    
    # model
    model = SimpleNet(
        opt.model_name,
        pretrained=True,
        layers=opt.layers,
        operation=opt.operation,
        generator_method=opt.generator_method,
        generator_learnable=opt.generator_learnable,
        sigma=opt.sigma,
    ).to(device)
    criterion = ComputeLoss(temperature=opt.temperature)
    optimizers = Optimizers(model, lr=1e-4, wd=1e-5)

    model_checkpoint = ModelCheckpoint()
    best_metrics = {'Accuracy': 0.0}
    for epoch in range(1, opt.epochs+1):
        # Callback: before_epoch
        train_one_epoch(model,
                        criterion,
                        optimizers,
                        train_dl,
                        epoch,
                        device)
        threshold, _ = get_optimal_threshold(model, train_dl, device)
        metric_monitor = evaluate(model,
                                  test_dl,
                                  device,
                                  epoch,
                                  draw=(epoch-1) % 10 == 0,
                                  threshold=threshold,
                                  track=True,
                                  category=opt.category)

        # Callback: after_epoch
        metrics = metric_monitor.get_metrics()
        model_checkpoint.step(metrics['Accuracy'], epoch, model)
        if metrics['Accuracy'] > best_metrics['Accuracy']:
            best_metrics = metrics

    metric_monitor_last = evaluate(model,
                                   test_dl,
                                   device,
                                   epoch='last',
                                   draw=True,
                                   track=False,
                                   category=opt.category)
    last_metrics = metric_monitor_last.get_metrics()

    # Callback: after_fit
    # save model
    save_model(state_dict=model_checkpoint.state_dict,
               model_name=f'{opt.category}-best.pth')
    save_model(model=model, model_name=f'{opt.category}-last.pth')

    # print result
    def _set_precision(d, precision=4):
        return ', '.join(
            [f'{k}: {v:.{precision}f}' for k, v in d.items()]
        )
    print(
        f'best epoch: {model_checkpoint.best_epoch}, best score: {_set_precision(best_metrics)}')
    print(f'last epoch: {epoch}, score: {_set_precision(last_metrics)}')
    return best_metrics, last_metrics


if __name__ == '__main__':
    train()
