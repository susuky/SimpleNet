
import argparse
import cv2
import numpy as np
import os
import torch

from pathlib import Path
from tqdm import tqdm

from dataset import MVTecADDataset
from modules import ComputeLoss, SimpleNet, save_model
from utils import (
    seed_everything, 
    MetricMonitor, 
    image_tonumpy, 
    compute_metrics, 
    normalize,
)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18', help='model name')
    parser.add_argument('--root', type=str, default='mvtecad_dataset', help='dataset root')
    parser.add_argument('--category', type=str, default='bottle', help='category')
    parser.add_argument('--model-path', type=str, default=f'./models/model.pth', help='model path')
    parser.add_argument('--img-size', '--imgsz', nargs='+', type=int, default=[256, 256], help='inference size h,w')
    parser.add_argument('--bs', '--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='global random seed')
    opt = parser.parse_args()
    return opt


def evaluate(model, dl, device, epoch=None, draw=False, track=False, **kwargs):
    metric_monitor = MetricMonitor()
    model.eval()

    if draw:
        category = kwargs.get('category', 'bottle')
        parent = os.path.join('run', category, f'{epoch}')
        if not os.path.exists(parent):
            os.makedirs(parent)

    if track:
        model.reset_minmax()
        for xs, ys, names in dl:
            xs = xs.to(device)
            anomaly_maps, image_scores = model(xs, track=True)

    scores = []
    labels = []
    print(f'score range: {model.max:.4f}, {model.min:.4f}')
    for i, (xs, ys, names) in enumerate(dl, 1):
        xs = xs.to(device)
        labels.extend(ys.numpy().tolist())

        anomaly_maps, image_scores = model(xs)
        scores.extend(image_scores.cpu().numpy().tolist())

        if draw:
            for img, y, path, anomaly_map in zip(xs, ys, names, anomaly_maps):
                label = 'good' if y == 0 else 'anomal'
                name = f'{Path(path).stem}-{label}'
                img = image_tonumpy(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(parent, f'{name}.jpg'), img)

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_map = normalize(anomaly_map, model.min, model.max)
                anomaly_map = ~ (anomaly_map * 255).astype(np.uint8)
                #anomaly_map = (anomaly_map * 255).astype(np.uint8)
                anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(parent, f'{name}-heatmap.jpg'), anomaly_map)

    metrics = compute_metrics(np.array(scores), np.array(labels), 0.0)
    metric_monitor.update_dict(metrics)
    model.threshold = metrics['threshold']
    print(metric_monitor)

    return metric_monitor


def one_epoch(model, criterion, optimizers, dl, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()

    stream = tqdm(dl)
    for i, (xs, *_) in enumerate(stream, 1):
        xs = xs.to(device)
        
        feats = model.backbone(xs)
        embedding = model.adaptor(feats)
        loss, p_ture, p_fake = criterion(embedding, model.generator, model.discriminator)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('p_true', p_ture.item())
        metric_monitor.update('p_fake', p_fake.item())

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
            stream.set_description(
                f'Epoch: {epoch}. Train.      {metric_monitor}'
            )


def train(opt=parse_opt()):
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = MVTecADDataset(opt.root, opt.category, 'train', opt.img_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=opt.bs, num_workers=4, shuffle=True)
    test_ds = MVTecADDataset(opt.root, opt.category, 'test', opt.img_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False)

    model = SimpleNet(opt.backbone, pretrained=True).to(device)

    criterion = ComputeLoss()
    optimizers = [
        torch.optim.Adam(model.adaptor.parameters(), lr=1e-4, weight_decay=1e-5),
        torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, weight_decay=1e-5),
    ]

    for epoch in range(1, opt.epochs+1):
        draw = (epoch-1) % 10 == 0
        one_epoch(model, criterion, optimizers, train_dl, epoch, device)
        evaluate(model, test_dl, device, epoch, draw=draw, track=True, category=opt.category)

    evaluate(model, test_dl, device, epoch='last', draw=True, track=False, category=opt.category)

    model_path = f'models/{opt.category}.pth'
    #save_model(model, path=model_path)
    


if __name__ == '__main__':
    train()