
import argparse
import torch

from tqdm import tqdm

from dataset import MVTecADDataset
from modules import ComputeLoss, SimpleNet
from utils import seed_everything, MetricMonitor


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='resnet18', help='model name')
    parser.add_argument('--root', type=str, default='mvtecad_dataset', help='root')
    parser.add_argument('--category', type=str, default='bottle', help='category')
    parser.add_argument('--model-path', type=str, default=f'.models/model.pth', help='model path')
    parser.add_argument('--img-size', '--imgsz', nargs='+', type=int, default=[312, 312], help='inference size h,w')
    parser.add_argument('--bs', '--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='global random seed')
    opt = parser.parse_args()
    return opt


def evaluate(model, dl, device, draw=False, epoch=None):
    metric_monitor = MetricMonitor()
    model.eval()

    scores = []
    labels = []
    stream = tqdm(dl)
    for i, (xs, ys, names) in enumerate(stream, 1):
        xs = xs.to(device)
        labels.extend(ys.numpy().tolist())

        anomaly_map, image_scores = model(xs)
        scores.extend(image_scores.cpu().numpy().tolist())

        metric_monitor.update('Loss', )
        stream.set_description(
            f'Epoch: {epoch}. Validation.      {metric_monitor}'
        )
        if draw:
            # TODO: save image
            ...


def one_epoch(model, criterion, optimizers, dl, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()

    stream = tqdm(dl)
    for i, (xs, *_) in enumerate(stream, 1):
        xs = xs.to(device)
        
        feats = model.backbone(xs)
        embedding = model.adaptor(feats)
        loss = criterion(embedding, model.generator, model.discriminator)
        metric_monitor.update('Loss', loss.item())

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
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=opt.bs, num_workers=4, shuffle=False)

    model = SimpleNet(opt.model_name, pretrained=True).to(device)

    criterion = ComputeLoss()
    optimizers = [
        torch.optim.Adam(model.adaptor.parameters(), lr=1e-4, weight_decay=1e-5),
        torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, weight_decay=1e-5),
    ]

    for epoch in range(1, opt.epochs+1):
        one_epoch(model, criterion, optimizers, train_dl, epoch, device)

        if epoch % 10 == 0:
            evaluate(model, test_dl, device, draw=True)
    torch.save(model.state_dict(), opt.model_path)


if __name__ == '__main__':
    train()