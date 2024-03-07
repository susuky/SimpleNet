
import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


DEBUG = __name__ == '__main__'

def get_transforms(phase='train', img_size=(312, 312)):

    if phase == 'train':
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    else:
        aug = []

    return A.Compose([
        A.Resize(*img_size),
        *aug,
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class MVTecADDataset(Dataset):
    def __init__(self, 
                 root='mvtecad_dataset', 
                 category='bottle',
                 phase='train', 
                 img_size=(312, 312)):
        self.root = root
        self.category = category
        self.phase = phase
        self.img_size = img_size
        self.transforms = get_transforms(phase=phase, img_size=img_size)
        self.data_path = Path(root, category, phase)

        self.load_data()

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        name = self.names[index]

        x = self.transforms(image=x)['image']

        return x, y, name

    def __len__(self):
        return len(self.ys)

    def load_data(self):
        xs = []
        ys = []
        names = []
        for path in self.data_path.rglob('*.png'):
            x = cv2.imread(path.as_posix())
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            y = int(path.parent.name == 'good')
            name = path.as_posix()

            xs.append(x)
            ys.append(y)
            names.append(name)

        self.xs = xs
        self.ys = ys
        self.names = names


if DEBUG:
    dataset = MVTecADDataset()
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    for x, y, name in loader:
        print(x.shape, y, name)

    dataset = MVTecADDataset(phase='test')
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for x, y, name in loader:
        print(x.shape, y, name)

