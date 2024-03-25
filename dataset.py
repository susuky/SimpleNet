
import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


DEBUG = __name__ == '__main__'


def get_transforms(phase='train', img_size=(312, 312)):
    # remove center crop
    #r = 224/256
    #crop_size = [int(s * r) for s in img_size]
    
    if phase == 'train':
        # aug = [
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        # ]
        aug = []
    else:
        aug = []

    return A.Compose([
        A.Resize(*img_size),
        #A.CenterCrop(*crop_size),
        *aug,
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])



class MVTecADDataset(Dataset):
    def __init__(self, 
                 root='mvtecad_dataset', 
                 category='bottle',
                 phase='train', 
                 with_mask=False,
                 img_size=(312, 312)):
        self.root = root
        self.category = category
        self.phase = phase
        self.img_size = img_size
        self.transforms = get_transforms(phase=phase, img_size=img_size)
        self.data_path = Path(root, category, phase)
        self.with_mask = with_mask and phase != 'train'
        if self.with_mask:
            self.mask_path = Path(root, category, 'ground_truth')
        self.load_data()

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        path = self.paths[index]
        mask = self.masks[index]

        x = self.transforms(image=x)['image']
        return x, y, path, mask

    def __len__(self):
        return len(self.ys)

    def load_data(self):

        xs = []
        ys = []
        paths = []
        masks = []

        for path in self.data_path.rglob('*.png'):
            label = path.parent.name

            mask = np.zeros(0, np.uint8)
            if self.with_mask:
                '''
                For example,
                `mvtecad_dataset/bottle/test/contamination/016.png`
                to
                `mvtecad_dataset/bottle/ground_truth/contamination/016_mask.png`
                '''
                mask_path = Path(self.mask_path, label, f'{path.stem}_mask.png')
                if mask_path.exists():
                    mask = cv2.imread(mask_path.as_posix(), flags=0)
                    mask = cv2.resize(mask, self.img_size)
                else:
                    mask = np.zeros_like(self.img_size, np.uint8)
                    assert mask.shape == self.img_size
                    
            x = cv2.imread(path.as_posix())
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            y = int(label != 'good')
            path = path.as_posix()

            xs.append(x)
            ys.append(y)
            paths.append(path)
            masks.append(mask)

        self.xs = xs
        self.ys = ys
        self.paths = paths
        self.masks = masks


if DEBUG:
    dataset = MVTecADDataset(category='cable')
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    for x, y, path, mask in loader:
        print(x.shape, y, path, mask)

    dataset = MVTecADDataset(phase='test', with_mask=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for x, y, path, mask in loader:
        print(x.shape, y, path, mask.shape)

