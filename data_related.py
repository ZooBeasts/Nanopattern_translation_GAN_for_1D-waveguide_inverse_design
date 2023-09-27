from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from os.path import join
import os
from PIL import Image


class DiyDataset(Dataset):
    def __init__(self, root_dir, transform, x_folder='inputs', y_folder='labels'):
        self.root_dir = root_dir
        self.data_list = os.listdir(join(root_dir, x_folder))
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(join(self.root_dir, self.x_folder, self.data_list[index])).convert('RGB')
        y = Image.open(join(self.root_dir, self.y_folder, self.data_list[index])).convert('RGB')
        x, y = self.transform(x), self.transform(y)
        return x, y

    def __len__(self):
        return len(self.data_list)


def get_loader(root_dir='circle_images', x_folder='inputs', y_folder='labels', transform=None,
               batch_size=128, shuffle=True):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],[0.5, 0.5, 0.5]),

        ])
    return DataLoader(
        DiyDataset(root_dir, transform, x_folder, y_folder),
        batch_size=batch_size,
        shuffle=shuffle,
    )


if __name__ == '__main__':
    loader = get_loader(batch_size=1)
    for x, y in loader:
        break
