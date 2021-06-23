from PIL import Image
import numpy as np
import torch
from torchvision import transforms


class WSI(torch.utils.data.Dataset):
    def __init__(
        self,
        file_list: list,
        classes: list = [0, 1, 2, 3],
        shape: tuple = None,
        transform=None,
        is_pred: bool = False
    ):
        self.file_list = file_list
        self.classes = classes  # class番号が入ったリスト
        self.shape = shape
        self.transform = transform
        self.is_pred = is_pred  # predmap作成時はTrue

    def __len__(self):
        return len(self.file_list)

    # pathからlabelを取得
    def get_label(self, path):
        def check_path(cl, path):
            if f"/{cl}/" in path:
                return True
            else:
                return False

        for idx in range(len(self.classes)):
            cl = self.classes[idx]

            if isinstance(cl, list):
                for sub_cl in cl:
                    if check_path(sub_cl, path):
                        label = idx
            else:
                if check_path(cl, path):
                    label = idx
        assert label is not None, "label is not included in {path}"
        return np.array(label)

    # 画像の前処理
    def preprocess(self, img_pil):
        if self.transform is not None:
            if self.transform['Resize']:
                img_pil = transforms.Resize(
                    self.shape, interpolation=0
                )(img_pil)
            if self.transform['HFlip']:
                img_pil = transforms.RandomHorizontalFlip(0.5)(img_pil)
            if self.transform['VFlip']:
                img_pil = transforms.RandomVerticalFlip(0.5)(img_pil)
        return np.asarray(img_pil)

    def transpose(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # For rgb or grayscale image
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        img_file = self.file_list[i]
        img_pil = Image.open(img_file)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        img = self.preprocess(img_pil)
        img = self.transpose(img)

        if self.is_pred:
            item = {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'name': img_file
            }
        else:
            label = self.get_label(img_file)
            item = {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'label': torch.from_numpy(label).type(torch.long),
                'name': img_file
            }

        return item
