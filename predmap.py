import os
import sys
import logging
import argparse
import glob
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from natsort import natsorted
from PIL import Image

from dataset_wsi import WSI
from model import build_model
from openslide_wsi import OpenSlideWSI

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class makePredmap(object):
    def __init__(
        self,
        wsi_name: str,
        classes: list,
        wsi_dir: str,
        overlaid_mask_dir: str
    ):
        self.wsi_name = wsi_name
        self.classes = classes

        self.wsi_dir = wsi_dir
        self.wsi_path = f"{self.wsi_dir}{self.wsi_name}.ndpi"

        # rgbのマスク画像
        self.overlaid_mask_dir = overlaid_mask_dir

        self.default_level = 5  # defaultの倍率
        self.level = 0  # patchの倍率
        self.length = 256  # patchのサイズ（一方向のpixel数）
        self.resized_size = (
            int(self.length / 2 ** (self.default_level - self.level)),
            int(self.length / 2 ** (self.default_level - self.level))
        )
        self.size = (self.length, self.length)
        self.stride = 256

    # 出力用のdirectory作成
    def make_output_dir(self, main_dir, wsi_name):
        output_dir = f"{main_dir}{wsi_name}/"
        os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    def get_wsi_name(self):
        return self.wsi_name

    # 各クラスの番号と色の対応
    def num_to_color(self, num):
        if isinstance(num, list):
            num = num[0]

        if num == 0:
            color = (200, 200, 200)
        elif num == 1:
            color = (255, 0, 0)
        elif num == 2:
            color = (255, 255, 0)
        elif num == 3:
            color = (0, 255, 0)
        elif num == 4:
            color = (0, 255, 255)
        elif num == 5:
            color = (0, 0, 255)
        elif num == 6:
            color = (255, 0, 255)
        elif num == 7:
            color = (128, 0, 0)
        elif num == 8:
            color = (128, 128, 0)
        elif num == 9:
            color = (0, 128, 0)
        elif num == 10:
            color = (0, 0, 128)
        elif num == 11:
            color = (64, 64, 64)
        else:
            sys.exit("invalid number:" + str(num))
        return color

    # 予測したパッチの着色
    def color_patch(self, y_classes, test_data_list, output_dir):
        for y, test_data in zip(y_classes, test_data_list):
            y = y.argmax(dim=0).numpy().copy()  # yはargmax前のsoftmax出力
            filename, _ = os.path.splitext(os.path.basename(test_data))
            canvas = np.zeros((256, 256, 3))
            for cl in range(len(self.classes)):
                canvas[y == cl] = self.num_to_color(self.classes[cl])
            canvas = Image.fromarray(np.uint8(canvas))
            canvas.save(output_dir + filename + ".png", "PNG", quality=100, optimize=True)

    # 予測したパッチをlikelihood-map用に着色
    def color_likelihood_patch(self, y_classes, test_data_list, output_dir):
        for y, test_data in zip(y_classes, test_data_list):
            y = y.numpy().copy()
            filename, _ = os.path.splitext(os.path.basename(test_data))
            for cl in range(len(self.classes)):
                color = self.num_to_color(self.classes[cl])
                patch_color = np.uint8(np.multiply(y[cl], color))
                canvas = np.full((256, 256, 3), patch_color)
                canvas = Image.fromarray(canvas)
                canvas.save(f"{output_dir}{filename}_cl{cl}.png", "PNG", quality=100, optimize=True)

    # パッチの結合
    def merge_patch(self, patch_dir, output_dir, suffix=None):
        img = OpenSlideWSI(self.wsi_path)
        img.patch_to_image(
            self.resized_size,
            self.level,
            self.size,
            self.stride,
            input_dir=patch_dir,
            output_dir=output_dir,
            output_name=self.wsi_name,
            suffix=suffix,
            cnt=0
        )

    # 背景&対象外領域をマスク
    def make_black_mask(self, input_dir, output_dir, suffix=None):
        if suffix is None:
            filename = self.wsi_name
        else:
            filename = self.wsi_name + suffix

        image = Image.open(
            input_dir + filename + ".png"
        )
        image_gt = Image.open(self.overlaid_mask_dir + self.wsi_name + "_overlaid.tif")

        WIDTH = image.size[0]
        HEIGHT = image.size[1]

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if image_gt.getpixel((x, y)) == (0, 0, 0):
                    image.putpixel((x, y), (0, 0, 0))
                elif image_gt.getpixel((x, y)) == (255, 255, 255):
                    image.putpixel((x, y), (255, 255, 255))

        image.save(
            output_dir + filename + ".png",
            "PNG",
            quality=100,
            optimize=True,
        )


def main():
    # ============ config ============ #
    parser = argparse.ArgumentParser(description='predmap')
    parser.add_argument('--title', type=str, default="predmap", metavar='N',
                        help='title of this project')
    parser.add_argument('--classes', type=list, default=[0, 1], metavar='N',
                        help='classes')
    parser.add_argument('--model_name', type=str, default="resnet50", metavar='N',
                        help='name of model name')
    parser.add_argument('--weight_path', type=str, default="/AAA.pth", metavar='N',
                        help='weight path of trained model')
    parser.add_argument('--is_likelihood', type=bool, default=False, metavar='N',
                        help='if you wanna make likelihood map, turn True')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--shape', type=tuple, default=(256, 256), metavar='N',
                        help='input-shape of patch img')
    args = parser.parse_args()

    # ================================ #

    # =============  各directoryのpath  ===================== #
    MAIN_DIR = "/mnt/ssdsub1/strage/"

    WSI_DIR = MAIN_DIR + "src/MF0012/origin/"  # wsi(ndpi)のあるdirectory
    MASK_DIR = MAIN_DIR + f"src/MF0012/mask_cancergrade/overlaid_{args.classes}/"  # maskのあるdirectory(rgb)

    PATCH_DIR = MAIN_DIR + "patch/MF0012/"  # 切り取ったpatchのあるdirectory
    PRED_DIR = MAIN_DIR + "pred_patch/MF0012/"  # 予測結果を着色したpatchを保存するdirectory
    OUTPUT_DIR = MAIN_DIR + "/output/MF0012/"  #  着色パッチを結合し，マスクした後のdirectory
    # ========================================================== #

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # predmapを作成するwsi名が入ったリスト
    # wsui (ndpi) の入ったdirectory等からwsi名を取得する
    test_wsis = []  # 要変更! (Ex. [0001_a-1, 0001_a-2, ...])

    # model
    net = build_model(
        args.model_name,
        num_classes=len(args.classes)
    )

    # 学習済みの重みをload
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(
        torch.load(args.weight_path, map_location=device))
    logging.info("Loading model {}".format(args.weight_path))

    net.eval()
    for wsi in test_wsis:
        logging.info(f"== {wsi} ==")
        PMAP = makePredmap(
            wsi,
            args.classes,
            wsi_dir=WSI_DIR,
            overlaid_mask_dir=MASK_DIR
        )

        # predmapを作成するwsiの全patchのpathリスト
        patch_list = natsorted(
            glob.glob(PATCH_DIR + f"/{wsi}/*.png", recursive=False))

        # pytorch用のDatasetを構築
        test_data = WSI(
            patch_list,
            args.classes,
            tuple(args.shape),
            transform={'Resize': True, 'HFlip': False, 'VFlip': False},
            is_pred=True
        )

        loader = DataLoader(
            test_data, batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)

        n_val = len(loader)  # the number of batch

        all_preds = []
        logging.info("predict class...")
        with tqdm(total=n_val, desc='predmap', unit='batch', leave=False) as pbar:
            for batch in loader:
                imgs = batch['image']
                imgs = imgs.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    preds = net(imgs)
                preds = nn.Softmax(dim=1)(preds).to('cpu').detach()
                all_preds.extend(preds)

                pbar.update()

        # 予測結果の着色パッチを作成
        logging.info("make color patch...")
        pred_out_dir = PMAP.make_output_dir(PRED_DIR, wsi)
        PMAP.color_patch(all_preds, patch_list, pred_out_dir)

        # 着色パッチを結合
        logging.info("merge color patch...")
        PMAP.merge_patch(pred_out_dir, OUTPUT_DIR)

        # 背景&対象外領域をマスク
        logging.info("mask bg & other classes area...")
        PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR)

        # likelihood mapの作成
        if args.is_likelihood:
            # 予測結果の着色パッチを作成
            logging.info("make color patch (likelihood)...")
            pred_out_dir = PMAP.make_output_dir(PRED_DIR, wsi)
            PMAP.color_likelihood_patch(all_preds, patch_list, pred_out_dir)

            for cl in range(len(args.classes)):
                # 着色パッチを結合
                logging.info("merge color patch (likelihood)...")
                PMAP.merge_patch(pred_out_dir, OUTPUT_DIR, suffix=f"_cl{cl}")

                # 背景&対象外領域をマスク
                logging.info("mask bg & other classes area (likelihood)...")
                PMAP.make_black_mask(OUTPUT_DIR, OUTPUT_DIR, suffix=f"_cl{cl}")


if __name__ == "__main__":
    main()
