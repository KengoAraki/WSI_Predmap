
# WSIの予測画像作成用

## Framework
```
pytorch
```

## Directoryの構成
```
# 元画像やマスク画像のあるdirectory
./src/
└── MF0012
    ├── mask_bg
    │      ├── 0001_a-1_mask_level05_bg.tif
    │      ├── 0001_a-2_mask_level05_bg.tif
    │      ├── 0002_a-1_mask_level05_bg.tif
    ├── mask_cancergrade
    │   ├── overlaid_[0, 1, 2]
    │   │      ├── 0001_a-1_overlaid.tif
    │   │      ├── 0001_a-2_overlaid.tif
    │   │      ├── 0002_a-1_overlaid.tif
    │   └── overlaid_[2, [1, 3]]
    ├── mask_cancergrade_gray
    │   ├── overlaid_[0, 1, 2]
    │   └── overlaid_[2, [1, 3]]
    └── origin
            ├── 0001_a-1.ndpi
            ├── 0001_a-2.ndpi
            ├── 0002_a-1.ndpi


# WSI(ndpi)から切り取ったpatch
./patch
└── MF0012
    ├── 0001_a-1
    │      ├── 0_0000000001.png
    │      ├── 0_0000000002.png
    │      ├── 0_0000000003.png
    ├── 0001_a-2
    ├── 0002_a-1

# 予測クラスの色に着色したpatch
./pred_patch
└── MF0012
    ├── 0001_a-1
            ├── 0_0000000001.png
            ├── 0_0000000002.png
            ├── 0_0000000003.png
    ├── 0001_a-2
    ├── 0002_a-1

# predmap 出力用
./output
└── MF0012
    ├── 0001_a-1_predmap.png
    ├── 0001_a-2_predmap.png
    ├── 0002_a-1_predmap.png

```