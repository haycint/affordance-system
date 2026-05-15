"""
Populate the image memory database from PIAD (Seen / Unseen) datasets.

For every image listed in Img_Train.txt / Img_Test.txt under ./Data/Seen and
./Data/Unseen, extract its Img_Encoder feature (using the MyNet / IAG_TextEmb
checkpoint in ./model_list/) and store it in the image memory store, indexed
by (object_category, affordance_label). At most 4 images per index.

Point clouds are NOT inserted -- this script only fills the image memory.

Usage
-----
    python populate_image_memory.py \
        --setting Seen \
        --split both \
        --ckpt ./model_list/iag_seen.pt \
        --store_dir ./image_memory_store_seen

Or run with defaults to populate both Seen and Unseen sequentially.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_system import ImageMemoryManager  # noqa: E402
from model.MyNet import MyNet  # noqa: E402


AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab',
]


class ImageOnlyPIAD(Dataset):
    """Lightweight loader: reads Img_*.txt / Box_*.txt only, no point clouds.

    Returns the cropped+resized image tensor plus parsed (object, affordance)
    labels and bounding boxes -- everything needed to write an image-memory
    entry, and nothing more.
    """

    def __init__(self, img_list_path: str, box_list_path: str,
                 img_size: Tuple[int, int] = (224, 224)):
        self.img_files = self._read_file(img_list_path)
        self.box_files = self._read_file(box_list_path)
        assert len(self.img_files) == len(self.box_files), \
            f"img/box count mismatch: {len(self.img_files)} vs {len(self.box_files)}"
        self.img_size = img_size
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _read_file(path: str) -> List[str]:
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _parse_labels(img_path: str) -> Tuple[str, str]:
        parts = os.path.basename(img_path).split('_')
        object_name = parts[-3]
        affordance = parts[-2]
        return object_name, affordance

    @staticmethod
    def _get_boxes(json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        sub_points, obj_points = [], []
        for box in data['shapes']:
            if box['label'] == 'subject':
                sub_points = box['points']
            elif box['label'] == 'object':
                obj_points = box['points']
        if len(sub_points) == 0:
            sub_points = [[0., 0.], [0., 0.]]
        sub_points = [*sub_points[0], *sub_points[1]]
        obj_points = [*obj_points[0], *obj_points[1]]
        return (np.array(sub_points, dtype=np.float32),
                np.array(obj_points, dtype=np.float32))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        box_path = self.box_files[index]

        try:
            img_pil = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError) as e:
            print(f"[WARN] Skipping unreadable image {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self))

        sub_box, obj_box = self._get_boxes(box_path)

        w0, h0 = img_pil.size
        img_resized = img_pil.resize(self.img_size)
        scale_w = self.img_size[1] / w0
        scale_h = self.img_size[0] / h0

        sub_box_scaled = sub_box.copy()
        sub_box_scaled[0::2] *= scale_w
        sub_box_scaled[1::2] *= scale_h
        obj_box_scaled = obj_box.copy()
        obj_box_scaled[0::2] *= scale_w
        obj_box_scaled[1::2] *= scale_h

        img_tensor = self.normalize(img_resized)

        object_name, affordance = self._parse_labels(img_path)

        img_np_uint8 = np.array(img_resized, dtype=np.uint8)

        return {
            'img_tensor': img_tensor,
            'img_np': img_np_uint8,
            'sub_box': torch.from_numpy(sub_box_scaled).float(),
            'obj_box': torch.from_numpy(obj_box_scaled).float(),
            'object': object_name,
            'affordance': affordance,
            'img_path': img_path,
        }


def collate_keep_dict(batch):
    return batch


@torch.no_grad()
def populate(setting: str, splits: List[str], ckpt_path: str,
             data_root: str, store_dir: str, device: str,
             max_per_key: int = 4, feature_dim: int = 512):
    print(f"\n=== Populating image memory: setting={setting}, splits={splits} ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Store dir:  {store_dir}")

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = MyNet(pre_train=False)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[WARN] Checkpoint not found, using ImageNet-pretrained Img_Encoder only.")
    model.eval().to(dev)

    manager = ImageMemoryManager(
        store_dir=store_dir,
        feature_dim=feature_dim,
        use_faiss=True,
        max_images_per_key=max_per_key,
        max_memory_images=max_per_key,
    )

    key_counts = {}

    for split in splits:
        img_txt = os.path.join(data_root, setting, f"Img_{split}.txt")
        box_txt = os.path.join(data_root, setting, f"Box_{split}.txt")
        if not (os.path.exists(img_txt) and os.path.exists(box_txt)):
            print(f"[WARN] Missing list files for {setting}/{split}, skipping.")
            continue

        dataset = ImageOnlyPIAD(img_txt, box_txt)
        loader = DataLoader(dataset, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn=collate_keep_dict)

        print(f"[{setting}/{split}] {len(dataset)} images")

        added = 0
        skipped = 0
        for i, batch in enumerate(loader):
            sample = batch[0]
            object_name = sample['object']
            affordance = sample['affordance']
            key = (object_name, affordance)

            if key_counts.get(key, 0) >= max_per_key:
                skipped += 1
                continue

            img_t = sample['img_tensor'].unsqueeze(0).to(dev)
            F_I = model.img_encoder(img_t)  # [1, C, h, w]
            feat_np = F_I.cpu().numpy().squeeze(0)  # [C, h, w]

            try:
                manager.store_image(
                    image=sample['img_np'],
                    image_feature=feat_np,
                    object_category=object_name,
                    affordance_label=affordance,
                    sub_box=sample['sub_box'].numpy(),
                    obj_box=sample['obj_box'].numpy(),
                    confidence=1.0,
                )
                key_counts[key] = key_counts.get(key, 0) + 1
                added += 1
            except Exception as e:
                print(f"[ERROR] {sample['img_path']}: {e}")

            if (i + 1) % 500 == 0:
                print(f"  ... processed {i + 1}/{len(dataset)} "
                      f"(added={added}, skipped={skipped}, keys={len(key_counts)})")

        print(f"[{setting}/{split}] done: added={added}, skipped={skipped}")

    manager.save()
    stats = manager.get_stats()
    print(f"\n=== Done: {setting} ===")
    print(f"Total images stored: {stats['total_images']}")
    print(f"Unique (object, affordance) keys: {len(stats['categories'])}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', choices=['Seen', 'Unseen', 'both'],
                        default='both')
    parser.add_argument('--split', choices=['Train', 'Test', 'both'],
                        default='both',
                        help='Which list(s) to enumerate.')
    parser.add_argument('--data_root', default='./Data')
    parser.add_argument('--ckpt_seen', default='./model_list/iag_seen.pt')
    parser.add_argument('--ckpt_unseen', default='./model_list/iag_seen.pt',
                        help='No Unseen ckpt provided; reuse the Seen one '
                             '(Img_Encoder is a generic ResNet18 backbone).')
    parser.add_argument('--store_dir_seen',
                        default='./image_memory_store_seen')
    parser.add_argument('--store_dir_unseen',
                        default='./image_memory_store_unseen')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_per_key', type=int, default=4)
    args = parser.parse_args()

    splits = ['Train', 'Test'] if args.split == 'both' else [args.split]
    settings = (['Seen', 'Unseen'] if args.setting == 'both'
                else [args.setting])

    for s in settings:
        populate(
            setting=s,
            splits=splits,
            ckpt_path=args.ckpt_seen if s == 'Seen' else args.ckpt_unseen,
            data_root=args.data_root,
            store_dir=(args.store_dir_seen if s == 'Seen'
                       else args.store_dir_unseen),
            device=args.device,
            max_per_key=args.max_per_key,
        )


if __name__ == '__main__':
    main()
