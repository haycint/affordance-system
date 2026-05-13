"""
PIAD Dataset for IAGNet
Point-Image Affordance Dataset
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import random
import os


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m
    

def img_normalize_train(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img


def img_normalize_val(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img


class PIAD(Dataset):
    """
    Point-Image Affordance Dataset
    """

    def __init__(self, run_type, setting_type, point_path, img_path, box_path, pair=2, img_size=(224, 224)):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.b_path = box_path
        self.pair_num = pair
        self.img_size = img_size
        '''
        print("=================================================================")
        print("img_path:",self.i_path,"...",img_path)
        print("box_path:",self.b_path,"...",box_path)
        print("point_path:",self.p_path,"...",point_path)
        print("=================================================================")
        '''
        # Affordance labels
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open',
                                      'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                                      'push', 'listen', 'wear', 'press', 'cut', 'stab']

        # Object categories for different settings
        if setting_type == 'Unseen':
            number_dict = {'Knife': 0, 'Refrigerator': 0, 'Earphone': 0,
                           'Bag': 0, 'Keyboard': 0, 'Chair': 0, 'Hat': 0, 'Door': 0, 'TrashCan': 0, 'Table': 0,
                           'Faucet': 0, 'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Display': 0, 'Mug': 0,
                           'Clock': 0}
        else:  # Seen
            number_dict = {'Earphone': 0, 'Bag': 0, 'Chair': 0, 'Refrigerator': 0, 'Knife': 0, 'Dishwasher': 0,
                           'Keyboard': 0, 'Scissors': 0, 'Table': 0,
                           'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Microwave': 0, 'Display': 0, 'TrashCan': 0,
                           'Hat': 0, 'Clock': 0,
                           'Door': 0, 'Mug': 0, 'Faucet': 0, 'Vase': 0, 'Laptop': 0, 'Bed': 0}

        self.img_files = self.read_file(self.i_path)
        self.box_files = self.read_file(self.b_path)

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        box_path = self.box_files[index]

        try:
            Img = Image.open(img_path).convert('RGB')
        except OSError as e:
            print(f"ERROR: 无法读取图片文件: {img_path}")
            print(f"错误信息: {e}")
            # 可以选择跳过这个样本，返回下一个样本
            return self.__getitem__((index + 1) % len(self))

        if self.run_type == 'val':
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('_')[-3]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0], range_[1]), self.pair_num)

        Img = Image.open(img_path).convert('RGB')

        if self.run_type == 'train':
            Img, subject, object_box = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Points_List = []
            affordance_label_List = []
            affordance_index_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                Points, _, _ = pc_normalize(Points)
                Points = Points.transpose()
                affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)
                Points_List.append(Points)
                affordance_label_List.append(affordance_label)
                affordance_index_List.append(affordance_index)

        else:
            subject, object_box = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Point, affordance_label = self.extract_point_file(point_path)
            Point, _, _ = pc_normalize(Point)
            Point = Point.transpose()

            affordance_label, _ = self.get_affordance_label(img_path, affordance_label)

        if self.run_type == 'train':
            return Img, Points_List, affordance_label_List, affordance_index_List, sub_box, obj_box
        else:
            return Img, Point, affordance_label, img_path, point_path, sub_box, obj_box

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict is not None:
                    object_ = file.split('_')[-2]
                    if object_ in number_dict:
                        number_dict[object_] += 1
                file_list.append(file)
            f.close()
        if number_dict is not None:
            return file_list, number_dict
        else:
            return file_list

    def extract_point_file(self, path):
        with open(path, 'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[:, 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str_path, label):
        cut_str = str_path.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)

        label = label[:, index]

        return label, index

    def get_crop(self, json_path, image, run_type):
        json_data = json.load(open(json_path, 'r'))
        sub_points, obj_points = [], []
        for box in json_data['shapes']:
            if box['label'] == 'subject':
                sub_points = box['points']
            elif box['label'] == 'object':
                obj_points = box['points']
        if len(sub_points) == 0:
            temp_box = [0.] * 2
            for i in range(2):
                sub_points.append(temp_box)

        if run_type == 'train':
            crop_img, crop_subpoints, crop_objpoints = self.random_crop_with_points(image, sub_points, obj_points)
            return crop_img, crop_subpoints, crop_objpoints
        else:
            sub_points = [*sub_points[0], *sub_points[1]]
            obj_points = [*obj_points[0], *obj_points[1]]
            sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
            return sub_points, obj_points

    def random_crop_with_points(self, image, sub_points, obj_points):
        points = []
        image = np.array(image)
        for obj_point in obj_points:
            points.append(obj_point)

        for sub_point in sub_points:
            points.append(sub_point)

        h, w = image.shape[0], image.shape[1]
        points = np.array(points, np.int32)
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

        t = random.randint(0, min_y) if min_y > 0 else 0
        b = random.randint(max_y + 1, h) if max_y + 1 < h else max_y + 1
        lft = random.randint(0, min_x) if min_x > 0 else 0
        r = random.randint(max_x + 1, w) if max_x + 1 < w else max_x + 1

        new_img = image[t: b, lft: r, :]

        new_img = Image.fromarray(new_img)
        obj_points_arr = points[0:2]
        new_objpoints = [[x - lft, y - t] for x, y in obj_points_arr]
        obj_LT = new_objpoints[0]
        obj_RB = new_objpoints[1]
        new_objpoints = [*obj_LT, *obj_RB]

        sub_points_arr = points[2:]
        new_subpoints = [[x - lft, y - t] for x, y in sub_points_arr]
        sub_LT = new_subpoints[0]
        sub_RB = new_subpoints[1]
        new_subpoints = [*sub_LT, *sub_RB]

        return new_img, new_subpoints, new_objpoints

    def get_resize_box(self, image, new_size, sub_box, obj_box):
        image = np.array(image)
        h_ = image.shape[0]
        w_ = image.shape[1]

        scale_h = new_size[0] / h_
        scale_w = new_size[1] / w_

        sub_box[0], sub_box[2] = sub_box[0] * scale_w, sub_box[2] * scale_w
        sub_box[1], sub_box[3] = sub_box[1] * scale_h, sub_box[3] * scale_h

        obj_box[0], obj_box[2] = obj_box[0] * scale_w, obj_box[2] * scale_w
        obj_box[1], obj_box[3] = obj_box[1] * scale_h, obj_box[3] * scale_h

        return sub_box, obj_box

"""
PIAD Dataset for IAGNet
Point-Image Affordance Dataset
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import random
import os


def pc_normalize(pc):
    """点云归一化到单位球内"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def pc_jitter(pc, sigma=0.01, clip=0.05):
    """
    点云抖动增强
    对每个点的坐标添加随机高斯噪声
    
    Args:
        pc: 点云坐标 (N, 3)
        sigma: 高斯噪声的标准差
        clip: 噪声的最大绝对值
    
    Returns:
        抖动后的点云 (N, 3)
    """
    jittered_pc = pc + np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return jittered_pc


def pc_rotate(pc, axis='y', angle_range=(-15, 15)):
    """
    点云随机旋转增强
    
    Args:
        pc: 点云坐标 (N, 3)
        axis: 旋转轴 ('x', 'y', 'z' 或 'all')
        angle_range: 旋转角度范围（度）
    
    Returns:
        旋转后的点云 (N, 3)
    """
    def rotation_matrix(axis, angle):
        """生成绕指定轴的旋转矩阵"""
        angle = np.radians(angle)
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
    
    if axis == 'all':
        # 绕三个轴随机旋转
        angles = [random.uniform(*angle_range) for _ in range(3)]
        R = rotation_matrix('x', angles[0]) @ rotation_matrix('y', angles[1]) @ rotation_matrix('z', angles[2])
    else:
        angle = random.uniform(*angle_range)
        R = rotation_matrix(axis, angle)
    
    rotated_pc = pc @ R.T
    return rotated_pc


def pc_scale(pc, scale_range=(0.8, 1.2)):
    """
    点云随机缩放增强
    
    Args:
        pc: 点云坐标 (N, 3)
        scale_range: 缩放因子范围
    
    Returns:
        缩放后的点云 (N, 3)
    """
    scale = random.uniform(*scale_range)
    return pc * scale


def pc_flip(pc, p=0.5):
    """
    点云随机翻转增强
    
    Args:
        pc: 点云坐标 (N, 3)
        p: 翻转概率
    
    Returns:
        翻转后的点云 (N, 3)
    """
    if random.random() < p:
        # 随机选择翻转轴（x或y，通常不翻转z）
        flip_axis = random.choice([0, 1])
        pc = pc.copy()
        pc[:, flip_axis] = -pc[:, flip_axis]
    return pc


class ImageColorJitter:
    """
    图像颜色抖动增强
    包括亮度、对比度、饱和度和色调的随机调整
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        return self.transform(img)


class ImageAugmentation:
    """
    组合图像增强策略
    """
    def __init__(self, 
                 color_jitter_prob=0.5,
                 brightness=0.2, 
                 contrast=0.2, 
                 saturation=0.2, 
                 hue=0.1,
                 grayscale_prob=0.1):
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.grayscale = transforms.RandomGrayscale(p=1.0)
    
    def __call__(self, img):
        # 颜色抖动
        if random.random() < self.color_jitter_prob:
            img = self.color_jitter(img)
        
        # 随机灰度化
        if random.random() < self.grayscale_prob:
            img = self.grayscale(img)
            # 如果转为灰度，需要转回RGB（3通道）
            if img.mode == 'L':
                img = img.convert('RGB')
        
        return img


def img_normalize_train(img, augment=False, aug_config=None):
    """
    图像归一化（可选增强）
    
    Args:
        img: PIL图像
        augment: 是否应用数据增强
        aug_config: 增强配置字典
    """
    if augment:
        if aug_config is None:
            aug_config = {
                'color_jitter_prob': 0.5,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
                'grayscale_prob': 0.05
            }
        augmentation = ImageAugmentation(**aug_config)
        img = augmentation(img)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img


def img_normalize_val(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img


class PIAD(Dataset):
    """
    Point-Image Affordance Dataset
    支持点云和图像的数据增强
    """

    def __init__(self, run_type, setting_type, point_path, img_path, box_path, 
                 pair=2, img_size=(224, 224),
                 # 数据增强配置
                 augment=False,
                 pc_aug_config=None,
                 img_aug_config=None):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.b_path = box_path
        self.pair_num = pair
        self.img_size = img_size
        
        # 数据增强开关
        self.augment = augment and (run_type == 'train')
        
        # 点云增强配置
        self.pc_aug_config = pc_aug_config if pc_aug_config else {
            'jitter': True,
            'jitter_sigma': 0.01,
            'jitter_clip': 0.05,
            'rotate': True,
            'rotate_axis': 'y',  # 'x', 'y', 'z', 或 'all'
            'rotate_angle_range': (-15, 15),
            'scale': True,
            'scale_range': (0.9, 1.1),
            'flip': True,
            'flip_prob': 0.5
        }
        
        # 图像增强配置
        self.img_aug_config = img_aug_config if img_aug_config else {
            'color_jitter_prob': 0.5,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'grayscale_prob': 0.05
        }

        # Affordance labels
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open',
                                      'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                                      'push', 'listen', 'wear', 'press', 'cut', 'stab']

        # Object categories for different settings
        if setting_type == 'Unseen':
            number_dict = {'Knife': 0, 'Refrigerator': 0, 'Earphone': 0,
                           'Bag': 0, 'Keyboard': 0, 'Chair': 0, 'Hat': 0, 'Door': 0, 'TrashCan': 0, 'Table': 0,
                           'Faucet': 0, 'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Display': 0, 'Mug': 0,
                           'Clock': 0}
        else:  # Seen
            number_dict = {'Earphone': 0, 'Bag': 0, 'Chair': 0, 'Refrigerator': 0, 'Knife': 0, 'Dishwasher': 0,
                           'Keyboard': 0, 'Scissors': 0, 'Table': 0,
                           'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Microwave': 0, 'Display': 0, 'TrashCan': 0,
                           'Hat': 0, 'Clock': 0,
                           'Door': 0, 'Mug': 0, 'Faucet': 0, 'Vase': 0, 'Laptop': 0, 'Bed': 0}

        self.img_files = self.read_file(self.i_path)
        self.box_files = self.read_file(self.b_path)

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)
    
    def apply_pc_augmentation(self, pc):
        """
        应用点云数据增强
        
        Args:
            pc: 点云坐标 (N, 3)
        
        Returns:
            增强后的点云 (N, 3)
        """
        augmented_pc = pc.copy()
        
        # 随机旋转
        if self.pc_aug_config.get('rotate', False):
            augmented_pc = pc_rotate(
                augmented_pc,
                axis=self.pc_aug_config.get('rotate_axis', 'y'),
                angle_range=self.pc_aug_config.get('rotate_angle_range', (-15, 15))
            )
        
        # 随机缩放
        if self.pc_aug_config.get('scale', False):
            augmented_pc = pc_scale(
                augmented_pc,
                scale_range=self.pc_aug_config.get('scale_range', (0.9, 1.1))
            )
        
        # 随机翻转
        if self.pc_aug_config.get('flip', False):
            augmented_pc = pc_flip(
                augmented_pc,
                p=self.pc_aug_config.get('flip_prob', 0.5)
            )
        
        # 点云抖动（通常放在最后）
        if self.pc_aug_config.get('jitter', False):
            augmented_pc = pc_jitter(
                augmented_pc,
                sigma=self.pc_aug_config.get('jitter_sigma', 0.01),
                clip=self.pc_aug_config.get('jitter_clip', 0.05)
            )
        
        return augmented_pc

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        box_path = self.box_files[index]

        if self.run_type == 'val':
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('_')[-3]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0], range_[1]), self.pair_num)

        Img = Image.open(img_path).convert('RGB')

        if self.run_type == 'train':
            Img, subject, object_box = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            # 应用图像增强（颜色抖动等）
            Img = img_normalize_train(Img, augment=self.augment, aug_config=self.img_aug_config)

            Points_List = []
            affordance_label_List = []
            affordance_index_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                # 点云归一化
                Points, _, _ = pc_normalize(Points)
                # 应用点云数据增强（旋转、缩放、抖动等）
                if self.augment:
                    Points = self.apply_pc_augmentation(Points)
                Points = Points.transpose()
                affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)
                Points_List.append(Points)
                affordance_label_List.append(affordance_label)
                affordance_index_List.append(affordance_index)

        else:
            subject, object_box = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            # 验证时不应用增强
            Img = img_normalize_train(Img, augment=False)

            Point, affordance_label = self.extract_point_file(point_path)
            Point, _, _ = pc_normalize(Point)
            Point = Point.transpose()

            affordance_label, _ = self.get_affordance_label(img_path, affordance_label)

        if self.run_type == 'train':
            return Img, Points_List, affordance_label_List, affordance_index_List, sub_box, obj_box
        else:
            return Img, Point, affordance_label, img_path, point_path, sub_box, obj_box

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict is not None:
                    object_ = file.split('_')[-2]
                    if object_ in number_dict:
                        number_dict[object_] += 1
                file_list.append(file)
            f.close()
        if number_dict is not None:
            return file_list, number_dict
        else:
            return file_list

    def extract_point_file(self, path):
        with open(path, 'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[:, 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str_path, label):
        cut_str = str_path.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)

        label = label[:, index]

        return label, index

    def get_crop(self, json_path, image, run_type):
        json_data = json.load(open(json_path, 'r'))
        sub_points, obj_points = [], []
        for box in json_data['shapes']:
            if box['label'] == 'subject':
                sub_points = box['points']
            elif box['label'] == 'object':
                obj_points = box['points']
        if len(sub_points) == 0:
            temp_box = [0.] * 2
            for i in range(2):
                sub_points.append(temp_box)

        if run_type == 'train':
            crop_img, crop_subpoints, crop_objpoints = self.random_crop_with_points(image, sub_points, obj_points)
            return crop_img, crop_subpoints, crop_objpoints
        else:
            sub_points = [*sub_points[0], *sub_points[1]]
            obj_points = [*obj_points[0], *obj_points[1]]
            sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
            return sub_points, obj_points

    def random_crop_with_points(self, image, sub_points, obj_points):
        points = []
        image = np.array(image)
        for obj_point in obj_points:
            points.append(obj_point)

        for sub_point in sub_points:
            points.append(sub_point)

        h, w = image.shape[0], image.shape[1]
        points = np.array(points, np.int32)
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

        t = random.randint(0, min_y) if min_y > 0 else 0
        b = random.randint(max_y + 1, h) if max_y + 1 < h else max_y + 1
        lft = random.randint(0, min_x) if min_x > 0 else 0
        r = random.randint(max_x + 1, w) if max_x + 1 < w else max_x + 1

        new_img = image[t: b, lft: r, :]

        new_img = Image.fromarray(new_img)
        obj_points_arr = points[0:2]
        new_objpoints = [[x - lft, y - t] for x, y in obj_points_arr]
        obj_LT = new_objpoints[0]
        obj_RB = new_objpoints[1]
        new_objpoints = [*obj_LT, *obj_RB]

        sub_points_arr = points[2:]
        new_subpoints = [[x - lft, y - t] for x, y in sub_points_arr]
        sub_LT = new_subpoints[0]
        sub_RB = new_subpoints[1]
        new_subpoints = [*sub_LT, *sub_RB]

        return new_img, new_subpoints, new_objpoints

    def get_resize_box(self, image, new_size, sub_box, obj_box):
        image = np.array(image)
        h_ = image.shape[0]
        w_ = image.shape[1]

        scale_h = new_size[0] / h_
        scale_w = new_size[1] / w_

        sub_box[0], sub_box[2] = sub_box[0] * scale_w, sub_box[2] * scale_w
        sub_box[1], sub_box[3] = sub_box[1] * scale_h, sub_box[3] * scale_h

        obj_box[0], obj_box[2] = obj_box[0] * scale_w, obj_box[2] * scale_w
        obj_box[1], obj_box[3] = obj_box[1] * scale_h, obj_box[3] * scale_h

        return sub_box, obj_box


class PIADInference(Dataset):
    """
    Dataset for inference/visualization
    """

    def __init__(self, point_path, img_path, box_path, img_size=(224, 224)):
        super().__init__()
        self.img_size = img_size
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open',
                                      'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                                      'push', 'listen', 'wear', 'press', 'cut', 'stab']

        self.img_files = self.read_file(img_path)
        self.box_files = self.read_file(box_path)
        self.point_files = self.read_file(point_path)

        # Shuffle the data
        combined = list(zip(self.img_files, self.box_files, self.point_files))
        random.shuffle(combined)
        self.img_files, self.box_files, self.point_files = zip(*combined)
        self.img_files = list(self.img_files)
        self.box_files = list(self.box_files)
        self.point_files = list(self.point_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        box_path = self.box_files[index]
        point_path = self.point_files[index]

        Img = Image.open(img_path).convert('RGB')

        subject, object_box = self.get_crop(box_path, Img)
        sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
        sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
        Img = Img.resize(self.img_size)
        Img = img_normalize_val(Img)

        Point, affordance_label = self.extract_point_file(point_path)
        Point, _, _ = pc_normalize(Point)
        Point = Point.transpose()

        affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)

        return Img, Point, affordance_label, img_path, point_path, sub_box, obj_box, affordance_index

    def read_file(self, path):
        file_list = []
        with open(path, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                file_list.append(file)
            f.close()
        return file_list

    def extract_point_file(self, path):
        with open(path, 'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[:, 3:]
        return points_coordinates, affordance_label

    def get_affordance_label(self, str_path, label):
        cut_str = str_path.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)
        label = label[:, index]
        return label, index

    def get_crop(self, json_path, image):
        json_data = json.load(open(json_path, 'r'))
        sub_points, obj_points = [], []
        for box in json_data['shapes']:
            if box['label'] == 'subject':
                sub_points = box['points']
            elif box['label'] == 'object':
                obj_points = box['points']
        if len(sub_points) == 0:
            temp_box = [0.] * 2
            for i in range(2):
                sub_points.append(temp_box)

        sub_points = [*sub_points[0], *sub_points[1]]
        obj_points = [*obj_points[0], *obj_points[1]]
        sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
        return sub_points, obj_points

    def get_resize_box(self, image, new_size, sub_box, obj_box):
        image = np.array(image)
        h_ = image.shape[0]
        w_ = image.shape[1]

        scale_h = new_size[0] / h_
        scale_w = new_size[1] / w_

        sub_box = np.array(sub_box, dtype=np.float32)
        obj_box = np.array(obj_box, dtype=np.float32)

        sub_box[0], sub_box[2] = sub_box[0] * scale_w, sub_box[2] * scale_w
        sub_box[1], sub_box[3] = sub_box[1] * scale_h, sub_box[3] * scale_h

        obj_box[0], obj_box[2] = obj_box[0] * scale_w, obj_box[2] * scale_w
        obj_box[1], obj_box[3] = obj_box[1] * scale_h, obj_box[3] * scale_h

        return sub_box.tolist(), obj_box.tolist()


class PIADUnseenFewShot(Dataset):
    """
    Few-shot learning for Unseen dataset
    在Unseen设置下，将部分测试集用作训练，剩余部分作为测试
    """

    def __init__(self, 
                 run_type,  # 'train' 或 'test'
                 setting_type,  # 固定为'Unseen'
                 point_path,  # Point_Test.txt路径
                 img_path,  # Img_Test.txt路径
                 box_path,  # Box_Test.txt路径
                 shot_num=5,  # 每个类别的少样本数量
                 img_size=(224, 224)):
        super().__init__()
        
        assert setting_type == 'Unseen', "此数据集类仅用于Unseen设置的Few-shot学习"
        
        self.run_type = run_type
        self.img_size = img_size
        self.shot_num = shot_num
        
        # 可承受性标签列表
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open',
                                      'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                                      'push', 'listen', 'wear', 'press', 'cut', 'stab']
        
        # 读取文件
        self.img_files = self.read_file(img_path)
        self.box_files = self.read_file(box_path)
        self.point_files = self.read_file(point_path)
        
        # 按照可承受性类别分组
        self.affordance_groups = self.group_by_affordance()
        
        # 划分训练集和测试集
        self.train_indices, self.test_indices = self.split_few_shot_data()
        
        # 根据run_type选择索引
        if self.run_type == 'train':
            self.indices = self.train_indices
        else:  # 'test' 或 'val'
            self.indices = self.test_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取实际索引
        actual_idx = self.indices[idx]
        
        img_path = self.img_files[actual_idx]
        box_path = self.box_files[actual_idx]
        point_path = self.point_files[actual_idx]
        
        # 加载和预处理图像
        Img = Image.open(img_path).convert('RGB')
        subject, object_box = self.get_crop(box_path, Img)
        sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object_box)
        sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
        Img = Img.resize(self.img_size)
        Img = img_normalize_val(Img)
        
        # 加载和预处理点云
        Point, affordance_label = self.extract_point_file(point_path)
        Point, _, _ = pc_normalize(Point)
        Point = Point.transpose()
        
        # 获取可承受性标签
        affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)
        
        # 为了与PIAD数据集格式一致，我们需要返回6个值
        # 训练模式下PIAD返回: Img, Points_List, affordance_label_List, affordance_index_List, sub_box, obj_box
        # 我们将单个样本包装成列表
        
        if self.run_type == 'train':
            # 训练模式：返回6个值
            return Img, [Point], [affordance_label], [affordance_index], sub_box, obj_box
        else:
            # 测试模式：返回7个值（与原始PIAD的val模式一致）
            return Img, Point, affordance_label, img_path, point_path, sub_box, obj_box
    def read_file(self, path):
        """读取文件列表"""
        file_list = []
        with open(path, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                file_list.append(file)
        return file_list
    
    def extract_point_file(self, path):
        """提取点云文件"""
        with open(path, 'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[:, 3:]
        return points_coordinates, affordance_label
    
    def get_affordance_label(self, str_path, label):
        """从文件名提取可承受性标签"""
        cut_str = str_path.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)
        label = label[:, index]
        return label, index
    
    def get_crop(self, json_path, image):
        """获取裁剪框"""
        json_data = json.load(open(json_path, 'r'))
        sub_points, obj_points = [], []
        for box in json_data['shapes']:
            if box['label'] == 'subject':
                sub_points = box['points']
            elif box['label'] == 'object':
                obj_points = box['points']
        if len(sub_points) == 0:
            temp_box = [0.] * 2
            for i in range(2):
                sub_points.append(temp_box)
        
        sub_points = [*sub_points[0], *sub_points[1]]
        obj_points = [*obj_points[0], *obj_points[1]]
        sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
        return sub_points, obj_points
    
    def get_resize_box(self, image, new_size, sub_box, obj_box):
        """调整框的大小"""
        image = np.array(image)
        h_ = image.shape[0]
        w_ = image.shape[1]
        
        scale_h = new_size[0] / h_
        scale_w = new_size[1] / w_
        
        sub_box = np.array(sub_box, dtype=np.float32)
        obj_box = np.array(obj_box, dtype=np.float32)
        
        sub_box[0], sub_box[2] = sub_box[0] * scale_w, sub_box[2] * scale_w
        sub_box[1], sub_box[3] = sub_box[1] * scale_h, sub_box[3] * scale_h
        
        obj_box[0], obj_box[2] = obj_box[0] * scale_w, obj_box[2] * scale_w
        obj_box[1], obj_box[3] = obj_box[1] * scale_h, obj_box[3] * scale_h
        
        return sub_box.tolist(), obj_box.tolist()
    
    def group_by_affordance(self):
        """按照可承受性类别分组样本"""
        groups = {}
        for i, img_path in enumerate(self.img_files):
            affordance = img_path.split('_')[-2]
            if affordance not in groups:
                groups[affordance] = []
            groups[affordance].append(i)
        return groups
    
    def split_few_shot_data(self, test_ratio=0.7):
        """
        划分Few-shot数据
        Args:
            test_ratio: 测试集比例
        """
        train_indices = []
        test_indices = []
        
        for affordance, indices in self.affordance_groups.items():
            # 打乱顺序
            random.shuffle(indices)
            
            # 计算每个类别的训练样本数
            total_samples = len(indices)
            train_samples = min(self.shot_num, total_samples)
            
            # 划分训练集和测试集
            train_indices.extend(indices[:train_samples])
            test_indices.extend(indices[train_samples:])
            
            print(f"Affordance '{affordance}': {train_samples} train, {len(indices)-train_samples} test")
        
        print(f"Total: {len(train_indices)} train samples, {len(test_indices)} test samples")
        
        return train_indices, test_indices

if __name__ == "__main__":
    # show the original pointcloud and 增强过d的点云
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 创建数据集实例
    dataset = PIAD(
        run_type='train',
        setting_type='Seen',
        point_path='./Data/Seen/Point_Train.txt',
        img_path='./Data/Seen/Img_Train.txt',
        box_path='./Data/Seen/Box_Train.txt',
        pair=2,
        img_size=(224, 224),
        augment=True,  # 启用数据增强
        pc_aug_config={
            'jitter': True,
            'jitter_sigma': 0.002,
            'jitter_clip': 0.05,
            'rotate': True,
            'rotate_axis': 'y',
            'rotate_angle_range': (-30, 30),
            'scale': True,
            'scale_range': (0.8, 1.2),
            'flip': True,
            'flip_prob': 0.5
        },
        img_aug_config={
            'color_jitter_prob': 0.7,
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1,
            'grayscale_prob': 0.1
        }
    )
    import random
    a=random.randint(0, len(dataset) - 1)
    # 获取一个样本
    img, points_list, affordance_labels, affordance_indices, sub_box, obj_box = dataset[a]
    # 同时可视化原始点云和增强后的点云
    fig = plt.figure(figsize=(12, 6))
    for i, points in enumerate(points_list):
        ax = fig.add_subplot(1, len(points_list), i + 1, projection='3d')
        ax.scatter(points[0], points[1], points[2], s=1)
        ax.set_title(f'Affordance: {dataset.affordance_label_list[affordance_indices[i]]}')
    plt.savefig('augmented_pointclouds.png')  # 保存图像
    plt.show()
    # 保存两个点云图像
    
   # 如果使用无数据增强：
    dataset1=PIAD(
        run_type='train',
        setting_type='Seen',
        point_path='./Data/Seen/Point_Train.txt',
        img_path='./Data/Seen/Img_Train.txt',
        box_path='./Data/Seen/Box_Train.txt',
        pair=2,
        img_size=(224, 224),
        augment=False  # 不启用数据增强
    )
    # 获取一个样本
    img1, points_list1, affordance_labels1, affordance_indices1, sub_box1, obj_box1 = dataset1[a]
    # 可视化原始点云
    fig = plt.figure(figsize=(12, 6))
    for i, points in enumerate(points_list1): 
        ax = fig.add_subplot(1, len(points_list1), i + 1, projection='3d')
        ax.scatter(points[0], points[1], points[2], s=1)
        ax.set_title(f'Affordance: {dataset1.affordance_label_list[affordance_indices1[i]]}')
    plt.show()  