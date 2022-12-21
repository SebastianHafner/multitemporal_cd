import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


def compose_transformations(cfg, no_augmentations: bool = False):
    if no_augmentations:
        return transforms.Compose([Numpy2Torch()])

    transformations = []

    # cropping
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'none':
        transformations.append(UniformCrop(cfg.MODEL.PATCH_SIZE))
    elif cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'importance':
        transformations.append(ImportanceRandomCrop(cfg.MODEL.PATCH_SIZE))
    else:
        raise Exception('Unknown oversampling type')

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_SHIFT:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        data, label_ch, labels_sem = args
        for i, x in enumerate(data):
            assert(len(x.shape) == 4)
            data[i] = torch.tensor(x.transpose(0, 3, 1, 2))
        label_ch = TF.to_tensor(label_ch)
        for i, y in enumerate(labels_sem):
            assert (len(y.shape) == 3)
            labels_sem[i] = TF.to_tensor(y)
        return data, label_ch, labels_sem


class RandomFlip(object):
    def __call__(self, args):
        data, label_ch, labels_sem = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            for i, x in enumerate(data):
                data[i] = np.flip(x, axis=2).copy()
            label_ch = np.flip(label_ch, axis=1).copy()
            for i, y in enumerate(labels_sem):
                labels_sem[i] = np.flip(y, axis=1).copy()

        if vertical_flip:
            for i, x in enumerate(data):
                data[i] = np.flip(x, axis=1).copy()
            label_ch = np.flip(label_ch, axis=0).copy()
            for i, y in enumerate(labels_sem):
                labels_sem[i] = np.flip(y, axis=0).copy()

        return data, label_ch, labels_sem


class RandomRotate(object):
    def __call__(self, args):
        data, label_ch, labels_sem = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        for i, x in enumerate(data):
            data[i] = np.rot90(x, k, axes=(1, 2)).copy()
        label_ch = np.rot90(label_ch, k, axes=(0, 1)).copy()
        for i, y in enumerate(labels_sem):
            labels_sem[i] = np.rot90(y, k, axes=(0, 1)).copy()
        return data, label_ch, labels_sem


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, args):
        x_min, x_max, y_min, y_max = self.random_crop_bounds(args)
        data_list, label_ch, labels_sem = args
        for i, data in enumerate(data_list):
            data_list[i] = data[:, y_min:y_max, x_min:x_max, ]
        label_ch = label_ch[y_min:y_max, x_min:x_max, ]
        for i, y in enumerate(labels_sem):
            labels_sem[i] = y[y_min:y_max, x_min:x_max, ]
        return data_list, label_ch, labels_sem

    def random_crop_bounds(self, args) -> tuple:
        _, label, _ = args
        height, width, _ = label.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)
        return x, x + self.crop_size, y, y + self.crop_size

    def __call__(self, args):
        data, label_ch, labels_sem = self.random_crop(args)
        return data, label_ch, labels_sem


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args, sample_size: int = 20, balancing_factor: int = 5):

        data_list, label_ch, labels_sem = args

        random_bounds = [self.random_crop_bounds(args) for _ in range(sample_size)]

        crop_weights = [(label_ch[y_min:y_max, x_min:x_max, ]).sum() for x_min, x_max, y_min, y_max in random_bounds]
        crop_weights = np.array(crop_weights) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()
        sample_idx = np.random.choice(sample_size, p=crop_weights)

        x_min, x_max, y_min, y_max = random_bounds[sample_idx]

        for i, data in enumerate(data_list):
            data_list[i] = data[:, y_min:y_max, x_min:x_max, ]
        label_ch = label_ch[y_min:y_max, x_min:x_max, ]
        for i, y in enumerate(labels_sem):
            labels_sem[i] = y[y_min:y_max, x_min:x_max, ]

        return data_list, label_ch, labels_sem


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        timeseries, label = args
        timeseries_length = timeseries.shape[0]
        factors = np.random.uniform(self.min_factor, self.max_factor, timeseries.shape[-1])
        timeseries_rescaled = np.clip(timeseries * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return timeseries_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        timeseries, label = args
        timeseries_length = timeseries.shape[0]
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, timeseries.shape[-1])
        timeseries_gamma_corrected = np.clip(np.power(timeseries, gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(
            np.float32)
        return timeseries_gamma_corrected, label