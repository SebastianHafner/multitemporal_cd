import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles, spacenet7_helpers


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type

        # unpacking config
        self.root_path = Path(cfg.PATHS.DATASET)
        self.sensor = cfg.DATALOADER.SENSOR
        self.timeseries_length = cfg.DATALOADER.TIMESERIES_LENGTH
        self.patch_size = cfg.MODEL.PATCH_SIZE

        self.metadata = geofiles.load_json(self.root_path / f'metadata_multimodal_cd.json')

        self.include_alpha = cfg.DATALOADER.INCLUDE_ALPHA
        if cfg.DATALOADER.SENSOR == 'planetscope':
            self.img_bands = 3 if not self.include_alpha else 4
        elif self.sensor == 'sentinel1':
            self.img_bands = len(cfg.DATALOADER.SENTINEL1_BANDS)
        elif self.sensor == 'sentinel2':
            self.img_bands = len(cfg.DATALOADER.SENTINEL2_BANDS)

        # creating boolean feature vector to subset sentinel 2 bands
        self.s1_indices = [['VV', 'VH'].index(band) for band in cfg.DATALOADER.SENTINEL1_BANDS]
        self.s2_indices = [['B2', 'B3', 'B4', 'B8'].index(band) for band in cfg.DATALOADER.SENTINEL2_BANDS]

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _load_planetscope_mosaic(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        img, _, _ = geofiles.read_tif(file)
        img = img / 255
        # 4th band (last oen) is alpha band
        if not self.include_alpha:
            img = img[:, :, :-1]
        return img.astype(np.float32)

    def _load_sentinel1_scene(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'sentinel1'
        file = folder / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
        img, *_ = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return img.astype(np.float32)

    def _load_sentinel2_scene(self, aoi_id: str, dataset: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / dataset / aoi_id / 'sentinel2'
        file = folder / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
        img, *_ = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return img.astype(np.float32)

    def _load_satellite_image(self, aoi_id: str, dataset: str, year: int, month: int,
                              padding: bool = True) -> np.ndarray:
        if self.sensor == 'planetscope':
            img = self._load_planetscope_mosaic(aoi_id, dataset, year, month)
        elif self.sensor == 'sentinel1':
            img = self._load_sentinel1_scene(aoi_id, dataset, year, month)
        elif self.sensor == 'sentinel2':
            img = self._load_sentinel2_scene(aoi_id, dataset, year, month)
        else:
            raise Exception('Unknown sensor')
        if padding:
            img = self.pad(img)
        return img

    def _load_satellite_timeseries(self, aoi_id: str, dataset: str, dates: list) -> np.ndarray:
        timeseries = []
        for year, month in dates:
            img = self._load_satellite_image(aoi_id, dataset, year, month)
            timeseries.append(img)
        timeseries = np.stack(timeseries)
        return timeseries

    def _load_building_label(self, aoi_id: str, year: int, month: int, padding: bool = True) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        label, _, _ = geofiles.read_tif(file)
        label = label > 0
        if padding:
            label = self.pad(label)
        return label.astype(np.float32)

    def _load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self._load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self._load_building_label(aoi_id, year_t2, month_t2)
        change = np.logical_and(building_t1 == 0, building_t2 == 1)
        return change.astype(np.float32)

    def _load_change_date_label(self, aoi_id: str, dates: list) -> np.ndarray:
        label_timeseries = []
        for year, month in dates:
            label = self._load_building_label(aoi_id, year, month)
            label_timeseries.append(label)
        label_timeseries = np.concatenate(label_timeseries, axis=-1)

        change_date_label = np.zeros((label_timeseries.shape[0], label_timeseries.shape[1]))
        for i in range(1, self.timeseries_length):
            prev_label, current_label = label_timeseries[:, :, i - 1], label_timeseries[:, :, i]
            change = np.logical_and(prev_label == 0, current_label == 1)
            change_date_label[change] = i

        return change_date_label[:, :, None]

    def _load_mask(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_mask.tif'
        mask, _, _ = geofiles.read_tif(file)
        return mask.astype(np.int8)

    def get_aoi_ids(self) -> list:
        return list(set([s['aoi_id'] for s in self.samples]))

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'

    @staticmethod
    def pad(arr: np.ndarray, res: int = 1024) -> np.ndarray:
        assert(len(arr.shape) == 3)
        h, w, _ = arr.shape
        arr = np.pad(arr, pad_width=((0, res - h), (0, res - w), (0, 0)), mode='edge')
        assert (arr.shape[0] == arr.shape[1])
        return arr


class SpaceNet7TrainingDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, disable_multiplier: bool = False,
                 no_augmentations: bool = False, dataset_mode: str = 'random'):
        super().__init__(cfg, 'training')

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg, no_augmentations)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)

        self.dataset_mode = dataset_mode

        if not disable_multiplier:
            self.aoi_ids = self.aoi_ids * cfg.DATALOADER.TRAINING_MULTIPLIER

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.metadata = manager.dict(self.metadata)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        aoi_id = self.aoi_ids[index]
        timestamps = self.metadata[aoi_id]
        dataset = timestamps[0]['dataset']

        if self.sensor == 'sentinel2' or self.sensor == 'sentinel2':
            timestamps = [ts for ts in timestamps if not ts['mask'] and ts[self.sensor]]
        else:
            timestamps = [ts for ts in timestamps if not ts['mask']]

        if self.dataset_mode == 'evenly_spaced':
            indices = [int(i) for i in np.linspace(0, len(timestamps) - 1, self.timeseries_length, endpoint=True)]
        else:
            indices = list(sorted(np.random.randint(0, len(timestamps), size=self.timeseries_length)))
        dates = [(timestamps[i]['year'], timestamps[i]['month']) for i in indices]

        timeseries = self._load_satellite_timeseries(aoi_id, dataset, dates)

        change = self._load_change_label(aoi_id, *dates[0], *dates[-1])
        # change_date = self._load_change_date_label(aoi_id, dates)

        timeseries, change = self.transform((timeseries, change))

        item = {
            'x': timeseries,
            'y': change,
            'aoi_id': aoi_id,
            'dates': dates,
        }

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class SpaceNet7EvaluationDataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str):
        super().__init__(cfg, run_type)

        self.enforce_patch_size = cfg.DATALOADER.ENFORCE_PATCH_SIZE

        # handling transformations of data
        self.transform = augmentations.compose_transformations(cfg, no_augmentations=True)

        # loading labeled samples (sn7 train set) and subset to run type aoi ids
        if run_type == 'training':
            self.aoi_ids = list(cfg.DATASET.TRAIN_IDS)
        elif run_type == 'validation':
            self.aoi_ids = list(cfg.DATASET.VALIDATION_IDS)
        elif run_type == 'test':
            self.aoi_ids = list(cfg.DATASET.TEST_IDS)
        else:
            raise Exception('Unknown run type')

        self.samples = []
        for aoi_id in self.aoi_ids:
            if self.enforce_patch_size:
                for i in (0, 1024, self.patch_size):
                    for j in (0, 1024, self.patch_size):
                        self.samples.append({
                            'aoi_id': aoi_id,
                            'i': i,
                            'j': j,
                        })
            else:
                self.samples.append({'aoi_id': aoi_id})

        manager = multiprocessing.Manager()
        self.aoi_ids = manager.list(self.aoi_ids)
        self.samples = manager.list(self.samples)
        self.metadata = manager.dict(self.metadata)
        self.patch_ids = manager.dict(self.patch_ids)

        self.length = len(self.aoi_ids)

    def __getitem__(self, index):

        sample = self.samples[index]
        aoi_id = sample['aoi_id']
        timestamps = self.metadata[aoi_id]
        dataset = timestamps[0]['dataset']

        if self.sensor == 'sentinel2' or self.sensor == 'sentinel2':
            timestamps = [ts for ts in timestamps if not ts['mask'] and ts[self.sensor]]
        else:
            timestamps = [ts for ts in timestamps if not ts['mask']]

        indices = [int(i) for i in np.linspace(0, len(timestamps) - 1, self.timeseries_length, endpoint=True)]
        dates = [(timestamps[i]['year'], timestamps[i]['month']) for i in indices]

        timeseries = self._load_satellite_timeseries(aoi_id, dataset, dates)

        change = self._load_change_label(aoi_id, *dates[0], *dates[-1])
        # change_date = self._load_change_date_label(aoi_id, dates)

        timeseries, change = self.transform((timeseries, change))

        if self.enforce_patch_size:
            assert(timeseries.size(-2) == timeseries.size(-1))
            assert(change.size(-2) == timeseries.size(-1))
            # crop to patch
            i, j = sample['i'], sample['j']
            timeseries = timeseries[:, :, i:i+self.patch_size, j:j+self.patch_size]
            change = change[:, i:i + self.patch_size, j:j + self.patch_size]

        item = {
            'x': timeseries,
            'y': change,
            'aoi_id': aoi_id,
            'dates': dates,
        }

        return item

    def get_index(self, aoi_id: str) -> int:
        for index, candidate_aoi_id in enumerate(self.aoi_ids):
            if aoi_id == candidate_aoi_id:
                return index
        return None

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'