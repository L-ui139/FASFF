from pytracking.evaluation.data import RgbtBaseDataset
from pytracking.evaluation.environment import env_settings

class RGBT210Dataset(RgbtBaseDataset):
    def __init__(self, dtype):
        base_path = env_settings().rgbt210_path
        dataset_name = 'RGBT210'
        rgb_sub_dir = 'visible'
        aux_sub_dir = 'infrared'
        rgb_anno_name = 'init.txt'
        aux_anno_name = 'init.txt'
        super().__init__(dtype, base_path, dataset_name, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)
