import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class RGBT234Dataset(BaseDataset):
    """ RGBT234 dataset for RGB-T tracking. """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt234_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)

        # 加载标注（优先使用 infrared.txt，若不存在则使用 visible.txt）
        anno_path = os.path.join(seq_dir, 'infrared.txt')
        if not os.path.exists(anno_path):
            anno_path = os.path.join(seq_dir, 'visible.txt')
        ground_truth_rect = load_text(anno_path, delimiter=',', dtype=np.float64)

        # 加载双模态图像路径
        frames_path_i = os.path.join(seq_dir, 'infrared')
        frames_path_v = os.path.join(seq_dir, 'visible')

        # 排序并构建图像路径列表
        frame_list_i = sorted([f for f in os.listdir(frames_path_i) if f.endswith(".jpg")],
                              key=lambda x: int(x.split('.')[0].replace('i', '')))
        frame_list_v = sorted([f for f in os.listdir(frames_path_v) if f.endswith(".jpg")],
                              key=lambda x: int(x.split('.')[0].replace('v', '')))

        frames_list_i = [os.path.join(frames_path_i, f) for f in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, f) for f in frame_list_v]

        # 合并双模态路径（假设帧数对齐）
        frames_list = [frames_list_v, frames_list_i]

        return Sequence(sequence_name, frames_list, 'rgbt234', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_file = os.path.join(self.base_path, 'rgbt234.txt')
        with open(list_file, 'r') as f:
            sequence_list = [line.strip() for line in f.readlines() if line.strip()]
        return sequence_list