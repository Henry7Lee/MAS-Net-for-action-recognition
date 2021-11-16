
import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import csv
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row, new_length):
        self._data = row
        self.new_length = new_length

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        #         return int(self._data[1]) - self.new_length + 1
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class UCF101VideoRecord(object):
    def __init__(self, row, new_length):
        self._data = row
        self._new_length = new_length

    @property
    def path(self):
        return self._data[2]

    @property
    def num_frames(self):
        _len = int(self._data[3])
        #         return _len - self._new_length + 1
        return _len

    @property
    def label(self):
        return int(self._data[1])


class TSNDataSet(data.Dataset):
    def __init__(self, dataset, root_path, list_file,
                 num_segments=8, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, twice_sample=False ,
                 dense_sample=False,dense_sample_num=10):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing

        self.dense_sample_num = dense_sample_num  # using dense sample as I3D
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print(('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx))))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]


    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= self.new_length]
        self.video_list = [VideoRecord(item, self.new_length) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """Random Sampling from each video segment

        :param record: VideoRecord
        :return: list
        """

        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        """Sampling for validation set

        Sample the middle frame from each video segment
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            # start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx = 0 if sample_pos == 1 else (sample_pos - 1) // 2
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=self.dense_sample_num, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [int(x * tick + start_idx) for x in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])
            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

            return offsets + 1

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        process_data,record_label = self.transform((images,record.label))
        return process_data, record_label


    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            print("path not exist",(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def __len__(self):
        return len(self.video_list)
