import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        self.uniform_sample = sample_config.get('uniform_sample', False)
        self.temporal_sampler = sample_config.get('temporal_sampler')
        self.temporal_p_interval = sample_config.get('temporal_p_interval', 1)
        self.temporal_test_mode = sample_config.get('temporal_test_mode', False)
        self.temporal_seed = sample_config.get('temporal_seed', 255)

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config.get('frames_skip_num', 0)

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.temporal_sampler == 'protogcn':
                    indices = self._protogcn_sample_indices(seq_len, frames_num)
                elif self.uniform_sample:
                    if seq_len < frames_num:
                        indices = np.arange(frames_num) % seq_len
                    else:
                        boundaries = np.linspace(0, seq_len, frames_num + 1, dtype=np.int64)
                        widths = boundaries[1:] - boundaries[:-1]
                        indices = boundaries[:-1] + np.asarray([
                            np.random.randint(width) if width > 0 else 0
                            for width in widths])
                elif self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)

        batch[0] = fras_batch
        return batch

    def _protogcn_sample_indices(self, num_frames, clip_len):
        """Port of ProtoGCN ``UniformSampleDecode`` for one temporal clip."""
        if num_frames <= 0:
            raise ValueError('ProtoGCN temporal sampling requires a non-empty sequence.')

        p_interval = self.temporal_p_interval
        if not isinstance(p_interval, (tuple, list)):
            p_interval = (p_interval, p_interval)
        if len(p_interval) != 2:
            raise ValueError('temporal_p_interval must be a scalar or a two-value interval.')
        rng = np.random.RandomState(self.temporal_seed) if self.temporal_test_mode else np.random
        ratio = rng.rand() * (p_interval[1] - p_interval[0]) + p_interval[0]
        sampled_frames = int(ratio * num_frames)
        # The original pipeline assumes p_interval produces at least one frame.
        sampled_frames = max(1, sampled_frames)
        offset = rng.randint(num_frames - sampled_frames + 1)

        if sampled_frames < clip_len:
            start = rng.randint(0, sampled_frames)
            indices = np.arange(start, start + clip_len) % sampled_frames
        elif sampled_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            chosen = rng.choice(clip_len + 1, sampled_frames - clip_len, replace=False)
            expansion = np.zeros(clip_len + 1, dtype=np.int64)
            expansion[chosen] = 1
            indices = basic + np.cumsum(expansion)[:-1]
        else:
            boundaries = np.asarray(
                [i * sampled_frames // clip_len for i in range(clip_len + 1)])
            widths = np.diff(boundaries)
            indices = boundaries[:-1] + rng.randint(widths)
        return (indices + offset).tolist()
