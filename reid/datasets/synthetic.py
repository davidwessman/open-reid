from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import random

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


random.seed(1)
class Synthetic(Dataset):
    def __init__(self, root, split_id=0, num_val=100, batch_id=None):
        super(Synthetic, self).__init__('%s-%s' % (root, batch_id), split_id=split_id)
        self.batch_id = batch_id
        self.nbr_chars = 1500

        if not self._check_integrity():
            self.prepare()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def prepare(self):
        if self._check_integrity():
            print("Files already prepared!")
            return

        import re
        # import hashlib
        import shutil
        from glob import glob
        # from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')

        if not osp.exists(raw_dir):
            raise RuntimeError("Could not find Synthetic dataset with batch_id %s on path %s" % (self.batch_id, raw_dir))

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # x nbr of characters and no "cameras"
        identities = [[[] for _ in range(10)] for _ in range(self.nbr_chars)]

        pattern = re.compile(r'([-\d]+)-')
        fpaths = sorted(glob(osp.join(raw_dir, self.batch_id, '*.png')))
        pids = set()
        for fpath in fpaths:
            cam = random.randint(0, 9)
            fname = osp.basename(fpath)
            pid = int(pattern.search(fname).group(1))
            assert 0 <= pid < self.nbr_chars  # pid == 0 means background
            pids.add(pid)
            fname = ('{:08d}_{:02d}_{:04d}.jpg'
                     .format(pid, cam, len(identities[pid][cam])))
            identities[pid][cam].append(fname)
            shutil.copy(fpath, osp.join(images_dir, fname))

        # Save meta information into a json file
        meta = {'name': 'Synthetic - %s' % self.batch_id, 'shot': 'multiple',
                'num_cameras': 10, 'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
