import json
import logging
import math
import os
import os.path as osp
import random
import ssl
import urllib.request
import zipfile

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

LEAF_NAMES = ["femnist", "celeba", "synthetic", "shakespeare", "twitter", "subreddit"]
IMAGE_SIZE = {"femnist": (28, 28), "celeba": (32, 32, 3)}  # fixed
CLASSES = {"femnist": list(range(62)), "celeba": list(range(2))}  # fixed
MODE = {"femnist": "L", "celeba": "RGB"}

logger = logging.getLogger(__name__)


def is_exists(path, names):
    exists_list = [osp.exists(osp.join(path, name)) for name in names]
    return False not in exists_list


def save_local_data(dir_path, train_data=None, train_targets=None, test_data=None, test_targets=None, val_data=None, val_targets=None):
    r"""
    Save data to disk. Source: \
    https://github.com/omarfoq/FedEM/blob/main/data/femnist/generate_data.py

    Args:
        train_data: x of train data
        train_targets: y of train data
        test_data: x of test data
        test_targets: y of test data
        val_data: x of validation data
        val_targets:y of validation data

    Note:
        save ``(`train_data`, `train_targets`)`` in ``{dir_path}/train.pt``, \
        ``(`val_data`, `val_targets`)`` in ``{dir_path}/val.pt`` \
        and ``(`test_data`, `test_targets`)`` in ``{dir_path}/test.pt``
    """

    if (train_data is not None) and (train_targets is not None):
        torch.save((train_data, train_targets), osp.join(dir_path, "train.pt"))

    if (test_data is not None) and (test_targets is not None):
        torch.save((test_data, test_targets), osp.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), osp.join(dir_path, "val.pt"))


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        logger.info(f"File {file} exists, use existing file.")
        return path

    logger.info(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


class LEAF(Dataset):
    """
    Base class for LEAF dataset from "LEAF: A Benchmark for Federated Settings"

    Arguments:
        root (str): root path.
        name (str): name of dataset, in `LEAF_NAMES`.
        transform: transform for x.
        target_transform: transform for y.

    """

    def __init__(self, root, name, transform_train, transform_val, target_transform):
        self.root = root
        self.name = name
        self.data_dict = {}
        if name not in LEAF_NAMES:
            raise ValueError(f"No leaf dataset named {self.name}")
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.target_transform = target_transform
        self.process_file()

    @property
    def raw_file_names(self):
        names = ["all_data.zip"]
        return names

    @property
    def extracted_file_names(self):
        names = ["all_data"]
        return names

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__len__()})"

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        for index in range(len(self.data_dict)):
            yield self.__getitem__(index)

    def download(self):
        raise NotImplementedError

    def extract(self):
        for name in self.raw_file_names:
            with zipfile.ZipFile(osp.join(self.raw_dir, name), "r") as f:
                f.extractall(self.raw_dir)

    def process_file(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        if len(os.listdir(self.processed_dir)) == 0:
            if not is_exists(self.raw_dir, self.extracted_file_names):
                if not is_exists(self.raw_dir, self.raw_file_names):
                    self.download()
                self.extract()
            self.process()

    def process(self):
        raise NotImplementedError


class CustomTensorDataset(Dataset):
    def __init__(self, name, data, targets, transform=None, target_transform=None) -> None:
        assert data.size(0) == targets.size(0)
        self.data = data
        self.targets = targets
        self.name = name
        self.transform = transform
        self.target_transform = target_transform
        self.classes = CLASSES[self.name]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.resize(img.numpy().astype(np.uint8), IMAGE_SIZE[self.name])
        img = Image.fromarray(img, mode=MODE[self.name])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.targets.size(0)


def file_name_to_idx(name):
    return int(name.split("_")[-1])


class CusteomLEAF(LEAF):
    """
    LEAF CV dataset from "LEAF: A Benchmark for Federated Settings"

    leaf.cmu.edu

    Arguments:
        root (str): root path.
        name (str): name of dataset, ‘femnist’ or ‘celeba’.
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
        train_tasks_frac (float): fraction of test tasks; default=1.0.
        transform_train: transform for x.
        transform_test: transform for x.
        target_transform: transform for y.

    """

    def __init__(self, root, name, s_frac=0.3, tr_frac=0.8, val_frac=0.0, seed=777, transform_train=None, transform_val=None, target_transform=None):
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.seed = seed
        self.classes = None
        super().__init__(root, name, transform_train, transform_val, target_transform)
        files = os.listdir(self.processed_dir)
        files = [f for f in files if f.startswith("task_")]

        train_data_total = torch.tensor([])
        train_targets_total = torch.tensor([], dtype=torch.int64)
        test_data_total = torch.tensor([])
        test_targets_total = torch.tensor([], dtype=torch.int64)
        val_data_total = torch.tensor([])
        val_targets_total = torch.tensor([], dtype=torch.int64)

        dict_users_train = {}
        dict_users_test = {}
        dict_users_val = {}

        if len(files):
            # Sort by idx
            files.sort(key=file_name_to_idx)

            for file in files:
                file_idx = file_name_to_idx(file)
                # train
                train_data, train_targets = torch.load(osp.join(self.processed_dir, file, "train.pt"))
                dict_users_train[file_idx] = torch.tensor(range(len(train_data_total), len(train_data_total) + len(train_data)), dtype=torch.int64)
                train_data_total = torch.cat((train_data_total, train_data), 0)
                train_targets_total = torch.cat((train_targets_total, train_targets), 0)

                # test
                test_data, test_targets = torch.load(osp.join(self.processed_dir, file, "test.pt"))
                dict_users_test[file_idx] = torch.tensor(range(len(test_data_total), len(test_data_total) + len(test_data)), dtype=torch.int64)
                test_data_total = torch.cat((test_data_total, test_data), 0)
                test_targets_total = torch.cat((test_targets_total, test_targets), 0)

                # val
                if osp.exists(osp.join(self.processed_dir, file, "val.pt")):
                    val_data, val_targets = torch.load(osp.join(self.processed_dir, file, "val.pt"))

                    dict_users_val[file_idx] = torch.tensor(range(len(val_data_total), len(val_data_total) + len(val_data)), dtype=torch.int64)
                    val_data_total = torch.cat((val_data_total, val_data), 0)
                    val_targets_total = torch.cat((val_targets_total, val_targets), 0)

        else:
            raise RuntimeError("Please delete ‘processed’ folder and try again!")

        self.dataset_train = CustomTensorDataset(name, train_data_total, train_targets_total, transform_train, target_transform)
        self.dataset_test = CustomTensorDataset(name, test_data_total, test_targets_total, transform_val, target_transform)
        self.dataset_val = CustomTensorDataset(name, val_data_total, val_targets_total, transform_val, target_transform)
        self.dict_users_train = dict_users_train
        self.dict_users_test = dict_users_test

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, f"processed-w-{self.s_frac}-{self.tr_frac}-{self.val_frac}-{self.seed}")

    @property
    def raw_file_names(self):
        names = [f"{self.name}_all_data.zip"]
        return names

    def download(self):
        # Download to `self.raw_dir`.
        url = "https://federatedscope.oss-cn-beijing.aliyuncs.com"
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            download_url(f"{url}/{name}", self.raw_dir)

    def process(self):
        raw_path = osp.join(self.raw_dir, "all_data")
        files = os.listdir(raw_path)
        files = [f for f in files if f.endswith(".json")]

        n_tasks = math.ceil(len(files) * self.s_frac)
        # random.shuffle(files)
        random.Random(self.seed).shuffle(files)
        files = files[:n_tasks]

        print("Preprocess data (Please leave enough space)...")

        idx = 0
        for num, file in enumerate(tqdm(files)):
            with open(osp.join(raw_path, file), "r") as f:
                raw_data = json.load(f)

            # Numpy to Tensor
            for writer, v in raw_data["user_data"].items():
                data, targets = v["x"], v["y"]

                if len(v["x"]) > 2:
                    data = torch.tensor(np.stack(data))
                    targets = torch.LongTensor(np.stack(targets))
                else:
                    data = torch.tensor(data)
                    targets = torch.LongTensor(targets)

                if self.name == "celeba":
                    original_size = (84, 84, 3)
                    data = data.float() / 255
                    data = [F.resize(F.to_pil_image(data[i].view(original_size).permute(2, 0, 1)), (32, 32)) for i in range(data.size(0))]
                    data = torch.stack([(F.to_tensor(data[i]) * 255).long().permute(1, 2, 0).flatten() for i in range(len(data))])

                train_data, test_data, train_targets, test_targets = train_test_split(data, targets, train_size=self.tr_frac, random_state=self.seed)

                if self.val_frac > 0:
                    val_data, test_data, val_targets, test_targets = train_test_split(
                        test_data,
                        test_targets,
                        train_size=self.val_frac / (1.0 - self.tr_frac),
                        random_state=self.seed,
                    )

                else:
                    val_data, val_targets = None, None
                save_path = osp.join(self.processed_dir, f"task_{writer}_{idx}")
                os.makedirs(save_path, exist_ok=True)

                save_local_data(
                    dir_path=save_path,
                    train_data=train_data,
                    train_targets=train_targets,
                    test_data=test_data,
                    test_targets=test_targets,
                    val_data=val_data,
                    val_targets=val_targets,
                )
                idx += 1
