import random
from copy import deepcopy
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from utils.leaf import CusteomLEAF
from utils.sampling import iid, iid_updated

trans_mnist = transforms.Compose(
    [
        transforms.ToTensor(),  # TODO: channel is 1
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
trans_emnist = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
trans_celeba = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_fmnist = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def add_rand_transform(args, transform):
    # copt transform
    transform = deepcopy(transform)
    # find index of ToTensor
    to_tensor_idx = [i for i, t in enumerate(transform.transforms) if isinstance(t, transforms.ToTensor)][0]
    interpolationMode = transforms.InterpolationMode.BICUBIC  # NEAREST, BILINEAR, BICUBIC
    randaug = transforms.RandAugment(args.ra_n, args.ra_m, interpolation=interpolationMode)
    transform.transforms.insert(to_tensor_idx, randaug)
    return transform


def update_transform(args, transform):
    if args.ra:
        transform = add_rand_transform(args, transform)
        print("-" * 50)
        print("Add RandAugment to transform")
        print(transform)
        print("-" * 50)
    return transform


# >>> data split from pFL-Bench
# https://github.com/alibaba/FederatedScope
def _split_according_to_prior(label, client_num, prior):  # 각 class에 대해서 prior와 같은 비율로 각 client에게 나눔
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))  # 각 client의 class별 개수
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)  # 각 class당 data 수 -> shape = (classes,)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]  # return이 tuple이라 [0]을 붙여줌
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] * len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):  # ceil이니까 nums_k가 더 많을 수 있으므로 그만큼을 랜덤하게 빼줌
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=1, prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py
    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError("Only support single-label tasks!")

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f"The number of sample should be " f"greater than" f" {client_num * min_size}."
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            if client_num <= 400:
                idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))]
            else:
                idx_k_slice = [idx.tolist() for idx in np.split(idx_k, prop)]
                idxs = np.arange(len(idx_k_slice))
                np.random.shuffle(idxs)
                idx_slice = [idx_j + idx_k_slice[idx] for idx_j, idx in zip(idx_slice, idxs)]

            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])

    dict_users = {client_idx: np.array(idx_slice[client_idx]) for client_idx in range(client_num)}
    return dict_users  # idx_slice


# <<< data split from pFL-Bench


def get_data(args, env="fed"):
    total_users = args.num_users + args.ood_users

    if len(args.dataset.split(",")) > 1:
        multi_dataset_type = getattr(args, "multi_dataset_type", 2)
        dataset_list = args.dataset.split(",")
        dataset_train_list, dataset_test_list = [], []
        classes = []
        class_cnts = []
        dataset_train_len = 0
        dataset_test_len = 0
        dict_users_train_total = dict()
        dict_users_test_total = dict()
        for i in range(total_users):
            dict_users_train_total[i] = np.array([], dtype=np.int64)
            dict_users_test_total[i] = np.array([], dtype=np.int64)

        orig_num_users = args.num_users
        orig_ood_users = args.ood_users
        for dataset_idx, dataset in enumerate(dataset_list):
            args.dataset = dataset
            if multi_dataset_type == 2:
                args.num_users = orig_num_users // len(dataset_list)
                args.ood_users = orig_ood_users // len(dataset_list)
                if dataset_idx == len(dataset_list) - 1:
                    args.num_users += orig_num_users % len(dataset_list)
                    args.ood_users += orig_ood_users % len(dataset_list)
            dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args, env="pfl-bench")
            dataset_train_list.append(dataset_train)
            dataset_test_list.append(dataset_test)
            # update target
            dataset_train.targets = torch.tensor(dataset_train.targets) + len(classes)
            dataset_test.targets = torch.tensor(dataset_test.targets) + len(classes)
            # update classes
            classes += dataset_train.classes
            class_cnts.append(len(dataset_train.classes))

            # update dict_users
            if multi_dataset_type == 0:
                users_to_update = range(total_users)
            elif multi_dataset_type == 1:
                users_to_update = list(
                    range(
                        dataset_idx * (args.num_users // len(dataset_list)),
                        args.num_users if dataset_idx + 1 == len(dataset_list) else (dataset_idx + 1) * (args.num_users // len(dataset_list)),
                    )
                )
                users_to_update.extend(
                    list(
                        range(
                            args.num_users + dataset_idx * (args.ood_users // len(dataset_list)),
                            args.num_users + args.ood_users if dataset_idx + 1 == len(dataset_list) else args.num_users + (dataset_idx + 1) * (args.ood_users // len(dataset_list)),
                        )
                    )
                )
            elif multi_dataset_type == 2:
                users_to_update = range(args.num_users + args.ood_users)
            else:
                raise NotImplementedError("multi_dataset_type should be 0, 1, or 2")

            data_cnt_train = 0
            data_cnt_test = 0
            for u in users_to_update:
                user_idx = u
                if multi_dataset_type == 2:
                    if u < args.num_users:  # participating user
                        user_idx = (orig_num_users // len(dataset_list)) * dataset_idx + u
                    if args.num_users <= u:  # ood user
                        user_idx = orig_num_users + (orig_ood_users // len(dataset_list)) * dataset_idx + u - args.num_users
                print(f"[+] user {user_idx:4d} assigned {dataset}: (train: {len(dict_users_train[u])}, test: {len(dict_users_test[u])})")
                dict_users_train_total[user_idx] = np.concatenate((dict_users_train_total[user_idx], dict_users_train[u] + dataset_train_len))
                dict_users_test_total[user_idx] = np.concatenate((dict_users_test_total[user_idx], dict_users_test[u] + dataset_test_len))
                data_cnt_train += len(dict_users_train[u])
                data_cnt_test += len(dict_users_test[u])
            dataset_train_len += len(dataset_train)
            dataset_test_len += len(dataset_test)
            print(f"[+] {dataset} data size: (train: {dataset_train_len}, test: {data_cnt_test})")

        args.dataset = ",".join(dataset_list)
        args.num_users = orig_num_users
        args.ood_users = orig_ood_users
        dataset_train = ConcatDataset(dataset_train_list)
        dataset_train.classes = classes
        dataset_train.targets = torch.concat([dataset_train.targets for dataset_train in dataset_train_list])
        dataset_test = ConcatDataset(dataset_test_list)
        dataset_test.classes = classes
        dataset_test.targets = torch.concat([dataset_test.targets for dataset_test in dataset_test_list])
        print(f"Multiple dataset {args.dataset} loaded (multi_dataset_type = {multi_dataset_type})")
        print(f"users          : {len(dict_users_train_total)}")
        print(f"train data size: {sum([len(dict_users_train_total[i]) for i in dict_users_train_total])}")
        print(f"test  data size: {sum([len(dict_users_test_total[i]) for i in dict_users_test_total])}")
        print(f"classes ({len(classes)})   : {classes}")
        dataset_train.class_cnts = class_cnts
        dataset_test.class_cnts = class_cnts
        return dataset_train, dataset_test, dict_users_train_total, dict_users_test_total

    # handle single dataset
    if args.dataset == "femnist":
        trans_emnist_train_updated = update_transform(args, trans_emnist)
        leaf_dataset = CusteomLEAF(
            "data",
            "femnist",
            s_frac=getattr(args, "s_frac", 1.0 if total_users > 400 else 0.1),
            tr_frac=getattr(args, "tr_frac", 0.8),
            val_frac=getattr(args, "val_frac", 0.0),
            seed=args.seed,
            transform_train=trans_emnist_train_updated,
            transform_val=trans_emnist,
        )
        dataset_train = leaf_dataset.dataset_train
        dataset_test = leaf_dataset.dataset_test
        dict_users_train = leaf_dataset.dict_users_train
        dict_users_test = leaf_dataset.dict_users_test
        print(f"LEAF dataset {args.dataset} loaded")
        print(f"users          : {len(dict_users_train)}")
        print(f"train data size: {sum([len(dict_users_train[i]) for i in dict_users_train])}")
        print(f"test  data size: {sum([len(dict_users_test[i]) for i in dict_users_test])}")
        return dataset_train, dataset_test, dict_users_train, dict_users_test

    elif args.dataset == "celeba":
        trans_celeba_train_updated = update_transform(args, trans_celeba)
        leaf_dataset = CusteomLEAF(
            "data",
            "celeba",
            s_frac=getattr(args, "s_frac", 1.0 if total_users > 500 else 0.05),
            tr_frac=getattr(args, "tr_frac", 0.8),
            val_frac=getattr(args, "val_frac", 0.0),
            seed=args.seed,
            transform_train=trans_celeba_train_updated,
            transform_val=trans_celeba,
        )
        dataset_train = leaf_dataset.dataset_train
        dataset_test = leaf_dataset.dataset_test
        dict_users_train = leaf_dataset.dict_users_train
        dict_users_test = leaf_dataset.dict_users_test
        print(f"LEAF dataset {args.dataset} loaded")
        print(f"users          : {len(dict_users_train)}")
        print(f"train data size: {sum([len(dict_users_train[i]) for i in dict_users_train])}")
        print(f"test  data size: {sum([len(dict_users_test[i]) for i in dict_users_test])}")
        return dataset_train, dataset_test, dict_users_train, dict_users_test

    elif args.dataset == "mnist":
        trans_mnist_train_updated = update_transform(args, trans_mnist)
        dataset_train = datasets.MNIST("data/mnist/", train=True, download=True, transform=trans_mnist_train_updated)
        dataset_test = datasets.MNIST("data/mnist/", train=False, download=True, transform=trans_mnist)
    elif args.dataset == "cifar10":
        trans_cifar10_train_updated = update_transform(args, trans_cifar10_train)
        dataset_train = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=trans_cifar10_train_updated)
        dataset_test = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=trans_cifar10_val)
    elif args.dataset == "cifar100":
        trans_cifar100_train_updated = update_transform(args, trans_cifar100_train)
        dataset_train = datasets.CIFAR100("data/cifar100", train=True, download=True, transform=trans_cifar100_train_updated)
        dataset_test = datasets.CIFAR100("data/cifar100", train=False, download=True, transform=trans_cifar100_val)
    elif args.dataset == "emnist":
        trans_emnist_train_updated = update_transform(args, trans_emnist)
        dataset_train = datasets.EMNIST("data/emnist", "byclass", train=True, download=True, transform=trans_emnist_train_updated)
        dataset_test = datasets.EMNIST("data/emnist", "byclass", train=False, download=True, transform=trans_emnist)
    elif args.dataset == "fmnist":
        trans_fmnist_train_updated = update_transform(args, trans_fmnist)
        dataset_train = datasets.FashionMNIST("data/fmnist", train=True, download=True, transform=trans_fmnist_train_updated)
        dataset_test = datasets.FashionMNIST("data/fmnist", train=False, download=True, transform=trans_fmnist)
    else:
        exit("Error: unrecognized dataset")

    if args.iid:
        dict_users_train = iid(dataset_train, total_users, args.server_data_ratio)
        dict_users_test = iid(dataset_test, total_users, args.server_data_ratio)
    else:
        dict_users_train = dirichlet_distribution_noniid_slice(np.array(dataset_train.targets), total_users, args.alpha)
        if args.test_dist == "dirichlet":
            dict_users_test = dirichlet_distribution_noniid_slice(np.array(dataset_test.targets), total_users, args.alpha)  # Test를 독립적으로 샘플
        elif args.test_dist == "consistent":
            train_label_distribution = [[dataset_train.targets[idx] for idx in dict_users_train[user_idx]] for user_idx in range(total_users)]
            dict_users_test = dirichlet_distribution_noniid_slice(np.array(dataset_test.targets), total_users, args.alpha, prior=train_label_distribution)  # prior에 맞춰 샘플
        elif args.test_dist == "uniform":
            dict_users_test = iid_updated(dataset_test, total_users, args.server_data_ratio)
        else:
            raise NotImplementedError("test_dist should be dirichlet, consistent, or uniform")

        # import ipdb; ipdb.set_trace(context=5)
        # for i in [39, 79, 119]:
        #     train_bin = torch.tensor(dataset_train.targets)[dict_users_train[i]].bincount(minlength=100)
        #     test_bin = torch.tensor(dataset_test.targets)[dict_users_test[i]].bincount(minlength=100)
        #     print(f"[+] train_bin (client: {i}): \n{train_bin}")
        #     print(f"[+] test_bin  (client: {i}): \n{test_bin}")

        # ###
    return dataset_train, dataset_test, dict_users_train, dict_users_test


from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


## MAML++ utils
def get_loss_weights(inner_loop, annealing_epochs, current_epoch):
    """Code from MAML++ paper AntreasAntoniou`s Pytorch Implementation(slightly modified for integration)
    return A tensor to be used to compute the weighted average of the loss, useful for the MSL(multi step loss)
    inner_loop : MAML inner loop number
    annealing_epochs : As learning progresses, weights are increased for higher inner loop
    current_epoch : Current epoch
    """
    loss_weights = np.ones(shape=(inner_loop)) * (1.0 / inner_loop)
    decay_rate = 1.0 / inner_loop / annealing_epochs
    min_value_for_non_final_losses = 0.03 / inner_loop
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(loss_weights[-1] + (current_epoch * (inner_loop - 1) * decay_rate), 1.0 - ((inner_loop - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights)
    return loss_weights


## Per-FedAvg utils
def approximate_hessian(w_local, functional, data_batch: Tuple[torch.Tensor, torch.Tensor], grad, delta=1e-4):
    """Code from Per-FedAvg KarhouTam's Pytorch Implementation(slightly modified for integration)
    return Hessian approximation which preserves all theoretical guarantees of MAML, without requiring access to second-order information(HF-MAML)
    """
    w_local = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in w_local]
    criterion = torch.nn.CrossEntropyLoss()
    x, y = data_batch

    wt_1 = [torch.Tensor(w) for w in w_local]
    wt_2 = [torch.Tensor(w) for w in w_local]

    wt_1 = [w + delta * g for w, g in zip(wt_1, grad)]
    wt_2 = [w - delta * g for w, g in zip(wt_1, grad)]

    logit_1 = functional(wt_1, x)
    loss_1 = criterion(logit_1, y)
    grads_1 = torch.autograd.grad(loss_1, w_local)

    logit_2 = functional(wt_2, x)
    loss_2 = criterion(logit_2, y)
    grads_2 = torch.autograd.grad(loss_2, w_local)

    with torch.no_grad():
        grads_2nd = deepcopy(grads_1)
        for g_2nd, g1, g2 in zip(grads_2nd, grads_1, grads_2):
            g_2nd.data = (g1 - g2) / (2 * delta)

    return grads_2nd


def clip_norm_(grads, max_norm, norm_type: float = 2.0):
    """This code is based on torch.nn.utils.clip_grad_norm_(inplace function that does gradient clipping to max_norm).
    the input of torch.nn.utils.clip_grad_norm_ is parameters
    but the input of clip_norm_ is list of gradients
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm


# def clip_norm_coef(grads, max_norm, norm_type: float = 2.0):
#     max_norm = float(max_norm)
#     norm_type = float(norm_type)

#     device = grads[0].device
#     total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
#     clip_coef = max_norm / (total_norm  + 1e-6)

#     clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

#     return clip_coef_clamped.to(device)


def clip_norm_coef(grads, max_norm, norm_type: float = 2.0):
    """This code looks similar to torch.nn.utils.clip_grad_norm_ and clip_norm_,
    but it is very different because it does not detach grads(important to MAML algorithm).
    return A scalar coefficient that normalizes the norm of gradients to the max_norm
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)


def clip_norm_coef_wo_logit(grads, max_norm, norm_type: float = 2.0):
    """This code looks similar to torch.nn.utils.clip_grad_norm_ and clip_norm_,
    but it is very different because it does not detach grads(important to MAML algorithm).
    return A scalar coefficient that normalizes the norm of gradients to the max_norm
    """
    logit_layer_num = 10  # to exclude NVDP logit layer
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for i, g in enumerate(grads) if i != logit_layer_num]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)


def calc_bins(preds, labels_oneh):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(preds, labels_oneh):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(preds, labels_oneh):
    #   import ipdb; ipdb.set_trace(context=5)
    ECE, MCE = get_metrics(preds, labels_oneh)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor="black", color="r", hatch="\\")

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor="black", color="b")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect("equal", adjustable="box")

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color="green", label="ECE = {:.2f}%".format(ECE * 100))
    MCE_patch = mpatches.Patch(color="red", label="MCE = {:.2f}%".format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    # plt.show()
    # plt.savefig('calibrated_network.png', bbox_inches='tight')
    # plt.close(fig)
    return fig, ECE, MCE


def plot_data_partition(dataset, dict_users, num_classes, num_sample_users, writer=None, tag="Data Partition"):
    dict_users_targets = {}
    targets = np.array(dataset.targets)

    dict_users_targets = {client_idx: targets[data_idxs] for client_idx, data_idxs in dict_users.items()}

    s = torch.stack([torch.bincount(torch.tensor(data_idxs), minlength=num_classes) for client_idx, data_idxs in dict_users_targets.items()])
    ss = torch.cumsum(s, 1)
    cmap = plt.cm.get_cmap("hsv", num_classes)
    fig, ax = plt.subplots(figsize=(20, num_sample_users))
    ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, 0], color=cmap(0))
    for c in range(1, num_classes):
        ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, c], left=ss[:num_sample_users, c - 1], color=cmap(c))
    # plt.show()
    if writer is not None:
        writer.add_figure(tag, fig)
