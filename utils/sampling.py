#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random

import numpy as np
import torch


def iid(dataset, num_users, server_data_ratio):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    if server_data_ratio > 0.0:
        dict_users["server"] = set(np.random.choice(all_idxs, int(len(dataset) * server_data_ratio), replace=False))

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def iid_updated(dataset, num_users, server_data_ratio):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    if server_data_ratio > 0.0:
        dict_users["server"] = set(np.random.choice(all_idxs, int(len(dataset) * server_data_ratio), replace=False))

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    random.shuffle(all_idxs)
    while len(all_idxs) > 0:
        for i in range(num_users):
            if len(all_idxs) == 0:
                break
            dict_users[i].add(all_idxs.pop())

    return dict_users


# For FedBE: server data allocation
def get_server_data_idxs(dataset, dict_users, ratio_per_user=0.01, type="random", verbose=False):
    num_data_orig = sum([len(data_idxs) for data_idxs in dict_users.values()])

    if type == "random":
        server_data_idxs = []
        for client_id, data_idxs in dict_users.items():
            num_per_user = int(len(data_idxs) * ratio_per_user)
            sampled_idxs = np.random.choice(range(len(data_idxs)), num_per_user, replace=False)
            server_data_idxs.append(data_idxs[sampled_idxs])
            dict_users[client_id] = np.delete(data_idxs, sampled_idxs)
        server_data_idxs = np.concatenate(server_data_idxs)
    else:
        raise NotImplementedError("Not implemented type")

    if verbose:
        num_data = sum([len(data_idxs) for data_idxs in dict_users.values()])
        print(f"get_server_data_idxs(ratio_per_user={ratio_per_user}, type={type})")
        print(f"- Before total client data size: {num_data_orig}")
        print(f"- Server data size:              {len(server_data_idxs)}")
        print(f"- After  total client data size: {num_data}")
        print(f"- Num overlapping elements:      {len(set(list(server_data_idxs)) & set(list(np.concatenate(list(dict_users.values())))))}")

    return server_data_idxs, dict_users
