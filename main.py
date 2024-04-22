#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from models.FedAvg import FedAvg
from models.FedAvgNvdpGausQ import FedAvgNvdpGausQ
from models.FedAvgPer import FedAvgPer
from models.FedAvgSnip import FedAvgSnip
from models.FedBE import FedBE
from models.FedProx import FedProx
from models.Maml import Maml
from models.MamlGausQ import MamlGausQ
from models.MamlSnip import MamlSnip
from models.NvdpGausQ import NvdpGausQ
from models.PerFedAvg import PerFedAvg
from models.PerFedAvgNvdpGausQ import PerFedAvgNvdpGausQ
from models.PerFedAvgSnip import PerFedAvgSnip
from models.Reptile import Reptile
from models.ReptileSnip import ReptileSnip
from models.VdGausEmQ import VdGausEmQ
from models.VdGausQ import VdGausQ
from utils.cyclical_annealing import CyclicalScheduler
from utils.sampling import get_server_data_idxs
from utils.train_utils import DatasetSplit, draw_reliability_graph, get_data, plot_data_partition, set_seed


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=777)

    # model
    parser.add_argument("--algorithm", type=str, default="reptile", choices=["fedavg", "fedavgnvdpgausq", "fedavgper", "fedavgsnip", "fedbe", "fedprox", "maml", "mamlgausq", "mamlsnip", "nvdpgausq", "perfedavg", "perfedavgnvdpgausq", "perfedavgsnip", "reptile", "reptilesnip", "vdgausemq", "vdgausq"], help="name of algorithm. use lower case")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=200)

    # federated arguments
    parser.add_argument("--num_rounds", type=int, default=1000, help="rounds of training")
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument("--ood_users", type=int, default=30)
    parser.add_argument("--frac_m", type=int, default=10, help="the nuumber of clients: C")

    parser.add_argument("--server_lr", type=float, default=0.9, help="parameter for proximal global SGD")
    parser.add_argument("--inner_lr", type=float, default=0.02, help="parameter for proximal local SGD")

    parser.add_argument("--local_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--local_epochs", type=int, default=5, help="the number of local SGD epochs in local epoch")

    parser.add_argument("--adaptation_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--adaptation_epochs", type=int, default=1)
    parser.add_argument("--adaptation_steps", type=int, default=1, help="negative value == 1 epoch.")

    # Reptile
    parser.add_argument("--momentum", type=float, default=0.9, help="server optimizer momentum")

    # MAML
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--client_lr", type=float, default=0.1, help="parameter for proximal client SGD")

    # NVDP
    parser.add_argument("--beta", type=float, default=10)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--drop_prob", type=float, default=0.9)
    parser.add_argument("--dropstart", type=float, default=1000)

    parser.add_argument("--momentum1", type=float, default=0.6, help="parameter for proximal global SGD")
    parser.add_argument("--decay1", type=float, default=5e-4, help="parameter for proximal local SGD")
    parser.add_argument("--momentum2", type=float, default=0.3, help="parameter for proximal global SGD")
    parser.add_argument("--decay2", type=float, default=5e-4, help="parameter for proximal local SGD")

    # fedprox
    parser.add_argument("--mu", type=float, default=0.1)

    # FedBE
    parser.add_argument("--num_teacher_samples", type=int, default=10)
    parser.add_argument("--ratio_per_user", type=float, default=0.04, help="FedBE: ratio of server data per user")
    parser.add_argument("--no_client_average", action="store_true")  # A
    parser.add_argument("--no_clients", action="store_true")  # C
    parser.add_argument("--no_teacher_samples", action="store_true")  # S
    parser.add_argument("--server_steps", type=int, default=72)
    parser.add_argument("--dist_type", type=int, default=2)
    parser.add_argument("--min_server_lr", type=float, default=9e-5, help="")
    parser.add_argument("--swa_lr_type", type=int, default=2)
    parser.add_argument("--swa_start_step", type=int, default=36)
    parser.add_argument("--swa_period", type=int, default=3)
    parser.add_argument("--server_bs", type=int, default=64)

    # pfl-bench
    parser.add_argument("--alpha", type=float, default=0.1)

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["femnist", "celeba", "mnist", "cifar10", "cifar100", "emnist", "fmnist"], help="name of dataset")  ### DATA
    parser.add_argument("--multi_dataset_type", type=int, default=0)
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")  ### DATA
    parser.add_argument("--server_data_ratio", type=float, default=0.0, help="The percentage of data that servers also have across data of all clients.")
    parser.add_argument("--test_dist", type=str, default="dirichlet", choices=["uniform", "dirichlet", "consistent"])
    parser.add_argument("--ra", action="store_true", help="RandAugment")
    parser.add_argument("--ra_n", type=int, default=2)
    parser.add_argument("--ra_m", type=int, default=9)

    # optimization
    parser.add_argument("--use_cyclical_beta", type=bool, default=True, help="using cyclical beta")
    parser.add_argument("--temp_type", type=str, default="sigmoid", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--temp_start", type=int, default=0)
    parser.add_argument("--temp_cycle", type=int, default=1)
    parser.add_argument("--temp_ratio", type=float, default=0.02)

    # SNIP
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--droplayer", nargs="+", default=[8])  ## ex: --droplayer 8 or --droplayer 8 10
    parser.add_argument("--pre", type=str, default="", help="prefix of experiment name")

    args = parser.parse_args()
    return args


def main(args):
    # Set Seed
    set_seed(args.seed)

    # Get Data
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args, env="pfl-bench")

    if "fedbe" in args.algorithm.lower():
        server_data_idxs, dict_users_train = get_server_data_idxs(dataset_train, dict_users_train, ratio_per_user=args.ratio_per_user, type="random", verbose=True)
    else:
        server_data_idxs = []

    users_datasize = [len(dd) for dd in dict_users_train.values()]
    num_classes = len(dataset_train.classes)

    ml_algorithms = {
        "Fedavg": FedAvg,
        "Fedavgnvdpgausq": FedAvgNvdpGausQ,
        "Fedavgper": FedAvgPer,
        "Fedprox": FedProx,
        "Fedbe": FedBE,
        "Reptile": Reptile,
        "Maml": Maml,
        "Mamlgausq": MamlGausQ,
        "Perfedavg": PerFedAvg,
        "Perfedavgnvdpgausq": PerFedAvgNvdpGausQ,
        "Vdgausq": VdGausQ,
        "Vdgausemq": VdGausEmQ,
        "Nvdpgausq": NvdpGausQ,
        "Fedavgsnip": FedAvgSnip,
        "Reptilesnip": ReptileSnip,
        "Mamlsnip": MamlSnip,
        "Perfedavgsnip": PerFedAvgSnip,
    }

    ml = ml_algorithms[args.algorithm.capitalize()](args, num_classes)
    # plot_data_partition(dataset_train, dict_users_train, num_classes, args.num_users + args.ood_users, writer=ml.writer, tag="train/data_partition") #! visualize data partition
    # plot_data_partition(dataset_test, dict_users_test, num_classes, args.num_users + args.ood_users, writer=ml.writer, tag="test/data_partition") #! visualize data partition

    val_loss_list = []
    val_kl_list = []
    val_acc_list = []
    val_oodacc_list = []

    if ml.args.use_cyclical_beta is True:
        temp_sced = CyclicalScheduler(type=ml.args.temp_type, start=ml.args.temp_start, stop=ml.args.beta, n_step=ml.args.num_rounds, cycle=ml.args.temp_cycle, ratio=ml.args.temp_ratio)

    step_iter = trange(ml.args.num_rounds)
    for iter in step_iter:
        if ml.args.use_cyclical_beta is True:
            ml.args.beta = temp_sced(iter)
            ml.writer.add_scalar("train/beta", ml.args.beta, iter)

        # Server choose clients to train
        user_idxs = np.random.choice(range(ml.args.num_users), ml.args.frac_m, replace=False)
        collected_weights = []

        val_loss_list.clear()
        val_kl_list.clear()
        val_acc_list.clear()

        for idx in user_idxs:
            ################################## Client #####################################
            ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=ml.args.local_bs, shuffle=True)  # drop_last=True (?)

            # Copy weight from SERVER to Client
            # wt = ml.hpnet(idx) # Weight Sampling from posteiror wt ~ q(w*)
            wt = ml.hpnet(idx) if iter < ml.args.dropstart else ml.hpnet(idx, ood=False, drop_prob=ml.args.drop_prob)  ## if you dont want to drop weights, set dropstart >= num_rounds
            w_local = [w.detach().clone() for w in wt]

            val_loss = 0
            val_acc = 0
            val_kl = 0

            for i in range(ml.args.local_epochs):
                wt, mean_loss, mean_acc, mean_kl = ml.client_update(ldr_train, wt, w_local)
                val_loss += mean_loss
                val_acc += mean_acc
                val_kl += mean_kl

            val_loss /= ml.args.local_epochs
            val_acc /= ml.args.local_epochs
            val_kl /= ml.args.local_epochs

            val_loss_list.append(val_loss.item())
            val_kl_list.append(val_kl.item())
            val_acc_list.append(val_acc)

            ############################################################################
            collected_weights.append([w.detach().clone() for w in wt])
            # collected_weights.append( [torch.Tensor(w.data).detach().clone() for w in wt] )

        ################################ Server ########################################
        ml.server_aggregation(user_idxs, users_datasize, [collected_weights, [], iter, ml.args.drop_prob, dataset_train, server_data_idxs])  #

        step_iter.set_description(
            f"Step:{iter}, KLD:{np.array(val_kl_list).mean():.4f}, Beta:{ml.args.beta}, AVG Loss: {np.array(val_loss_list).mean():.4f},  AVG Acc: {np.array(val_acc_list).mean():.4f}"
        )

        if iter == 0:
            if ml.args.algorithm in ["fedavgsnip", "reptilesnip", "mamlsnip", "perfedavgsnip"]:
                ml.snip(user_idxs, users_datasize, [collected_weights, [], iter, ml.args.drop_prob, dataset_train, server_data_idxs])  #

        if iter % 10 == 0 or iter == (ml.args.num_rounds - 1):
            wt = ml.hpnet(idx)  # if iter <= ml.args.dropstart else ml.hpnet(idx, ood=False, drop_prob=ml.args.drop_prob)
            ml.train_report(iter, val_loss_list, val_acc_list, val_kl_list, wt, user_idxs, users_datasize)

        if iter % 20 == 0 or iter == (ml.args.num_rounds - 1):  # Testing
            val_loss_list.clear()
            val_kl_list.clear()
            val_acc_list.clear()

            preds_list = []
            labels_oneh_list = []

            if iter == (ml.args.num_rounds - 1):
                user_idxs = range(ml.args.num_users)
                set_seed(args.seed)  # ! data loader seed setting
            else:
                num_test_users = 20
                user_idxs = np.random.choice(range(ml.args.num_users), num_test_users, replace=False)

            for idx in user_idxs:
                if ml.args.algorithm in [
                    "fedavgnvdpgausq",
                    "mamlgausq",
                    "vdgausq",
                    "vdgausemq",
                    "nvdpgausq",
                    "perfedavgnvdpgausq",
                ]:
                    wt = ml.hpnet(idx, ood=True)
                else:
                    # w_local = ml.hpnet(idx) if iter < ml.args.dropstart else ml.hpnet(idx, ood=False, drop_prob=ml.args.drop_prob)
                    wt = ml.hpnet(idx)

                ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=ml.args.adaptation_bs, shuffle=True)
                ldr_test = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size=100, shuffle=True)

                for i in range(args.adaptation_epochs):
                    wt = ml.client_adapt(ldr_train, wt)

                if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                    mean_loss, mean_acc, preds, labels_oneh = ml.client_test_with_calibration(ldr_test, wt)
                    preds_list.append(preds)
                    labels_oneh_list.append(labels_oneh)
                else:
                    mean_loss, mean_acc = ml.client_test(ldr_test, wt)

                val_loss_list.append(mean_loss)
                val_acc_list.append(mean_acc)

            if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                preds_total = torch.cat(preds_list).numpy()
                labels_total = torch.cat(labels_oneh_list).numpy()
                fig, ECE, MCE = draw_reliability_graph(preds_total, labels_total)
                ml.writer.add_scalar("test/ECE", ECE * 100, iter)
                ml.writer.add_scalar("test/MCE", MCE * 100, iter)
                ml.writer.add_figure("test/calibrated_network", fig, iter, close=True)

            print(f"Step:{iter}, (test) AVG Loss: {np.array(val_loss_list).mean():.4f}, (test) AVG Acc: {np.array(val_acc_list).mean():.4f}")

            #  uniform average
            ml.writer.add_scalar("test/AVG_loss", np.array(val_loss_list).mean(), iter)
            ml.writer.add_scalar("test/AVG_acc", np.array(val_acc_list).mean() * 100, iter)
            ml.writer.add_scalar("test/least_acc", np.array(val_acc_list).min() * 100, iter)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            ml.writer.add_scalar("test/weighted_loss", np.average(val_loss_list, weights=weights), iter)
            ml.writer.add_scalar("test/weighted_acc", np.average(val_acc_list, weights=weights) * 100, iter)

            # multi dataset (single_domain)
            if getattr(args, "multi_dataset_type", 0):
                dataset_list = args.dataset.split(",")
                for dataset_idx, dataset in enumerate(dataset_list):
                    user_idxs_dataset = []
                    val_acc_list_dataset = []
                    min_idx = dataset_idx * (args.num_users // len(dataset_list))
                    max_idx = args.num_users if dataset_idx + 1 == len(dataset_list) else (dataset_idx + 1) * (args.num_users // len(dataset_list))
                    for i, idx in enumerate(user_idxs):
                        if min_idx <= idx < max_idx:
                            user_idxs_dataset.append(idx)
                            val_acc_list_dataset.append(val_acc_list[i])
                    weights_size_dataset = [users_datasize[idx] for idx in user_idxs_dataset]
                    weights_dataset = torch.Tensor(weights_size_dataset) / sum(weights_size_dataset)
                    print(f"[+] (test) multi acc ({dataset}): {user_idxs_dataset}")
                    ml.writer.add_scalar(f"test/weighted_acc_{dataset}", np.average(val_acc_list_dataset, weights=weights_dataset) * 100, iter)

            ml.writer.flush()

        if iter % 100 == 0 or iter == (ml.args.num_rounds - 1):  # OOD
            val_loss_list.clear()
            val_oodacc_list.clear()

            preds_list = []
            labels_oneh_list = []

            if iter == (ml.args.num_rounds - 1):
                set_seed(args.seed)  # ! data loader seed setting

            user_idxs = range(ml.args.num_users, ml.args.num_users + ml.args.ood_users)
            for idx in user_idxs:
                ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=ml.args.adaptation_bs, shuffle=True)
                ldr_test = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size=100, shuffle=True)

                if ml.args.algorithm in [
                    "fedavgnvdpgausq",
                    "mamlgausq",
                    "vdgausq",
                    "vdgausemq",
                    "nvdpgausq",
                    "perfedavgnvdpgausq",
                ]:
                    wt = ml.hpnet(idx, ood=True)
                else:
                    wt = ml.hpnet(idx)  # Weight Sampling from posteiror wt ~ q(w*)

                for i in range(args.adaptation_epochs):
                    wt = ml.client_adapt(ldr_train, wt)

                if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                    mean_loss, mean_acc, preds, labels_oneh = ml.client_test_with_calibration(ldr_test, wt)
                    preds_list.append(preds)
                    labels_oneh_list.append(labels_oneh)
                else:
                    mean_loss, mean_acc = ml.client_test(ldr_test, wt)

                val_loss_list.append(mean_loss)
                val_oodacc_list.append(mean_acc)

            if iter == (ml.args.num_rounds - 1):  # Measure Calibration
                preds_total = torch.cat(preds_list).numpy()
                labels_total = torch.cat(labels_oneh_list).numpy()
                fig, ECE, MCE = draw_reliability_graph(preds_total, labels_total)
                ml.writer.add_scalar("OOD/ECE", ECE * 100, iter)
                ml.writer.add_scalar("OOD/MCE", MCE * 100, iter)
                ml.writer.add_figure("OOD/calibrated_network", fig, iter, close=True)

                ml.writer.add_scalar("OOD/GAP", (np.array(val_oodacc_list).mean() - np.array(val_acc_list).mean()) * 100, iter)  # TODO !!

            print(f"Step:{iter}, (OOD) AVG Loss: {np.array(val_loss_list).mean():.4f}, (OOD) AVG Acc: {np.array(val_oodacc_list).mean():.4f}")
            #  uniform average
            ml.writer.add_scalar("OOD/AVG_loss", np.array(val_loss_list).mean(), iter)
            ml.writer.add_scalar("OOD/AVG_acc", np.array(val_oodacc_list).mean() * 100, iter)
            ml.writer.add_scalar("OOD/least_acc", np.array(val_acc_list).min() * 100, iter)

            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights2 = torch.Tensor(weights_size) / sum(weights_size)
            ml.writer.add_scalar("OOD/weighted_loss", np.average(val_loss_list, weights=weights2), iter)
            ml.writer.add_scalar("OOD/weighted_acc", np.average(val_oodacc_list, weights=weights2) * 100, iter)

            # multi dataset (single_domain)
            if getattr(args, "multi_dataset_type", 0):
                dataset_list = args.dataset.split(",")
                for dataset_idx, dataset in enumerate(dataset_list):
                    user_idxs_dataset = []
                    val_oodacc_list_dataset = []
                    min_idx = args.num_users + dataset_idx * (args.ood_users // len(dataset_list))
                    max_idx = args.num_users + args.ood_users if dataset_idx + 1 == len(dataset_list) else args.num_users + (dataset_idx + 1) * (args.ood_users // len(dataset_list))
                    for i, idx in enumerate(user_idxs):
                        if min_idx <= idx < max_idx:
                            user_idxs_dataset.append(idx)
                            val_oodacc_list_dataset.append(val_oodacc_list[i])
                    weights_size_dataset = [users_datasize[idx] for idx in user_idxs_dataset]
                    weights_dataset = torch.Tensor(weights_size_dataset) / sum(weights_size_dataset)
                    print(f"[+] (OOD) multi acc ({dataset}): {user_idxs_dataset}")
                    ml.writer.add_scalar(f"OOD/weighted_acc_{dataset}", np.average(val_oodacc_list_dataset, weights=weights_dataset) * 100, iter)

            ml.writer.flush()

        if iter == (ml.args.num_rounds - 1):
            torch.save({"args": ml.args, "model": ml.model.state_dict(), "hpnet": ml.hpnet.state_dict()}, ml.args.ckpt_dir + "/ckpt.pt")

    torch.cuda.empty_cache()
    # return np.array(val_acc_list).mean()*100, np.array(val_oodacc_list).mean()*100
    return np.average(val_acc_list, weights=weights) * 100, np.average(val_oodacc_list, weights=weights2) * 100


if __name__ == "__main__":
    args = args_parser()
    main(args)
