import os
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn

from utils.train_utils import get_data, set_seed

# from utils.models import FCmodel, CNNCifar, CNNCifarVD, CNNCifarNVDP
# # from utils.HyperNetClasses import IdentityNet
# from utils.HyperNetClasses import IdentityModel, VDModel, NVDPModel

class BaseModel:
    def __init__(self, args):

        #### TensorBoard Writer ####
        args.log_dir = f'./runs/{args.pre}/{args.dataset}/log' + '/' + args.exp
        args.ckpt_dir = f'./runs/{args.pre}/{args.dataset}/ckpt' + '/' + args.exp
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
        self.writer = SummaryWriter(args.log_dir)
        #############################

        # # Get Data
        # self.dataset_train, self.dataset_test, self.dict_users_train, self.dict_users_test = get_data(args)
        # self.num_classes = len(self.dataset_train.classes)

        self.criteria = torch.nn.CrossEntropyLoss()

        self.args = args

    def train_report(self,iter,val_loss_list, val_acc_list, val_kl_list, wt, user_idxs, users_datasize):
            # weighted average
            weights_size = []
            for idx in user_idxs:
                weights_size.append(users_datasize[idx])
            weights = torch.Tensor(weights_size) / sum(weights_size)
            self.writer.add_scalar('train/weighted_loss', np.average(val_loss_list, weights=weights), iter)
            self.writer.add_scalar('train/weighted_kl', np.average(val_kl_list, weights=weights), iter)
            self.writer.add_scalar('train/weighted_acc', np.average(val_acc_list, weights=weights)*100, iter)
            
            #  uniform average
            self.writer.add_scalar('train/AVG_loss', np.array(val_loss_list).mean(), iter)
            self.writer.add_scalar('train/AVG_kl', np.array(val_kl_list).mean(), iter)
            self.writer.add_scalar('train/AVG_acc', np.array(val_acc_list).mean()*100, iter)

            if self.args.algorithm in ['mamlgaus', 'mamlgausq', 'vdgaus', 'vdgausem', 'nvdpgaus', 'nvdpgausv2',  'nvdpgausq', 'perfedavgnvdpgaus', 'perfedavgnvdpgausq']:
                with torch.no_grad():
                    alpha = ( wt[10].clamp(-8.5,8.5).exp() / (wt[8]**2 + 1e-10) ).view(1,-1)
                    probs = alpha / (1+alpha)
                    self.writer.add_histogram('train/probs', probs , iter)

            if self.args.algorithm in ['nvdpgausqplus']:
                with torch.no_grad():
                    alpha = ( wt[11].clamp(-8.5,8.5).exp() / (wt[9]**2 + 1e-10) ).view(1,-1)
                    probs = alpha / (1+alpha)
                    self.writer.add_histogram('train/probs', probs , iter)

            # if self.args.algrithm in ['nvdpgausqplus']:
            #      with torch.no_grad():
            #         alpha = ( wt[10].clamp(-8.5,8.5).exp() / (wt[8]**2 + 1e-10) ).view(1,-1)
            #         probs = alpha / (1+alpha)
            #         self.writer.add_histogram('train/probs', probs , iter)

            self.writer.flush()

    @torch.no_grad()
    def client_test(self, ldr_test, wt):
        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()
                y_pred = self.functional(wt, x.cuda())
                loss = self.criteria(y_pred, y)
                val_loss += loss.mean().item()
                val_acc += y_pred.argmax(1).eq(y).sum().item() / len(y)

            val_loss /= (batch_idx+1)
            val_acc /= (batch_idx+1)

        return  val_loss, val_acc

    @torch.no_grad()
    def client_test_multi(self, ldr_test, wt):
        with torch.no_grad():
            dataset_list = self.args.dataset.split(',')
            class_cnts = ldr_test.dataset.dataset.class_cnts
            # dataset_labels: {dataset: [min_label, max_label+1]}
            dataset_labels = {dataset:[sum(class_cnts[:i]), sum(class_cnts[:i+1])] for i, dataset in enumerate(dataset_list)}
            multi_accs = {dataset:0 for dataset in dataset_list}
            multi_cnts = {dataset:0 for dataset in dataset_list}

            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()
                y_pred = self.functional(wt, x.cuda())
                # multi accs
                for dataset in dataset_list:
                    dataset_label = dataset_labels[dataset]
                    idxs = (y>=dataset_label[0]) & (y<dataset_label[1])
                    multi_accs[dataset] += y_pred.argmax(1).eq(y)[idxs].sum().item()
                    multi_cnts[dataset] += idxs.sum().item()
            
            for dataset in dataset_list:
                multi_accs[dataset] /= multi_cnts[dataset]
            assert sum(multi_cnts.values()) == len(ldr_test.dataset)
        return  multi_accs, multi_cnts


    def client_test_with_calibration_multi(self, ldr_test, wt):

        preds = []
        labels_oneh = []
        sm = nn.Softmax(dim=1)

        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            dataset_list = self.args.dataset.split(',')
            class_cnts = ldr_test.dataset.dataset.class_cnts
            dataset_labels = {dataset:[sum(class_cnts[:i]), sum(class_cnts[:i+1])] for i, dataset in enumerate(dataset_list)}
            multi_accs = {dataset:0 for dataset in dataset_list}
            multi_cnts = {dataset:0 for dataset in dataset_list}

            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()
                y_pred = self.functional(wt, x.cuda())
                loss = self.criteria(y_pred, y)
                val_loss += loss.mean().item()
                val_acc += y_pred.argmax(1).eq(y).sum().item() / len(y)

                ## ADDED for calibration
                # pred = y_pred.cpu().detach().numpy()
                label_oneh = torch.nn.functional.one_hot(labels, num_classes=len(ldr_test.dataset.dataset.classes))
                # label_oneh = label_oneh.cpu().detach().numpy()

                preds.extend(sm(y_pred).cpu())
                labels_oneh.extend(label_oneh)
                
                # multi accs
                for dataset in dataset_list:
                    dataset_label = dataset_labels[dataset]
                    idxs = (y>=dataset_label[0]) & (y<dataset_label[1])
                    multi_accs[dataset] += y_pred.argmax(1).eq(y)[idxs].sum().item()
                    multi_cnts[dataset] += idxs.sum().item()

            val_loss /= (batch_idx+1)
            val_acc /= (batch_idx+1)

            # preds = np.array(preds).flatten()
            # labels_oneh = np.array(labels_oneh).flatten()
            
            for dataset in dataset_list:
                multi_accs[dataset] /= multi_cnts[dataset]
            assert sum(multi_cnts.values()) == len(ldr_test.dataset)

        return  val_loss, val_acc, torch.cat(preds), torch.cat(labels_oneh), multi_accs, multi_cnts
    

    def client_test_with_calibration(self, ldr_test, wt):

        preds = []
        labels_oneh = []
        sm = nn.Softmax(dim=1)

        with torch.no_grad():
            val_loss = 0
            val_acc = 0

            ## Then we perform validation on the rest of data
            for batch_idx, (images, labels) in enumerate(ldr_test):
                x = images.cuda()
                y = labels.cuda()
                y_pred = self.functional(wt, x.cuda())
                loss = self.criteria(y_pred, y)
                val_loss += loss.mean().item()
                val_acc += y_pred.argmax(1).eq(y).sum().item() / len(y)

                ## ADDED for calibration
                # pred = y_pred.cpu().detach().numpy()
                label_oneh = torch.nn.functional.one_hot(labels, num_classes=len(ldr_test.dataset.dataset.classes))
                # label_oneh = label_oneh.cpu().detach().numpy()

                preds.extend(sm(y_pred).cpu())
                labels_oneh.extend(label_oneh)

            val_loss /= (batch_idx+1)
            val_acc /= (batch_idx+1)

            # preds = np.array(preds).flatten()
            # labels_oneh = np.array(labels_oneh).flatten()

        return  val_loss, val_acc, torch.cat(preds), torch.cat(labels_oneh)


    def report_probs_embedding(self, iter, num_users=None):
        # user_idxs=np.random.choice(range(self.args.num_users), num_users if num_users else self.args.num_users, replace=False)
        # user_idxs=sorted(user_idxs)

        with torch.no_grad():
            user_idxs = range(0, self.args.num_users)
            probs_total = torch.tensor([]).cuda()
            for idx in user_idxs:
                w_local=self.hpnet(idx)
                # wt=[torch.Tensor(w) for w in w_local]
                alpha = ( w_local[10].clamp(-8.5, 8.5).exp() / (w_local[8]**2 + 1e-10) ).view(1,-1)
                probs = alpha / (1+alpha)
                probs_total = torch.cat((probs_total, probs))
            # self.writer.add_embedding(probs_total, metadata=user_idxs, global_step=iter, tag='probs_embedding')

            cos_val = 0
            counter = 0
            # cos_max = torch.zeros(100,100)
            for i in range(0, len(probs_total)):
                for j in range(i, len(probs_total)):
                    # cos_max[i,j] = F.cosine_similarity(probs_total[i], probs_total[j], dim=0)
                    cos_val += F.cosine_similarity(probs_total[i], probs_total[j], dim=0)
                    counter += 1

            mean_cos_similarity = cos_val / counter

            self.writer.add_scalar('test/cos_similarity', mean_cos_similarity.data , iter)
            self.writer.flush()

            # cos = torch.nn.CosineSimilarity(dim=1)
            # v_mean = probs_total.mean(0)
            # output = 0
            # for v in probs_total:
            #     output += torch.dot(v, v_mean)
            # similarity_score = output / len(probs_total)

            # self.writer.add_scalar('test/probs_difference', similarity_score , iter)
            # self.writer.flush()



