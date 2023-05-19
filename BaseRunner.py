# coding=utf-8

import torch
import logging
from time import time

import torch.nn.functional as F
# from utils import utils, global_p
from tqdm import tqdm
import numpy as np
import os
import copy
import datetime

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, accuracy_score

import controller,loss_formula
from ltr import NeuralRanker, load_multiple_data, metric_results_to_string, evaluation, \
    DataProcessor, RankMSE


class baserunner():
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="AUC",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        parser.add_argument('--skip_rate', type=float, default=1.005, help='bad loss skip rate')
        parser.add_argument('--rej_rate', type=float, default=1.005, help='bad training reject rate')
        parser.add_argument('--skip_lim', type=float, default=1e-5, help='bad loss skip limit')
        parser.add_argument('--rej_lim', type=float, default=1e-5, help='bad training reject limit')
        parser.add_argument('--lower_bound_zero_gradient', type=float, default=1e-4,
                            help='bound to check zero gradient')
        parser.add_argument('--search_train_epoch', type=int, default=1,help='epoch num for training when searching loss')
        parser.add_argument('--step_train_epoch', type=int, default=1, help='epoch num for training each step')

        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='AUC,RMSE', check_epoch=10, early_stop=1, controller=None,
                 loss_formula=None,controller_optimizer=None, args=None, gpu=True, device="cuda:0"):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2

        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        self.train_results, self.valid_results, self.test_results = [], [], []

        self.controller = controller
        self.loss_formula = loss_formula
        self.controller_optimizer = controller_optimizer
        self.args = args
        self.print_prediction = {}
        self.gpu, self.device = gpu, device



    # def _build_optimizer(self, model):
    #     optimizer_name = self.optimizer_name.lower()
    #     if optimizer_name == 'gd':
    #         logging.info("Optimizer: GD")
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
    #     # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
    #     elif optimizer_name == 'adagrad':
    #         logging.info("Optimizer: Adagrad")
    #         optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
    #     # optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
    #     elif optimizer_name == 'adam':
    #         logging.info("Optimizer: Adam")
    #         optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
    #     # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    #     else:
    #         logging.error("Unknown Optimizer: " + self.optimizer_name)
    #         assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
    #     # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
    #     return optimizer
    # def fit(self, model, data, data_processor, epoch=-1, loss_fun=None, sample_arc=None,
    #         regularizer=True):  # fit the results for an input set
    #
    #     if model.optimizer is None:
    #         model.optimizer = self._build_optimizer(model)
    #     batches = data_processor.prepare_batches(data, self.batch_size, train=True)
    #     batches = self.batches_add_control(batches, train=True)
    #     batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
    #     model.train()  # tensorflow,nnmodel中,train() model.
    #     accumulate_size = 0
    #     to_show = batches if self.args.search_loss else tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
    #                                                          ncols=100, mininterval=1)
    #     for batch in to_show:
    #         accumulate_size += len(batch['Y'])
    #         model.optimizer.zero_grad()
    #         output_dict = model(batch)
    #         loss = output_dict['loss'] + model.l2() * self.l2_weight
    #         if loss_fun is not None and sample_arc is not None:
    #             loss = loss_fun(output_dict['prediction'], batch['Y'], sample_arc)
    #             if regularizer:
    #                 loss += model.l2() * self.l2_weight
    #         loss.backward()
    #         torch.nn.utils.clip_grad_value_(model.parameters(), 50)
    #         if accumulate_size >= batch_size or batch is batches[-1]:
    #             model.optimizer.step()
    #             accumulate_size = 0
    #     model.eval()
    #     return output_dict

    def fit(self,model,data,loss_fun=None,sample_arc=None, regularizer=True):
        epoch = 10
        num_queries = 0
        model.train_mode()
        for step in range(epoch):
            for batch_ids, batch_q_doc_vectors, batch_std_labels in data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]

                num_queries += len(batch_ids)
                if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)
                batch_preds = model(batch_q_doc_vectors)
                model.optimizer.zero_grad()
                # if loss_fun is None:
                #     for epoch_k in range(1, epoch + 1):
                #         torch_fold_k_epoch_k_loss = model.train(train_data=data, epoch_k=epoch_k, presort=True)

                if loss_fun is not None and sample_arc is not None:
                    loss = loss_fun(batch_preds, batch_std_labels, sample_arc)
                    if regularizer:
                        loss += model.l2() * self.l2_weight
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 50)
                model.optimizer.step()
        model.eval_mode()











    def train(self, search_loss=False,data_id=None, dir_data=None, model_id=None):

        model=RankMSE()
        model.init()  # initialize or reset with the same random initialization
        # if torch.cuda.device_count() > 0:
        #     # model = model.to('cuda:0')
        #     model = model.cuda()

        train_data, test_data,validation_data = DataProcessor(data_id, dir_data)
        epochs=1
        min_reward = torch.tensor(-1.0).cuda()
        train_with_optim = False
        try:
            for epoch in range(1, epochs + 1):
                self.loss_formula.eval()
                self.controller.zero_grad()

                if search_loss:
                    start_auc=pre_evaluate(model,test_data)
                    baseline = torch.tensor(start_auc).cuda()
                    cur_model= copy.deepcopy(model)
                    grad_dict = dict()
                    test_pred = torch.rand(20).cuda() * 0.8 + 0.1  # change range here
                    test_label = torch.rand(20).cuda()
                    test_pred.requires_grad = True
                    max_reward = min_reward.clone().detach()
                    best_arc = None
                    for i in range(10):
                        while True:
                            reward = None
                            self.controller()  # perform forward pass to generate a new architecture
                            sample_arc = self.controller.sample_arc
                            if test_pred.grad is not None:
                                test_pred.grad.data.zero_()
                            test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
                            try:
                                test_loss.backward()
                            except RuntimeError:
                                pass
                            if test_pred.grad is None or torch.norm(test_pred.grad,float('inf')) < self.args.lower_bound_zero_gradient:
                                reward = min_reward.clone().detach()
                            if reward is None:
                                for key, value in grad_dict.items():
                                    if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
                                        reward = value.clone().detach()
                                        break
                            if reward is None:
                                model.zero_grad()
                                for j in range(self.args.search_train_epoch):
                                    last_batch = self.fit(model,train_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
                                reward = torch.tensor(pre_evaluate(model,test_data)).cuda()
                                grad_dict[test_pred.grad.clone().detach()] = reward.clone().detach()
                                model = copy.deepcopy(cur_model)
                            if reward < baseline - self.args.skip_lim:
                                reward = min_reward.clone().detach()
                                reward += self.args.controller_entropy_weight * self.controller.sample_entropy
                            else:
                                if reward > max_reward:
                                    max_reward = reward.clone().detach()
                                    if train_with_optim:
                                        best_arc = copy.deepcopy(sample_arc)
                                reward += self.args.controller_entropy_weight * self.controller.sample_entropy
                                baseline -= (1 - self.args.controller_bl_dec) * (baseline - reward)
                            baseline = baseline.detach()

                            ctrl_loss = -1 * self.controller.sample_log_prob * (reward - baseline)
                            ctrl_loss /= self.controller.num_aggregate
                            if (i + 1) % self.controller.num_aggregate == 0:
                                ctrl_loss.backward()
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                                                           self.args.child_grad_bound)
                                self.controller_optimizer.step()
                                self.controller.zero_grad()
                            else:
                                ctrl_loss.backward(retain_graph=True)
                            break
                    self.controller.eval()

                    logging.info(
                        'Best auc during controller train: %.3f; Starting auc: %.3f' % (max_reward.item(), start_auc))
                    last_search_cnt = 0
                    if self.args.train_with_optim and best_arc is not None and max_reward > start_auc - self.args.rej_lim:
                        sample_arc = copy.deepcopy(best_arc)
                        for j in range(self.args.search_train_epoch):
                            last_batch = self.fit(model,train_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
                        new_auc = torch.tensor(pre_evaluate(model,test_data)).cuda()
                        print('Optimal: ',
                              self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
                    else:
                        grad_dict = dict()
                        self.controller.zero_grad()
                        while True:
                            with torch.no_grad():
                                self.controller(sampling=True)
                                last_search_cnt += 1
                            sample_arc = self.controller.sample_arc
                            if test_pred.grad is not None:
                                test_pred.grad.data.zero_()
                            test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
                            try:
                                test_loss.backward()
                            except RuntimeError:
                                pass
                            if test_pred.grad is None or torch.norm(test_pred.grad,float('inf')) < self.args.lower_bound_zero_gradient:
                                continue
                            dup_flag = False
                            for key in grad_dict.keys():
                                if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
                                    dup_flag = True
                                    break
                            if dup_flag:
                                continue
                            print(self.loss_formula.log_formula(sample_arc=sample_arc,
                                                                id=self.loss_formula.num_layers - 1))
                            grad_dict[test_pred.grad.clone().detach()] = True
                            model = copy.deepcopy(cur_model)
                            model.zero_grad()
                            for j in range(self.args.search_train_epoch):
                                last_batch = self.fit(model,train_data,loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
                            new_auc =torch.tensor(pre_evaluate(model,test_data)).cuda()
                            if new_auc > start_auc - self.args.rej_lim:
                                break
                            print('Epoch %d: Reject!' % (epoch + 1))

                    last_search_cnt = max(last_search_cnt // 10,self.controller.num_aggregate * self.args.controller_train_steps)
                    if last_search_cnt % self.controller.num_aggregate != 0:
                        last_search_cnt = (last_search_cnt // self.controller.num_aggregate + 1) * self.controller.num_aggregate
                    logging.info(self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
                    self.controller.train()




                else:
                    # for epoch_k in range(1, epochs + 1):
                    #     torch_fold_k_epoch_k_loss = model.train(train_data=train_data, epoch_k=epoch_k, presort=True)

                    evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)

        except KeyboardInterrupt:
            print("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()
                self.controller.save_model()
                self.loss_formula.save_model()


        # self.controller.load_model()
        # self.loss_formula.load_model()

def pre_evaluate(model,data):
    model_id = 'RankMSE'
    ranker = model
    fold_num = 5
    cutoffs = [1,3,5,7,9,10]
    epochs = 1

    l2r_cv_avg_scores = np.zeros(len(cutoffs)) # fold average

    for fold_k in range(1, fold_num + 1):
        #ranker.init()           # initialize or reset with the same random initialization
        #test_data = None

        # for epoch_k in range(1, epochs + 1):
        #     torch_fold_k_epoch_k_loss = ranker.train(train_data=data, epoch_k=epoch_k, presort=True)
        torch_fold_ndcg_ks = ranker.ndcg_at_ks(test_data=data, ks=cutoffs, device='cpu', presort=True)
        fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()
        l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks) # sum for later cv-performance

    # time_end = datetime.datetime.now()  # overall timing
    # elapsed_time_str = str(time_end - time_begin)
    # print('Elapsed time:\t', elapsed_time_str + "\n\n")

    l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
    eval_prefix = str(fold_num) + '-fold average scores:'
    print(model_id, eval_prefix, metric_results_to_string(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))  # print either cv or average performance

    mean_l2r_cv_avg_scores=np.mean(l2r_cv_avg_scores)
    print(mean_l2r_cv_avg_scores)
    return mean_l2r_cv_avg_scores
