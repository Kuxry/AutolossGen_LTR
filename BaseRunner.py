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
from autoloss_ltr import NeuralRanker, load_multiple_data, metric_results_to_string, evaluation, DataProcessor


class baserunner(NeuralRanker):
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
        parser.add_argument('--search_train_epoch', type=int, default=1,
                            help='epoch num for training when searching loss')
        parser.add_argument('--step_train_epoch', type=int, default=1, help='epoch num for training each step')

        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='AUC,RMSE', check_epoch=10, early_stop=1, controller=None,
                 loss_formula=None,
                 controller_optimizer=None, args=None):
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

    def train(self, search_loss=False,data_id=None, dir_data=None, model_id=None):
        model=NeuralRanker()
        train, test,vali = DataProcessor(data_id=data_id, dir_data=dir_data)
        epochs=1
        min_reward = torch.tensor(-1.0).cuda()
        try:
            for epoch_k in range(1, epochs + 1):
                # self.loss_formula.eval()
                # self.controller.zero_grad()

                if search_loss:
                    start=evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)
                    baseline = torch.tensor(start).cuda()
                    cur_model= copy.deepcopy(model)
                    grad_dict = dict()
                    test_pred = torch.rand(20).cuda() * 0.8 + 0.1  # change range here
                    test_label = torch.rand(20).cuda()
                    test_pred.requires_grad = True
                    max_reward = min_reward.clone().detach()
                    best_arc = None
                    for i in range(100):
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
                            if test_pred.grad is None or torch.norm(test_pred.grad,
                                                                    float('inf')) < self.args.lower_bound_zero_gradient:
                                reward = min_reward.clone().detach()
                            if reward is None:
                                for key, value in grad_dict.items():
                                    if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
                                        reward = value.clone().detach()
                                        break
                            # if reward is None:
                            #     model.zero_grad()
                            #     for j in range(self.args.search_train_epoch):
                            #         last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch,
                            #                               loss_fun=self.loss_formula, sample_arc=sample_arc,
                            #                               regularizer=False)
                            #     reward = torch.tensor(self.evaluate(model, validation_data, data_processor)[0]).cuda()
                            #     grad_dict[test_pred.grad.clone().detach()] = reward.clone().detach()
                            #     model = copy.deepcopy(cur_model)
                            if reward < baseline - self.args.skip_lim:
                                reward = min_reward.clone().detach()
                                reward += self.args.controller_entropy_weight * self.controller.sample_entropy
                            else:
                                if reward > max_reward:
                                    max_reward = reward.clone().detach()
                                    if self.args.train_with_optim:
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

