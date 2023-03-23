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

from autoloss_ltr import NeuralRanker, load_multiple_data, metric_results_to_string, evaluation


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

    def train(self, model, epochs,search_loss=False):
        for epoch_k in range(1, epochs + 1):
            if  search_loss:
                start=evaluation()
                start_ndcg10=start.min()
                baseline = torch.tensor(start_ndcg10).cuda()


            else:
            loss_formula.eval()
            controller.zero_grad()

