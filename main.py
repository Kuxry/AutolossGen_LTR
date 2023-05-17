# coding=utf-8

import argparse
import torch
import os
import sys
import numpy as np


import BaseRunner
#只是调用了这个文件，还需调用文件下的这个构造函数
from autoloss_ltr import DataProcessor
from controller import Controller
from loss_formula import LossFormula



def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')

    parser.add_argument('--log_file', type=str, default='../log/log_0.txt', help='Logging file path')
    parser.add_argument('--result_file', type=str, default='../result/result.npy', help='Result file path')
    parser.add_argument('--random_seed', type=int, default=40, help='Random seed of numpy and pytorch')
    parser.add_argument('--model_name', type=str, default='BiasedMF', help='Choose model to run.')

    parser.add_argument('--child_num_layers', type=int, default=12)
    parser.add_argument('--child_num_branches', type=int, default=8)  # different layers
    parser.add_argument('--child_out_filters', type=int, default=36)
    parser.add_argument('--sample_branch_id', action='store_true')
    parser.add_argument('--sample_skip_id', action='store_true')
    parser.add_argument('--search_loss', action='store_true', help="To search a loss or verify a loss")
    parser.add_argument('--train_with_optim', action='store_true')
    parser.add_argument('--child_grad_bound', type=float, default=5.0)
    parser.add_argument('--smooth_coef', type=float, default=1e-6)
    parser.add_argument('--layers', type=str, default='[64, 16]',
                        help="Size of each layer. (For Deep RS Model.)")
    parser.add_argument('--loss_func', type=str, default='BCE',
                        help='Loss Function. Choose from ["BCE", "MSE", "Hinge", "Focal", "MaxR", "SumR", "LogMin"]')

    parser = BaseRunner.baserunner.parse_runner_args(parser)
    parser = Controller.parse_Ctrl_args(parser)
    parser = LossFormula.parse_Formula_args(parser)
    args, extras = parser.parse_known_args()

    # random seed & gpu
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # testing
    data_id = 'MQ2008_Super'
    dir_data = 'D:/Data/MQ2008/'
    model_id = 'RankMSE'  # RankMSE, RankNet, LambdaRank
    # print(model_id)
    # evaluation(data_id=data_id, dir_data=dir_data, model_id=model_id, batch_size=100)
    data_processor = DataProcessor(data_id=data_id, dir_data=dir_data)

    controller = Controller()
    controller = controller.cuda()

    loss_formula = LossFormula()
    loss_formula = loss_formula.cuda()

    runner = BaseRunner.baserunner(loss_formula=loss_formula, controller=controller, args=args)
    runner.train(search_loss=True, data_id=data_id, dir_data=dir_data, model_id=model_id)

if __name__ == '__main__':
	main()
