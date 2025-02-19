import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evalute.evaluator import Eval_thread
from evalute.dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor
def main(cfg):
    version = cfg.pred_dir.split('/')[-1]
    file = open(cfg.save_dir+'result.txt', 'a+')
    file.write(version+': \n')
    file.close()

    # root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = ''
    # gt_dir = osp.join(root_dir, 'gt')
    # pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(cfg.pred_dir)
    else:
        method_names = cfg.methods.split(' ')

    
    threads = []
    for method in method_names:
        if cfg.datasets is None:
            dataset_names = os.listdir(osp.join(cfg.pred_dir, method))
        else:
            dataset_names = cfg.datasets.split(' ')
        for dataset in dataset_names:
            loader = EvalDataset(osp.join(cfg.pred_dir, method, dataset), osp.join(cfg.gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)


    for thread in threads:
        print(thread.run())

def my_evalutor(methods = None, datasets = None, save_dir = '', gt_dir = None, pred_dir = None, cuda = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=methods)
    parser.add_argument('--datasets', type=str, default=datasets)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    parser.add_argument('--gt_dir', type=str, default=gt_dir)
    parser.add_argument('--pred_dir', type=str, default=pred_dir)
    parser.add_argument('--cuda', type=bool, default=cuda)
    config = parser.parse_args()
    main(config)
