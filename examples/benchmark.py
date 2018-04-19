import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from reid import models
from reid.dist_metric import DistanceMetric
from reid.evaluators import Evaluator
from reid.utils.serialization import load_checkpoint

from softmax_loss import get_data

working_dir = os.path.dirname(os.path.abspath(__file__))

def plot(args, directory, values, testset, plot_range=100):
    plt.plot(list(range(plot_range)), values[0:plot_range], 'r.')
    plt.axis([0, plot_range, 0.0, 1.0])
    plt.title(args.batch_id)
    plt.xlabel('Rank')
    plt.ylabel('Cumulative accuracy')
    filename = '%s_%s_%s.png' % (testset, args.architecture, args.loss)
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight')
    plt.close()

def load_dataset(architecture, dataset='synthetic', batch_id=''):
    batch_size, workers, combine_trainval = 32, 4, False
    height, width = (144, 56) if architecture == 'inception' else (256, 128)
    dataset, num_classes, _, _, loader = \
        get_data(dataset, 0, os.path.join(working_dir, 'data'), \
                 height, width, batch_size, workers, combine_trainval, batch_id)
    return dataset, num_classes, loader

def setup_model(loss, architecture, num_classes):
    if loss == 'triplet':
        dropout = 0
        num_features = 1024
        num_classes = 128
    elif loss == 'softmax':
        dropout = 0.5
        num_features = 128
    return models.create(architecture, num_features=num_features,
                         dropout=dropout, num_classes=num_classes)

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True # Not sure about this one
    metric = DistanceMetric(algorithm='euclidean')
    benchmark_dir = os.path.join(working_dir, 'benchmarks', args.batch_id,
                                 args.loss, args.architecture)


    _, num_classes, _ = load_dataset(args.architecture, dataset='synthetic',
                                     batch_id=args.batch_id)
    for test_set in args.test_sets:
        dataset, _, loader = load_dataset(args.architecture, dataset=test_set)

        model = setup_model(args.loss, args.architecture, num_classes)
        model = nn.DataParallel(model).cuda()
        checkpoint = load_checkpoint(os.path.join(benchmark_dir, 'model_best.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        evaluator = Evaluator(model)

        cmc = evaluator.evaluate(loader, dataset.query, dataset.gallery, metric)
        plot(args, os.path.join(working_dir, 'benchmarks', args.batch_id), cmc, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate smaller dataset')
    parser.add_argument('batch_id', help='batch id')
    parser.add_argument('-a', '--architecture', type=str, default='inception')
    parser.add_argument('-l', '--loss', type=str, default='softmax')
    parser.add_argument('--test-sets', nargs='+', type=str, default=['cuhk03', 'dukemtmc', 'market1501'])
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    main(args)
