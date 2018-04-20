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

def plot(args, directory, batches, testset, plot_range=100):
    plots = []
    for batch, cmc in batches.items():
        plots.append(plt.plot(list(range(plot_range)), cmc[0:plot_range], '.', label=batch))
    plt.legend()
    plt.axis([0, plot_range, 0.0, 1.0])
    plt.title("%s - %s, %s" % (testset, args.architecture, args.loss))
    plt.xlabel('Rank')
    plt.ylabel('Cumulative accuracy')
    filename = '%s_%s_%s' % (testset, args.architecture, args.loss)
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight', format='pdf')
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


    for test_set in args.test_sets:
        dataset, _, loader = load_dataset(args.architecture, dataset=test_set)
        cmcs = {}
        for batch_id in args.batch_ids:
            benchmark_dir = os.path.join(working_dir, 'benchmarks', batch_id,
                                         args.loss, args.architecture)
            _, num_classes, _ = load_dataset(args.architecture, dataset='synthetic',
                                             batch_id=batch_id)

            model = setup_model(args.loss, args.architecture, num_classes)
            model = nn.DataParallel(model).cuda()
            checkpoint = load_checkpoint(os.path.join(benchmark_dir, 'model_best.pth.tar'))
            model.module.load_state_dict(checkpoint['state_dict'])
            evaluator = Evaluator(model)

            cmcs[batch_id] = evaluator.test(loader, dataset.query, dataset.gallery, metric)

        plot(args, os.path.join(working_dir, 'plots'), cmcs, test_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate results and plot!')
    parser.add_argument('--batch-ids', action='append', help='Batch ids')
    parser.add_argument('-a', '--architecture', type=str, default='inception')
    parser.add_argument('-l', '--loss', type=str, default='softmax')
    parser.add_argument('--test-sets', nargs='+', type=str, default=['cuhk03', 'dukemtmc', 'market1501'])
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    main(args)
