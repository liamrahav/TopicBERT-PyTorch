from argparse import ArgumentParser
import json
import os
import random
import sys

import torch
import numpy as np

import training.train_topicbert as ttb
from datasets import Vocabulary, Reuters8Dataset, BOWDataset
from datasets.utils import partition_dataset

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run experiments with TopicBERT.',
        epilog='Use -s or --save to PATH generate a reusable JSON config when calling this script.'
        ' It can be loaded with -l or --load PATH. \n'
    )

    parser.add_argument(
        '-d', '--dataset',
        default='reuters8',
        choices=['reuters8'],
        help='Which Dataset wrapper to use (DO NOT USE UNLESSS MORE DATASETS ARE ADDED).'
    )

    parser.add_argument(
        '--label-path', '--labels-path',
        help='Path to text file containing labels separated by newlines.'
    )

    parser.add_argument(
        '--train-dataset-path',
        help='Path to the file containing training examples. Must be in the format of the dataset'
        ' indicated by -d/--dataset.'
    )

    parser.add_argument(
        '--val-dataset-path',
        help='Path to the file containing validation examples. Must be in the format of the dataset'
        ' indicated by -d/--dataset. Not necessary if --val-freq == 0'
    )

    parser.add_argument(
        '--test-dataset-path',
        help='Path to the file containing test examples. Must be in the format of the dataset'
        ' indicated by -d/--dataset. Not necessary if --test-freq == 0'
    )

    parser.add_argument(
        '--partition-factor',
        type=int,
        default=1,
        help='How much further to split examples relative to 512. E.g. if partition-factor =='
        ' 4, examples are partitioned such that they are a maximum of 512 / 4 = 128 tokens long'
        ' while each partition retains the same label. (default: 1)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of workers to use for the dataloaders (how many background processes). (default: 8)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for the dataloader. (default: 8)'
    )

    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=10,
        help='Number of warmup steps for the learning rate scheduler. (default: 10)'
    )

    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate to be passed to the optimizer. (default: 2e-5)'
    )

    parser.add_argument(
        '--alpha',
        metavar='[0, 1]',
        type=float,
        default=0.9,
        help='Controls how much TopicBERT should weight its own loss or its topic model\'s loss'
        ' Must be between 0 and 1. (default: 0.9)'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout to apply to the concatenated hidden state of the model. (default: 0.1)'
    )

    parser.add_argument(
        '--num-epochs', '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train for. (default: 10)'
    )

    parser.add_argument(
        '--clip', '--grad-clip',
        type=float,
        default=1.,
        help='Norm-based gradient clipping threshold. (default: 1.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='If provide, fixes the global random seed to SEED for consistency.'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Which PyTorch device to run on. (default: "cpu")'
    )

    parser.add_argument(
        '--val-freq',
        type=int,
        default=0,
        help='How frequently in terms of epochs to gather the model\'s validation set metrics'
        ' (default: 0.0)'
    )

    parser.add_argument(
        '--test-freq',
        type=int,
        default=0,
        help='How frequently in terms of epochs to gather the model\'s test set metrics'
        ' (default: 0.0)'
    )

    parser.add_argument(
        '--resume', '--use-checkpoint', '--load-checkpoint',
        metavar='CHECKPOINT_DIR',
        default='',
        help='If provided, resumes training from the given checkpoint directory. Use'
        ' --save-checkpoint-only to just save (no load).'
    )

    parser.add_argument(
        '--save-checkpoint-only',
        action='store_true',
        help='If set, will NOT load a checkpoint, but WILL save it to the checkpoint directory'
        ' provided at --resume/--use-checkpoint/--load-checkpoint'
    )

    parser.add_argument(
        '--disable-tensorboard',
        action='store_true',
        help='If set, disables PyTorch\'s Tensorboard for training visualization'
    )

    parser.add_argument(
        '--tensorboard-dir',
        help='Directory to place the tensorboard logs in.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='If set, (slightly) more training details will be outputted to stdout'
    )

    parser.add_argument(
        '--silent',
        action='store_true',
        help='If set, stdout will be redirected to /dev/null *during the main training loop*'
    )

    parser.add_argument(
        '-l', '--load',
        metavar='CONFIG_JSON_PATH',
        help='Loads command line arguments to this script from a JSON file. Use -s or --save to'
        ' create this file with desired settings.'
    )

    parser.add_argument(
        '-s', '--save',
        metavar='CONFIG_JSON_PATH',
        help='Saves command line arguments given to this script as a JSON config file.'
    )

    # Get options, load/save JSON config file if flagged
    opts = vars(parser.parse_args())
    if opts['load']:
        with open(opts['load'], 'r') as f:
            for key, val in json.loads(f.read()).items():
                opts[key] = val
    elif opts['save']:
        with open(opts['save'], 'w') as f:
            f.write(json.dumps(opts))

    verbose = opts['verbose']

    # Set seeds
    if opts['seed']:
        random.seed(opts['seed'])
        np.random.seed(opts['seed'])
        torch.manual_seed(opts['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts['seed'])

    # Construct datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None

    vocab = Vocabulary([opts['train_dataset_path'],
                        opts['val_dataset_path'], opts['test_dataset_path']])
    if verbose:
        print(' [*] Vocabulary built.')

    if opts['dataset'] == 'reuters8':
        train_dataset = Reuters8Dataset(
            opts['train_dataset_path'], opts['label_path'], vocab)
        train_dataset = BOWDataset(train_dataset, train_dataset.vocab)
        if verbose:
            print(' [*] Train dataset built.')
        val_dataset = Reuters8Dataset(
            opts['val_dataset_path'], opts['label_path'], vocab)
        val_dataset = BOWDataset(val_dataset, val_dataset.vocab)
        if verbose:
            print(' [*] Validation dataset built.')
        test_dataset = Reuters8Dataset(
            opts['test_dataset_path'], opts['label_path'], vocab)
        test_dataset = BOWDataset(test_dataset, test_dataset.vocab)
        if verbose:
            print(' [*] Test dataset built.')

    pf = opts['partition_factor']
    if pf > 1:
        old_size = len(train_dataset)
        partition_dataset(train_dataset, partition_factor=pf)
        if verbose:
            print(' [*] Partitioned training examples by factor of {}. (ex.\'s: {} --> {})'.format(
                pf, old_size, len(train_dataset)))

    # Train
    tensorboard = not opts['disable_tensorboard']

    load_ckpt = bool(opts['resume'])
    if opts['save_checkpoint_only']:
        load_ckpt = False

    ttb.train(
        train_dataset,
        batch_size=opts['batch_size'],
        num_warmup_steps=opts['warmup_steps'],
        lr=opts['lr'],
        alpha=opts['alpha'],
        dropout=opts['dropout'],
        num_epochs=opts['num_epochs'],
        clip=opts['clip'],
        device=opts['device'],
        val_frequency=opts['val_freq'],
        val_dataset=val_dataset,
        test_frequency=opts['test_freq'],
        test_dataset=test_dataset,
        num_workers=opts['num_workers'],
        should_load_ckpt=load_ckpt,
        ckpt_dir=opts['resume'],
        tensorboard=tensorboard,
        tensorboard_dir=opts['tensorboard_dir'],
        verbose=verbose,
        silent=opts['silent']
    )
