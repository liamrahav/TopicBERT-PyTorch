'''This module contains utilities to help with training models.'''

import os
import warnings

import torch


def save_ckpt(state_dict, ckpt_dir, ckpt_tag=''):
    '''Saves a checkpoint at the given directory. Defaults to
    `ckpt_dir/checkpoint{ckpt_tag}.pt`

    Args:
        state_dict (dict): Any dictionary can be saved. Most likely has the model's
            state_dict included.
        ckpt_dir (str): The directory to save the checkpoint in.
        ckpt_tag (str): A tag to be added to the checkpoint name.
    '''
    if not os.path.isdir(ckpt_dir):
        try:
            os.makedirs(ckpt_dir)
        except:
            raise FileExistsError(
                'The directory {} does not exist'.format(ckpt_dir))

    with warnings.catch_warnings():
        # Ignore warning about needing to save optimizer if scheduler is being saved
        warnings.simplefilter("ignore")
        torch.save(state_dict, os.path.join(
            ckpt_dir, 'checkpoint{}.pt'.format(ckpt_tag)))


def load_ckpt(ckpt_dir, ckpt_tag=''):
    if not os.path.isdir(ckpt_dir):
        raise FileExistsError(
            'The directory {} does not exist'.format(ckpt_dir))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(os.path.join(ckpt_dir, 'checkpoint{}.pt'.format(ckpt_tag)))
