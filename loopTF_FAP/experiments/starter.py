import os
from ..custom import Config, Pipeline, DataModule, TrainingManager
from simtransformer.module_base import DirectoryHandler
import argparse
import torch
import numpy as np
import random
import re

def train_start(**kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    task_dir = os.path.join(parent_dir)

    dir_handler = DirectoryHandler(
        load_data_abs_dir=os.path.join(task_dir, 'data'),
        data_file_name=kwargs['data_file_name'],
        vocab_file_name='dag_vocab.yaml',
        load_config_abs_dir=os.path.join(task_dir, 'configurations'),
        load_ckpt_abs_path=None,
        output_abs_dir=None,
        create_run_under_abs_dir=task_dir, # will create new folder 'run'
        training_name=None,
    )

    training_manager = TrainingManager(
            dir_handler=dir_handler,
            abstract_config=Config, 
            abstract_pipeline=Pipeline,
            abstract_datamodule=DataModule,
            **kwargs,
        )
    
    training_manager.fit()
    
    
def train_continue(**kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    task_dir = os.path.join(parent_dir)
    
    last_run_dir = os.path.join(task_dir, 'run', kwargs['last_run_name'])
    load_ckpt_abs_path = os.path.join(last_run_dir, kwargs['ckpt_file_name'])
    # check if the 'last_run_name' start with '_v', if not set the training name to be '_v1'+last_run_name. If so, get the number after '_v' and increment by 1
    if kwargs['last_run_name'].startswith('v'):
        match = re.search(r'\d+', kwargs['last_run_name'][1:])
        if match:
            version_number = int(match.group()) + 1
            training_name = 'v' + str(version_number) + kwargs['last_run_name'][match.end()+1:]
        else:
            training_name = 'v1_' + kwargs['last_run_name'][2:]
    else:
        training_name = 'v1_' + kwargs['last_run_name']
    
    dir_handler = DirectoryHandler(
        load_data_abs_dir=os.path.join(task_dir, 'data'),
        data_file_name=kwargs['data_file_name'],
        vocab_file_name='dag_vocab.yaml',
        load_config_abs_dir=os.path.join(last_run_dir, 'configurations'),
        load_ckpt_abs_path=load_ckpt_abs_path,
        output_abs_dir=None,
        create_run_under_abs_dir=task_dir, # will create new folder 'run'
        training_name=training_name,
    )