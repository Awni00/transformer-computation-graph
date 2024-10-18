import os
from ..custom import Config, Pipeline, DataModule, TrainingManager
from simtransformer.module_base import DirectoryHandler
import argparse
import torch
import numpy as np
import random

def train_start(**kwargs):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    task_dir = os.path.join(parent_dir)

    dir_handler = DirectoryHandler(
        load_data_abs_dir=os.path.join(task_dir, 'data'),
        data_file_name='dag.json',
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