import sys, time, copy
sys.path.append("..")
# from simtransformer.model_bank import GPT2Standard
from simtransformer.module_base import ConfigBase, DataModuleBase, PipelineBase
import torch
import torch.nn as nn
from torch.utils.data import random_split
from simtransformer.manager import TrainingManagerBase
from simtransformer.model_bank import LoopGPTBlock
from simtransformer.utils import MRR_fn, EasyDict
from simtransformer.model_base import LinearWithChannel
from typing import Optional
from torch.utils.data import DataLoader

probe_pos_len = 4

class Config(ConfigBase):
    pass

class TrainingManager(TrainingManagerBase):
    def __init__(self, dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule, only_dst=False):
        self.only_dst = only_dst
        super(TrainingManager, self).__init__(dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule)
        

    def get_training_name(self):
        training_name = f'L{self.model_config.num_layers}H{self.model_config.num_heads}N{self.data_config.dag_config.num_nodes}T' + time.strftime("%m%d-%H%M%S") # default
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = LoopGPTBlock(self.model_config, input_size=1, output_size=len(self.vocab), weight_tie=True)
        loss_p_model = nn.CrossEntropyLoss()
        loss_n_model = None
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model,
            "only_dst": self.only_dst
        }
    
    def config_datamodule(self):
        if "batch_size" not in self.data_config.to_dict().keys():
            self.data_config.batch_size = self.train_config.batch_size
        return {
            "data_config": self.data_config, 
            "dir_handler": self.dir_handler,
        }

class Pipeline(PipelineBase):

    def __init__(self, train_config, training_model, loss_p_model, loss_n_model):
        """
        add only_dst parameter to Pipeline class
        """
        super(Pipeline, self).__init__(train_config, training_model, loss_p_model, loss_n_model)


    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, batch_info = self._unpack_batch(batch)
        x, y, mask = train_batch
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # mask (batch_size, seq_len)

        output = self.training_model(x)

        # compute the loss for the masked position
        # y_msk_p = self._mask_select(y, mask)

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            loss_p_dst = self.loss_p_model(output_msk_p_dst, y_msk_p_dst)
            loss_p_rel = 0.0
        else:
            # (assuming each seq_len has exactly two True values)
            mask_rel, mask_dst = self._split_mask(mask)

            # Select the targets and outputs corresponding to the relation (rel) and destination (dst)
            y_msk_p_rel = self._mask_select(y, mask_rel)  # Select for relation
            output_msk_p_rel = self._mask_select(output, mask_rel)  # Model's output for relation

            y_msk_p_dst = self._mask_select(y, mask_dst)  # Select for destination
            output_msk_p_dst = self._mask_select(output, mask_dst) 

            loss_p_rel = self.loss_p_model(output_msk_p_rel, y_msk_p_rel)
            loss_p_dst = self.loss_p_model(output_msk_p_dst, y_msk_p_dst)

        loss_p = loss_p_rel + loss_p_dst

        # compute the loss for the non-masked position
        y_msk_n = self._mask_select(y, ~mask)
        output_msk_n = self._mask_select(output, ~mask)

        if len(y_msk_n) > 0 and self.loss_n_model is not None:
            loss_n = self.loss_n_model(output_msk_n, y_msk_n)
        else:
            loss_n = 0.0

        ## --------- log training loss --------- ##
        if step_type is not None:
            if self.loss_n_model is not None:
                self.log(step_type + "_loss_n", loss_n, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
                self.log(step_type + "_loss_p", loss_p, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(step_type + "_loss", loss_p + loss_n, prog_bar=True, on_epoch=True, logger=True, batch_size=self.len_batch(batch)) # should always log the total loss as 'val_loss' is used for ckpt saving

            if not self.only_dst:
                self.log(step_type + "_loss_rel", loss_p_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(step_type + "_loss_dst", loss_p_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            # don't change the log name step_type + '_loss' as it is used for ckpt saving

        return loss_p, loss_n, output

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs['loss']
        loss_p = training_step_outputs['loss_p']
        loss_n = training_step_outputs['loss_n']
        output = training_step_outputs['output']
        batch = training_step_outputs['batch']
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)
        
        else:
            mask_rel, mask_dst = self._split_mask(mask)

            y_msk_p_rel = self._mask_select(y, mask_rel)
            output_msk_p_rel = self._mask_select(output, mask_rel)

            y_msk_p_dst = self._mask_select(y, mask_dst)
            output_msk_p_dst = self._mask_select(output, mask_dst)

            # y_msk_p = self._mask_select(y, mask)
            # output_msk_p = self._mask_select(output, mask)
            
            # mrr = MRR_fn(output_msk_p, y_msk_p)
            mrr_rel = MRR_fn(output_msk_p_rel, y_msk_p_rel)
            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)

        # log mrr
        # self.log('train_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        if not self.only_dst:
            self.log('train_mrr_rel', mrr_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log('train_mrr_dst', mrr_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        
    def validation_step_end(self, validation_step_outputs):
        loss = validation_step_outputs['loss']
        loss_p = validation_step_outputs['loss_p']
        loss_n = validation_step_outputs['loss_n']
        output = validation_step_outputs['output']
        batch = validation_step_outputs['batch']
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch

        if self.only_dst:
            y_msk_p_dst = self._mask_select(y, mask)
            output_msk_p_dst = self._mask_select(output, mask)

            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)
        
        else:
            mask_rel, mask_dst = self._split_mask(mask)

            y_msk_p_rel = self._mask_select(y, mask_rel)
            output_msk_p_rel = self._mask_select(output, mask_rel)

            y_msk_p_dst = self._mask_select(y, mask_dst)
            output_msk_p_dst = self._mask_select(output, mask_dst)
            
            # y_msk_p = self._mask_select(y, mask)
            # output_msk_p = self._mask_select(output, mask)
            
            # mrr = MRR_fn(output_msk_p, y_msk_p)
            mrr_rel = MRR_fn(output_msk_p_rel, y_msk_p_rel)
            mrr_dst = MRR_fn(output_msk_p_dst, y_msk_p_dst)

        # log mrr
        # self.log('val_mrr', mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        if not self.only_dst:
            self.log('val_mrr_rel', mrr_rel, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log('val_mrr_dst', mrr_dst, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        
        

class DataModule(DataModuleBase):
    def __init__(self, data_config, dir_handler, only_dst=False):
        super(DataModule, self).__init__(data_config, dir_handler)
        self.only_dst = only_dst
        self.tokenizer

    def train_val_test_split(self, data):
        # split data into train, validation, and test sets by ratio 90:5:5
        data_train, data_test = random_split(data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])
        data_train, data_val = random_split(data_train, [int(0.9*len(data_train)), len(data_train)-int(0.9*len(data_train))])
        return data_train, data_val, data_test
    
    def transform_batch(self, batch, dataloader_idx):
        """
        Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
        """
        sentences = [' , '.join(sample['eqs']).split(' ') + ['<eos>'] for sample in batch]
        max_seq_len = max([len(s) for s in sentences])
        
        x_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.long).fill_(self.vocab["<pad>"])
        y_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
        dep_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
        batch_info = []
        msk_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        probe_msk_tensor = None
        probe_label = None
        
        for i, sample in enumerate(batch):
            sentence = sentences[i]
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded))
            
            # find the position of '=' and add 1 to get the position of the target
            para_pos = sentence_idx.index(self.vocab['=']) + 1
            # the depth of the target is given by batch[i]['depths'], which is a list of the same size as para_pos
            dep_tensor[i, para_pos] = torch.tensor(sample['depths'], dtype=torch.long)
            
            y_tensor[i, para_pos] = torch.tensor(sample['values'], dtype=torch.long)
            
            msk_tensor[i, para_pos] = True
            
            # batch_info.append({
            #     "pos": para_pos,
            #     "depth": depth
            # })
            
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": {
                "dep": dep_tensor
            }
        })
    