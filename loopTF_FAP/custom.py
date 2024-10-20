import sys, time, copy
sys.path.append("..")
from simtransformer.model_bank import GPT2Standard
from simtransformer.model_base import ReadOut
from simtransformer.module_base import ConfigBase, DataModuleBase, PipelineBase
import torch
import torch.nn as nn
from torch.utils.data import random_split
from simtransformer.manager import TrainingManagerBase
from simtransformer.utils import MRR_fn, EasyDict, token_accuracy
from simtransformer.model_base import LinearWithChannel
from typing import Optional
from torch.utils.data import DataLoader
import re

probe_pos_len = 4

class LoopGPTBlock(GPT2Standard):
    def __init__(self, config: EasyDict, vocab_size: int, weight_tie: bool = False):
        super().__init__(config, vocab_size, weight_tie)
        self.med_target_readout = ReadOut(config.hidden_size, vocab_size)
        self.med_parent_readout = ReadOut(config.hidden_size, vocab_size)

class Config(ConfigBase):
    pass

class TrainingManager(TrainingManagerBase):
    def __init__(self, dir_handler, abstract_config, abstract_pipeline, abstract_datamodule, **kwargs):
        super(TrainingManager, self).__init__(dir_handler, abstract_config, abstract_pipeline, abstract_datamodule, **kwargs)
        

    def get_training_name(self):
        add_med_loss_prob_str = '[' + ','.join(f'{prob:.1e}'.replace('e-0', 'e-').replace('e+0', 'e+') for prob in self.train_config.med_loss_ratio) + ']'
        if "dag_ADDonly" in self.dir_handler.data_file_name:
            Operation_type = '[ADD]'
        else:
            Operation_type = '[ADD,MUL]'
        if self.train_config.use_loss_n:
            ntp_scale = f'{self.train_config.loss_n_scale:.1f}'
        else:
            ntp_scale = '0.0'
        if self.train_config.use_parent_loss:
            parent_scale = f'{self.train_config.loss_parent_scale:.1f}'
        else:
            parent_scale = '0.0'
        training_name = (
            'L' + str(self.model_config.num_layers) +
            'H' + str(self.model_config.num_heads) +
            'D' + str(self.model_config.hidden_size) + 
            '-' +
            'N' + str(self.data_config.dag_config.num_nodes) +
            'DP' + str(self.train_config.max_dep) + 
            Operation_type + 
            '-' +
            'MR' + add_med_loss_prob_str +
            'NTP' + ntp_scale + 
            'PAR' + parent_scale +
            '-' +
            time.strftime("%m%d-%H%M")
        )  # default
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = LoopGPTBlock(self.model_config, len(self.vocab), weight_tie=True)
        loss_p_model = nn.CrossEntropyLoss()
        loss_n_model = nn.CrossEntropyLoss()
        return  {
            "train_config": self.train_config,
            "training_model": training_model,
            "loss_p_model": loss_p_model,
            "loss_n_model": loss_n_model,
        }
    
    def config_datamodule(self):
        if "batch_size" not in self.data_config.to_dict().keys():
            self.data_config.batch_size = self.train_config.batch_size
        return {
            "data_config": self.data_config, 
            "dir_handler": self.dir_handler,
        }
       

class DataModule(DataModuleBase):
    def __init__(self, data_config, dir_handler, only_dst=False):
        super(DataModule, self).__init__(data_config, dir_handler)
        self.only_dst = only_dst

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
        
        # initialize the depth and operation tensors with a large number
        dep_tensor = torch.full((len(batch), max_seq_len), torch.iinfo(torch.long).max, dtype=torch.long)
        oper_tensor = torch.full((len(batch), max_seq_len), torch.iinfo(torch.long).max, dtype=torch.long)
        msk_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        msk_pa_tensor = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)

        probe_msk_tensor = None
        probe_label = None
        
        for i, sample in enumerate(batch):
            sentence = sentences[i]
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded))
            
            # find all positions of '=' and add 1 to get the positions of the targets
            var_pos = [idx + 1 for idx, word in enumerate(sentence_idx) if word == self.vocab['=']]

            # Build a lookup table for variable names and their values
            var_lookup = {}
            for pos, value, depth, oper in zip(var_pos, sample['values'], sample['depths'], sample['opers']):
                var_name = sentence[pos]  # Get the variable name from the sentence
                var_lookup[var_name] = {
                    "value": value,
                    "depth": depth,
                    "oper": oper
                }

            # Update tensors based on the lookup table
            for idx, word in enumerate(sentence):
                if re.match(r'^[a-zA-Z]_\d+$', word) and word in var_lookup:
                    y_tensor[i, idx] = var_lookup[word]["value"]
                    dep_tensor[i, idx] = var_lookup[word]["depth"]
                    oper_tensor[i, idx] = var_lookup[word]["oper"]
                    msk_pa_tensor[i, idx] = True

            msk_tensor[i, var_pos] = True 
        
        # exclude msk_tensor == 1's position from msk_pa_tensor
        msk_pa_tensor = msk_pa_tensor & ~msk_tensor
            
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": {
                "dep": dep_tensor, 
                "oper": oper_tensor, 
                "msk_pa": msk_pa_tensor
            }
        })

class Pipeline(PipelineBase):

    def __init__(self, train_config, training_model, loss_p_model, loss_n_model):
        """
        add only_dst parameter to Pipeline class
        """
        super(Pipeline, self).__init__(train_config, training_model, loss_p_model, loss_n_model)
        self.max_oper = self.train_config.max_oper
        self.max_dep = self.train_config.max_dep

        self.med_loss_counter = [0 for _ in range(self.max_dep)]

        self.loss_parent_scale = self.train_config.loss_parent_scale if self.train_config.use_parent_loss else 0.0
        

    def intermediate_loss(self, hidden_state, y, indices, loss_type: str, step_type: str, batch_size: int, cur_dep: int):
        """
        Select the intermediate target for each depth. 
        
        Args:
        - hidden_state: tensor of shape (batch_size, seq_len, hidden_size)
        - y: tensor of shape (batch_size, seq_len)
        - indices: tensor of shape (batch_size, seq_len)
        - loss_type: str, 'target' or 'parent'
        - step_type: str, 'train' or 'val' or 'test'
        - batch_size: int
        - cur_dep: int, current depth
        
        Returns:
        - loss: tensor of shape (1, )
        - mrr: tensor of shape (1, )
        - acc: tensor of shape (1, )
        - num_target: int
        """
        # find all the indices in dep that are equal to cur_dep and oper <= max_oper, and msk_target == True
        selected_state = self._mask_select(hidden_state, indices)
        selected_label = self._mask_select(y, indices)
        if selected_state.numel() == 0:
            return None, None, None, None

        if loss_type == 'target':
            selected_output = self.training_model.med_target_readout(selected_state)
        elif loss_type == 'parent':
            selected_output = self.training_model.med_parent_readout(selected_state)
        elif loss_type == 'NTPvar' or loss_type == 'NTPoper':
            selected_output = self.training_model.readout(selected_state)
        
        # compute mrr 
        mrr = MRR_fn(selected_output, selected_label)
        # compute loss
        loss = self.loss_p_model(selected_output, selected_label)
        # compute accuracy
        acc = token_accuracy(selected_output, selected_label)
        
        self.log(f"{step_type}_{loss_type}_loss_dep_{cur_dep}", loss, prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f"{step_type}_{loss_type}_mrr_dep_{cur_dep}", mrr, prog_bar=True, logger=True, batch_size=batch_size) if self.train_config.log_mrr else None
        self.log(f"{step_type}_{loss_type}_acc_dep_{cur_dep}", acc, prog_bar=True, logger=True, batch_size=batch_size) if self.train_config.log_acc else None
        
        return loss, mrr, acc, len(selected_state)

    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, batch_info = self._unpack_batch(batch)
        x, y, msk_target = train_batch
        dep, oper, msk_pa = batch_info['dep'], batch_info['oper'], batch_info['msk_pa']
        batch_size = self.len_batch(batch)

        
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # msk_target (batch_size, seq_len)
        
        loss_ls = []
        mrr_ls = []
        acc_ls = []
        loss_parent_ls = []
        mrr_parent_ls = []
        acc_parent_ls = []
        med_loss_ratio = []
        
        token_embedding = self.training_model.readin(x)
        for cur_dep in range(self.max_dep):
            
            if cur_dep == 0:
                hidden_state = self.training_model.encoder(token_embedding)
            else: 
                hidden_input = hidden_state + token_embedding if self.train_config.add_input_per_loop else hidden_state
                hidden_state = self.training_model.encoder(hidden_input)
                # shape (batch_size, seq_len, hidden_size)
                # Here, we also add the token embedding to the hidden_state to make the model aware of the input, see https://arxiv.org/pdf/2409.15647 
            
            
            # compute intermediate target loss
            indices = torch.logical_and(torch.logical_and(dep == cur_dep, msk_target == True), oper <= self.max_oper)
            med_target_loss, med_target_mrr, med_target_acc, num_target = self.intermediate_loss(hidden_state, y, indices, 'target', step_type, batch_size, cur_dep)
            loss_ls.append(med_target_loss)
            mrr_ls.append(med_target_mrr)
            acc_ls.append(med_target_acc)

            # compute intermediate parent loss always
            indices = torch.where(torch.logical_and(torch.logical_and(dep == cur_dep - 1, msk_pa == True), oper <= self.max_oper))
            hidden_state_for_parent = hidden_state if self.train_config.use_parent_loss else hidden_state.detach()
            med_parent_loss, med_parent_mrr, med_parent_acc, num_parent = self.intermediate_loss(hidden_state_for_parent, y, indices, 'parent', step_type, batch_size, cur_dep)
            loss_parent_ls.append(med_parent_loss)
            mrr_parent_ls.append(med_parent_mrr)
            acc_parent_ls.append(med_parent_acc)

            med_loss_ratio.append(self.train_config.med_loss_ratio[cur_dep] if len(self.train_config.med_loss_ratio) > cur_dep else 1.0)
            
            # # update loss counter
            # self.med_loss_counter[cur_dep] += num_target if med_target_loss is not None else 0

        # final output
        # also do next token prediction loss 
        var_msk_to_predict = msk_target | msk_pa
        var_msk_to_predict = var_msk_to_predict[:, 1:]  # exclude the first token to make it match the y_n
        # prepare the output and label for next token prediction loss
        hidden_state_n = hidden_state[:, :-1, :] if self.train_config.use_loss_n else hidden_state[:, :-1, :].detach()
        y_n = x[:, 1:]
        
        cur_dep = self.max_dep - 1
        loss_n_var, _, _, num_var = self.intermediate_loss(hidden_state_n, y_n, var_msk_to_predict, 'NTPvar', step_type, batch_size, cur_dep)
        loss_n_oper, _, _, num_oper = self.intermediate_loss(hidden_state_n, y_n, ~var_msk_to_predict, 'NTPoper', step_type, batch_size, cur_dep)
        
        # note that the number of operations is roughly the same as the number of variables in the sentence, so we don't do any scaling here
        loss_n = loss_n_var * num_var + loss_n_oper * num_oper / (num_var + num_oper)

        loss_p = 0.0
        loss_parent = 0.0
        mrr = 0.0
        mrr_parent = 0.0
        acc = 0.0
        acc_parent = 0.0
        
        add_med_loss_prob_total = 0.0
        
        # calculate loss_p as the inner product of loss and med_loss_ratio by ignoring None
        loss_target = sum([loss_i * med_loss_ratio[i] for i, loss_i in enumerate(loss_ls) if loss_i is not None]) / sum([med_loss_ratio[i] for i, loss_i in enumerate(loss_ls) if loss_i is not None])
        
        mrr_target = sum([mrr_ls[i] for i, mrr_i in enumerate(mrr_ls) if mrr_i is not None]) / sum([1 for i, mrr_i in enumerate(mrr_ls) if mrr_i is not None])
        
        acc_target = sum([acc_ls[i] for i, acc_i in enumerate(acc_ls) if acc_i is not None]) / sum([1 for i, acc_i in enumerate(acc_ls) if acc_i is not None])
        
        loss_parent = sum([loss_parent_ls[i] * med_loss_ratio[i] for i, loss_parent_i in enumerate(loss_parent_ls) if loss_parent_i is not None]) / sum([med_loss_ratio[i] for i, loss_parent_i in enumerate(loss_parent_ls) if loss_parent_i is not None])
        
        mrr_parent = sum([mrr_parent_ls[i] for i, mrr_parent_i in enumerate(mrr_parent_ls) if mrr_parent_i is not None]) / sum([1 for i, mrr_parent_i in enumerate(mrr_parent_ls) if mrr_parent_i is not None])
        
        acc_parent = sum([acc_parent_ls[i] for i, acc_parent_i in enumerate(acc_parent_ls) if acc_parent_i is not None]) / sum([1 for i, acc_parent_i in enumerate(acc_parent_ls) if acc_parent_i is not None])

        self.log(f"{step_type}_target_loss", loss_target, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log(f"{step_type}_target_mrr", mrr_target, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
        self.log(f"{step_type}_target_acc", acc_target, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None
        
        self.log(f"{step_type}_parent_loss", loss_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log(f"{step_type}_parent_mrr", mrr_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
        self.log(f"{step_type}_parent_acc", acc_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None
        
        
        loss_p = (loss_target + loss_parent * self.loss_parent_scale) / (1.0 + self.loss_parent_scale)
        loss = (loss_p + loss_n * self.loss_n_scale) / (1.0 + self.loss_n_scale)
        self.log(f"{step_type}_loss", loss, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        
        return loss_p, loss_n, 0.0

        
 