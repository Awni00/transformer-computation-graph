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
            'NTP' + f'{self.train_config.loss_n_scale:.1f}' + 
            '-' +
            time.strftime("%m%d-%H%M")
        )  # default
        print(f"Current training run: {training_name}")
        return training_name
    
    def config_pipeline(self):
        training_model = LoopGPTBlock(self.model_config, len(self.vocab), weight_tie=True)
        loss_p_model = nn.CrossEntropyLoss()
        loss_n_model = nn.CrossEntropyLoss() if self.train_config.use_loss_n else None
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

    def compute_intermediate_target_loss(self, hidden_state, y, indices):
        """
        Select the intermediate target for each depth. 
        """
        # find all the indices in dep that are equal to cur_dep and oper <= max_oper, and mask == True
        selected_state = self._mask_select(hidden_state, indices)
        selected_label = self._mask_select(y, indices)
        if selected_state.numel() == 0:
            return None, None, None

        selected_output = self.training_model.med_target_readout(selected_state)
        # compute mrr 
        mrr = MRR_fn(selected_output, selected_label)
        # compute loss
        loss = self.loss_p_model(selected_output, selected_label)
        # compute accuracy
        acc = token_accuracy(selected_output, selected_label)
        return loss, mrr, acc, len(selected_state)

    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, batch_info = self._unpack_batch(batch)
        x, y, mask = train_batch
        dep, oper, msk_pa = batch_info['dep'], batch_info['oper'], batch_info['msk_pa']
        
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # mask (batch_size, seq_len)

        token_embedding = self.training_model.readin(x)
        hidden_state = token_embedding
        loss_ls = []
        mrr_ls = []
        acc_ls = []
        loss_parent_ls = []
        mrr_parent_ls = []
        acc_parent_ls = []
        med_loss_ratio = []
        
        for cur_dep in range(self.max_dep):
            
            if cur_dep == 0:
                hidden_state = self.training_model.encoder(token_embedding)
            else: 
                hidden_input = hidden_state + token_embedding if self.train_config.add_input_per_loop else hidden_state
                hidden_state = self.training_model.encoder(hidden_input)
                # shape (batch_size, seq_len, hidden_size)
                # Here, we also add the token embedding to the hidden_state to make the model aware of the input, see https://arxiv.org/pdf/2409.15647 
            
            # compute intermediate target loss
            indices = torch.where(torch.logical_and(torch.logical_and(dep == cur_dep, oper <= self.max_oper), mask == True))
            med_target_loss, med_target_mrr, med_target_acc, num_target = self.compute_intermediate_target_loss(hidden_state, y, indices)
            loss_ls.append(med_target_loss)
            mrr_ls.append(med_target_mrr)
            acc_ls.append(med_target_acc)

            # compute intermediate parent loss
            if self.train_config.use_parent_loss:
                indices = torch.where(torch.logical_and(torch.logical_and(dep < cur_dep, msk_pa == True), oper <= self.max_oper))
                med_parent_loss, med_parent_mrr, med_parent_acc, num_parent = self.compute_intermediate_parent_loss(cur_dep, hidden_state, y, msk_pa, dep)
                loss_parent_ls.append(med_parent_loss)
                mrr_parent_ls.append(med_parent_mrr)
                acc_parent_ls.append(med_parent_acc)

            med_loss_ratio.append(self.train_config.med_loss_ratio[cur_dep] if len(self.train_config.med_loss_ratio) > cur_dep else 1.0)
            
            # # update loss counter
            # self.med_loss_counter[cur_dep] += num_target if med_target_loss is not None else 0

        # final output
        output = self.training_model.readout(hidden_state)

        # also do next token prediction loss 
        if self.loss_n_model is not None:
            output_n = output[:, :-1, :]
            y_n = y[:, 1:]
            loss_n = self.loss_n_model(output_n.reshape(-1, output_n.size(-1)), y_n.reshape(-1))
        else: 
            loss_n = 0.0

        loss_p = 0.0
        loss_parent = 0.0
        mrr = 0.0
        mrr_parent = 0.0
        acc = 0.0
        acc_parent = 0.0
        num_effective_dep = 0
        add_med_loss_prob_total = 0.0
        for i, loss_i in enumerate(loss_ls):
            if loss_i is not None:
                
                loss_p += loss_i * med_loss_ratio[i]
                mrr += mrr_ls[i]
                acc += acc_ls[i]
                
                if self.train_config.use_parent_loss:
                    loss_parent += loss_parent_ls[i] * med_loss_ratio[i]
                    mrr_parent += mrr_parent_ls[i]
                    acc_parent += acc_parent_ls[i]
                
                num_effective_dep += 1
                add_med_loss_prob_total += med_loss_ratio[i]
                # The total number of parameters with small depth outnumbers the total number of parameters with large depth, so we don't want the mrr of small depth to dominate the mrr of large depth. We want to focus more on large depth. So is for the loss.

                self.log(f"{step_type}_loss_dep_{i}", loss_i, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
                self.log(f"{step_type}_mrr_dep_{i}", mrr_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
                self.log(f"{step_type}_acc_dep_{i}", acc_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None

                if self.train_config.use_parent_loss:
                    self.log(f"{step_type}_loss_pa_dep_{i}", loss_parent_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch))
                    self.log(f"{step_type}_mrr_pa_dep_{i}", mrr_parent_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
                    self.log(f"{step_type}_acc_pa_dep_{i}", acc_parent_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None
        
        if num_effective_dep > 0:
            loss_p /= add_med_loss_prob_total
            mrr /= num_effective_dep
            acc /= num_effective_dep
            if self.train_config.use_parent_loss:
                loss_parent /= add_med_loss_prob_total
                mrr_parent /= num_effective_dep
                acc_parent /= num_effective_dep
        else:
            return None, None, None

        self.log(f"{step_type}_loss", loss_p, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log(f"{step_type}_mrr", mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
        self.log(f"{step_type}_acc", acc, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None
        
        if self.train_config.use_parent_loss:
            self.log(f"{step_type}_loss_pa", loss_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(f"{step_type}_mrr_pa", mrr_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_mrr else None
            self.log(f"{step_type}_acc_pa", acc_parent, prog_bar=True, logger=True, batch_size=self.len_batch(batch)) if self.train_config.log_acc else None
            
        if self.loss_n_model is not None:
            self.log(f"{step_type}_loss_n", loss_n, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        return loss_p + loss_parent * self.train_config.parent_loss_scale, loss_n, output

        
 