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
    def __init__(self, dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule):
        super(TrainingManager, self).__init__(dir_handler, use_wandb, abstract_config, abstract_pipeline, abstract_datamodule)
        

    def get_training_name(self):
        training_name = f'L{self.model_config.num_layers}H{self.model_config.num_heads}N{self.data_config.dag_config.num_nodes}T' + time.strftime("%m%d-%H%M%S") # default
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
        
        msk_tensor = None
        probe_msk_tensor = None
        probe_label = None
        
        for i, sample in enumerate(batch):
            sentence = sentences[i]
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded))
            
            # find all positions of '=' and add 1 to get the positions of the targets
            var_pos = [idx + 1 for idx, word in enumerate(sentence_idx) if word == self.vocab['=']]
            # the depth of the target is given by batch[i]['depths'], which is a list of the same size as var_pos
            dep_tensor[i, var_pos] = torch.tensor(sample['depths'], dtype=torch.long)

            oper_tensor[i, var_pos] = torch.tensor(sample['opers'], dtype=torch.long)
            
            y_tensor[i, var_pos] = torch.tensor(sample['values'], dtype=torch.long)
        
            
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": {
                "dep": dep_tensor, 
                "oper": oper_tensor
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



    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, batch_info = self._unpack_batch(batch)
        x, y, mask = train_batch
        dep, oper = batch_info['dep'], batch_info['oper']
        
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # mask (batch_size, seq_len)

        token_embedding = self.training_model.readin(x)
        hidden_state = token_embedding
        loss_ls = []
        mrr_ls = []
        loss_counter = []
        
        for cur_dep in range(self.max_dep):
            loss_ls.append(None)
            mrr_ls.append(None)
            loss_counter.append(0)
            
            if cur_dep == 0:
                hidden_state = self.training_model.encoder(token_embedding)
            else: 
                hidden_state = self.training_model.encoder(hidden_state + token_embedding) # shape (batch_size, seq_len, hidden_size)
                # Here, we also add the token embedding to the hidden_state to make the model aware of the input, see https://arxiv.org/pdf/2409.15647 
            
            # find all the indices in dep that are equal to cur_dep and dep < max_dep
            indices = torch.where(torch.logical_and(dep == cur_dep, dep < self.max_dep))
            
            selected_state = self._mask_select(hidden_state, indices)
            selected_label = self._mask_select(y, indices)
            if selected_state.numel() == 0:
                continue
            
            add_med_loss_prob = self.train_config.add_med_loss_prob[cur_dep] if len(self.train_config.add_med_loss_prob) > cur_dep else 1.0
            
            num_selected = selected_state.size(0)
            # select a subset of range(num_selected) to add the medium loss
            med_loss_indices = torch.randperm(num_selected)[:int(add_med_loss_prob*num_selected)]

            if len(med_loss_indices) == 0:
                continue

            selected_state = selected_state[med_loss_indices]
            selected_label = selected_label[med_loss_indices]
            
            # update loss counter
            self.med_loss_counter[cur_dep] += len(med_loss_indices)
            loss_counter[cur_dep] += len(med_loss_indices)

            selected_output = self.training_model.readout(selected_state)
            # compute loss
            loss_ls[-1] = self.loss_p_model(selected_output, selected_label)

            # compute mrr 
            mrr_ls[-1] = MRR_fn(selected_output, selected_label)

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
        mrr = 0.0
        num_effective_dep = 0
        for i, loss_i in enumerate(loss_ls):
            if loss_i is not None:
                loss_p += loss_i
                mrr += mrr_ls[i] 
                num_effective_dep += 1
                # The total number of parameters with small depth outnumbers the total number of parameters with large depth, so we don't want the mrr of small depth to dominate the mrr of large depth. We want to focus more on large depth. So is for the loss.

                self.log(f"{step_type}_loss_dep_{i}", loss_i, prog_bar=True, logger=True, batch_size=self.len_batch(batch), on_step=True, on_epoch=True)
                self.log(f"{step_type}_mrr_dep_{i}", mrr_ls[i], prog_bar=True, logger=True, batch_size=self.len_batch(batch), on_step=True, on_epoch=True)
        
        if num_effective_dep > 0:
            loss_p /= num_effective_dep
            mrr /= num_effective_dep
        else:
            return None

        self.log(f"{step_type}_loss", loss_p, prog_bar=True, logger=True, batch_size=self.len_batch(batch), on_step=True, on_epoch=True)
        self.log(f"{step_type}_mrr", mrr, prog_bar=True, logger=True, batch_size=self.len_batch(batch), on_step=True, on_epoch=True)
        if self.loss_n_model is not None:
            self.log(f"{step_type}_loss_n", loss_n, prog_bar=True, logger=True, batch_size=self.len_batch(batch), on_step=True, on_epoch=True)

        return loss_p, loss_n, output

        
 