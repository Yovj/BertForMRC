from config import Config
from transformers import BertTokenizer,BertForQuestionAnswering
import torch.nn as nn
import os
import torch

config = Config()

class BertForQA(nn.Module):
    def __init__(self,config):
        super(BertForQA, self).__init__()
        self.BertModule = BertForQuestionAnswering.from_pretrained(os.path.join(config.model_dir,config.model_name),output_hidden_states=True)
        self.type_linear = nn.Linear(config.hidden_size,config.num_type)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,inputs):
        outputs  = self.BertModule(**inputs)
        loss = outputs.get('loss',torch.tensor([]))

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        hidden_states = outputs['hidden_states'][0]

        type_logit = self.type_linear(hidden_states[:,0:,])
        type_prob = self.softmax(type_logit)





        return loss,start_logits,end_logits,type_prob

