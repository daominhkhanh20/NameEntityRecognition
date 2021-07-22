import torch 
from torch import nn

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.config=config
        self.model=config.MODEL
        self.linear_pos=nn.Linear(self.model.config.hidden_size,config.NUM_POS)
        self.linear_tag=nn.Linear(self.model.config.hidden_size,config.NUM_TAG)

        self.norm1=nn.LayerNorm(self.model.config.hidden_size)
        self.norm2=nn.LayerNorm(self.model.config.hidden_size)
        self.loss_fn=nn.CrossEntropyLoss()
    
    def forward(self,input_ids,attention_mask,target_pos,target_tag):
        bert_outputs=self.model(input_ids,attention_mask)
        last_hidden_state=bert_outputs.last_hidden_state#batch_size*seq_length*hidden_size
        out_tag=self.norm1(last_hidden_state)
        out_pos=self.norm2(last_hidden_state)
        tag=self.linear_tag(out_tag)#batch_size*seq_length*n_tags
        pos=self.linear_pos(out_pos)#batch_size*seq_length*n_pos
        #print("Tag ---->",end="")
        loss_tag=self.loss_func(tag,target_tag,attention_mask,self.config.NUM_TAG)
        #print("Pos ---->",end="")
        loss_pos=self.loss_func(pos,target_pos,attention_mask,self.config.NUM_POS)
        loss=(loss_pos+loss_tag)/2
        return pos,tag,loss 

    def loss_func(self,outputs,targets,attention_mask,num_outputs):
        activate_loss=attention_mask.view(-1)==1
        activate_logit=outputs.view(-1,num_outputs)#(batch_size*seq,num_outputs)
        activate_label=torch.where(
            activate_loss, #condition
            targets.view(-1), #if true
            torch.tensor(self.loss_fn.ignore_index,dtype=torch.long).to(DEVICE) # if False
        )
        loss=self.loss_fn(activate_logit,activate_label)
        #print(loss.item())
        return loss