import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence 

class EntityDataset(Dataset):
    def __init__(self,texts,pos,tags,config):
        self.texts=texts
        self.pos=pos
        self.tags=tags
        self.pos_pad=config.POS_PAD
        self.tag_pad=config.TAG_PAD
        self.config=config

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        pos=self.pos[idx]
        tag=self.tags[idx]

        input_ids=[]
        target_pos=[]
        target_tag=[]

        for i,word in enumerate(text):
            token=self.config.TOKENIZER(
                word,
                add_special_tokens=False
            )
            in_ids=token.input_ids
            input_lens=len(in_ids)
            input_ids.extend(in_ids)
            target_pos+=[pos[i]]*input_lens
            target_tag+=[tag[i]]*input_lens
        input_ids=input_ids[:self.config.MAX_LENGTH-2]
        target_pos=target_pos[:self.config.MAX_LENGTH-2]
        target_tag=target_tag[:self.config.MAX_LENGTH-2]

        input_ids=[self.config.VOCAB['<s>']]+input_ids+[self.config.VOCAB['</s>']]# add [CLS] and [SEP] token
        target_pos=[self.pos_pad]+target_pos+[self.pos_pad]
        target_tag=[self.tag_pad]+target_tag+[self.tag_pad]

        attention_mask=[1]*len(input_ids)
        #token_type_ids=[0]*len(input_ids)

        # if len(input_ids)<self.config.MAX_LENGTH:
        #   padding_length=self.config.MAX_LENGTH-len(input_ids)
        #   input_ids=input_ids+[self.config.VOCAB['[PAD]']]*padding_length
        #   attention_mask=attention_mask+[0]*padding_length
        #   target_pos=target_pos+[self.pos_pad]*padding_length
        #   target_tag=target_tag+[self.tag_pad]*padding_length
        return {
            "input_ids":torch.tensor(input_ids,dtype=torch.long),
            "attention_mask":torch.tensor(attention_mask,dtype=torch.long),
            "target_pos":torch.tensor(target_pos,dtype=torch.long),
            "target_tag":torch.tensor(target_tag,dtype=torch.long)
        }


class MyCollate:
    def __init__(self,ids_pad,pos_pad,tag_pad): 
        self.ids_pad=ids_pad
        self.pos_pad=pos_pad
        self.tag_pad=tag_pad
    
    def __call__(self,batch):
        input_ids=[features['input_ids'] for features in batch]
        attention_mask=[features['attention_mask'] for features in batch]
        target_pos=[features['target_pos'] for features in batch]
        target_tag=[features['target_tag'] for features in batch]

        input_ids=pad_sequence(input_ids,batch_first=True,padding_value=self.ids_pad)
        attention_mask=pad_sequence(attention_mask,batch_first=True,padding_value=0)
        target_pos=pad_sequence(target_pos,batch_first=True,padding_value=self.pos_pad)
        target_tag=pad_sequence(target_tag,batch_first=True,padding_value=self.tag_pad)
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "target_pos":target_pos,
            "target_tag":target_tag
        }