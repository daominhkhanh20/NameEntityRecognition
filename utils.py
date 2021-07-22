from torch.autograd.grad_mode import no_grad
from data import EntityDataset,MyCollate
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import torch 
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pickle 
import os 
import time 

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loader(train_dataset,val_dataset,config):
    train_loader=DataLoader(train_dataset,batch_size=config.BATCH_SIZE,num_workers=2,shuffle=True,collate_fn=MyCollate(config.VOCAB['<pad>'],config.POS_PAD,config.TAG_PAD))
    val_loader=DataLoader(val_dataset,batch_size=config.BATCH_SIZE,num_workers=2,shuffle=True,collate_fn=MyCollate(config.VOCAB['<pad>'],config.POS_PAD,config.TAG_PAD))
    return train_loader,val_loader


def preprocess(config):
    data=pd.read_csv(config.PATH_FILE,encoding='latin1',error_bad_lines=False)
    config.NUM_POS=len(data.POS.unique())
    config.NUM_TAG=len(data.Tag.unique())
    data.loc[:,'Sentence #']=data['Sentence #'].fillna(method='ffill')#forward fille
    encoder_tag=LabelEncoder()
    encoder_pos=LabelEncoder()
    data.loc[:,'POS']=encoder_pos.fit_transform(data['POS'])
    data.loc[:,'Tag']=encoder_tag.fit_transform(data['Tag'])
    tag_pad=encoder_tag.transform(['O'])[0]
    pos_pad=encoder_pos.transform(['$'])[0]
    config.ENCODER_TAG=encoder_tag
    config.ENCODER_POS=encoder_pos

    sentences=data.groupby("Sentence #")["Word"].apply(list).values
    pos=data.groupby("Sentence #")["POS"].apply(list).values
    tags=data.groupby("Sentence #")["Tag"].apply(list).values
    return sentences,pos,tags,tag_pad,pos_pad

def get_dataloader(config):
    sentences,pos,tags,tag_pad,pos_pad=preprocess(config)
    config.POS_PAD=pos_pad
    config.TAG_PAD=tag_pad
    train_sentences,val_sentences,train_pos,val_pos,train_tag,val_tag=train_test_split(
        sentences,pos,tags,test_size=0.2,shuffle=True
    )
    train_dataset=EntityDataset(train_sentences,train_pos,train_tag,config)
    val_dataset=EntityDataset(val_sentences,val_pos,val_tag,config)
    train_loader,val_loader=get_loader(train_dataset,val_dataset,config)
    return train_loader,val_loader


def evaluate(model,val_loader,num_pos,num_tags):
    print('------------------TIME FOR EVALUATE---------------------')
    model.eval()
    preds_pos,preds_tag,targets_pos,targets_tag=[],[],[],[]
    with torch.no_grad():
        val_loss=0
        for idx,features in enumerate(val_loader):
            for i,v in features.items():
                features[i]=v.to(DEVICE)
            pos,tag,loss=model(**features)
            val_loss+=loss.item()
            pos=pos.view(-1,num_pos)
            tag=tag.view(-1,num_tags)
            pos_predict=torch.argmax(pos,dim=1)
            tag_predict=torch.argmax(tag,dim=1)
            preds_pos.extend(pos_predict.cpu().detach().numpy().tolist())
            preds_tag.extend(tag_predict.cpu().detach().numpy().tolist())
            targets_pos.extend(features['target_pos'].view(-1).cpu().detach().numpy().tolist())
            targets_tag.extend(features['target_tag'].view(-1).cpu().detach().numpy().tolist())
            if idx%100==0:
                print(idx,end=" ")
    print() 
    return val_loss/len(val_loader),accuracy_score(targets_pos,preds_pos),accuracy_score(targets_tag,preds_tag)


def save_checkpoint(model,optimizer,scheduler,history,epoch):
#     path="/content/drive/MyDrive/Model"
#     if os.path.exists(path) is False:
#         os.makedirs(path,exist_ok=True)

    with open(f"/hitory{epoch}.pickle",'wb') as file:
        pickle.dump(history,file,protocol=pickle.HIGHEST_PROTOCOL)
    print("History done")
    model_state={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
    }
    torch.save(model_state,f'/model{epoch}.pth')
    print("Save model done")

def train_model(model,train_loader,val_loader,optimizer,scheduler,epochs,num_pos,num_tags):
    model.train()
    history=defaultdict(list)
    for epoch in range(epochs):
        print('-------------------------TIME FOR TRAINING---------------------')
        start_time=time.time()
        preds_pos,preds_tag,targets_pos,targets_tag=[],[],[],[]
        train_loss=0
        for idx,features in enumerate(train_loader):
            for i,v in features.items():
                features[i]=v.to(DEVICE)
            
            optimizer.zero_grad()
            pos,tag,loss=model(**features)#batch_size*seq_length*n_classes
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss+=loss.item()
            pos=pos.view(-1,num_pos)
            tag=tag.view(-1,num_tags)
            pos_predict=torch.argmax(pos,dim=1)
            tag_predict=torch.argmax(tag,dim=1)
            preds_pos.extend(pos_predict.cpu().detach().numpy().tolist())
            preds_tag.extend(tag_predict.cpu().detach().numpy().tolist())
            targets_pos.extend(features['target_pos'].view(-1).cpu().detach().numpy().tolist())
            targets_tag.extend(features['target_tag'].view(-1).cpu().detach().numpy().tolist())

            if idx%100==0:
                print(idx,end=" ")

        print() 

        train_loss/=len(train_loader)
        train_acc_pos=accuracy_score(targets_pos,preds_pos)
        train_acc_tag=accuracy_score(targets_tag,preds_tag)
        val_loss,val_acc_pos,val_acc_tag=evaluate(model,val_loader,num_pos,num_tags)
        print(f"Epoch:{epoch}--Train loss:{train_loss}--Val loss:{val_loss}\n"+
                f"       --Train pos acc:{train_acc_pos}--Val pos acc:{val_acc_pos}\n"+
                f"       --Train tag acc:{train_acc_tag}--Val tag acc:{val_acc_tag}--Time:{time.time()-start_time}")
        history['train_loss']=train_loss
        history['val_loss']=val_loss
        history['train_acc_pos']=train_acc_pos
        history['val_acc_pos']=val_acc_pos
        history['train_acc_tag']=train_acc_tag
        history['val_acc_tag']=val_acc_tag
        if epoch >0 and epoch%5==0:
            save_checkpoint(model,optimizer,scheduler,history,epoch)
