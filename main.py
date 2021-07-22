from collections import defaultdict
from numpy.testing._private.utils import HAS_LAPACK64
from transformers import AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,RobertaModel,RobertaTokenizer
from utils import get_dataloader
from model import Model 
from utils import train_model 
import argparse
import torch 
DEVICE=torch.device("cuda" if torch.cuda.is_availabel() else "cpu")
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,required=False,default='bert-base-uncased')
parser.add_argument('--file_train',type=str,required=True,help='Fath to file train')
parser.add_argument('--batch_size',type=int,required=True,help='Number example in batch')
parser.add_argument('--max_length',type=int,required=False,default=128)
parser.add_argument('--n_epochs',type=int,required=True,help="Num epochs for training")
parser.add_argument('--lr',type=float,required=True,help='Learning rate')

arg=parser.parse_args()

class Config:
    BATCH_SIZE=64
    PATH_FILE='/Data/ner_dataset.csv'
    MAX_LENGTH=128
    MODEL=RobertaModel.from_pretrained('roberta-base')
    TOKENIZER=RobertaTokenizer.from_pretrained('roberta-base')
    VOCAB=TOKENIZER.get_vocab()
    EPOCHS=5

train_loader,val_loader=get_dataloader(Config)
model=Model(Config).to(DEVICE)
optimizer=AdamW(model.parameters(),lr=1e-5)
scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=len(train_loader)*Config.EPOCHS,
    num_warmup_steps=50
)

train_model(model,optimizer,train_loader,val_loader,scheduler,arg.n_epochs)

#python3 main.py --batch_size=32 --n_epochs 100 --lr 1e-5
