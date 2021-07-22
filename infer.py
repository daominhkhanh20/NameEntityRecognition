from data import EntityDataset
import torch
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--string',type=str,required=True)

arg=parser.parse_args()

DEVICE=torch.device("cuda" if torch.cuda.is_availabel() else "cpu")
def infer(sentence,model,Config):
    test_dataset=EntityDataset([sentence.split()],[[0]*len(sentence)],[[0]*len(sentence)],Config)

    model.eval()
    with torch.no_grad():
        data=test_dataset.__getitem__(0)
        for k,v in data.items():
            data[k]=v.unsqueeze(dim=0).to(DEVICE)
        pos,tag,loss=model(**data)
        tag=tag.squeeze(dim=0)
        pos=pos.squeeze(dim=0)
        pos_index=pos.argmax(dim=1).cpu().detach().numpy().tolist()
        tag_index=tag.argmax(dim=1).cpu().detach().numpy().tolist()
        print(sentence)
        sub_words=[Config.VOCAB_INVERSE[value] for value in data['input_ids'][0].cpu().detach().numpy().tolist()]
        print(sub_words)
    #     print(pos_index)
    #     print(tag_index)
        print(Config.ENCODER_POS.inverse_transform(pos_index))
        print(Config.ENCODER_TAG.inverse_transform(tag_index))
    