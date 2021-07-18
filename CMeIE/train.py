import torch
import torch as t
from torchcrf import CRF
from torch import nn
import sys
sys.path.append('./')
from preprocess import getdata
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset, collate_fn
from bbc import BBC
from torch import optim

if __name__ == '__main__':
    para = {'batch_size': 256, 'embedding_dim': 768, 'hidden_dim': 800, 'tagset_size': 91, 'dropout_rate': 0, 'epochs': 40}
    path = 'CMeIE/CMeEE_train.json'
    myDataset = CustomDataset(path, mode='train')
    dataloader = DataLoader(myDataset, batch_size=para['batch_size'], shuffle=True, collate_fn=collate_fn)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BBC(para).cuda()
    model.load_state_dict(torch.load('params.pkl'))
    # optimizer = optim.SGD(params=model.parameters(), lr=2e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_recoders = []
    loss_recoder = 0
    for epoch in range(para['epochs']):
        for index, (samples, tags, length) in enumerate(iter(dataloader)):
            optimizer.zero_grad()
            loss = model.get_loss(samples=samples, length=length, tags=tags)
            loss.backward()
            optimizer.step()
            print("epoch:%d batch:%d loss:%f" % (epoch, index, loss))
            loss_recoder += loss.item()
            if(index%40 == 39):
                print(loss_recoder)
                loss_recoders.append(loss_recoder)
                loss_recoder = 0
        torch.save(model.state_dict(), 'params.pkl')
    with open('loss.txt', 'w', encoding='utf-8') as f:
        for i in loss_recoders:
            f.write(str(i)+'\n')
    print("finished!")

