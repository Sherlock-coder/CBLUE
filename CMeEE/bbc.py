import torch as t
from torchcrf import  CRF
from torch import nn
import sys
sys.path.append('./')
from preprocess import getdata
import numpy as np
from dataset import CustomDataset, collate_fn


class BBC(nn.Module):
    def __init__(self, parameter):
        super(BBC, self).__init__()
        self.batch_size = parameter['batch_size']
        self.embedding_dim = parameter['embedding_dim']
        self.hidden_dim = parameter['hidden_dim']
        self.tagset_size = parameter['tagset_size']
        self.dropout_rate = parameter['dropout_rate']
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, bidirectional=True, batch_first=True, dropout=self.dropout_rate)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
        self.crf.reset_parameters()

    def _get_lstm_out(self, **kwargs):
        samples, length = kwargs['samples'], kwargs['length']
        samples_pack = nn.utils.rnn.pack_padded_sequence(samples, length, batch_first=True)
        lstm_out, _ = self.lstm(samples_pack)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.hidden2tag(lstm_out)
        return lstm_out

    def get_loss(self, **kwargs):
        samples, length, tags = kwargs['samples'], kwargs['length'], kwargs['tags']
        lstm_out = self._get_lstm_out(**kwargs)
        mask = (tags != 0)
        loss = -self.crf(lstm_out, tags, mask=mask)
        return loss

    def forward(self, **kwargs):
        lstm_out = self._get_lstm_out(**kwargs)
        res = self.crf.decode(lstm_out)
        res = t.tensor(res)
        return res

if __name__ == "__main__":
    para = {'batch_size': 40, 'embedding_dim': 768, 'hidden_dim': 20, 'tagset_size': 20, 'dropout_rate': 0, 'epoch': 100}
    x = BBC(para)
    for name, para in x.named_parameters():
        print(name, para)
    path = 'CMeEE/CMeEE/CMeEE_train.json'
    myDataset = CustomDataset(path)
    dataloader = DataLoader(myDataset, batch_size=40, shuffle=True, collate_fn=collate_fn)
    samples, tags, length = next(iter(dataloader))
    # loss = x.get_loss(samples=samples, length=length, tags=tags)
    # print(loss)