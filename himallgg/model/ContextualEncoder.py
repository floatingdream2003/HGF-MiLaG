import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextualEncoder(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(ContextualEncoder, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, text_len_tensor, text_tensor):
        packed = pack_padded_sequence(
            text_tensor, #输入文本的数据
            text_len_tensor.to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )
        rnn_out, (_, _) = self.rnn(packed, None)#LSTM
        # rnn_out, _ = self.rnn(packed, None)#GRU
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out
