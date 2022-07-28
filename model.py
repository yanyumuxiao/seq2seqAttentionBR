import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer.GGNN import GlobalAggregator


class Encoder(nn.Module):
    def __init__(self, config, embedding, cate_embedding, adj):
        super(Encoder, self).__init__()
        self.config = config
        self.enc_embedding = embedding
        self.cate_embedding = cate_embedding

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden), padding=0) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.hidden)

        self.adj = adj
        self.ggnn = GlobalAggregator(config.hidden, config.item_count + 2, config.n_steps)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    # def forward(self, enc_inputs):
    #     # ggnn_feature = self.ggnn(self.enc_embedding.weight, self.adj)
    #     # (1) 提取全局特征供cnn卷积
    #     # (2) 提取全局特征作为辅助特征plus、concat、element-wise
    #     # 这样写把eos和pad的embedding也放进去ggnn，会不会导致特征也被训练，或者说被训练之后有影响吗
    #     # item_feature = ggnn_feature[enc_inputs]
    #
    #     init_item_embedding = self.enc_embedding.weight[:self.config.item_count]
    #     init_cate_embedding = self.cate_embedding(self.config.cate_list[:self.config.item_count])
    #     init_embedding = torch.cat([init_item_embedding, init_cate_embedding], -1)
    #     ggnn_feature = self.ggnn(init_embedding, self.adj)
    #     out = ggnn_feature[enc_inputs]
    #
    #     mask = (enc_inputs != self.config.pad).unsqueeze(-1).to(self.config.device)
    #     out = out * mask
    #
    #     pad_size = max(0, self.config.filter_sizes[-1] - out.shape[1])
    #     out = F.pad(out, (0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
    #     out = out.unsqueeze(1)
    #     out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
    #     out = self.dropout(out)
    #     out = self.fc(out)
    #
    #     return out

    def forward(self, enc_inputs):
        # ggnn_feature = self.ggnn(self.enc_embedding.weight, self.adj)
        # (1) 提取全局特征供cnn卷积
        # (2) 提取全局特征作为辅助特征plus、concat、element-wise
        # 这样写把eos和pad的embedding也放进去ggnn，会不会导致特征也被训练，或者说被训练之后有影响吗
        # item_feature = ggnn_feature[enc_inputs]

        out = self.enc_embedding(enc_inputs)

        cate_inputs = self.config.cate_list[enc_inputs].to(self.config.device)
        cate_feature = self.cate_embedding(cate_inputs)
        out = torch.cat([out, cate_feature], 2)

        mask = (enc_inputs != self.config.pad).unsqueeze(-1).to(self.config.device)
        out = out * mask

        pad_size = max(0, self.config.filter_sizes[-1] - out.shape[1])
        out = F.pad(out, (0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


        # mask = (enc_inputs != self.config.pad).unsqueeze(-1).to(self.config.device)
        # hidden = torch.sum(item_feature * mask, 1) / torch.sum(mask, 1)
        # return hidden

        # pos_emb = self.pos_embedding.weight[:len]
        # pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        #
        # hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        # hs = hs.unsqueeze(-2).repeat(1, len, 1)
        # nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        # nh = torch.tanh(nh)
        # nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        # beta = torch.matmul(nh, self.w_2)
        # beta = beta * mask
        # select = torch.sum(beta * hidden, 1)


class Decoder(nn.Module):
    def __init__(self, config, embedding, cate_embedding):
        super(Decoder, self).__init__()
        self.config = config
        self.dec_embedding = embedding
        self.cate_embedding = cate_embedding
        self.bidirectional = config.bidirectional

        self.gru = nn.GRU(input_size=config.hidden + config.dec_hidden, hidden_size=config.hidden,
                          num_layers=config.num_layers)
        self.fc = nn.Linear(config.hidden + config.dec_hidden * 2, config.item_count + 3)
        # 输出尺寸item_count加不加2

        self.dropout = nn.Dropout(config.dropout)

    def get_cate_embedding(self, inputs):
        item_embedding = self.dec_embedding(inputs)
        cate_inputs = self.config.cate_list[inputs]
        cate_embedding = self.cate_embedding(cate_inputs)
        outputs = torch.cat([item_embedding, cate_embedding], 2)
        return outputs

    def forward(self, dec_inputs, context, hidden):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        dec_inputs = dec_inputs.unsqueeze(0)
        # dec_inputs = [1, batch size]

        embedded = self.dropout(self.get_cate_embedding(dec_inputs))
        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.gru(emb_con, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc(output)
        # prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, config, adj):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.item_count + 3, config.hidden // 2)
        self.cate_embedding = nn.Embedding(config.cate_count + 1, config.hidden // 2)

        self.encoder = Encoder(config, self.embedding, self.cate_embedding, adj)
        self.decoder = Decoder(config, self.embedding, self.cate_embedding)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, enc_inputs, pos, teacher_forcing_ratio):
        # enc_inputs = self.embedding(enc_inputs)
        batch_size = pos.shape[0]
        dec_max_len = pos.shape[1] + 1

        go_symbol = (torch.ones(batch_size, 1) * self.config.go_symbol).long().to(self.config.device)
        pos = torch.cat([go_symbol, pos], 1)

        # 编码器的输出作为上下文向量
        context = self.encoder(enc_inputs).unsqueeze(0)

        # 解码器的初始输入，item_count作为<start>
        dec_inputs = pos[:, 0]
        # dec_inputs = (torch.ones(batch_size) * self.config.item_count).long().to(self.config.device)

        hidden = context

        # 创建outputs张量存储decoder输出
        outputs = torch.zeros(dec_max_len, batch_size, self.config.item_count + 3).to(self.config.device)

        for t in range(1, dec_max_len):
            output, hidden = self.decoder(dec_inputs, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            # 取出概率最大值的索引，即单词在字典里对应的id
            pred_token_id = output.max(1)[1]

            # 可能取真实值作为下一时刻的输入，也有可能取这一次的预测值作为下一时刻的输入
            dec_inputs = pos[:, t] if teacher_force else pred_token_id

        return outputs, context
