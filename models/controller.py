"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F
import torch.nn as nn

import utils


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        hidden_size = args.controller_hidden_size
        num_funcs = args.controller_num_functions
        #self.g_emb = nn.Linear(1, hidden_size)
        self.g_emb = torch.zeros([1, hidden_size], dtype=torch.float32)
        self.w_emb = nn.Linear(hidden_size, num_funcs)
        # self.w_emb =
        self.attention_w_1 = nn.Linear(hidden_size, hidden_size)
        self.attention_w_2 = nn.Linear(hidden_size, hidden_size)
        self.attention_v = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.lstm = nn.LSTMCell(args.controller_hidden_size, args.controller_hidden_size)
        self.init_weights()


    def init_weights(self):
        initrange = 0.01
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.g_emb.data.uniform_(-initrange, initrange)

        self.w_emb.bias.data.zero_()
        self.w_emb.weight.data.uniform_(-initrange, initrange)

        self.attention_w_1.bias.data.zero_()
        self.attention_w_1.weight.data.uniform_(-initrange, initrange)

        self.attention_w_2.bias.data.zero_()
        self.attention_w_2.weight.data.uniform_(-initrange, initrange)

        self.attention_v.bias.data.zero_()
        self.attention_v.weight.data.uniform_(-initrange, initrange)


    def forward(self):
        pass


    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """Samples a architecture
        """

        hidden_size = self.args.controller_hidden_size
        num_layers = self.args.controller_num_layers

        arc_seq = []
        sample_log_probs = []
        sample_entropy = []
        all_h = [torch.zeros([1, hidden_size], dtype=torch.float32)]
        all_h_w = [torch.zeros([1, hidden_size], dtype=torch.float32)]

        inputs = self.g_emb
        prev_c = torch.zeros([1, hidden_size], dtype=torch.float32)
        prev_h = torch.zeros([1, hidden_size], dtype=torch.float32)


        for layer_id in range(1, num_layers + 1):

            # sample previous node
            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_h, prev_c = next_h, next_c
            all_h.append(next_h)
            all_h_w.append(self.attention_w_1(next_h))

            query = self.attention_w_2(next_h)
            query = query + torch.cat(all_h_w[:-1], 0)
            query = torch.tanh(query)
            logits = self.attention_v(query)
            logits = torch.reshape(logits, [1, layer_id])


            if self.args.controller_temperature:
                logits /= self.args.controller_temperature
            if self.args.controller_tanh_constant:
                logits = self.args.controller_tanh_constant * torch.tanh(logits)

            diff = (layer_id - torch.arange(0, layer_id)) ** 2
            diff  = diff.float()
            logits -= torch.reshape(diff, [1, layer_id]) / 6.0

            #
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False)
            )

            arc_seq.append(action[:, 0])
            sample_log_probs.append(selected_log_prob[:, 0])

            entropy = selected_log_prob[:, 0] * torch.exp(-selected_log_prob[:, 0])
            sample_entropy.append(entropy)


            # sample activation function
            inputs = torch.cat(all_h[:-1], 0)[action[:, 0]]
            inputs /= (0.1 + (layer_id - action[:, 0])).float()

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_h, prev_c = next_h, next_c

            logits = self.w_emb(next_h)

            if self.args.controller_temperature:
                logits /= self.args.controller_temperature
            if self.args.controller_tanh_constant:
                logits = self.args.controller_tanh_constant * torch.tanh(logits)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False)
            )
            arc_seq.append(action[:, 0])

            sample_log_probs.append(selected_log_prob[:, 0])
            entropy = selected_log_prob[:, 0] * torch.exp(-selected_log_prob[:, 0])
            sample_entropy.append(entropy)
            inputs = self.w_emb.weight.data[action[:, 0]]

        arc_seq = torch.cat(arc_seq, 0)
        # self.sample_arc = arc_seq
        #
        # self.sample_log_probs = torch.cat(sample_log_probs, 0)
        # self.ppl = torch.exp(torch.reduce_mean(self.sample_log_probs))
        #
        # sample_entropy = torch.cat(sample_entropy, 0)
        # self.sample_entropy = torch.reduce_sum(sample_entropy)
        #
        # self.all_h = all_h
        return arc_seq, torch.cat(sample_log_probs), torch.cat(sample_entropy)
