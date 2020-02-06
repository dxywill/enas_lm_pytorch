"""Module containing the shared RNN model."""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import models.shared_base
import utils


logger = utils.get_logger()


def _gen_mask(shape, drop_prob):
    """Generate a droppout mask."""
    keep_prob = 1. - drop_prob
    #mask = tf.random_uniform(shape, dtype=tf.float32)
    mask = torch.FloatTensor(shape[0], shape[1]).uniform_(0, 1)
    mask = torch.floor(mask + keep_prob) / keep_prob
    return mask

class LockedDropout(nn.Module):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def _set_default_params(params):
    """Set default hyper-parameters."""
    params.add_hparam('alpha', 0.0)  # activation L2 reg
    params.add_hparam('beta', 1.)  # activation slowness reg
    params.add_hparam('best_valid_ppl_threshold', 5)

    params.add_hparam('batch_size', FLAGS.child_batch_size)
    params.add_hparam('bptt_steps', FLAGS.child_bptt_steps)

    # for dropouts: dropping rate, NOT keeping rate
    params.add_hparam('drop_e', 0.10)  # word
    params.add_hparam('drop_i', 0.20)  # embeddings
    params.add_hparam('drop_x', 0.75)  # input to RNN cells
    params.add_hparam('drop_l', 0.25)  # between layers
    params.add_hparam('drop_o', 0.75)  # output
    params.add_hparam('drop_w', 0.00)  # weight

    params.add_hparam('grad_bound', 0.1)
    params.add_hparam('hidden_size', 200)
    params.add_hparam('init_range', 0.04)
    params.add_hparam('learning_rate', 20.)
    params.add_hparam('num_train_epochs', 600)
    params.add_hparam('vocab_size', 10000)

    params.add_hparam('weight_decay', 8e-7)
    return params


class RNN(models.shared_base.SharedModel):
    """Shared RNN model."""
    def __init__(self, args, corpus):
        models.shared_base.SharedModel.__init__(self)

        self.args = args
        self.corpus = corpus

        hidden_size = args.shared_hid
        num_layers = args.num_layers
        num_func = args.num_funcs

        self.decoder = nn.Linear(args.shared_hid, corpus.num_tokens)
        self.embeddings = nn.Embedding(corpus.num_tokens,
                                        args.shared_embed)
        self.lockdrop = LockedDropout()
        self.w_prev = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        # if self.args.tie_weights:
        #     self.decoder.weight = self.encoder.weight

        # NOTE(brendan): Since W^{x, c} and W^{h, c} are always summed, there
        # is no point duplicating their bias offset parameter. Likewise for
        # W^{x, h} and W^{h, h}.
        # self.w_xc = nn.Linear(args.shared_embed, args.shared_hid)
        # self.w_xh = nn.Linear(args.shared_embed, args.shared_hid)
        #
        # The raw weights are stored here because the hidden-to-hidden weights
        # are weight dropped on the forward pass.
        # self.w_hc_raw = torch.nn.Parameter(
        #     torch.Tensor(args.shared_hid, args.shared_hid))
        # self.w_hh_raw = torch.nn.Parameter(
        #     torch.Tensor(args.shared_hid, args.shared_hid))
        # self.w_hc = None
        # self.w_hh = None

        self.w_h = collections.defaultdict(dict)
        self.w_c = collections.defaultdict(dict)

        self.w_combined = collections.defaultdict(dict)

        for idx in range(num_layers):
            for jdx in range(idx + 1, num_layers):
                self.w_combined[idx][jdx] = []
                for f in num_func:
                    self.w_h[idx][jdx] = nn.Linear(args.shared_hid,
                                                   args.shared_hid,
                                                   bias=False)
                    self.w_c[idx][jdx] = nn.Linear(args.shared_hid,
                                                   args.shared_hid,
                                                   bias=False)
                    self.w_combined[idx][jdx].append(nn.Linear(args.shared_hid,
                                                   args.shared_hid * 2,
                                                   bias=False))


        # self._w_h = nn.ModuleList([self.w_h[idx][jdx]
        #                            for idx in self.w_h
        #                            for jdx in self.w_h[idx]])
        # self._w_c = nn.ModuleList([self.w_c[idx][jdx]
        #                            for idx in self.w_c
        #                            for jdx in self.w_c[idx]])


        # if args.mode == 'train':
        #     self.batch_norm = nn.BatchNorm1d(args.shared_hid)
        # else:
        #     self.batch_norm = None
        #
        # self.reset_parameters()
        # self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        #
        # logger.info(f'# of parameters: {format(self.num_parameters, ",d")}')

    def _rnn_fn(self, sample_arc, x, prev_s, input_mask, layer_mask,
                params):
        """Multi-layer LSTM.

        Args:
            sample_arc: [num_layers * 2], sequence of tokens representing architecture.
            x: [batch_size, num_steps, hidden_size].
            prev_s: [batch_size, hidden_size].
            w_prev: [2 * hidden_size, 2 * hidden_size].
            w_skip: [None, [hidden_size, 2 * hidden_size] * (num_layers-1)].
            input_mask: `[batch_size, hidden_size]`.
            layer_mask: `[batch_size, hidden_size]`.
            params: hyper-params object.

        Returns:
            next_s: [batch_size, hidden_size].
            all_s: [[batch_size, num_steps, hidden_size] * num_layers].
        """
        batch_size = x.size()[0]
        num_steps = x.size()[1]
        num_layers = len(sample_arc) // 2

        all_s = []

        # extract the relevant variables, so that you only do L2-reg on them.
        u_skip = []
        start_idx = 0
        for layer_id in range(num_layers):
            prev_idx = sample_arc[start_idx]
            func_idx = sample_arc[start_idx + 1]
            u_skip.append(self.w_combined[prev_idx][layer_id][func_idx])
            start_idx += 2
        w_skip = u_skip
        var_s = [self.w_prev] + w_skip[1:]

        def _select_function(h, function_id):
            h = torch.stack([F.tanh(h), F.relu(h), F.sigmoid(h), h], dim=0)
            h = h[function_id]
            return h

        step = 0
        while step < num_steps:

            """Body function."""
            inp = x[:, step, :]

            # important change: first input uses a tanh()
            if layer_mask is not None:
                assert input_mask is not None
                ht = self.w_prev(torch.cat([inp * input_mask, prev_s * layer_mask],
                                            dim=1))
            else:
                ht = self.w_prev(torch.cat([inp, prev_s], dim=1))
            h, t = torch.split(ht, 2, dim=1)
            h = F.tanh(h)
            t = F.sigmoid(t)
            s = prev_s + t * (h - prev_s)
            layers = [s]

            start_idx = 0
            used = []
            for layer_id in range(num_layers):
                prev_idx = sample_arc[start_idx]
                func_idx = sample_arc[start_idx + 1]
                # used.append(tf.one_hot(prev_idx, depth=num_layers, dtype=tf.int32)) not used?
                prev_s = torch.stack(layers, dim=0)[prev_idx]
                if layer_mask is not None:
                    ht = self.w_combined[prev_idx][layer_id][func_idx](prev_s * layer_mask)
                else:
                    ht = self.w_combined[prev_idx][layer_id][func_idx](prev_s)
                h, t = torch.split(ht, 2, dim=1)

                h = _select_function(h, func_idx)
                t = F.sigmoid(t)
                s = prev_s + t * (h - prev_s)
                # s.set_shape([batch_size, self.hidden_size])
                s = s.view(batch_size, self.hidden_size)
                layers.append(s)
                start_idx += 2

            next_s = torch.sum(layers[1:], dim=0) / num_layers
            all_s.append(next_s)

            prev_s = next_s
            step += 1

        # all_s = tf.transpose(all_s.stack(), [1, 0, 2])

        return next_s, all_s, var_s

    def forward(self, x, y, model_params, init_states, is_training=False):
        """Computes the logits.

        Args:
            x: [batch_size, num_steps], input batch.
            y: [batch_size, num_steps], output batch.
            model_params: a `dict` of params to use.
            init_states: a `dict` of params to use.
            is_training: if `True`, will apply regularizations.

        Returns:
            loss: scalar, cross-entropy loss
        """

        emb = self.embeddings(x)
        batch_size = self.params.batch_size
        hidden_size = self.params.hidden_size
        sample_arc = self.sample_arc
        prev_s = self.init_hidden(batch_size)

        if is_training:
            # emb = tf.layers.dropout(
            #         emb, self.params.drop_i, [batch_size, 1, hidden_size], training=True)
            emb  = self.lockdrop(emb, self.args.drop_i)

            input_mask = _gen_mask([batch_size, hidden_size], self.args.drop_x)
            layer_mask = _gen_mask([batch_size, hidden_size], self.args.drop_l)
        else:
            input_mask = None
            layer_mask = None

        out_s, all_s, var_s = self._rnn_fn(sample_arc, emb, prev_s,
                                        input_mask, layer_mask, params=self.params)

        top_s = all_s
        if is_training:
            # top_s = tf.layers.dropout(
            #         top_s, self.params.drop_o,
            #         [self.params.batch_size, 1, self.params.hidden_size], training=True)
            top_s = self.lockdrop(top_s, self.args.drop_o)

        prev_s = out_s
        logits = torch.einsum('bnh,vh->bnv', top_s, self.embeddings.weight.data)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
        #                                                     logits=logits)
        loss = self.loss_fn(logits, y)
        #loss = torch.reduce_mean(loss)

        reg_loss = loss  # `loss + regularization_terms` is for training only
        # if is_training:
        #     # L2 weight reg
        #     self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(w ** 2) for w in var_s])
        #     reg_loss += self.params.weight_decay * self.l2_reg_loss
        #
        #     # activation L2 reg
        #     reg_loss += self.params.alpha * tf.reduce_mean(all_s ** 2)
        #
        #     # activation slowness reg
        #     reg_loss += self.params.beta * tf.reduce_mean(
        #             (all_s[:, 1:, :] - all_s[:, :-1, :]) ** 2)


        return reg_loss, loss

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.shared_hid)
        return utils.get_variable(zeros, self.args.cuda, requires_grad=False)


    def get_num_cell_parameters(self, dag):
        num = 0

        num += models.shared_base.size(self.w_xc)
        num += models.shared_base.size(self.w_xh)

        q = collections.deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    assert len(nodes) == 1, 'parent of leaf node should have only one child'
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                num += models.shared_base.size(w_h)
                num += models.shared_base.size(w_c)

                q.append(next_id)

        logger.debug(f'# of cell parameters: '
                     f'{format(self.num_parameters, ",d")}')
        return num

    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
