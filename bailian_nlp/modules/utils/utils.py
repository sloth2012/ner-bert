# coding: utf8
import numpy as np


def bert_labels2tokens(dl, labels):
    res_tokens = []
    res_labels = []

    for f, l in zip(dl.dataset, labels):
        res_tokens.append(f.bert_tokens[1:])
        res_labels.append(l[1:])

    return res_tokens, res_labels


def tokens2spans_(tokens_, labels_):
    res = []
    idx_ = 0

    # print('model cut result:')
    # for i, (t, l) in enumerate(zip(tokens_, labels_)):
    #     print(i, t, l)

    while idx_ < len(labels_):
        label = labels_[idx_]

        if label in ["I_O", "B_O", "O"]:
            res.append((tokens_[idx_], "O"))
            idx_ += 1
        elif label == "[SEP]" or label == "<eos>":
            break
        elif label == "[CLS]" or label == "<bos>":
            res.append((tokens_[idx_], label))
            idx_ += 1
        else:
            span = [tokens_[idx_]]
            try:
                span_label = labels_[idx_].split("_")[1]
            except IndexError:
                print('error:', label, labels_[idx_].split("_"))
                span_label = None
            idx_ += 1
            while idx_ < len(labels_) and labels_[idx_] not in ["I_O", "B_O", "O"] \
                    and labels_[idx_].split("_")[0] in ["I", "E"]:
                if span_label == labels_[idx_].split("_")[1]:
                    span.append(tokens_[idx_])
                    idx_ += 1
                else:
                    break
            res.append((" ".join(span), span_label))
    return res


def tokens2spans(tokens, labels):
    assert len(tokens) == len(labels)

    return list(map(lambda x: tokens2spans_(*x), zip(tokens, labels)))


def encode_position(pos, emb_dim=10):
    """The sinusoid position encoding"""

    # keep dim 0 for padding token position encoding zero vector
    if pos == 0:
        return np.zeros(emb_dim)
    position_enc = np.array(
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)])

    # apply sin on 0th,2nd,4th...emb_dim
    position_enc[0::2] = np.sin(position_enc[0::2])
    # apply cos on 1st,3rd,5th...emb_dim
    position_enc[1::2] = np.cos(position_enc[1::2])
    return list(position_enc.reshape(-1))


def recover_model_from_config(config: dict):
    # TODO 这里先只用bert
    from ..models import bert_models

    if not isinstance(config, dict) or 'name' not in config:
        return config

    model_type = getattr(bert_models, config['name'])

    return model_type.from_config(config['params'])


# TODO 模型显存预估函数待完成
# def modelsize(model, batch, type_size=4, optimizer="adam"):
#     '''
#     模型显存占用监测函数
#     :param model: 输入的模型
#     :param batch: 实际中需要输入的Tensor变量
#     :param type_size: 默认为 4 默认类型为 float32
#     :return:
#     '''
#     import torch.nn as nn
#     import torch
#     para = sum([np.prod(list(p.size())) for p in model.parameters()])
#     print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
#
#     with torch.no_grad():
#         input_ = batch
#
#         mods = list(model.modules())
#         out_sizes = []
#
#         def size_counter(o):
#             if isinstance(o, tuple) or isinstance(o, list):
#                 for i in o:
#                     size_counter(i)
#             else:
#                 out_sizes.append(np.array(o.size()))
#
#         print(len(mods))
#         for i in range(1, len(mods)):
#             m = mods[i]
#             if isinstance(m, nn.ReLU):
#                 if m.inplace:
#                     continue
#             try:
#                 out = m.forward(input_)
#             except ValueError:
#                 out = m.forward(*input_)
#
#             size_counter(out)
#             print(out_sizes)
#             input_ = out
#
#         total_nums = 0
#         for i in range(len(out_sizes)):
#             s = out_sizes[i]
#             nums = np.prod(np.array(s))
#             total_nums += nums
#
#     '''
#         显存占用 = 模型自身参数 × n + batch size × 输出参数量 × 2 + 一个batch的输入数据（往往忽略）
#         n是根据优化算法来定的，如果选用SGD， 则 n = 2， 如果选择Adam， 则 n = 4.
#     '''
#     base_n = 4 if optimizer.lower() == 'adam' else 2
#     print('Model {} : intermedite variables: {:3f} M (without backward)'
#           .format(model._get_name(), total_nums * type_size / 1000 / 1000))
#     print('Model {} : intermedite variables: {:3f} M (with backward)'
#           .format(model._get_name(), total_nums * type_size * base_n / 1000 / 1000))
#
#
# if __name__ == '__main__':
#     from bailian_nlp.modules import BertData
#     from bailian_nlp.released.settings import DEFAULT_POS_TRAIN_FILE, DEFAULT_POS_VALID_FILE, CHINESE_BERT_VOCAB_FILE, \
#         CHINESE_BERT_MODEL_CONFIG_FILE, CHINESE_BERT_MODEL_FILE
#
#     import os
#     train_path = os.path.join(os.path.dirname(DEFAULT_POS_TRAIN_FILE), 'train_small.csv')
#     data = BertData.create(
#         train_path,
#         DEFAULT_POS_VALID_FILE,
#         CHINESE_BERT_VOCAB_FILE,
#         data_type="bert_uncased",
#         max_seq_len=128,
#         batch_size=128
#     )
#
#     from bailian_nlp.modules.models import bert_models
#
#     model = bert_models.BertBiLSTMAttnCRF.create(
#         len(data.label2idx),
#         CHINESE_BERT_MODEL_CONFIG_FILE,
#         CHINESE_BERT_MODEL_FILE,
#         enc_hidden_dim=256,
#         freeze=False,
#     )
#
#     for batch in data.train_dl:
#         modelsize(model, batch)
#         break
