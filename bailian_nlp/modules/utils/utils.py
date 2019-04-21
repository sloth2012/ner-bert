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


