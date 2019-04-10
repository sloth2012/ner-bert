from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import json
from web.utils.common import timer

delimiter = '△△△'


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, labels, labels_ids, labels_mask, tok_map, cls=None, cls_idx=None, meta=None):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
        data[3]: list, tokens meta info (if meta is not None)
        ...
        data[-2]: list, labels mask
        data[-1]: list, labels ids
        """
        self.data = []
        # Bert data
        self.bert_tokens = bert_tokens
        self.input_ids = input_ids
        self.data.append(input_ids)
        self.input_mask = input_mask
        self.data.append(input_mask)
        self.input_type_ids = input_type_ids
        self.data.append(input_type_ids)
        # Meta data
        self.meta = meta
        if meta is not None:
            self.data.append(meta)
        # Origin data
        self.tokens = tokens
        self.labels = labels
        # Used for joint model
        self.cls = cls
        self.cls_idx = cls_idx
        if cls is not None:
            self.data.append(cls_idx)
        # Labels data
        self.labels_mask = labels_mask
        self.data.append(labels_mask)
        self.labels_ids = labels_ids
        self.data.append(labels_ids)
        self.tok_map = tok_map


class DataLoaderForTrain(DataLoader):

    def __init__(self, data_set, shuffle, cuda, **kwargs):
        super().__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            **kwargs
        )

        self.cuda = cuda and torch.cuda.is_available()

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        label_ml = max(map(lambda x_: sum(x_.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x_: sum(x_.data[1]), data)))[::-1]
        for idx in sorted_idx:
            f = data[idx]
            example = []
            for idx_, x in enumerate(f.data[:-2]):
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            example.append(f.data[-2][:label_ml])
            example.append(f.data[-1][:label_ml])
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            if data[0].meta is not None and idx == 3:
                res_.append(torch.FloatTensor(x))
            else:
                res_.append(torch.LongTensor(x))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
        return res_


class DataLoaderForPredict(DataLoader):

    def __init__(self, data_set, cuda, **kwargs):
        super().__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.cuda = cuda and torch.cuda.is_available()

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[1]), data))
        label_ml = max(map(lambda x_: sum(x_.data[-2]), data))
        sorted_idx = np.argsort(list(map(lambda x_: sum(x_.data[1]), data)))[::-1]
        for idx in sorted_idx:
            f = data[idx]
            example = []
            for x in f.data[:-2]:
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            example.append(f.data[-2][:label_ml])
            example.append(f.data[-1][:label_ml])
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            if data[0].meta is not None and idx == 3:
                res_.append(torch.FloatTensor(x))
            else:
                res_.append(torch.LongTensor(x))
        sorted_idx = torch.LongTensor(list(sorted_idx))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
            sorted_idx = sorted_idx.cuda()
        return res_, sorted_idx


@timer
def get_data(
        df, tokenizer, label2idx=None, max_seq_len=424, pad="<pad>", cls2idx=None,
        is_cls=False, is_meta=False):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1}
    features = []
    all_args = []
    if is_cls:
        # Use joint model
        if cls2idx is None:
            cls2idx = dict()
        all_args.extend([df["1"].tolist(), df["0"].tolist(), df["2"].tolist()])
    else:
        all_args.extend([df["1"].tolist(), df["0"].tolist()])
    if is_meta:
        all_args.append(df["3"].tolist())
    total = len(df["0"].tolist())
    cls = None
    meta = None
    for args in tqdm(enumerate(zip(*all_args)), total=total, desc="bert data"):
        if is_cls:
            if is_meta:
                idx, (text, labels, cls, meta) = args
            else:
                idx, (text, labels, cls) = args
        else:
            if is_meta:
                idx, (text, labels, meta) = args
            else:
                idx, (text, labels) = args

        tok_map = []
        meta_tokens = []
        if is_meta:
            meta = json.loads(meta)
            meta_tokens.append([0] * len(meta[0]))
        bert_tokens = []
        bert_labels = []
        bert_tokens.append("[CLS]")
        bert_labels.append("[CLS]")
        orig_tokens = []

        orig_tokens.extend(str(text).split())
        labels = str(labels).split()
        pad_idx = label2idx[pad]
        assert len(orig_tokens) == len(labels)
        prev_label = ""
        for idx_, (orig_token, label) in enumerate(zip(orig_tokens, labels)):
            # Fix BIO to IO as BERT proposed https://arxiv.org/pdf/1810.04805.pdf
            prefix = "B_"
            if label != "O":
                label = label.split("_")[1]
                if label == prev_label:
                    prefix = "I_"
                prev_label = label
            else:
                prev_label = label

            cur_tokens = tokenizer.tokenize(orig_token)
            if max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                break
            tok_map.append(len(bert_tokens))
            if is_meta:
                meta_tokens.extend([meta[idx_]] * len(cur_tokens))
            bert_tokens.extend(cur_tokens)
            # ["I_" + label] * (len(cur_tokens) - 1)
            bert_label = [prefix + label] + ["X"] * (len(cur_tokens) - 1)
            bert_labels.extend(bert_label)
        # bert_tokens.append("[SEP]")
        # bert_labels.append("[SEP]")
        if is_meta:
            meta_tokens.append([0] * len(meta[0]))
        # + ["[SEP]"]
        orig_tokens = ["[CLS]"] + orig_tokens

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        labels = bert_labels
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        labels_ids = [label2idx[l] for l in labels]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        labels_mask = [1] * len(labels_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            labels_ids.append(pad_idx)
            labels_mask.append(0)
            tok_map.append(-1)
            if is_meta:
                meta_tokens.append([0] * len(meta[0]))
        # assert len(input_ids) == len(bert_labels_ids)
        input_type_ids = [0] * len(input_ids)
        # For joint model
        cls_idx = None
        if is_cls:
            if cls not in cls2idx:
                cls2idx[cls] = len(cls2idx)
            cls_idx = cls2idx[cls]
        if is_meta:
            meta = meta_tokens
        features.append(InputFeatures(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            # Origin data
            tokens=orig_tokens,
            labels=labels,
            labels_ids=labels_ids,
            labels_mask=labels_mask,
            tok_map=tok_map,
            # Joint data
            cls=cls,
            cls_idx=cls_idx,
            # Meta data
            meta=meta
        ))
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(input_type_ids)
        if len(input_ids) != len(labels_ids):
            print(len(input_ids), len(labels_ids), orig_tokens, labels)
            raise Exception('len(input_ids) != len(labels_ids):')
        assert len(input_ids) == len(labels_mask)
    if is_cls:
        return features, (label2idx, cls2idx)
    return features, label2idx


def get_bert_data_loaders(train, valid, vocab_file, batch_size=16, cuda=True, is_cls=False,
                          do_lower_case=False, max_seq_len=424, is_meta=False, label2idx=None, cls2idx=None):
    global delimiter
    train = pd.read_csv(train, delimiter=delimiter)
    valid = pd.read_csv(valid, delimiter=delimiter)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_f, label2idx = get_data(
        train, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls, max_seq_len=max_seq_len, is_meta=is_meta)
    if is_cls:
        label2idx, cls2idx = label2idx
    train_dl = DataLoaderForTrain(
        train_f, batch_size=batch_size, shuffle=True, cuda=cuda)
    valid_f, label2idx = get_data(
        valid, tokenizer, label2idx, cls2idx=cls2idx, is_cls=is_cls, max_seq_len=max_seq_len, is_meta=is_meta)
    if is_cls:
        label2idx, cls2idx = label2idx
    valid_dl = DataLoaderForTrain(
        valid_f, batch_size=batch_size, cuda=cuda, shuffle=False)
    if is_cls:
        return train_dl, valid_dl, tokenizer, label2idx, max_seq_len, cls2idx
    return train_dl, valid_dl, tokenizer, label2idx, max_seq_len


def get_bert_data_loader_for_predict(path, learner):
    global delimiter
    df = pd.read_csv(path, delimiter=delimiter)
    f, _ = get_data(df, tokenizer=learner.data.tokenizer,
                    label2idx=learner.data.label2idx, cls2idx=learner.data.cls2idx,
                    is_cls=learner.data.is_cls,
                    max_seq_len=learner.data.max_seq_len, is_meta=learner.data.is_meta)
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=True)

    return dl


# TODO 暂时未用到cls和meta，所以先不考虑
def split_text(input_text_arr: str, max_seq_len, cls=None, meta=None):
    replace_chars = [
        '\x97',
        '\uf076',
        "\ue405",
        "\ue105",
        "\ue415",
        '\x07',
        '\x7f',
        '\u3000',
        '\xa0',
        ' '
    ]

    punctuation = ',，.。：:！;! '

    # 记录一些索引，用于还原。三元组，第一个元素：文本在输入list中的索引；第二个元素：0表示为换行，1表示直接追加；第三个元素：表示起始位置；
    # 这里三元组用于将各个输入的字符串数组进行分解组织，以供后续还原
    line_marker = []

    clean_text_arr = []

    for idx, input_text in enumerate(input_text_arr):
        # 设置两个指针，进行符合长度的串的提取
        pointer_st = 0
        pointer_ed = 0
        last_valid_punc_pos = -1

        text_list = list(input_text)
        for i, ch in enumerate(text_list):
            if ch in replace_chars:
                text_list[i] = 'unk'

            if (pointer_ed - pointer_st) > max_seq_len:
                if last_valid_punc_pos == -1:
                    valid_text_list = text_list[pointer_st:pointer_ed]

                    clean_text_arr.append((
                        ' '.join(valid_text_list),
                        ' '.join('O' * len(valid_text_list))
                    ))

                    line_marker.append((
                        idx, 1, (pointer_st, pointer_ed)
                    ))
                    pointer_st = pointer_ed

                else:
                    ed = last_valid_punc_pos + 1
                    valid_text_list = text_list[pointer_st:ed]

                    clean_text_arr.append((
                        ' '.join(valid_text_list),
                        ' '.join('O' * len(valid_text_list))
                    ))

                    line_marker.append((
                        idx, 1, (pointer_st, ed)
                    ))
                    pointer_st = ed

                last_valid_punc_pos = -1

            else:
                if ch == '\n':
                    valid_text_list = text_list[pointer_st: i]
                    if len(valid_text_list) != 0:
                        clean_text_arr.append((
                            ' '.join(valid_text_list),
                            ' '.join('O' * len(valid_text_list))
                        ))

                    line_marker.append((
                        idx, 0, (pointer_st, i)
                    ))
                    pointer_st = i + 1

                    last_valid_punc_pos = -1

                elif ch in punctuation:
                    last_valid_punc_pos = i

            pointer_ed = i + 1

        if pointer_st != pointer_ed:
            valid_text_list = text_list[pointer_st: pointer_ed]
            if len(valid_text_list) != 0:
                clean_text_arr.append((
                    ' '.join(valid_text_list),
                    ' '.join('O' * len(valid_text_list))
                ))

                line_marker.append((
                    idx, 1, (pointer_st, pointer_ed)
                ))

    return clean_text_arr, line_marker


# 先主要针对中文序列标注（单字），转换空格。一篇文本，不含换行符号
@timer
def text_array_for_predict(input_text_arr, learner):
    # 记录空行的索引，以供插入
    clean_text_arr, line_marker = split_text(
        input_text_arr=input_text_arr,
        max_seq_len=learner.data.max_seq_len
    )

    df = pd.DataFrame(clean_text_arr, columns=['1', '0'])
    # print(repr(df.values[0][0]), repr(df.values[0][1]))
    f, _ = get_data(
        df,
        tokenizer=learner.data.tokenizer,
        label2idx=learner.data.label2idx,
        cls2idx=learner.data.cls2idx,
        is_cls=learner.data.is_cls,
        max_seq_len=learner.data.max_seq_len,
        is_meta=learner.data.is_meta
    )
    cuda = torch.cuda.is_available()
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=cuda)

    # 此处耗时较多
    preds = learner.predict(dl)

    from modules.utils.utils import bert_labels2tokens, first_choicer, tokens2spans

    tokens, labels = bert_labels2tokens(dl, preds, fn=first_choicer)
    span_preds = tokens2spans(tokens, labels)

    results = []
    pred_counter = 0

    result = []
    input_text = ''
    last_index = -1
    for idx, (arr_index, marker, (pointer_st, pointer_ed)) in enumerate(line_marker):
        if arr_index != last_index:
            input_text = input_text_arr[arr_index]

            if last_index != -1:
                results.append(result)
                result = []

        if pointer_ed != pointer_st:
            # 下边为恢复机制
            pred = span_preds[pred_counter]
            st = 0
            text = input_text[pointer_st:pointer_ed]
            text_size = len(text)

            for token, lab in pred:
                valid_st = st
                tok_size = len(token)
                tok_st = 0

                # 目前就两个自定义边界
                while st < text_size and tok_st < tok_size:
                    if token[tok_st] == ' ':
                        tok_st += 1
                        continue
                    if text[st] != token[tok_st]:
                        # 标记为unk
                        tok_ed1 = tok_st + 3
                        if tok_ed1 > tok_size:
                            # print('err:', token, lab, token[tok_st:tok_ed1])
                            raise Exception('边界识别错误:unk')

                        if token[tok_st:tok_ed1] == 'unk':
                            tok_st = tok_ed1
                            st += 1
                        else:
                            # 标记为[UNK]
                            tok_ed2 = tok_st + 5
                            if tok_ed2 > tok_size:
                                raise Exception('边界识别错误1:[UNK]')

                            # print('err:', token, lab, token[tok_st:tok_ed2])
                            if token[tok_st:tok_ed2] == '[UNK]':
                                tok_st = tok_ed2
                                st += 1
                            else:
                                raise Exception('边界识别错误2:[UNK]')

                    else:
                        st += 1
                        tok_st += 1

                if tok_st < tok_size:
                    raise Exception('识别边界出错')

                result.append((text[valid_st: st], lab))

            pred_counter += 1

        if marker == 0:
            result.append(('\n', 'w'))

        last_index = arr_index

    if len(result) != 0:
        results.append(result)

    return results


class BertNerData(object):

    @property
    def config(self):
        config = {
            "train_path": self.train_path,
            "valid_path": self.valid_path,
            "vocab_file": self.vocab_file,
            "data_type": self.data_type,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "is_cls": self.is_cls,
            "cuda": self.cuda,
            "is_meta": self.is_meta,
            "label2idx": self.label2idx,
            "cls2idx": self.cls2idx
        }
        return config

    def __init__(self, train_path, valid_path, vocab_file, data_type,
                 train_dl=None, valid_dl=None, tokenizer=None,
                 label2idx=None, max_seq_len=424,
                 cls2idx=None, batch_size=16, cuda=True, is_meta=False):
        self.train_path = train_path
        self.valid_path = valid_path
        self.data_type = data_type
        self.vocab_file = vocab_file
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.cls2idx = cls2idx
        self.batch_size = batch_size
        self.is_meta = is_meta
        self.cuda = cuda and torch.cuda.is_available()
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])
        self.is_cls = False
        self.max_seq_len = max_seq_len
        if cls2idx is not None:
            self.is_cls = True
            self.id2cls = sorted(cls2idx.keys(), key=lambda x: cls2idx[x])

    def reload_dl(self, path, for_train=True):
        '''
            重新加载数据集
        :param path:
        :param for_train: 是否为了训练
        :return:
        '''

        global delimiter
        df = pd.read_csv(path, delimiter=delimiter)
        features, label2idx = get_data(
            df,
            tokenizer=self.tokenizer,
            label2idx=self.label2idx,
            cls2idx=self.cls2idx,
            is_cls=self.is_cls,
            max_seq_len=self.max_seq_len,
            is_meta=self.is_meta
        )

        f = DataLoaderForPredict if not for_train else DataLoaderForTrain
        dl = f(
            features,
            batch_size=self.batch_size,
            shuffle=for_train,
            cuda=self.cuda
        )

        self.label2idx = label2idx

        return dl

    # TODO: write docs
    @classmethod
    def from_config(cls, config, for_train=True):
        if config["data_type"] == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif config["data_type"] == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        if config["train_path"] and config["valid_path"] and for_train:
            fn_res = fn(config["train_path"], config["valid_path"], config["vocab_file"], config["batch_size"],
                        config["cuda"], config["is_cls"], do_lower_case, config["max_seq_len"], config["is_meta"],
                        label2idx=config["label2idx"], cls2idx=config["cls2idx"])
        else:
            fn_res = (None, None, tokenization.FullTokenizer(
                vocab_file=config["vocab_file"], do_lower_case=do_lower_case), config["label2idx"],
                      config["max_seq_len"], config["cls2idx"])
        return cls(
            config["train_path"], config["valid_path"], config["vocab_file"], config["data_type"],
            *fn_res, batch_size=config["batch_size"], cuda=config["cuda"], is_meta=config["is_meta"])

        # with open(config_path, "w") as f:
        #    json.dump(config, f)

    @classmethod
    def create(cls,
               train_path, valid_path, vocab_file, batch_size=16, cuda=True, is_cls=False,
               data_type="bert_cased", max_seq_len=424, is_meta=False):
        if data_type == "bert_cased":
            do_lower_case = False
            fn = get_bert_data_loaders
        elif data_type == "bert_uncased":
            do_lower_case = True
            fn = get_bert_data_loaders
        else:
            raise NotImplementedError("No requested mode :(.")
        return cls(
            train_path,
            valid_path,
            vocab_file,
            data_type,
            *fn(train_path, valid_path, vocab_file, batch_size, cuda, is_cls, do_lower_case, max_seq_len, is_meta),
            batch_size=batch_size,
            cuda=cuda,
            is_meta=is_meta
        )
