from torch.utils.data import DataLoader
from . import tokenization
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from bailian_nlp.web.utils.common import timer
from ..settings import DELIMITER


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens,
            input_ids,
            input_mask,
            input_type_ids,

            # Origin data
            labels,
            labels_ids,
            labels_mask,
    ):
        """
        Data has the following structure.
        data[0]: list, tokens ids
        data[1]: list, tokens mask
        data[2]: list, tokens type ids (for bert)
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

        # Origin data
        self.labels = labels

        # Labels data
        self.labels_mask = labels_mask
        self.data.append(labels_mask)
        self.labels_ids = labels_ids
        self.data.append(labels_ids)


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
            res_.append(torch.LongTensor(x))
        sorted_idx = torch.LongTensor(list(sorted_idx))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
            sorted_idx = sorted_idx.cuda()
        return res_, sorted_idx


@timer
def get_data(
        df,
        tokenizer,
        label2idx=None,
        max_seq_len=424,
        pad="<pad>",
):
    if label2idx is None:
        label2idx = {pad: 0, '[CLS]': 1}
    features = []
    all_args = [df["0"].tolist(), df["1"].tolist()]
    total = len(df)
    for args in tqdm(enumerate(zip(*all_args)), total=total, desc="bert data"):
        idx, (text, text_label) = args

        bert_tokens = []
        bert_labels = []
        bert_tokens.append("[CLS]")
        bert_labels.append("[CLS]")

        tokens = str(text).split()
        labels = str(text_label).split()

        pad_idx = label2idx[pad]

        limit_size = min(max_seq_len - 2, len(tokens))
        bert_tokens.extend(tokens[:limit_size])
        bert_labels.extend(labels[:limit_size])

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        for l in bert_labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)
        labels_ids = [label2idx[l] for l in bert_labels]

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
        # assert len(input_ids) == len(bert_labels_ids)
        input_type_ids = [0] * len(input_ids)
        features.append(InputFeatures(
            # Bert data
            bert_tokens=bert_tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            # Origin data
            labels=bert_labels,
            labels_ids=labels_ids,
            labels_mask=labels_mask,
        ))
        assert len(input_ids) == len(input_mask)
        assert len(input_ids) == len(input_type_ids)
        if len(input_ids) != len(labels_ids):
            print(len(input_ids), len(labels_ids), bert_tokens, bert_labels)
            raise Exception('len(input_ids) != len(labels_ids):')
        assert len(input_ids) == len(labels_mask)
    return features, label2idx


def get_bert_data_loaders(
        train,
        valid,
        vocab_file,
        batch_size=16,
        cuda=True,
        do_lower_case=True,
        max_seq_len=424,
        label2idx=None,
):
    train = pd.read_csv(train, delimiter=DELIMITER)
    valid = pd.read_csv(valid, delimiter=DELIMITER)

    tokenizer = tokenization.BailianTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case
    )

    train_f, label2idx = get_data(
        train,
        tokenizer,
        label2idx,
        max_seq_len=max_seq_len
    )
    train_dl = DataLoaderForTrain(
        train_f,
        batch_size=batch_size,
        shuffle=True,
        cuda=cuda
    )
    valid_f, label2idx = get_data(
        valid,
        tokenizer,
        label2idx,
        max_seq_len=max_seq_len,
    )
    valid_dl = DataLoaderForTrain(
        valid_f,
        batch_size=batch_size,
        cuda=cuda,
        shuffle=False
    )
    return train_dl, valid_dl, tokenizer, label2idx, max_seq_len


def get_bert_data_loader_for_predict(path, learner):
    df = pd.read_csv(path, delimiter=DELIMITER)
    f, _ = get_data(
        df,
        tokenizer=learner.data.tokenizer,
        label2idx=learner.data.label2idx,
        max_seq_len=learner.data.max_seq_len,
    )
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=True)

    return dl


def text_array_for_predict(input_text_arr: list, learner):
    tokenizer = learner.data.tokenizer

    marker = []

    format_text_arr = []

    for index, input_text in enumerate(input_text_arr):
        sents, sent_marker = tokenizer.tokenize_for_predict(
            input_text,
            max_seq_len=learner.data.max_seq_len - 2
        )
        format_text_arr.extend(sents)
        marker.append((len(sents), sent_marker))

    df = pd.DataFrame(format_text_arr, columns=["0", "1"])

    f, _ = get_data(
        df,
        tokenizer=learner.data.tokenizer,
        label2idx=learner.data.label2idx,
        max_seq_len=learner.data.max_seq_len,

    )

    cuda = learner.model.use_cuda
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=cuda)

    preds = learner.predict(dl)
    from ..utils.utils import bert_labels2tokens, tokens2spans
    tokens, labels = bert_labels2tokens(dl, preds)
    span_preds = tokens2spans(tokens, labels)

    pointer = 0
    results = []
    for idx, (size, sen_marker) in enumerate(marker):
        input_text = input_text_arr[idx]
        import itertools
        ed = pointer + size
        pred_labels = list(itertools.chain(*span_preds[pointer:ed]))
        pointer = ed

        result = tokenizer.recover_text(
            input_text,
            dl.dataset[idx].bert_tokens,
            pred_labels,
            marker=sent_marker
        )

        results.append(result)

    return results


class BertData(object):

    @property
    def config(self):
        config = {
            "train_path": self.train_path,
            "valid_path": self.valid_path,
            "vocab_file": self.vocab_file,
            "data_type": self.data_type,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "cuda": self.cuda,
            "label2idx": self.label2idx,
        }
        return config

    def __init__(
            self,
            train_path,
            valid_path,
            vocab_file,
            data_type,
            train_dl=None,
            valid_dl=None,
            tokenizer=None,
            label2idx=None,
            max_seq_len=424,
            batch_size=16,
            cuda=True,
    ):

        self.train_path = train_path
        self.valid_path = valid_path
        self.data_type = data_type
        self.vocab_file = vocab_file
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.batch_size = batch_size
        self.cuda = cuda and torch.cuda.is_available()
        self.id2label = sorted(label2idx.keys(), key=lambda x: label2idx[x])
        self.max_seq_len = max_seq_len

    def reload_dl(self, path, for_train=True):
        '''
        重新加载数据集
        :param path:
        :param for_train: 是否为了训练
        :return:
        '''

        df = pd.read_csv(path, delimiter=DELIMITER)
        features, label2idx = get_data(
            df,
            tokenizer=self.tokenizer,
            label2idx=self.label2idx,
            max_seq_len=self.max_seq_len,
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
            fn_res = fn(
                config["train_path"],
                config["valid_path"],
                config["vocab_file"],
                config["batch_size"],
                config["cuda"],
                do_lower_case,
                config["max_seq_len"],
                label2idx=config["label2idx"]
            )
        else:
            fn_res = (
                None,
                None,
                tokenization.BailianTokenizer(
                    vocab_file=config["vocab_file"],
                    do_lower_case=do_lower_case
                ),
                config["label2idx"],
                config["max_seq_len"]
            )
        return cls(
            config["train_path"],
            config["valid_path"],
            config["vocab_file"],
            config["data_type"],
            *fn_res,
            batch_size=config["batch_size"],
            cuda=config["cuda"],
        )

        # with open(config_path, "w") as f:
        #    json.dump(config, f)

    @classmethod
    def create(
            cls,
            train_path,
            valid_path,
            vocab_file,
            batch_size=16,
            cuda=True,
            data_type="bert_uncased",
            max_seq_len=424,
    ):
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
            *fn(
                train_path,
                valid_path,
                vocab_file,
                batch_size,
                cuda,
                do_lower_case,
                max_seq_len,
            ),
            batch_size=batch_size,
            cuda=cuda
        )
