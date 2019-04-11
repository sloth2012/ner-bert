# coding: utf8
from . import settings
from . import dictionary
from collections import defaultdict

_DICTIONARY = dictionary.Dictionary()


class PosTagger:
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.learner = None
        self.init_env()

    def cut(self, text, ignore=False):
        '''

        :param text: list或单个字符串
        :param ignore:
        :return:
        '''
        from modules.data.bert_data import text_array_for_predict

        # TODO 接收list类型输入
        text = self._check_input(text, ignore)
        res = text_array_for_predict(text, learner=self.learner)

        return res

    def init_env(self, for_train=False):
        from modules.train import train
        if self.config_file is not None:
            self.learner = train.NerLearner.from_config(self.config_file, for_train=for_train)
            self.learner.load_model()

        else:
            mapping = {
                'train_path': settings.DEFAULT_POS_TRAIN_FILE,
                'valid_path': settings.DEFAULT_POS_VALID_FILE,
                'vocab_file': settings.CHINESE_BERT_VOCAB_FILE,
                'bert_config_file': settings.CHINESE_BERT_MODEL_CONFIG_FILE,
                'init_checkpoint_pt': settings.CHINESE_BERT_MODEL_FILE,
                'best_model_path': settings.DEFAULT_POS_MODEL_FILE
            }

            from .utils import valid_config
            config = valid_config(settings.DEFAULT_POS_MODEL_CONFIG_FILE, mapping)

            self.learner = train.NerLearner.from_config(config, for_train=for_train)
            self.learner.load_model()

    @staticmethod
    def _check_input(text, ignore=False):
        if not text:
            return []

        if not isinstance(text, list):
            text = [text]

        null_index = [i for i, t in enumerate(text) if not t]
        if null_index and not ignore:
            raise Exception("null text in input ")

        return text

    @staticmethod
    def load_userdict(path):
        _DICTIONARY.add_dict(path)

    @staticmethod
    def delete_userdict():
        _DICTIONARY.delete_dict()

    def lexerCustom(self, text):
        all_words = self.cut(text)

        pos_words = []
        if _DICTIONARY.sizes != 0:
            for sent, words in zip(text, all_words):
                words = self._merge_user_words(sent, words)
                pos_words.append(words)
        else:
            pos_words = all_words
        return pos_words


    @staticmethod
    def _merge_user_words(text, seg_results):
        if not _DICTIONARY:
            return seg_results

        matchs = _DICTIONARY.parse_words(text)
        graph = defaultdict(lambda: defaultdict(tuple))

        text_len = len(text)

        for i in range(text_len):
            graph[i][i + 1] = (1.0, dictionary._UNKNOWN_LABEL)

        index = 0

        for w, p in seg_results:
            w_len = len(w)
            graph[index][index + w_len] = (_DICTIONARY.get_weight(w) + w_len, p)
            index += w_len

        for m in matchs:
            graph[m.start][m.end] = (
                _DICTIONARY.get_weight(m.keyword) * len(m.keyword),
                _DICTIONARY.get_label(m.keyword)
            )

        route = {}
        route[text_len] = (0, 0, dictionary._UNKNOWN_LABEL)

        for idx in range(text_len - 1, -1, -1):
            m = [((graph.get(idx).get(k)[0] + route[k][0]), k, graph.get(idx).get(k)[1]) for k in graph.get(idx).keys()]
            mm = max(m)
            route[idx] = mm

        index = 0
        path = [index]
        pos_words = []

        while index < text_len:
            ind_y = route[index][1]
            path.append(ind_y)
            word = text[index:ind_y]
            label = route[index][2]
            pos_words.append((word, label))
            index = ind_y

        return pos_words
