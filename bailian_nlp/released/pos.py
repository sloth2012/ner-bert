# coding: utf8
from . import dictionary, settings
from collections import defaultdict
from ..web.utils.logger import getLogger
import os

_DICTIONARY = dictionary.Dictionary()

has_load_default = False


class PosTagger:
    def __init__(
            self,
            config_file=None
    ):
        global has_load_default
        self.config_file = config_file
        self.learner = None

        self.logger = getLogger(__name__)

        if not has_load_default:
            if os.path.exists(settings.DEFAULT_USER_DICT):
                self.logger.info(f'load default user_dict in {settings.DEFAULT_USER_DICT}')
                _DICTIONARY.add_dict(settings.DEFAULT_USER_DICT)
                has_load_default = True

        self.init_env()

    def cut(self, text, ignore=False, checked=False):
        '''

        :param text: list或单个字符串
        :param ignore:
        :param checked: 是否校验过
        :return:
        '''
        if not checked:
            text = self._check_input(text, ignore)

        from ..modules.data.data_loader import text_array_for_predict
        self.learner.data.tokenizer.tokenize(text)
        res = text_array_for_predict(text, learner=self.learner)

        return res

    def load_model(self, path):
        if self.learner is None:
            self.logger.error('please init pos model first!')
        else:
            self.logger.info(f'loadding model file in {path}')
            try:
                self.learner.load_model(path)
                self.logger.info('pos model loads success!')
            except Exception as e:
                self.logger.error('pos model loads fail!!')
                self.logger.error(e, exc_info=True)
                raise

    def init_env(self, for_train=False):
        from bailian_nlp.modules.train import train
        if self.config_file is not None:
            self.learner = train.NerLearner.from_config(self.config_file, for_train=for_train)
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

        import os
        if os.path.exists(self.learner.best_model_path):
            self.logger.info(f'found pos model file in {self.learner.best_model_path}')
            self.learner.load_model()
            self.logger.info('pos model loads success!')

        else:
            self.logger.warning(f'no model file found!')

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

    def lexerCustom(self, text, ignore=False):
        text = self._check_input(text, ignore)
        all_words = self.cut(text, checked=True)

        pos_words = []
        if _DICTIONARY.sizes != 0:
            for sent, words in zip(text, all_words):
                try:
                    words = self._merge_user_words(sent, words)
                    pos_words.append(words)
                except:
                    print('error text:', sent)
                    raise
        else:
            pos_words = all_words
        return pos_words

    def lexer2str(self, pos_sents):
        results = []
        for pos_sent in pos_sents:
            result = []
            for word, flag in pos_sent:
                result.append(f'{word}/{flag}')
            results.append(' '.join(result))

        return results

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

            # print(w, p, index, index + w_len)
            graph[index][index + w_len] = (_DICTIONARY.get_weight(w) + w_len, p)
            index += w_len

        for m in matchs:
            graph[m.start][m.end] = (
                _DICTIONARY.get_weight(m.keyword) * (len(m.keyword) + 1),  # 加1平滑下
                _DICTIONARY.get_label(m.keyword)
            )

        route = {}
        route[text_len] = (0, 0, dictionary._UNKNOWN_LABEL)

        for idx in range(text_len - 1, -1, -1):
            # print(graph)
            # print(idx, route)

            m = [((graph.get(idx).get(k)[0] + route[k][0]), k, graph.get(idx).get(k)[1]) for k in graph.get(idx).keys()]

            # print('**************')
            # for i, j, k in m:
            #     print(text[j - 1], i, k)

            mm = max(m)
            # print(idx, text[idx], mm)
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
