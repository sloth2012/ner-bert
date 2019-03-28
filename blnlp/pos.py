# coding: utf8
from . import settings


class PosTagger:
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.learner = None
        self.init_env()

    def cut(self, text):
        if not text:
            raise Exception('Empty Input Text!')
        from modules.data.bert_data import single_example_for_predict

        res = single_example_for_predict(text, learner=self.learner)

        return res

    def init_env(self):
        from modules.models import released_models
        if self.config_file is not None:
            self.learner = released_models.recover_from_config_file(self.config_file, for_train=False)
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

            from . import utils
            config = utils.valid_config(settings.DEFAULT_POS_MODEL_CONFIG_FILE, mapping)

            self.learner = released_models.recover_from_config(config, for_train=False)
            self.learner.load_model()

