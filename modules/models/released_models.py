from modules.layers.encoders import *
from modules.layers.decoders import *


released_models = {
    "BertBiLSTMNCRF": {
        "encoder": BertBiLSTMEncoder,
        "decoder": NCRFDecoder
    },
    "BertBiLSTMAttnCRF": {
        "encoder": BertBiLSTMEncoder,
        "decoder": AttnCRFDecoder
    }
}


def recover_model_from_config(config: dict):
    # TODO 这里先只用bert
    from modules.models import bert_models

    if not isinstance(config, dict) or 'name' not in config:
        return config

    model_type = getattr(bert_models, config['name'])

    params = {
        k: recover_model_from_config(v)
        for k, v in config['params'].items()
    }

    if issubclass(model_type.__class__, bert_models.NerModel.__class__):
        return model_type(**params)
    else:
        return model_type.create(**params)


def recover_from_config(path, for_train=True):

    from modules.data import bert_data
    import json
    with open(path, "r") as file:
        config = json.load(file)

    data = bert_data.BertNerData.from_config(config["data"], for_train)
    model_config = config["model"]

    model = recover_model_from_config(model_config)

    from modules.train.train import NerLearner

    return NerLearner(model=model, data=data, **config["learner"])

