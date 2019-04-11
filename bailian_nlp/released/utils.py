# coding: utf8


# 递归检测替换模型的加载参数
def config_recursion(config, mapping: dict):
    if not isinstance(config, dict):
        return config

    return {
        k: mapping.get(k, config_recursion(v, mapping))
        for k, v in config.items()
    }


# 检查模型参数，进行更正
def valid_config(conf_file, mapping: dict):
    with open(conf_file, 'r') as f:
        import json
        config = json.load(f)

    return config_recursion(config, mapping)
