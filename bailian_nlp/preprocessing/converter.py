# coding: utf8

# 转换tf转化的ernie模型，详见：https://github.com/ArthurRizar/tensorflow_ernie
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch


def converter():
    import os
    base_dir = '/Users/lx/Downloads/baidu_ernie/'
    tf_checkpoint_path = os.path.join(base_dir, 'bert_model.ckpt')
    bert_config_file = os.path.join(base_dir, 'bert_config.json')

    root_dir, _ = os.path.split(os.path.realpath(__file__))
    dump_dir = os.path.join(os.path.dirname(root_dir), 'datadrive/models/ernie')
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir, exist_ok=True)

    dump_path = os.path.join(dump_dir, 'ernie_model.bin')

    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, dump_path)


if __name__ == '__main__':
    converter()


