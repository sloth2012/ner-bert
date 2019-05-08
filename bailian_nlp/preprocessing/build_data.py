# coding: utf8
import os
from bailian_nlp.modules.settings import DELIMITER


# TODO 整理baidu分词中的url，针对其进行更改
def build_pos_fake_data():
    '''
    构造一些较为奇葩的样本数据，形如:
    1、  美年大健康产业（集团）有限公司美年大健康产业（集团）有限公司美年大健康产业（集团）有限公司
    2.  雷军马化腾李彦宏
    3.  马化腾深圳
    4.  深圳马化腾
    ...
    以及上述变种的组合
    '''
    root_dir, _ = os.path.split(os.path.realpath(__file__))
    data_dir = os.path.join(os.path.dirname(root_dir), 'datadrive/bailian')

    materials_dir = os.path.join(data_dir, 'materials')

    city_path = os.path.join(materials_dir, 'city.txt')
    addr_path = os.path.join(materials_dir, 'location.dic')
    org_path = os.path.join(materials_dir, 'org.txt')
    per_path = os.path.join(materials_dir, 'han_names.utf8')
    url_path = os.path.join(materials_dir, 'url.txt')
    single_path = os.path.join(data_dir, 'single.txt')

    out_path = os.path.join(data_dir, 'fake.txt')

    from collections import defaultdict

    materials = defaultdict(set)
    with open(city_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            materials['ns'].add(line)

    with open(addr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            materials['ns'].add(line)

    with open(org_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            materials['nt'].add(line)

    with open(per_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            materials['nr'].add(line)

    with open(url_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            materials['xu'].add(line)

    with open(single_path) as f:
        import re

        p = re.compile(r'(.+?)/(?:([a-z]{1,2})(?:$| ))')

        for line in f:
            line = line.strip()
            if not line:
                continue
            for word, label in p.findall(line):
                materials[label].add(word)

    for k, v in materials.items():
        materials[k] = list(v)

    import random

    fake_num = 200000

    with open(out_path, 'w') as f:

        for i in range(fake_num):
            score = random.random()

            if score <= 0.3:
                # 同词性相同词多次重复
                key = random.choice(list(materials.keys()))

                word = random.choice(materials[key])
                if word == '[UNK]':
                    continue

                text = ' '.join(random.randint(1, 4) * [word + '/' + key])

            elif score <= 0.6:
                # 同词性不同词多次重复
                key = random.choice(list(materials.keys()))
                words = [
                    random.choice(materials[key])
                    for _ in range(random.randint(1, 4))
                ]

                text = ' '.join([
                    word + '/' + key
                    for word in words
                    if word != '[UNK]'
                ])
            elif score <= 0.9:
                # 不同词性词组合
                words = []

                for _ in range(random.randint(1, 6)):
                    key = random.choice(list(materials.keys()))
                    word = random.choice(materials[key])
                    if word == '[UNK]':
                        continue
                    words.append(
                        word + '/' + key
                    )

                if len(words) == 0:
                    continue

                text = ' '.join(words)
            else:
                # 直接输出
                key = random.choice(list(materials.keys()))
                word = random.choice(materials[key])
                if word == '[UNK]':
                    continue

                text = word + '/' + key

            if text == '':
                continue

            f.write(text)
            f.write('\n')


def build_pos_train_and_valid_data(vocab_file=None):
    '''
    构造词性标注模型的训练和验证数据，根据百度分词结果（自定义词库）和上面的fake数据来组织进行。
    :return:
    '''

    root_dir, _ = os.path.split(os.path.realpath(__file__))
    data_dir = os.path.join(os.path.dirname(root_dir), 'datadrive/bailian')

    seg_file = os.path.join(data_dir, 'final_baidu-23w.txt')
    fake_file = os.path.join(data_dir, 'fake.txt')
    special_file = os.path.join(data_dir, 'special.txt')
    dict_file = os.path.join(data_dir, 'single.txt')

    train_path = os.path.join(data_dir, 'pos/train.csv')
    valid_path = os.path.join(data_dir, 'pos/valid.csv')
    test_path = os.path.join(data_dir, 'pos/test.csv')

    with open(seg_file) as fin1, \
            open(fake_file) as fin2, \
            open(special_file) as fin3, \
            open(dict_file) as fin4, \
            open(train_path, 'w') as train_f, \
            open(valid_path, 'w') as valid_f, \
            open(test_path, 'w') as test_f:

        train_f.write(f'0{DELIMITER}1\n')
        valid_f.write(f'0{DELIMITER}1\n')
        test_f.write(f'0{DELIMITER}1\n')

        fins = [fin1, fin2, fin3, fin4]

        from bailian_nlp.modules.data.tokenization import BailianTokenizer
        tokenizer = BailianTokenizer(vocab_file=vocab_file)

        total_counter = 0
        error_counter = 0
        for k, fin in enumerate(fins):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total_counter += 1
                import random
                score = random.random()

                if k < 2:
                    if score > 0.01:
                        fout = train_f
                    elif score > 0.004:
                        fout = valid_f
                    else:
                        fout = test_f
                else:
                    fout = train_f

                try:
                    tokens, labels = tokenizer.tokenize_with_pos_text(line)
                    fout.write(DELIMITER.join([
                        ' '.join(labels),
                        ' '.join(tokens)
                    ]))
                    fout.write('\n')
                except KeyboardInterrupt:
                    raise
                except:
                    error_counter += 1
                    if total_counter % 100 == 0:
                        print(error_counter, total_counter, error_counter / total_counter)
                    continue

        print(error_counter, total_counter, error_counter / total_counter)


def build_pos_single_data_from_hanlp_dict():
    '''
    通过hanlp的一些词典构造语料，主要用于训练单个单词的样本
    :return:
    '''

    root_dir, _ = os.path.split(os.path.realpath(__file__))
    data_dir = os.path.join(os.path.dirname(root_dir), 'datadrive/bailian')

    single_file = os.path.join(data_dir, 'single.txt')

    hanlp_dict_dir = '/Users/lx/Programs/research/HanLP/data/dictionary'

    custom_file = os.path.join(hanlp_dict_dir, 'custom/CustomDictionary.txt')
    nt_file = os.path.join(hanlp_dict_dir, 'custom/机构名词典.txt')
    additional_file = os.path.join(hanlp_dict_dir, 'custom/现代汉语补充词库.txt')
    core_file = os.path.join(hanlp_dict_dir, 'CoreNatureDictionary.txt')

    # hanlp的词性转成目前百度相对于的词性，可能会自定义一些。暂时只列一些专门的映射
    pos_mapping = {
        'n': 'n',  # 名词
        'f': 'f',  # 方位名词
        's': 's',  # 处所名词
        't': 't',  # 时间名词
        'nr': 'nr',  # 人名
        'ns': 'ns',  # 地名
        'nt': 'nt',  # 机构团体名
        'nw': 'nw',  # 作品名，暂无此映射
        'nz': 'nz',  # 其它专名
        'v': 'v',  # 动词
        'vd': 'vd',  # 动副词
        'vn': 'vn',  # 动名词
        'a': 'a',  # 形容词
        'ad': 'ad',  # 副形词,
        'an': 'an',  # 名形词,
        'd': 'd',  # 副词
        'm': 'm',  # 数词
        'q': 'q',  # 量词
        'r': 'r',  # 代词
        'p': 'p',  # 介词
        'c': 'c',  # 连词
        'u': 'u',  # 助词
        'xc': 'xc',  # 其它虚词, 无此映射
        'w': 'w',  # 标点符号

        # 下边为hanlp独有的
        'i': 'i',  # 成语
        'l': 'l',  # 习用语
        'xu': 'xu',  # 网址
        'j': 'j',  # 简称略语
        'nnt': 'ti',  # 职务职称,
        'nrf': 'nr',  # 音译人名
        'nrj': 'nr',  # 日语人名
        'ntc': 'nt',  # 公司名
        'ntcb': 'nt',  # 银行
        'ntcf': 'nt',  # 工厂
        'ntch': 'nt',  # 酒店宾馆
        'nth': 'nt',  # 医院
        'nto': 'nt',  # 政府机构
        'nts': 'nt',  # 中小学
        'nxu': 'nt',  # 大学
        'nit': 'nt',  # 教育相关机构,
        'nm': 'n',  # 物品名
        'nmc': 'n',  # 化学用品
        'nnd': 'n',  # 职业
        'ni': 'n',  # 机构相关
        'nh': 'n',  # 医疗疾病相关名词
        'nhd': 'n',  # 疾病
        'nhm': 'n',  # 药品
        'nic': 'n',  # 下属机构
        'nf': 'n',  # 食品
        'nba': 'n',  # 动物名
        'nb': 'n',  # 生物名
        'nbp': 'n',  # 植物名

    }

    with open(single_file, 'w') as fout, \
            open(custom_file) as fin1, \
            open(nt_file) as fin2, \
            open(additional_file) as fin3, \
            open(core_file) as fin4:

        for fin in [
            fin1,
            fin2,
            fin3,
            fin4
        ]:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                linelist = line.split()
                length = len(linelist)
                if length % 2 == 0:
                    continue

                length = length // 2
                for i in range(length):
                    flag = pos_mapping.get(linelist[i * 2 + 1])

                    word = linelist[0].strip()

                    if not word or not flag:
                        continue

                    from pytorch_pretrained_bert.tokenization import _is_control
                    if len(word) == 1 and _is_control(word):
                        break
                    fout.write(f'{word}/{flag}\n')


if __name__ == '__main__':
    # build_pos_fake_data()

    # build_pos_single_data_from_hanlp_dict()

    build_pos_train_and_valid_data()
