# coding: utf8
import os


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
                text = ' '.join(random.randint(1, 4) * [random.choice(materials[key]) + '/' + key])

            elif score <= 0.6:
                # 同词性不同词多次重复
                key = random.choice(list(materials.keys()))
                text = ' '.join([
                    random.choice(materials[key]) + '/' + key
                    for _ in range(random.randint(1, 4))
                ])
            elif score <= 0.9:
                # 不同词性词组合
                words = []

                for _ in range(random.randint(1, 6)):
                    key = random.choice(list(materials.keys()))
                    words.append(
                        random.choice(materials[key]) + '/' + key
                    )

                text = ' '.join(words)
            else:
                # 直接输出
                key = random.choice(list(materials.keys()))
                text = random.choice(materials[key]) + '/' + key

            f.write(text)
            f.write('\n')


def build_pos_train_and_valid_data():
    '''
    构造词性标注模型的训练和验证数据，根据百度分词结果（自定义词库）和上面的fake数据来组织进行。
    :return:
    '''
    import re

    p = re.compile(r'(.+?)/(?:([a-z]{1,2})(?:$| ))')

    root_dir, _ = os.path.split(os.path.realpath(__file__))
    data_dir = os.path.join(os.path.dirname(root_dir), 'datadrive/bailian')

    seg_file = os.path.join(data_dir, 'final_baidu-23w.txt')
    fake_file = os.path.join(data_dir, 'fake.txt')

    train_path = os.path.join(data_dir, 'pos/train.csv')
    valid_path = os.path.join(data_dir, 'pos/valid.csv')

    delimiter = '△△△'

    # 注意这里的逻辑需要和modules.bert_data中预测预处理的的split_text函数中的一致
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
        '\u2005'
        ' '
    ]
    with open(seg_file) as fin1, \
            open(fake_file) as fin2, \
            open(train_path, 'w') as train_f, \
            open(valid_path, 'w') as valid_f:

        train_f.write(f'0{delimiter}1\n')
        valid_f.write(f'0{delimiter}1\n')

        fins = [fin1, fin2]
        for fin in fins:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                import random
                score = random.random()

                fout = train_f if score > 0.006 else valid_f
                words = []
                flags = []
                for word, flag in p.findall(line):
                    char_list = ['unk' if c in replace_chars or c.isspace() else c for c in list(word)]

                    char_size = len(char_list)
                    if char_size == 1:
                        # 一些错误的单个字符实体剔除掉
                        if flag in ['nt', 'ti', 'nr', 'ns', 'nz']:
                            flag = 'xx'
                        # 单个
                        tag_list = [f'S_{flag}']
                    else:
                        tag_list = [f'B_{flag}'] + [f'I_{flag}'] * (len(char_list) - 2) + [f'E_{flag}']

                    if char_size != len(tag_list):
                        print(line)
                        print(word, flag)
                        print(char_list, tag_list)

                    words.extend(char_list)
                    flags.extend(tag_list)

                assert len(words) == len(flags)

                fout.write(delimiter.join([
                    ' '.join(flags),
                    ' '.join(words)
                ]))
                fout.write('\n')


if __name__ == '__main__':
    build_pos_fake_data()
