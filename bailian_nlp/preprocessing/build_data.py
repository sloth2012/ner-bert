# coding: utf8
import os

'''
构造一些较为奇葩的样本数据，形如:
1、  美年大健康产业（集团）有限公司美年大健康产业（集团）有限公司美年大健康产业（集团）有限公司
2.  雷军马化腾李彦宏
3.  马化腾深圳
4.  深圳马化腾
...
以及上述变种的组合
'''


def build_fake_data():
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


if __name__ == '__main__':
    build_fake_data()


