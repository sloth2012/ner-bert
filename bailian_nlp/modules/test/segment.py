# coding: utf8
from bailian_nlp.web.utils.common import timer


@timer
def segment_test(sample=False):
    import os
    root_dir, _ = os.path.split(os.path.realpath(__file__))
    for i in range(2):
        root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, 'datadrive/icwb2-data')
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    raw_data_file = os.path.join(root_dir, 'datadrive/bailian/final_baidu-23w.txt')
    vocab_file = os.path.join(test_dir, 'vocab.txt')
    truth_file = os.path.join(test_dir, 'truth.txt')
    test_file = os.path.join(test_dir, 'test.txt')
    score_result_file = os.path.join(test_dir, 'score.txt')

    score_file = os.path.join(data_dir, 'scripts/score')

    # 若重新进行采样
    if sample:
        vocab = set()

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

        import re

        p = re.compile(r'(.+?)/(?:([a-z]{1,2})(?:$| ))')

        with open(raw_data_file) as fin, \
                open(truth_file, 'w') as truth_f, \
                open(test_file, 'w') as test_f, \
                open(vocab_file, 'w') as vocab_f:

            test_text_arr = []
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                import random
                score = random.random()

                if score > 0.006:
                    continue

                words = []
                for word, _ in p.findall(line):
                    from pytorch_pretrained_bert.tokenization import _is_control
                    char_list = [
                        c for c in list(word)
                        if not (c in replace_chars or c.isspace() or _is_control(c))
                    ]

                    char_size = len(char_list)

                    if char_size == 0:
                        continue

                    words.append(word)
                    vocab.add(word)

                if len(words) != 0:
                    truth_f.write(' '.join(words) + '\n')
                    test_text_arr.append(''.join(words))

            from bailian_nlp.released.pos import PosTagger
            tagger = PosTagger()

            results = tagger.lexerCustom(test_text_arr)

            for result in results:
                words = []
                for word, _ in result:
                    words.append(word)

                test_f.write('  '.join(words) + '\n')

            for word in vocab:
                vocab_f.write(word + '\n')

    cmd = f'{score_file} {vocab_file} {truth_file} {test_file}'

    from subprocess import Popen, PIPE
    process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf8', errors='ignore')
    stdout, stderr = process.communicate()
    print(stdout)
    with open(score_result_file, 'w') as f:
        f.write(stdout)


if __name__ == '__main__':
    segment_test(sample=True)
