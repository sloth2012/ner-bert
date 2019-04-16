# coding: utf8
from bailian_nlp.web.utils.common import timer


@timer
def segment_test():
    import os
    root_dir, _ = os.path.split(os.path.realpath(__file__))
    for i in range(2):
        root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, 'datadrive/icwb2-data')
    test_file = os.path.join(data_dir, 'testing/pku_test.utf8')

    out_file = os.path.join(data_dir, 'testing/pku_out.utf8')
    with open(test_file) as fin, open(out_file, 'w') as fout:

        sents = []
        for line in fin:
            line = line.strip()
            if not line:
                continue

            sents.append(line)

        from bailian_nlp.released.pos import PosTagger

        tagger = PosTagger()

        results = tagger.lexerCustom(sents)

        for result in results:
            words = []
            for word, flag in result:
                words.append(word)

            fout.write('  '.join(words) + '\n')


if __name__ == '__main__':
    segment_test()
