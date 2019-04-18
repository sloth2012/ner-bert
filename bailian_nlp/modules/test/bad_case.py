# coding: utf8


def pos_check():
    import os
    root_dir, _ = os.path.split(os.path.realpath(__file__))
    for i in range(2):
        root_dir = os.path.dirname(root_dir)
    data_dir = os.path.join(root_dir, 'datadrive/bailian')
    test_file = os.path.join(data_dir, 'pos_bad_case')

    with open(test_file) as fin:

        sents = []
        for line in fin:
            line = line.strip()
            if not line:
                continue

            sents.append(line)

        from bailian_nlp.released.pos import PosTagger

        tagger = PosTagger()

        results = tagger.lexerCustom(sents)

        for pos_str in tagger.lexer2str(results):
            print(pos_str)


if __name__ == '__main__':
    pos_check()
