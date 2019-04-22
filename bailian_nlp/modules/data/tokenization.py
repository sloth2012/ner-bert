# coding=utf-8
from pytorch_pretrained_bert import tokenization
import unicodedata
import six
from ..settings import UNKNOWN_TEXT_LABEL
import re


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


class BailianTokenizer(object):
    def __init__(
            self,
            vocab_file=None,
            unk_token=UNKNOWN_TEXT_LABEL,
            do_lower_case=True,
            max_input_chars_per_word=200
    ):
        if vocab_file is None:
            from ...released.settings import CHINESE_BERT_VOCAB_FILE
            vocab_file = CHINESE_BERT_VOCAB_FILE
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.pos_pattern = re.compile(r'(.+?)/(?:([a-z]{1,2})(?:$| ))')

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def basic_tokenize(self, text, marker=[], index=0):
        token = ''

        output_tokens = []
        last_idx = index
        for idx, char in enumerate(text):
            if tokenization._is_punctuation(char):
                if token != '':
                    sub_output_tokens = self.wordpiece_tokenize(token, marker, index=last_idx)
                    output_tokens.extend(sub_output_tokens)
                sub_output_tokens = self.wordpiece_tokenize(char, marker, index=index + idx)
                output_tokens.extend(sub_output_tokens)
                last_idx = index + idx + 1
                token = ''
            else:
                token += char
        if token != '':
            sub_output_tokens = self.wordpiece_tokenize(token, marker, last_idx)
            output_tokens.extend(sub_output_tokens)

        return output_tokens

    def wordpiece_tokenize(self, text, marker=[], index=0):
        text = convert_to_unicode(text)
        output_tokens = []

        token = ''
        last_idx = index
        for idx, char in enumerate(text):
            if tokenization._is_whitespace(char):
                if token != '':
                    sub_output_tokens = self.piece_token(token, marker, index=last_idx)
                    output_tokens.extend(sub_output_tokens)
                    token = ''

                marker.append((
                    0, (index + idx, index + idx + 1)
                ))
                last_idx = index + idx + 1
            else:
                token += char

        if token != '':
            sub_output_tokens = self.piece_token(token, marker, index=last_idx)
            output_tokens.extend(sub_output_tokens)

        return output_tokens

    def piece_token(self, token, marker=[], index=0):

        output_tokens = []
        idx = index
        if len(token) > self.max_input_chars_per_word:
            # 这里转换为成未知字符时，长度可能溢出，也可能变短
            marker.append((
                1, (idx, idx + len(token))
            ))
            output_tokens.append(self.unk_token)

        else:
            is_bad = False
            start = 0
            sub_tokens = []
            prefix = '##'

            while start < len(token):
                end = len(token)
                cur_substr = None
                while start < end:
                    substr = "".join(token[start:end])
                    if start > 0:
                        substr = prefix + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # 这里转换为成未知字符时，长度可能溢出，也可能变短
                marker.append((
                    1, (idx, idx + len(token))
                ))
                output_tokens.append(self.unk_token)

            else:
                last_idx = idx
                for sub_idx, sub_token in enumerate(sub_tokens):
                    sub_token_size = len(sub_token)
                    if sub_idx != 0:
                        sub_token_size -= len(prefix)

                    idx_ed = last_idx + sub_token_size
                    marker.append((
                        1, (last_idx, idx_ed)
                    ))
                    last_idx = idx_ed

                    output_tokens.append(sub_token)

        return output_tokens

    def tokenize(self, text):
        split_tokens = []
        marker = []

        text = convert_to_unicode(text)

        token = ''

        last_idx = 0

        # 针对不能处理中文引号的问题，这里替换成英文符号
        replace_chars_mapping = {
            '“': '"',
            '”': '"',
            '‘': '\'',
            '’': '\''
        }

        for idx, char in enumerate(text):
            char = replace_chars_mapping.get(char, char)
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or tokenization._is_control(char):
                # 一些特殊字符去除，这种表示可能出现在marker中的某个区间内
                marker.append((
                    -1, (idx, idx + 1)
                ))
                continue

            if tokenization._is_whitespace(char):
                if token != '':
                    if self.do_lower_case:
                        token = token.lower()

                    sub_split_tokens = self.basic_tokenize(token, marker, index=last_idx)
                    split_tokens.extend(sub_split_tokens)
                    token = ''

                marker.append((
                    0, (idx, idx + 1)
                ))

                last_idx = idx + 1

            else:
                format_char = unicodedata.normalize("NFD", char)
                if len(format_char) > len(char) and unicodedata.category(format_char[1]) == 'Mn':
                    char = format_char[0]
                    cp = ord(char)
                elif unicodedata.category(char) == 'Mn':
                    # 表示这种特殊元音去除的情况，也可能出现在后面的某个区间内
                    marker.append((
                        -2, (idx, idx + 1)
                    ))
                    continue

                if self._is_chinese_char(cp):
                    if token != '':
                        if self.do_lower_case:
                            token = token.lower()

                        sub_split_tokens = self.basic_tokenize(token, marker, index=last_idx)
                        split_tokens.extend(sub_split_tokens)

                    sub_split_tokens = self.basic_tokenize(char, marker, index=idx)
                    split_tokens.extend(sub_split_tokens)
                    token = ''
                    last_idx = idx + 1
                else:
                    token += char

        if token != '':
            sub_split_tokens = self.basic_tokenize(token, marker, index=last_idx)
            split_tokens.extend(sub_split_tokens)

        return split_tokens, marker

    # 用于恢复粗粒度的文本
    @staticmethod
    def recover_text_striped(text, tokens, labels, marker, default_label='w'):
        assert len(tokens) == len(labels) and len(marker) >= len(tokens)

        # 有效token指针
        pointer = 0
        cache_marker = []

        # 最终结果
        span_tokens = []
        span_labels = []

        # 中间变量
        cache_tokens = []
        strip_offset = 0

        cache_label = None

        last_ed = -1
        for idx, (tp, (st, ed)) in enumerate(marker):
            if tp == 1:
                if last_ed != st:
                    strip_offset = 0

                temp_offset = strip_offset
                current_prefix, current_label = labels[pointer].split('_')

                if (current_label == cache_label and current_prefix in ['I', 'E']) or cache_label is None:
                    do_append = True
                else:
                    do_append = False
                    # 说明为新的开始，这时应该将已有的处理下
                    span_tokens.append(''.join(cache_tokens))
                    span_labels.append(cache_label)
                    cache_tokens = []

                cache_remover = []
                accent_counter = 0
                # 先找到有效原token
                for cache_tp, cache_st, cache_ed in cache_marker:

                    # 这种情况一般发生在tp=1第一个位置
                    if cache_ed <= st + strip_offset \
                            or (cache_st == st + strip_offset and cache_ed <= ed + temp_offset):
                        token = text[cache_st:cache_ed]
                        # 这种一般都是tp为0,空格、控制符之类的，所以为绝对位置索引，无需考虑间断的strip数量
                        if cache_tp == -2 and cache_st == st + strip_offset:
                            accent_counter += 1
                            temp_offset += cache_ed - cache_st

                        if not do_append:
                            # 表示为元音字，需要补充在上一个token后面以供还原
                            if cache_tp == -2 and len(span_tokens) > 0:
                                span_tokens[-1] += token
                            else:
                                span_tokens.append(token)
                                span_labels.append(default_label)
                        else:
                            if cache_label is None:
                                span_tokens.append(token)
                                span_labels.append(default_label)
                            else:
                                cache_tokens.append(token)

                        cache_remover.append((cache_tp, cache_st, cache_ed))
                    elif cache_st >= st + strip_offset and cache_ed <= ed + temp_offset:
                        temp_offset += cache_ed - cache_st

                        cache_remover.append((cache_tp, cache_st, cache_ed))
                    else:
                        break

                [cache_marker.remove(c) for c in cache_remover]
                token = text[st + strip_offset + accent_counter: ed + temp_offset]

                # print('marker', pointer)
                # print(idx, st, ed, strip_offset, temp_offset,  accent_counter, token)
                # print('**************************')

                cache_tokens.append(token)

                cache_label = current_label
                pointer += 1
                strip_offset = temp_offset

            else:
                cache_marker.append((tp, st, ed))

                # 目前trip的marker位置都是绝对位置，无需添加offset
                strip_offset = 0

            last_ed = ed

        if len(cache_marker) != 0:
            for cache_tp, cache_st, cache_ed in cache_marker:
                token = text[cache_st:cache_ed]
                if cache_tp == -2 and len(cache_tokens) > 0:
                    cache_tokens[-1] += token
                else:
                    if len(cache_tokens) != 0:
                        span_tokens.append(''.join(cache_tokens))
                        span_labels.append(cache_label)
                        cache_tokens = []
                    span_tokens.append(token)
                    span_labels.append(default_label)

        if len(cache_tokens) != 0:
            span_tokens.append(''.join(cache_tokens))
            span_labels.append(cache_label)

        return span_tokens, span_labels

    # 用于从词性文本构造符合格式的训练数据，用于训练阶段的数据处理
    def tokenize_with_pos_text(self, pos_sent):
        sent = ''
        pos_marker = []
        pointer = 0

        for word, flag in self.pos_pattern.findall(pos_sent):
            # print(repr(word), repr(flag), len(word))
            sent += word
            pos_marker.extend([(flag, pointer)] * len(word))
            pointer += 1

        tokens, marker = self.tokenize(sent)
        cache_marker = []
        cache_labels = []

        strip_offset = 0
        last_ed = -1
        for tp, (st, ed) in marker:
            if tp == 1:
                if last_ed != st:
                    strip_offset = 0

                temp_offset = 0
                cache_remover = []
                accent_counter = 0
                for cache_tp, cache_st, cache_ed in cache_marker:
                    if cache_st == st + strip_offset and cache_ed <= ed + temp_offset:
                        if cache_tp == -2:
                            accent_counter += 1
                            temp_offset += cache_ed - cache_st
                        cache_remover.append((cache_tp, cache_st, cache_ed))
                    elif cache_st >= st + strip_offset and cache_ed <= ed + temp_offset:
                        temp_offset += cache_ed - cache_st
                        cache_remover.append((cache_tp, cache_st, cache_ed))
                    else:
                        break

                [cache_marker.remove(c) for c in cache_remover]
                label = pos_marker[st + strip_offset + accent_counter:ed + temp_offset]
                cache_labels.append(label)

                strip_offset = temp_offset

            else:
                cache_marker.append((tp, st, ed))
                strip_offset = 0

            last_ed = ed

        labels = []
        cache_counter = 0
        last_cache_label = (None, None)

        for token, cache_label in zip(tokens, cache_labels):
            current_label = None
            for idx, lab_pos in enumerate(cache_label):
                if current_label is None:
                    current_label = lab_pos

                if current_label != lab_pos:
                    raise Exception('原有分词与现有tokenizer有冲突')

            if last_cache_label is None or last_cache_label == current_label:
                cache_counter += 1
            else:
                size = cache_counter
                if size == 1:
                    label = last_cache_label[0]
                    labels.append(f'S_{label}')
                elif size > 1:
                    label = last_cache_label[0]
                    labels.extend(
                        [f'B_{label}']
                        + [f'I_{label}'] * (size - 2)
                        + [f'E_{label}']
                    )

                cache_counter = 1

            last_cache_label = current_label

        size = cache_counter
        if size == 1:
            label = last_cache_label[0]
            labels.append(f'S_{label}')
        elif size > 1:
            label = last_cache_label[0]
            labels.extend(
                [f'B_{label}']
                + [f'I_{label}'] * (size - 2)
                + [f'E_{label}']
            )

        assert len(tokens) == len(labels)

        # for token, label in zip(tokens, labels):
        #     print(token, label)

        return tokens, labels

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    # 预测专用，输出假label
    def tokenize_for_predict(self, text, max_seq_len=128):
        tokens, marker = self.tokenize(text)

        last_punctuation_idx = -1

        results = []
        cache = []
        for idx, token in enumerate(tokens):
            cache.append(token)
            if len(token) == 1 and tokenization._is_punctuation(token):
                last_punctuation_idx = len(cache)

            # 训练的max_seq_len需要比这个加1
            if len(cache) > max_seq_len - 1:
                if last_punctuation_idx != -1:
                    target = cache[:last_punctuation_idx]
                    cache = cache[last_punctuation_idx:]
                    last_punctuation_idx = -1
                else:
                    target = cache
                    cache = []
                    last_punctuation_idx = -1

                results.append([
                    ' '.join(target),
                    ' '.join(['O'] * len(target))
                ])

        if len(cache) != 0:
            target = cache
            results.append([
                ' '.join(target),
                ' '.join(['O'] * len(target))
            ])

        return results, marker
