# coding=utf-8
from pytorch_pretrained_bert import tokenization
import unicodedata
import six
from ..settings import UNKNOWN_CHAR


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
            vocab_file,
            unk_token=UNKNOWN_CHAR,
            do_lower_case=True,
            max_input_chars_per_word=200
    ):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

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
            if len(self.unk_token) >= len(token):
                marker.append((
                    1, (idx, idx + len(token))
                ))
            else:
                marker.append((
                    1, (idx, idx + len(self.unk_token))
                ))
                marker.append((
                    0, (idx + len(self.unk_token), idx + len(token))
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
                if len(self.unk_token) >= len(token):
                    marker.append((
                        1, (idx, idx + len(token))
                    ))
                else:
                    marker.append((
                        1, (idx, idx + len(self.unk_token))
                    ))
                    marker.append((
                        0, (idx + len(self.unk_token), idx + len(token))
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
        for idx, char in enumerate(text):
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
                        -1, (idx, idx + 1)
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
