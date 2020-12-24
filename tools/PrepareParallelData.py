'''
Convert a fairseq vocab to a NiuTrans.NMT vocab
Help: python3 PrepareParallelData.py -h

Training data format (binary):
first 8 bit: number of sentence pairs
subsequent segements:
source sentence length (4 bit)
target sentence length (4 bit)
source tokens (4 bit per token)
target tokens (4 bit per token)
'''

from struct import pack
import argparse

# User defined words
PAD = 1
SOS = 2
EOS = 2
UNK = 3

# The maximum length for a sentence
MAX_SENT_LEN = 1024

parser = argparse.ArgumentParser(
    description='Prepare parallel data for nmt training')
parser.add_argument('-src', help='Source language file', type=str, default='')
parser.add_argument('-tgt', help='Target language file', type=str, default='')
parser.add_argument(
    '-src_vocab', help='Source language vocab file', type=str, default='')
parser.add_argument(
    '-tgt_vocab', help='Target language vocab file', type=str, default='')
parser.add_argument('-output', help='Training file', type=str, default='')
args = parser.parse_args()

src_vocab = dict()
tgt_vocab = dict()
cut_num = 0

def load_vocab(vocab, file):
    with open(file, 'r', encoding='utf8') as f:
        vocab_size = int(f.readline().split()[0])
        for l in f:
            l = l.split()
            vocab[l[0]] = int(l[1])
    print("{}: {} types".format(file, vocab_size))
    return vocab_size


def get_id(vocab, word, is_src=True):
    if word in vocab.keys():
        return vocab[word]
    else:
        return UNK


src_vocab_size = load_vocab(src_vocab, args.src_vocab)
tgt_vocab_size = load_vocab(tgt_vocab, args.tgt_vocab)
if (not isinstance(src_vocab_size, int)) or (src_vocab_size < 0):
    raise ValueError("Invalid source vocab size")
if (not isinstance(tgt_vocab_size, int)) or (src_vocab_size < 0):
    raise ValueError("Invalid source vocab size")


with open(args.src, 'r', encoding='utf8') as fs:
    with open(args.tgt, 'r', encoding='utf8') as ft:
        src_sentences, tgt_sentences = list(), list()
        for ls in fs:
            ls = ls.split()
            lt = ft.readline().split()
            if len(ls) >= MAX_SENT_LEN:
                cut_num += 1
                ls = ls[:MAX_SENT_LEN - 1]
            if len(lt) >= MAX_SENT_LEN:
                cut_num += 1
                lt = lt[:MAX_SENT_LEN - 1]
            src_sent = [get_id(src_vocab, w) for w in ls] + [EOS]
            tgt_sent = [SOS] + [get_id(tgt_vocab, w, False) for w in lt]

            src_sentences.append(src_sent)
            tgt_sentences.append(tgt_sent)

        src_tokens = sum([len(s) - 1 for s in src_sentences])
        tgt_tokens = sum([len(t) - 1 for t in tgt_sentences])
        print("{}: {} sents, {} tokens, {:.2f} replaced by <UNK>".format(
            args.src, len(src_sentences), src_tokens, sum([s.count(UNK) for s in src_sentences]) / src_tokens))
        print("{}: {} sents, {} tokens, {:.2f} replaced by <UNK>".format(
            args.tgt, len(tgt_sentences), tgt_tokens, sum([s.count(UNK) for s in tgt_sentences]) / tgt_tokens))

        with open(args.output, 'wb') as fo:
            # seg 1: source and target vocabulary size
            vocab_size = [src_vocab_size, tgt_vocab_size]
            vocab_size_pack = pack("i" * len(vocab_size), *vocab_size)
            fo.write(vocab_size_pack)

            # seg 2: number of sentence pairs (8 bit per number)
            sent_num = [len(src_sentences)]
            sent_num_pack = pack("Q", *sent_num)
            fo.write(sent_num_pack)

            for i in range(len(src_sentences)):
                src_sent = src_sentences[i]
                tgt_sent = tgt_sentences[i]

                # seg 3: number of source and target sentence length (4 bit per number)
                src_tgt_length = [len(src_sent), len(tgt_sent)]
                src_tgt_length_pack = pack(
                    "i" * len(src_tgt_length), *src_tgt_length)
                fo.write(src_tgt_length_pack)

                # seg 4: source sentence and target sentence pairs (4 bit per token)
                # print(src_sent)
                src_sent_pack = pack("i" * len(src_sent), *src_sent)
                fo.write(src_sent_pack)
                tgt_sent_pack = pack("i" * len(tgt_sent), *tgt_sent)
                fo.write(tgt_sent_pack)
