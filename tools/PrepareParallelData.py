'''
Binarize the training data for NiuTrans.NMT
Help: python3 PrepareParallelData.py -h

Training data format (binary):
1. first 16 bits: source & target vocabulary size
2. second 8 bits: number of sentence pairs
3. subsequent segements:
source sentence length (4 bits)
target sentence length (4 bits)
source tokens (4 bits per token)
target tokens (4 bits per token)
'''

import argparse
from struct import pack

# User defined words
PAD = 1
SOS = 2
EOS = 2
UNK = 3

parser = argparse.ArgumentParser(
    description='Binarize the training data for NiuTrans.NMT')
parser.add_argument('-src', help='Path to the source language file',
                    type=str, required=True, default='')
parser.add_argument('-tgt', help='Path to the target language file',
                    type=str, required=True, default='')
parser.add_argument(
    '-maxsrc', help='The maximum source sentence length, default: 200', type=int, default=200)
parser.add_argument(
    '-maxtgt', help='The maximum target sentence length, default: 200', type=int, default=200)
parser.add_argument(
    '-sv', help='Path to the source language vocab file', type=str, default='')
parser.add_argument(
    '-tv', help='Path to the target language vocab file', type=str, default='')
parser.add_argument('-output', help='Path to the binarized training file',
                    type=str, required=True, default='')
args = parser.parse_args()

sv = dict()
tv = dict()
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


# load the vocabularies
sv_size = load_vocab(sv, args.sv)
tv_size = load_vocab(tv, args.tv)
if (not isinstance(sv_size, int)) or (sv_size <= 0):
    raise ValueError("Invalid source vocabulary size")
if (not isinstance(tv_size, int)) or (sv_size <= 0):
    raise ValueError("Invalid target vocabulary size")


with open(args.src, 'r', encoding='utf8') as fs:
    with open(args.tgt, 'r', encoding='utf8') as ft:
        src_sentences, tgt_sentences = list(), list()
        for ls in fs:
            ls = ls.split()
            lt = ft.readline().split()

            # limit the source/target sequence length
            if len(ls) >= args.maxsrc:
                cut_num += 1
                ls = ls[:args.maxsrc - 1]
            if len(lt) >= args.maxtgt:
                cut_num += 1
                lt = lt[:args.maxtgt - 1]

            # append EOS to the begin of source sequence
            src_sent = [get_id(sv, w) for w in ls] + [EOS]

            # append SOS to the end of target sequence
            tgt_sent = [SOS] + [get_id(tv, w, False) for w in lt]

            src_sentences.append(src_sent)
            tgt_sentences.append(tgt_sent)

        # print information
        src_tokens = sum([len(s) - 1 for s in src_sentences])
        tgt_tokens = sum([len(t) - 1 for t in tgt_sentences])
        print("{}: {} sents, {} tokens, {:.2f} replaced by <UNK>".format(
            args.src, len(src_sentences), src_tokens, sum([s.count(UNK) for s in src_sentences]) / src_tokens))
        print("{}: {} sents, {} tokens, {:.2f} replaced by <UNK>".format(
            args.tgt, len(tgt_sentences), tgt_tokens, sum([s.count(UNK) for s in tgt_sentences]) / tgt_tokens))

        with open(args.output, 'wb') as fo:
            # seg 1: source and target vocabulary size (4 bits per size, 8 bits in total)
            vocab_size = [sv_size, tv_size]
            vocab_size_pack = pack("i" * len(vocab_size), *vocab_size)
            fo.write(vocab_size_pack)

            # seg 2: user-defined tokens (4 bits per token, 16 bits in total)
            user_defined_tokens = [PAD, SOS, EOS, UNK]
            user_defined_tokens_pack = pack(
                "i" * len(user_defined_tokens), *user_defined_tokens)
            fo.write(user_defined_tokens_pack)

            # seg 3: number of sentence pairs (4 bits)
            sent_num = [len(src_sentences)]
            sent_num_pack = pack("i", *sent_num)
            fo.write(sent_num_pack)

            # seg 4: length and contents of sentence pairs
            for i in range(len(src_sentences)):
                src_sent = src_sentences[i]
                tgt_sent = tgt_sentences[i]

                # number of source and target sentence length (4 bit per number)
                src_tgt_length = [len(src_sent), len(tgt_sent)]
                src_tgt_length_pack = pack(
                    "i" * len(src_tgt_length), *src_tgt_length)
                fo.write(src_tgt_length_pack)

                # source sentence and target sentence pairs (4 bit per token)
                src_sent_pack = pack("i" * len(src_sent), *src_sent)
                fo.write(src_sent_pack)
                tgt_sent_pack = pack("i" * len(tgt_sent), *tgt_sent)
                fo.write(tgt_sent_pack)
