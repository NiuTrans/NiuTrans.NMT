'''
Convert a fairseq vocab to a niutensor vocab
Usage: python3 PrepareParallelData.py -src [src_file] -tgt [tgt_file] -src_vocab [src_vocab_file] -tgt_vocab [tgt_vocab_file] -output [training_file]
Help: python3 PrepareParallelData.py -h
'''

# User defined words
PAD=1
SOS=2
EOS=2
UNK=3

import argparse

parser = argparse.ArgumentParser(description='prepare parallel data for nmt training')
parser.add_argument('-src', help='source language file', type=str, default='')
parser.add_argument('-tgt', help='target language file', type=str, default='')
parser.add_argument('-src_vocab', help='source language vocab file', type=str, default='')
parser.add_argument('-tgt_vocab', help='target language vocab file', type=str, default='')
parser.add_argument('-output', help='training file', type=str, default='')
args = parser.parse_args()

src_vocab=dict()
tgt_vocab=dict()

def load_vocab(vocab, file):
    with open(file, 'r', encoding='utf8') as f:
        f.readline()
        for l in f:
            l = l.split()
            vocab[l[0]] = vocab[l[1]]

def get_id(vocab, word):
    if word in vocab.keys():
        return vocab[word]
    else:
        return UNK

load_vocab(src_vocab, args.src_vocab)
load_vocab(tgt_vocab, args.tgt_vocab)

with open(args.src, 'r', encoding='utf8') as fs:
    with open(args.tgt, 'r', encoding='utf8') as ft:
        with open(args.output, 'w', encoding='utf8') as fo:
            for ls in fs:
                ls = ls.split()
                src_sent = [SOS] + [get_id(src_vocab, w) for w in ls] + [EOS]
                lt = ft.readline().split()
                tgt_sent = [SOS] + [get_id(tgt_vocab, w) for w in lt] + [EOS]
                fo.write(' '.join([str(w) for w in src_sent]))
                fo.write(' ||| ')
                fo.write(' '.join([str(w) for w in tgt_sent]))
                fo.write('\n')