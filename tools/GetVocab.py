'''
Convert a bpe vocabulary to a NiuTrans.NMT vocab
Usage: python3 GetVocab.py -src [bpe_vocab] -tgt [niutrans_nmt_vocab]
'''

import sys
import argparse

parser = argparse.ArgumentParser(description='prepare parallel data for nmt training')
parser.add_argument('-src', help='source language vocabulary file', type=str, default='')
parser.add_argument('-tgt', help='target language vocabulary file', type=str, default='')
args = parser.parse_args()

# User defined words
PAD=1
SOS=2
EOS=2
UNK=3

with open(args.src, "r", encoding="utf8") as fi:
    with open(args.tgt, "w", encoding="utf8") as fo:

        all_lines = fi.readlines()
        vocab_size = len(all_lines) + UNK + 1

        # make sure the vocabulary size is divisible by 8
        vocab_size += (8 - vocab_size % 8)

        start_id = UNK + 1

        # first line: vocab size, start id
        fo.write("{} {}\n".format(vocab_size, start_id))

        # other lines: word, id
        for l in all_lines:
            
            fo.write("{} {}\n".format(l.split()[0], start_id))
            start_id += 1