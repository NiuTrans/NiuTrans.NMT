'''
Convert a fairseq vocab to a NiuTrans.NMT vocab
Usage: python3 VocabConverter.py [fairseq_vocab] [niutrans_nmt_vocab]
'''

import sys

# User defined words
PAD=1
SOS=2
EOS=2
UNK=3

with open(sys.argv[1], "r", encoding="utf8") as fi:
    with open(sys.argv[2], "w", encoding="utf8") as fo:
        lines = fi.readlines()

        # the first several indices are reserved
        start_id = UNK + 1
        
        # the first line: vocab_size, start_id
        fo.write("{} {}\n".format(len(lines)+start_id, start_id))

        # other lines: word, id
        for l in lines:
            fo.write("{} {}\n".format(l.split()[0], start_id))
            start_id += 1