'''
Convert a fairseq vocab to a niutensor vocab
Usage: python3 VocabConverter.py [fairseq_vocab] [niutensor_vocab]
'''

import sys

with open(sys.argv[1], "r", encoding="utf8") as fi:
    with open(sys.argv[2], "w", encoding="utf8") as fo:
        lines = fi.readlines()

        # the first 4 indices are reserved
        start_id = 4
        
        # the first line: vocab_size, start_id
        fo.write("{} {}\n".format(len(lines)+start_id, start_id))

        # other lines: words, ids
        for l in lines:
            fo.write("{} {}\n".format(l.split()[0], start_id))
            start_id += 1