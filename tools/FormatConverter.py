'''
Convert the format of a model.
Usage: python3 FormatConverter.py -src <raw_model> -tgt <new_model>
Help: python3 FormatConverter.py -h
'''

import argparse
import numpy as np
from glob import glob
from struct import pack
from struct import unpack

parser = argparse.ArgumentParser(
    description='The format converter for NiuTrans.NMT')
parser.add_argument('-input', help='Path of the raw model file',
                    type=str, default='')
parser.add_argument('-output', help='Path of the new model file',
                    type=str, default='')
parser.add_argument('-format', help='Target storage format, FP16 (Default) or FP32', type=str, default='fp16')
args = parser.parse_args()
args.format = args.format.lower()

META_INFO_NUM = 12

meta_infos = None
parameters = None

if args.format == 'fp32':
    PARAM_LEN = 2
elif args.format == 'fp16':
    PARAM_LEN = 4
else:
    raise NotImplementedError("Unsupported data type")

with open(args.input, "rb") as f:
    meta_infos = f.read(META_INFO_NUM * 4)
    data = f.read()
    if args.format == 'fp32':
        values = unpack('e' * (len(data) // PARAM_LEN), data)
    elif args.format == 'fp16':
        values = unpack('f' * (len(data) // PARAM_LEN), data)
    print("Loaded {} parameters from: {}".format(len(values), args.input))
    parameters = np.array(values)

with open(args.output, "wb") as f:
    f.write(meta_infos)
    if args.format == 'fp32':
        values = pack("f" * len(parameters), *(parameters.astype(np.float32)))
    elif args.format == 'fp16':
        values = pack("e" * len(parameters), *(parameters.astype(np.float16)))
    f.write(values)