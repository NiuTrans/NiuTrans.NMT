'''
Convert the format of a NiuTrans.NMT model (FP32 <-> FP16).
Usage: python3 FormatConverter.py -input <raw_model> -output <new_model>
Help: python3 FormatConverter.py -h
'''

import argparse
import numpy as np
from glob import glob
from struct import pack
from struct import unpack

parser = argparse.ArgumentParser(
    description='The format converter for NiuTrans.NMT (FP32 <-> FP16)')
parser.add_argument('-input', help='Path of the raw model file',
                    type=str, default='')
parser.add_argument('-output', help='Path of the new model file',
                    type=str, default='')
parser.add_argument(
    '-format', help='Target storage format, FP16 (Default) or FP32', type=str, default='fp16')
args = parser.parse_args()
args.format = args.format.lower()


if args.format == 'fp32':
    PARAM_LEN = 2
elif args.format == 'fp16':
    PARAM_LEN = 4
else:
    raise NotImplementedError("Unsupported data type")

with open(args.input, "rb") as f:

    # meta infomation includes 11 booleans and 18 integers, detailed in Model.cpp:InitModel()
    meta_info = f.read(11 * 1 + 18 * 4)
    data = f.read()
    if args.format == 'fp32':
        values = unpack('e' * (len(data) // PARAM_LEN), data)
    elif args.format == 'fp16':
        values = unpack('f' * (len(data) // PARAM_LEN), data)
    print("Loaded {} parameters from: {}".format(len(values), args.input))
    parameters = np.array(values)

with open(args.output, "wb") as f:
    f.write(meta_info)
    if args.format == 'fp32':
        values = pack("f" * len(parameters), *(parameters.astype(np.float32)))
    elif args.format == 'fp16':
        values = pack("e" * len(parameters), *(parameters.astype(np.float16)))
    f.write(values)
