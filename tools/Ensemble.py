'''
Ensemble multiple models by checkpoint averaging.
Usage: python3 Ensemble.py -src <model_files> -tgt <ensembled_model>
Help: python3 ModelConverter.py -h
'''


import argparse
import numpy as np
from glob import glob
from struct import pack
from struct import unpack

parser = argparse.ArgumentParser(
    description='A model ensemble tool for NiuTrans.NMT')
parser.add_argument('-input', help='Model file pattern, e.g., \'model.bin.*\'',
                    type=str, default='model.bin.*')
parser.add_argument('-output', help='The ensembled model',
                    type=str, default='model.ensemble')
args = parser.parse_args()

model_files = glob(args.input)

meta_infos = None
parameters = []

for file in model_files:
    with open(file, "rb") as f:
        meta_infos = f.read(12 * 4)
        data = f.read()
        values = unpack('f' * (len(data) // 4), data)
        print("Loaded {} parameters from: {}".format(len(values), file))
        parameters.append(np.array(values))

parameters = np.mean(np.array(parameters), axis=0)

with open(args.output, "wb") as f:
    f.write(meta_infos)
    values = pack("f" * len(parameters), *parameters)
    f.write(values)

print("Model ensemble finished")
