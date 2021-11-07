'''
Ensemble multiple NiuTrans.NMT models by checkpoint averaging.
Usage: python3 Ensemble.py -input <model_files> -output <ensembled_model>
Example: python Ensemble.py -input 'model.bin.epoch.00*' -output model.ensemble
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
parser.add_argument('-output', help='The ensembled model, e.g., model.ensemble',
                    type=str, default='model.ensemble')
args = parser.parse_args()

model_files = glob(args.input)

meta_info = None
parameters = []

for file in model_files:
    with open(file, "rb") as f:
        # meta infomation includes 11 booleans and 18 integers, detailed in Model.cpp:InitModel()
        meta_info = f.read(11 * 1 + 18 * 4)
        data = f.read()
        values = unpack('f' * (len(data) // 4), data)
        print("Loaded {} parameters from: {}".format(len(values), file))
        parameters.append(np.array(values))

parameters = np.mean(np.array(parameters), axis=0)

with open(args.output, "wb") as f:
    f.write(meta_info)
    values = pack("f" * len(parameters), *parameters)
    f.write(values)

print("Model ensemble finished")
