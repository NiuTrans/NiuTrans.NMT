/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __MODEL_H__
#define __MODEL_H__

#include "Encoder.h"
#include "Decoder.h"
#include "layer/FNN.h"
#include "layer/Output.h"
#include "Utility.h"
#include "layer/Attention.h"

namespace nmt
{

/* a nmt model that keeps parameters of the encoder,
   the decoder and the output layer (softmax). */
class Model
{
public:
    /* device id */
    int devID;

    /* the encoder */
    AttEncoder* encoder;

    /* the decoder */
    AttDecoder* decoder;

    /* output layer */
    Output* outputLayer;

    /* indicates whether the model is running for language modeling */
    bool isLM;

    /* indicates whether the model is running for machine translation */
    bool isMT;

    /* indicates whether the model is running with FP16 data type */
    bool useFP16;

    /* number of heads in the attention model */
    int nhead;

    /* indicates whether share encoders embeddings with decoders */
    int shareAllEmbeddings;

    /* indicates whether share decoder embeddings with output weights */
    int shareDecInputOutputWeight;

public:
    /* constructor */
    Model();

    /* de-constructor */
    ~Model();

    /* initialize the model */
    void InitModel(Config& config);

    /* print model configurations */
    void ShowModelConfig(Config& config);

    /* make the encoding network */
    XTensor MakeEncoder(XTensor& input, XTensor* mask, bool isTraining);

    /* make the encoding network */
    XTensor MakeDecoder(XTensor& inputEnc, XTensor& inputDec, XTensor* mask,
        XTensor& MaskEncDec, bool isTraining);

    /* make the network for language modeling (with the output softmax layer) */
    void MakeLM(XTensor& input, XTensor& output, XTensor& padding, bool isTraining);

    /* make the network for machine translation (with the output softmax layer) */
    void MakeMT(XTensor& inputEnc, XTensor& inputDec, XTensor& output,
        XTensor& paddingEnc, XTensor& paddingDec, bool isTraining);

    /* make the mask for training MT models */
    void MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
        XTensor& paddingEnc, XTensor& paddingDec,
        XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec);

    /* make the mask of the encoder */
    void MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc);

    /* make the mask of the decoder */
    void MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
        XTensor& maskDec, XTensor& maskEncDec);

    /* get parameter matrices */
    void GetParams(TensorList& list);

    /* dump the model to a file */
    void Dump(const char* fn);

    /* read the parameters */
    void Read(FILE* file);
};

}

#endif
