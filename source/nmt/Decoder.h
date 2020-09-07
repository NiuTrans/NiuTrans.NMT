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

#ifndef __DECODER_H__
#define __DECODER_H__

#include "Encoder.h"
#include "Utility.h"

namespace nmt
{

class AttDecoder
{
public:

    /* device id */
    int devID;

    /* layer number */
    int nlayer;

    /* hidden layer size of the FNN layer */
    int hSize;

    /* embedding size */
    int eSize;

    /* vocabulary size */
    int vSize;

    /* dropout probability */
    DTYPE dropoutP;

    /* embedding of word at each position */
    Embedder embedder;

    /* FNN model of each layer */
    FNN* fnns;

    /* attention model of each layer */
    Attention* selfAtt;

    /* layer normalization for attention */
    LN* selfAttLayerNorms;

    /* layer normalization for fnn */
    LN* fnnLayerNorms;

    /* layer normalization for decoder */
    LN* decoderLayerNorm;

    /* encoder-decoder attention model of each layer */
    Attention* enDeAtt;

    /* layer normalization for encoder-decoder attention */
    LN* enDeAttLayerNorms;

    /* layer cache list */
    Cache* selfAttCache;

    /* layer cache list */
    Cache* enDeAttCache;

    /* the location of layer normalization */
    bool preNorm;

public:
    /* constructor */
    AttDecoder();

    /* de-constructor */
    ~AttDecoder();

    /* initialize the model */
    void InitModel(Config& config);

    /* make the decoding network */
    XTensor Make(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                 XTensor* maskEncDec, int nstep, bool isTraining);

    /* make the decoding network (pre norm) */
    XTensor MakeFast(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                     XTensor* maskEncDec, int nstep, bool isTraining);
};

}

#endif