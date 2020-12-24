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

#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "Utility.h"
#include "layer/FNN.h"
#include "layer/Attention.h"
#include "layer/Embedding.h"
#include "layer/LayerNorm.h"
#include "../niutensor/network/XNet.h"
#include "layer/LayerHistory.h"

using namespace nts;

namespace nmt
{

/*
base class of the encoder
*/
class Encoder
{
public:
    virtual XTensor Make(XTensor& input, XTensor* mask, XTensor& mask2, bool isTraining) = 0;
};

/*
the encoder based on self-attention
*/
class AttEncoder : Encoder
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

    /* some positions can be ignored in attention. this is useful in lm where the first position needs
       special design for the attention model. */
    int ignored;

    /* embedding of word at each position */
    Embedder embedder;

    /* FNN model of each layer */
    FNN* fnns;

    /* attention model of each layer */
    Attention* selfAtt;

    /* layer normalizations for attention */
    LN* attLayerNorms;

    /* layer normalization for fnn */
    LN* fnnLayerNorms;

    /* layer normalization for encoder */
    LN* encoderLayerNorm;

    /* dynamic layer history */
    LayerHistory* history;

    /* the location of layer normalization */
    bool preNorm;

    /* add LN to the encoder output or not */
    bool finalNorm;

    /* reserve history for layers or not */
    bool useHistory;

public:
    /* constructor */
    AttEncoder();

    /* de-constructor */
    ~AttEncoder();

    /* initialize the model */
    void InitModel(Config& config);

    /* make the encoding network */
    XTensor Make(XTensor& input, XTensor* mask, XTensor& maskEncDec, bool isTraining);

    /* make the encoding network (wrapper) */
    XTensor Make(XTensor& input, XTensor* mask, bool isTraining);
};

}

#endif
