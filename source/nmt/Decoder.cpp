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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-10-09
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Decoder.h"
#include "Utility.h"
#include "layer/LayerNorm.h"
#include "layer/CommonModules.h"
#include "../niutensor/tensor/core/CHeader.h"

namespace nmt
{

/* constructor */
AttDecoder::AttDecoder()
{
    selfAtt = NULL;
    fnns = NULL;
    selfAttLayerNorms = NULL;
    fnnLayerNorms = NULL;
    enDeAtt = NULL;
    enDeAttLayerNorms = NULL;
    decoderLayerNorm = NULL;
    selfAttCache = NULL;
    enDeAttCache = NULL;
}

/* de-constructor */
AttDecoder::~AttDecoder()
{
    delete[] selfAttCache;
    delete[] enDeAttCache;
    delete[] selfAtt;
    delete[] fnns;
    delete[] selfAttLayerNorms;
    delete[] fnnLayerNorms;
    delete[] enDeAtt;
    delete[] enDeAttLayerNorms;
    if (preNorm)
        delete decoderLayerNorm;
}

/*
initialize the model
>> config - configurations of the model
*/
void AttDecoder::InitModel(Config& config)
{
    devID = config.devID;
    nlayer = config.nDecLayer;
    hSize = config.modelSize;
    eSize = config.embSize;
    vSize = config.tgtVocabSize;
    dropoutP = config.dropout;
    preNorm = config.preNorm;

    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");
    CheckNTErrors(vSize > 1, "set vocabulary size by \"-vsizetgt\"");

    /* embedding model */
    embedder.InitModel(config, false);

    selfAtt = new Attention[nlayer];
    fnns = new FNN[nlayer];
    selfAttLayerNorms = new LN[nlayer];
    enDeAtt = new Attention[nlayer];
    enDeAttLayerNorms = new LN[nlayer];
    fnnLayerNorms = new LN[nlayer];

    selfAttCache = new Cache[nlayer];
    enDeAttCache = new Cache[nlayer];
    if (preNorm)
        decoderLayerNorm = new LN;

    /* initialize the stacked layers */
    for (int i = 0; i < nlayer; i++) {
        selfAtt[i].InitModel(config);
        fnns[i].InitModel(config);
        selfAttLayerNorms[i].InitModel(config);
        fnnLayerNorms[i].InitModel(config);
        enDeAtt[i].InitModel(config);
        enDeAttLayerNorms[i].InitModel(config);
        selfAttCache[i].enable = true;
        enDeAttCache[i].enable = true;
    }
    if (preNorm)
        decoderLayerNorm->InitModel(config);
}

/*
make the decoding network
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::Make(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                         XTensor* maskEncDec, int nstep, bool isTraining)
{
    XTensor x;

    x = embedder.Make(inputDec, true, isTraining, nstep);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP);

    for (int i = 0; i < nlayer; i++) {
        XTensor att;
        XTensor ende;
        XTensor fnn;
        XTensor res;
        XTensor selfAttnBefore;
        XTensor selfAttnAfter;
        XTensor endeAttnBefore;
        XTensor endeAttnAfter;
        XTensor fnnBefore;

        /* layer normalization with pre-norm for self-attn */
        selfAttnBefore = LayerNorm(x, selfAttLayerNorms[i], preNorm, true, false);

        /******************/
        /* self attention */
        att = selfAtt[i].Make(selfAttnBefore, selfAttnBefore, selfAttnBefore, 
                              mask, isTraining, &selfAttCache[i], SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP);

        /* residual connection */
        res = Sum(att, x);

        /* layer normalization with post-norm for self-attention */
        selfAttnAfter = LayerNorm(res, selfAttLayerNorms[i], preNorm, false, true);

        /* layer normalization with pre-norm for encoder-decoder attention */
        endeAttnBefore = LayerNorm(selfAttnAfter, enDeAttLayerNorms[i], preNorm, true, false);

        /* encoder-decoder attention */
        ende = enDeAtt[i].Make(outputEnc, endeAttnBefore, outputEnc, maskEncDec, 
                               isTraining, &enDeAttCache[i], EN_DE_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            ende = Dropout(ende, dropoutP);

        /* residual connection */
        res = Sum(ende, selfAttnAfter);

        /* layer normalization with post-norm for encoder-decoder attention */
        endeAttnAfter = LayerNorm(res, enDeAttLayerNorms[i], preNorm, false, true);

        /* layer normalization with pre-norm for fnn */
        fnnBefore = LayerNorm(endeAttnAfter, fnnLayerNorms[i], preNorm, true, false);

        /* fnn */
        fnn = fnns[i].Make(fnnBefore, isTraining);

        /* dropout */
        if (isTraining && dropoutP > 0)
            fnn = Dropout(fnn, dropoutP);

        /* residual connection */
        res = Sum(fnn, endeAttnAfter);

        /* layer normalization with post-norm for fnn */
        x = LayerNorm(res, fnnLayerNorms[i], preNorm, false, true);
    }

    if (preNorm)
        return decoderLayerNorm->Make(x);

    return x;
}

/*
make the decoding network
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::MakeFast(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                             XTensor* maskEncDec, int nstep, bool isTraining)
{
    XTensor x;

    x = embedder.Make(inputDec, true, isTraining, nstep);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP);

    for (int i = 0; i < nlayer; i++) {
        XTensor res;

        res = x;

        /* layer normalization with pre-norm for self-attn */
        x = selfAttLayerNorms[i].Make(x);

        /******************/
        /* self attention */
        x = selfAtt[i].Make(x, x, x, mask, isTraining, &selfAttCache[i], SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            x = Dropout(x, dropoutP);

        /* residual connection */
        x = Sum(res, x);

        res = x;

        /* layer normalization with pre-norm for encoder-decoder attention */
        x = enDeAttLayerNorms[i].Make(x);

        /* encoder-decoder attention */
        x = enDeAtt[i].Make(outputEnc, x, outputEnc, maskEncDec,
                            isTraining, &enDeAttCache[i], EN_DE_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            x = Dropout(x, dropoutP);

        /* residual connection */
        x = Sum(res, x);

        res = x;

        /* layer normalization with pre-norm for fnn */
        x = fnnLayerNorms[i].Make(x);

        /* fnn */
        x = fnns[i].Make(x, isTraining);

        /* dropout */
        if (isTraining && dropoutP > 0)
            x = Dropout(x, dropoutP);

        /* residual connection */
        x = Sum(res, x);
    }

    x = decoderLayerNorm->Make(x);

    return x;
}

}