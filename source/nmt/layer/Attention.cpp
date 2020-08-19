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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include "Attention.h"
#include "Embedding.h"
#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"

namespace nmt
{
/* constructor */
Attention::Attention()
{
    nhead = -1;
    dk = -1;
    dv = -1;
    d = -1;
}

/* de-constructor */
Attention::~Attention()
{
}

/*
initialize the model
>> config - the configurations of the network
*/
void Attention::InitModel(Config& config)
{
    devID = config.devID;
    useRPR = config.useRPR;

    nhead = config.nhead;
    d = config.modelSize;
    dk = config.modelSize;
    dv = config.modelSize;
    maxRP = config.maxRP;
    dropoutP = config.attDropout;

    /* initialize the parameters */
    InitTensor2D(&weightQ, d, d, X_FLOAT, devID);
    InitTensor1D(&biasQ, d, X_FLOAT, devID);
    InitTensor2D(&weightK, d, d, X_FLOAT, devID);
    InitTensor1D(&biasK, d, X_FLOAT, devID);
    InitTensor2D(&weightV, d, d, X_FLOAT, devID);
    InitTensor1D(&biasV, d, X_FLOAT, devID);
    if (useRPR)
        InitTensor2D(&RPEmbK, maxRP * 2 + 1, d / nhead, X_FLOAT, devID);
    InitTensor2D(&weightO, d, d, X_FLOAT, devID);
    InitTensor1D(&biasO, d, X_FLOAT, devID);

    float scale = 1.0F;
    _SetDataFanInOut(&weightK, scale);
    _SetDataFanInOut(&weightQ, scale);
    _SetDataFanInOut(&weightV, scale);
    _SetDataFanInOut(&weightO, scale);
    if (useRPR)
        _SetDataFanInOut(&RPEmbK, scale);
    biasK.SetZeroAll();
    biasQ.SetZeroAll();
    biasV.SetZeroAll();
    biasO.SetZeroAll();

    biasK.SetDataRand(-(DTYPE)sqrt(6.0F / d), (DTYPE)sqrt(6.0F / d));
    biasV.SetDataRand(-(DTYPE)sqrt(6.0F / d), (DTYPE)sqrt(6.0F / d));
}

/*
make the network
>> k - keys, B * L * H for encoders, B * 1 * H for decoders
       where B = batch size, L = sequence length,
       and H = vector size of each position
>> q - queries, B * L * H
>> v - values, B * L * H for encoders, B * 1 * H for decoders
>> mask - as it is
>> isTraining - indicates whether the model is used for training
>> cache - decoder cache
>> cacheType - type of cache, e.g., self-attention
<< return - multi-attention result
*/
XTensor Attention::Make(XTensor& k, XTensor& q, XTensor& v, XTensor* mask,
                        bool isTraining, Cache* cache, int cacheType)
{
    const bool isEnc = (!cache) ? true : false;

    /* linear transformation before self-attention */
    XTensor q2, k2, v2;

    q2 = MulAndShift(q, weightQ, biasQ);

    if (!cache || isTraining || !(cache->enable)) {
        /* self attention for encoder layers */
        k2 = MulAndShift(k, weightK, biasK);
        v2 = MulAndShift(v, weightV, biasV);

        if (useRPR)
            return MakeRPRAttention(k2, q2, v2, mask, isTraining, isEnc);
        return MakeAttention(k2, q2, v2, mask, isTraining);
    }

    else {
        if (cacheType == SELF_ATT) {
            k2 = MulAndShift(k, weightK, biasK);
            v2 = MulAndShift(v, weightV, biasV);

            /* if hit, we only concat the cache with the new token */
            if (!cache->miss) {
                k2 = Concatenate(cache->key, k2, 1);
                v2 = Concatenate(cache->value, v2, 1);
            }
            cache->key = k2;
            cache->value = v2;
            cache->miss = false;

            if (useRPR)
                return MakeRPRAttention(cache->key, q2, cache->value, mask, isTraining, isEnc);
            return MakeAttention(cache->key, q2, cache->value, mask, isTraining);
        }
        else if (cacheType == EN_DE_ATT) {
            if (cache->miss) {
                cache->key = MulAndShift(k, weightK, biasK);
                cache->value = MulAndShift(v, weightV, biasV);
                cache->miss = false;
            }

            return MakeAttention(cache->key, q2, cache->value, mask, isTraining);
        }
        CheckNTErrors(0, "invalid cache type");
    }
}

/*
make the attention network given keys, queries and values (after linear transformation)
>> k - keys, B * L * H
>> q - queries, B * L * H
>> v - values, B * L * H
>> mask - as it is
>> isTraining - indicates whether the model is used for training
*/
XTensor Attention::MakeAttention(XTensor& k, XTensor& q, XTensor& v,
                                 XTensor* mask, bool isTraining)
{
    XTensor kheads;
    XTensor qheads;
    XTensor vheads;

    const auto dataType = k.dataType;

    /* multi head */
    kheads = Split(k, k.order - 1, nhead);
    qheads = Split(q, q.order - 1, nhead);
    vheads = Split(v, v.order - 1, nhead);

    XTensor att;
    XTensor dot;
    XTensor scalar;

    /* Some operations may cause numerical overflow under FP16 including
       BMMul, Mask, Div and Softmax. So we need to cast the input to FP32 */

    if (qheads.dataType == X_FLOAT16) {
        qheads = ConvertDataType(qheads, X_FLOAT);
        kheads = ConvertDataType(kheads, X_FLOAT);
    }

    /* scalar = softmax(Q * K^T / sqrt(dk)) * V */
    dot = BMMul(qheads, X_NOTRANS, kheads, X_TRANS);

    if (mask)
        dot = dot + (*mask);

    dot = Linear(dot, 1.0F / (float)sqrt((float)dk / nhead));

    scalar = Softmax(dot, -1);

    if (isTraining && dropoutP > 0)
        scalar = Dropout(scalar, dropoutP);

    if (vheads.dataType != scalar.dataType)
        vheads = ConvertDataType(vheads, scalar.dataType);

    att = BMMul(scalar, vheads);

    if (dataType != att.dataType)
        att = ConvertDataType(att, dataType);

    /* concatenate the heads */
    return MulAndShift(Merge(att, att.order - 1), weightO, biasO);
}

/*
make the attention network by incorporating the relative position representation
with the given keys, queries and values (after linear transformation)
>> k - keys, B * L * H
>> q - queries, B * L * H
>> v - values, B * L * H
>> mask - as it is
>> isTraining - indicates whether the model is used for training
>> isEnc - indicates whether it is encoder
*/
XTensor Attention::MakeRPRAttention(XTensor& k, XTensor& q, XTensor& v,
                                    XTensor* mask, bool isTraining, bool isEnc)
{
    XTensor kheads;
    XTensor qheads;
    XTensor vheads;

    const int batchSize = q.dimSize[0];
    const int lenQ = q.dimSize[1];
    const int lenKV = k.dimSize[1];

    const auto dataType = k.dataType;

    /* multi head */
    kheads = Split(k, k.order - 1, nhead);
    qheads = Split(q, q.order - 1, nhead);
    vheads = Split(v, v.order - 1, nhead);

    XTensor att;
    XTensor dot;
    XTensor scalar;

    XTensor embMatrix, relativeKey;

    /* generate the relative emb index (L_q, L_kv) */
    embMatrix = GetRPEmbedding(lenQ, lenKV, maxRP, isEnc);

    /* generate the relative key from the RPEmbK (L_q, L_kv, H/K) */
    relativeKey = Gather(RPEmbK, embMatrix);

    if (qheads.dataType == X_FLOAT16) {
        qheads = ConvertDataType(qheads, X_FLOAT);
        kheads = ConvertDataType(kheads, X_FLOAT);
        relativeKey = ConvertDataType(relativeKey, X_FLOAT);
    }

    qheads = ScaleAndShift(qheads, 1.0F / float(nhead));

    dot = RPDotProduct(qheads, kheads, relativeKey, true);

    if (mask)
        dot = dot + (*mask);

    /* softmax */
    scalar = Softmax(dot, -1);

    if (isTraining && dropoutP > 0)
        scalar = Dropout(scalar, dropoutP);

    if (vheads.dataType != scalar.dataType)
        vheads = ConvertDataType(vheads, scalar.dataType);

    /* generate the relative attention output (K, B, L_q, H/K) */
    att = BMMul(scalar, vheads);

    if (dataType != att.dataType)
        att = ConvertDataType(att, dataType);

    /* concatenate the heads */
    return MulAndShift(Merge(att, att.order - 1), weightO, biasO);
}

/*
generate relative position embeddings
>> lenQ - the length of query
>> lenKV - the length of key and value
>> maxRelativeLen - the maximum length of relative position
*/
XTensor Attention::GetRPEmbedding(const int lenQ, const int lenKV,
                                  const int maxRelativeLen, const bool isEnc)
{
    XTensor range;
    XTensor embMatrix;
    InitTensor1D(&range, lenKV, X_INT, devID);
    int* index = new int[lenKV];

    if (isEnc) {
        for (int i = 0; i < lenKV; i++)
            index[i] = i;
        range.SetData(index, lenKV);
        XTensor range2D;
        XTensor range2DTrans;
        range2D = Unsqueeze(range, 0, lenQ);
        range2DTrans = Transpose(range2D, 0, 1);
        embMatrix = Sum(range2D, range2DTrans, -1);
    }
    else {
        for (int i = 0; i < lenKV; i++)
            index[i] = -lenKV + i + 1;
        range.SetData(index, lenKV);
        embMatrix = Unsqueeze(range, 0, lenQ);
    }

    embMatrix = Clip(embMatrix, -float(maxRelativeLen), float(maxRelativeLen));
    embMatrix = ScaleAndShift(embMatrix, 1.0F, float(maxRelativeLen));

    delete[] index;
    return embMatrix;
}

/*
relative position-aware dot-product attention inner calculation.
>> x - Tensor with shape [batch_size*heads, length, length or depth].
>> y - Tensor with shape [batch_size*heads, length, depth].
>> z - Tensor with shape [length, length, depth].
>> isKey - Whether y is key.
<< return - A Tensor with shape [batch_size*heads, length, length or depth].
*/
XTensor Attention::RPDotProduct(XTensor& x, XTensor& y, XTensor& z, const bool isKey)
{
    const int headNum = nhead;
    const int batchSize = x.dimSize[1];
    const int lenQ = x.dimSize[2];
    const int lenKV = y.dimSize[2];
    const int depth = y.dimSize[3];

    const int lastDim = isKey ? lenKV : depth;
    MATRIX_TRANS_TYPE transposeFlag = isKey ? X_TRANS : X_NOTRANS;

    XTensor context;
    context = MatrixMulBatched(x, X_NOTRANS, y, transposeFlag);

    int mergeDims[] = { headNum * batchSize, lenQ, x.dimSize[3] };
    x = Reshape(x, 3, mergeDims);
    //x.Reshape(3, mergeDims);

    XTensor xTrans;
    xTrans = Transpose(x, 0, 1);

    XTensor relative;
    relative = MatrixMulBatched(xTrans, X_NOTRANS, z, transposeFlag);

    XTensor relativeTrans;
    relativeTrans = Transpose(relative, 0, 1);

    int splitDims[] = { headNum, batchSize, lenQ, lastDim };

    relativeTrans = Reshape(relativeTrans, 4, splitDims);
    //relativeTrans.Reshape(4, splitDims);

    return Sum(context, relativeTrans);
}

/* constructor */
Cache::Cache()
{
    miss = true;
    enable = true;
}

/* update the states cache */
void Cache::Update(XTensor&& k, XTensor&& v)
{
    key = k;
    value = v;
    miss = false;
}

/* keep alive states */
void Cache::KeepAlive(XTensor& aliveIdx)
{
    if (!miss) {
        key = AutoGather(key, aliveIdx);
        value = AutoGather(value, aliveIdx);
    }
}

/* reorder alive states */
void Cache::Reorder(XTensor& reorder)
{
    if (!miss) {
        key = AutoGather(key, reorder);
        value = AutoGather(value, reorder);
    }
}
}