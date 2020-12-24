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

#include <cstdint>

#include "Model.h"
#include "Utility.h"
#include "../niutensor/tensor/XUtility.h"
#include "../niutensor/tensor/core/CHeader.h"

namespace nmt
{

/* constructor */
Model::Model()
{
    devID = -1;
    isLM = false;
    isMT = false;
    useFP16 = false;
    shareAllEmbeddings = 0;
    shareDecInputOutputWeight = 0;
    nhead = 1;

    encoder = new AttEncoder();
    decoder = new AttDecoder();
    outputLayer = new Output();
}

/* de-constructor */
Model::~Model()
{
    delete encoder;
    delete decoder;
    delete outputLayer;
}

/*
initialize the model
>> config - configurations of the model
*/
void Model::InitModel(Config& config)
{
    devID = config.devID;
    isMT = config.isMT;
    isLM = !isMT;
    useFP16 = config.useFP16;

    /* configurations for the model */
    int* metaInfo[] = {
        &config.nEncLayer, &config.nDecLayer,
        &config.fnnHiddenSize, &config.modelSize,
        &config.embSize, &config.srcVocabSize,
        &config.tgtVocabSize, &config.nhead,
        &config.maxRP, &config.shareAllEmbeddings,
        &config.shareDecInputOutputWeight,
        &config.maxPosition
    };

    FILE* modelFile = NULL;

    /* read model configurations */
    if (!config.isTraining || strcmp(config.pretrainedModel, "") != 0) {
        if (strcmp(config.pretrainedModel, "") != 0)
            modelFile = fopen(config.pretrainedModel, "rb");
        else
            modelFile = fopen(config.modelFN, "rb");
        CheckNTErrors(modelFile, "Failed to open the model file");
        for (auto& meta : metaInfo) {
            fread(meta, sizeof(int), 1, modelFile);
        }
    }
    if (config.isTraining) {
        /* read the source and target vocab size */
        FILE* trainF = fopen(config.trainFN, "rb");
        CheckNTErrors(trainF, "Failed to open the training file");

        fread(&config.srcVocabSize, sizeof(config.srcVocabSize), 1, trainF);
        fread(&config.tgtVocabSize, sizeof(config.tgtVocabSize), 1, trainF);
        CheckNTErrors(config.srcVocabSize > 0, "Invalid source vocabulary size");
        CheckNTErrors(config.tgtVocabSize > 0, "Invalid target vocabulary size");
        fclose(trainF);
    }

    nhead = config.nhead;
    shareAllEmbeddings = config.shareAllEmbeddings;
    shareDecInputOutputWeight = config.shareDecInputOutputWeight;

    ShowModelConfig(config);

    encoder->InitModel(config);
    outputLayer->InitModel(config);

    if (isMT)
        decoder->InitModel(config);

    /* load parameters */
    if (!config.isTraining || strcmp(config.pretrainedModel, "") != 0)
        Read(modelFile);

    if (config.isTraining) {
        TensorList params;
        GetParams(params);
        for (int i = 0; i < params.Size(); i++)
            params[i]->SetVarFlag();
    }

    if (modelFile != NULL)
        fclose(modelFile);
}

/*
print model configurations
>> config - model configurations
*/
void Model::ShowModelConfig(Config& config)
{
    /* TODO: output more info */
    XPRINT1(0, stderr, "encoder layer: %d\n", config.nEncLayer);
    XPRINT1(0, stderr, "decoder layer: %d\n", config.nDecLayer);
    XPRINT1(0, stderr, "attention heads: %d\n", config.nhead);
    XPRINT1(0, stderr, "model size: %d\n", config.modelSize);
    XPRINT1(0, stderr, "source vocab size: %d\n", config.srcVocabSize);
    XPRINT1(0, stderr, "target vocab size: %d\n", config.tgtVocabSize);
}

/*
make the encoding network
>> input - input tensor, (batchSize, srcLen)
>> mask - the mask for encoder self-attention, (headNum, batchSize, srcLen, srcLen)
>> isTraining - indicates whether we are training the model
<< return - encoding result, (batchSize, srcLen, hiddenDim)
*/
XTensor Model::MakeEncoder(XTensor& input, XTensor* mask, bool isTraining)
{
    XTensor nothing;

    return encoder->Make(input, mask, nothing, isTraining);
}

/*
make the decoding network
>> inputDec - input tensor of the decoder, (batchSize, tgtLen)
>> outputEnc - output tensor of the encoder, (batchSize, srcLen, hiddenDim)
>> mask - mask for decoder self-attention, (headNum, batchSize, tgtLen, tgtLen)
>> maskEncDec - mask for the encoder-decoder attention, (headNum, batchSize, tgtLen, srcLen)
>> isTraining - indicates whether we are training the model
<< return - decoding result, (batchSize, tgtLen, hiddenDim)
*/
XTensor Model::MakeDecoder(XTensor& inputDec, XTensor& outputEnc,
    XTensor* mask, XTensor& maskEncDec, bool isTraining)
{
    return decoder->Make(inputDec, outputEnc, mask, &maskEncDec,
                         inputDec.GetDim(1), isTraining);
}

/*
make the network for language modeling (with the output softmax layer)
>> input - input tensor
>> output - output tensor (distribution)
>> padding - padding of the sequences
>> isTraining - indicates whether the model is for training
*/
void Model::MakeLM(XTensor& input, XTensor& output, XTensor& padding, bool isTraining)
{
    int len = padding.GetDim(padding.order - 1);
    int* dims = new int[padding.order + 2];
    for (int i = 0; i < padding.order; i++)
        dims[i + 1] = padding.GetDim(i);
    dims[0] = nhead;
    dims[padding.order + 1] = len;
    XTensor mask;
    InitTensor(&mask, padding.order + 2, dims, X_FLOAT, padding.devID);

    delete[] dims;

    /* a upper triangular matrix where the cells of the upper triangular are set to -1e-9.
        this matrix can be used to prevent the attention to current or following words in
        a given sequence. */
    _SetDataLowTri(&mask, 1e9F, 0);
    ScaleAndShiftMe(mask, 1.0F, -1e9F);

    /* forward */
    XTensor encoding;

    encoding = MakeEncoder(input, &mask, isTraining);
    outputLayer->Make(encoding, output, true, true);
}

/*
make the network for machine translation (with the output softmax layer)
>> inputEnc - input tensor of the encoder, (batchSize, srcLen)
>> inputDec - input tensor of the decoder, (batchSize, tgtLen)
>> output - output tensor (distribution), (batchSize, tgtLen, hiddenDim)
>> paddingEnc - padding of the sequences (on the encoder side), (batchSize, srcLen)
>> paddingDec - padding of the sequences (on the decoder side), (batchSize, tgtLen)
>> isTraining - indicates whether the model is for training
*/
void Model::MakeMT(XTensor& inputEnc, XTensor& inputDec, XTensor& output,
                   XTensor& paddingEnc, XTensor& paddingDec, bool isTraining)
{
    XTensor encoding;
    XTensor decoding;

    XTensor maskEnc;
    XTensor maskDec;
    XTensor maskEncDec;

    bool debug(false);

    /* encoder mask */
    MakeMTMaskEnc(paddingEnc, maskEnc);

    /* decoder mask */
    MakeMTMaskDec(paddingEnc, paddingDec, maskDec, maskEncDec);

    encoding = MakeEncoder(inputEnc, &maskEnc, isTraining);

    if (debug) {
        LOG("after encoding:");
        encoding.mem->ShowMemUsage(stderr);
    }
    
    decoding = MakeDecoder(inputDec, encoding, &maskDec, maskEncDec, isTraining);

    if (debug) {
        LOG("after decoding:");
        encoding.mem->ShowMemUsage(stderr);
    }

    outputLayer->Make(decoding, output, true, true);

    if (debug) {
        LOG("after outputing:");
        encoding.mem->ShowMemUsage(stderr);
        exit(0);
    }
}

/*
make the mask for training MT models
>> inputEnc - input of the encoder
>> inputDec - input of the decoder
>> paddingEnc - padding of the encoder input
>> paddingDec - padding of the decoder input
>> maskEnc - mask of the encoder self-attention
>> maksDec - mask of the decoder self-attention
>> maksEncDec - mask of the decoder enc-dec attention
*/
void Model::MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
                       XTensor& paddingEnc, XTensor& paddingDec,
                       XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec)
{
    int len = inputDec.GetDim(inputDec.order - 1);
    int* dims = new int[inputDec.order + 2];
    for (int i = 0; i < inputDec.order; i++)
        dims[i + 1] = inputDec.GetDim(i);
    dims[0] = nhead;
    dims[inputDec.order + 1] = len;
    InitTensor(&maskDec, inputDec.order + 2, dims, X_FLOAT, paddingDec.devID);

    /* an upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       this matrix can be used to prevent the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);
    ScaleAndShiftMe(maskDec, 1.0F, -1e9F);

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    dims[inputDec.order + 1] = inputEnc.GetDim(inputEnc.order - 1);
    InitTensor(&maskEncDec, inputDec.order + 2, dims, X_FLOAT, paddingEnc.devID);

    XTensor* maskEncDecTMPEnc = NewTensorBufV2(paddingEnc.order + 1, dims + 1,
        paddingEnc.dataType, 1.0F, paddingEnc.devID, paddingEnc.mem);
    XTensor* maskEncDecTMPDec = NewTensorBufV2(maskEncDecTMPEnc, paddingEnc.devID, paddingEnc.mem);

    _Unsqueeze(&paddingEnc, maskEncDecTMPEnc, paddingEnc.order - 1, paddingDec.GetDim(-1));
    _ScaleAndShiftMe(maskEncDecTMPEnc, 1e9F, -1e9F);
    _Unsqueeze(maskEncDecTMPEnc, &maskEncDec, 0, dims[0]);

    DelTensorBuf(maskEncDecTMPDec);
    DelTensorBuf(maskEncDecTMPEnc);

    /* padding on the source side */
    int* dimsPadding = new int[paddingEnc.order + 2];
    for (int i = 0; i < paddingEnc.order - 1; i++)
        dimsPadding[i] = paddingEnc.GetDim(i);
    dimsPadding[paddingEnc.order - 1] = paddingEnc.GetDim(-1);
    dimsPadding[paddingEnc.order] = paddingEnc.GetDim(-1);

    XTensor* padding2 = NewTensorBufV2(paddingEnc.order + 1, dimsPadding, paddingEnc.dataType, 1.0F,
        paddingEnc.devID, paddingEnc.mem);

    for (int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = nhead;

    XTensor* padding3 = NewTensorBufV2(paddingEnc.order + 2, dimsPadding, paddingEnc.dataType, 1.0F, paddingEnc.devID, paddingEnc.mem);

    /* mask of the padding */
    _Unsqueeze(&paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    _Unsqueeze(padding2, padding3, 0, nhead);

    _ScaleAndShiftMe(padding3, 1e9F, -1e9F);

    InitTensor(&maskEnc, padding3);
    maskEnc.SetZeroAll();

    /* generate the mask on the source language side (for padding) */
    _Sum(&maskEnc, padding3, &maskEnc);

    delete[] dims;
    delete[] dimsPadding;

    DelTensorBuf(padding3);
    DelTensorBuf(padding2);
}

/*
make the mask of the encoder
>> paddingEnc - padding of the encoder input, (batchSize, srcLen)
>> maskEnc - mask of the encoder self-attention, (headNum, batchSize, srcLen, srcLen)
*/
void Model::MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc)
{
    XTensor padding2;

    /* mask of the padding */
    Unsqueeze(paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));

    Unsqueeze(padding2, maskEnc, 0, nhead);
    ScaleAndShiftMe(maskEnc, 1e9F, -1e9F);
}

/*
make the mask of the decoder
>> paddingEnc - padding of the encoder input, (batchSize, srcLen)
>> paddingDec - padding of the decoder input, (batchSize, tgtLen)
>> maksDec - mask of the decoder self-attention, (headNum, batchSize, tgtLen, tgtLen)
>> maksEncDec - mask of the decoder enc-dec attention, (headNum, batchSize, tgtLen, srcLen)
*/
void Model::MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
                          XTensor& maskDec, XTensor& maskEncDec)
{
    int len = paddingDec.GetDim(paddingDec.order - 1);
    int* dims = new int[paddingDec.order + 2];
    for (int i = 0; i < paddingDec.order; i++)
        dims[i + 1] = paddingDec.GetDim(i);
    dims[0] = nhead;
    dims[paddingDec.order + 1] = len;
    InitTensor(&maskDec, paddingDec.order + 2, dims, X_FLOAT, paddingDec.devID);

    /* An upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       This matrix can be used to block the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);
    ScaleAndShiftMe(maskDec, 1.0F, -1e9F);

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    XTensor maskEncDecTMP;

    Unsqueeze(paddingEnc, maskEncDecTMP, paddingEnc.order - 1, paddingDec.GetDim(-1));
    ScaleAndShiftMe(maskEncDecTMP, 1e9F, -1e9F);

    Unsqueeze(maskEncDecTMP, maskEncDec, 0, dims[0]);

    delete[] dims;
}

/*
get parameter matrices
>> list - the list that keeps the parameter matrics
*/
void Model::GetParams(TensorList& list)
{
    list.Clear();

    /* encoder parameters */
    if (encoder->useHistory) {
        for (int i = 0; i < encoder->nlayer + 1; i++)
            list.Add(&encoder->history->weights[i]);
        for (int i = 0; i < encoder->nlayer; i++) {
            list.Add(&encoder->history->layerNorms[i].weight);
            list.Add(&encoder->history->layerNorms[i].bias);
        }
    }
    for (int i = 0; i < encoder->nlayer; i++) {
        list.Add(&encoder->selfAtt[i].weightQ);
        list.Add(&encoder->selfAtt[i].weightK);
        list.Add(&encoder->selfAtt[i].weightV);
        list.Add(&encoder->selfAtt[i].biasQ);
        list.Add(&encoder->selfAtt[i].biasK);
        list.Add(&encoder->selfAtt[i].biasV);
        if (encoder->selfAtt[i].useRPR)
            list.Add(&encoder->selfAtt[i].RPEmbK);
        list.Add(&encoder->selfAtt[i].weightO);
        list.Add(&encoder->selfAtt[i].biasO);
        list.Add(&encoder->fnns[i].w1);
        list.Add(&encoder->fnns[i].b1);
        list.Add(&encoder->fnns[i].w2);
        list.Add(&encoder->fnns[i].b2);
        list.Add(&encoder->attLayerNorms[i].weight);
        list.Add(&encoder->attLayerNorms[i].bias);
        list.Add(&encoder->fnnLayerNorms[i].weight);
        list.Add(&encoder->fnnLayerNorms[i].bias);
    }
    if (encoder->finalNorm) {
        list.Add(&encoder->encoderLayerNorm->weight);
        list.Add(&encoder->encoderLayerNorm->bias);
    }

    if (isMT) {
        /* decoder parameters */
        if (decoder->useHistory) {
            for (int i = 0; i < decoder->nlayer + 1; i++)
                list.Add(&decoder->history->weights[i]);
            for (int i = 0; i < decoder->nlayer; i++) {
                list.Add(&decoder->history->layerNorms[i].weight);
                list.Add(&decoder->history->layerNorms[i].bias);
            }
        }

        for (int i = 0; i < decoder->nlayer; i++) {
            list.Add(&decoder->selfAtt[i].weightQ);
            list.Add(&decoder->selfAtt[i].weightK);
            list.Add(&decoder->selfAtt[i].weightV);
            list.Add(&decoder->selfAtt[i].biasQ);
            list.Add(&decoder->selfAtt[i].biasK);
            list.Add(&decoder->selfAtt[i].biasV);
            if (decoder->selfAtt[i].useRPR)
                list.Add(&decoder->selfAtt[i].RPEmbK);
            list.Add(&decoder->selfAtt[i].weightO);
            list.Add(&decoder->selfAtt[i].biasO);
            list.Add(&decoder->selfAttLayerNorms[i].weight);
            list.Add(&decoder->selfAttLayerNorms[i].bias);
            list.Add(&decoder->enDeAtt[i].weightQ);
            list.Add(&decoder->enDeAtt[i].weightK);
            list.Add(&decoder->enDeAtt[i].weightV);
            list.Add(&decoder->enDeAtt[i].biasQ);
            list.Add(&decoder->enDeAtt[i].biasK);
            list.Add(&decoder->enDeAtt[i].biasV);
            list.Add(&decoder->enDeAtt[i].weightO);
            list.Add(&decoder->enDeAtt[i].biasO);
            list.Add(&decoder->enDeAttLayerNorms[i].weight);
            list.Add(&decoder->enDeAttLayerNorms[i].bias);
            list.Add(&decoder->fnns[i].w1);
            list.Add(&decoder->fnns[i].b1);
            list.Add(&decoder->fnns[i].w2);
            list.Add(&decoder->fnns[i].b2);
            list.Add(&decoder->fnnLayerNorms[i].weight);
            list.Add(&decoder->fnnLayerNorms[i].bias);
        }
        if (decoder->finalNorm) {
            list.Add(&decoder->decoderLayerNorm->weight);
            list.Add(&decoder->decoderLayerNorm->bias);
        }
    }

    list.Add(&encoder->embedder.w);

    if (isMT && (shareAllEmbeddings == 0)) {
        list.Add(&decoder->embedder.w);
    }

    if (shareDecInputOutputWeight == 0) {
        list.Add(&outputLayer->w);
    }
}

/*
dump the model to a file
>> fn - where to save the model
>> model - the model
*/
void Model::Dump(const char* fn)
{
    double startT = GetClockSec();

    FILE* file = fopen(fn, "wb");
    CheckNTErrors(file, "Cannot open the model file");

    TensorList params;

    GetParams(params);

    int metaInfo[]{
        encoder->nlayer, decoder->nlayer,
        encoder->fnns->hSize, encoder->selfAtt->d,
        encoder->embedder.eSize, encoder->embedder.vSize,
        decoder->embedder.vSize, encoder->selfAtt->nhead,
        encoder->selfAtt->maxRP, shareAllEmbeddings,
        shareDecInputOutputWeight, encoder->embedder.maxLength - 1 - 1,
    };

    /* part 1: hyper-parameters */
    fwrite(metaInfo, sizeof(int), sizeof(metaInfo) / sizeof(int), file);

    /* part 2: model parameters */
    for (int i = 0; i < params.Size(); i++) {
        params[i]->BinaryDump(file);
    }

    fclose(file);

    double elapsed = GetClockSec() - startT;

    LOG("model saved (took %.1fs)", elapsed);
}

/* read the parameters */
void Model::Read(FILE* file)
{
    double startT = GetClockSec();

    TensorList params;
    GetParams(params);
    LOG("params count: %ld", params.Size());
    int size = 0;
    for (int i = 0; i < params.Size(); i++) {
        size += params[i]->unitNum;
    }
    LOG("params size: %d", size);

    /* convert parameters to FP16 before reading files */
    if (useFP16) {
        LOG("Convert parameters to FP16");
        for (int i = 0; i < params.Size(); i++) {
            XTensor* p = params[i];
            InitTensor(p, p->order, p->dimSize, X_FLOAT16, p->devID, p->enableGrad && X_ENABLE_GRAD);
        }

        auto& encEmb = encoder->embedder.posEmbeddingBase;
        auto& decEmb = decoder->embedder.posEmbeddingBase;
        encEmb = ConvertDataType(encEmb, X_FLOAT16);
        decEmb = ConvertDataType(decEmb, X_FLOAT16);
    }

    for (int i = 0; i < params.Size(); i++)
        params[i]->BinaryRead(file);

    /* share all embeddings */
    if (shareAllEmbeddings == 1) {
        _CopyValues(&encoder->embedder.w, &decoder->embedder.w);
        LOG("sharing encoder decoder embeddings");
    }

    /* share embeddings with output weights */
    if (shareDecInputOutputWeight == 1) {
        _CopyValues(&decoder->embedder.w, &outputLayer->w);
        LOG("sharing decoder embeddings with output weights");
    }

    double elapsed = GetClockSec() - startT;
    LOG("model loaded (took %.1fs)", elapsed);
}

}