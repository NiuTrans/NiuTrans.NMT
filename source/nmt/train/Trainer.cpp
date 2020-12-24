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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-02
 */

#include "Trainer.h"
#include "../Utility.h"
#include "../../niutensor/network/XNoder.h"
#include "../../niutensor/tensor/XUtility.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/loss/LHeader.h"


#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif
#include "../../niutensor/tensor/XMem.h"

namespace nmt
{

/* constructor */
Trainer::Trainer()
{
    nwarmup = -1;
    cfg = NULL;
}

/* de-constructor */
Trainer::~Trainer()
{
    for (int i = 0; i < moments.count; i++) {
        XTensor* m = (XTensor*)moments.Get(i);
        delete m;
    }

    for (int i = 0; i < moments2nd.count; i++) {
        XTensor* m = (XTensor*)moments2nd.Get(i);
        delete m;
    }
}

/*
initialization
>> config - configurations of the training process
*/
void Trainer::Init(Config& config)
{
    cfg = &config;
    
    lrate = config.lrate;
    lrbias = config.lrbias;
    sBatchSize = config.sBatchSize;
    wBatchSize = config.wBatchSize;
    bucketSize = config.bucketSize;
    nstep = config.nstep;
    nepoch = config.nepoch;
    logInterval = config.logInterval;
    maxCheckpoint = config.maxCheckpoint;
    d = config.modelSize;
    nwarmup = config.nwarmup;
    vSize = config.srcVocabSize;
    vSizeTgt = config.tgtVocabSize;
    useAdam = config.useAdam;
    adamBeta1 = config.adamBeta1;
    adamBeta2 = config.adamBeta2;
    adamDelta = config.adamDelta;
    isShuffled = config.isShuffled;
    labelSmoothingP = config.labelSmoothingP;
    nStepCheckpoint = config.nStepCheckpoint;
    useEpochCheckpoint = config.useEpochCheckpoint;
    updateStep = config.updateStep;
    isLenSorted = config.isLenSorted;

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;

    batchLoader.startID = config.startID;
    batchLoader.endID = config.endID;
    batchLoader.unkID = config.unkID;
    batchLoader.padID = config.padID;
    batchLoader.maxSrcLen = config.maxSrcLen;
    batchLoader.maxTgtLen = config.maxTgtLen;
}

/*
train the model
>> fn - training data file
>> validFN - validation data file
>> modelFN - where we keep the model
>> model - model to train
*/
void Trainer::Train(const char* fn, const char* validFN, 
                    const char* modelFN, Model* model)
{
    /* disable cache during training */
    for (int i = 0; i < model->decoder->nlayer; i++) {
        model->decoder->selfAttCache[i].enable = false;
        model->decoder->enDeAttCache[i].enable = false;
    }
    int step = 0;
    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int batchCountTotal = 0;
    bool isEnd = false;
    float loss = 0;
    float lr = 0;
    int nStepCheck = 0;
    int nCheckpoint = 0;
    int nSkipped = 0;
    int gradStep = 0;
    int validStep = 0;
    int epoch = 0;

    char* trainFN = new char[(int)strlen(fn) + 10];
    strcpy(trainFN, fn);

#ifndef WIN32
    if (isShuffled)
        sprintf(trainFN, "%s.random", fn);
#endif

    int devID = model->devID;

    PrepareModel(model);

    double startT = GetClockSec();

    batchLoader.Init(fn, bucketSize, true);

    for (epoch = 1; epoch <= nepoch; epoch++) {

        wordCount = 0;
        loss = 0;

        /* reset the batch loader */
        batchLoader.ClearBuf();

        while (!batchLoader.IsEmpty())
        {
            XNet net;
            net.Clear();

            /* batch of sequences (on the encoder and decoder sides) */
            XTensor batchEnc;
            XTensor batchDec;

            /* labels */
            XTensor label;

            /* padding */
            XTensor paddingEnc;
            XTensor paddingDec;

            UInt64List info = batchLoader.LoadBatch(&batchEnc, &paddingEnc, &batchDec, &paddingDec, &label, 
                                                    sBatchSize, wBatchSize, devID);

            wc = info[0];
            ws = info[1];
            CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");

            /* output probabilities */
            XTensor output;

            /* make the network */
            if (model->isLM) {
                model->MakeLM(batchEnc, output, paddingEnc, true);
            }
            else if (model->isMT) {
                model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, true);
            }
            else {
                ShowNTErrors("Illegal model type!");
            }

            /* get loss and probabilities */
            XTensor labelOnehot;
            XTensor lossTensor;

            labelOnehot = IndexToOnehot(label, vSizeTgt, labelSmoothingP);

            lossTensor = CrossEntropy(output, labelOnehot, paddingDec);

            float lossBatch = ReduceSumAllValue(lossTensor);

            DTYPE lossLocal = lossBatch / wc;
            bool doUpdate = (!IsNAN(lossLocal) && !IsINF(lossLocal) && lossLocal < 1e3F);

            net.isGradEfficient = true;

            bool debug(false);
            if (debug) {
                LOG("after forward:");
                batchEnc.mem->ShowMemUsage(stderr);
                exit(0);
            }

            if (doUpdate) {

                net.Backward(lossTensor);

                if (model->encoder->useHistory)
                    model->encoder->history->ClearHistory(/*reset=*/false);
                if (model->decoder->useHistory)
                    model->decoder->history->ClearHistory(/*reset=*/false);

                gradStep += 1;
                loss += lossBatch;
                wordCount += wc;
                wordCountTotal += wc;
                batchCountTotal += ws;

                /* update the parameters */
                if (gradStep == updateStep) {

                    float warmupEndLR = lrate;
                    float warmupInitLR = 1e-7;
                    float lrStep = (warmupEndLR - warmupInitLR) / nwarmup;
                    float decayFactor = warmupEndLR * pow(float(nwarmup), 0.5F);

                    /* learning rate, scheduled by inverse square root */
                    if (step < nwarmup)
                        lr = warmupInitLR + step * lrStep;
                    else
                        lr = decayFactor * pow((float)step, -0.5F);


                    /* model update */
                    Update(model, lr);

                    gradStep = 0;
                    validStep++;
                }
            }
            else
                nSkipped++;

            if (++step >= nstep) {
                isEnd = true;
                break;
            }

            if (step % logInterval == 0) {
                double elapsed = GetClockSec() - startT;
                LOG("elapsed=%.1fs, step=%d, epoch=%d, "
                    "total word=%d, total batch=%d, loss=%.3f, ppl=%.3f, lr=%.2e", 
                    elapsed, step, epoch, wordCountTotal, batchCountTotal,
                    loss / wordCount / log(2.0), exp(loss / wordCount), lr);
                
                if (!doUpdate)
                    XPRINT(0, stderr, " (no update)");
            }

            if (nStepCheckpoint > 0 && ++nStepCheck >= nStepCheckpoint) {
                MakeCheckpoint(model, validFN, modelFN, "step", step);
                nStepCheck = 0;
                nCheckpoint++;
            }
        }

        if (isEnd)
            break;

        if (useEpochCheckpoint)
            MakeCheckpoint(model, validFN, modelFN, "epoch", epoch);
    }

    double elapsed = GetClockSec() - startT;

    epoch = MIN(epoch, nepoch);

    LOG("lr=%.2e, elapsed=%.1fs, step=%d, "
        "epoch=%d, word=%d, loss=%.3f, ppl=%.3f",
        lr, elapsed, step, epoch, wordCountTotal, loss / wordCount / log(2.0), exp(loss / wordCount));
    LOG("training finished (took %.1fs, step=%d, "
        "skipped=%d and epoch=%d)", elapsed, step, nSkipped, epoch);

    LOG("saving the final model");
    model->Dump(modelFN);

    delete[] trainFN;
}

/*
test the model
>> fn - test data file
>> ofn - output data file
>> model - model that is trained
*/
void Trainer::Validate(const char* fn, const char* ofn, Model* model)
{
    DISABLE_GRAD;

    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int sentCount = 0;
    float loss = 0;

    /* data files */
    batchLoader.Init(fn, 0, false);

    double startT = GetClockSec();

    while (!batchLoader.IsEmpty())
    {
        /* batch of input sequences */
        XTensor batchEnc;
        XTensor batchDec;

        /* label */
        XTensor label;

        /* padding */
        XTensor paddingEnc;
        XTensor paddingDec;

        /* output probabilities */
        XTensor output;

        /* prediction probabilities */
        XTensor labelOnehot;
        XTensor lossTensor;

        UInt64List info = batchLoader.LoadBatch(&batchEnc, &paddingEnc, &batchDec, &paddingDec, &label, 
                                                sBatchSize, 0, model->devID);
        wc = info[0];
        ws = info[1];
        CheckNTErrors(batchEnc.order == 2, "Wrong tensor order of the sequence batch");

        /* make the network */
        if (model->isLM)
            model->MakeLM(batchEnc, output, paddingEnc, false);
        else if (model->isMT)
            model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, false);
        else {
            ShowNTErrors("Illegal model type!");
        }

        int bSize = output.GetDim(0);
        int length = output.GetDim(1);

        labelOnehot = IndexToOnehot(label, vSizeTgt, 0);

        lossTensor = CrossEntropy(output, labelOnehot, paddingDec);

        float lossBatch = ReduceSumAllValue(lossTensor);

        loss += lossBatch;

        wordCount += wc;
        sentCount += bSize;
    }

    double elapsed = GetClockSec() - startT;

    ENABLE_GRAD;

    LOG("validating finished (took %.1fs, sentence=%d, word=%d, loss=%.3f and ppl=%.3f)",
        elapsed, sentCount, wordCount, loss / wordCount / log(2.0), exp(loss / wordCount));
}

/*
make a checkpoint
>> model - the model
>> validFN - validation data file
>> modelFN - model data file
>> label - label of the model
>> id - id of the checkpoint
*/
void Trainer::MakeCheckpoint(Model* model, const char* validFN, 
                             const char* modelFN, const char* label, int id)
{
    LOG("make a checkpoint");
    char* fn = new char[MAX_LINE_LENGTH];

    Trainer validator;
    validator.Init(*cfg);
    
    /* save last checkpoints */
    id = validator.maxCheckpoint - (maxCheckpoint--);
    if (maxCheckpoint == 0)
        maxCheckpoint = validator.maxCheckpoint;
    sprintf(fn, "%s.%s.%03d", modelFN, label, id);

    model->Dump(fn);
    delete[] fn;

    char* fn2 = new char[MAX_LINE_LENGTH];
    sprintf(fn2, "%s.%s.%03d.output", modelFN, label, id);
    if (validFN != NULL) {
        
        validator.Validate(validFN, fn2, model);
    }
    delete[] fn2;
}

/*
update the model by delta rule
\theta_{new} = \theta - \lrate * grad
where
\lrate = d^-0.5 * min(stepNum^{-0.5}, stepNum * warmupStepNum^{-1.5})
>> model - the  model
>> lr - learning rate
*/
void Trainer::Update(Model* model, const float lr)
{
    TensorList ws;

    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XTensor* paraGrad = para->grad;

        if (paraGrad == NULL)
            continue;

        CheckNTErrors(para != NULL, "NULL parameter tensor!");
        CheckNTErrors(paraGrad != NULL, "NULL gradient tensor!");

        if (useAdam) {
            adamBeta1T *= adamBeta1;
            adamBeta2T *= adamBeta2;
            DTYPE e = lr * (DTYPE)sqrt(1 - adamBeta2T) / (1 - adamBeta1T);
            DTYPE d = adamDelta * (DTYPE)sqrt(1 - adamBeta2T);

            /* m = beta_1 * m + (1-beta_1) * grad */
            XTensor* m = (XTensor*)moments.Get(i);
            _ScaleAndShiftMe(m, adamBeta1, 0);
            _Sum(m, paraGrad, m, (1.0F - adamBeta1));

            /* v = beta_2 * v + (1-beta_2) * grad * grad*/
            XTensor* v = (XTensor*)moments2nd.Get(i);
            _Multiply(paraGrad, paraGrad, v, adamBeta2 / (1.0F - adamBeta2));
            _ScaleAndShiftMe(v, (1.0F - adamBeta2), 0);

            /* v2 = m / (sqrt(v) + delta) */
            XTensor* v2 = NewTensorBufV2(v, v->devID, v->mem);
            _Power(v, v2, 0.5F);
            _ScaleAndShiftMe(v2, 1.0F, d);
            _Div(m, v2, v2);

            /* the delta rule */
            _Sum(para, v2, para, -e);

            DelTensorBuf(v2);
        }
        else {
            /* the delta rule */
            _Sum(para, paraGrad, para, -lr);
        }

        /* clear gradient */
        paraGrad->SetZeroAll();
    }
}

/*
prepare model for training
>> model - the model for training
*/
void Trainer::PrepareModel(Model* model)
{
    moments.Clear();
    moments2nd.Clear();

    TensorList ws;

    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XNoder::MakeGrad(para);

        if (useAdam) {
            XTensor* m = new XTensor(para);
            XTensor* m2 = new XTensor(para);
            m->SetZeroAll();
            m2->SetZeroAll();
            moments.Add(m);
            moments2nd.Add(m2);
        }
    }

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

}