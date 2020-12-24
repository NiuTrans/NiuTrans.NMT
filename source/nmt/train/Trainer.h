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

#ifndef __TRAINER_H__
#define __TRAINER_H__

#include "../Model.h"
#include "TrainDataSet.h"

using namespace nts;

namespace nmt
{

/* trainer of the  model */
class Trainer
{
public:

    /* configurations */
    Config* cfg;

    /* dimension size of each inner layer */
    int d;

    /* step number of warm-up for training */
    int nwarmup;

    /* vocabulary size of the source side */
    int vSize;

    /* vocabulary size of the target side */
    int vSizeTgt;

    /* learning rate */
    float lrate;

    /* the parameter that controls the maximum learning rate in training */
    float lrbias;

    /* sentence batch size */
    int sBatchSize;

    /* word batch size */
    int wBatchSize;

    /* size of bucket for grouping data by length */
    int bucketSize;

    /* training epoch number */
    int nepoch;

    /* traing step number */
    int nstep;

    /* interval step for logging */
    int logInterval;

    /* the maximum number of saved checkpoints */
    int maxCheckpoint;

    /* indicates whether we use adam */
    bool useAdam;

    /* hyper parameters of adam*/
    float adamBeta1;
    float adamBeta2;
    float adamDelta;
    float adamBeta1T;
    float adamBeta2T;

    /* list of the moment of the parameter matrices */
    TensorList moments;

    /* list of the 2nd order moment of the parameter matrices */
    TensorList moments2nd;

    /* indicates whether the data file is shuffled for training */
    bool isShuffled;

    /* the factor of label smoothing */
    DTYPE labelSmoothingP;

    /* number of steps after which we make a checkpoint */
    int nStepCheckpoint;

    /* indicates whether we make a checkpoint after each training epoch */
    bool useEpochCheckpoint;

    /* number of batches on which we do model update */
    int updateStep;

    /* indicates whether the sequence is sorted by length */
    bool isLenSorted;

    /* used for loading batches */
    TrainDataSet batchLoader;

public:
    /* constructor */
    Trainer();

    /* de-constructor */
    ~Trainer();

    /* initialize the trainer */
    void Init(Config& config);

    /* train the model */
    void Train(const char* fn, const char* validFN, const char* modelFN, Model* model);

    /* test the model */
    void Validate(const char* fn, const char* ofn, Model* model);

    /* make a checkpoint */
    void MakeCheckpoint(Model* model, const char* validFN, const char* modelFN, const char* label, int id);

    /* update the model by delta rule */
    void Update(Model* model, const float lr);

    /* prepare model for training */
    void PrepareModel(Model* model);
};

}

#endif
