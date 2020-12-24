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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * A week with no trips :)
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#ifndef __TESTER_H__
#define __TESTER_H__

#include "Search.h"
#include "TranslateDataSet.h"

namespace nmt
{

/* This class translates test sentences with a trained model. */
class Translator
{
public:
    /* vocabulary size of the source side */
    int vSize;

    /* vocabulary size of the target side */
    int vSizeTgt;

    /* batch size for sentences */
    int sentBatch;

    /* batch size for words */
    int wordBatch;

    /* beam size */
    int beamSize;

    /* for batching */
    DataSet batchLoader;

    /* decoder for inference */
    void* seacher;

public:
    /* constructor */
    Translator();

    /* de-constructor */
    ~Translator();

    /* initialize the model */
    void Init(Config& config);

    /* test the model */
    void Translate(const char* ifn, const char* vfn, const char* ofn, 
                   const char* tfn, Model* model);

    /* dump the result into the file */
    void Dump(FILE* file, XTensor* output);
};

}

#endif