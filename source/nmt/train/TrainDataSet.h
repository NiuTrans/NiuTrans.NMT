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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2019-04-03
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#ifndef __TRAIN_DATASET_H__
#define __TRAIN_DATASET_H__

#include <cstdio>
#include <vector>
#include <fstream>

#include "../../niutensor/tensor/XList.h"
#include "../../niutensor/tensor/XTensor.h"
#include "../../niutensor/tensor/XGlobal.h"

using namespace std;

namespace nts {

/* a class of sentence pairs for training */
struct TrainExample {

    /* id of the sentence pair */
    int id;

    /* source language setence (tokenized) */
    IntList srcSent;

    /* target language setence (tokenized) */
    IntList tgtSent;

    /* the key used to shuffle items in a bucket */
    int key;

    /* the key used to shuffle buckets */
    int bucketKey;
};

/* A `TrainDataSet` is associated with a file which contains training data. */
struct TrainDataSet {

public:
    /* the data buffer */
    TrainBufferType buffer;

    /* a list of empty line number */
    IntList emptyLines;

    /* the pointer to file stream */
    FILE* fp;

    /* current index in the buffer */
    int curIdx;

    /* size of used data in the buffer */
    int bufferUsed;

    /* size of the bucket used for grouping sentences */
    int bucketSize;

    /* indicates whether it is used for training */
    bool isTraining;

    /* the padding id */
    int padID;

    /* the unk id */
    int unkID;

    /* start symbol */
    int startID;

    /* end symbol */
    int endID;

    /* the maximum length for a source sentence */
    int maxSrcLen;

    /* the maximum length for a target sentence */
    int maxTgtLen;

public:

    /* get the maximum source sentence length in a range */
    int MaxSrcLen(int begin, int end);

    /* get the maximum target sentence length in a range */
    int MaxTgtLen(int begin, int end);

    /* sort the input by source sentence length (in descending order) */
    void SortBySrcLength();

    /* sort the input by target sentence length (in descending order) */
    void SortByTgtLength();

    /* sort buckets by key (in descending order) */
    void SortBuckets();

    /* shuffle buckets */
    void ShuffleBuckets();

    /* sort the output by key (in descending order) */
    void SortInBucket(int begin, int end);

    /* load data from a file to the buffer */
    void LoadDataToBuffer();

    /* generate a mini-batch */
    UInt64List LoadBatch(XTensor* batchEnc, XTensor* paddingEnc,
                         XTensor* batchDec, XTensor* paddingDec, XTensor* label,
                         int minSentBatch, int batchSize, int devID);

    /* initialization function */
    void Init(const char* dataFile, int bucketSize, bool training);

    /* check if the buffer is empty */
    bool IsEmpty();

    /* reset the buffer */
    void ClearBuf();

    /* group data into buckets with similar length */
    void BuildBucket();

    /* de-constructor */
    ~TrainDataSet();
};
}

#endif // __TRAIN_DATASET_H__