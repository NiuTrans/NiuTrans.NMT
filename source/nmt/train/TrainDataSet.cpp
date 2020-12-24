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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-08-09
 * $Updated by: CAO Hang and Wu Siming 2020-12-13
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "TrainDataSet.h"
#include "../Utility.h"
#include "../translate/Vocab.h"

using namespace nmt;

namespace nts {

/* get the maximum source sentence length in a range */
int TrainDataSet::MaxSrcLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0) && (end <= buffer.count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        maxLen = MAX(buffer[i]->srcSent.Size(), maxLen);
    }
    return maxLen;
}

/* get the maximum target sentence length in a range */
int TrainDataSet::MaxTgtLen(int begin, int end) {
    CheckNTErrors(end > begin, "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        maxLen = MAX(buffer[i]->tgtSent.Size(), maxLen);
    }
    return maxLen;
}

/* sort the dataset by source sentence length (in descending order) */
void TrainDataSet::SortBySrcLength() {
    stable_sort(buffer.items, buffer.items + buffer.count,
                [](TrainExample* a, TrainExample* b) {
                    return (a->srcSent.Size())
                         < (b->srcSent.Size());
                });
}

/* sort the dataset by target sentence length (in descending order) */
void TrainDataSet::SortByTgtLength() {
    stable_sort(buffer.items, buffer.items + buffer.count,
                [](TrainExample* a, TrainExample* b) {
                    return (a->tgtSent.Size())
                         < (b->tgtSent.Size());
                });
}

/* sort buckets by key (in descending order) */
void TrainDataSet::SortBuckets() {
    stable_sort(buffer.items, buffer.items + buffer.count,
        [](TrainExample* a, TrainExample* b) {
            return a->bucketKey < b->bucketKey;
        });
}

/* shuffle buckets */
void TrainDataSet::ShuffleBuckets()
{
    /* assign random keys for different buckets */
    int i = 0;
    int key = 0;

    while (i < buffer.count - 1) {

        /* determine the range of a bucket */
        key = buffer[i]->bucketKey;
        int cur = i;
        while ((buffer[i]->bucketKey == key) && (i < (buffer.count - 1))) {
            i++;
        }

        /* update the bucket key */
        key = rand();
        while (cur < i) {
            buffer[cur++]->bucketKey = key;
        }
    }

    SortBuckets();
}

/*
sort the output by key in a range (in descending order)
>> begin - the first index of the range
>> end - the last index of the range
*/
void TrainDataSet::SortInBucket(int begin, int end) {
    sort(buffer.items + begin, buffer.items + end,
        [](TrainExample* a, TrainExample* b) {
            return (a->key < b->key);
        });
}

/*
load all data from a file to the buffer
training data format (binary):
first 8 bit: number of sentence pairs
subsequent segements:
source sentence length (4 bit)
target sentence length (4 bit)
source tokens (4 bit per token)
target tokens (4 bit per token)
*/
void TrainDataSet::LoadDataToBuffer()
{
    buffer.Clear();
    curIdx = 0;

    int id = 0;
    uint64_t sentNum = 0;

    int srcVocabSize = 0;
    int tgtVocabSize = 0;
    fread(&srcVocabSize, sizeof(srcVocabSize), 1, fp);
    fread(&tgtVocabSize, sizeof(tgtVocabSize), 1, fp);

    fread(&sentNum, sizeof(uint64_t), 1, fp);
    CheckNTErrors(sentNum > 0, "Invalid sentence pairs number");

    while (id < sentNum) {
        int srcLen = 0;
        int tgtLen = 0;
        fread(&srcLen, sizeof(int), 1, fp);
        fread(&tgtLen, sizeof(int), 1, fp);
        CheckNTErrors(srcLen > 0, "Invalid source sentence length");
        CheckNTErrors(tgtLen > 0, "Invalid target sentence length");

        IntList srcSent;
        IntList tgtSent;
        srcSent.ReadFromFile(fp, srcLen);
        tgtSent.ReadFromFile(fp, tgtLen);

        /* reserve the first maxSrcLen words if the source is too long */
        if (srcLen > maxSrcLen) {
            for (int i = maxSrcLen; i < srcLen; i++)
                srcSent.Remove(maxSrcLen);
        }

        /* reserve the first maxTgtLen words if the target is too long */
        if (tgtLen > maxTgtLen) {
            for (int i = maxTgtLen; i < tgtLen; i++)
                srcSent.Remove(maxTgtLen);
        }

        TrainExample* example = new TrainExample;
        example->id = id++;
        example->key = MAX(srcSent.count, tgtSent.count);
        example->srcSent = srcSent;
        example->tgtSent = tgtSent;

        buffer.Add(example);
    }

    fclose(fp);

    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", id);
}

/*
load a mini-batch to a device (for training)
>> batchEnc - a tensor to store the batch of encoder input
>> paddingEnc - a tensor to store the batch of encoder paddings
>> batchDec - a tensor to store the batch of decoder input
>> paddingDec - a tensor to store the batch of decoder paddings
>> label - a tensor to store the label of input
>> minSentBatch - the minimum number of sentence batch
>> batchSize - the maxium number of words in a batch
>> devID - the device id, -1 for the CPU
<< return - number of target tokens and sentences
*/
UInt64List TrainDataSet::LoadBatch(XTensor* batchEnc, XTensor* paddingEnc,
                                   XTensor* batchDec, XTensor* paddingDec, XTensor* label,
                                   int minSentBatch, int batchSize, int devID)
{
    UInt64List info;
    int srcTokenNum = 0;
    int tgtTokenNum = 0;
    int realBatchSize = 0;

    /* use a fixed batch size for validation */
    if (!isTraining)
        realBatchSize = minSentBatch;

    /* dynamic batching for sentences, enabled when the dataset is used for training */
    if (isTraining) {
        int bucketKey = buffer[curIdx]->bucketKey;
        while ((realBatchSize < (buffer.Size() - curIdx)) &&
            (buffer[curIdx + realBatchSize]->bucketKey == bucketKey)) {
            realBatchSize++;
        }
    }

    realBatchSize = MIN(realBatchSize, (buffer.Size() - curIdx));
    CheckNTErrors(realBatchSize > 0, "Invalid batch size");

    /* get the maximum target sentence length in a mini-batch */
    int maxSrcLen = MaxSrcLen(curIdx, curIdx + realBatchSize);
    int maxTgtLen = MaxTgtLen(curIdx, curIdx + realBatchSize);

    CheckNTErrors(maxSrcLen > 0, "Invalid source length for batching");
    CheckNTErrors(maxTgtLen > 0, "Invalid target length for batching");

    int* batchEncValues = new int[realBatchSize * maxSrcLen];
    float* paddingEncValues = new float[realBatchSize * maxSrcLen];

    int* labelVaues = new int[realBatchSize * maxTgtLen];
    int* batchDecValues = new int[realBatchSize * maxTgtLen];
    float* paddingDecValues = new float[realBatchSize * maxTgtLen];

    for (int i = 0; i < realBatchSize * maxSrcLen; i++) {
        batchEncValues[i] = padID;
        paddingEncValues[i] = 1.0F;
    }
    for (int i = 0; i < realBatchSize * maxTgtLen; i++) {
        batchDecValues[i] = padID;
        labelVaues[i] = padID;
        paddingDecValues[i] = 1.0F;
    }

    int curSrc = 0;
    int curTgt = 0;

    /*
    batchEnc: end with EOS (left padding)
    batchDec: begin with SOS (right padding)
    label:    end with EOS (right padding)
    */
    for (int i = 0; i < realBatchSize; ++i) {

        srcTokenNum += buffer[curIdx + i]->srcSent.Size();
        tgtTokenNum += buffer[curIdx + i]->tgtSent.Size();

        curSrc = maxSrcLen * i;
        for (int j = 0; j < buffer[curIdx + i]->srcSent.Size(); j++) {
            batchEncValues[curSrc++] = buffer[curIdx + i]->srcSent[j];
        }

        curTgt = maxTgtLen * i;
        for (int j = 0; j < buffer[curIdx + i]->tgtSent.Size(); j++) {
            if (j > 0)
                labelVaues[curTgt - 1] = buffer[curIdx + i]->tgtSent[j];
            batchDecValues[curTgt++] = buffer[curIdx + i]->tgtSent[j];
        }
        labelVaues[curTgt - 1] = endID;
        while (curSrc < maxSrcLen * (i + 1))
            paddingEncValues[curSrc++] = 0;
        while (curTgt < maxTgtLen * (i + 1))
            paddingDecValues[curTgt++] = 0;
    }

    InitTensor2D(batchEnc, realBatchSize, maxSrcLen, X_INT, devID);
    InitTensor2D(paddingEnc, realBatchSize, maxSrcLen, X_FLOAT, devID);
    InitTensor2D(batchDec, realBatchSize, maxTgtLen, X_INT, devID);
    InitTensor2D(paddingDec, realBatchSize, maxTgtLen, X_FLOAT, devID);
    InitTensor2D(label, realBatchSize, maxTgtLen, X_INT, devID);

    curIdx += realBatchSize;

    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingEncValues, paddingEnc->unitNum);
    batchDec->SetData(batchDecValues, batchDec->unitNum);
    paddingDec->SetData(paddingDecValues, paddingDec->unitNum);
    label->SetData(labelVaues, label->unitNum);

    delete[] batchEncValues;
    delete[] paddingEncValues;
    delete[] batchDecValues;
    delete[] paddingDecValues;
    delete[] labelVaues;

    info.Add(tgtTokenNum);
    info.Add(realBatchSize);
    return info;
}

/*
the constructor of DataSet
>> dataFile - path of the data file
>> bucketSize - size of the bucket to keep similar length sentence pairs
>> training - indicates whether it is used for training
*/
void TrainDataSet::Init(const char* dataFile, int myBucketSize, bool training)
{
    fp = fopen(dataFile, "rb");
    CheckNTErrors(fp, "can not open the training file");
    curIdx = 0;
    bucketSize = myBucketSize;
    isTraining = training;

    LoadDataToBuffer();

    SortByTgtLength();

    SortBySrcLength();

    if (isTraining) {
        BuildBucket();
    }

}

/* check if the buffer is empty */
bool TrainDataSet::IsEmpty() {
    if (curIdx < buffer.Size())
        return false;
    return true;
}

/* reset the buffer */
void TrainDataSet::ClearBuf()
{
    curIdx = 0;

    /* make different batches in different epochs */
    if (isTraining)
        ShuffleBuckets();
}

/* group data with similar length into buckets */
void TrainDataSet::BuildBucket()
{
    int idx = 0;

    IntList list;

    /* build buckets by the length of source and target sentence */
    while (idx < buffer.Size()) {

        /* sentence number in a bucket */
        int sentNum = 1;

        /* get the maximum source sentence length in a bucket */
        int maxSrcLen = MaxSrcLen(idx, idx + sentNum);
        int maxTgtLen = MaxTgtLen(idx, idx + sentNum);
        int maxLen = MAX(maxSrcLen, maxTgtLen);

        /* max sentence number in a bucket */
        const int MAX_SENT_NUM = 5120;

        while ((sentNum < (buffer.count - idx))
            && (sentNum < MAX_SENT_NUM)
            && (sentNum * maxLen <= bucketSize)) {
            sentNum++;
            maxSrcLen = MaxSrcLen(idx, idx + sentNum);
            maxTgtLen = MaxTgtLen(idx, idx + sentNum);
            maxLen = MAX(maxSrcLen, maxTgtLen);
        }

        /* make sure the number is valid */
        if ((sentNum)*maxLen > bucketSize || sentNum >= MAX_SENT_NUM) {
            sentNum--;
            sentNum = max(8 * (sentNum / 8), sentNum % 8);
        }
        if ((buffer.Size() - idx) < sentNum)
            sentNum = buffer.Size() - idx;

        /* assign the same key for items in a bucket */
        int randomKey = rand();
        for (int i = 0; i < sentNum; i++) {
            buffer[idx + i]->bucketKey = randomKey;
        }

        idx += sentNum;
        list.Add(sentNum);
    }

    LOG("number of batches: %d", list.count);

    ShuffleBuckets();
}

/* de-constructor */
TrainDataSet::~TrainDataSet()
{

    /* release the buffer */
    for (int i = 0; i < buffer.Size(); i++)
        delete buffer[i];
}

}
