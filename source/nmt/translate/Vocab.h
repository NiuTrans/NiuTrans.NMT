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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-01-03
 */

#ifndef __VOCAB_H__
#define __VOCAB_H__

#include <cstdio>
#include <unordered_map>

using namespace std;

namespace nts {

/* the vocabulary class */
struct Vocab
{
    /* the start id for words */
    int startID;

    /* size of the vocabulary */
    int vocabSize;

    /* a dict that maps words to ids */
    unordered_map<string, int> word2id;

    /* a dict that maps ids to words */
    unordered_map<int, string> id2word;

    /* load a vocabulary from a file */
    void Load(const string& src);

    /* save a vocabulary to a file */
    void Save(const string& src);

    /* copy data from another vocab */
    void CopyFrom(const Vocab& v);
};

}

#endif