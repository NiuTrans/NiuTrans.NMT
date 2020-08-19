/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University. 
 * All rights reserved.
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
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-01-20
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include "XGlobal.h"

#if !defined( WIN32 ) && !defined( _WIN32 )
    #include "sys/time.h"
    #include "time.h"
    #include "iconv.h"
#else
    #include "time.h"
    #include "windows.h"
    #include "process.h"
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* memory pool setting */
int MAX_MEM_BLOCK_NUM = 1024;
int MAX_MEM_BLOCK_SIZE = 1024 * 1024 * 256;
int MIN_MEM_BLOCK_SIZE = 1024 * 1024 * 64;
int MINOR_MEM_BLOCK_SIZE = 1024 * 1024 * 256;
int MAX_MEM_BUF_SIZE = 1024 * 1024 * 256;
int MIN_MEM_BUF_SIZE = 1024 * 1024 * 32;
int TRAINING_SAMPLE_BUF_SIZE = 1024 * 1024 * 16;
int CONST_MINUSONE = -1;
bool CONST_TRUE = true;

int verboseLevel = 0;

FILE * tmpLog = NULL;
double myTime = 0;
double myTime2 = 0;
double myTime3 = 0;
double myTime4 = 0;
double myTime5 = 0;
double myTime6 = 0;
double myTime7 = 0;
double myTime8 = 0;
double myTime9 = 0;
double myTimeForward1 = 0;
double myTimeForward2 = 0;
double myTimeForward3 = 0;
double myTimeBackward1 = 0;
double myTimeBackward2 = 0;
double myTimeBackward3 = 0;
double myTimeBackward4 = 0;

int dEdWCount = 0;
FILE * tF;

/* initialization of the global stuff */
void InitGlobalAll()
{
    srand((unsigned int)time(NULL));
}

#ifndef WIN32

/* print calling stack for debug in linux */
void PrintCallStack() {
    int j, nptrs;
#define SIZE 100
    void* buffer[100];
    char** strings;

    nptrs = backtrace(buffer, SIZE);
    printf("backtrace() returned %d addresses\n", nptrs);

    /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
        would produce similar output to the following: */

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < nptrs; j++)
        printf("%s\n", strings[j]);

    free(strings);
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

void print_trace() {
    char pid_buf[30];
    sprintf(pid_buf, "%d", getpid());
    char name_buf[512];
    name_buf[readlink("/proc/self/exe", name_buf, 511)] = 0;
    int child_pid = fork();
    if (!child_pid) {
        dup2(2, 1); // redirect output to stderr
        fprintf(stdout, "stack trace for %s pid=%s\n", name_buf, pid_buf);
        execlp("gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
        abort(); /* If gdb failed to start */
    }
    else {
        waitpid(child_pid, NULL, 0);
    }
}

#endif // !WIN32

} /* end of the nts (NiuTrans.Tensor) namespace */
