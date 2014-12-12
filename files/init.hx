#ifndef _INIT_H
#define _INIT_H

#include "globals.h"

// ============================ INIT =========================================//

//int initSimParams_default(SimParams & s);
//int setSimParams_RsErrSd(SimParams &s, float z_Rs, float z_eSD);
//int setSimParams_trMax(SimParams &s, float z_trD);
//int setSimParams_kAO(SimParams &s, float z_kA, float z_kO);

int read_execution_config_file(string filename);

int allocArrays();
void freeArrays();

int initStateArrays();


#endif
