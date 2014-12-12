#ifndef ALTRUISM_H
#define ALTRUISM_H

// includes, system

#include "globals.h"

// =================== GRAPHICS DECLARATIONS =================================//


// ============================ MAIN SIMULATION FUNCTIONS  ===================//

// run these functions in this order ONLY.

int calcFitness();
void writeState();
int select_reproduce();

int advanceGen();

//int runCudaAnimate();
//int mainLoop();
//int launchExpt();


#endif
