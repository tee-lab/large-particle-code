#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <cuda_runtime.h>
#include <GL/freeglut.h>

#include "globals.h"

// k = 0 for display calls, = 1 for kernel calls
void calcFPS(int k);

// GL functionality
bool initGL(int *argc, char** argv, SimParams &s);

// rendering callbacks
void display();
void timerEvent(int value);
void keyPress(unsigned char key, int x, int y);
void reshape(int w, int h);
//void mousePress(int button, int state, int x, int y);
//void mouseMove(int x, int y);

int executeCommand(string cmd);


#endif
