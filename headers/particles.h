#ifndef PARTICLES_H
#define PARTICLES_H

#include <string>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "../utils/simple_initializer.h"
#include "../utils/simple_timer.h"


enum Strategy {Cooperate = 1, Defect = 0};

// Definition of class particle and associated functions. This can function as an independent library
// variables can be added to this class

// All types beginning wA in class Particle MUST BE 4 bytes
class Particle{
	public:
	float2 pos;		// position
	float2 vel;		// velocity

	// all types after this point MUST BE 4 bytes
	int	   wA;		// Strategy
	float  Rs;		// radius of flocking interaction
	float  kA;		// k attraction
	float  kO;		// k alignment
	int gID;		// group ID
	int ng;			// group size
	int kg;			// number of cooperators in group
	int ancID;		// original ancestor from which this individual descends
	float fitness;	// fitness
	
	Particle(){
		pos = vel = make_float2(0,0);
		wA = Rs = kA = kO = 0;
		gID = ng = kg =  ancID = 0;
	}
};


// Struct of movement parameters
// Must use struct for C compatibility in kernel functions
struct Movementparams{
	// time step
	float dt;

	// particle constants
	float Rr;
	float Rs;
	float kA;
	float kO;
	float speed;
	float copyErrSd;
	float turnRateMax;
	float cosphi;   // cos(P_turnRateMax*dt) !!
		
	// boundaries
	float xmin; 
	float xmax;
	float ymin; 
	float ymax;

	// number of particles
	int N;
	float cellSize;
	int nCellsX;
	int nCellsXY;
};



// Particle system class
// Particle system allows for 2 types of particles, (e.g. cooperators, defectors)

class ParticleSystem{
	public:
	string name;

	int N, K;
	vector <Particle> pvec;

	float dx, dy;
	
	map <int, int> g2ng_map, g2kg_map;
	float r, r2; 

	Movementparams par;

	int igen, istep;	// counters for generation number and step number
	int nStepsLifetime;
	int genMax;
	
	int K0; 	// initial value of K
	
	bool b_anim_on;					// is simulation update on?
	SimpleCounter kernelCounter;	// counter to count kernel calls/sec 
	
	// GPU stuff
	float2 *pos_dev, *pos_new_dev;
	float2 *vel_dev, *vel_new_dev;
	float  *Rs_dev;
	dim3 blockDims, gridDims;
	
	// grid arrays
	int * cellIds_dev; 			// cellId[i] = cell ID of particle i
	int * cellCounts_dev;		// cellCounts[i] = number of particles in cell i
	int * cellParticles_dev; 	// cellParticles[i,i+1,i+2,i+3] = indices of particles in cell i (max 4)
	
	public:
//	ParticleSystem();
	
	void  printParticles(int n = 5);
	void  summarize();

	void  init(Initializer &I);		// init particle ans particle system params
	cudaError __host__ copyParamsToDevice();

	void  launch_movement_kernel();
	int   step();
	int   animate();
	int   launchSim();

	void  updateGroupIndices_fast(float rGrp = -1);
	void  updateGroupIndices(float rGrp = -1);
	int   updateGroupSizes();
	float update_r();
	int   updateGroups();
	int   disperse(int R = -1);

};



#endif


