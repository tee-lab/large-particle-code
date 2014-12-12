#include <vector>
#include <cmath>
using namespace std;

#include "altruism.h"

#include "utils/simple_utils.h"
#include "utils/cuda_vector_math.cuh"
#include "utils/cuda_device.h"
#include "globals.h"
#include "graphics.h"


//// output all arrays to file
void writeState(){
	if (!dataOut) return;
	
	// calc avg fitness and avg Rs of A and D
	float fitA_avg = 0, fitD_avg = 0, rsA_avg = 0, rsD_avg = 0;
	for (int i= 0; i < nFish; ++i) {
		Particle &p = animals[i];
		if (p.wA == Cooperate) {
			fitA_avg += p.fitness;
			rsA_avg += p.Rs;
		}
		else{
			fitD_avg += p.fitness;
			rsD_avg += p.Rs;
		}
	}	
	fitA_avg /= (nCoop + 1e-6);
	fitD_avg /= (nFish - nCoop + 1e-6);
	rsA_avg /= (nCoop + 1e-6);
	rsD_avg /= (nFish - nCoop + 1e-6);

	// print p and related measures
	p_fout[0]   << 0 << "\t"
				<< pbar << "\t"
				<< EpgNg << "\t"
				<< varPg << "\t"
				<< r << "\t"
				<< Skg2bNg << "\t"
				<< SkgbNg << "\t"
				<< r2 << "\t"
				<< fitA_avg << "\t"
				<< fitD_avg << "\t"
				<< rsA_avg << "\t"
				<< rsD_avg << "\t"
				<< endl;

	return;
}




// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Use the group sizes and wA to calculate fitness for each individual
// Relies on : g2ng_map, g2kg_map, groupIndices
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//float pMinFitG = 0;
int calcFitness(){

	vector <float> fitness(nFish);	// array of absolute fitnesses
	
	// calc fitness (group wise) and population minimum fitness
	float pop_min_fit = b+1e6;  // NOTE: init with b+100 as fitness calc below will never exceed b 
	for (int i=0; i<nFish; ++i) {

		Particle *p = &animals[i];
		if (p->ng == 1){
			if ( p->wA == Cooperate){
				p->fitness = -c;
			}
			else{
				p->fitness = 0;
			}
		}
		else{
			if ( p->wA == Cooperate){	// individual is cooperator
				p->fitness = (p->kg-1)*b/(p->ng-1) - c;
			}
			else {	// individual is defector
				p->fitness = p->kg*b/(p->ng-1);
			}
		}
		p->fitness -= cS*(p->Rs*p->Rs);	// additional flocking cost

		pop_min_fit = fmin(pop_min_fit, p->fitness);
	}

	// shift the fitnesses such that least fitness is = fitness_base
	for (int i=0; i<nFish; ++i){
		Particle *p = &animals[i];
		p->fitness = p->fitness-pop_min_fit + fitness_base;
	}
	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// use fitness values to choose which individuals reproduce and 
// init positions/velocities/wA/wS for next generation
// relies on: fitnesses

//   0  1     2       3        4       5  6 7   8 ...  nFish-1			<-- fish number
// |--|---|--------|----|------------|---|-|-|--- ... ---------|		<-- fitnesses mapped onto 0-1
// 0            ^                                 ...          1		<-- range_selector
// selected fish = 3 (in this case)
	
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int select_reproduce(){

	// normalize fitness to get sum 1
	float sum_fit = 0;
	for (int i=0; i<nFish; ++i){
		sum_fit += animals[i].fitness;
	}
	
	// create ranges for random numbers 
	vector <float> ranges(nFish+1);
	ranges[0] = 0;
	for (int i=0; i<nFish; ++i){
		ranges[i+1] = ranges[i] + animals[i].fitness/sum_fit;
	}

	// init offspring with mutation
	for (int i=0; i<nFish; ++i){
		// select reproducing individual (parent)
		float range_selector = runif();
		vector <float>::iterator hi = upper_bound(ranges.begin(), ranges.end(), range_selector);	// ranges vector is naturally ascending, can use upper_bound directly
		int reproducing_id = (hi-1) - ranges.begin();
		int parent_Ad    = reproducing_id;

		// Copy reproducing parent into offspring, then add mutations
		// NOTE: Ancestor index is naturally copied.
		offspring[i] = animals[parent_Ad];
		
		// Mutate offspring's Rs 
		offspring[i].Rs += rnorm(0.0f, RsNoiseSd);		// add mutation to ws: normal random number with sd = wsNoisesd
		offspring[i].Rs = clamp(offspring[i].Rs, 1.0f, 10.0f); 
		if (b_baseline) offspring[i].Rs = Rs_base;		// in baseline experiment, set Rs = Rs_base

		// Mutate offspring's wA with some probability
		if (runif() < mu[0]/100.0f) offspring[i].wA = (offspring[i].wA == Cooperate)? Defect:Cooperate;

	}
	
	// kill parents and init new generation from offspring
	for (int i=0; i<nFish; ++i){
		animals[i] = offspring[i];
	}
	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This function advances the generation after movement is done.
// performs: Group formation, Fitness calc,   Selection,  Reproduction, Output
// relies on: ^ groupIndices, ^ ng & kg maps,  ^ fitness,  ^ fitness
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int advanceGen(){

	// copy latest positions to CPU
	cudaMemcpy2D( (void*)&(animals[0].pos),  sizeof(Particle), (void*) pos_dev,  sizeof(float2), sizeof(float2), nFish, cudaMemcpyDeviceToHost);

	// Advance generation
	updateGroups();		// calc group indices 
	calcFitness();		// calc fitnesses
	writeState();			// output data if desired
	
	select_reproduce();	// init next generation 
	disperse(rDisp);	// dispersal within rDisp units. Random dispersal if rDisp = -1

	// copy new arrays back to GPU
	cudaMemcpy2D( (void*) pos_dev, sizeof(float2), (void*)&(animals[0].pos), sizeof(Particle), sizeof(float2), nFish, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) vel_dev, sizeof(float2), (void*)&(animals[0].vel), sizeof(Particle), sizeof(float2), nFish, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) Rs_dev,  sizeof(float),  (void*)&(animals[0].Rs),  sizeof(Particle), sizeof(float),  nFish, cudaMemcpyHostToDevice);

	return 0;
}



