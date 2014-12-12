#include "../headers/particles.h"
#include "../headers/graphics.h"
#include "../utils/cuda_vector_math.cuh"
#include "../utils/simple_io.h"
#include "../utils/cuda_device.h"
using namespace std;

//cudaError __host__ copyParamsToDevice();

void ParticleSystem::printParticles(int n){
	if (n > pvec.size()) n = pvec.size();
	cout << "Particles:\n";
	cout << "Sr. n" << "\t"
//		 << "ancID " << "\t" 
//		 << "gID   " << "\t" 
		 << "wA   " << "\t" 
		 << "Rs   " << "\t" 
		 << "ng   " << "\t" 
		 << "kg   " << "\t" 
//		 << "fit  " << "\t" 
		 << "px  " << "\t"
		 << "py  " << "\t"
		 << "vx  " << "\t"
		 << "vy  " << "\t"
		 << "\n";
	for (int i=0; i<n; ++i){
		cout << i << "\t"
//			 << pvec[i].ancID << "\t" 
//			 << pvec[i].gID << "\t" 
			 << pvec[i].wA << "\t" 
			 << pvec[i].Rs << "\t" 
			 << pvec[i].ng << "\t" 
			 << pvec[i].kg << "\t" 
//			 << pvec[i].fitness << "\t" 
			 << pvec[i].pos.x << "\t"
			 << pvec[i].pos.y << "\t"
			 << pvec[i].vel.x << "\t"
			 << pvec[i].vel.y << "\t"
			 << "\n";
	}
	cout << "\n";
}


void ParticleSystem::init(Initializer &I){

	cout << "init particle system" << endl;

	// init variables
	name = I.getString("exptName");
	N = I.getScalar("particles");
	K = K0 = int(I.getScalar("fC0")*N);
	nStepsLifetime = I.getArray("nStepsLife")[0];
	genMax = int(I.getScalar("genMax"));
	b_anim_on = bool(I.getScalar("b_anim_on"));
	
	// init movement params
	par.dt = I.getScalar("dt");
	par.Rr = I.getScalar("Rr");
	par.Rs = I.getArray("Rs0")[0];	// this is initial value of Rs. If baseline, this must not change
	par.kA = I.getScalar("kA");
	par.kO = I.getScalar("kO");
	par.speed = I.getScalar("speed");
	par.copyErrSd = I.getScalar("copyErrSd");
	par.turnRateMax = I.getScalar("turnRateMax")*pi/180;
	par.cosphi = cos(par.turnRateMax*par.dt);

	par.xmin = -I.getScalar("arenaSize")/2;
	par.xmax =  I.getScalar("arenaSize")/2;
	par.ymin = -I.getScalar("arenaSize")/2;
	par.ymax =  I.getScalar("arenaSize")/2;

	// grid properties
	par.N = N;
	par.cellSize = par.Rr;	// this MUST BE equal to Rr. Otherwise code will fail.
	par.nCellsX  = ceil((par.xmax-par.xmin)/(par.cellSize));
	par.nCellsXY = par.nCellsX*par.nCellsX;


	blockDims.x = I.getScalar("blockSize");
	gridDims.x = (N-1)/blockDims.x + 1;

	pvec.resize(N);
	// -------- INIT particles ------------------
	for (int i=0; i<N; ++i){

		pvec[i].pos = runif2(par.xmax, par.ymax); 
		pvec[i].vel = runif2(1.0); // make_float2(0,1); //
		pvec[i].wA  = (i< K0)? Cooperate:Defect; 
		pvec[i].kA  = par.kA; //	runif(); // 
		pvec[i].kO  = par.kO; //	runif(); // 
		pvec[i].ancID = i;	// each fish within a block gets unique ancestor ID
		pvec[i].fitness = 0;
		if (pvec[i].wA == Cooperate) pvec[i].Rs = 2; // par.Rs; // 1.3;
		else 						 pvec[i].Rs = 3; // par.Rs; // 1.1;
//		pvec[i].Rs = 1+float(i)/(N-1)*(10-1);
	}

//	SimpleTimer T; T.start();
//	updateGroupIndices_fast();
//	T.stop(); T.printTime();
	
//	pvec[0].pos = make_float2(0,0);
//	pvec[1].pos = make_float2(par.xmax,par.ymax);
//	pvec[2].pos = make_float2(par.xmin,par.ymin);
	
	printParticles(20);

	cout << "blocks: " << gridDims.x << ", threads: " << blockDims.x << ", Total threads: " << blockDims.x*gridDims.x << endl; 
	
	// allocate memory for grid arrays on device
	cudaMalloc((void**)&cellIds_dev, par.N*sizeof(int));
	cudaMalloc((void**)&cellParticles_dev, 4*par.nCellsXY*sizeof(int));
	cudaMalloc((void**)&cellCounts_dev, par.nCellsXY*sizeof(int));
	
//	// ~~~~~~~~~~~~~~~~~ EXPT DESC ~~~~~~~~~~~~~~~~~~~~	 
//	stringstream sout;
//	sout << setprecision(3);
//	sout << I.getString("exptName");
//	if (b_baseline) sout << "_base";
//	sout << "_n("   << N
//		 << ")_nm(" << I.getScalar("moveStepsPerGen")
//		 << ")_rd(" << I.getScalar("rDisp")
//	 	 << ")_mu(" << I.getScalar("mu")[0]
//		 << ")_fb(" << I.getScalar("fitness_base")
//		 << ")_as(" << I.getScalar("arenaSize")
//	 	 << ")_rg(";
//	if (b_constRg) sout << I.getScalar("rGrp");
//	else sout << "-1";
//	if (b_baseline)  sout << ")_Rs(" << I.getScalar(Rs_base;
//	sout << ")";
//	exptDesc = sout.str(); sout.clear();


	// ~~~~~~~~~~~~~~ initial state ~~~~~~~~~~~~~~~~~~~

//	// copy arrays to device
//	//             v dst           v dst pitch     v src                     v src pitch       v bytes/elem    v n_elem       v direction
//	cudaMemcpy2D( (void*) pos_dev, sizeof(float2), (void*)&(animals[0].pos), sizeof(Particle), sizeof(float2), nFish, cudaMemcpyHostToDevice);
//	cudaMemcpy2D( (void*) vel_dev, sizeof(float2), (void*)&(animals[0].vel), sizeof(Particle), sizeof(float2), nFish, cudaMemcpyHostToDevice);
//	cudaMemcpy2D( (void*) Rs_dev,  sizeof(float),  (void*)&(animals[0].Rs),  sizeof(Particle), sizeof(float),  nFish, cudaMemcpyHostToDevice);

//	// grid
//	cellSize = 10;
//	nCellsX = int(arenaSize/(cellSize+1e-6))+1;
//	nCells = nCellsX * nCellsX;	
	
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 	Indentify groups of particles based on equivalence classes algo
//
//  Inputs: rGrp - if rGrp < 0, particle Rs is used. Else, rGrp is used
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void ParticleSystem::updateGroupIndices(float rGrp){

	vector <int> eq(N);	// temporary array of group indices

	// loop row by row over the lower triangular matrix
	for (int myID=0; myID<N; ++myID){
		eq[myID] = myID;
		for (int otherID=0; otherID< myID; ++otherID) {	

			eq[otherID] = eq[eq[otherID]];
			// calculate distance
			float2 v2other = periodicDisplacement( pvec[myID].pos, pvec[otherID].pos, dx, dy);
			float d2other = length(v2other);
			
			// set Radius of grouping from const or Rs
			float R_grp = (rGrp < 0)? pvec[myID].Rs : rGrp;	// rGrp < 0 means use particle Rs. Else, use rGrp
			// if distance is < R_grp, assign same group
			if (d2other < R_grp){
				eq[eq[eq[otherID]]] = myID;
			} 

		}
	}
	
	// complete assignment of eq. class for all individuals.
	for (int j = 0; j < N; j++) eq[j] = eq[eq[j]]; 

	// copy these group indices into the iblock'th row of particles
	memcpy2D( (Particle*) &(pvec[0].gID), (int*) &eq[0], sizeof(int), N);
	
}

int root(int q, int* par){
	while (q != par[q]){
		par[q] = par[par[q]];
		q = par[q];
	}
	return q;
}

//bool find(int p, int q, int *par){
//	return root(p) == root(q);
//}

void unite(int p, int q, int *par, int *sz){
	int i = root(p, par);
	int j = root(q, par);
	if (sz[i] < sz[j]) {par[i]=j; sz[j] += sz[i];}
	else 			   {par[j]=i; sz[i] += sz[j];}
}

void ParticleSystem::updateGroupIndices_fast(float rGrp){
	vector <int> par(N);
	vector <int> sz(N,1);
	for (int i=0; i<N; ++i) par[i] = i;
	
	for (int p=0; p<N; ++p){
		for (int q=0; q< p; ++q) {

			float2 v2other = periodicDisplacement( pvec[p].pos, pvec[q].pos, dx, dy);
			float d2other = length(v2other);
			
			// set Radius of grouping from const or Rs
			float R_grp = (rGrp < 0)? pvec[p].Rs : rGrp;	// rGrp < 0 means use particle Rs. Else, use rGrp
			// if distance is < R_grp, assign same group
			if (d2other < R_grp){
				unite(p,q,&par[0],&sz[0]);
			} 
			
		}
	}
	
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// update group-sizes and cooperators/group from group Indices 
// relies on : groupIndices - must be called after updateGroups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::updateGroupSizes(){

	// delete previous group sizes
	g2ng_map.clear(); g2kg_map.clear();
	K = 0;
	
	// calculate new group sizes and indices
	for (int i=0; i<N; ++i) {
		Particle &p = pvec[i];
	
		++g2ng_map[p.gID]; 
		if (p.wA == Cooperate ) {
			++g2kg_map[p.gID];
			++K;
		}
	}
	
	return K;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// calculate r by 2 different methods. 
// Assumes fitness  V = (k-wA)b/n - c wA 
// relies on : ng and kg maps updated by updateGroupSizes()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float ParticleSystem::update_r(){
	// calculate r and related quantities
	float pbar = K/float(N);
	r = 0;
	float varPg = 0, EpgNg = 0;
	for (int i=0; i<N; ++i){
		Particle *p = &pvec[i];
		p->kg = g2kg_map[p->gID];	// number of cooperators in the group
		p->ng = g2ng_map[p->gID];		// number of individuals in the group

		EpgNg += float(p->kg)/p->ng/p->ng;
		varPg += (float(p->kg)/p->ng-pbar)*(float(p->kg)/p->ng-pbar);
	}
	EpgNg /= N;
	varPg /= N;
	
	// calc r by another method (should match with r calc above)
	r2 = 0;
	float Skg2bNg = 0, SkgbNg = 0;
	for (map <int,int>::iterator it = g2kg_map.begin(); it != g2kg_map.end(); ++it){
		float kg_g = it->second;
		float ng_g = g2ng_map[it->first];
		Skg2bNg += kg_g*kg_g/ng_g;
		SkgbNg  += kg_g/ng_g;
	}

	if (K == 0 || K == N) r = r2 = -1e20;	// put nan if p is 0 or 1
	else {
		r  = varPg/pbar/(1-pbar) - EpgNg/pbar;
		r2 = float(N)/K/(N-K)*Skg2bNg - float(K)/(N-K) - SkgbNg/K;
	}

	return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// call the 3 functions above to update all info about groups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::updateGroups(){
	updateGroupIndices();
	updateGroupSizes();
	update_r();
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Disperse particles to random locations within radius R of current pos 
// if R == -1, disperse in the entire space
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::disperse(int R){
	for (int i=0; i<N; ++i){
		Particle *p = &pvec[i];
		
		// new velocity in random direction
		p->vel = runif2(1.0); 
		
		if (R == -1){ // random dispersal
			p->pos  = runif2(par.xmax, par.ymax); 
		}
		else{	// dispersal within radius R
			float2 dx_new = runif2(R, R);  // disperse offspring within R radius of parent
			p->pos += dx_new;
			makePeriodic(p->pos.x, par.xmin, par.xmax); 
			makePeriodic(p->pos.y, par.ymin, par.ymax); 
		}
	}	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// move particles for 1 movement step
// launch_movement_kernel() is defined in kernels.cu
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::step(){

    // Execute single kernel launch
	launch_movement_kernel();
	getLastCudaError("movement_kernel_launch");

	//glRenderer->swap = 1-glRenderer->swap;

//	cudaMemcpy2D( (void*)&(pvec[0].pos), sizeof(Particle), (void*) pos_dev,  sizeof(float2), sizeof(float2), N,    cudaMemcpyDeviceToHost);			
//	printParticles(5);

	kernelCounter.increment();
	++istep;

	return 0;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// execute a single step and check generation advance, sim completion etc.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::animate(){
	// animate particles
	if (b_anim_on) step();

	if (   glRenderer->updateMode == Step  
		&& istep % glRenderer->nSkip == 0
		&& glRenderer->quality > 0)  glutPostRedisplay();	// update display if every step update is on
	
	if (istep >= nStepsLifetime){
		istep = 0; ++igen;
//			advanceGen();	// Note: CudaMemCpy at start and end of advanceGen() ensure that all movement kernel calls are done
	}

	// when genMax genenrations are done, end run
	if (igen >= genMax) return 1;
	else return 0;
}

#include <unistd.h>

int ParticleSystem::launchSim(){

	while(1){	// infinite loop needed to poll anim_on signal.
		if (glRenderer->quality > 0) glutMainLoopEvent();
		
		int i = animate();
		usleep(2*1e3);
		
		if (i ==1) break;
	}

	return 0;
}




