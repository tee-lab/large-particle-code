//#include "graphics.h"

#include <cuda_runtime.h>
#include <GL/freeglut.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <jpeglib.h>
using namespace std;

#include "utils/simple_utils.h"
#include "utils/cuda_vector_math.cuh"

#include "params.h"
#include "globals.h"
#include "init.h"
//#include "altruism.h"
#include "graphics.h"
#include "particles.h" 

// openGL temporary global variables 

// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

// variables
const float tailLen_def = 0.03;
const int fpsSamplingInterval = 500; 	//  FPS sampling interval in ms

int sleepDur;							// sleep for these many ms between each glutTimerFunc call
float tailLen = tailLen_def;
static SimpleTimer fpsTimer;

// rendering options
bool b_renderText = true;
bool renderFlockingRadius = false;
bool renderLabels = false;
bool console_on = false;
bool grid_on = true;

enum FishAttribute {Group = 0, Fitness = 1, Ancestry = 2, AorD = 3, Col_kA = 4, Col_kO = 5, Col_kAO = 6, Col_Rs = 7};
int colourByAttribute = Group;
int colourTailByAttribute = AorD;
int maxColourByAttribute = 8;	// gID, fitness, ancestry, strategy, kA, kO
string fishAttributes[] = {"Group", "Fitness", "Ancestry", "AorD", "kA", "kO", "kA / kO", "Rs"};

// string to store command to be executed from GUI console
string command;

// layers - by default only layer 0 is visible
bool layerVis[] = {true,false,false,false};

// colour palette to be used. Will be used in this file only.
vector <Colour_rgb> palette;
vector <Colour_rgb> palette_rand;


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// calculate FPS
// k = 0 for display calls, = 1 for kernel calls
void calcFPS(int k){
	static float framesElapsed = 0;
	static float kernelCalls = 0;
	float currentTime = fpsTimer.getTime();
	//cout << currentTime << '\n';
	if (k == 0) ++framesElapsed;
	else if (k == 1) ++kernelCalls;
	if (currentTime > fpsSamplingInterval){
        stringstream sout; 
        sout << fixed << setprecision(1) << "Altruism! kcps = " << kernelCalls/currentTime*1000 
        								 << ", dcps = " << framesElapsed/currentTime*1000 
        								 << ", Run = " << gRun
        								 << ", s = " << genNum << "." << stepNum;
        if (graphicsQual > 0) glutSetWindowTitle(sout.str().c_str());

		fpsTimer.reset();
		framesElapsed = 0;
		kernelCalls = 0;
	}
}

// function to output gl window to jpeg image
int write_jpeg_file( const char *filename, unsigned char* raw_image, int width, int height ){

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	/* this is a pointer to one row of image data */
	JSAMPROW row_pointer[1];
	FILE *outfile = fopen( filename, "wb" );
	
	if ( !outfile ){
		printf("Error opening output jpeg file %s!\n", filename );
		return -1;
	}
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	// Setting the parameters of the output file here 
	cinfo.image_width = width;	
	cinfo.image_height = height;
	cinfo.input_components = 3; //bytes_per_pixel;
	cinfo.in_color_space = JCS_RGB; //color_space;
	// default compression parameters, we shouldn't be worried about these
	jpeg_set_defaults( &cinfo );
	// Now do the compression ..
	jpeg_start_compress( &cinfo, TRUE );
	// like reading a file, this time write one row at a time
	while( cinfo.next_scanline < cinfo.image_height )
	{
		row_pointer[0] = &raw_image[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines( &cinfo, row_pointer, 1 );
	}
	// similar to read file, clean up after we're done compressing 
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	fclose( outfile );
	// success code is 1!
	return 1;
}


int writeFrame(){
	static int nframe = 0;
	
	GLubyte * image = new GLubyte[window_width * window_height * 3];
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, image);

	stringstream ss;
	ss << framesDir << "/frame_" << nframe << ".jpg";
	
	write_jpeg_file(ss.str().c_str(), image, window_width, window_height);
	++nframe;
	
	delete [] image;
}


// set the openGL colour based on colouring style and particle properties
void setParticleColor(Particle &p){

	// determine the colour index from palette based on selected trait
	if (colourByAttribute == Group){ 	// colour by group ID
		int col_id = p.gID;
		glColor3f(palette_rand[col_id].r, palette_rand[col_id].g, palette_rand[col_id].b); // color vertices by grp
	}
	else if (colourByAttribute == Fitness){	// color by fitness
		// find max and min fitness
		float max_f, min_f;
		max_f = min_f = animals[ix2(0,gRun,nFish)].fitness;
		for (int j=0; j<nFish; ++j) {
			max_f = fmax(max_f, animals[ix2(j,gRun,nFish)].fitness);
			min_f = fmin(min_f, animals[ix2(j,gRun,nFish)].fitness);
		}
		int col_id = (p.fitness-min_f)/(max_f-min_f+1e-4)*(nFish-1);
		glColor3f(palette[col_id].r, palette[col_id].g, palette[col_id].b); 
	}
	else if (colourByAttribute == AorD){	// color by strategy
		float c = (p.wA == Cooperate)? 0.0:1.0;
		glColor3f(c, (1-c), 0.0);	
	}
	else if (colourByAttribute == Ancestry){	// colour by ancestry
		int col_id = p.ancID;
		glColor3f(palette[col_id].r, palette[col_id].g, palette[col_id].b); 
	}
	else if (colourByAttribute == Col_kA){	// colour by kA
		glColor3f((0.2+p.kA)/1.2, 0, 0); 
	}
	else if (colourByAttribute == Col_kO){	// colour by kO
		int col_id = int(p.kO*(nFish-1));
		glColor3f(0, (0.2+p.kO)/1.2, 0); 
	}
	else if (colourByAttribute == Col_kAO){	// colour by kAO
		float r = float(p.kA > p.kO);
		float scale = (p.kA > p.kO)? p.kA : p.kO;
		float col_r = r*scale;
		float col_g = (1-r)*scale;
		glColor3f(col_r, col_g, 0); 
	}
//	else if (colourByAttribute == Col_Rs){	// colour by Rs
//		float c = (p.Rs-1)/(rsMax-1+1e-6);
//		glColor3f(c, 1-c, 0); 
//	}

}


// ================================
// set the coordinate system. 
// 0 is for coordinates as per host_params 
// 1 is for normalized = 0-100 in both axes
// ================================
void setCoordSys(int l){	
	switch(l){
		case 0:
			// projection
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluOrtho2D(host_params.xmin, host_params.xmax, host_params.ymin, host_params.ymax);
			glMatrixMode(GL_MODELVIEW);
		break;

		case 1:
			// projection
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluOrtho2D(0, 100, 0, 100);
			glMatrixMode(GL_MODELVIEW);
		break;

		case 2:
			// projection
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluOrtho2D(0, 200, 0, 200);
			glMatrixMode(GL_MODELVIEW);
		break;
	}
}

// ============================ GL INIT ======================================//


bool initGL(int *argc, char **argv, SimParams &s){

	// create colour palettes
	palette = createPalette_rainbow(nFish, 0, 0.75);
	palette_rand = createPalette_random(nFish);
	//printPalette(palette_rand);

	// init display interval
	sleepDur = dispInterval;	
	
	// init
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Altruism!");

	// Callbacks
	glutDisplayFunc(display); 
//	glutIdleFunc(NULL);	// start animation immediately. Otherwise init with NULL	
//	glutMouseFunc(mousePress);
//	glutMotionFunc(mouseMove);
	glutKeyboardFunc(keyPress);

	if (dispInterval > 0) glutTimerFunc(sleepDur, timerEvent, 0);
	glutReshapeFunc(reshape);
   
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

	// enable large points!
	if (graphicsQual >=2){
		glEnable( GL_POINT_SMOOTH );
		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	}
	glPointSize( 5.0 );
	glLineWidth( 1.7 );

	setCoordSys(0);
//	// projection
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluOrtho2D(s.xmin, s.xmax, s.ymin, s.ymax);
//	glMatrixMode(GL_MODELVIEW);

	// Create timer
	fpsTimer.start();
 
    return true;
}


// ============================ RENDER COMPONENTS ============================//


inline void drawAxes(float lim){
	// draw axes 
	glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(-lim,0,0); glVertex3f(lim,0,0);
		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(0,-lim,0); glVertex3f(0,lim,0);
		glColor3f(0.0, 0.0, 1.0);
		glVertex3f(0,0,-lim); glVertex3f(0,0,lim);
	glEnd();
}

inline void drawGrid(float dx, float dy){
	glColor3f(0.4,0.4,0.4);
	
	float Dx = host_params.xmax-host_params.xmin;
	float Dy = host_params.ymax-host_params.ymin;

	glLineWidth(1.5);
	glBegin(GL_LINES);
	// draw vertical lines 
		for (int i=0; i< Dx/dx+1; ++i){
			glVertex2f(host_params.xmin+i*dx, host_params.ymin);
			glVertex2f(host_params.xmin+i*dx, host_params.ymax);
		}
	// draw horizontal lines 
		for (int i=0; i< Dy/dy+1; ++i){
			glVertex2f(host_params.xmin, host_params.ymin+i*dy);
			glVertex2f(host_params.xmax, host_params.ymin+i*dy);
		}
	glEnd();
	glLineWidth(1.7);
}


inline void drawFish(){
	glPushMatrix();
		float s = 0.6;
		glScalef(s,s,s);
		glBegin(GL_POLYGON);
			glVertex2f(0,-5);
			glVertex2f(1, 0);
			glVertex2f(0.3,1);
			glVertex2f(-0.3,1);
			glVertex2f(-1, 0);
			glVertex2f(0,-5);
		glEnd();
	glPopMatrix();
}



int renderTails(){
	glBegin(GL_LINES);
		for (int i=0; i<nFish; ++i){
			Particle &p = animals[ix2(i, gRun, nFish)]; // particles from gRun'th block are chosen for display
			float c = (p.wA == Cooperate)? 0.0:1.0;
			glColor3f(c, 1-c, 0.0);	// color tails by strategy

			float2 tailTip = p.pos - p.vel*tailLen*(0.5+ p.Rs/3);
			glVertex2f(p.pos.x, p.pos.y);
			glVertex2f(tailTip.x, tailTip.y);
		}
	glEnd();
}


int renderFish(){

	// calc Max Rs is colouring is by Rs
	float rsMax = 0;
	if (colourByAttribute == Col_Rs){	// colour by rs
		for (int i=0; i<nFish; ++i) rsMax += animals[ix2(i,gRun,nFish)].Rs;
		rsMax /= nFish;
	}

	// at low graphics qual, render particles as points
	if (graphicsQual < 3){
		glPointSize(5);
		glBegin(GL_POINTS);
			for (int i=0; i<nFish; ++i){
				// current particle (reference)
				Particle &p = animals[ix2(i, gRun, nFish)]; // particles from gRun'th block are chosen for display
				setParticleColor(p);
				glVertex2f(p.pos.x, p.pos.y);	
			}
		glEnd();
		glPointSize(1);
	}
	else{
		// render particles fancily
		for (int i=0; i<nFish; ++i){
			Particle &p = animals[ix2(i, gRun, nFish)]; // particles from gRun'th block are chosen for display
			glPushMatrix();
				glTranslatef(p.pos.x, p.pos.y, 0);
				glRotatef(atan2(-p.vel.x, p.vel.y)*180/pi, 0,0,1);
				setParticleColor(p);
				drawFish();
			glPopMatrix();
		}
	}
	
	
	// draw other fancy stuff
	for (int i=0; i<nFish; ++i){
		// current particle (reference)
		Particle &p = animals[ix2(i, gRun, nFish)]; // particles from gRun'th block are chosen for display

		glPushMatrix();
			glTranslatef(p.pos.x, p.pos.y, 0);

			// render labels
			if (renderLabels){
				glPushMatrix();
					glTranslatef(0,0,0);
					glScalef(.02,.02,.02);
					float c = (p.wA == Cooperate)? 0.0:1.0;
					glColor4f(c, 1-c, 0.0, 0.4);	
					drawString(as_string(i));
				glPopMatrix();
			}

			// render flocking radius
			if (renderFlockingRadius){
				float c = (p.wA == Cooperate)? 0.0:1.0;
				glColor4f(c, 1-c, 0.0, 0.4);	
				drawCircle(p.Rs);
			}

		glPopMatrix();
		
	}

	// render tails if desired
//	if (graphicsQual == 2) renderTails();
	
}


int renderText(){
	glPushMatrix();
		glTranslatef(host_params.xmin+2, host_params.ymax-5-2, 0);
		float charScale = 0.05;
		glScalef(charScale,charScale,charScale);
		glPointSize(1);
		glColor4f(1,1,1, 0.4);
		string s = "Colouring "+ fishAttributes[colourByAttribute];
		for (int i=0; i<s.size(); ++i){
			glutStrokeCharacter(GLUT_STROKE_ROMAN, s.c_str()[i]);
		}
	glPopMatrix();
}

int renderConsole(){
	glPushMatrix();
		float charScale = 0.14;
		SimParams &s = host_params;
		glTranslatef(s.xmin+0.1*(s.xmax-s.xmin), 
					 s.ymin+0.1*(s.ymax-s.ymin), 0);

		glScalef(charScale,charScale,charScale);
		glLineWidth(5);
		for (int i=0; i<command.size()+2; ++i){
			glutStrokeCharacter(GLUT_STROKE_ROMAN, ("> "+command).c_str()[i]);
		}
		glLineWidth(1.7);
	glPopMatrix();
}

// ========================== CHARTS ===========================================
/*
inline int renderMovingPies(){

	// calcluate group centroids
	map <int, float2> g2centroid_map;
	map <int, int> npc_map;	// number of particles used to calculate centroid (so as to avois running pies in boundary crossing groups)
	for (int i=0; i<nFish; ++i){
		Particle &p = animals[ix2(i,gRun,nFish)];
		if (p.ng > 1){	// ignore solitary individuals
			if (npc_map[p.gID] == 0 || length(g2centroid_map[p.gID] - p.pos) < 50){
				g2centroid_map[p.gID] += p.pos;
				npc_map[p.gID] += 1;
			}
		}
	}
	for (map<int,int>::iterator it=npc_map.begin(); it != npc_map.end(); ++it){
		g2centroid_map[it->first] /= npc_map[it->first];
	}
	
	for (map<int,int>::iterator it=npc_map.begin(); it != npc_map.end(); ++it){
		float2 cent = g2centroid_map[it->first];
		glPushMatrix();
			glTranslatef(cent.x,cent.y,0);
			pie2(float(g2kg_map[it->first])/g2ng_map[it->first], 0.5+g2ng_map[it->first]/2.f);
		glPopMatrix();
	}
	
}


inline int renderCharts(int w){

//	glViewport(0,w/2, w/2,w/2);

//	vector <float> ng_vec(g2ng_map.size()), kg_vec(g2ng_map.size()), gid_vec(g2ng_map.size());
//	int i=0;
//	for (map<int,int>::iterator it = g2ng_map.begin(); it != g2ng_map.end(); ++it){
//		ng_vec[i] = it->second;
//		kg_vec[i] = g2kg_map[it->first];
//		gid_vec[i] = it->first;
//		++i;
//	}

//	double breaks_array[9] = {0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 512.5};
//	vector <double> breaks(breaks_array, breaks_array+9);

//	for (int i=0; i<nFish; ++i) pg_all[i] = float(animals[i].kg)/animals[i].ng;
//	Histogram gs_hist(ng_vec, breaks);


	glViewport(0,w/2, w/2,w/2);

	double breaks_array[9] = {0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 512.5};
	vector <double> breaks(breaks_array, breaks_array+9);

//	vector <float> ng_all(nFish);
//	for (int i=0; i<nFish; ++i) ng_all[i] = animals[i].ng;
//	Histogram gs_hist1(ng_all, breaks);
//	vector <float> counts1  = gs_hist1.getCounts();
//	vector <float> breaksf1 = gs_hist1.getBreaks();
//	hist(breaksf1, counts1);

	vector <float> ng_alt, ng_def;
	for (int i=0; i<nFish; ++i){
		Particle &p = animals[ix2(i, gRun, nFish)];
		if (p.wA == Cooperate) ng_alt.push_back(p.ng);
		else ng_def.push_back(p.ng);
	}
	Histogram ng_hist_A(ng_alt, breaks);
	Histogram ng_hist_D(ng_def, breaks);
	vector <float> ng_bins = ng_hist_A.getBreaks();
	vector <float> ng_c_A = ng_hist_A.getCounts();
	vector <float> ng_c_D = ng_hist_D.getCounts();
	hist2(ng_bins, ng_c_D, ng_c_A);


	glViewport(w/2,w/2, w/2, w/2);

	// histogram of proportion of altruists in groups
	vector <float> pg_all(nFish);
	for (int i=0; i<nFish; ++i) {
		Particle &p = animals[ix2(i,gRun,nFish)];
		pg_all[i] = float(p.kg)/p.ng;
	}
	Histogram pg_hist(pg_all, 10, 0,1+1e-6);
	vector <float> counts  = pg_hist.getCounts();
	vector <float> breaksf = pg_hist.getBreaks();
	hist(breaksf, counts);


	glViewport(w/2,0, w/2, w/2);

//	// histogram of proportion of orientors (kO>kA) in groups
//	map <int, int> g2wk_map;	// map from group size to number of wO dominant individuals
//	for (int i=0; i<nFish; ++i) {
//		Particle &p = animals[ix2(i,gRun,nFish)];
//		if (p.kO > p.kA ) {
//			++g2wk_map[p.gID];
//		}
//	}
//	vector <float> wk_all(nFish);
//	for (int i=0; i<nFish; ++i) {
//		Particle &p = animals[ix2(i,gRun,nFish)];
//		p.wkg = g2wk_map[p.gID];
//		wk_all[i] = float(p.wkg)/p.ng;
//	}
//	Histogram wk_hist(wk_all, 10, 0,1+1e-6);
//	vector <float> counts1  = wk_hist.getCounts();
//	vector <float> breaksf1 = wk_hist.getBreaks();
//	hist(breaksf1, counts1);
	
	// histogram of high Rs individuals
	float rsMean = 0;
	for (int i=0; i<nFish; ++i) rsMean += animals[ix2(i,gRun,nFish)].Rs;
	rsMean /= nFish;
	map <int, int> g2rs_map;	// map from group size to number of wO dominant individuals
	for (int i=0; i<nFish; ++i) {
		Particle &p = animals[ix2(i,gRun,nFish)];
		if (p.Rs > rsMean ) {
			++g2rs_map[p.gID];
		}
	}
	vector <float> prs_all(nFish);
	for (int i=0; i<nFish; ++i) {
		Particle &p = animals[ix2(i,gRun,nFish)];
		prs_all[i] = float(g2rs_map[p.gID])/p.ng;
	}
	Histogram prs_hist(prs_all, 10, 0,1+1e-6);
	vector <float> counts2  = prs_hist.getCounts();
	vector <float> breaksf2 = prs_hist.getBreaks();
	hist(breaksf2, counts2);


//	vector <float> ng_vec_all(nFish), pg_vec_all(nFish);
//	vector <int> wa_vec_all(nFish);
//	for (int i=0; i<nFish; ++i) {
//		Particle &p = animals[ix2(i,gRun,nFish)];
//		pg_vec_all[i] = float(p.kg)/p.ng + runif_cpp(-0.02,0.02);
//		ng_vec_all[i] = p.ng*(1 + runif_cpp(0,0.04));
//		wa_vec_all[i] = (p.wA == Cooperate)? 256:0;
//	}
//	scatter(ng_vec_all, pg_vec_all, wa_vec_all, palette, true, 0.8, 9);

}


int renderCharts_genchange(int w){

	glViewport(w/4,3*w/4,w/4,w/4);
	glPushMatrix();
		glTranslatef(50-10,50,0);
		pie2(float(nCoop)/nFish, 30);
		if (r>0) glColor3f(0,1,0);
		else     glColor3f(1,0,0);
		drawCircle(5, 32, true);

		glTranslatef(20,0,0);
		if (r*b-c[gRun]>0) glColor3f(0,1,0);
		else     glColor3f(1,0,0);
		drawCircle(5, 32, true);

	glPopMatrix();
	

	glViewport(0,0,w/2,w/2);

	vector <float> rs_alt, rs_def;
	for (int i=0; i<nFish; ++i){
		Particle &p = animals[ix2(i, gRun, nFish)];
		if (p.wA == Cooperate) rs_alt.push_back(p.Rs);
		else rs_def.push_back(p.Rs);
	}
	Histogram rs_hist_A(rs_alt, 10, 0, 6); rs_hist_A.convertToPdf();
	Histogram rs_hist_D(rs_def, 10, 0, 6); rs_hist_D.convertToPdf();
	vector <float> rs_bins = rs_hist_A.getBreaks();
	vector <float> rs_c_A = rs_hist_A.getCounts();
	vector <float> rs_c_D = rs_hist_D.getCounts();
	hist2(rs_bins, rs_c_D, rs_c_A);


	glViewport(w/2,w/2,w/2,w/2);

//	vector <float> rg_alt, rg_def;
//	for (int i=0; i<nFish; ++i){
//		Particle &p = animals[ix2(i, gRun, nFish)];
//		if (p.wA == Cooperate) rg_alt.push_back(p.Rg);
//		else rg_def.push_back(p.Rg);
//	}
//	Histogram rg_hist_A(rg_alt, 10, 0, 10); rg_hist_A.convertToPdf();
//	Histogram rg_hist_D(rg_def, 10, 0, 10); rg_hist_D.convertToPdf();
//	vector <float> rg_bins = rg_hist_A.getBreaks();
//	vector <float> rg_c_A = rg_hist_A.getCounts();
//	vector <float> rg_c_D = rg_hist_D.getCounts();
//	hist2(rg_bins, rg_c_D, rg_c_A);

}

*/
// -----------------------------------------------------------------------------


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// OpenGL DISPLAY function
// This function expects groups to be updated
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void display(){

	if (graphicsQual == 0) return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glViewport(0,0,512,512);
	int wd = glutGet(GLUT_WINDOW_WIDTH);
	int ht = glutGet(GLUT_WINDOW_HEIGHT);
	int w = min(wd,ht); 

	// moving charts layer
	if (layerVis[2]){
		setCoordSys(0);		// set viewport same as fish

		glViewport(0,0,w,w);
//		renderMovingPies();		// pie charts that move with the fish groups
	}

	// fish layer
	if (layerVis[0]){
		setCoordSys(0);
		glViewport(0,0,w,w);

		renderFish();							// render Fish

		if (b_renderText) renderText();			// draw static text, if any
		if (console_on) renderConsole();		// render console commands
		if (grid_on) drawGrid(cellSize,cellSize);
		//drawAxes(200);						// render axes
	}	


	// Coarse charts layer
	if (layerVis[1]){
		setCoordSys(1);

//		renderCharts(w);	
//		renderCharts_genchange(w);	
	}

	calcFPS(0);	// calculate display rate
	glutSwapBuffers();
	
	if (framesOut) writeFrame();

}


// ============================ CALLBACKS ====================================//

void timerEvent(int value){
	// display particles from gRun and set timer function
	displayDevArrays();
	sleepDur = (dispInterval < 0)? 1000:dispInterval;
	glutTimerFunc(sleepDur, timerEvent,0);
}

void reshape(int w, int h){
	int x = min(w,h); 	// keep window square
    // viewport
    glViewport(0, 0, x, x);
    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
//    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
	gluOrtho2D(host_params.xmin, host_params.xmax, host_params.ymin, host_params.ymax);

	tailLen = tailLen_def * host_params.xmax*float(window_height)/float(x);	
}


void keyPress(unsigned char key, int x, int y){
	if (!console_on){

		if (key == 'a' || key == 32){
			b_anim_on = !b_anim_on;
		}
		
//		else if (key == 'n'){
//			++gRun;
//			if (gRun >= nBlocks) gRun = 0;
//			displayDevArrays();
//		}
		
//		else if (key == 'b'){
//			--gRun;
//			if (gRun < 0) gRun = nBlocks-1;
//			displayDevArrays();
//		}
		
		else if (key == 27){
			cout << "\n\n~~~ Simulation ABORTED! ~~~\n\n";
			freeArrays();
			exit(0);
		}	
		
		else if (key == 'c'){
			++colourByAttribute;
			if (colourByAttribute >= maxColourByAttribute) colourByAttribute = 0;
			b_renderText = true;
		}
		else if (key == 'v'){
			--colourByAttribute;
			if (colourByAttribute < 0) colourByAttribute = maxColourByAttribute-1;
			b_renderText = true;
		}

		else if (key == 't'){
			b_renderText = !b_renderText;
		}
		else if (key == 'r'){
			renderFlockingRadius = !renderFlockingRadius;
		}
		else if (key == 'l'){
			renderLabels = !renderLabels;
		}
		else if (key == 'g'){
			grid_on = !grid_on;
		}


		else if (key == 'f'){	// update fitness values
			updateGroups();
//			calcFitness(gRun);
			printParticles(&animals[ix2(0,gRun,nFish)],nFish);
		}
		
		else if (key == 'p'){	// print first 10 particles
			printParticles(&animals[ix2(0,gRun,nFish)],10);
		}

		else if (key == 'x'){
			console_on = true;
			cout << "Command-line turned on.\n";
		}
			
		else if (key >= '0' && key <= '9'){
			//cout << "number pressed: " << int(key) << '\n';
			layerVis[key-'0'] = !layerVis[key-'0'];
		}
		else{
		}

	}
	else{	// console is on. keys will be sent to command buffer.
		switch (key){
			case 27:	// esc
				executeCommand("exit");
			break;
					
			case 13:	// enter
				executeCommand(command);
			break;
			
			case 8:		// backspace
				if (command.size() != 0) command = command.substr(0, command.size()-1);
			break;
			
			default:
				command += key;
				//cout << "command = " << command << '\n';
			break;
		}
	}
		
	glutPostRedisplay();

}


int executeCommand(string cmd){
	
	if (cmd == "exit"){
		console_on = false;
		cout << "Command-line turned off.\n";
	}

	else if (cmd == "p anc"){
		cout << "ancestor Indices: \n";
		for (int i=0; i<nFish; ++i){
			cout << animals[ix2(i,gRun,nFish)].ancID << " ";
		}
		cout << '\n';
	}

	else if (cmd == "p ka"){
		cout << "kA: \n";
		for (int i=0; i<nFish; ++i){
			cout << animals[ix2(i,gRun,nFish)].kA << " ";
		}
		cout << '\n';
	}

	else if (cmd == "p ko"){
		cout << "kO: \n";
		for (int i=0; i<nFish; ++i){
			cout << animals[ix2(i,gRun,nFish)].kO << " ";
		}
		cout << '\n';
	}

	else{}
	
	command = "";

}






