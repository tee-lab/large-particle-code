#include "../headers/graphics.h"
#include "../headers/particles.h"
#include "../utils/simple_io.h"
#include "../utils/simple_math.h"


Renderer * glRenderer;


void loadShader(string filename, GLuint &shader_id, GLenum shader_type){

	ifstream fin(filename.c_str());
	string c((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
	const char * glsl_src = c.c_str();

	shader_id = glCreateShader(shader_type);
	glShaderSource(shader_id, 1, &glsl_src, NULL);
	glCompileShader(shader_id);
	if (GL_NO_ERROR != glGetError()) cout << "Error compiling shader!\n";
}


void setActiveRenderer(Renderer * R){
	glRenderer = R;
}

// ===========================================================
// class Shape
// ===========================================================

Shape::Shape(string obj_name, bool dbuff){
	objName = obj_name;
	vertexShaderFile = "shader_vertex_" + obj_name + ".glsl";
	fragmentShaderFile = "shader_fragment_" + obj_name + ".glsl";
	doubleBuffered = dbuff;
}

void Shape::init(string obj_name, bool dbuff){
	objName = obj_name;
	vertexShaderFile = "shader_vertex_" + obj_name + ".glsl";
	fragmentShaderFile = "shader_fragment_" + obj_name + ".glsl";
	doubleBuffered = dbuff;
}

void Shape::createVBO(void* data, int nbytes){
	cout << "creating buffers for " << objName << endl;
	// create buffers 
	glGenBuffers(2, vbo_ids);					// create buffer ids and store in array

	// alllocate space and copy initial data to 1st buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0]); 	// Bring 1st buffer into current openGL context
	glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_DYNAMIC_DRAW); 

	if (doubleBuffered){
	// allocate space for 2nd buffer, but dont copy any data	
	glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1]); 	// bring 2nd buffer into current openGL context
	glBufferData(GL_ARRAY_BUFFER, nbytes, NULL, GL_DYNAMIC_DRAW);	
	}
	
	// remove buffers from curent context. (appropriate buffers will be set bu CUDA resources)
	glBindBuffer(GL_ARRAY_BUFFER, 0);			
}

void Shape::createShaders(){
	GLenum ErrorCheckValue = glGetError();

	cout << "creating shaders for " << objName << endl;

	loadShader(vertexShaderFile, vertexShader_id, GL_VERTEX_SHADER);	
	loadShader(fragmentShaderFile, fragmentShader_id, GL_FRAGMENT_SHADER);

	program_id = glCreateProgram();
	glAttachShader(program_id, vertexShader_id);
	glAttachShader(program_id, fragmentShader_id);
	glLinkProgram(program_id);
	
	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR){
		cout << "ERROR: Could not create the shaders: " << gluErrorString(ErrorCheckValue) << endl;
	}

}

void Shape::deleteVBO(){
	glDeleteBuffers(2, vbo_ids);
}


void Shape::deleteShaders(){
	glUseProgram(0);

	glDetachShader(program_id, vertexShader_id);
	glDetachShader(program_id, fragmentShader_id);

	glDeleteShader(fragmentShader_id);
	glDeleteShader(vertexShader_id);

	glDeleteProgram(program_id);
}

void Shape::createColorBuffer(){
	glGenBuffers(1, &colorBuffer_id);
}

void Shape::setColors(float4 *colData, int nColors){
	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer_id);
	glBufferData(GL_ARRAY_BUFFER, nColors*sizeof(float4), colData, GL_DYNAMIC_DRAW); 
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::deleteColorBuffer(){
	glDeleteBuffers(1, &colorBuffer_id);
}

void Shape::useProgram(){
	glUseProgram(program_id);
}


// ===========================================================
// class Renderer
// ===========================================================

void Renderer::init(Initializer &I){
	tailLen_def = 0.03;
	maxColorAttributes = 9;

	b_renderText = true;
	b_renderLabels = true;
	b_renderRs = true;
	b_renderConsole = false;
	b_renderGrid = false;
	b_renderAxes = false;

	window_width = 512;
	window_height = 512;

	int t = I.getScalar("dispInterval");
	if (t < 0) {
		nSkip = -t;
		updateMode = Step;
		displayInterval = 50;	// in this case, only title will be updated in timer function (at 20 fps)
	}
	else {
		displayInterval = t;
		nSkip = -1;
		updateMode = Time;
	}

	quality = I.getScalar("graphicsQual");
	//layerVis
	
	// init shapes
	pos_shape.init("psys", false);
	vel_shape.init("psys_vel", false);
	grid_shape.init("grid", false);

}

void Renderer::connect(ParticleSystem *P){

	cout << "connecting particle system" << endl;
	// set the renderer particle system
	psys = P;

	// set the arena bounds
	xmin = P->par.xmin; //-I.getScalar("arenaSize")/2; 
	xmax = P->par.xmax; // I.getScalar("arenaSize")/2; 
	ymin = P->par.ymin; //-I.getScalar("arenaSize")/2; 
	ymax = P->par.ymax; // I.getScalar("arenaSize")/2; 

	// create colour palettes
	int np = P->N; //I.getScalar("particles");
	int n;
	if (np<10) n=10;
	else n=np;
	palette = createPalette_rainbow(n, 0, 0.75);
	palette_rand = createPalette_random(n);
	//printPalette(palette_rand);
	
	// particle colour properties
	maxColorAttributes = 9;
	iColorAttribute = 0;
	string names[] = {"wA",  "Rs",    "kA",    "kO",    "gID", "ng",  "kg",  "ancID", "fitness"};
	string types[] = {"int", "float", "float", "float", "int", "int", "int", "int",   "float"};
	float  cvmin[] = { 0,     1,       0,       0,       0,     0,     0,     0,       0};
	float  cvmax[] = { 1,    10,       1,       1,       np,    np,    np,    np,      110};
	
	colorAttrNames.assign(names, names+maxColorAttributes);
	colorAttrTypes.assign(types, types+maxColorAttributes);
	colorValueMin.assign(cvmin, cvmin+maxColorAttributes);
	colorValueMax.assign(cvmax, cvmax+maxColorAttributes);

	//particleColorAttribute = (void*)&psys->pvec[0].wA;
	setColorAttr(iColorAttribute);
	cout << "Coloring by " << colorAttrNames[iColorAttribute] << endl;
	
	// create buffers and shaders
	createShaders();
	createVBO();

}

void Renderer::disconnect(){
	deleteVBO();
	deleteShaders();
}


float Renderer::getParticleColorValue(int i){
	string s = colorAttrTypes[iColorAttribute];
	if (s=="float") 	return *(float*)((Particle*)particleColorAttribute+i);
	else if (s=="int") 	return *(int*)((Particle*)particleColorAttribute+i);
	else return 0;
}


void Renderer::setColorAttr(int j){
	if (j >= maxColorAttributes || j<0) return;
	particleColorAttribute = (void*)((float*)&psys->pvec[0].wA+j);
	iColorAttribute = j;
	cout << "Colouring by " << colorAttrNames[j] << endl;
}

// made this into separate data so that colour scheme can be changed dynamically
void Renderer::setColorBufferData(){
	float4 * col_tmp = new float4[psys->N];
	for (int i=0; i<psys->N; ++i) {
		Colour_rgb c = palette[discretize_index(getParticleColorValue(i), palette.size(), colorValueMin[iColorAttribute],colorValueMax[iColorAttribute])];
		col_tmp[i] = make_float4(c.r, c.g, c.b, 1);
//		cout << "particle " << i << ": " << getParticleColorValue(i) <<endl; //<< "(" << colorValueMax[iColorAttribute] <<  ")" << endl;
	}

	pos_shape.setColors(col_tmp, psys->N);

	delete [] col_tmp;
}

void Renderer::createVBO(){

	// this vertex array is implicitly required for god-knows-what reason, so create it :/
	glGenVertexArrays(1, &vao_id);
	glBindVertexArray(vao_id);

	// create temporary copies of pos and vel
	float2 * pos_tmp = new float2[psys->N];
	float2 * vel_tmp = new float2[psys->N];
	for (int i=0; i<psys->N; ++i){
		pos_tmp[i] = psys->pvec[i].pos;
		vel_tmp[i] = psys->pvec[i].vel;
	}

	// create the buffers
	pos_shape.createVBO(pos_tmp, psys->N*sizeof(float2));
	vel_shape.createVBO(vel_tmp, psys->N*sizeof(float2));
	pos_shape.createColorBuffer();
	setColorBufferData();

	// register pos and vel buffers with CUDA
	cudaGraphicsGLRegisterBuffer(&pos_cgr, pos_shape.vbo_ids[0], cudaGraphicsMapFlagsWriteDiscard);	
	cudaGraphicsGLRegisterBuffer(&vel_cgr, vel_shape.vbo_ids[0], cudaGraphicsMapFlagsWriteDiscard);	

	// delete temporary copies
	delete [] pos_tmp;
	delete [] vel_tmp;

	// set current active buffers
	swap = 0;

	// init grid shape
	float Dx = xmax-xmin;
	float Dy = ymax-ymin;
	float dx = psys->par.cellSize;
	float dy = dx;
	grid_shape.nVertices = int(Dx/dx+1+Dy/dy+1)*2;

	float2 grid_tmp[grid_shape.nVertices];
	// vertical lines 
	cout << Dx << " " << Dy << " " << dx << " " << dy << " " << grid_shape.nVertices << endl;
	int k = 0;
	for (int i=0; i< Dx/dx+1; ++i){
		grid_tmp[k++] = make_float2(xmin+i*dx, ymin);
		grid_tmp[k++] = make_float2(xmin+i*dx, ymax);
	}
	// horizontal lines 
	for (int i=0; i< Dy/dy+1; ++i){
		grid_tmp[k++] = make_float2(xmin, ymin+i*dy);
		grid_tmp[k++] = make_float2(xmax, ymin+i*dy);
	}

	grid_shape.createVBO(grid_tmp, grid_shape.nVertices*sizeof(float2));

	// map resources initially so that kernel execution can begin immediately.
    cudaGraphicsMapResources(1, &pos_cgr, 0);
    cudaGraphicsMapResources(1, &vel_cgr, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&psys->pos_dev, &num_bytes, pos_cgr);
    cudaGraphicsResourceGetMappedPointer((void**)&psys->vel_dev, &num_bytes, vel_cgr);

	// ==== VBOs for vertex colours ===========================================
	
	// allocate vel_new_dev and Rs_dev on GPU
	cudaMalloc((void**)&psys->Rs_dev, psys->N*sizeof(float));
	cudaMalloc((void**)&psys->vel_new_dev, psys->N*sizeof(float2));
	cudaMemcpy2D((void*)psys->Rs_dev, sizeof(float), (void*)&psys->pvec[0].Rs, sizeof(Particle), sizeof(float),  psys->N, cudaMemcpyHostToDevice);

	cout << "DONE" << endl;

}

void Renderer::deleteVBO(){
    cudaGraphicsUnmapResources(1, &pos_cgr, 0);
    cudaGraphicsUnmapResources(1, &vel_cgr, 0);

	cudaGraphicsUnregisterResource(pos_cgr);
	cudaGraphicsUnregisterResource(vel_cgr);

	glDeleteVertexArrays(1, &vao_id);

	pos_shape.deleteVBO();
	vel_shape.deleteVBO();
	pos_shape.deleteColorBuffer();
}


void Renderer::createShaders(){
	pos_shape.createShaders();
	grid_shape.createShaders();
}

void Renderer::deleteShaders(){
	pos_shape.deleteShaders();
	grid_shape.deleteShaders();
}


void Renderer::renderGrid(){
	// draw grid
	if (b_renderGrid){
		grid_shape.useProgram();

		GLuint loc = glGetUniformLocation(grid_shape.program_id, "bounds");
		glUniform4f(loc, xmin, xmax, ymin, ymax);

		glBindBuffer(GL_ARRAY_BUFFER, grid_shape.vbo_ids[0]);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);	
	
		glDrawArrays(GL_LINES, 0, grid_shape.nVertices);
	}
}

void Renderer::renderParticles(){
	
	// unmap resources from CUDA to make them available to OpenGL
	// these functions will wait till all kernels called previously have finished 
    cudaGraphicsUnmapResources(1, &pos_cgr, 0);
    cudaGraphicsUnmapResources(1, &vel_cgr, 0);
	// draw particles 

	// use program from pos_shape
	pos_shape.useProgram();

	// set the coord system bounds for getting orthograhic projection in vertex-shader
	GLuint loc = glGetUniformLocation(pos_shape.program_id, "bounds");
	glUniform4f(loc, xmin, xmax, ymin, ymax);
	
	// set the point size to match physical scale
	loc = glGetUniformLocation(pos_shape.program_id, "psize");
	float window_size_x = min(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
	float psize = window_size_x/(xmax-xmin)*psys->par.Rr;
	if (psize < 1) psize = 1;
	glUniform1f(loc, psize);

	glBindBuffer(GL_ARRAY_BUFFER, pos_shape.vbo_ids[swap]);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);	

	glBindBuffer(GL_ARRAY_BUFFER, pos_shape.colorBuffer_id);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);	

	glDrawArrays(GL_POINTS, 0, psys->N);

	// map resources again for use by CUDA,
	// kernel execution will resume once graphics rendering is complete
    cudaGraphicsMapResources(1, &pos_cgr, 0);
    cudaGraphicsMapResources(1, &vel_cgr, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&psys->pos_dev, &num_bytes, pos_cgr);
    cudaGraphicsResourceGetMappedPointer((void**)&psys->vel_dev, &num_bytes, vel_cgr);
    
}



int Renderer::getDisplayInterval(){
	return displayInterval;
}


void Renderer::toggleAnim(){
	psys->b_anim_on = !psys->b_anim_on;
	
}

void Renderer::toggleConsole(){
	command = "";
	b_renderConsole = !b_renderConsole;
}

void Renderer::toggleText(){
	b_renderText = !b_renderText;
}

void Renderer::toggleGrid(){
	b_renderGrid = !b_renderGrid;
}

void Renderer::toggleAxes(){
	b_renderAxes = !b_renderAxes;
}

void Renderer::receiveConsoleChar(char key){
	switch (key){
		case 27:	// esc
			b_renderConsole = false;
			cout << "Command-line turned off.\n";
		break;
				
		case 13:	// enter
			executeCommand();
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


int Renderer::executeCommand(){
	vector <string> args = parse(command);
	
	if (args[0] == "exit"){
		b_renderConsole = false;
		cout << "Command-line turned off.\n";
	}

	if (args[0] == "set"){
		if (args[1] == "background" || args[1] == "bg" ){
			if (args.size() >= 5){
				float r = as_float(args[2]), g = as_float(args[3]), b = as_float(args[4]); //, a = as_float(args[5]);
				glClearColor(r,g,b,1);
			}
		}
	}

	else{}
	
	command = "";

}


string Renderer::makeTitle(){
    const unsigned char * ver = glGetString(GL_VERSION);
    stringstream sout; 
    sout << fixed << setprecision(1) << psys-> N << " Particles" //<< "GL" << ver 
    								 << ", kcps = " << psys->kernelCounter.fps
    								 << ", dcps = " << frameCounter.fps
    								 << ", s = " << psys->igen << "." << psys->istep;
	return sout.str();
}



// =================================================================================
//
//			OpenGL functions and callbacks
//				Init				
//				Display
//				Timer
//				reshape
//				keypress
//				
//			These functions assume that
//				glRenderer points to a valid renderer R
//				psys points to a valid Particle System P
//				P is connect()-ed to R
//
// =================================================================================


// function to initialize opengl.
// The Renderer R must be created, initialized and connected to a particleSystem
// before this function can be called.
bool initGL(Renderer *R, int *argc, char **argv){
	cout << "init GL" << endl;
	
	glRenderer = R;

	// init
	glutInit(argc, argv);
	glutInitContextVersion(4, 0);
	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	glutInitContextProfile(GLUT_CORE_PROFILE);

	glutInitWindowSize(glRenderer->window_width, glRenderer->window_height);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow("Altruism!");

	glewExperimental = GL_TRUE;
	glewInit();

	// Callbacks
	glutDisplayFunc(display); 
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyPress);
//	glutIdleFunc(NULL);	// start animation immediately. Otherwise init with NULL	
	glutTimerFunc(glRenderer->getDisplayInterval(), timerEvent, 0);
	glutCloseFunc(cleanup);
	
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnable(GL_PROGRAM_POINT_SIZE);
//  glDisable(GL_DEPTH_TEST);
//	glEnable( GL_POINT_SMOOTH );
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	// enable large points!
//	glPointSize( 5.0 );
//	glLineWidth( 1.7 );

    return true;
}

void cleanup(){
	glRenderer->disconnect();
}


// ===================== DISPLAY FUNCTION ====================================//

void display(){
	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glRenderer->renderParticles();	
	glRenderer->renderGrid();	

	glRenderer->frameCounter.increment();	// calculate display rate

//	glutPostRedisplay();
	glutSwapBuffers();


}

// ============================ CALLBACKS ====================================//

void timerEvent(int value){
	
//	glRenderer->psys->animate();

    glutSetWindowTitle(glRenderer->makeTitle().c_str());

	if (glRenderer->updateMode == Time) glutPostRedisplay();
	glutTimerFunc(glRenderer->getDisplayInterval(), timerEvent, 0);
}


void reshape(int w, int h){
	int x = min(w,h); 	// keep window square
    // viewport
    glViewport(0, 0, x, x);
//	glRenderer->tailLen = glRenderer->tailLen_def * glRenderer->xmax*float(glRenderer->window_height)/float(x);	
}


void keyPress(unsigned char key, int x, int y){
	if (!glRenderer->b_renderConsole){		
		
		if (key == 32){
			glRenderer->toggleAnim();
		}
		
		if (key == 'a'){
			glRenderer->toggleAxes();
			cout << "axes ";
			if (glRenderer->b_renderAxes) cout << "on\n";
			else cout << "off\n";
		}
		
		else if (key == 27){
			cout << "\n\n~~~ Simulation ABORTED! ~~~\n\n";
			exit(0);
		}	
		else if (key == 'g'){
			glRenderer->toggleGrid();
			cout << "grid ";
			if (glRenderer->b_renderGrid) cout << "on\n";
			else cout << "off\n";
		}

		else if (key == 'x'){
			glRenderer->toggleConsole();
			cout << "Command-line turned on.\n";
		}
		else if (key == 'p'){
			cout << "Command-line turned on.\n";
		}

			
		else if (key >= '0' && key <= '9'){
			//cout << "number pressed: " << int(key) << '\n';
			//layerVis[key-'0'] = !layerVis[key-'0'];
			glRenderer->setColorAttr(key-'0');
			glRenderer->setColorBufferData();
		}
		else{
		}

	}
	else{	// console is on. keys will be sent to command buffer.
		glRenderer->receiveConsoleChar(key);
	}
		
	glutPostRedisplay();

}


//int Renderer::createVBO_grid(){
//	// ==== VBO for grid =======================================================
//	glGenBuffers(1, &grid_vbo_id);
//	glBindBuffer(GL_ARRAY_BUFFER, grid_vbo_id);
//	float Dx = xmax-xmin;
//	float Dy = ymax-ymin;
//	float dx = psys->par.cellSize;
//	float dy = dx;
//	float2 grid_tmp[int(Dx/dx+1+Dy/dy+1)*2];
//	// vertical lines 
//	cout << Dx << " " << Dy << " " << dx << " " << dy << " " << int(Dx/dx+1+Dy/dy+1) << endl;
//	int k = 0;
//	for (int i=0; i< Dx/dx+1; ++i){
//		grid_tmp[k++] = make_float2(xmin+i*dx, ymin);
//		grid_tmp[k++] = make_float2(xmin+i*dx, ymax);
//	}
//	// horizontal lines 
//	for (int i=0; i< Dy/dy+1; ++i){
//		grid_tmp[k++] = make_float2(xmin, ymin+i*dy);
//		grid_tmp[k++] = make_float2(xmax, ymin+i*dy);
//	}
//	glBufferData(GL_ARRAY_BUFFER, k*sizeof(float2), grid_tmp, GL_DYNAMIC_DRAW); 
//}




