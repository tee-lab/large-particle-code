#ifndef IO_UTILS_H__
#define IO_UTILS_H__

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <cstring>
using namespace std;

// a very simple terminal progress bar that can use either an internal counter or an external counter
class SimpleProgressBar{
	private:
	int length;
	int dt;
	int t_now;	// internal counter
	int * steps_total;	// number of steps (external variable)
	int * steps_now;	// external counter
	
	public:
	string title;
	
	public:
	inline SimpleProgressBar(){
		t_now = 0;
		length = 40;
		*steps_total = 40;
		steps_now = &t_now; 
		dt = *steps_total/length;
	};
	
	// constructor to use external counter
	inline SimpleProgressBar(int* nsteps, int* stepsnow, string ti = "", int len = 40){
		cout << "Calling func:\n";
		t_now = 0;
		steps_total = nsteps;
		steps_now = stepsnow;
		title = ti;
		length = len;
		dt = *steps_total/length;
		cout << "tnow= " << t_now << "\n";
		cout << "steps_now= " << steps_now << "\n";
		cout << "steps_total= " << steps_total << "\n";
	}

	// constructor to use internal counter
	inline SimpleProgressBar(int* nsteps, string ti="", int len = 40){
//		SimpleProgressBar(nsteps, &t_now, ti, len); 	>>> Initialization fails if this is used. Why?
		t_now = 0;
		steps_total = nsteps;
		steps_now = &t_now;		// use internal counter
		title = ti;
		length = len;
		dt = *steps_total/length;
	}
	

	inline void start(){
		cout << "\n> Launch: " << title << "\n";
		cout << "   > ";
		for (int i=0; i<length; ++i) cout << "_";
		cout << "\n   > ";
		//cout << "   > Simulate " << *steps_total << " generations, plot after every " << plotStep << " steps.\n   > "; 
		cout.flush();
	}
	inline void increment(){
		++t_now;
	}
	inline void print(){
		if (*steps_now % dt == 0) { cout << "."; cout.flush(); }
	}
	inline void stop(){
		cout << "> DONE.\n"; cout.flush();
	}
	inline void update(){
		++(*steps_now);
		print();
		if (*steps_now >= *steps_total) stop();
	}
};


template <class T>
string as_string(T f, int prec = 5, bool fmt_fixed = false){
	stringstream ss;
	if (fmt_fixed) ss << fixed;
	ss << setprecision(prec) << f;
	return ss.str();
}

inline float as_float(string s){
	float f;
	stringstream ss; ss.str(s);
	ss >> f;
	return f;
}

template <class T, class U>
void printMap(map <int, T> &m, map <int, U> &m2, string name = "map"){
	cout << name << ":\n";
	for (typename map<int, T>::iterator i = m.begin(); i!= m.end(); ++i){
		cout << i->first << "|" << i->second << ", " << m2[i->first] << '\n';
	}
	cout << '\n';
}


template <class T>
void printMap(map <int, T> &m, string name = "map"){
	cout << name << ":\n";
	for (typename map<int, T>::iterator i = m.begin(); i!= m.end(); ++i){
		cout << i->first << "|" << i->second << '\n';
	}
	cout << '\n';
}

template <class T>
void printArray(T *v, int n, string name = "v", string nameSep=" ", string endSep = ""){
	cout << name << ":" << nameSep;
	for (int i = 0; i< n; ++i){
		cout << v[i] << " ";
	}
	cout << "\n" << endSep;
}

inline void printTime_hhmm(float clkt){
	int nhrs = clkt/1000.0f/60.f/60.f;
	float nmins = clkt/1000.0f/60.f - nhrs*60;
	cout << "| t = " << nhrs << ":" << nmins << " hh:mm.\n\n";
}

template <class SRC, class DST>
inline DST* memcpy2D(DST * dst, SRC * src, int bytes_per_elem, int n_elements){
	if (bytes_per_elem > sizeof(DST)) {
		cout << "Error: overlapping copy (bytes per element > destination pitch)\n\n";
		return NULL;
	}
	DST * d = dst; SRC * s = src;
	for (int i=0; i<n_elements; ++i){
		memcpy((void*)d, (void*)s, bytes_per_elem);
		++d; ++s;
	}
	return dst;
}

inline vector <string> parse(string cmd){
	vector <string> tokens;
	stringstream ss; ss.str(cmd);
//	cout << "Tokens: \n";
	while (!ss.eof()){
		string s; ss >> s;
		tokens.push_back(s);
//		cout << s << "\n";
	} 
//	cout << "----\n";
	return tokens;
}


#endif


