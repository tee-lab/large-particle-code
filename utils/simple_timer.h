/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* CUda UTility Library */

#ifndef _STOPWATCH_LINUX_H_
#define _STOPWATCH_LINUX_H_

// includes, system
#include <ctime>
#include <sys/time.h>
#include <iostream>
using namespace std;

//! Windows specific implementation of StopWatch
class SimpleTimer{
public:

    SimpleTimer(){};
    ~SimpleTimer(){};

    //! Start time measurement
    inline void start();

    //! Stop time measurement
    inline void stop();

    //! Reset time counters to zero
    inline void reset();

    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned, otherwise the
    //! time between the last start() and stop call is returned
    inline float getTime() const;

    //! Mean time to date based on the number of times the stopwatch has been 
    //! _stopped_ (ie finished sessions) and the current total time
    inline float getAverageTime() const;
    
    inline void printTime();

private:

    // helper functions
  
    //! Get difference between start time and current time
    inline float getDiffTime() const;

private:

    // member variables

    //! Start of measurement
    struct timeval  start_time;

    //! Time difference between the last start and stop
    float  diff_time;

    //! TOTAL time difference between starts and stops
    float  total_time;

    //! flag if the stop watch is running
    bool running;

    //! Number of times clock has been started
    //! and stopped to allow averaging
    int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
SimpleTimer::start() {

  gettimeofday( &start_time, 0);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
SimpleTimer::stop() {

  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does 
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
SimpleTimer::reset() 
{
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if( running )
    gettimeofday( &start_time, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the 
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float 
SimpleTimer::getTime() const 
{
    // Return the TOTAL time to date
    float retval = total_time;
    if( running) {

        retval += getDiffTime();
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float 
SimpleTimer::getAverageTime() const
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}


// print time as string
inline void SimpleTimer::printTime(){
	float ms = getTime();
	cout << "t = " << ms << " ms.\n";
}


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float
SimpleTimer::getDiffTime() const 
{
  struct timeval t_time;
  gettimeofday( &t_time, 0);

  // time difference in milli-seconds
  return  (float) (1000.0 * ( t_time.tv_sec - start_time.tv_sec) 
                + (0.001 * (t_time.tv_usec - start_time.tv_usec)) );
}


class SimpleCounter{
	private:
	float samplingInterval;
	SimpleTimer fpsTimer;
	
	public:
	int count;
	float fps;
	
	public:
	SimpleCounter(){
		samplingInterval = 500;
		fpsTimer.reset();
		fpsTimer.start();
		fps = count = 0;
	}
	SimpleCounter(int si){
		samplingInterval = si;
		fpsTimer.reset();
		fpsTimer.start();
		fps = count = 0;
	}
	
	void increment(){
		++count;
		float currentTime = fpsTimer.getTime();
		if (currentTime > samplingInterval){
			fps = count/(currentTime+1e-6)*1000;
			fpsTimer.reset();
			count = 0;
		}
		
	}

};

#endif // _STOPWATCH_LINUX_H_

