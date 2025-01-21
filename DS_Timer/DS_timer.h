// #pragma once
#ifndef _DS_TIMER_H
#define _DS_TIMER_H

#include <string> // std string

// #ifndef uint
// typedef unsigned int uint;
// #endif

#ifdef _WIN32
// For windows
#include <Windows.h>
typedef LARGE_INTEGER TIME_VAL;
#else
// For Unix/Linux
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h> // c string
typedef struct timeval TIME_VAL;
#endif

#define TIMER_ON true
#define TIMER_OFF false

class DS_timer
{
private:
	bool turnOn;

	uint numTimer;
	uint numCounter;

	// For timers
	bool *timerStates;
	TIME_VAL ticksPerSecond;
	TIME_VAL *start_ticks;
	TIME_VAL *end_ticks;
	TIME_VAL *totalTicks;

	char timerTitle[255];
	std::string *timerName;

	// For counters
	uint *counters;

	void memAllocCounters(void);
	void memAllocTimers(void);
	void releaseCounters(void);
	void releaseTimers(void);

public:
	DS_timer(int _numTimer = 1, int _numCount = 1, bool _trunOn = true);
	~DS_timer(void);

	// For configurations
	inline void timerOn(void) { turnOn = TIMER_ON; }
	inline void timerOff(void) { turnOn = TIMER_OFF; }

	uint getNumTimer(void);
	uint getNumCounter(void);
	uint setTimer(uint _numTimer);
	uint setCounter(uint _numCounter);

	// For timers

	void initTimer(uint id);
	void initTimers(void);
	void onTimer(uint id);
	void offTimer(uint id);
	double getTimer_ms(uint id);

	void setTimerTitle(char *_name)
	{
		memset(timerTitle, 0, sizeof(char) * 255);
		memcpy(timerTitle, _name, strlen(_name));
	}

	void setTimerName(uint id, std::string &_name) { timerName[id] = _name; }
	void setTimerName(uint id, char *_name) { timerName[id] = _name; }

	// For counters

	void incCounter(uint id);
	void initCounters(void);
	void initCounter(uint id);
	void add2Counter(uint id, uint num);
	uint getCounter(uint id);

	// For reports

	void printTimer(float _denominator = 1);
	void printToFile(char *fileName, int _id = -1);
	void printTimerNameToFile(char *fileName);
};

#endif