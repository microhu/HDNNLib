#pragma once

#include "state.h"

using namespace std;

struct TransMatrix
{
	wstring name;
	int nState; // number of states
	double trans[5][5]; // transition probabilty matrix
};

class TransMatrixSet
{
private:
static map<wstring, TransMatrix*> transMatrixPointer;
public:
	static void Clear();
	static TransMatrix* Get(wstring name);
	static void Add(TransMatrix *transMatrix);
};

struct Hmm
{
	TransMatrix *transMatrix;
	int state[5]; // each state
};


