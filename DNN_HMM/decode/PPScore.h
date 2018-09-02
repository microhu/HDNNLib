#pragma once

#include "state.h"

class PPScore // -loglikelihood score in a pp file
{
private:
	double **score; // score is a two-dimension array, score[i][j] means the -loglikelihood of frame i assign to state j
	int n, m; // the size of array score
public:
	PPScore(const char *inFile);
	~PPScore(void);
	int Length(void);
	double** Score(void);
};