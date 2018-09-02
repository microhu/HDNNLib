#pragma once
#include "TransGraph.h"
#include "PPScore.h"
#include <algorithm>

class TriphoneRecognition: public TransGraph
{
private:
	Node *root;
public:
	TriphoneRecognition(LanguageModel *lModel = NULL);
	TriphoneRecognition(LanguageModel *lModel, int pruning);
	void Process(double **score, int length, FILE *out);
};