#pragma once

#include "TransGraph.h"
#include "PPscore.h"

class MonophoneRecognition: public TransGraph
{
private:
	Node *root;
public:
	MonophoneRecognition(LanguageModel *lModel = NULL);
	MonophoneRecognition(LanguageModel *lModel, int pruning);
	void Process(double **score, int length, FILE *out);
};