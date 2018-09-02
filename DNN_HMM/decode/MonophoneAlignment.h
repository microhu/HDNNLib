#pragma once
#include "TransGraph.h"
#include "PPScore.h"

class MonophoneAlignment: public TransGraph
{
private:
	Node *root, *endNode;
	void Init(void);
	void PushWord(wstring name);
	 // state list, stay probabilty list, list length, the word
public:
	void Process(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel);
};