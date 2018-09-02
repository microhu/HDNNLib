#pragma once
#include "TransGraph.h"
#include "PPScore.h"

class TriphoneAlignment: public TransGraph
{
private:
	void Init(void);
	int releaseCount = 0;
public:
	void Process(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel);
	vector<WordDuration> GetResult(vector<wstring> wordList, double **score, int length);
	WordDuration DecodeSingleWord(Phone *pre, Word *word, Phone *next, double **score, int startFrame, int endFrame);
	void ReleaseAllofSource();
};
