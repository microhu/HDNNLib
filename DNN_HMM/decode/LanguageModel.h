#pragma once

#include "word.h"

using namespace std;

class LanguageModel
{
private:
	size_t nWord;
	double exponent, wordScore;
	double *bigram, *unigram_p, *unigram_b; // pre-processing data
	void AddBigram(const wstring& s, const wstring& t, double dist);
	void AddUnigram_p(const wstring& s, double p);
	void AddUnigram_b(const wstring& s, double b);
	static const double INF;
public:
	LanguageModel(double exponent, double wordScore);
	void ReadLM(const wchar_t *lmFile);
	double Dist(Word *s, Word *t); // Get the language score by pre-processing data
	double GetP(Word *s);
	~LanguageModel(void);
};
