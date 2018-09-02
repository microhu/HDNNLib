#pragma once

#include "state.h"
#include "hmm.h"
#include "phone.h"
#include "word.h"
#include "LanguageModel.h"
#include "TransGraph.h"
#include "PPScore.h"
#include "MonophoneRecognition.h"
#include "MonophoneAlignment.h"
#include "TriphoneAlignment.h"
#include "TriphoneRecognition.h"
#include "ppl.h"

using namespace std;

class Decode
{
private:
	__declspec(thread) static LanguageModel *lModel;
	static bool triPhone;
	__declspec(thread) static bool recogniton;
	__declspec(thread) static TransGraph *decoder;

public:
	//choose 1 of the 2
	static void InitMonoPhone(const wchar_t *modelFile, const wchar_t *dictFile);
	static void InitTriPhone(const wchar_t *modelFile, const wchar_t *tiedFile, const wchar_t *dictFile);
	static void InitTriPhone2(const wchar_t *modelFile, const wchar_t *tiedFile, const wchar_t *evadictFile, const wchar_t *errdictFile);
	
	// Optional
	static void InitLanguageModelFile(const wchar_t *lmFile, double exponent, double wordScore);
	
	//choose 1 of the 2
	static void InitRecognition(int pruning = 0);
	static void InitAlignment(void);
	
	//choose 1 of the 2 for each data
	static void Recognition(double **score, int length, FILE *out);
	static void Alignment(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel = false);

	static vector<WordDuration> Alignment(vector<wstring> wordList, double **score, int length);
	static vector<CompeteWordDuration> CompeteAlignment(vector<wstring> wordList, double **score, int length);
	static WordDuration AlignmentSingleWord(Phone *pre, Word *word, Phone *next, double **score, int startFrame, int endFrame);
	static vector<CompeteWordDuration> CompeteAlignment2(vector<wstring> wordList, double **score, int length);
	static vector<vector<WordDuration>> Decode::CompeteAlignment_fixBoundary(vector<wstring> wordList, double **score, int length);
	static vector<vector<WordDuration>> Decode::CompeteAlignment_relaxBoundary(vector<wstring> wordList, double **score, int length);
	//Dispose TransGraph object
	static void DisposeTransGraphObject();
};