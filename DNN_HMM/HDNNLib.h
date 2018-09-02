#pragma once
#include <stdio.h>

#ifdef DNN_HMM_EXPORTS
#define DNNHMMDECODEAPI extern "C"  _declspec(dllexport)
#else
#define DNNHMMDECODEAPI _declspec(dllimport)
#endif

typedef unsigned char byte;

DNNHMMDECODEAPI void LoadModel(wchar_t *dataPath, bool likelihood=true, int cores=0);
DNNHMMDECODEAPI int EvaluateStrictBoundary(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI int EvaluateRelaxBoundary(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI void unLoadModel();

// extern interface for research experiments
DNNHMMDECODEAPI int LikelihoodWithGivenPhoneBoundary(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI int ForceAlignWithCanonicalWords_Memo(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI int ForceAlignWithCanonicalWords(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI int EvaluateWithCanonicalWords(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength);
DNNHMMDECODEAPI int EvaluateWithCompetingPhones(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength, int competingPhoneNumber, bool fixPhoneBoundary);
DNNHMMDECODEAPI int EvaluateWithCompetingPhonesMemoStream(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength, int competingPhoneNumber, bool fixPhoneBoundary);