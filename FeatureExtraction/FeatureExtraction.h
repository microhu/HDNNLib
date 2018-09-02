#pragma once

#include "HCopy.h"

#ifdef FEATUREEXTRACTION_EXPORTS
#define FEATUREEXTRACTIONAPI extern "C"  _declspec(dllexport)
#else
#define FEATUREEXTRACTIONAPI extern "C" _declspec(dllimport)
#endif 

FEATUREEXTRACTIONAPI void Initialization(char* cmdline);
FEATUREEXTRACTIONAPI int FeatureExtractionFromFile(char* wavfile, byte** outdata);
FEATUREEXTRACTIONAPI int FeatureExtractionFromMemory(byte* inputdata, int len, byte** outdata);
FEATUREEXTRACTIONAPI void UnInitializationFeatureExaction();
