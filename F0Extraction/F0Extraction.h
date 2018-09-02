#pragma once

#include "HCopy.h"
//#include "f0_main.h"

#ifdef F0EXTRACTION_EXPORTS
#define F0EXTRACTIONAPI extern "C"  _declspec(dllexport)
#else
#define F0EXTRACTIONAPI extern "C" _declspec(dllimport)
#endif

F0EXTRACTIONAPI void F0ExtractionInitialization(char* cmdline);
F0EXTRACTIONAPI int F0ExtractionFromFile(char* wavfile, byte** f0data);
F0EXTRACTIONAPI int F0ExtractionFromMemory(byte* inputdata, int len, byte** f0data);
F0EXTRACTIONAPI void UnF0ExtractionInitialization();