#include <cstdio>
#include <fstream>
#include "FeatureExtraction.h"
#include <iostream>
#include <vector>

using namespace std;

__declspec(thread) static byte mfcdata[STACKSIZE];


void Initialization(char* cmdline)
{
   int argc;
   int i, j;
   char **argv = new char *[100];
   for (i = 0; i < 100; i++)
   {
      argv[i] = new char[1024];
   }
   Str2Arg(cmdline, argc, argv);

   HcopyConfigParametersIntialization(argc, argv);

   for (int i = 0; i < 100; i++)
   {
      delete[1024]argv[i];
      argv[i] = NULL;
   }
   delete[100]argv;
   argv = NULL;
}

void UnInitializationFeatureExaction()
{
	UnInitialization();
}

int FeatureExtractionFromFile(char* wavfile, byte** outdata)
{
   int pos = 0;
   off = 0.0;

   HcopySystemParametersIntialization();

   OpenSpeechFile(wavfile);               /* Load initial file  S1 */

   PutTargetToArray(mfcdata, &pos);

   if (trace & T_MEM) PrintAllHeapStats();

   ResetHeap(&lStack);
   ResetHeap(&iStack);
   ResetHeap(&oStack);
   if (chopF) ResetHeap(&cStack);

   *outdata = mfcdata;
   return pos;
}

int FeatureExtractionFromMemory(byte* inputdata, int len, byte** outdata)
{
   int pos = 0;


   try
   {
    HcopySystemParametersIntialization();
    OpenSpeechFileFromMemory(inputdata, len);  /* Load initial file  S1 */

    PutTargetToArray(mfcdata, &pos);
   }
   catch (int e)
   {
       pos = 0;
   }            


   if (trace & T_MEM) PrintAllHeapStats();

   // Release all of source
   ReleaseSource();

   *outdata = mfcdata;
   return pos;
}

/*
byte wav[10000000];

int main()
{
   Initialization("HTKfunctions -C D:\\LF\\data\\config\\hcopy.config");

   //ifstream inFile("D:\\feature", ifstream::binary);
   ifstream inFile("D:\\audioData\\01AA010H.wav", ifstream::binary);
   if( !inFile )
   cout << "File error" << endl;

   inFile.seekg(0, inFile.end);
   int length = inFile.tellg();
   inFile.seekg(0, inFile.beg);

   inFile.read((char*)wav, length);

   byte* ptr;
   int resLen = FeatureExtractionFromMemory(wav, length, &ptr);

   ofstream outFile("D:\\feature", ofstream::binary);
   outFile.write((char*)ptr, resLen);

   return 0;
}
*/