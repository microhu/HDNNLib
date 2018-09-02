#include "F0Extraction.h" // remove it for test or debug
//#include "HCopy.h";

#include <vector>
#include <iostream>
#include <fstream>
#include <errno.h>
#include "io.h"
#include "get_f0.h"
using namespace std;
// add for f0 extraction
__declspec(thread) union ByteDoubUnion
{
   double dvalue;
   byte bytevalue[sizeof(double) / sizeof(byte)];
}ByteDU;

__declspec( thread ) static byte pitchData[STACKSIZE];

void F0ExtractionInitialization(char* cmdline)
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

void UnF0ExtractionInitialization()
{
	UnInitialzation();
}

// later 

typedef struct _Wave{   /* Internal wave file representation */
   MemHeap *mem;        /* memory heap for this wave rec */
   FileFormat fmt;      /* Format of associated source file */
   Boolean isPipe;      /* Source is a pipe */
   HTime sampPeriod;    /* Sample period in 100ns units */
   int  hdrSize;        /* Header size in bytes */
   long nSamples;       /* No of samples in data */
   long nAvail;         /* Num samples allocated for data */
   short *data;         /* Actual data (always short once loaded) */
   int frSize;          /* Num samples per frame */
   int frRate;          /* Frame rate */
   int frIdx;           /* Start of next frame */
}WaveRec;


int F0ExtractionFromMemory(byte* inputdata, int len, byte** outdata)
{
   int pos = 0;
   vector<Pitch> pit;
   double st_wave, en_wave;
   F0_params *f0_par = new F0_params;

   try
   {
       HcopySystemParametersIntialization();

       SetF0Params(f0_par);

       OpenSpeechFileFromMemory(inputdata, len);
       if (wv == NULL)
       {
           wv = GetPBWave(pb);
       }
       st_wave = 0.0;
       en_wave = wv->nSamples*wv->sampPeriod / 1.0e7;
       pit = get_f0(wv, f0_par, st_wave, en_wave);

       for (int i = 0; i < pit.size(); i++)
       {
           ByteDU.dvalue = pit[i].rec_F0;
           for (int j = 0; j < sizeof(double) / sizeof(byte); j++)
           {
               pitchData[pos++] = ByteDU.bytevalue[j];
           }
       }
   }
   catch (int e)
   {
       pos = 0;
   }

   // Release all of source
   delete f0_par;

   ReleaseSource();

   *outdata = pitchData;
   return pos;
}

int F0ExtractionFromFile(char* wavfile, byte** f0data)
{
   int pos = 0;
   vector<Pitch> pit;
   double st_wave, en_wave;

   HcopySystemParametersIntialization();

   F0_params *f0_par = new F0_params;
   SetF0Params(f0_par);

   OpenSpeechFile(wavfile);
   if (wv == NULL)
   {
      wv = GetPBWave(pb);
   }
   st_wave = 0.0;
   en_wave = wv->nSamples*wv->sampPeriod / 1.0e7;
   pit = get_f0(wv, f0_par, st_wave, en_wave);

   for (int i = 0; i < pit.size(); i++)
   {
      ByteDU.dvalue = pit[i].rec_F0;
      for (int j = 0; j < sizeof(double) / sizeof(byte); j++)
      {
         pitchData[pos++] = ByteDU.bytevalue[j];
      }
   }
   *f0data = pitchData;
   return pos;
}

/*
int main()
{
   char* wavfile = "D:\\code\\CAPT\\src\\Components\\FeatureExtraction\\MFCCExtractor\\TestFeature\\testdata\\01AA010C.wav";
   char* commandline = "HTKfunctions -C D:\\code\\CAPT\\src\\Components\\FeatureExtraction\\MFCCExtractor\\TestFeature\\testdata\\Get_f0.config";
   byte* returndata = NULL;
   long length = 0;
   F0ExtractionInitialization(commandline);
   int startpos = 0;
   int* pos = &startpos;
   FILE *fp = fopen(wavfile, "rb");
   fseek(fp, 0, SEEK_END);
   length = ftell(fp);
   byte*data = (byte*)malloc(length);
   fseek(fp, 0, SEEK_SET);
   fread(data, sizeof(byte), length, fp);
   fclose(fp);
   F0ExtractionFromFile(wavfile, &returndata);

   //F0ExtractionFromMemory(data,length, &returndata);

   return 0;
}
*/