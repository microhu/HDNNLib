/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */ 
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*      File: HCopy.c: Copy one Speech File to another         */
/* ----------------------------------------------------------- */

//#ifndef _HCOPY_H
//#define _HCOPY_H

char *hcopy_version = "!HVER!HCopy:   3.4.1 [CUED 12/03/09]";
char *hcopy_vc_id = "$Id: HCopy.c,v 1.1.1.1 2006/10/11 09:54:59 jal58 Exp $";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HVQ.h"
#include "HAudio.h"
#include "HParm.h"



/* -------------------------- Trace Flags & Vars ------------------------ */

#define T_TOP     001           /* basic progress reporting */
#define T_KINDS   002           /* report file formats and parm kinds */
#define T_SEGMENT 004           /* output segment label calculations */
#define T_MEM     010           /* debug memory usage */

__declspec(thread) static int  trace = 0;         /* Trace level */
typedef struct _TrList *TrPtr;  /* simple linked list for trace info */
typedef struct _TrList {      
   char *str;                   /* output string */
   TrPtr next;                  /* pointer to next in list */
} TrL;
__declspec(thread) static TrL trList;              /* 1st element in trace linked list */
__declspec(thread) static TrPtr trStr;   /* ptr to it */

static int traceWidth = 70;     /* print this many chars before wrapping ln */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */

/* ---------------------- Global Variables ----------------------- */

FileFormat srcFF = UNDEFF;   /* I/O configuration options */
FileFormat tgtFF = UNDEFF;
FileFormat srcLabFF = UNDEFF;
FileFormat tgtLabFF = UNDEFF;
ParmKind srcPK = ANON;
ParmKind tgtPK = ANON;
HTime srcSampRate = 0.0;
HTime tgtSampRate = 0.0;
Boolean saveAsVQ = FALSE;
int swidth0 = 1;

__declspec(thread) static HTime st = 0.0;            /* start of samples to copy */
__declspec(thread) static HTime en = 0.0;            /* end of samples to copy */
__declspec(thread) static HTime xMargin = 0.0;       /* margin to include around extracted labs */
static Boolean stenSet = FALSE;   /* set if either st or en set */
__declspec(thread) static int labstidx = 0;          /* label start index (if set) */
__declspec(thread) static int labenidx = 0;          /* label end index (if set) */
__declspec(thread) static int curstidx = 0;          /* label start index (if set) */
__declspec(thread) static int curenidx = 0;          /* label end index (if set) */
__declspec(thread) static int labRep = 1;            /* repetition of named label */
__declspec(thread) static int auxLab = 0;          /* auxiliary label to use (0==primary) */
static Boolean chopF = FALSE;   /* set if we should truncate files/trans */

//static LabId labName = NULL;    /* name of label to extract (if set) */
static Boolean useMLF = FALSE;    /* set if we are saving to an mlf */
static Boolean labF = FALSE;      /* set if we should  process label files too */
__declspec(thread) static char *labDir = NULL;     /* label file directory */
__declspec(thread) static char *outLabDir = NULL;  /* output label dir */
__declspec(thread) static char *labExt = "lab";    /* label file extension */

__declspec(thread) static Wave wv;                 /* main waveform; cat all input to this */
__declspec(thread) static ParmBuf pb;              /* main parmBuf; cat input, xform wv to this */
//static Transcription *trans=NULL;/* main labels; cat all input to this */
//static Transcription *tr;       /* current transcription */
__declspec(thread) static char labFile[255];       /* current source of trans */
__declspec(thread) static HTime off = 0.0;         /* length of files appended so far */

/* ---------------- Memory Management ------------------------- */

#define STACKSIZE 4500000        /* assume ~300K wave files */
__declspec(thread) static MemHeap iStack;          /* input stack */
__declspec(thread) static MemHeap oStack;          /* output stack */
__declspec(thread) static MemHeap cStack;          /* chop stack */
__declspec(thread) static MemHeap lStack;          /* label i/o  stack */
__declspec(thread) static MemHeap tStack;          /* trace list  stack */

/* ---------------- Process Command Line ------------------------- */

#define MAXTIME 1E13            /* maximum HTime (1E6 secs) for GetChkdFlt */

int Str2Arg(char * szInputStr, int &argc, char *argv[])
{
   short  nLen = strlen(szInputStr);
   int i;
   short nPosLast = -1;
   short nPos;
   argc = 0;
   for (i = 0; i <= nLen; i++){
      if (szInputStr[i] == ' ' || szInputStr[i] == '\0'){
         nPos = i;
         if (nPos - nPosLast > 1){
            strncpy(argv[argc], szInputStr + nPosLast + 1, nPos - nPosLast);
            argv[argc][nPos - nPosLast - 1] = '\0';
            argc++;
            nPosLast = nPos;
         }
         else{
            nPosLast = i;
         }
      }
   }
   return 0;
}

void ReportUsage(void)
{
   printf("\nUSAGE: HCopy [options] src [ + src ...] tgt ...\n\n");
   printf(" Option                                       Default\n\n");
   printf(" -a i     Use level i labels                  1\n");
   printf(" -e t     End copy at time t                  EOF\n");
   printf(" -i mlf   Save labels to mlf s                null\n");
   printf(" -l dir   Output target label files to dir    current\n");
   printf(" -m t     Set margin of t around x/n segs     0\n");
   printf(" -n i [j] Extract i'th [to j'th] label        off\n");
   printf(" -s t     Start copy at time t                0\n");
   printf(" -t n     Set trace line width to n           70\n");
   printf(" -x s [n] Extract [n'th occ of] label  s      off\n");
   PrintStdOpts("FGILPOX");
}

/* SetConfParms: set conf parms relevant to this tool */
void SetConfParms(void)
{
   int i;
   Boolean b;
   char buf[MAXSTRLEN];

   nParm = GetConfig("HCOPY", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfBool(cParm,nParm,"SAVEASVQ",&b)) saveAsVQ = b;
      if (GetConfInt(cParm,nParm,"NSTREAMS",&i)) swidth0 = i;
      if (GetConfStr(cParm,nParm,"SOURCEFORMAT",buf))
         srcFF = Str2Format(buf);
      if (GetConfStr(cParm,nParm,"TARGETFORMAT",buf))
         tgtFF = Str2Format(buf);
      if (GetConfStr(cParm,nParm,"SOURCEKIND",buf))
         srcPK = Str2ParmKind(buf);
      if (GetConfStr(cParm,nParm,"TARGETKIND",buf)) {
         tgtPK = Str2ParmKind(buf);
         if (tgtPK&HASNULLE) 
            HError(1019, "SetConfParms: incompatible TARGETKIND=%s for coding", buf);
      }
   }
}

/* FixOptions: Check and set config options */
void FixOptions(void)
{
   if (srcFF == UNDEFF) srcFF = HTK;
   if (tgtFF == UNDEFF) tgtFF = HTK;
   if (tgtPK == ANON) tgtPK = srcPK;
}


/* ----------------- Trace linked list handling ------------------------ */

/* AppendTrace: insert a string to trStr for basic tracing */
void AppendTrace(char *str)
{
   TrPtr tmp = trStr;

   /* Seek to end of list */
   while (tmp->str != NULL) tmp = tmp->next;
   tmp->str =  CopyString(&tStack, str);
   tmp->next = (TrPtr)New(&tStack,sizeof(trList));
   tmp->next->str = NULL;
   tmp->next->next = NULL;
}

/* PrintTrace: Print trace linked list */
void PrintTrace(void)
{
   int linelen = 0;
   TrPtr tmp = trStr;

   /* print all entries in list */
   while (tmp->next != NULL){
      printf("%s ",tmp->str);
      linelen += strlen(tmp->str) + 1;
      if (linelen > traceWidth && tmp->next->next!=NULL){
         printf("\n    ");  /* wrap line where appropriate */
         linelen = 0;
      }
      tmp = tmp->next;
   }
   if(linelen > 0) printf("\n");
}

/* ------------------- Utility Routines ------------------------ */

/* ClampStEn: set/clamp  st/en times */
void ClampStEn(HTime length, HTime *st, HTime *en)
{  
   *st -= xMargin;
   if (*st < 0) *st = 0;
   
   if( *en > 0.0 ){             /* Absolute time */
      *en += xMargin;
   }
   else if( *en < 0.0 ){        /* Relative to end */
      *en = length + *en + xMargin;
      if (*en >= length) *en = length;
      if (*en < *st) *en = *st;
      if (*st > *en) *st = *en;
   }
   else                         /* default to eof */
      *en = length - xMargin;

   /* Now clamp */
   if (*en >= length) *en = length;
   if (*en < *st) *en = *st;
   if (*st > *en) *st = *en;
}

/* ----------------- Label Manipulation ------------------------ */

/* FixLabIdxs: -ve idxs count from end, so set +ve and check */
void FixLabIdxs(int nlabs)
{
   if (labstidx<0) curstidx = nlabs + 1 + labstidx;
   else curstidx = labstidx;
   if (labenidx<0) curenidx = nlabs + 1 + labenidx;
   else curenidx = labenidx;
   if (curstidx < 0 || curstidx > nlabs)
      HError(1030,"FixLabIdxs: label start index [%d] out of range",curstidx);
   if (curenidx < curstidx || curstidx > nlabs)
      HError(1030,"FixLabIdxs: label end index  [%d] out of range",curenidx);
}


/* ----------------------- Wave File Handling ------------------------ */

/* ChopWave: return wave chopped to st and end. end = 0 means all */
Wave ChopWave(Wave srcW, HTime start, HTime end, HTime sampRate)
{
   Wave tgtW;
   HTime length;                /* HTime length of file */
   long stSamp, endSamp, nSamps;
   short *data;
   
   data = GetWaveDirect(srcW,&nSamps);
   length = nSamps * sampRate;
   if(start >= length)
      HError(1030,"ChopWave: Source too short to get data from %.0f",start); 
   ClampStEn(length,&start,&end);
   if(trace & T_SEGMENT)
      printf("ChopWave: Extracting data %.0f to %.0f\n",start,end);
   stSamp = (long) (start/sampRate);
   endSamp = (long) (end/sampRate);
   nSamps = endSamp - stSamp;
   if(nSamps <= 0)
      HError(1030,"ChopWave: Truncation options result in zero-length file"); 
   tgtW = OpenWaveOutput(&cStack,&sampRate,nSamps);
   PutWaveSample(tgtW,nSamps,data + stSamp);
   CloseWaveInput(srcW);
 //  if(chopF && labF) ChopLabs(tr,start,end);
   return(tgtW);
}

/* IsWave: check config parms to see if target is a waveform */
Boolean IsWave(char *srcFile)
{
   FILE *f;
   long nSamp,sampP, hdrS;
   short sampS,kind;
   Boolean isPipe,bSwap,isWave;
   if(tgtPK == WAVEFORM)
	   isWave=TRUE;
   else
	   isWave=FALSE;
   //isWave = tgtPK == WAVEFORM;
   if (tgtPK == ANON){
      if ((srcFF == HTK || srcFF == ESIG) && srcFile != NULL){
         if ((f=FOpen(srcFile,WaveFilter,&isPipe)) == NULL)
            HError(1011,"IsWave: cannot open File %s",srcFile);
         switch (srcFF) {
         case HTK:
            if (!ReadHTKHeader(f,&nSamp,&sampP,&sampS,&kind,&bSwap))
               HError(1013, "IsWave: cannot read HTK Header in File %s",
                      srcFile);
            break;
         case ESIG:
            if (!ReadEsignalHeader(f, &nSamp, &sampP, &sampS,
                                   &kind, &bSwap, &hdrS, isPipe))
               HError(1013, "IsWave: cannot read Esignal Header in File %s",
                      srcFile);             
            break;
         }
		 if(kind==WAVEFORM)
			 isWave=TRUE;
		 else
			 isWave=FALSE;
      //   isWave = kind == WAVEFORM;
         FClose(f,isPipe);
      } else
         isWave = TRUE;
   }
   return isWave;
}

/* OpenWaveFile: open source wave file and extract portion if indicated */
HTime OpenWaveFile(char *src)
{
   Wave w, cw;
   long nSamps;
   short *data;

   if((w = OpenWaveInput(&iStack,src,srcFF,0,0,&srcSampRate))==NULL)
      HError(1013,"OpenWaveFile: OpenWaveInput failed");
   srcPK = WAVEFORM;
   tgtSampRate = srcSampRate;
   cw = (chopF)?ChopWave(w,st,en,srcSampRate) : w;
   data = GetWaveDirect(cw,&nSamps);
   wv = OpenWaveOutput(&oStack, &srcSampRate, nSamps);
   PutWaveSample(wv,nSamps,data);
   CloseWaveInput(cw);
   return(nSamps*srcSampRate);
}

HTime OpenWaveFileFromMemory(byte* inputdata, int len)
{
   Wave w, cw;
   long nSamps;
   short *data;

   if ((w = OpenWaveInputFromMemory(&iStack, inputdata, len, srcFF, 0, 0, &srcSampRate)) == NULL)
      HError(1013, "OpenWaveFileFromMemory: OpenWaveInput failed");
   srcPK = WAVEFORM;
   tgtSampRate = srcSampRate;
   cw = (chopF) ? ChopWave(w, st, en, srcSampRate) : w;
   data = GetWaveDirect(cw, &nSamps);
   wv = OpenWaveOutput(&oStack, &srcSampRate, nSamps);
   PutWaveSample(wv, nSamps, data);
   CloseWaveInput(cw);
   return(nSamps*srcSampRate);
}

/* AppendWave: append the src file to global wave wv */
HTime AppendWave(char *src)
{
   Wave w, cw;
   HTime period=0.0;
   long nSamps;
   short *data;

   if((w = OpenWaveInput(&iStack,src, srcFF, 0, 0, &period))==NULL)
      HError(1013,"AppendWave: OpenWaveInput failed");
   if(trace & T_KINDS )
      printf("Appending file %s format: %s [WAVEFORM]\n",src,
             Format2Str(WaveFormat(w)));   
   if(period != srcSampRate)
      HError(1032,"AppendWave: Input file %s has inconsistent sampling rate",src);
   cw = (chopF)? ChopWave(w,st,en,srcSampRate) : w;
   data = GetWaveDirect(cw,&nSamps);
   PutWaveSample(wv,nSamps,data);
   CloseWaveInput(cw);
   return(nSamps*period);
}

/* ----------------------- Parm File Handling ------------------------ */

/* ChopParm: return parm chopped to st and end. end = 0 means all */
ParmBuf ChopParm(ParmBuf b, HTime start, HTime end, HTime sampRate)
{  
   int stObs, endObs, nObs, i;
   HTime length;
   short swidth[SMAX];
   Boolean eSep;
   ParmBuf cb;
   Observation o;
   BufferInfo info;

   length =  ObsInBuffer(b) * sampRate;
   ClampStEn(length,&start,&end);
   if(start >= length)
      HError(1030,"ChopParm: Src file too short to get data from %.0f",start);
   if(trace & T_SEGMENT)
      printf("ChopParm: Extracting segment %.0f to %.0f\n",start,end);
   stObs = (int) (start/sampRate);
   endObs = (int) (end/sampRate);
   nObs = endObs -stObs;
   if(nObs <= 0)
      HError(1030,"ChopParm: Truncation options result in zero-length file");
   GetBufferInfo(b,&info);
   ZeroStreamWidths(swidth0,swidth);
   SetStreamWidths(tgtPK,info.tgtVecSize,swidth,&eSep);
   o = MakeObservation(&cStack, swidth, info.tgtPK, saveAsVQ, eSep);
   if (saveAsVQ){
      if (info.tgtPK&HASNULLE){
         info.tgtPK=DISCRETE+HASNULLE;
      }else{
         info.tgtPK=DISCRETE;
      }
   }
   cb =  EmptyBuffer(&cStack, nObs, o, info);
   for (i=stObs; i < endObs; i++){
      ReadAsTable(b, i, &o);
      AddToBuffer(cb, o);
   }
   CloseBuffer(b);

   return(cb);
}

/* AppendParm: append the src file to current Buffer pb. Return appended len */
HTime AppendParm(char *src)
{  
   int i;
   char bf1[MAXSTRLEN]; 
   char bf2[MAXSTRLEN]; 
   short swidth[SMAX];
   Boolean eSep;
   ParmBuf b, cb;
   Observation o;
   BufferInfo info;

   if((b =  OpenBuffer(&iStack,src,0,srcFF,TRI_UNDEF,TRI_UNDEF))==NULL)
      HError(1050,"AppendParm: Config parameters invalid");
   GetBufferInfo(b,&info);
   if(trace & T_KINDS ){
      printf("Appending file %s format: %s [%s]->[%s]\n",src,
             Format2Str(info.srcFF), ParmKind2Str(info.srcPK,bf1),
             ParmKind2Str(info.tgtPK,bf2));
   }
   if  (tgtSampRate != info.tgtSampRate)
      HError(1032,"AppendParm: Input file %s has inconsistent sample rate",src);
   if ( BaseParmKind(tgtPK) != BaseParmKind(info.tgtPK))
      HError(1032,"AppendParm: Input file %s has inconsistent tgt format",src);
   cb = (chopF)?ChopParm(b,st,en,info.tgtSampRate) : b;
   ZeroStreamWidths(swidth0,swidth);
   SetStreamWidths(info.tgtPK,info.tgtVecSize,swidth,&eSep);
   o = MakeObservation(&iStack, swidth, info.tgtPK, saveAsVQ, eSep);
   for (i=0; i < ObsInBuffer(cb); i++){
      ReadAsTable(cb, i, &o);
      AddToBuffer(pb, o);
   }
   CloseBuffer(cb);
   return(i*info.tgtSampRate);
}

/* OpenParmFile: open source parm file and return length */
HTime OpenParmFile(char *src)
{
   int i;
   ParmBuf b, cb;
   short swidth[SMAX];
   Boolean eSep;
   Observation o;
   BufferInfo info;

   if((b =  OpenBuffer(&iStack,src,0,srcFF,TRI_UNDEF,TRI_UNDEF))==NULL)
      HError(1050,"OpenParmFile: Config parameters invalid");
   GetBufferInfo(b,&info);
   srcSampRate = info.srcSampRate;
   tgtSampRate = info.tgtSampRate;
   srcPK = info.srcPK; tgtPK = info.tgtPK;
   cb = chopF?ChopParm(b,st,en,info.tgtSampRate):b;
   ZeroStreamWidths(swidth0,swidth);
   SetStreamWidths(info.tgtPK,info.tgtVecSize,swidth,&eSep);
   o = MakeObservation(&oStack, swidth, info.tgtPK, saveAsVQ, eSep);
   if (saveAsVQ){
      if (info.tgtPK&HASNULLE){
         info.tgtPK=DISCRETE+HASNULLE;
      }else{
         info.tgtPK=DISCRETE;
      }
   }
   pb =  EmptyBuffer(&oStack, ObsInBuffer(cb), o, info);
   for(i=0; i < ObsInBuffer(cb); i++){
      ReadAsTable(cb, i, &o);
      AddToBuffer(pb, o);
   }
   CloseBuffer(cb);
   if( info.nSamples > 0 )
      return(info.nSamples*srcSampRate);
   else
      return(ObsInBuffer(pb)*info.tgtSampRate);
}

/* OpenParmFileFromMemory: open source parm file from memory and return length */
HTime OpenParmFileFromMemory(byte* inputdata, int len)
{
   int i;
   ParmBuf b, cb;
   short swidth[SMAX];
   Boolean eSep;
   Observation o;
   BufferInfo info;

   if ((b = OpenBufferFromMemory(&iStack, inputdata, len, 0, srcFF, TRI_UNDEF, TRI_UNDEF)) == NULL)
      HError(1050, "OpenParmFileFromMemory: Config parameters invalid");
   GetBufferInfo(b, &info);
   srcSampRate = info.srcSampRate;
   tgtSampRate = info.tgtSampRate;
   srcPK = info.srcPK; tgtPK = info.tgtPK;
   cb = chopF ? ChopParm(b, st, en, info.tgtSampRate) : b;
   ZeroStreamWidths(swidth0, swidth);
   SetStreamWidths(info.tgtPK, info.tgtVecSize, swidth, &eSep);
   o = MakeObservation(&oStack, swidth, info.tgtPK, saveAsVQ, eSep);
   if (saveAsVQ){
      if (info.tgtPK&HASNULLE){
         info.tgtPK = DISCRETE + HASNULLE;
      }
      else{
         info.tgtPK = DISCRETE;
      }
   }
   pb = EmptyBuffer(&oStack, ObsInBuffer(cb), o, info);
   for (i = 0; i < ObsInBuffer(cb); i++){
      ReadAsTable(cb, i, &o);
      AddToBuffer(pb, o);
   }
   CloseBuffer(cb);

   if (info.nSamples > 0)
      return(info.nSamples*srcSampRate);
   else
      return(ObsInBuffer(pb)*info.tgtSampRate);
}

/* --------------------- Speech File Handling ---------------------- */

/* OpenSpeechFile: open waveform or parm file */
void OpenSpeechFile(char *s)
{
   HTime len;
   char buf[MAXSTRLEN];

   if (tgtPK == WAVEFORM)
      len = OpenWaveFile(s);
   else
      len = OpenParmFile(s);

   // if(labF) AppendLabs(tr,len);
   if (trace & T_TOP) AppendTrace(s);
   if (tgtPK == ANON) tgtPK = srcPK;
   if (trace & T_KINDS){
      printf("Source file format: %s [%s]\n",
         Format2Str(srcFF), ParmKind2Str(srcPK, buf));
      printf("Target file format: %s [%s]\n",
         Format2Str(tgtFF), ParmKind2Str(tgtPK, buf));
      printf("Source rate: %.0f Target rate: %.0f \n",
         srcSampRate, tgtSampRate);
   }
}

/* OpenSpeechFileFromMemory: open waveform or parm file */
void OpenSpeechFileFromMemory(byte* inputdata, int ilen)
{
   HTime len;
   char buf[MAXSTRLEN];

   if (tgtPK == WAVEFORM)
      len = OpenWaveFileFromMemory(inputdata, ilen);
   else 
      len = OpenParmFileFromMemory(inputdata, ilen);

   // if(labF) AppendLabs(tr,len);
   if (tgtPK == ANON) tgtPK = srcPK;
   if (trace & T_KINDS){
      printf("Source file format: %s [%s]\n",
         Format2Str(srcFF), ParmKind2Str(srcPK, buf));
      printf("Target file format: %s [%s]\n",
         Format2Str(tgtFF), ParmKind2Str(tgtPK, buf));
      printf("Source rate: %.0f Target rate: %.0f \n",
         srcSampRate, tgtSampRate);
   }
}

/* AppendSpeechFile: open waveform or parm file */
void AppendSpeechFile(char *s)
{
   HTime len;

   // if (labF) tr = LoadTransLabs(s);
   if (tgtPK == WAVEFORM)
      len = AppendWave(s);
   else
      len = AppendParm(s);
   if (labF){
      //    AppendLabs(tr,len);
   }
   if (trace & T_TOP) {
      AppendTrace("+"); AppendTrace(s);
   }
}

/* PutTargetFile: close and store waveform or parm file */
void PutTargetFile(char *s)
{
   if (tgtPK == WAVEFORM) {
      if (CloseWaveOutput(wv, tgtFF, s) < SUCCESS)
         HError(1014, "PutTargetFile: Could not save waveform file %s", s);
   }
   else {
      if (SaveBuffer(pb, s, tgtFF) < SUCCESS)
         HError(1014, "PutTargetFile: Could not save parm file %s", s);
      CloseBuffer(pb);
   }
   if (trace & T_TOP){
      AppendTrace("->"); AppendTrace(s);
      PrintTrace();
      ResetHeap(&tStack);
      trList.str = NULL;
   }
   // if(trans != NULL)
   //   SaveLabs(s,trans);
}

void PutTargetToArray(byte* mfcdata, int *pos)
{
   if (tgtPK != WAVEFORM)
   {
      if (SaveBufferToArray(pb, mfcdata, pos, tgtFF) < SUCCESS)
         HError(1014, "PutTargetToArray: Could not save parm to array");
      CloseBuffer(pb);
   }
}

void HcopyConfigParametersIntialization(int argc, char*argv[])
{
	if (InitShell(argc, argv, hcopy_version, hcopy_vc_id) < SUCCESS)
		HError(1000, "HCopy: InitShell failed");
	InitMem();
	InitMath();
	InitSigP();
	InitWave();
	InitAudio();
	InitVQ();

	if (InitParm() < SUCCESS)
		HError(1000, "HCopy: InitParm failed");

	SetConfParms();
}

void HcopySystemParametersIntialization()
{
   InitMem_gstack();
   InitSigPHeap();
   /* initial trace string is null */
   trList.str = NULL;
   trStr = &trList;

   CreateHeap(&iStack, "InBuf", MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   CreateHeap(&oStack, "OutBuf", MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   CreateHeap(&cStack, "ChopBuf", MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   CreateHeap(&lStack, "LabBuf", MSTAK, 1, 0.0, 10000, LONG_MAX);
   CreateHeap(&tStack, "Trace", MSTAK, 1, 0.0, 100, 200);

   FixOptions();
}

void ReleaseSource()
{
	// Release Source
	DeleteHeap(&lStack);
	DeleteHeap(&iStack);
	DeleteHeap(&oStack);
	DeleteHeap(&cStack);
	DeleteHeap(&tStack);

    UnInitSigPHeap();
	UnInitMem_gstack();

	// Release trace
	if (trStr != NULL)
	{
		TrPtr tmpBack = trStr;
		TrPtr tmpFore = trStr->next;

		/* Seek to end of list */
		while ((tmpFore!= NULL)&& (tmpFore->str != NULL) && (tmpFore->next != NULL))
		{
			free(tmpBack->str);
			free(tmpBack->next);

			tmpBack = tmpFore;
			tmpFore = tmpFore->next;
		}

		free(tmpBack->str);
		free(tmpBack->next);

		trStr = NULL;
	}
}

void UnInitialization()
{
	ResetShell();
	UnInitSigP();
	UnInitMem();
	UnInitVQ();
	UnInitParm();
}