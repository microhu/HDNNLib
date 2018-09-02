
/*
 * This material contains unpublished, proprietary software of 
 * Entropic Research Laboratory, Inc. Any reproduction, distribution, 
 * or publication of this work must be authorized in writing by Entropic 
 * Research Laboratory, Inc., and must bear the notice: 
 *
 *    "Copyright (c) 1990-1996 Entropic Research Laboratory, Inc. 
 *                   All rights reserved"
 *
 * The copyright notice above does not evidence any actual or intended 
 * publication of this source code.     
 *
 * Written by:  Derek Lin
 * Checked by:
 * Revised by:  David Talkin
 *
 * Brief description:  Estimates F0 using normalized cross correlation and
 *   dynamic programming.
 *
 */
static char *sccs_id = "@(#)get_f0.c	1.14	10/21/96	ERL";

#include "get_f0.h"

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

char	    *ProgName = "get_f0";
static char *Version = "1.14";
static char *Date = "10/21/96";
/*
extern double g_CandThresh,g_LagWeight,g_FreqWeight,g_TransCost,g_TransAmp,
    g_TransSpec,g_VoiceBias,g_DoubleCost,g_MinF0,g_MaxF0,g_FrameStep,
    g_WindDur,g_MeanF0,g_MeanF0Weight,g_StartTime,g_EndTime;
extern int g_nCands,g_Conditioning,g_Trace;
*/
std::vector<Pitch> get_f0(Wave w, F0_params *par, double start_time, double end_time)
{
  float *fdata;
  float *f0p, *vuvp, *rms_speech, *acpkp;
  int done, ndone=0, vecsize;
  int sf;
  long buff_size, actsize, s_rec, e_rec, sdstep=0, total_samps;
  int init_dp_f0(int,F0_params*,long*,long*),check_f0_params(F0_params*,int),
      dp_f0(float*,int,int,int,F0_params*,float**,float**,float**,float**,int*,int);
  void free_dp_f0();
  void Free_Get_Cands();

  std::vector<Pitch> pit;
  Pitch p;

  sf = (int)(1.0e7/w->sampPeriod);
  if (sf == 0.0) {
    Fprintf(stderr, "%s: no sampling frequency---exiting.\n", ProgName);
    exit(1);
  }
  if(check_f0_params(par, sf)){
    Fprintf(stderr, "%s: invalid/inconsistent parameters -- exiting.\n",
	    ProgName);
    exit(1);
  }

  s_rec = (int)(start_time * sf + 0.5);
  e_rec = (int)(end_time * sf + 0.5);
  if (s_rec < 0) s_rec = 0;
  if (e_rec >= (w->nSamples - 1) || e_rec < 0)
      e_rec = w->nSamples - 1;
  if (s_rec > e_rec) return pit;

  total_samps = e_rec - s_rec + 1;
  if(total_samps < ((par->frame_step * 2.0) + par->wind_dur) * sf) {
    // at least 3 frames
    Fprintf(stderr, "%s: input range too small for analysis by get_f0.\n",
	    ProgName);
    exit(1);
  }

  /* Initialize variables in get_f0.c; allocate data structures;
   * determine length and overlap of input frames to read.
   */
  if (init_dp_f0(sf, par, &buff_size, &sdstep)
      || buff_size > INT_MAX || sdstep > INT_MAX)
  {
    Fprintf(stderr, "%s: problem in init_dp_f0().\n", ProgName);
    exit(1);
  }


  if (buff_size > total_samps)
    buff_size = total_samps;

  actsize = min(buff_size, w->nSamples);
  fdata = new float[max(buff_size, sdstep)];
  ndone = s_rec;

  float *fd;
  short *sd;
  int i;
  while (TRUE) {

    done = (actsize < buff_size) || (total_samps == buff_size);
    for (i=0,fd=fdata,sd=w->data+ndone; i<actsize; i++)
      *fd++ = *sd++;

    if (dp_f0(fdata, (int) actsize, (int) sdstep, sf, par,
	      &f0p, &vuvp, &rms_speech, &acpkp, &vecsize, done)) {
      Fprintf(stderr, "%s: problem in dp_f0().\n", ProgName);
      exit(1);
    }

    for (i = vecsize - 1; i >= 0; i--) {
        p.rec_F0 = f0p[i];
        p.rec_pv = vuvp[i];
        p.rec_rms = rms_speech[i];
        p.rec_acp = acpkp[i];
        pit.push_back (p);
    }

    if (done)
      break;
    
    ndone += sdstep;
    actsize = min(buff_size, w->nSamples-ndone);
    total_samps -= sdstep;

    if (actsize > total_samps)
      actsize = total_samps;
  }

  Free_Get_Cands();
  delete fdata;
  free_dp_f0();
  return pit;
}

void SetF0Params(F0_params *par)
{
  par->cand_thresh = 0.3F;//(float)g_CandThresh;
  par->lag_weight = 0.3F;//(float)g_LagWeight;
  par->freq_weight = 0.02F;//(float)g_FreqWeight;
  par->trans_cost = 0.005F;//(float)g_TransCost;
  par->trans_amp = 0.5F;//(float)g_TransAmp;
  par->trans_spec = 0.5F;//(float)g_TransSpec;
  par->voice_bias =0.0F;// (float)g_VoiceBias;
  par->double_cost = 0.35F;//(float)g_DoubleCost;
  par->min_f0 = 50;//(float)g_MinF0;
  par->max_f0 = 400;//(float)g_MaxF0;
  par->frame_step = 0.01F;//(float)g_FrameStep;
  par->wind_dur = 0.0075F;//(float)g_WindDur;
  par->n_cands = 20;//g_nCands;
  par->mean_f0 =200;// (float)g_MeanF0;     /* unused */
  par->mean_f0_weight =0.0F;// (float)g_MeanF0Weight;  /* unused */
  par->conditioning =0;// g_Conditioning;    /*unused */
  
}

/*
 * Some consistency checks on parameter values.
 * Return a positive integer if any errors detected, 0 if none.
 */

int check_f0_params(register F0_params *par, register int sample_freq)
{
  int	  error = 0;
  double  dstep;

  if((par->cand_thresh < 0.01) || (par->cand_thresh > 0.99)) {
    Fprintf(stderr,
	    "%s: ERROR: cand_thresh parameter must be between [0.01, 0.99].\n",
	    ProgName);
    error++;
  }
  if((par->wind_dur > .1) || (par->wind_dur < .0001)) {
    Fprintf(stderr,
	    "ERROR: wind_dur parameter must be between [0.0001, 0.1].\n",
	    ProgName);
    error++;
  }
  if((par->n_cands > 100) || (par->n_cands < 3)){
    Fprintf(stderr,
	    "%s: ERROR: n_cands parameter must be between [3,100].\n",
	    ProgName); 
    error++;
  }
  if((par->max_f0 <= par->min_f0) || (par->max_f0 >= (sample_freq/2.0)) ||
     (par->min_f0 < (sample_freq/10000.0))){
    Fprintf(stderr,
	    "%s: ERROR: min(max)_f0 parameter inconsistent with sampling frequency.\n",
	    ProgName); 
    error++;
  }
  dstep = ((double)((int)(0.5 + (sample_freq * par->frame_step))))/sample_freq;
  if(dstep != par->frame_step) {
  //  if(g_Trace)
      Fprintf(stderr,
	      "%s: Frame step set to %f to exactly match signal sample rate.\n",
	      ProgName, dstep);
    par->frame_step = (float)dstep;
  }
  if((par->frame_step > 0.1) || (par->frame_step < (1.0/sample_freq))){
    Fprintf(stderr,
	    "%s: ERROR: frame_step parameter must be between [1/sampling rate, 0.1].\n",
	    ProgName); 
    error++;
  }

  return(error);
}
  

