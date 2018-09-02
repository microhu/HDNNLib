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
 * Written by:  David Talkin
 * Checked by:
 * Revised by:
 * @(#)f0_structs.h	1.7 9/9/96 ERL
 * Brief description:
 *
 */


/* f0_structs.h */

#define BIGSORD 100

typedef struct pitch_contour { /* for storing the pitch contour information */
  double rec_F0;              /* fundamental frequency estimate */
  double rec_pv;              /* probability of voicing */
  double rec_rms;             /* local root mean squared measurements */
  double rec_acp;             /* the peak normalized cross-correlation value 
                                 that was found to determine the output F0 */

  //add the following 4 feature as the related f0 feature , data: 07,10,31 by feng zhang
  double rec_deltaF0; //delta F0
  double rec_accF0; //accelate F0
  double rec_VoiceIndex;//voice index
  double rec_longSpanF0;//long span f0
} Pitch;

typedef struct cross_rec { /* for storing the crosscorrelation information */
	float	rms;	/* rms energy in the reference window */
	float	maxval;	/* max in the crosscorr. fun. q15 */
	short	maxloc; /* lag # at which max occured	*/
	short	firstlag; /* the first non-zero lag computed */
	float	*correl; /* the normalized corsscor. fun. q15 */
} Cross;

typedef struct dp_rec { /* for storing the DP information */
	short	ncands;	/* # of candidate pitch intervals in the frame */
	short	*locs; /* locations of the candidates */
	float	*pvals; /* peak values of the candidates */
	float	*mpvals; /* modified peak values of the candidates */
	short	*prept; /* pointers to best previous cands. */
	float	*dpvals; /* cumulative error for each candidate */
} Dprec;

typedef struct windstat_rec {  /* for lpc stat measure in a window */
    float rho[BIGSORD+1];
    float err;
    float rms;
} Windstat;

typedef struct sta_rec {  /* for stationarity measure */
  float *stat;
  float *rms;
  float *rms_ratio;
} Stat;


typedef struct frame_rec{
  Cross *cp;
  Dprec *dp;
  float rms;
  struct frame_rec *next;
  struct frame_rec *prev;
} Frame;

extern   Frame * alloc_frame(register int nlags, register int ncands);


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
 * Written by:  David Talkin
 * Checked by:
 * Revised by:
 * @(#)f0.h	1.4 9/9/96 ERL
 * Brief description:
 *
 */


/* f0.h */
/* Some definitions used by the "Pitch Tracker Software". */

typedef struct f0_params {
float cand_thresh,	/* only correlation peaks above this are considered */
      lag_weight,	/* degree to which shorter lags are weighted */
      freq_weight,	/* weighting given to F0 trajectory smoothness */
      trans_cost,	/* fixed cost for a voicing-state transition */
      trans_amp,	/* amplitude-change-modulated VUV trans. cost */
      trans_spec,	/* spectral-change-modulated VUV trans. cost */
      voice_bias,	/* fixed bias towards the voiced hypothesis */
      double_cost,	/* cost for octave F0 jumps */
      mean_f0,		/* talker-specific mean F0 (Hz) */
      mean_f0_weight,	/* weight to be given to deviations from mean F0 */
      min_f0,		/* min. F0 to search for (Hz) */
      max_f0,		/* max. F0 to search for (Hz) */
      frame_step,	/* inter-frame-interval (sec) */
      wind_dur;		/* duration of correlation window (sec) */
int   n_cands,		/* max. # of F0 cands. to consider at each frame */
      conditioning; /* Specify optional signal pre-conditioning. */
} F0_params;

/* Possible values returned by the function f0(). */
#define F0_OK		0
#define F0_NO_RETURNS	1
#define F0_TOO_FEW_SAMPLES	2
#define F0_NO_INPUT	3
#define F0_NO_PAR	4
#define F0_BAD_PAR	5
#define F0_BAD_INPUT	6
#define F0_INTERNAL_ERR	7

/* Bits to specify optional pre-conditioning of speech signals by f0() */
/* These may be OR'ed together to specify all preprocessing. */
#define F0_PC_NONE	0x00		/* no pre-processing */
#define F0_PC_DC	0x01		/* remove DC */
#define F0_PC_LP2000	0x02		/* 2000 Hz lowpass */
#define F0_PC_HP100	0x04		/* 100 Hz highpass */
#define F0_PC_AR	0x08		/* inf_order-order LPC inverse filter */
#define F0_PC_DIFF	0x010		/* 1st-order difference */

/* Definitions of mathematical constants */
#define M_PI       3.14159265358979323846

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

void SetF0Params(F0_params *par);
extern int lpc(int,float,int,float*,float*,float*,float*,float*,float*,float,int);
extern void get_fast_cands(float *fdata, float *fdsdata, int ind, int step, int size, int dec, 
                    int start, int nlags, float *engref, int *maxloc, float *maxval, Cross *cp, 
                    float *peaks, int *locs, int *ncand, F0_params *par);
extern void a_to_aca(float*,float*,float*,int);
extern void crossf(float*,int,int,int,float*,int*,float*,float*);
extern void crossfi(float*,int,int,int,int,float*,int*,float*,float*,int*,int);

#define Fprintf (void)fprintf
