// matrixquantization.h -- code for quantizing (gradient) matrices. This code is shared between CPU (MPI) and GPU code
//
// F. Seide, Dec 2013
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/matrixquantization.h $
// 
// 77    6/17/14 11:35a Fseide
// switched to ZERO_THRESHOLD_FOR_1BIT mode by default
// 
// 76    6/13/14 6:18p Fseide
// assuming zero mean for 1-bit quantization estimate (but not actual
// threshold)
// 
// 75    6/13/14 10:56a Fseide
// zero-mean quantization now has a #define so that we can log about it
// elsewhere
// 
// 74    6/11/14 7:17p Fseide
// comment
// 
// 73    4/30/14 9:34a Fseide
// valuequantizer: set the threshold for "zero" 10^6 times smaller
// 
// 72    3/19/14 9:09a Fseide
// added code branch for NORESIDUAL tests
// 
// 71    2/07/14 17:25 Fseide
// assert() caused a problem when compiling in CUDA, so we defined it away
// for CUDA builds
// 
// 70    2/03/14 11:32 Fseide
// enabled tying of 1-bit reconstruction values to enforce symmetry
// 
// 69    1/26/14 18:54 Fseide
// disabled PPL for now (does not seem to help too much)
// 
// 68    1/23/14 18:20 Fseide
// enabled PPL--seemed to help a little on the GCD HPC farm
// 
// 67    1/23/14 5:50p Fseide
// optimized quantization a little (did not help much)
// 
// 66    1/23/14 5:16p Fseide
// using PPL for unquant
// 
// 65    1/23/14 5:03p Fseide
// optimized unquant for 1-bit altlayout (the critical case), about 30%
// speed-up, not enough
// 
// 64    1/23/14 4:22p Fseide
// computerange() now assumes zero mean, to save one data pass (accuracy
// impact to be verified experimentally)
// 
// 63    1/22/14 13:10 Fseide
// centralized the 'accuracy' variable into computerange() and renamed it
// to 'stddevs' (this fixed a bug since its value was inconsistent between
// CPU and GPU code)
// 
// 62    1/22/14 10:20 Fseide
// fixed a matrix access of the end element which was correct but
// triggered an out-of-bounds access anyway
// 
// 61    1/21/14 11:49 Fseide
// bug fix in skipping the recomputation: forgot to scale to #nodes;
// 'accuracy' now set to 5 stddevs (before: 2), big difference for 16 bit
// in early iterations;
// bg thread disabled for now to get better-comparable log output
// 
// 60    1/21/14 10:36 Fseide
// added testing code for clipping the residual (disabled)
// 
// 59    1/20/14 9:35 Fseide
// (avoiding div by 0 in quant-32 case, does not really matter)
// 
// 58    1/18/14 19:21 Fseide
// fix of >> overflow in computing rangeend (which would cause a bit mask
// to be 0, in an unfortunate sequence of events)
// 
// 57    1/18/14 18:21 Fseide
// comments;
// new methods lsl() and lsr() (not used yet)
// 
// 56    1/17/14 5:12p V-haofu
// add 32-bit case in quantize() and unquantize() for debugging (we
// actually return the float value that encoded in int form for
// quantization)
// 
// 55    1/17/14 8:53 Fseide
// (moved the ISSUE TRACKER comment to parallelrbmmatrix.h)
// 
// 54    1/16/14 20:47 Fseide
// made 'qbwordbits' a (static) class member once again since I used it
// all the time
// 
// 53    1/16/14 20:38 Fseide
// presumably fixed the altlayout issue, to be tested
// 
// 52    1/16/14 19:25 Fseide
// (fixed a broken #ifdef)
// 
// 51    1/16/14 18:44 Fseide
// (added an issue to track)
// 
// 50    1/16/14 18:31 Fseide
// fixed unquantize() for altlayout (but not yet quantize())
// 
// 49    1/16/14 16:18 Fseide
// towards fixing the altlayout problem (not completed)
// 
// 48    1/16/14 12:35 Fseide
// added another #define for testing
// 
// 47    1/16/14 12:15 Fseide
// added #defines to control the testing of altlayout (no functional
// change, altlayout remains disabled)
// 
// 46    1/16/14 11:51 Fseide
// bug fix: quantize1() should use >= not > for consistency;
// temporarily disabled altlayout since it does not give the same result
// 
// 45    1/16/14 11:17 Fseide
// added issue tracker
// 
// 44    1/15/14 18:59 Fseide
// moved masktable[] from unquantizeqbword1() into a new method
// float4::maskby(size_t) with truly static intialization of the table
// 
// 43    1/15/14 18:25 Fseide
// (minor optimization of unquantizeqbword1())
// 
// 42    1/15/14 18:14 Fseide
// 1-bit optimized version of unquantizeqbword()
// 
// 41    1/15/14 17:31 Fseide
// added specially optimized branch for 1 bit for quantizeqbword() (but
// makes no difference on GPU)
// 
// 40    1/14/14 20:06 Fseide
// SSE version of unquantizeqbword1()
// 
// 39    1/14/14 19:01 Fseide
// (removed a check in unquantize(), CPU-side)
// 
// 38    1/14/14 18:35 Fseide
// now skipping second computerange() (instead, we reuse the range
// determined on our local stripe)--to be verified once we have multiple
// nodes
// 
// 37    1/14/14 18:11 Fseide
// more efficient unquantize() (with alt layout)
// 
// 36    1/14/14 17:38 Fseide
// SSE implementation of quantizeqbword1()
// 
// 35    1/14/14 15:17 Fseide
// optimized the 1-bit version for CPU, twice as fast but not enough
// 
// 34    1/14/14 11:08 Fseide
// (un)quantize1bitaltlayout() implemented;
// computerange() tidied up (no more i0/i1);
// some functions force-inlined
// 
// 33    1/14/14 10:12 Fseide
// cleaned up quantization-related CPU-side interfaces to assume a matrix
// patch to specify the rect dimension, rather than passing the dims
// around all the time (in prep for SSE-based quantization, since now we
// can enforce alignment)
// 
// 32    1/14/14 9:48 Fseide
// added ssematrix::patch;
// towards using this in (un)quantize() (for SSE-optimizing quantization)
// 
// 31    1/14/14 8:38 Fseide
// towards using patches for base matrix, for use in quantize()
// 
// 30    1/13/14 16:50 Fseide
// tidied up CUDA computerange()
// 
// 29    1/13/14 16:39 Fseide
// again using columnquantizer::computerange() in CUDA code, after
// extending it with reducer lambdas :)
// 
// 28    1/13/14 15:40 Fseide
// implemented parallel version of computerange(), but still has some
// problems
// 
// 27    1/13/14 14:18 Fseide
// clean-up of 'qbits' (renamed to qbword, fixed variable names)
// 
// 26    1/13/14 13:50 Fseide
// new struct quantizedcolumn to represent the header and the bit data for
// one
// 
// 25    1/13/14 13:11 Fseide
// tidied up per-int quantization
// 
// 24    1/13/14 10:48 Fseide
// special code branch for 1-bit
// 
// 23    1/13/14 10:25 Fseide
// (made qbwordbits a static const member)
// 
// 22    1/13/14 10:19 Fseide
// further factoring of int-val based loop
// 
// 21    1/13/14 9:28 Fseide
// recast quantization as loop over 32-bit ints, for better CUDA collation
// (to be tested and cleaned up)
// 
// 20    1/11/14 15:20 Fseide
// hardened quantizer class against zero intervals and all-zero input
// vectors
// 
// 19    1/10/14 20:08 Fseide
// (comments)
// 
// 18    1/10/14 16:51 Fseide
// two bug fixes in bit fiddling for quantization
// 
// 17    1/10/14 15:01 Fseide
// (forgot to add i0,i1 args to one function)
// 
// 16    1/10/14 13:31 Fseide
// bug fix: column quantizer now takes a row range (needed to support
// quantizing sub-stripes of the bias vectors)
// 
// 15    1/10/14 11:48 Fseide
// CPU-side matrix quantization can now pass a patch region;
// agg residual now allocated per stripe, in order to match the dimensions
// and layout of agg accumulator
// 
// 14    1/10/14 10:48 Fseide
// (split quantizer into 3 classes)
// 
// 13    1/09/14 16:41 Fseide
// unquantizeandaggregatestripe() implemented
// 
// 12    1/09/14 16:34 Fseide
// implemented full-matrix (un)quantization code, now correctly as a
// 'static' function (previous version was bogus)
// 
// 11    1/09/14 16:05 Fseide
// added whole-matrix versions of quantize()/unquantize() to class
// quantizer;
// unquantize() now allows to add the unquantized column to the existing
// matrix, instead of overwriting it
// 
// 10    1/09/14 13:53 Fseide
// (renamed in/outresidual to cur/newresidual)
// 
// 9     1/09/14 13:52 Fseide
// quantization now peruses the residual in-place (instead of first adding
// it explicitly to the raw gradient)--better separation of concerns
// (residual belongs to quantization);
// bug fix in quantizeandfetchqstripe(): forgot to apply the patch to the
// residual
// 
// 8     1/08/14 9:44 Fseide
// qstripe no longer knows about patch dimension and 'bits' parameter
// (none of its business)
// 
// 7     1/03/14 17:36 Fseide
// fixed hack for mean0/1 -> lower/upper per Hao's discovery of the bug
// 
// 6     1/03/14 13:00 Fseide
// moved ld() to quantized class
// 
// 5     1/03/14 12:34 Fseide
// moved unquantize() to matrixquantization.h
// 
// 4     1/03/14 12:18 Fseide
// moved CPU/GPU-sharable quantization code to matrixquantization.h
// 
// 3     1/03/14 12:19a V-haofu
// change the initialization of quantizer
// 
// 2     12/20/13 15:03 Fseide
// added the quantizer class here (as a dummy copy, not used yet)
// 
// 1     12/20/13 14:50 Fseide
// added new module (inline header) for the shared quantization code

#pragma once

#ifdef __device__  // this can be used in CUDA; if this is not defined, then we are compiling in a non-CUDA context
#define cudacode       __device__           // CUDA: we assume we ONLY run these functions on CUDA (otherwise we'd need to mess with specifiers of matrixref)
#define cudasharedcode __device__ __host__  // shared on both CUDA and CPU; note that such functions cannot call into __device__ only functions like matrixref::operator(,)
#undef assert
#define assert(c)
#else
#define cudacode  // non-CUDA context: defines to nothing
#define cudasharedcode
//#define QUANTUSEPPL
#endif

#ifdef QUANTUSEPPL
#include <ppl.h>    // in non-CUDA: also use PPL lib
#endif

// options for handling the mean for 1-bit quantization
#undef  ASSUME_ZERO_MEAN_1BIT   // in 1-bit quantization, assume column mean is 0 (saves time to calculate the mean)
#define ZERO_THRESHOLD_FOR_1BIT // force 1-bit quant to threshold against 0 rather than the midpoint between lower and upper, but use asymmetrical reconstruction values

namespace msra { namespace math {

// ---------------------------------------------------------------------------
// quantization of single values
// ---------------------------------------------------------------------------

class valuequantizer
{
protected:

    /*const*/ size_t ldNbits;   // must be power of two
    /*const*/ size_t Nbits;     // now we quantized to 4 bits i.e. [0, 16)
    /*const*/ unsigned int rangeend;
    /*const*/ float quantimin, quantimax;   // quantization range
#ifndef ZERO_THRESHOLD_FOR_1BIT
    /*const*/ float quantimid;              // quantization threshold for 1-bit case
#endif
    /*const*/ float qfactor;    // precomputed factor for quantizating
    /*const*/ float ufactor;    // and for unquantizing
public:
    // constructor
    cudacode
    valuequantizer (size_t ldNbits, float lower, float upper) :
        ldNbits(ldNbits), Nbits(1 << ldNbits), rangeend(1 << Nbits), quantimin(lower), quantimax(upper)
    {
        if (Nbits >= 8*sizeof(rangeend))        // post-fix for incorrect shift for no-quant hack (Nbits=32): << arg is taken mod 32!
            rangeend = 0;                       // in this case, it's only used as (rangeend-1) which is now correct (before it was 0!)
        if (quantimax - quantimin < 1e-36f || rangeend == 0)// must protect against NaN: interval is 0 -> quantization is futile, just emit 0
            qfactor = ufactor = 0.0f;
        else
        {
            qfactor = rangeend / (quantimax - quantimin);   // precompute this for quantize() (see comment there)
            ufactor = (quantimax - quantimin) / rangeend;   // and for unquantize()
        }
#ifndef ZERO_THRESHOLD_FOR_1BIT
        // set the quantization threshold for the special case of 1-bit
        quantimid = 0.5f * (quantimax + quantimin);
#endif
    }

    // quantize for 32-bits case (special case that allows to bypass quantization, for testing/debugging purposes)
    cudasharedcode
    unsigned int quantize32 (const float u) const
    {
        assert (Nbits == 32 && sizeof (unsigned int) == 4);
        return *(unsigned int*)&u;  // we return the bit pattern that encodes the float value
    }

    // quantize one value --special version for 1 bit
    cudasharedcode
    bool quantize1 (const float u) const
    {
        assert (Nbits == 1);
#ifndef ZERO_THRESHOLD_FOR_1BIT
        return u >= quantimid;
#else
        return u >= 0.0f;
#endif
    }

    // quantize one value
    // TODO: we can optimize for 1 bit here very simply... use a template arg 'isonebit'
    cudacode
    unsigned int quantize (const float u) const
    {
        // 32-bits case for hacking
        if (Nbits == 32)
            return quantize32 (u);
        else if (ldNbits == 0)      // TODO: we may need to optimize this by a template arg
            return quantize1 (u) ? 1 : 0;
        else
        {
            //int result = (int) ((u - quantimin) / (quantimax - quantimin) * rangeend);
            int result = (int) ((u - quantimin) * qfactor);
            if (result < 0) // (note: '(int)' rounds asymmetrically towards 0, but that's OK since we clip against 0
                return 0;
            else if ((unsigned int) result >= rangeend)
                return rangeend - 1;
            else
                return (unsigned int) result;
        }
    }

    // unquantize one value
    cudasharedcode
    float unquantize (const unsigned int u) const
    {
        // 32-bits case for hacking
        if (Nbits == 32)
            return *(float *)&u;
        // Note: in 1-bit case, we want 0.5 -> mean0, 1.5 -> mean1
        //float val = (u + 0.5f) / rangeend * (quantimax - quantimin) + quantimin;
        float val = (u + 0.5f) * ufactor + quantimin;
#if 0
#ifndef __device__
        if (_isnan (val))
            sin(1.0);
#endif
#endif
        return val;
    }

    // unquantize one value  --special case for 1 bit
    static
    cudasharedcode
    float unquantize1 (const bool u, float val0, float val1)
    {
        return u ? val1 : val0;
    }

    // helper: compute the binary log of a power of two (utility function to convert 'Nbits' into 'ldNbits'
    static size_t ld (size_t v)
    {
        if (v == 1)
            return 0;
        else if (v & 1) // not a power of two
            throw std::runtime_error ("ld: 'bits' must be a power of two");
        else
            return 1 + ld (v >> 1);
    }
};

// ---------------------------------------------------------------------------
// quantization of whole columns and matrices
// ---------------------------------------------------------------------------

class columnquantizer : public valuequantizer
{

public:
    // quantized values are stored in groups of 'qbwords' = unsigned ints (which happen to memory-align with 'float' as used in 'quantizedcolumn' structure)
    typedef unsigned int qbword;                            // one word of storage containing multiple bits
    static const size_t qbwordbits = 8 * sizeof (qbword);   // number of bits in a qbword (32)

    // sane shift operations that do not take the argument mod 32
    qbword lsl (qbword v, size_t k)         // shift left
    {
        if (k >= sizeof (v)*8) return 0;    // overflow
        return v << k;
    }
    qbword lsr (qbword v, size_t k)         // shift right
    {
        if (k >= sizeof (v)*8) return 0;    // overflow
        return v >> k;
    }

    static size_t qbwordspercol (size_t rows, size_t Nbits) // compute #qbwords per column of a given height
    {
        const size_t valsperqbword = qbwordbits / Nbits;    // how many quantized values fit into one qbword (32 in 1-bit case)
        return (rows + valsperqbword - 1) / valsperqbword;  // how many qbwords do we need to store the column
    }
    size_t qbwordspercol (size_t rows) const { return qbwordspercol (rows, Nbits); }

    // constructor
    cudacode
    columnquantizer (size_t logNbits, float lower, float upper) : valuequantizer (logNbits, lower, upper) { }

    // -----------------------------------------------------------------------
    // columns
    // -----------------------------------------------------------------------

    // determine quantization range of one column
    // This code is written so that it can run in parallel threads on CUDA for collated memory access;
    // set 'subsets' to >1 and pass cross-thread reducer functions for 'float' and 'size_t' (which would reduce through using CUDA __shared__ memory).
    // TODO: further opportunity for speed-up: use 'mean' from last round for 1-bit and stddev calc
    template<class MATRIX, class F1, class F2>
    cudacode
    static void computerange (const MATRIX & us, size_t j, size_t bits, float & lower, float & upper,
                              size_t subset, size_t subsets, F1 allreducef, F2 allreducen)
    {
        const float stddevs = 5.0f;                         // quantization range, cut off after how many standard deviations (make this a parameter if we care)
        const size_t rows = us.rows();
        // compute mean
#if defined (ASSUME_ZERO_MEAN_1BIT) || defined (ZERO_THRESHOLD_FOR_1BIT)   // computing the mean is expensive; we assume there is no reason for asymmetry and thus a zero mean
        // an initial experiment showed that this is significantly worse (36.0 vs. 37.7% frame acc) at the start, but seems to recover nearly (minor gap)
        // thought:
        //  - we could set the threshold at 0
        //  - but keep the quantization values for 0 and 1 separate
        // i.e.
        //  - do not symmetrize/pool the quantization values for 0 and 1
        //  - but hard-code the quantization threshold to be 0 instead of the mean of the two bounds
        // This should give us the best of all--fast operation yet ability to be asymmetric within a column
        const float mean = 0.0f;
#else
        float meanacc = 0.0f;
        for (size_t i = subset; i < rows; i += subsets)     // (subset: compute subset sum)
            meanacc += us(i,j);
        allreducef (meanacc);                               // multi-subset (CUDA): reduce to one thread
        const float mean = meanacc / rows;
#endif

        if (bits == 1)
        {
            // 1-bit case:
            // We want to minimize the (squared) reconstruction error within the two levels.
            // I.e. we should reconstruct to the respective means of each level.
            // To be able to express the range by two floats, we approximate the level threshold as the av. of the two level means.
            // compute the two level means
            float meanacc0 = 0.0f, meanacc1 = 0.0f;
            size_t num0 = 0, num1 = 0;
            for (size_t i = subset; i < rows; i += subsets) // (subset: compute subset sum)
            {
                if (us(i,j) < mean)
                {
                    meanacc0 += us(i,j);
                    num0++;
                }
                else
                {
                    meanacc1 += us(i,j);
                    num1++;
                }
            }
            allreducef (meanacc0);                          // multi-subset (CUDA): reduce to one thread
            allreducef (meanacc1);
            allreducen (num0);
            allreducen (num1);
#ifndef ZERO_THRESHOLD_FOR_1BIT     // we minimize the error jointly across positive and negative numbers to make things symmetrical around the mean (which may be non-zero)
            // tying the two sides
            const float devacc0 = num0 * mean - meanacc0;
            const float devacc1 = meanacc1 - num1 * mean;
            const float dev = (devacc0 + devacc1) / rows;   // both deviations tied, to ensure consistent mean
            const float radius = 2.0f * dev;
            const float newmean = mean;
#else       // we keep two separate reconstruction values to allow for asymmetries--but we instead hard-code that the threshold is 0
            if (num0 == 0) num0 = 1;                        // happens for all-zero columns which do exist (mean0 is 0 in that case)
            if (num1 == 0) num1 = 1;
            const float mean0 = meanacc0 / num0;
            const float mean1 = meanacc1 / num1;
            const float newmean = 0.5f * (mean0 + mean1);   // approximate by using their average as the threshold between 0 and 1
            const float radius = 2.0f * (mean1 - newmean);  // with these values, bits (0,1) which mean values (0.5,1.5) will reconstruct to mean0/1
#endif
            if (subset == 0)
            {
                lower = newmean - radius;
                upper = newmean + radius;
            }
        }
        else
        {
            // >1 bit:
            // We linearly quantize between 'stddevs' standard deviations.
            float varacc = 0.0f;
            for (size_t i = subset; i < rows; i += subsets) // (subset: compute subset sum)
                varacc += (us(i,j) - mean) * (us(i,j) - mean);
            allreducef (varacc);                            // multi-subset (CUDA): reduce to one thread
            const float stddev = sqrt (varacc / rows);
            if (subset == 0)
            {
                lower = mean - stddevs * stddev;            // stddevs = how many stddevs from the mean until outside of quantization range
                upper = mean + stddevs * stddev;
            }
        }
    }
    template<class MATRIX>  // workaround for not being able to declare a default argument for lambda parameters
    cudacode
    static void computerange (const MATRIX & us, size_t j, size_t bits, float & lower, float & upper)
    {
        computerange (us, j, bits, lower, upper, 0, 1, [](float){}, [](size_t){}/*dummy reducers do nothing in linear CPU version*/);
    }

    // compute one qbword value of a quantized matrix column
    template<class MATRIX>
    cudacode
    __forceinline qbword quantizeqbword (const MATRIX & us, const MATRIX & curresidual, const size_t ibegin, const size_t iend, size_t numqbwordspercol, const size_t j, MATRIX & newresidual) const
    {
        qbword bitbuf = 0;
#define USE1BITOPT  // (to allow to test the unoptimized routine--they are actually confirmed to give the same result)
#ifdef USE1BITOPT
        if (Nbits == 1 && &curresidual(0,0) == &newresidual(0,0)/*in-place*/)
        {
            const float val0 = valuequantizer::unquantize (0);
            const float val1 = valuequantizer::unquantize (1);
            const float * usibj     = &us(ibegin,j);
            const float * usibjend  = usibj + (iend - ibegin);
            float *       resibj    = &newresidual(ibegin,j);
            for (qbword bitmask = 1;
                 usibj < usibjend;      // we know that the range covers at most 'qbwordbits' bits
                 bitmask <<= 1, usibj += numqbwordspercol, resibj += numqbwordspercol)
            {
                // quantize   --we access element (i,j) through the three increasing pointers
                const float val = *usibj/*us(i,j)*/ + *resibj/*curresidual(i,j)*/;
                bool qval = valuequantizer::quantize1 (val);
                // save it
                if (qval)
                    bitbuf |= bitmask;
                // compute residual
                float uval = valuequantizer::unquantize1 (qval, val0, val1);
#undef NORESIDUAL  // set this to test without residual--does it still work?
#ifdef NORESIDUAL
                *resibj/*newresidual(i,j)*/ = 0.0f;
#else
                *resibj/*newresidual(i,j)*/ = val - uval;
#endif
            }
        }
        else
#endif
        {
            const unsigned int qbwordbits = 8 * sizeof (qbword);     // number of bits in a qbword
            size_t i = ibegin;
            for (size_t k = 0; k < qbwordbits && i < iend; k += Nbits, i += numqbwordspercol)
            {
                // quantize
                const float val = us(i,j) + curresidual(i,j);
                unsigned int qval = valuequantizer::quantize (val);
                // compute residual
                float uval = valuequantizer::unquantize (qval);
#ifdef NORESIDUAL
                float r = 0.0f;
#else
                float r = val - uval;
#endif
#if 0
                const float clip = quantimax - quantimin;
                if (r > clip)
                    r = clip;
                else if (r < -clip)
                    r = -clip;
#endif
                newresidual(i,j) = r;
                // save it
                bitbuf |= (qval << k);
            }
        }
        return bitbuf;
    }

    // quantize a matrix column into qcoldata
    // The current value of 'curresidual' is added to the matrix, and 'newresidual' gets updated with the new residual; &curresidual = &newresidual is allowed (intended)
    template<class MATRIX>
    cudacode
    void quantize (const MATRIX & us, const MATRIX & curresidual, const size_t j, qbword * qcolbits, MATRIX & newresidual) const
    {
        // we loop over qbword values
        // E.g. there are 35 ints for a 1100-dim column (at 1-bit quantization).
        // For better CUDA memory collating, we interleave memory such that computing consecutive ints triggers consecutive memory accesses
        // (although for the CPU side, it breaks caching; we could do in-place op)
        // E.g., int  0 accesses elements 0, 35, 70, etc.
        // while int  1 accesses elements 1, 36, 71, etc
        // up to int 34 accesses elements 34, 69, 104, etc.
        const size_t numqbwordspercol = qbwordspercol (us.rows());
        for (size_t iqbword = 0; iqbword < numqbwordspercol; iqbword++)
            qcolbits[iqbword] = quantizeqbword (us, curresidual, iqbword, us.rows(), numqbwordspercol, j, newresidual);
    }

    // compute one qbword value of a quantized matrix column --special version for 1 bit, no striping (all bits represent consecutive memory locations)
    template<class MATRIX>
    __forceinline qbword quantizeqbword1 (const MATRIX & us, MATRIX & residual, const size_t ibegin, const size_t numbits/*32 or less*/, const size_t j) const
    {
        assert (Nbits == 1);
        qbword bitbuf = 0;
//#define _SSEFLOAT4_H  // yak: this conflicts with CUDA type 'float4'
#ifdef _SSEFLOAT4_H     // yak: caller must #include <ssefloat4.h> to activate this
        const float4 val0 = valuequantizer::unquantize (0);  // the two values for 0 and 1 (for unquantization)
        const float4 val1 = valuequantizer::unquantize (1);
        const float4 val1m0 = val1 - val0;
#ifndef ZERO_THRESHOLD_FOR_1BIT
        const float4 mid = quantimid;
#else
        const float4 mid = 0.0f;
#endif
        const float4 * usibj  = (const float4 *) &us(ibegin,j); // we know they are aligned
        float4 *       resibj = (float4 *)       &residual(ibegin,j);
        for (size_t k = 0; k < numbits; k += 4, usibj++, resibj++)
        {
            // quantize
            const float4 val = *usibj + *resibj;
            const float4/*compareresult*/ qval = (val >= mid);  // (creates a sort of 'int4', -1 for 'true', 0 for 'false')
            // save it
            const qbword qvalasbits = qval.comparesultasbits(); // 4 bits; 1 if val >= quantimid
            bitbuf |= (qvalasbits << k);
            // compute residual
            const float4 uval = val0 + val1m0.maskby (qval);
#ifdef NORESIDUAL
            *resibj = 0.0f;
#else
            *resibj = val - uval;
#endif
        }
        // SSE may give us a few extra bits--clear them out
        const size_t bitstopushout = qbwordbits - numbits/*bits to keep*/;  // this many leading bits must be zero
        bitbuf <<= bitstopushout; bitbuf >>= bitstopushout;                 // push them out at the top
#else   // (we may disable the code below if ssefloat is not included)
        const float val0 = valuequantizer::unquantize (0);  // the two values for 0 and 1 (for unquantization)
        const float val1 = valuequantizer::unquantize (1);
        const float * usibj  = &us(ibegin,j);
        float *       resibj = &residual(ibegin,j);
        qbword bitmask = 1;
        for (size_t k = 0; k < numbits; k++, bitmask <<= 1)
        {
            // quantize
            const float val = usibj[k] + resibj[k];
            bool qval = valuequantizer::quantize1 (val);
            // save it
            if (qval)
                bitbuf |= bitmask;
            // compute residual
            float uval = valuequantizer::unquantize1 (qval, val0, val1);
#ifdef NORESIDUAL
            resibj[k] = 0.0f;
#else
            resibj[k] = val - uval;
#endif
        }
#endif
        return bitbuf;
    }

    // determine how the GPU shuffling lays out the data in memory
    // It determines
    //  - how many bits are used in each qbword (it's not always 32)
    //  - how many qbwords use that number of bits; the remaining use one bit less
    // Once more:
    // The first 'maxusingwords' qbwords have 'maxusedbits' valid bits; and all remaining have one valid bit less.
    void get1bitlayout (size_t rows, size_t & numqbwords, size_t & maxusedbits, size_t & maxusingwords) const
    {
        numqbwords = qbwordspercol (rows);                 // we have this many qbwords to process for this column
        // compute how many bits are used in each qbword (some bits are unused unless rows divides by 32)
        // examples:
        //    32 (32 'groups' of 1) -> 32             // 'groups' are #bits consumed; of X is the #qbwords
        //    33 (17 groups of 2)   -> 1*17+1*16
        //    65 (22 groups of 3)   -> 2*22+1*21
        //   170 (29 groups of 6)   -> 2*29+4*28
        //  3201 (32 groups of 101) -> 70*32+31*31
        //  5000 (32 groups of 157) -> 133*32+24*31
        const size_t groupsize = (rows + (qbwordbits-1)) / qbwordbits;  // group size
        assert (groupsize == numqbwords);                               // one 32-bit word per group
        maxusedbits = (rows + (groupsize-1)) / groupsize;  // max #bits used
        const size_t totalunusedbits = maxusedbits * groupsize - rows;  // so many words use 1 bit less
        maxusingwords = numqbwords - totalunusedbits;      // so many words use the maximum number of bits; the remaining use 1 bit less
        // verify it
        const size_t thisgives = maxusingwords * maxusedbits + totalunusedbits * (maxusedbits-1);
        assert (thisgives == rows);
    }

    // quantize a matrix column into qcoldata
    // Same as quantize() but assumes 1 bit and allowing reshuffled layout that is only valid for straight aggregation.
    // Optimized like hell for CPU side (caching and other).
    template<class MATRIX>
    __forceinline void quantize1bitaltlayout (const MATRIX & us, MATRIX & residual/*in/out*/, const size_t j, qbword * qcolbits, const bool verify) const
    {
        const size_t rows = us.rows();
        size_t numqbwords, maxusedbits, maxusingwords;
        get1bitlayout (rows, numqbwords, maxusedbits, maxusingwords);
        size_t iqbword = 0;     // next qbword to write
        size_t totalbits = 0;   // (for sanity check only; total #bits written)
        size_t ibegin = 0;
        // special case: 32 bits (this code was derived and simplified from the generic code below)
        if (maxusedbits == qbwordbits)
        {
            assert (maxusingwords <= numqbwords);
            for ( ; iqbword < maxusingwords; ibegin += qbwordbits)
            {
                size_t thisnumbits = rows-ibegin;       // number of bits we produce from this; 32 except for last step
                if (thisnumbits < qbwordbits)
                    break;                              // a partial word: leave to generic case
                qbword bits = quantizeqbword1 (us, residual, ibegin, qbwordbits/*constant 32, can unroll*/, j);
                assert (iqbword < numqbwords);
                qcolbits[iqbword++] = bits;
            }
            totalbits += ibegin;                        // (sanity check only)
        }
        // generic case
        qbword bits = 0;        // bit buffer
        size_t numbits = 0;     // current number of bits in buffer
        size_t iend;
        for ( ; ibegin < rows; ibegin = iend)
        {
            size_t thisnumbits = rows-ibegin;           // number of bits we produce from this; 32 except for last step
            if (thisnumbits > qbwordbits)
                thisnumbits = qbwordbits;
            iend = ibegin + thisnumbits;
            // produce the 32 bits
            qbword thisbits;
            if (thisnumbits == qbwordbits)
                thisbits = quantizeqbword1 (us, residual, ibegin, qbwordbits/*constant 32, can unroll*/, j);
            else
                thisbits = quantizeqbword1 (us, residual, ibegin, thisnumbits, j);
            // append these bits to our buffer
            assert (numbits < qbwordbits);
            const size_t space = qbwordbits - numbits;  // we are only able to store this many in 'bits'
            bits |= thisbits << numbits;                // concatenate with our buffer; if too many bits, then those get pushed out at the top; those are (thisbits >> space)
            qbword highbits = (space < qbwordbits) ? (thisbits >> space) : 0;   // overflowing bits (will be 0 if no overflow) (note that >>32 seems to do nothing!!)
            numbits += thisnumbits;                     // we've got this many bits in our buffer now
            totalbits += thisnumbits;                   // (sanity check only)
            assert (numbits < 2 * qbwordbits);
            // we now have a bit stream in (bits,highbits)
            // flush out as much as we can
            for (;;)
            {
                // how many bits are meant to be stuffed into this qbword?
                const size_t outnumbits = iqbword < maxusingwords ? maxusedbits : (maxusedbits-1); // first 'maxusingwords' use the max; remaining use 1 bit less
                if (numbits < outnumbits)
                    break;                              // not enough bits left in our buffer
                // get 'outnumbits' from our buffer
                qbword outbits = bits;
                const size_t bitstopushout = qbwordbits - outnumbits/*bits to keep*/;   // clear any bits above
                outbits <<= bitstopushout; outbits >>= bitstopushout;                   // push them out at the top
                // save them
                assert (iqbword < numqbwords);
#if 0
                // special function: verify once again
                if (verify)
                {
                    if (qcolbits[iqbword] != outbits)
                        throw std::logic_error ("quantize1bitaltlayout not reconstructing the input");
                    assert (qcolbits[iqbword] == outbits);
                }
#endif
                qcolbits[iqbword++] = outbits;
                // and remove them from out buffer
                if (outnumbits == qbwordbits)                       // (we need this branch because x>>32==x--yak!)
                {
                    bits = highbits;
                    highbits = 0;
                }
                else
                {
                    bits >>= outnumbits;                            // we now have a vacuum of 'outnumbits' bits that will suck in bits from 'highbits'
                    bits |= highbits << (qbwordbits - outnumbits);  // shift the lowest 'outnumbits' to max. position and or it
                    highbits >>= outnumbits;                        // and consume those bits
                }
                numbits -= outnumbits;
                assert (highbits == 0);                             // we can never have over 32 bits left-over since we'd be able to flush something if we did
            }
        }
        assert (numbits == 0 && bits == 0);                         // buffer must be empty
        assert (iqbword == numqbwords);                             // must have filled expected number of qbwords
        assert (totalbits == rows); totalbits;                      // processed correct number of bits
    }

    // unquantize one qbword of a quantized matrix column
    template<class MATRIX>
    cudacode
    __forceinline void unquantizeqbword (MATRIX & us, const size_t ibegin, const size_t iend, size_t numqbwordspercol, size_t j, qbword bitbuf, const bool add) const
    {
#ifdef USE1BITOPT
        if (Nbits == 1)   // special case for 1 bit
        {
            const float val0 = valuequantizer::unquantize (0);
            const float val1 = valuequantizer::unquantize (1);
            float * usibj           = &us(ibegin,j);
            const float * usibjend  = usibj + (iend - ibegin);
            for ( ; usibj < usibjend; usibj += numqbwordspercol)
            {
                // get value
                const bool qval = (bitbuf & 1) != 0;    // bitbuf is shifted in-place
                bitbuf >>= 1;                           // and get bitbuf into next position
                // unquantize
                float val = valuequantizer::unquantize1 (qval, val0, val1);
                if (add)            // add previous value
                    val += *usibj/*us(i,j)*/;
                *usibj/*us(i,j)*/ = val;
            }
        }
        else
#endif
        {
            const size_t bitmask = rangeend -1;                     // (rangeend MUST be a power of two; ensured by constructing off ldNbits)
            size_t i = ibegin;
            for (size_t k = 0; k < qbwordbits && i < iend; k += Nbits, i += numqbwordspercol)
            {
                // get value
                const unsigned int qval = (bitbuf >> k) & bitmask;  // % 2^Nbits
                // unquantize
                float val = valuequantizer::unquantize (qval);
                if (add)            // add previous value
                    val += us(i,j);
                us(i,j) = val;
            }
        }
    }

    // unquantize a matrix column from qcoldata
    // If 'add' then add to the existing content of the matrix (this is a common thing to do; saves a buffer).
    template<class MATRIX>
    cudacode
    void unquantize (MATRIX & us, size_t j, const qbword * qcolbits, const bool add) const
    {
        // loop over qbword values
        const size_t numqbwordspercol = qbwordspercol (us.rows());
        for (size_t iqbword = 0; iqbword < numqbwordspercol; iqbword++)
            unquantizeqbword (us, iqbword, us.rows(), numqbwordspercol, j, qcolbits[iqbword], add);
    }

    // unquantize one qbword of a quantized matrix column
    template<class MATRIX>
    __forceinline void unquantizeqbword1 (MATRIX & us, const size_t ibegin, const size_t numbits, size_t j, qbword bitbuf, const bool add) const
    {
        assert (Nbits == 1);
        // ensure well-formed bitbuf (no garbage bits in the upper unused region if any)
        const size_t bitstopushout = qbwordbits - numbits/*bits to keep*/;   // clear any bits above
        assert ((bitbuf << bitstopushout) >> bitstopushout == bitbuf); bitstopushout;
#ifdef _SSEFLOAT4_H     // yak: caller must #include <ssefloat4.h> to activate this
        const float4 val0 = valuequantizer::unquantize (0);  // the two values for 0 and 1 (for unquantization)
        const float4 val1 = valuequantizer::unquantize (1);
        const float4 val1m0 = val1 - val0;
        float4 *       usibj     = (float4 *) &us(ibegin,j);
        const float4 * usibjend  = (float4 *) (&us(ibegin,j) + numbits);
        if (add)
        {
            for ( ; usibj < usibjend; usibj++/*this is a step of 4 floats*/)
            {
                // get bits and unquantize
                qbword qvalasbits = bitbuf & 0xf;   // bitbuf shifted in-place
                bitbuf >>= 4;                       // advance to next position
                // map to val0/val1
                float4 uval = val0 + val1m0.maskby (qvalasbits);    // bitbuf contains 1 for set bits and 0 for unset ones
                // save it
                uval += *usibj;                     // add previous value
                *usibj = uval;
            }
        }
        else    // code dup from above to avoid a test in the inner loop (may matter indeed)
        {
            for ( ; usibj < usibjend; usibj++/*this is a step of 4 floats*/)
            {
                // get bits and unquantize
                qbword qvalasbits = bitbuf & 0xf;   // bitbuf shifted in-place
                bitbuf >>= 4;                       // advance to next position
                // map to val0/val1
                float4 uval = val0 + val1m0.maskby (qvalasbits);    // bitbuf contains 1 for set bits and 0 for unset ones
                // save it
                //uval += *usibj;                     // add previous value
                *usibj = uval;
            }
        }
#else
        const float val0 = valuequantizer::unquantize (0);  // the two values for 0 and 1 (for unquantization)
        const float val1 = valuequantizer::unquantize (1);
        float * usibj  = &us(ibegin,j);
        qbword bitmask = 1;
        for (size_t k = 0; k < numbits; k++, bitmask <<= 1)
        {
            // get bit and unquantize
            float val = (bitbuf & bitmask) ? val1 : val0;
            if (add)            // add previous value
                val += usibj[k];
            usibj[k] = val;
        }
#endif
    }

    // unquantize a matrix column from qcoldata
    // Same as unquantize() but assumes 1 bit and allowing reshuffled layout that is only valid for straight aggregation, as created by quantize1bitaltlayout().
    // This is nasty--the striped (GPU-efficient) bit layout is not contiguous--qbwords do not always use all bits.
    template<class MATRIX>
    __forceinline void unquantize1bitaltlayout (MATRIX & us, size_t j, const qbword * qcolbits, const bool add) const
    {
        const size_t rows = us.rows();
        size_t numqbwords, maxusedbits, maxusingwords;
        get1bitlayout (rows, numqbwords, maxusedbits, maxusingwords);
        // algorithm:
        // loop over qbwords and construct a stream of 32-bit groups
        size_t totalbits = 0;   // (for sanity check only; total #bits consumed)
        size_t ibegin = 0;      // next matrix element to unquantize (we unquantize 32 starting from this one)
        size_t iqbword = 0;
        // special case: 32 bits (this code was derived and simplified from the generic code below)
        if (maxusedbits == qbwordbits)
        {
            assert (maxusingwords <= numqbwords);
            for ( ; iqbword < maxusingwords; iqbword++)
            {
                assert (ibegin < rows);
                assert (iqbword < maxusingwords && maxusedbits == qbwordbits);
                qbword bits = qcolbits[iqbword];    // consume one word
                assert (us.rows() - ibegin >= qbwordbits);
                unquantizeqbword1 (us, ibegin, qbwordbits/*constant 32, can unroll*/, j, bits, add);
                // and shift them out of our buffer
                ibegin += qbwordbits;
            }
            totalbits += qbwordbits * iqbword;                   // (sanity check only)
        }
        // generic case
        qbword bits = 0;        // ongoing bit buffer
        size_t numbits = 0;     // #bits we got in our buffer; whenever we hit 'qbwordbits', we will flush them, i.e. unquantize them
        for ( ; iqbword < numqbwords; iqbword++)
        {
            assert (ibegin < rows);
            qbword thisbits = qcolbits[iqbword];    // consume one word
            // how many bits are in this qbword?
            const size_t thisnumbits = iqbword < maxusingwords ? maxusedbits : (maxusedbits-1); // first 'maxusingwords' use the max; remaining use 1 bit less
            const size_t bitstopushout = qbwordbits - thisnumbits/*bits to keep*/;  // ensure there's no garbage in the upper bits
            assert ((thisbits << bitstopushout) >> bitstopushout == thisbits); bitstopushout;
            // append these bits to our buffer
            assert (numbits < qbwordbits);
            const size_t space = qbwordbits - numbits;  // we are only able to store this many in 'bits'
            bits |= thisbits << numbits;                // concatenate with our buffer; if too many bits, then those get pushed out at the top; those are (thisbits >> space)
            numbits += thisnumbits;                     // we've got this many bits in our buffer now
            totalbits += thisnumbits;                   // (sanity check only)
            assert (numbits < 2 * qbwordbits);
            // consume the bits if we got enough (32), meaning, unquantize them
            if (numbits >= qbwordbits)
            {
                assert (us.rows() - ibegin >= qbwordbits);
                unquantizeqbword1 (us, ibegin, qbwordbits/*constant 32, can unroll*/, j, bits, add);
                // and shift them out of our buffer
                bits = (space < qbwordbits) ? (thisbits >> space) : 0; // after removing 'qbwordbits' bits from the buffer, what's left are the overflowing bits (will be 0 if no overflow) (note that >>32 seems to do nothing!!)
                numbits -= qbwordbits;
                ibegin += qbwordbits;
            }
        }
        // last one
        if (numbits > 0)                                // still some left-over
            unquantizeqbword1 (us, ibegin, numbits, j, bits, add);
        else
            assert (bits == 0);                         // buffer must be empty
        // sanity-check that we processed the correct number of elements
        assert (ibegin + numbits/*last one*/ == rows);  // we must have written all values
        assert (totalbits == rows); totalbits;          // processed correct number of bits
    }
};

// ---------------------------------------------------------------------------
// entire matrices
// This is only used on the CPU side. The GPU calls column functions directly from kernels.
// ---------------------------------------------------------------------------

class matrixquantizer : public columnquantizer  // (we inherit for some static methods only)
{
    matrixquantizer();

public:

    // A q-package represents a patch of a matrix as quantized values, as a byte stream of this format:
    //  - array of                              // one for each column
    //     - lower bound: float
    //     - upper bound: float
    //     - array of 'qbwords'                 // same for each column, rounded to multiple of 32 bits
    // The q-package does not store #bits or stripe dimensions, those are taken from context and must match on all nodes.

    // one quantized column with header
    // This is a variable-length structure.
    // A matrix is an array of these.
    struct quantizedcolumn
    {
        float lower;                            // quantization range for this column
        float upper;
        qbword bits[1/*variable*/];             // variable-size array to hold the bits, grouped into 'qbwords'

        // required storage size of quantized package in bytes for a given column (incl. header, aligned to 4 bytes for 'float')
        cudasharedcode
        static size_t columnsize (size_t bits, size_t rows)    // one column
        {
            const size_t columndatasize = (rows * bits + (qbwordbits-1)) / qbwordbits * sizeof(qbword);       // bit array for one column, rounded to multiple of 4 bytes
            // TODO: can this ^^ be computed using qbwordspercol()?
            return 2 * sizeof (float) + columndatasize;
        }
    };

    // storage requirement for quantization package of an entire matrix (patch)
    static size_t buffersize (size_t bits, size_t rows, size_t cols)
    {
        // 'bits' must fit into a 32-bit integer (we'll use 1 or 2 anyway...)
        if (qbwordbits / bits * bits != qbwordbits)
            throw std::logic_error ("buffersize: 'bits' must be a divisor of 32");
        return quantizedcolumn::columnsize (bits, rows) * cols;
    }

    // quantize an entire matrix, calling quantize() for each column
    // This is only used on the CPU side; the GPU calls into here column by column (on different GPU threads).
    // 'allowaltlayout' allows an alternate, CPU-cache/SSE-optimized layout that does not match the original matrix but still allows aggregation
    // (pass this flag to the matching unquantize()/quantize() pair that surrouns the aggregation call).
    // BUGBUG: with altlayout, the result is not precisely the same; that means that we are missing some bits somewhere??
    // If 'reuserangescaled' then the target qpackage already contains valid quantization ranges--do not recompute (to save time), but instead scale by this factor.
    template<class MATRIX>
    static void quantize (const MATRIX & us, MATRIX & curresidual, char * qpackage, const char * qpackageend, size_t bits, MATRIX & newresidual, bool allowaltlayout, size_t reuserangescaled)
    {
#undef DISABLEALTLAYOUT    // #define to disable the 'altlayout' option until we have fixed the missing-bits bug (see quantize1bitaltlayout())
#ifdef DISABLEALTLAYOUT
allowaltlayout = false;     // BUGBUG: 'true' will lead to a different result
#endif
        if ((ptrdiff_t) buffersize (bits, us.rows(), us.cols()) != qpackageend - qpackage)
            throw std::logic_error ("quantizestripe: dimension of patch to be quantized does not match allocated buffer size for quantized data");
        const size_t ldNbits = ld (bits);
        const size_t colsize = quantizedcolumn::columnsize (bits, us.rows());
#ifdef QUANTUSEPPL
        Concurrency::parallel_for ((size_t) 0, us.cols(), [&] (size_t j)
#else
        for (size_t j = 0; j < us.cols(); j++)
#endif
        {
            auto & qcol = *(quantizedcolumn *) &qpackage[colsize * j];
            if (reuserangescaled == 0)      // (optimization: reuse sub-batch ranges that were computed by the GPU)
            {
                columnquantizer q (0, 0.0f, 1.0f);   // (dummy to workaround broken lambda compilation for QUANTUSEPPL, otherwise not needed)
                q.computerange (us, j, bits, qcol.lower, qcol.upper);
            }
            else                            // we reuse the range, but since we added K nodes, we need to scale
            {
                qcol.lower *= reuserangescaled;
                qcol.upper *= reuserangescaled;
            }
            if (bits == 1 && allowaltlayout)
            {
                assert (&curresidual(0,0) == &newresidual(0,0)/*in-place*/);    // we assume this
                columnquantizer q (0/*ldNbits*/, qcol.lower, qcol.upper);
                q.quantize1bitaltlayout (us, newresidual, j, qcol.bits, false);
            }
            else
            {
                columnquantizer q (ldNbits, qcol.lower, qcol.upper);
                q.quantize (us, curresidual, j, qcol.bits, newresidual);
            }
        }
#ifdef QUANTUSEPPL
        );
#endif
    }

    // unquantize an entire matrix, calling unquantize() for each column
    template<class MATRIX>
    static void unquantize (const char * qpackage, const char * qpackageend, size_t bits, MATRIX & us, bool add, bool allowaltlayout)
    {
#ifdef DISABLEALTLAYOUT
allowaltlayout = false;     // BUGBUG: 'true' will lead to a different result
#endif
        if ((ptrdiff_t) buffersize (bits, us.rows(), us.cols()) != qpackageend - qpackage)
            throw std::logic_error ("unquantize: dimension of patch to be unquantized does not match size of quantized data");
        const size_t ldNbits = ld (bits);
        const size_t colsize = quantizedcolumn::columnsize (bits, us.rows());
#ifdef QUANTUSEPPL
        Concurrency::parallel_for ((size_t) 0, us.cols(), [&] (size_t j)
#else
        for (size_t j = 0; j < us.cols(); j++)
#endif
        {
            const auto & qcol = *(const quantizedcolumn *) &qpackage[colsize * j];
            if (bits == 1 && allowaltlayout)
            {
                columnquantizer q (0/*ldNbits*/, qcol.lower, qcol.upper);   // use a '0' here so compiler can optimize out a lot (hopefully it will)
                q.unquantize1bitaltlayout (us, j, qcol.bits, add);
            }
            else
            {
                columnquantizer q (ldNbits, qcol.lower, qcol.upper);
                q.unquantize (us, j, qcol.bits, add);
            }
        }
#ifdef QUANTUSEPPL
        );
#endif
    }
};

};};
