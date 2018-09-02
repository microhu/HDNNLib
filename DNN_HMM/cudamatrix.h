// cudamatrix.h -- matrix with CUDA execution
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrix.h $
// 
// 148   7/19/14 20:37 Fseide
// disabled BEGINTIME1/ENDTIME1 macro to stop HVite from clogging the log
// and the server
// 
// 147   6/06/14 4:47p Fseide
// softplus implemented
// 
// 146   5/20/14 3:39p Fseide
// updated setbackpropagationerrorsignalhsmoothing() (added one parameter
// for frame dropping)
// 
// 145   5/15/14 7:24p Fseide
// variance normalization now implemented in class mvn;
// new method meanvaracc();
// meanvarnorm() now takes parameter whether to apply the mean or not
// 
// 144   5/15/14 13:26 Fseide
// extended addtoallcolumns() to take an optional weight;
// implemented meanvarnorm() through addtoallcolumns (., -1) (to add
// variance norm, I'd rather extend addtoallcolumns() further)
// 
// 143   5/14/14 15:20 Fseide
// (comment)
// 
// 142   5/14/14 15:19 Fseide
// added new method meanvarnorm(), not completed yet
// 
// 141   5/12/14 13:53 Fseide
// removed dropout pre-scaling everywhere incl. CUDA side
// 
// 140   4/30/14 11:17a Fseide
// bug fix: gradientfixups() now takes an optional qstripe to make it run
// on the associated CUDA stream, same as unquantization (without, it may
// run on the wrong stream and thus induce some non-deterministism);
// titied up some logging
// 
// 139   4/29/14 1:42p Fseide
// unquantizeandaggregatestripe() now also takes the learning rate;
// gradientfixups() now multiplies the gradient with an additional
// learning-rate parameter -> this way it can compute the final-final
// thing that gets added to the model;
// syncassignaggregatedstripeandunquantize() now takes bpinfo so it can
// know about 'distributefixedcost' mode
// 
// 138   4/29/14 10:33a Fseide
// forgot a = 0
// 
// 137   4/29/14 10:24a Fseide
// new method gradientfixups(), to be tested
// 
// 136   3/19/14 5:51p Fseide
// new methods scalerowwise() and unscalerowwise()
// 
// 135   2/20/14 4:27p Fseide
// adadenom() now takes a mean accumulator as well (currently only used
// for diagnostics, not in actual AdaGrad normalizations, assuming a fixed
// target is given);
// likewise, network layer now keeps a mean accumulator matching the sqr
// acc for AdaGrad;
// applyadagrad() now again computes avdenom even if it is not used (i.e.
// when fixed target is given), for diagnostics;
// 
// 134   2/17/14 2:08p Fseide
// new method disableasynccudaoperation() to support new --gpuforcesync
// cmd-line option
// 
// 133   2/14/14 4:05p Fseide
// new faster AdaGrad function that combines two: adagradientfromsqracc()
// 
// 132   2/12/14 16:20 Fseide
// disabled BEGIN/ENDTIME again
// 
// 131   2/12/14 13:58 Fseide
// timing function now outputs the thread id
// 
// 130   2/11/14 22:06 Fseide
// back to outputting timing info again to track down that strange glitch
// that kills us
// 
// 129   2/11/14 16:59 Fseide
// disabled detailed timing measurements again
// 
// 128   2/11/14 3:04p Fseide
// added equivalents to synchronize() that specifically wait after fetch()
// and assign(), implemented as stream syncs rather than global device
// syncs, hoping for better efficiency
// 
// 127   2/11/14 10:25 Fseide
// added second set of macros BEGINTIME1/ENDTIME1 which can be disabled
// separately
// 
// 126   2/10/14 13:58 Fseide
// disabled the timer
// 
// 125   2/08/14 7:02p F-gli
// fixed mismatch between size_t and int
// 
// 124   2/07/14 19:00 Fseide
// moved 'timeme' class to cudamatrix.h so we can use it inside
// cudamatrix.cpp
// 
// 123   2/07/14 14:05 Fseide
// completed the GPU code for requantization, but not giving the same
// result yet so I keep it disabled;
// quantizeandfetchqstripe() now supports the 'reuserangescaled' flag just
// like the CPU version
// 
// 122   2/07/14 11:58 Fseide
// syncassignqstripeandunquantize() now takes an 'add' parameter;
// new flag 'prioritystream' to newqstripe(), but not used anywhere yet
// 
// 121   2/03/14 19:42 Fseide
// some (unelegant) renaming of CUDA quant sync functions for more clarity
// 
// 120   2/03/14 17:47 Fseide
// implemented AdaGrad-like pre-scaling for quantization, but does not
// work (disabled)
// 
// 119   2/03/14 16:09 Fseide
// changed AdaGrad implementation to isolate it out;
// minor fix of AdaGrad in interpreting old-style parameters (256 frames,
// now compatible with double-buffering=128 frames)
// 
// 118   1/17/14 12:23 Fseide
// changed qstripe to no longer remember the CPU-side buffer, but instead
// just a GPU-side buffer and associated events
// 
// 117   1/17/14 9:17 Fseide
// new method newsharedtransferbuffer()
// 
// 116   1/09/14 15:06 Fseide
// towards implementing the lambdas that the MPI aggregator needs
// 
// 115   1/09/14 13:53 Fseide
// (renamed in/outresidual to cur/newresidual)
// 
// 114   1/09/14 13:52 Fseide
// quantization now peruses the residual in-place (instead of first adding
// it explicitly to the raw gradient)--better separation of concerns
// (residual belongs to quantization);
// bug fix in quantizeandfetchqstripe(): forgot to apply the patch to the
// residual
// 
// 113   1/08/14 9:44 Fseide
// qstripe no longer knows about patch dimension and 'bits' parameter
// (none of its business)
// 
// 112   1/08/14 8:49 Fseide
// qstripe now does not take a buffer+offset but begin and end iterator;
// qstripe is now handed back from cudamatrix lib as a shared_ptr with
// custom deleter;
// MPI initialization sequence: attachmpiaggregator() determines the
// needed stripe sizes and buffer offsets, while first use (within-layer)
// will lazily allocate (cross-layer) stripe buffers;
// split off mpihelper from mpiaggregator (mpihelper handles basic MPI
// interaction)
// 
// 111   1/03/14 17:26 Fseide
// initialization sequence of MPI stuff stratified, new methods
// attachmpiaggregator() and detachmpiaggregator() in rbm.h, called by
// respective functions in dbn.h;
// moved logic to determine buffer size etc. from CUDA-side qstripe to a
// new CPU-side structure mpistripebufferref (since stripes may live on
// different GPUs, the GPU side cannot do this);
// stripes are now associated with a GPU (in theory--the actual
// determination of the GPU device is not implemented)
// 
// 110   12/19/13 16:54 Fseide
// added skeleton for quantization of matrix stripes (supposed to be used
// for data parallelism, for exchanging gradient data as highly quantized
// buffers)
// 
// 109   9/29/13 13:48 Fseide
// (renamed args of gemm() and gems() to be more clear on what they mean)
// 
// 108   9/28/13 19:36 Fseide
// tidied up the last posteriorstats() fix (removed the dummy args, made
// function arg names consistent, lots of comments)
// 
// 107   9/28/13 10:17 Fseide
// CUDA posteriorstats() now returns the stats values directly by
// cudaMemcpy() inside that function (more efficient since only 3 floats)
// 
// 106   9/27/13 18:45 Fseide
// model-parallel version of posteriorstats() implemented;
// changed argument order of posteriorstats() CUDA function to be more
// logical
// 
// 105   9/27/13 18:05 Fseide
// merged stripedposteriorstats() and posteriorstats() (posteriorstats()
// can now run striped)
// 
// 104   9/26/13 15:43 Fseide
// stripedsoftmaxstep1() now takes the full matrix, not just a stripe;
// softmax() currently not working for K>1;
// softmaxs0t() now returns the frame-wise partial sum not as a return
// value but by writing it into a location whose address is
// passed--otherwise it would not work (compiler bug?)
// 
// 103   9/25/13 19:11 Fseide
// implemented proper model parallelism in softmax(), using the trick we
// already used in pipeline training
// 
// 102   9/24/13 10:57 Fseide
// new method matrix::setsubstream();
// added associated wrapper class onsubstream
// 
// 101   9/17/13 3:43p Fseide
// steps towards implementing send()
// 
// 100   9/17/13 12:36p Fseide
// new methods send() and receive() for asynchronous peer-to-peer transfer
// 
// 99    9/11/13 3:35p V-haofu
// to be consistent with noncuda adagrad changes
// 
// 98    9/10/13 1:47p V-haofu
// fix not complie problem due to previous interface change
// 
// 97    9/10/13 10:31a V-haofu
// revert to the original parameter order in accumulatesqr
// 
// 96    9/10/13 10:12a V-haofu
// 
// 95    8/07/13 4:40p T-paswie
// mulbydmaxout kernel, (temporary?) use curand in dropout function
// 
// 94    7/05/13 9:02p T-paswie
// colwise norms now propagated around the matrices code and used with
// maxouts
// 
// 93    7/02/13 8:10p T-paswie
// colwisenorm kernel +matrix infrastracture for it
// 
// 92    6/09/13 19:21 Fseide
// replaced dllimporty by no declspec, which works for both static and DLL
// linkage
// 
// 91    6/09/13 18:13 Fseide
// enabled warning level 4 and warning = errors;
// fixed the fallout (unused arguments, variables, and functions)
// 
// 90    6/07/13 20:37 Fseide
// added experimental generalization of recitifed linear units using an
// optional non-linearity (n-th root) and leakiness
// 
// 89    6/06/13 11:28 Fseide
// dropout() now takes another parameter 'prescale' to scale remaining
// values up so that the average scale remains 1.0, avoiding the need to
// post-process the model
// 
// 88    6/02/13 8:14 Fseide
// backpropagationstats() implemented for LRU;
// mulbydlru() implemented
// 
// 87    6/02/13 4:03 Fseide
// towards RLUs
// 
// 86    1/09/13 3:29p V-hansu
// add setbackpropagationerrorsignalhsmoothing() and related kernel to
// prepare for cuda based hsmoothing
// 
// 85    12/07/12 5:15a Adame
// convolution/maxpool support
// 
// 84    11/27/12 4:51p Fseide
// moved vectorbase<> to its original place, so that we don't see
// unnecessary diffs when comparing
// 
// 83    11/27/12 4:44p V-hansu
// move vectorbase<> like ushortvector from cudamatrix.h to cudalattice.h
// 
// 82    11/27/12 4:10p V-hansu
// add senone2keepmodelupdate to setbackpropagationerrorsignal()
// 
// 81    11/27/12 3:43p V-hansu
// move vectorbase<> like ushortvector from cudalattice.h to cudamatrix.h
// 
// 80    11/04/12 7:52a Fseide
// new class matrixaccumulator
// 
// 79    11/01/12 10:09a T-simonw
// restore lost check-in comments (no code changes had been lost)
// 
// 78    10/31/12 10:06a T-simonw
// correct spelling error
//
// 77    10/29/12 3:50p T-simonw
// add dot product, nrm2, elementwise operations, weighteddot, and special
// purpose method (sethessianvectorsignal)
// 
// 76    10/16/12 11:25a Fseide
// two new methods dropout() and scale(), for implementing Hinton's
// drop-out method
// 
// 75    10/11/12 3:44p V-hansu
// (add a space after the setvalue function)
// 
// 74    10/10/12 9:58a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new languages.
// 
// 73    9/26/12 9:43p V-hansu
// change setzero() to setvalue()
// 
// 72    9/24/12 3:25p Fseide
// adadenom() no longer takes numsummands, as it has become obsolete
// 
// 71    9/24/12 3:00p Fseide
// AdaGrad adjustment now clipped to 10 x against the average of the
// respective parameter matrix/vector, only afterwards is it scaled to the
// user-specified target. This is to prevent clipping if the dynamics
// change.
// 
// 70    9/21/12 3:24p Fseide
// added nosoftmax mode, to speed up sequence training by bypassing the
// unnecessary expensive softmax() computation
// 
// 69    9/21/12 8:14a Fseide
// fixed missing symbol EXCEPTION_EXECUTE_HANDLER when compiling in
// cudamatrix project
// 
// 68    9/21/12 8:10a Fseide
// new method msra::cuda::numcudadevices() inside cudamatrix.h, which
// determines the # devices but does not crash if CUDA DLL missing
// (returning 0 instead), this was factored out from
// msra::dbn::numcudadevices() so we can share it with lattice code;
// parallelstate() constructor now uses proper numcudadevices() function
// to determine whether CUDA is available (before just assumed it is,
// which was an early hack)
// 
// 67    9/18/12 11:15a Fseide
// made asum() const
// 
// 66    9/18/12 11:10a Fseide
// new method adagradientfromdenom()
// 
// 65    9/18/12 10:27a Fseide
// two new methods for AdaGrad
// 
// 64    9/18/12 10:03a Fseide
// switched accumulatesqr() to CUDA mode
// 
// 63    9/17/12 6:10p Fseide
// new method accumulatesqr()
// 
// 62    9/16/12 4:34p Fseide
// new method setzero()
// 
// 61    9/04/12 1:54p Fseide
// new class readonlymatrix, which is accessed through the texture
// machinery and its special caching structure
// 
// 60    8/28/12 5:05p Fseide
// (bug fix; resize -> allocate())
// 
// 59    8/28/12 4:06p Fseide
// rename vectorbase::resize() to allocate(), since it does not resize()
// like STL which retains the content;
// fixed ~vectorbase()
// 
// 58    8/28/12 3:56p Fseide
// implemented vectorbase::assign() and fetch() (not tested)
// 
// 57    8/28/12 3:26p Fseide
// changed vectorbase::size() to const no-throw
// 
// 56    8/28/12 3:14p Fseide
// added a std::vector<> version of fetch() and assign() to vectorbase<>
// 
// 55    8/28/12 2:59p Fseide
// (fixed tabs...meh)
// 
// 54    8/28/12 2:50p Fseide
// new (base) class vectorbase<>;
// changed WaitFor() to inline to avoid error from dup definition
// 
// 53    8/07/12 9:42 Fseide
// now defines CopyFlags even if NOCUDA, to make some code easier
// 
// 52    8/02/12 12:24p F-gli
// changed check-in on 7/17/12 by Adame outside NOCUDA, because it is not
// related to latgen build
// 
// 51    7/27/12 2:51p V-hansu
// encoding in GB2312, I do not know why vs2010 ask me to change it again
// and again. Acutally nothing has been changed.
// 
// 50    7/17/12 5:31p Adame
// Update for no-sync framework
// async copy fixes
// 
// 49    6/24/12 9:27p V-xieche
// switch code into a work point(an old version as well).
// 
// 47    6/08/12 8:36p V-xieche
// add a flag to decide to use async copy or sync copy. Need to improve it
// later.
// 
// 46    4/05/12 9:52p V-xieche
// add functions for posteriorstats in striped toplayer pipeline training.
// not finished yet.
// 
// 45    4/01/12 2:05p Fseide
// seterrorsignal now takes an offset parameter so that it can work for
// vertical stripes
// 
// 44    4/01/12 2:00p V-xieche
// code for striped seterror signal
// 
// 43    4/01/12 11:24a V-xieche
// add code for striped softmax computation in 2 gpu.
// 
// 42    3/31/12 19:16 Fseide
// new method assign() from another matrix
// 
// 41    2/26/12 6:58p V-xieche
// Add codes for coping date between CUDA device.
// 
// 40    2/25/12 5:24p V-xieche
// Add helpler function for coping date in CUDA device
// 
// 39    1/01/12 10:33a Fseide
// (added a comment)
// 
// 38    12/06/11 5:47p Dongyu
// #include <stdexcept>
// 
// 37    11/28/11 5:56p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 36    11/23/11 4:03p Dongyu
// add reshape and KhatriRaoProduct
// 
// 35    11/04/11 14:54 Fseide
// new parameter for addrowsum()
// 
// 34    10/25/11 5:17p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 33    10/06/11 5:16p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 32    6/21/11 13:40 Fseide
// added frame for new function patchasblockdiagonal(), but inner loop not
// implemented yet
// 
// 31    6/10/11 7:46 Fseide
// removed explicit #undef NOCUDA so we can #define it inside the CPP file
// 
// 30    3/02/11 9:35a Dongyu
// add setto0ifabsbelow definition
// 
// 29    2/26/11 4:31p Fseide
// new method softmax()
// 
// 28    2/25/11 5:55p Fseide
// new method synchronize();
// assign(0 and fetch() now take a parameter to run sync or async
// 
// 27    2/11/11 1:50p Fseide
// rolled back previous check-in
// 
// 26    2/11/11 1:47p Fseide
// 
// 25    2/10/11 1:14p Fseide
// new method posteriorstats()
// 
// 24    2/10/11 11:21a Fseide
// new method mulbydsigm
// 
// 23    2/10/11 10:56a Fseide
// new method setbackpropagationerrorsignal()
// 
// 22    2/07/11 9:34p Fseide
// new method llstats()
// 
// 21    2/07/11 7:03p Fseide
// new method addtoallcolumns()
// 
// 20    2/07/11 6:38p Fseide
// new method samplebinary()
// 
// 19    2/07/11 6:13p Fseide
// new method sigmoid()
// 
// 18    2/07/11 6:04p Fseide
// new method addrowsum()
// 
// 17    2/05/11 8:55p Fseide
// new method patch()
// 
// 16    2/02/11 8:03a Fseide
// gemm() now allows B also to be transposed
// 
// 15    2/01/11 4:52p Fseide
// deleted addcol()
// 
// 14    2/01/11 15:32 Fseide
// new CUDA method addcol for column-wise addition (to add bias)
// 
// 13    2/01/11 14:55 Fseide
// replaced dummy operator+= by method gems()
// 
// 12    2/01/11 13:52 Fseide
// added NOCUDA compilation mode
// 
// 11    1/31/11 3:31p Fseide
// (forgot to make operator+= pure virtual)
// 
// 10    1/31/11 2:47p Fseide
// matrix is now an interface
// 
// 9     1/31/11 8:38a Fseide
// (added a test() function)
// 
// 8     1/30/11 11:44p Fseide
// renamed numdevices() to getnumdevices() as it seemed to have conflicted
// with the other declaration
// 
// 7     1/30/11 11:37p Fseide
// fixed wrong #pragma
// 
// 6     1/30/11 11:30p Fseide
// now references the cudamatrix DLL
// 
// 5     1/30/11 11:21p Fseide
// added numdevices() to msra::cuda in cudamatrix.h
// 
// 4     1/30/11 11:19p Fseide
// changed to DLL-export cudamatrix instead of cudalib
// 
// 3     1/30/11 17:54 Fseide
// updated the #include
// 
// 2     1/30/11 16:37 Fseide
// added missing #pragma once
// 
// 1     1/30/11 16:29 Fseide
// CUDA-related source files added (currently empty placeholders)

#pragma once
#include <stdexcept>    // (for NOCUDA version only)
#include <vector>       // for quantization

//#define NOCUDA      // define this to skip CUDA components (will act as if no CUDA device)

#ifndef NOCUDA
#ifdef DLLEXPORT
#define EXPORT __declspec(dllexport)
#else
//#define EXPORT __declspec(dllimport)  // won't work if we link to cudamatrix.lib statically
#define EXPORT                          // this works for both static and DLL linking
#pragma comment (lib, "cudamatrix.lib")
#endif
#endif

namespace msra { namespace cuda {

// parameters needed for convolution
struct convolutionParams
{
    size_t prevBands;
    size_t prevKernels; 
    size_t minibatchSize; 
    size_t bands; 
    size_t kernels;
    size_t poolSize;
    size_t poolingBandShift; 
    size_t filterSize;
    size_t numFeatureSegments;
    convolutionParams(size_t prevBands, size_t prevKernels, size_t minibatchSize, size_t bands, size_t kernels, size_t poolSize, size_t poolingBandShift, size_t filterSize, size_t numFeatureSegments) :
        minibatchSize(minibatchSize), numFeatureSegments(numFeatureSegments), kernels(kernels), bands(bands), prevKernels(prevKernels),
            prevBands(prevBands), poolSize(poolSize), poolingBandShift(poolingBandShift), filterSize(filterSize)
    {}

    // default constructor will initialize to reasonable values
    // really only needed for file constructor
    convolutionParams() :
    minibatchSize(1024), numFeatureSegments(3), kernels(84), bands(20), prevKernels(45), prevBands(41), poolSize(6), poolingBandShift(2), filterSize(8)
    {}

};
struct/*interface*/ matrix
{
    virtual ~matrix() { }
    virtual void setdevice (size_t deviceid) = 0;
    virtual size_t setsubstream (size_t substream) = 0;
    virtual void allocate (size_t n, size_t m) = 0;
    virtual size_t rows() const throw() = 0;
    virtual size_t cols() const throw() = 0;
    virtual void reshape(const size_t newrows, const size_t newcols) = 0;
    virtual void KhatriRaoProduct(const matrix & m1, const matrix & m2) = 0;
    virtual matrix * patch (size_t i0, size_t i1, size_t j0, size_t j1) = 0;
    virtual const matrix * patch (size_t i0, size_t i1, size_t j0, size_t j1) const = 0;
    // transfer
    virtual void assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride, bool synchronize) = 0;
    virtual void syncassign() const = 0;
    virtual void fetch (size_t i0, size_t i1, size_t j0, size_t j1, float * pi0j0, size_t colstride, bool synchronize) const = 0;
	virtual void syncfetch() const = 0;
    // peer-to-peer transfer
    virtual void send (matrix & to, size_t i0, size_t i1, size_t j0, size_t j1) const = 0;
    virtual void receive (matrix & from, size_t i0, size_t i1, size_t j0, size_t j1) = 0;
    // sync --consider all functions asynchronous, call this if needed for data access or time measurement
	// Note: This does a device sync which is pretty brutal for efficiency. Avoid.
    virtual void synchronize() const = 0;
    // CUBLAS functions
    virtual void gemm (float thisscale, const matrix & A, bool Aistransposed, const matrix & B, bool Bistransposed, float ABweight) = 0;
    virtual void gems (float thisscale, const matrix & other, float otherweight) = 0;
    virtual float dot(const matrix & other) const = 0;
    virtual float nrm2() const = 0;
    virtual void colwisenrm2(matrix & ref, float maxcolnorm) const = 0;
    // additional specialized helpers with our own kernels
    virtual void setto0ifabsbelow (float threshold) = 0;
    virtual void setto0ifabsbelow2 (matrix &  ref, float threshold)=0;
    virtual void setto0ifabsabove2 (matrix &  ref, float threshold)=0;
    virtual void patchasblockdiagonal (size_t diagblocks, bool averageblocks, size_t firstcol) = 0;
    virtual void addrowsum (float beta, const matrix & othercols, float alpha) = 0;
    virtual void addrowsumpool (float beta, const matrix & other, float alpha, size_t poolSize, size_t bands, size_t kernels) = 0;
    virtual void sigmoid() = 0;
    virtual void samplebinary (const matrix & P, unsigned int randomseed) = 0;
    virtual void addtoallcolumns (const matrix & other, float otherweight = 1.0f) = 0;
    virtual void llstats (const matrix & v1, matrix & logllsums, bool gaussian) const = 0;
    virtual void softmax() = 0;
    virtual void sethessianvectorsignal(const matrix & Pu, const matrix &forwardStatistics) = 0;
    virtual void setdiagonalpreconditioner(const matrix & gradientsquared, float nobservations, float lambda, float alpha) = 0;
    virtual void elementwisedivision(const matrix &a, const matrix &b) = 0;
    virtual void elementwisesquare(const matrix &a) = 0;
    virtual float weighteddot(const matrix &weightingmatrix, const matrix &a) const = 0;
    virtual void setbackpropagationerrorsignal (const matrix & uids, const matrix & Pu, size_t i0, const matrix & senone2keepmodelupdate) = 0;
    virtual void setbackpropagationerrorsignalwithklreg (const matrix & uids, const matrix & Pu, const matrix & refPu, const float alpha) = 0;
    virtual void setbackpropagationerrorsignalhsmoothing (const matrix & uids, const matrix & Pu, const matrix & refmat, const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh) = 0;
    virtual void mulbydsigm (const matrix & sigm) = 0;
    virtual void posteriorstats (const matrix & Pu, size_t s0, bool nosoftmax, matrix & logpps, matrix & pps, matrix & maxlogpps) const = 0;
    virtual void reshapecolumnproduct (const matrix & eh, const matrix & h, const bool isehtransposed) = 0;
    // quantization
    virtual std::shared_ptr<char> newsharedtransferbuffer (size_t size) const = 0;
    virtual std::shared_ptr<struct qstripe> newqstripe (size_t size, bool prioritystream) const = 0;
    virtual void quantizeandfetchqstripe (const matrix & curresidual, size_t i0, size_t i1, size_t j0, size_t j1, qstripe * q, char * cpubufferbegin, size_t bits, matrix & newresidual, size_t reuserangescaled) const = 0;
    virtual void cpuneedsfetchqstripe (const struct qstripe * q) const = 0;
    virtual void assignqstripe (struct qstripe * q, const char * cpubufferbegin) const = 0;
    virtual void syncassignqstripeandunquantize (const qstripe * q, size_t bits, size_t i0, size_t i1, size_t j0, size_t j1, bool add) = 0;
    virtual void cpuneedsassignqstripe (const struct qstripe * q) const = 0;
    virtual void gradientfixups (const qstripe * q, matrix & adagradsqracc, float adagradkeepweight, float targetadagradavdenom_x_adagradsqrframes, float momentumkeepweight, matrix & smoothedgradient, float learningratepersample) = 0;
    // transfer data between matrices, potentially across devices
    virtual void assign (matrix & other, float * pi0j0/*CPU buffer in case it's needed*/, size_t colstride, bool synchronize, int copyFlags) = 0;
    virtual void stripedsoftmaxstep1 (matrix & partialframesums, size_t thisdevice) = 0;
    virtual void stripedsoftmaxstep2 (const matrix & partialframesums, size_t thisdevice) = 0;
#if 0  // transfer data between device. [v-xieche]
    virtual float * getpfromcuda () = 0;  // get the data point from cuda device. [v-xieche]
    virtual void copydatafromcudatocuda (matrix &dst, matrix & src, size_t size) = 0;
#endif

    virtual void reorder (matrix & to, const convolutionParams &convParams) const = 0;
    virtual void convolutionForward (matrix & out, const matrix & weight, const matrix & bias, const convolutionParams &convParams) = 0;
    virtual void computeCnnDeltaW (const matrix & deltaM, const matrix & vM, matrix & deltatM, matrix & vtM, float thisscale, float vhscale, const convolutionParams &convParams) = 0;
    virtual void maxpoolForward (matrix & out, matrix & maxIndex, const convolutionParams &convParams) = 0;
    virtual void maxpoolBack (matrix & out, const matrix & maxIndex, const convolutionParams &convParams) = 0;    
    virtual void dump(char* name) const = 0;    // dump the matrix for debug purposes
    virtual void setvalue (float value) = 0;
    // functions for AdaGrad
    virtual void accumulatesqr (const matrix & other, float keepweight) = 0;
    virtual void adadenom (const matrix & sqracc, float numframes, const matrix & meanacc, float meannumframes, size_t mbframes) = 0;
    virtual float asum() const = 0;
    virtual void adagradientfromdenom (const matrix & denom, const matrix & gradient, float actualavdenom, float targetavdenom, size_t mbframes) = 0;
    virtual void adagradientfromsqracc (const matrix & gradient, const matrix & sqracc, float numframes, float actualavdenom, float lrfudgefactor) = 0;
    virtual void prescaleforquantization (const matrix & sqrsum, double numframes, bool unscale) = 0;
    // this += other * weight, both in accelerated memory
    // This is used for model update.
    virtual void addweighted (const matrix & other, float weight) = 0;
    // for dropout
    virtual void dropout (float factor, unsigned int randomseed) = 0;
    virtual void scale (float factor) = 0;
    // for maxouts and others
    virtual void scalecolwise (const matrix & factors) = 0;
    virtual void scalerowwise (const matrix & factors) = 0;
    virtual void unscalerowwise (const matrix & factors) = 0;
    // for ReLUs
    virtual void setto0ifbelow (float threshold) = 0;
    virtual void mulbydlru (const matrix & lruvals) = 0;
    // for softplus
    virtual void softplus() = 0;
    virtual void mulbydsoftplus (const matrix & softpvals) = 0;
    // for leaky-root experiments (these may be deleted again if this does not work out, which is likely)
    virtual void leakyroot (size_t rootorder, float leakiness) = 0;
    virtual void mulbydleakyroot (const matrix & lruvals, size_t rootorder, float leakiness) = 0;
    virtual void mulbydmaxout (const matrix & h) = 0;
    // for MVN-SGD
    virtual void meanvarnorm (const matrix & mean, bool subtractmean, const matrix & diagvar, matrix & out) const = 0;
    virtual void meanvaracc (bool add, matrix & meanacc, const matrix & mean, matrix & varacc) = 0;
};

class onsubstream   // wrapper for switching a matrix to a substream and back at end of block (through destructor)
{
    void operator= (onsubstream &); onsubstream (onsubstream &);
    size_t prevsubstream;
    matrix & m;
public:
    onsubstream (matrix & m, size_t substream) : m(m) { prevsubstream = m.setsubstream (substream); }
    ~onsubstream () {  m.setsubstream (prevsubstream); }
};

// special version that implements a 'double' matrix, but only used for accumulation (we may change the name someday if used for other purposes)
struct/*interface*/ matrixaccumulator
{
    virtual ~matrixaccumulator() { }
    virtual void setdevice (size_t deviceid) = 0;
    virtual void allocate (size_t n, size_t m) = 0;
    // reset: reset accumulator to 0
    virtual void reset() = 0;
    // accumulate: this = thisscale * this + otherweight * other
    virtual void accumulate (float thisscale, const matrix & other, float otherweight) = 0;
    // read out result: to(,) = (float) this(,)
    virtual void tomatrix (matrix & to) const = 0;
};

// special version implemented through textures, which provides caching
struct/*interface*/ readonlymatrix
{
    virtual ~readonlymatrix() { }
    virtual void setdevice (size_t deviceid) = 0;
    virtual void allocate (size_t n, size_t m) = 0;
    virtual size_t rows() const throw() = 0;
    virtual size_t cols() const throw() = 0;
    virtual void assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride, bool synchronize) = 0;
};

// a vector type; use this as a basetype
// Note that this will only be instantiated for types known inside this lib, and a newvector<> function must be exported for each.
template<typename ELEMTYPE> struct/*interface*/ vectorbase
{
    virtual ~vectorbase() { }
    virtual void setdevice (size_t deviceid) = 0;
    virtual void allocate (size_t n) = 0;
    virtual size_t size() const throw() = 0;
    virtual void assign (const ELEMTYPE * p, size_t n, bool synchronize) = 0;
    template<class VECTOR> void assign (const VECTOR & v, bool synchronize) { allocate (v.size()); if (!v.empty()) assign (&v[0], v.size(), synchronize); }
    virtual void fetch (ELEMTYPE * p, size_t n, bool synchronize) const = 0;
    template<class VECTOR> void fetch (VECTOR & v, bool synchronize) const { v.resize (size()); if (!v.empty()) fetch (&v[0], v.size(), synchronize); }
    typedef ELEMTYPE elemtype;
};

enum CopyFlags
{
    copySync = 0,	        // use synchronous copy
    copyAsync = 1,	        // use asynchronous copy
    copyUsePassedBuffer = 2,	// use the passed buffer for Async
    copyDirect = 4,             // use universal addressing (UA), or peer to peer copy
    copyUseDestinationBuffers = 8, // for async copy we usually use the CPU buffers associated with the source GPU
                                // this flag swaps the usage to the destination buffers instead
};

#ifndef NOCUDA
//Events that can be fired and waited for, one set for each device
EXPORT enum Event{
    eventNil,	// invalid event code
    eventComputeComplete,	// compute event to be fired by user when needed
	eventUser1 = eventComputeComplete, // user event to be repurposed and renamed as needed
    eventComputeReady,		// compute event to be fired by user when needed
	eventUser2 = eventComputeReady, // user event to be repurposed and renamed as needed
    eventCopyToComplete,	// event fired by copy routines when CopyTo portion complete
    eventCopyToReady,		// copy to event to be fired by user when needed
	eventUser3 = eventCopyToReady, // user event to be repurposed and renamed as needed
    eventCopyFromComplete,	// event fired by copy routines when CopyFrom portion complete
    eventCopyFromReady,		// copy from event to be fired by user when needed
	eventUser4 = eventCopyFromReady, // user event to be repurposed and renamed as needed
};

// Operation (aligns with streams in cuda)
EXPORT enum Operation{
    operationNothing, // nothing to do next
    operationCompute, // compute next
    operationCopyTo, // copyTo is next
    operationCopyFrom, // copyFrom is next
};

// event fire/wait
struct EventContext;
struct OperationContext;
EXPORT void FireEvent(EventContext event);
EXPORT void WaitForEvent(EventContext event, OperationContext operation);
EXPORT void FireAndWaitEvent(EventContext event, OperationContext nextOperation);
EXPORT void PinBuffer(const float *bufHost, size_t size);
// unpin the buffer, if it wasn't pinned, do nothing
// WARNING: Unpin operations do a CPU sync
EXPORT void UnpinBuffer(const float *bufHost);
EXPORT bool IsPinned(const float *bufHost);
EXPORT void SyncDevice(size_t device);

// Operation context specifying the device that can fire the event, and the event ID
struct OperationContext
{
    size_t deviceId;
    Operation operationId;

    // create one
    OperationContext(size_t p_deviceId, Operation p_operationId) :
    deviceId(p_deviceId), operationId(p_operationId)
    {}

    void WaitFor(EventContext event);
};

// Event context specifying the device that can fire the event, and the event ID
struct EventContext
{
    size_t deviceId;
    Event eventId;

    // create one
    EventContext(size_t p_deviceId, Event p_eventId) :
    deviceId(p_deviceId), eventId(p_eventId)
    {}

    void Fire() {FireEvent(*this);}
    void ThenDo(OperationContext nextOperation) {WaitForEvent(*this, nextOperation);}
};

inline void OperationContext::WaitFor(EventContext event) {WaitForEvent(event, *this);}

//Copy commands:
EXPORT EventContext CopyBuffer(size_t deviceTo, float* deviceToBuffer, size_t deviceFrom, float* deviceFromBuffer, size_t sizeInBytes, float* hostBuffer=NULL);
EXPORT EventContext CopyBufferWait(EventContext eventToWaitFor, size_t deviceTo, float* deviceToBuffer, size_t deviceFrom, float* deviceFromBuffer, size_t sizeInBytes, float* hostBuffer=NULL);

//Comments:
//CopyBuffer() will wait for the last computeComplete event submitted to the queue to occur on the From device before starting the copy.
//CopyBufferWait() will wait for the eventToWaitFor event to occur before starting the copy
//Return value: the event that will fire when the copy is complete

const size_t deviceHost = 0xff; // (invalid device value) to signify a device buffer
//examples: 
//CopyBuffer(1, bufferDevice, deviceHost, bufferHost, sizeOfBuffer) ?this would copy from a host buffer to a device buffer
//CopyBuffer(1, bufferDevice, 2, bufferDevice2, sizeOfBuffer, bufferHostForCopy) ?copies from device2 to device1, using the bufferHostForCopy as an intermediate copy buffer
//CopyBuffer(deviceHost, bufferHost, 1, bufferDevice, sizeOfBuffer) ?this would copy from a device buffer to a host buffer


//example:
//sgemm(buf1, buf2, buf3,?);
//cudaEvent_t eventCopyComplete = CopyBuffer(1, buf2, 2, bufferDevice2, sizeOfBuffer, bufferHostForCopy);
//WaitForEvent(eventCopyComplete, 2); 
//These instruction will start a copy and when complete eventCopyComplete will be raised, all compute activity submitted to the queue on device 2 will wait for this event before running
//
#endif

#ifndef NOCUDA
// helper functions
// NOTE: there is a bug here if cudamatrix is built into a separate DLL with a mismatching compiler: caller cannot call delete in that case
EXPORT size_t getnumdevices();
EXPORT void disableasynccudaoperation();    // for debugging
EXPORT matrix * newmatrix();
EXPORT matrixaccumulator * newmatrixaccumulator();
EXPORT readonlymatrix * newreadonlymatrix();
EXPORT void test(); // test function
#else
static inline size_t getnumdevices() { return 0; }
static inline void disableasynccudaoperation() { }
static inline matrix * newmatrix() { throw std::runtime_error ("should not be here"); }
static inline matrixaccumulator * newmatrixaccumulator() { throw std::runtime_error ("should not be here"); }
static inline readonlymatrix * newreadonlymatrix() { throw std::runtime_error ("should not be here"); }
static inline void test() {}
#endif

// return number of CUDA devices, while catching a missing DLL as having 0 devices
static inline size_t numcudadevices()
{
    __try
    {
        return msra::cuda::getnumdevices();
    }
    __except (1/*EXCEPTION_EXECUTE_HANDLER, see excpt.h--not using constant to avoid Windows header in here*/)
    {
        fprintf (stderr, "numcudadevices: cudamatrix.dll or underlying CUDA DLLs not installed\n");
        return 0;
    }
}

};};

// -----------------------------------------------------------------------
// helper for timing
// -----------------------------------------------------------------------

#include <Windows.h>

// simple timer
// auto_timer timer; run(); double seconds = timer; // now can abandon the object
class timeme_base
{
    LARGE_INTEGER freq, start;
    timeme_base (const timeme_base &); void operator= (const timeme_base &);
public:
    timeme_base()
    {
        if (!QueryPerformanceFrequency (&freq)) // count ticks per second
            throw std::runtime_error ("timeme_base: QueryPerformanceFrequency failure");
        QueryPerformanceCounter (&start);
    }
    operator double() const     // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter (&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    }
};
class timeme : timeme_base
{
    const char * label;
    static int & getindent()
    {
        __declspec(thread) static int indent = 3-1;  // outer-most level has 3 +++
        return indent;
    }
public:
    timeme (const char * label) : label (label) { getindent()++; }
    ~timeme()
    {
        double elapsed = *this;
        size_t indent = getindent()--;
        fprintf (stderr, "%8.3f ms", elapsed * 1000.0/*to ms*/);
        for (size_t k = 0; k < 10; k++)
            fputc (k < indent ? ' ' : '-', stderr);
        fprintf (stderr, "%s [%d]\n", label, (int) ::GetCurrentThreadId());
        indent--;
    }
};
#define BEGINTIME(label) // { ::timeme ___(label);
#define ENDTIME() // }
#define BEGINTIME1(label) // { ::timeme ___(label);
#define ENDTIME1() // }
