// rbm.h -- implementation of layers for deep neural networks such as Hinton's Restricted Boltzmann Machine
//
// F. Seide, Nov 2010 based on code provided by Yu Dong, MSR Speech Research Group
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/rbm.h $
// 
// 539   7/04/14 13:53 Fseide
// SVD: moved 'w' buffer further up, and added more documentation on what
// the dimensions mean
// 
// 538   7/04/14 12:10 Fseide
// commented dimensions of SVD matrices
// 
// 537   7/04/14 12:01 Fseide
// moved w[] vector to inside RBM's svd() function, since it is not
// exposed or needed outside at all
// 
// 536   7/04/14 11:27 Fseide
// SVD: moved transposition of V inside the layer's svd() function;
// SVD now uses parallel_for() over layers
// 
// 535   7/04/14 11:07 Fseide
// some clean-up of SVD code
// 
// 534   7/04/14 10:48 Fseide
// rbmbernoullibernoulli() from vector of vectors is now 'const' correct
// and calls the renamed setfrom() function
// 
// 533   6/24/14 3:29p Ganl
// remove a warning statement
// 
// 532   6/11/14 6:51p Fseide
// bug fix: previous change incorrectly set the layer to linearkind,
// whereas that should be the other layer which was already set correctly
// (the original code was correct after all)
// 
// 531   6/11/14 4:43p Fseide
// (fixed a message)
// 
// 530   6/11/14 4:20p Fseide
// further clean-up of the SVD mess, to be tested someday
// 
// 529   6/06/14 6:05p Fseide
// bug fix in noptnetwork, template argument name was shadowed by a member
// of the same name;
// comments
// 
// 528   6/06/14 4:08p Fseide
// softplusnetwork implemented
// 
// 527   6/06/14 4:04p Fseide
// towards implementing the softplus non-linearity
// 
// 526   6/06/14 2:56p Fseide
// implemented the V in MVN, also added the full formula derivation as a
// comment (still does not work);
// fixed the mean/var update, now done consistently after the model update
// so that model update does not use an inconsistent value
// 
// 525   6/05/14 6:04p Fseide
// enablemvn flag (hack) now inside entercomputation();
// copyfrom() for linearnetworks tested and enabled
// 
// 524   5/30/14 4:34p Fseide
// (comment)
// 
// 523   5/30/14 4:28p Fseide
// preliminary bug fix in pushtorawgradient()--if 0 frames are passed in,
// it should nevertheless reset the gradient buffer, such that subsequent
// code (such as quantization) do not have to check for this condition.
// This would correctly be done inside the CUDA functions (they must
// interpret 'keepweight' correct in presence of an empty input) but this
// has not been done.
// 
// 522   5/26/14 2:39p Fseide
// (removed an unnecessary condition for mean/var update of MVN-SGD)
// 
// 521   5/22/14 3:27p Fseide
// in addgradienttomodel(), factored the various explicit calls to
// addweighted() into a shared function addtomodel() which now also takes
// care of MVN-SGD;
// reimplemented MVN-SGD with an in-place patch of v into v-v0 and
// post-correction in addtomodel()--this should be correct in presence of
// further gradient-processing steps like quantization, AdaGrad, or
// momentum
// 
// 520   5/20/14 9:25a Fseide
// (comments)
// 
// 519   5/16/14 5:41p Fseide
// added MVN-SGD, somewhat hacky in this version (to enable, locally
// change enablemvn to 1)
// 
// 518   5/15/14 7:24p Fseide
// variance normalization now implemented in class mvn;
// new method meanvaracc();
// meanvarnorm() now takes parameter whether to apply the mean or not
// 
// 517   5/15/14 5:33p Fseide
// (added init code for mvn var)
// 
// 516   5/15/14 5:29p Fseide
// added MPI to class mvn;
// glimpse() now works outside entercomputation()
// 
// 515   5/15/14 3:14p Fseide
// two minor bug fixes in class mvn, it now estimates and applies means
// (full test pending)
// 
// 514   5/15/14 14:17 Fseide
// changed gems() to use applyonsubstreams(), to ensure compatibility with
// model parallelism;
// completed class mvn (not tested yet)
// 
// 513   5/14/14 15:20 Fseide
// implemented mvn::forwardprop() (but underlying CUDA code not complete)
// 
// 512   5/14/14 14:40 Fseide
// new class mvn, to implement mean-normalized SGD;
// cleaned up the interface definition and added dummy throwing
// implementations for serveral more functions
// 
// 511   5/13/14 2:23p Fseide
// initrandom(): skipratio disabled again, it makes ReLU training slower,
// rather
// 
// 510   5/12/14 7:05p Fseide
// changed initrandom() to connect to a subset rather than using random
// values, since it got strange results (but just noticed a silly bug that
// could have been the cause)
// 
// 509   5/12/14 18:20 Fseide
// (comments)
// 
// 508   5/12/14 4:18p Fseide
// fixed random initialization (previous version tested nonlinearitykind
// before it was initialized by parent constructors)
// 
// 507   5/12/14 16:06 Fseide
// towards MN-SGD
// 
// 506   5/12/14 14:29 Fseide
// incompatible change: initialization for ReLU layers now using 100 x
// smaller random values
// 
// 505   5/12/14 13:36 Fseide
// added infrastructure to no longer require dropout models to be scaled
// later (they are now scaled upon exitcomputation(), and unscaled upon
// entercomputation())--note: still need to fix the actual forwardprop()
// function;
// GenerateMeanModel() renamed to dropoutscaling() (dbn.h) and
// dropoutscaleweights (rbm.h) since it is now also used to undo the
// change
// 
// 504   5/05/14 4:25p F-gli
// fixed bug in entercomputation(), added check if matrix is empty before
// setzero
// 
// 503   4/30/14 1:28p Fseide
// tidied up now that 'distributefixedcost' mode works
// 
// 502   4/30/14 11:17a Fseide
// bug fix: gradientfixups() now takes an optional qstripe to make it run
// on the associated CUDA stream, same as unquantization (without, it may
// run on the wrong stream and thus induce some non-deterministism);
// titied up some logging
// 
// 501   4/30/14 10:38a Fseide
// model update now done in unquantize function directly, saving a
// CUDA-side memory access
// 
// 500   4/30/14 10:31a Fseide
// learning-rate scaling now works when done in gradientfixups(), but
// currently suboptimally due to beign on the wrong CUDA stream
// 
// 499   4/29/14 3:26p Fseide
// this version works but does not add gradient directly from unquantizing
// 
// 498   4/29/14 2:22p Fseide
// towards learning-rate scaling in gradientfixups()
// 
// 497   4/29/14 1:44p Fseide
// minor fix in previous check-in
// 
// 496   4/29/14 1:42p Fseide
// unquantizeandaggregatestripe() now also takes the learning rate;
// gradientfixups() now multiplies the gradient with an additional
// learning-rate parameter -> this way it can compute the final-final
// thing that gets added to the model;
// syncassignaggregatedstripeandunquantize() now takes bpinfo so it can
// know about 'distributefixedcost' mode
// 
// 495   4/28/14 4:06p Fseide
// 'distributefixedcost' mode enabling logic implemented & tested,
// arguments passed to unquantizeandaggregatestripe(), but actual
// math/kernel not there yet (and thus, the mode is not enabled in the
// checked-in version)
// 
// 494   4/28/14 2:40p Fseide
// implemented direct use of MPI_Allreduce() in non-quantized case (as one
// would prefer for model-averaging case);
// made some methods robust to empty matrix dimensions;
// new methods mpiallreducegradient(), mpihelper::allreducescalar(), and
// allreduce() for matrices
// 
// 493   4/28/14 10:31a Fseide
// using the true mbframes across nodes (no rounding) now debugged and
// tested
// 
// 492   4/28/14 9:36 Fseide
// (comment)
// 
// 491   4/27/14 16:59 Fseide
// (comment)
// 
// 490   4/27/14 16:58 Fseide
// qpackages now contain a header (struct mpistripeheader) that stores the
// number of frames of the gradient they represent;
// syncassignaggregatedstripeandunquantize() no longer takes the
// 'numstripes' argument (it was unused and is superfluous now)
// 
// 489   4/27/14 16:11 Fseide
// new modelupdateinfo parameter 'distributefixedcost';
// unquantizeandaggregatestripe() now takes all parameters needed to
// distribute fixed cost (AdaGrad, momentum)
// 
// 488   4/27/14 15:35 Fseide
// (fix of last check-in)
// 
// 487   4/27/14 15:34 Fseide
// (towards distributed fixed cost)
// 
// 486   4/27/14 15:07 Fseide
// (comments)
// 
// 485   4/27/14 15:03 Fseide
// data parallelism: first steps towards doing part of fixed cost inside
// the stripe operation
// 
// 484   4/27/14 14:49 Fseide
// (minor tidying-up of previous check-in)
// 
// 483   4/27/14 14:44 Fseide
// moved maxnorm out from addgradienttomodel() into main
// backpropagationmodelupdate3() function
// 
// 482   4/09/14 16:21 Fseide
// renamed Ph to Eh (expectation of h) which is more correct
// 
// 481   4/09/14 16:01 Fseide
// (comments)
// 
// 480   4/09/14 8:41 Fseide
// changed last check-in to use sqrt(extranormgrowth) instead of
// extranormgrowth, to make the gradient scale consistently with the
// weight matrix
// 
// 479   4/08/14 21:42 Fseide
// now considering non-square matrices (but it does not seem to work
// better)
// 
// 478   4/06/14 20:46 Fseide
// updated the ReLU AWE code to get the first targetnorm from the first
// layer, as to avoid initial fluctuations (which did kill the training at
// one point)
// 
// 477   4/04/14 21:33 Fseide
// addgradienttomodel(): sorted out the old maxnorm implementation (by
// Pawel) against my experimental stuff (which is no longer compile-time
// enabled but enabled by passing -1 as the reg param)
// 
// 476   4/01/14 3:24p Fseide
// removed 'completed' from local-loop message so that we can grep again
// for this to get the epoch stats
// 
// 475   4/01/14 9:41a Fseide
// changed local loop to do averaging;
// fixed a warning about an unused function
// 
// 474   3/26/14 7:31p F-gli
// added a helper function nonlinearitykindtostring
// 
// 473   3/20/14 6:57p Fseide
// (removed an fflush())
// 
// 472   3/20/14 5:54p Fseide
// changed local-loop process to use a back-up model (because we need to
// restore that model as well)
// 
// 471   3/20/14 4:15p Fseide
// fixed the missing call to update3() in update2()
// 
// 470   3/20/14 4:07p Fseide
// some refactoring and initial code for local loop for data parallelism
// 
// 469   3/20/14 3:06p Fseide
// towards Kaldi-style model averaging for data parallelism
// 
// 468   3/20/14 2:20p Fseide
// addgradienttomodel() now takes the gradient variables as args, in prep
// for bypassing momentum
// 
// 467   3/20/14 11:39a Fseide
// (fixed a log msg)
// 
// 466   3/19/14 5:54p Fseide
// added REDISTRELU code path
// 
// 465   3/19/14 11:30a Fseide
// removed a log message for ReLU
// 
// 464   2/28/14 14:15 Fseide
// (added some commented-out code)
// 
// 463   2/27/14 20:19 Fseide
// disabled emulated double-buffering again (should not have been checked
// in enabled)
// 
// 462   2/27/14 19:21 Fseide
// (removed an unnecessary 'if' in entercomputation())
// 
// 461   2/27/14 18:52 Fseide
// new members raw2_dW,a,mbframes to allow to simulate double buffering
// without actual MPI
// 
// 460   2/25/14 4:15p Fseide
// switched AdaGrad default back to raw gradient (no benefit from sub
// gradient, it seems)
// 
// 459   2/25/14 1:46p Fseide
// (now counting #frames actually pushed to AdaGrad)
// 
// 458   2/25/14 11:51 Fseide
// comment
// 
// 457   2/24/14 18:24 Fseide
// onpartialsubgradient mode now allows for computing the avdenom on the
// first chunk, to at least have some tracking
// 
// 456   2/24/14 17:44 Fseide
// implemented partial-minibatch AdaGrad (tested the partial update, but
// not yet with AdaGrad), disabling for now
// 
// 455   2/24/14 15:45 Fseide
// pushtorawgradient() now takes the accumulators it should push to by
// reference, in prep for partial-minibatch AdaGrad;
// removed some old commented-out code
// 
// 454   2/24/14 15:26 Fseide
// (comment)
// 
// 453   2/22/14 3:35p Fseide
// changed to AdaGrad pre quantization;
// applyadagrad() now handles an empty AdaGrad accumulator correctly (by
// setting the output to 0)
// 
// 452   2/21/14 13:50 Fseide
// backpropagationmodelupdate2() now takes 'bpinfo' like all others;
// implemented sub-gradient AdaGrad in there (to be tested)
// 
// 451   2/21/14 13:44 Fseide
// replaced bool adagradonsmoothedgradient by an enum value
// 'adagradwhere', in prep of allowing to apply AdaGrad on the
// sub-gradients before quantization
// 
// 450   2/21/14 13:30 Fseide
// entermpiaggregation() now takes 'bits' as an explicit parameter
// 
// 449   2/20/14 7:46p Fseide
// (implementing lrfudgefactor slightly differently)
// 
// 448   2/20/14 6:07p Fseide
// switched AdaGrad to operate on raw gradient instead of smoothed one
// 
// 447   2/20/14 5:49p Fseide
// AdaGrad mean seems not working, now avoiding the mem alloc unless an
// #if 0 is changed to reenable it
// 
// 446   2/20/14 5:20p Fseide
// AdaGrad can now operate on the raw gradient as well (not tested yet,
// and not enabled)
// 
// 445   2/20/14 4:27p Fseide
// adadenom() now takes a mean accumulator as well (currently only used
// for diagnostics, not in actual AdaGrad normalizations, assuming a fixed
// target is given);
// likewise, network layer now keeps a mean accumulator matching the sqr
// acc for AdaGrad;
// applyadagrad() now again computes avdenom even if it is not used (i.e.
// when fixed target is given), for diagnostics;
// 
// 444   2/19/14 3:17p Fseide
// tidied up AdaGrad target avdenom, e.g. it is no longer divided by 32
// (which had been done for compat with even older cmd lines)
// 
// 443   2/19/14 3:01p Fseide
// cleaned up left-overs from when the LR was still part of the smoothed
// gradient;
// moved targetavdenom directly into addgradienttomodel since we now can
// do it there
// 
// 442   2/19/14 2:38p Fseide
// hard-coded a target av denom
// 
// 441   2/19/14 11:53a Fseide
// learning rate moved out of smoothed gradient
// 
// 440   2/19/14 11:19 Fseide
// (comment)
// 
// 439   2/19/14 9:20 Fseide
// addgradienttomodel() now takes a learning-rate parameter, in prep for
// finally removing the weird pre-scaling of the smoothed gradient (LR *
// 1/(1-momentum)) once and for all;
// some clean-up of comments and debug code
// 
// 438   2/18/14 4:54p Fseide
// fixed a missing pointer initialization in modelupdateparams constructor
// 
// 437   2/18/14 2:50p Fseide
// implemented multi-GPU support (model parallelism) for MPI (data
// parallelism), not tested yet but should at least work for single GPU;
// allocatetransferbuffer() now allocates on the correct GPU
// 
// 436   2/18/14 11:45a Fseide
// implemented sharing AdaGrad avdenom across all layers
// 
// 435   2/18/14 10:34a Fseide
// new class adagradstate_t to hold the cross-layer AdaGrad average
// implemented (passed around but not used yet)
// 
// 434   2/18/14 9:57a Fseide
// raw_dmbframes is now updated under the control of MPI aggregation. This
// is a preparation for separating out AdaGrad from ...update3().
// 
// 433   2/13/14 15:21 Fseide
// bug fix for variable Kopt (#nodes in MPI mode): in MPI mode, model is
// now being sync'ed from node 0 to all others at start of each epoch, as
// to get all nodes the latest model even if they did not participate in
// previous epochs and thus did not get their model updates
// 
// 432   2/07/14 14:52 Fseide
// unquantizeandaggregatestripe() now uses different buffers for each
// stream
// 
// 431   2/07/14 8:55 Fseide
// renamed quantizeaggregatedstripe() to
// quantizeandassignaggregatedstripe() and combined it with its subsequent
// call to assignaggregatedstripe(), for upcoming move to doing this on
// the GPU
// 
// 430   2/05/14 19:58 Fseide
// now switched to no targetavdenom, using the actual average instead (and
// no fudge factor either)
// 
// 429   2/05/14 19:26 Fseide
// adagradient() now accepts a fudge factor instead of a user-chosen
// target (not used yet)
// 
// 428   2/05/14 19:18 Fseide
// moved 'updateactualavdenom' flag out from parallelrbmmatrix
// 
// 427   2/05/14 18:46 Fseide
// shifted up the strange weight of targetavdenom by 32 out of the actual
// AdaGrad routines (we will eventually eliminate it)
// 
// 426   2/05/14 16:53 Fseide
// eliminated resetmomentum flag
// 
// 425   2/05/14 16:38 Fseide
// refactored the reset of the AdaGrad accumulators, in prep of removing
// the resetmomentum flag
// 
// 424   2/03/14 21:01 Fseide
// bug fix for deferred update: resetmomentum did not happen when
// mbiter.isfirst() was in a deferred chunk
// 
// 423   2/03/14 17:47 Fseide
// implemented AdaGrad-like pre-scaling for quantization, but does not
// work (disabled)
// 
// 422   2/03/14 16:32 Fseide
// (minor refactoring)
// 
// 421   2/03/14 16:09 Fseide
// changed AdaGrad implementation to isolate it out;
// minor fix of AdaGrad in interpreting old-style parameters (256 frames,
// now compatible with double-buffering=128 frames)
// 
// 420   2/02/14 22:19 Fseide
// AdaGrad now skips first update when double-buffering (which would use
// the fake zero gradient);
// check for first zero gradient moved from matrix function to rbm.h;
// forceaggregate=true for now so that we can test quant without data
// parallelism interference;
// disabled double-buffering for now since it does not play with AdaGrad,
// something wrong there
// 
// 419   1/30/14 19:16 Fseide
// gradient accs are now always reset to zero in entercomputation();
// momentum is now scaled again based on aggregate #frames *disregarding*
// double-buffering because it was wrong to do so
// 
// 418   1/27/14 10:14 Fseide
// redefined currently unused function backpropagationmodelupdate2() to
// run after a deferred update but before aggregation, so that we can do
// AdaGrad scaling here.
// 
// 417   1/26/14 20:04 Fseide
// switched momentum to do the "right thing": account for double
// buffering, and not brute-force momentum to 0 for deferred update
// 
// 416   1/26/14 14:50 Fseide
// recovered previous momentum behavior for deferred update (for
// compatibility)
// 
// 415   1/26/14 11:19 Fseide
// completely moved 'deferupdate' into dbn.h by just not calling aggregate
// and backpropagationmodelupdate3(), and rather have pushtorawgradient()
// and updategradient() control the state
// 
// 414   1/26/14 11:10 Fseide
// next step towards deferred update in raw gradient
// 
// 413   1/26/14 10:49 Fseide
// towards moving deferupdate to the raw gradient
// 
// 412   1/24/14 19:13 Fseide
// temp fix of raw_dmbframes, needs better fix
// 
// 411   1/24/14 11:06 Fseide
// towards setting numframes correctly in ...update3()
// 
// 410   1/21/14 11:49 Fseide
// bug fix in skipping the recomputation: forgot to scale to #nodes;
// 'accuracy' now set to 5 stddevs (before: 2), big difference for 16 bit
// in early iterations;
// bg thread disabled for now to get better-comparable log output
// 
// 409   1/20/14 17:04 Fseide
// bug fix in backpropagationmodelupdate3(): now post-corrects 'numframes'
// in presence of MPI data parallelism
// 
// 408   1/17/14 12:22 Fseide
// (added a missing 'return')
// 
// 407   1/17/14 11:32 Fseide
// allocatetransferbuffer() implemented, now uses cudamatrix lib's
// newsharedtransferbuffer() when in CUDA mode
// 
// 406   1/17/14 10:54 Fseide
// MPI: changed bufferbegin/end to bufferbegin/size
// 
// 405   1/17/14 9:21 Fseide
// streamlined enter/exitmpiaggregation() a little, such that on matrix
// level, those functions no longer know about the mpiaggregator
// 
// 404   1/14/14 18:35 Fseide
// now skipping second computerange() (instead, we reuse the range
// determined on our local stripe)--to be verified once we have multiple
// nodes
// 
// 403   1/10/14 17:13 Fseide
// bug fix: corrected invalid parameter list for override of
// quantizeaggregatedstripe()
// 
// 402   1/10/14 16:51 Fseide
// syncassignaggregatedstripeandunquantize() now takes CPU-side buffer so
// that it can operate on the CPU without GPU
// 
// 401   1/10/14 11:22 Fseide
// moved MPI quantization residuals out of rbmbase into
// parallelrbmmatrix.h where they belong
// 
// 400   1/10/14 11:01 Fseide
// renamed at/detachmpiaggregator() to enter/exitmpiaggregation()
// 
// 399   1/09/14 17:53 Fseide
// began to implement the big fat MPI aggregator using the lambdas
// 
// 398   1/09/14 17:06 Fseide
// last lambda for MPI aggregation implemented, now on to the MPI
// aggregator function itself!
// 
// 397   1/09/14 16:55 Fseide
// quantizeaggregatedstripe() implemented
// 
// 396   1/09/14 15:55 Fseide
// added more lambdas for MPI aggregation
// 
// 395   1/09/14 15:06 Fseide
// towards implementing the lambdas that the MPI aggregator needs
// 
// 394   1/09/14 9:17 Fseide
// backpropagationmodelupdate2() now just takes the mpiaggregator directly
// (i.e it now knows that it is for this specific purpose)
// 
// 393   1/08/14 18:25 Fseide
// bug fix: it can now again write IPE-compatible 'linearkind' files that
// arise out of SVD (so we can use their older version to read them)
// 
// 392   1/08/14 18:10 Fseide
// documented a bug
// 
// 391   1/08/14 18:07 Fseide
// networktypedesc_t() failed to bit-blast itself to 0 first, so that
// operator!= can use memcmp() safely
// 
// 390   1/08/14 18:04 Fseide
// bug fix in reading old IPE SVD models (islinearoverride was tested at
// the wrong place)
// 
// 389   1/08/14 17:11 Fseide
// towards MPI and quantization (currently totally broken since MPI
// transfer is not cross-layer);
// mpiaggregate() now takes a second residual parameter
// 
// 388   1/08/14 11:00 Fseide
// (comments)
// 
// 387   1/08/14 10:52 Fseide
// (some tidying-up of comments; moved updategradient() back to where it
// belongs)
// 
// 386   1/08/14 10:42 Fseide
// renamed updatedeltas3() to updategradient() and adddeltas() to
// addgradienttomodel()
// 
// 385   1/08/14 10:34 Fseide
// mpistriperefs_dx and qstripes_dx moved to parallelrbmmatrix.h
// 
// 384   1/08/14 9:54 Fseide
// towards moving MPI data exchange from rbm.h to parallelrbmmatrix.h
// 
// 383   1/08/14 9:44 Fseide
// qstripe no longer knows about patch dimension and 'bits' parameter
// (none of its business)
// 
// 382   1/08/14 8:52 Fseide
// (comments)
// 
// 381   1/08/14 8:49 Fseide
// qstripe now does not take a buffer+offset but begin and end iterator;
// qstripe is now handed back from cudamatrix lib as a shared_ptr with
// custom deleter;
// MPI initialization sequence: entermpiaggregation() determines the
// needed stripe sizes and buffer offsets, while first use (within-layer)
// will lazily allocate (cross-layer) stripe buffers;
// split off mpihelper from mpiaggregator (mpihelper handles basic MPI
// interaction)
// 
// 380   1/06/14 9:45p V-haofu
// change the nucudadevices() condtion error
// 
// 379   1/03/14 19:14 Fseide
// (added comments on create_task())
// 
// 378   1/03/14 17:47 Fseide
// backpropagationmodelupdate2() should now be correct for the new
// structure
// 
// 377   1/03/14 17:26 Fseide
// initialization sequence of MPI stuff stratified, new methods
// entermpiaggregation() and exitmpiaggregation() in rbm.h, called by
// respective functions in dbn.h;
// moved logic to determine buffer size etc. from CUDA-side qstripe to a
// new CPU-side structure mpistripebufferref (since stripes may live on
// different GPUs, the GPU side cannot do this);
// stripes are now associated with a GPU (in theory--the actual
// determination of the GPU device is not implemented)
// 
// 376   1/03/14 16:01 Fseide
// towards MPI data-parallel gradient aggregation
// 
// 375   1/03/14 13:02 Fseide
// changed quantizeunquantize() to use quantizer class, for testing it
// 
// 374   1/03/14 10:48 Fseide
// some code simplifications that arose out of the previous change
// 
// 373   1/03/14 10:28 Fseide
// finally removed NEWGRADIENTSCALING as a #define-based option, we always
// use this anyway
// 
// 372   1/03/14 10:15 Fseide
// renamed updatedeltas1() to pushtorawgradient();
// cleaned up some derived types to use backpropagationmodelupdate1..3;
// removed updatedeltas2() and old backpropagationmodelupdate() (kept
// #if-0'ed out for reference)
// 
// 371   1/03/14 12:21a V-haofu
// use new structure to implement backpropagationmodelupdate2 (not
// verified)
// 
// 370   12/20/13 7:18p V-haofu
// fake split all other versions of backpropagationmodelupdate into 3
// functions;
// for each layer call backpropagationmodelupdate1..3 instead;
// 
// 369   12/20/13 5:49p V-haofu
// split backpropagationmodelupdate function into 3 steps and enable it.
// 
// 368   12/20/13 4:55p V-haofu
// move updatedeltas1,2 at the very beginning in the
// backpropagationmodelupdate function;
// cancle all the calls of updatedeltas in rbm class
// 
// 367   12/20/13 4:40p V-haofu
// comment out updatedeltas2(NULL) in pretrainingmodelupdate function
// 
// 366   12/20/13 4:37p V-haofu
// fix the mistake: comment wrong place of previous check in
// 
// 365   12/20/13 4:35p V-haofu
// comment out unnecessary updatedeltas call in backpropagationmodelupdate
// function, deferupdate case
// 
// 364   12/20/13 4:21p V-haofu
// delete the class quantizer in rbm.h;
// replace calls of updatedeltas() with updatedeltas1..3();
// 
// 363   12/20/13 15:24 Fseide
// minor reshuffling between updatedelta() pieces
// 
// 362   12/20/13 3:08p V-haofu
// change the if conditions in updatedeltas1,2;
// enable the code inside updatedeltas();
// 
// 361   12/20/13 2:54p V-haofu
// move initialization(and enter/exit computation) of raw_dW/a/b out of
// mpi initialization functions;
// move updatedeltas1..3 out side of class rbmbase;
// disabled 3 functions in updatedeltas;
// 
// 360   12/20/13 2:10p V-haofu
// enables updatedeltas1..3 inside the function updatedeltas
// 
// 359   12/20/13 2:08p V-haofu
// move other codes after updatedeltas into updategradient
// 
// 358   12/19/13 10:00p V-haofu
// put checks into #if 1 and move checks into updatedeltas1
// 
// 357   12/19/13 9:55p V-haofu
// move updategradient before all cases
// 
// 356   12/19/13 9:14p V-haofu
// move updatedeltas2 before all cases
// 
// 355   12/19/13 9:02p V-haofu
// move updatedeltas1 after declaration of class convolutional 
// 
// 354   12/19/13 8:54p V-haofu
// Start incoprating branches into 3 functinos. Firstly, move branches
// into updatedeltas1 and move it to the begining.
// need to handle the updateb case and convolutional case  BUG: not
// compiled, need to move after declaration of class convolutional
// 
// 353   12/19/13 7:51p V-haofu
// implement updatedeltas2 for step2 (only for mpi case)
// 
// 352   12/19/13 7:46p V-haofu
// change interface of updatedeltas1 and updategradient
// 
// 351   12/19/13 7:34p V-haofu
// implement updategradient for step3 (for all cases)
// 
// 350   12/19/13 7:30p V-haofu
// implement updatedeltas1 for step1(not for convolutional case)
// 
// 349   12/19/13 7:24p V-haofu
// seperat gradients calculation into 2 steps for convolutional case and
// common case
// 
// 348   12/19/13 7:16p V-haofu
// change comments in #else part, mpi case
// 
// 347   12/19/13 7:10p V-haofu
// start refactoring step by step: add raw_db and its inialization
// function; put lazyinit at the beggining of all cases
// 
// 346   12/19/13 5:36p V-haofu
// split the gradients accumulation into 2(3) functions in all cases:
// calcualte raw gradients, mpi all reduce(if in mpi mode), accumulate raw
// gradients into momentum sum.  TODO: to verify the correctness for
// convolutional case
// 
// 345   12/18/13 10:21 Fseide
// design comment on data parallelism async stuff
// 
// 344   12/18/13 10:03 Fseide
// (design comment added regarding data parallelism)
// 
// 343   11/28/13 3:57p V-haofu
// modify the quantization for 1 bit case(use mean to unquantize the data)
// 
// 342   11/01/13 4:33p V-haofu
// implementation of quantization for data parallelization
// 
// 341   11/01/13 4:30p V-haofu
// 
// 340   10/21/13 4:27p V-haofu
// add entercomputation() and exitcomputation() to fix a bug
// 
// 339   10/18/13 4:47p V-haofu
// fix a bug of declaration of raw_dW,raw_da
// 
// 338   10/16/13 6:01p V-haofu
// move the declaration of raw_da,dw right before the updatedeltas(). do
// resize before using raw_da,dw
// 
// 337   10/16/13 5:43p V-haofu
// in mpi mode, we calculate and exchange sum of gradients(before momentum
// accumulation), then do momentum accumulation in each node
// 
// 336   9/29/13 13:18 Fseide
// now passes a buffer variable to matprod_mm() to support model
// parallelism
// 
// 335   9/27/13 11:37 Fseide
// renamed vtoh() to vtoz() since that is more correct
// 
// 334   9/25/13 17:51 Fseide
// towards model parallelism for softmax()--added the buffer
// 
// 333   9/16/13 3:34p Fseide
// disabled #define COMPACTRAINER and made it compile again
// 
// 332   9/16/13 3:20p Fseide
// disabled #define MULTICUDA and made it compile again (some code was not
// guarded)
// 
// 331   9/16/13 10:51a Fseide
// (added a comment)
// 
// 330   9/16/13 10:11a Fseide
// applytransform() now implemented
// 
// 329   9/16/13 9:56a Fseide
// new methods flipsigmoids(), flippolarity() and applytransform() for
// some weird experiment aimed at understanding seeming sparseness
// 
// 328   9/11/13 6:12p Fseide
// added comments for further tidy-up of AdaGrad
// 
// 327   9/11/13 10:48a Fseide
// tidied up adagradient(), removing the weird factor of sqrt(mbframes)
// (... and putting it back in where needed to maintain back compat, but
// at least now we know where it really goes, and thus where we shall
// remove it if we decide to break compat)
// 
// 326   9/10/13 10:15a V-haofu
// delete mbframes parameter in all accumulatesqr related functions
// 
// 325   9/06/13 3:22p Fseide
// yups, virtual destructor must be public; and we shouldn't have removed
// the default constructor
// 
// 324   9/06/13 3:17p V-haofu
// Iannlayer() lacked a virtual destructor, so object was never properly
// torn down, which showed only on CV where we reload the model
// 
// 323   8/28/13 3:15p Fseide
// enabled copyfrom() in relunetwork
// 
// 322   8/28/13 11:23a Fseide
// changed the combo of decltype and auto to explicitly putting the type
// there, as otherwise I get compiler and internal linker errors (bugs)
// 
// 321   8/28/13 10:58a Fseide
// copyfrom() implemented in all layer classes (some as dummies, some
// throwing "untested"), but compilation now fails (checking in so I can
// stepwise modify things to see where it comes from--in Release it's an
// internal link error, so possibly a linker bug related to decltype())
// 
// 320   8/27/13 8:22p Fseide
// new methods model::backto()/restorefrom() and
// Iannlayer::clone()/copyfrom() to support model backup for lookahead LR
// tuning (clone() and copyfrom() are actually not implemented yet for any
// class)
// 
// 319   8/26/13 6:17p V-haofu
// added code for learningrate adjustment of top layer(softmax)
// 
// 318   8/23/13 8:52a Fseide
// added a missing NULL check in addgradienttomodel() to Pawel's code
// 
// 317   8/15/13 4:16p V-haofu
// a bug fix in exitcomputation() for norms
// 
// 316   8/15/13 3:25p V-haofu
// revert back to earlier AdaGrad setup that worked well
// 
// 315   8/07/13 4:18p T-paswie
// mainly maxout-realated code, columnwise regularization now moved out
// for general use thourgh , for example: --regularization L2C --regparams
// 2.5, small changes to maxpool class (backpropagation for maxouts)
// 
// 314   7/08/13 4:52p T-paswie
// some bugfixes to columnwise-based normalisation
// 
// 313   7/05/13 9:04p T-paswie
// colwise norms now propagated around the matrices code and used with
// maxouts
// 
// 312   7/04/13 9:30 Fseide
// fixed a few non-ASCII characters for ' " and -- that had snuck in
// through a copy-paste operation from an e-mail and caused issues for
// non-English codepage workstations
// 
// 311   7/03/13 6:25p T-paswie
// releasing lock
// 
// 310   6/28/13 4:34p T-paswie
// added poolSize to maxout constructors, commeting out some debug
// messages
// 
// 309   6/28/13 3:50p T-paswie
// maxouts debugged, looks like are learning something. 
// 
// 308   6/27/13 8:44p T-paswie
// maxouts cont. - implemented, not tested + Frank's comment regarding
// convnolutional nets
// 
// 307   6/21/13 6:45p T-paswie
// some infrastructure for maxout units
// 
// 306   6/10/13 17:57 Fseide
// began to comment convolutional nets (not completed, not even correct
// yet)
// 
// 305   6/07/13 20:37 Fseide
// added experimental generalization of recitifed linear units using an
// optional non-linearity (n-th root) and leakiness
// 
// 304   6/07/13 18:46 Fseide
// (fixed a message)
// 
// 303   6/07/13 18:15 Fseide
// new method avsqr();
// now prints diagnostics of gradient value range for relus, to check
// balance
// 
// 302   6/07/13 14:18 Fseide
// MPI mode now gets momentum right (subject to testing)
// 
// 301   6/06/13 20:10 Fseide
// updatedeltas() now performs MPI-based aggregation if enabled (but no
// code yet to enable it);
// new method rbmmodelmatrixbase::mpiallreduce() to implement this
// 
// 300   6/06/13 5:16p T-paswie
// 1) copymodel command 2) fix to dowrite() altering default description
// of output layer to softmax (reaults in BTYP tag is not saved and the
// DBN structure is backward compatible with latgen)
// 
// 299   6/06/13 14:34 Fseide
// global model::convparams and its setter removed, convolution-related
// networks now get their config through the layerconfigparameters;
// layerconfigval type casts now implemented, i.e. it is fully functional;
// --convolutionparams is now mapped to the layerconfigparameters
// mechanism, from which 'convolutional' and 'maxpool' will now retrieve
// their parameters
// 
// 298   6/04/13 15:39 Fseide
// -D option now implemented
// 
// 297   6/04/13 14:26 Fseide
// infrastructure for passing creation parameters to network
// instantiations (actual parsing and use to be implemented)
// 
// 296   6/03/13 21:34 Fseide
// renamed class rlu to relunetwork (we will use relu for a non-linearity
// layer for use with CNNs)
// 
// 295   6/03/13 19:12 Fseide
// (added a comment)
// 
// 294   6/03/13 7:32 Fseide
// sorted the functions in rbmbase, esp. all that experimental stuff
// towards the end
// 
// 293   6/03/13 6:44 Fseide
// perceptron::forwardprop() removed, since covered by base class
// 
// 292   6/03/13 6:36 Fseide
// moved forwardprop() and backpropagationstats() to rbmbase class since
// they are now shared by linearnetwork and perceptron
// 
// 291   6/03/13 6:29 Fseide
// perceptron::backpropagationstats() removed since now covered by
// rbmbase;
// bug fix: rbmbase::write() forgot to write the desc size
// 
// 290   6/03/13 4:51 Fseide
// rbmbase construct from file now prints the type descriptor if one is
// found
// 
// 289   6/02/13 13:28 Fseide
// write() now writes a networktypedesc_t structure (BTYP/ETYP) if it is
// not the default
// 
// 288   6/02/13 8:14 Fseide
// backpropagationstats() implemented for LRU;
// mulbydlru() implemented
// 
// 287   6/02/13 7:49 Fseide
// towards implementing the ReLU as a RBM with nonlinearity = relukind
// 
// 286   6/02/13 7:25 Fseide
// infrastructure for storing non-linearity kind in the model file;
// removed some duplicate file-reading functions (FILE*, HANDLE) into
// function templates
// 
// 285   6/02/13 4:03 Fseide
// towards ReLUs
// 
// 284   6/02/13 3:34 Fseide
// lots of code hygiene for SVD implementation, including rename 'flag'
// (in an interface!) to something meaningful and lots of formatting
// inconsistencies;
// 
// 283   4/09/13 8:53p V-hansu
// (fix some comments)
// 
// 282   4/08/13 7:09p V-hansu
// remove NO_A_UPDATE_FOR_TOP
// 
// 281   4/04/13 10:09a Jianxue
// Changed for SVD decomposition and retraining. Add a flag for class
// Iannlayer, default is 0. If it's 1, means no nonlinear function. Add
// function Do_SVD.
// 
// 280   1/03/13 8:54p Kaisheny
// Asynchronous SGD using data pipe.
// 
// 279   12/19/12 10:46p Kaisheny
// Initial version of asynchronous stochastic gradient descent algorithm
// for distributed training of DNN. 
// 
// 278   12/07/12 5:24a Adame
// convolution/maxpool support (GPU only)
// --convolutionalParams flag to support convolution parameters
// --addEnergy flag to add energy to datasets (such as HVT)
// --asyncopy flag to enable asynccopy on multi-GPU setups with pipeline
// trainer
// zero out all arrays on creation (eliminate NANs)
// 
// 277   11/20/12 14:00 Fseide
// fixed int/size_t correctness in some HF functions for Win32 builds
// 
// 276   11/17/12 4:11p V-hansu
// add the derivation of deltah to forwardpropdelta()
// 
// 275   11/17/12 3:55p Fseide
// unseen-state compensation now covers 'a' as well
// 
// 274   11/16/12 9:10p V-hansu
// modify forwardpropdelta() because one step is already done in
// errorbackprop
// 
// 273   11/16/12 7:16p Fseide
// forwardpropwithoutbias() cleaned up
// 
// 272   11/16/12 6:45p Fseide
// compensationupdate() no longer needed/used in this form
// 
// 271   11/16/12 5:48p Fseide
// forwardpropdelta() now returns its 'eps', which is then passed to
// compensationupdate()
// 
// 270   11/16/12 5:39p Fseide
// forwardpropdelta() now returns the eps it used
// 
// 269   11/16/12 5:37p Fseide
// fixed forwardpropdelta()
// 
// 268   11/16/12 5:34p Fseide
// forwardpropdelta() now takes learning rate per frame and momentum per
// sample;
// so does unseenstatecompensaion(), but that is not fully implemented yet
// 
// 267   11/16/12 4:58p Fseide
// added even more dim checks to forwardpropdelta()
// 
// 266   11/16/12 4:56p Fseide
// added more dim checks to forwardpropdelta()
// 
// 265   11/16/12 4:54p Fseide
// forwardpropdelta() now checks for empty deltav
// 
// 264   11/16/12 4:17p V-hansu
// modify the forwardpropdelta(), seems done, not tested
// 
// 263   11/16/12 2:44p V-hansu
// add some code to forwardpropdelta()
// 
// 262   11/16/12 11:02a V-hansu
// add eh into forwardpropdelta() since it will be used
// 
// 261   11/16/12 10:50a V-hansu
// modify scalederrorE to layerstatenorm, and scaleE to vnorms, add method
// allocatestatevectors()
// 
// 260   11/15/12 7:49p V-hansu
// (add comments to forwardpropdelta())
// 
// 259   11/15/12 7:35p V-hansu
// add another matrixref scaledE in forwardpropdelta() to save middle
// results
// 
// 258   11/14/12 7:26p V-hansu
// add forwardpropdelta(), not finished
// 
// 257   11/11/12 5:16p V-hansu
// modify code relating to compensation, change the compensateupdate()
// 
// 256   11/10/12 5:18p V-hansu
// add NO_A_UPDATE_FOR_TOP, not turned on
// 
// 255   11/10/12 3:26p V-hansu
// add compensationupdate(), evtoeh() and forwardpropwithoutbias()
// 
// 254   11/08/12 4:32p T-simonw
// add double precision accumulators and corresponding methods
// 
// 253   11/02/12 4:32p T-simonw
// code formatting and documentation
// 
// 252   11/02/12 11:38a Fseide
// doublenodes(): updated the second part (receiving layer) to match the
// previous change for output layer
// 
// 251   11/02/12 11:36a Fseide
// doublenodes() changed to split by copying a whole section rather than
// neighboring nodes
// 
// 250   10/31/12 1:47p T-simonw
// Hessian-free: bugfix, use correct setvalue method
// 
// 249   10/31/12 1:44p T-simonw
// use setzero instead of setvalue for initialization of Hessian-free
// statistics
// 
// 248   10/31/12 10:05a T-simonw
// add Hessian free optimization methods
// 
// 247   10/28/12 10:15p V-hansu
// (change tabs to spaces)
// 
// 246   10/17/12 5:00p Dongyu
// fixed several copy&paste errors in the dropout code. 
// 
// 245   10/16/12 11:39a Fseide
// forwardprop() now supports Hinton's drop-out method (currently requires
// a compile-time #if 0 to be changed to #if 1)
// 
// 244   10/12/12 1:49p Dongyu
// added support of dropout training for DNN (frame level training only). 
// addes support to convert the model based on dropout rate used in the
// training and/or senone sections used in multilingual training.
// 
// 243   10/10/12 10:03a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new langauges.
// 
// 242   10/09/12 7:22p Fseide
// moved definition of matrix, matrixbase, matrixstripe, and vector from
// rbm.h to seematrix.h
// 
// 241   10/09/12 6:44p Fseide
// changed rand() to ::rand() because there was some conflict
// 
// 240   10/05/12 1:13p Fseide
// (added a comment)
// 
// 239   10/04/12 3:29p Fseide
// adagradient() is now passed a manually adjusted targetadagradavdenom
// (this is experimental, trying to accomodate for the different numeric
// characteristics of the layer types)
// 
// 238   9/24/12 3:00p Fseide
// AdaGrad adjustment now clipped to 10 x against the average of the
// respective parameter matrix/vector, only afterwards is it scaled to the
// user-specified target. This is to prevent clipping if the dynamics
// change.
// 
// 237   9/23/12 8:25p Fseide
// lifted initialization of AdaGrad parameters from rbm.h to main.cpp
// 
// 236   9/23/12 8:01p Fseide
// const correctness for modelupdateinfo parameters
// (backpropagationmodelupdate(), updatedeltas(), preflayer)
// 
// 235   9/23/12 5:35p Fseide
// AdaGrad now displays the actual avdenom for diagnostics, although we
// hand-fix it
// 
// 234   9/23/12 4:18p Fseide
// 
// 233   9/21/12 5:47p T-simonw
// added member variables method forwardpropHessianVectorProduct for
// forwarding Hessian vector product statistics
// method is not working yet
// 
// 232   9/21/12 6:29p Fseide
// reduced hand-tuned AdaGrad avdenom 5-fold, instead assuming user will
// increase learning rates 5-fold: this will lead to less clipping of the
// weights
// 
// 231   9/21/12 3:24p Fseide
// added nosoftmax mode, to speed up sequence training by bypassing the
// unnecessary expensive softmax() computation
// 
// 230   9/20/12 7:18p Fseide
// AdaGrad now uses a fixed 'avdenom' across all layers--experimental;
// fixed a bug in updatedeltas() pointed out by Gang --thanks
// 
// 229   9/20/12 4:38p Fseide
// AdaGrad now enforces to use the same av for the biases as for the W
// matrix (we basically ignore the bias in the avg)
// 
// 228   9/19/12 7:01p Fseide
// now no longer falling back to traditional gradient, which caused
// strange drops to 0 accuracy
// 
// 227   9/19/12 7:58a Fseide
// fixed the log msg that prints the adagrad denom count
// 
// 226   9/18/12 10:02a Fseide
// changed AdaGrad to use a forgetting factor (time constant 2 hours of
// data)
// 
// 225   9/17/12 3:30p Fseide
// more steps towards AdaGrad
// 
// 224   9/16/12 5:43p Fseide
// modelupdateinfo now has a constructor;
// further steps towards AdaGrad
// 
// 223   9/02/12 6:19p Fseide
// implemented I-smoothing
// 
// 222   9/02/12 5:58p Fseide
// bug fix in L2 regularization--it should now be the same as previous
// except for not affecting the momentum term
// 
// 221   9/02/12 5:47p Fseide
// regularizationtype::alpha renamed to more concise L2weightpersample
// 
// 220   9/02/12 5:31p Fseide
// implemented a new version of L2 regularization in addgradienttomodel() (which no
// longer changes the momentum accumulators)
// 
// 219   9/02/12 5:07p Fseide
// addweighted() now takes a 'thisscale' parameter, and addgradienttomodel() now
// also takes the modelupdateinfo parameters, in prep for L2
// regularization
// 
// 218   9/02/12 4:59p Fseide
// merged the two versions of updatedeltas() into one, which can now take
// a NULL pointer as bpinfo (now called modelupdateparameters)
// 
// 217   9/02/12 12:36a V-hansu
// comment out adagrad initialization
// 
// 216   8/31/12 9:34p F-gli
// changed adagrad code according to Frank's comments
// 
// 215   8/31/12 4:57p F-gli
// checked in temp code about adagrad
// 
// 214   8/31/12 15:00 Fseide
// added comments regarding AdaGrad implementation
// 
// 213   8/30/12 6:48p F-gli
// updated code for AdaGrad factor
// 
// 212   8/30/12 12:46a F-gli
// added tmp code for AdaGrad
// 
// 211   8/29/12 1:17p Fseide
// (added a comment)
// 
// 210   8/08/12 11:11 Fseide
// disabled momentum for deferupdate mode (its implementation was wrong,
// and for such huge blocks, it will have little impact anyway)
// 
// 209   8/07/12 18:14 Fseide
// completed implementation of deferupdate flag (still to be tested)
// 
// 208   8/07/12 17:46 Fseide
// new option to backpropagationmodelupdate(): deferupdate, used to
// implement batches of batches, for an MMI experiment
// 
// 207   8/07/12 5:04p V-hansu
// delete the ROUNDUPMODEL and INJECTTOPSECONDLAYER macro
// 
// 206   8/07/12 9:15 Fseide
// added Frank's weird sampling experiment
// 
// 205   7/23/12 10:54a V-hansu
// add macro INJECTTOPSECONDLAYER to do adaptation using second top layer
// 
// 204   7/19/12 11:46a V-hansu
// modify "roundup" related sentences
// 
// 203   7/06/12 9:17p V-hansu
// add numstream and numroundup in rbmbase so as to record the blowup
// information
// 
// 202   7/05/12 8:01p V-hansu
// chang the interface of blow up to let it able to return roundup unit
// 
// 201   7/02/12 4:26p V-hansu
// add function setlinearlayerweight to use GMM adaptation matrix to
// initialize
// 
// 200   6/30/12 2:29p V-hansu
// modify blowup function
// 
// 199   6/27/12 9:24p V-hansu
// add print(FILE *f) for debugging
// 
// 198   6/27/12 10:41a V-hansu
// modify dumplayer function for debugging
// 
// 197   6/26/12 3:01p V-hansu
// modify blowup function
// 
// 196   6/22/12 2:14p V-hansu
// complete the blowup function and did some change to previous interface
// of blowup
// 
// 195   6/05/12 4:21p V-hansu
// change the blowup function of class rbmbase, rbm and perceptron, not
// complete yet
// 
// 194   6/05/12 2:02p V-hansu
// add blowup function to several classes, not fully complete yet
// 
// 193   5/31/12 10:54p V-xieche
// fix a bug in exitcomputation function for more than 2 cuda devices on
// top layer
// 
// 192   5/13/12 11:00p V-xieche
// add initial code to make toplayer support more than 2 cuda devices in
// pipeline training. not finish yet
// 
// 191   5/09/12 4:25p F-gli
// 
// 190   5/09/12 4:24p F-gli
// 
// 189   4/18/12 4:01p V-xieche
// clean up all code related to target propagation and margin term.
// 
// 188   4/04/12 10:08p V-xieche
// add some commend and delete some debug code and old code won't use
// anymore.
// 
// 187   4/03/12 8:39p V-xieche
// check in all the code for pipeline training, stripe top layer and lies
// them on two cuda devices. need to add comments and adjust the code make
// it easy to read.
// 
// 186   3/27/12 1:14a V-xieche
// Add code for pipeline training with multi cuda devices
// 
// 185   3/11/12 7:05p V-xieche
// add code for a compact trainer. make it run in CUDA directly.
// 
// 184   3/08/12 10:34p V-xieche
// add code to make forward and backward prop do in CUDA directly.
// verified the training is correct, while speed faster than previous.
// need to debug it.
// 
// 183   3/06/12 10:51p V-xieche
// add code for compact trainer. Not finished.
// 
// 182   3/01/12 7:27p V-xieche
// add virtual function in dtnn class to make the code compilable for
// flatten sigmoid training.
// 
// 181   2/07/12 2:29p Dongyu
// fixed momentum invalid problem when learning rate is 0
// 
// 180   1/04/12 7:08p Fseide
// now handles momentum == 0
// 
// 179   11/29/11 5:20p F-gli
// implement peekweightmatrix() and peekbias() to Iannlayer derived class
// 
// 178   11/29/11 11:01a F-gli
// add peekweightmatrix() peekbias() forwardpropwithoutnonlinearity()
// implementation to Iannlayer derived class
// 
// 177   11/23/11 4:30p Dongyu
// refactorize rbmbase to support dtnn and other layer types. added
// Iannlayer as the major interface. May still need to update the
// definition of Iannlayer to fully support dtnn and other layer types.
// 
// 176   11/04/11 16:26 Fseide
// gradient scaling fixed
// 
// 175   11/04/11 14:46 Fseide
// (added a comment)
// 
// 174   11/04/11 14:22 Fseide
// refactored gradient weighting to allow for eliminating the gradient
// scaling by 1/(1-momentum)
// 
// 173   11/04/11 13:49 Fseide
// (editorial)
// 
// 172   11/03/11 15:18 Fseide
// momentum now passed down to network-update functions as momentum per
// sample, in prep for also taking the scaling out of the gradient
// 
// 171   10/28/11 14:51 Fseide
// formal change to use the new otherweight parameters in model update
// (but currently passing 1.0)
// 
// 170   10/28/11 13:36 Fseide
// changed 'momentum' to 'double' in prep of pushing in the scaling
// 
// 169   10/25/11 5:19p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 168   10/18/11 9:06p V-xieche
// modify the code to implement a true steeper or flat sigmoid function.
// i.e. scale the bias as well
// 
// 167   10/08/11 10:25 Fseide
// new special-purpose access methods peekweightmatrix() and peekbias()
// 
// 166   10/06/11 5:18p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 165   9/26/11 8:43p V-xieche
// Add some codes for log(sigmoid + epison) experiment.
// 
// 164   9/20/11 2:46p V-xieche
// fix a minor bug for steeper sigmoid experiment
// 
// 163   9/19/11 10:47p V-xieche
// Add two function to get and set weight matrix when computing for tmp
// experiment
// 
// 162   8/24/11 9:07p V-xieche
// remove a log infomation for adding margin term.
// 
// 161   8/23/11 7:57p V-xieche
// add margin-based training code for dbn according to Heigold's thesis.
// 
// 160   8/16/11 10:36p V-xieche
// add code for targetpropagation v4
// 
// 159   8/02/11 12:30a V-xieche
// add function to implement targetpropagation using b=w*h instead of h
// 
// 158   7/28/11 2:34p V-xieche
// add some indicatioin and comments modified by v-xieche
// 
// 157   7/26/11 1:07p V-xieche
// fix some TAB format
// 
// 156   7/25/11 10:17a V-xieche
// Put the setvalue() function into the #if #else block.
// 
// 155   7/23/11 5:13p V-xieche
// Add getweight and getbias function to get value from a specific
// location of a specific layer.
// 
// 154   7/20/11 4:02p V-xieche
// Add the dumplayer function and delete the unused variable layer in the
// creat function
// 
// 152   7/13/11 19:02 Fseide
// new method forwardpropwithoutnonlinearity() for a single target
// dimension, intended for use for a specific state index
// 
// 151   7/11/11 11:16a V-xieche
// Add the function for cheating experiment on hidden layer. Add a creat
// function to only create a layer
// 
// 150   7/08/11 11:07 Fseide
// documented the fact that dW, da, and db are SCALED versions of the
// low-pass filtered gradient, which is compensated for when calling
// addgradienttomodel()
// 
// 149   7/07/11 12:11 Fseide
// fixed the momentum bug in the refactoring that was the bug fix for
// 1-frame minibatches
// 
// 148   7/06/11 14:14 Fseide
// added comments and documented a potential bug in momentum handling
// 
// 147   7/06/11 14:03 Fseide
// linearnetwork::backpropagationmodelupdate() now just calls the base
// class and post-processes, to avoid code duplication
// 
// 146   7/06/11 13:59 Fseide
// further cleanup w.r.t. momentumfiltergain
// 
// 145   7/06/11 13:56 Fseide
// (some factoring w.r.t. momentumfiltergain)
// 
// 144   7/06/11 13:52 Fseide
// pushed weighting of learning rate through to rbm
// 
// 143   6/30/11 8:04a Fseide
// added a log message in linearnetwork constructor
// 
// 142   6/22/11 5:10p V-xieche
// put the setblockdiagonal function after addgradienttomodel to make sure the
// out-of-diag is 0.
// also setblockdiagonal function for a in the pooled situation.
// 
// 141   6/21/11 5:20p V-xieche
// just execute the setblockdiagonal functuion(previous comment it).
// 
// 140   6/20/11 10:19p V-xieche
// comment the setblockdiagonal just for temporary test purpose
// 
// 139   6/20/11 12:33p V-xieche
// No need to initial b. remove it
// 
// 138   6/20/11 7:51 Fseide
// changed backpropagationmodelupdate() to be a virtual function;
// added an override to backpropagationmodelupdate() to implement the
// block-diagonal structure
// 
// 137   6/20/11 7:25 Fseide
// factored network construction by type string out from dbn.h into a
// factory class in rbm.h where it belongs;
// moved linearnetwork::initial() out from linearnetwork to rbmbase next
// to initrandom() since it structurally seems to belong (it makes no
// assumption on linearnetwork) there although it is only used by
// linearnetwork
// 
// 136   6/19/11 3:42p V-xieche
// Initial b in the linear network also
// 
// 135   6/19/11 2:43p V-xieche
// Initial the linearnetwork, W to be a identity matrix and A to be a zero
// matrix.
// 
// 134   6/18/11 16:51 Fseide
// (renamed a variable)
// 
// 133   6/17/11 11:21 Fseide
// added class members and respective reading/writing code to
// lineartransform
// 
// 132   6/17/11 11:07 Fseide
// added comments and renamed a variable
// 
// 131   6/16/11 18:42 Fseide
// (comments)
// 
// 130   6/14/11 11:01 Fseide
// added new class 'linearnetwork'
// 
// 129   6/12/11 18:48 Fseide
// new method forwardpropwithoutnonlinearity() to support bottleneck
// features
// 
// 128   5/10/11 7:41a Fseide
// (refined logging of stats)
// 
// 127   5/09/11 15:23 Fseide
// temporarily made 'a' public for a hacked analysis tool
// 
// 126   4/11/11 3:18p Fseide
// (fixed a compiler warning)
// 
// 125   3/23/11 11:50a Fseide
// new method setweights()
// 
// 124   3/13/11 20:59 Fseide
// (a minor bug commented)
// 
// 123   3/05/11 8:30p Fseide
// printmatvaluedistribution() and checkmodel() now compute/print the
// overall number of non-null parameters in aggregate
// 
// 122   3/04/11 6:17a Dongyu
// added model weight distribution analysis and dumping functionality
// through the "checkmodel" switch
// 
// 121   3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 120   2/10/11 10:02a Fseide
// scaleandaddallcols() prototype was simplified;
// documented and partially fixed spelling error 'rmbmodelmatrix' --oops!
// 
// 119   2/08/11 5:33p Fseide
// bug fix in addgradienttomodel(): now no longer adds db in backprop mode (b is
// unused at this point anyway, but it was wrong nevertheless)
// 
// 118   2/08/11 4:23p Fseide
// moved three resizeonce() calls from updatedeltas() to inside their NUMA
// counterparts (they are not used in CUDA, so no need to allocate them)
// 
// 117   2/08/11 2:19p Fseide
// (an outdated comment deleted)
// 
// 116   2/07/11 4:29p Fseide
// removed a few checknan() that do not play well with the new
// architecture
// 
// 115   2/07/11 3:25p Fseide
// added typedefs for rbmstatevectorsrefread/writing
// 
// 114   2/05/11 8:23p Fseide
// fixed an incorrect assertion in updatedeltas()
// 
// 113   2/05/11 7:23p Fseide
// fixed an assertion in updatedeltas()
// 
// 112   2/05/11 7:00p Fseide
// moved mulbydsigm() and samplebinary() to rbmstatevectorsref
// 
// 111   2/02/11 11:22a Fseide
// moved sigmoid() and softmax() to rbmstatevectors;
// replace the now empty matrixbase class by a typedef
// 
// 110   2/02/11 10:48a Fseide
// switched all matrixbase & to rbmstatevectorsref & (not tested yet
// because underlying classes still cannot handle it)
// 
// 109   2/02/11 10:27a Fseide
// switched over from matrix/matrixbase to dummy implementations of
// rbmstatevectorsbase/rbmstatevectorsrefbase
// 
// 108   2/02/11 10:23a Fseide
// added typedef matrixstripe rbmstatevectorsref
// 
// 107   2/02/11 9:25a Fseide
// defined rbmstatevectors, but currently identical to 'matrix', need to
// solve the stripe problem first
// 
// 106   2/02/11 8:38a Fseide
// changed model parameter types from acceleratedmatrix to rbmmodelmatrix
// 
// 105   2/02/11 8:24a Fseide
// (added a comment)
// 
// 104   2/02/11 8:22a Fseide
// pushed some math ops on updatedeltas() down to acceleratedmatrix, for
// further CUDA optimization
// 
// 103   2/01/11 7:53p Fseide
// added performance comments
// 
// 102   2/01/11 6:44p Fseide
// (added a comment)
// 
// 101   2/01/11 4:53p Fseide
// added one more cache
// 
// 100   2/01/11 15:24 Fseide
// now gets cachedmatrix from inside acceleratedmatrix
// 
// 99    2/01/11 15:00 Fseide
// matprod_m*m() functions now take one additional cache object for moving
// data to/from CUDA
// 
// 98    2/01/11 14:57 Fseide
// stratified interface to acceleratedmatrix a little
// 
// 97    2/01/11 11:47a Fseide
// fixed entercomputation() protocol w.r.t. allocation of deltas and
// updatedeltas()
// 
// 96    1/30/11 16:37 Fseide
// (added a comment)
// 
// 95    1/30/11 16:33 Fseide
// (added a comment)
// 
// 94    1/30/11 16:33 Fseide
// acceleratedmatrixbase and cachedmatrixbase moved to parallelrbmmatrix.h
// 
// 93    1/30/11 16:28 Fseide
// changed acceleratedmatrix and cachedmatrix to class templates, so we
// can move them to a separate header
// 
// 92    1/30/11 15:56 Fseide
// further abstraction of cachedmatrix, ready to be reused for CUDA
// version
// 
// 91    1/28/11 17:13 Fseide
// commented the four key matrix functions in acceleratedmatrix, which are
// to be adapted to CUDA
// 
// 90    1/28/11 16:54 Fseide
// comments on what happens where
// 
// 89    1/28/11 16:38 Fseide
// changed acceleratedmatrix to derive from 'matrix' protected to make all
// calls into 'matrix' explicit;
// added call-through methods to acceleratedmatrix for all calls into
// 'matrix';
// parallelized matrix product moved inside acceleratedmatrix
// 
// 88    1/28/11 15:36 Fseide
// changed model parameters to acceleratedmatrix (first step)
// 
// 87    1/28/11 15:16 Fseide
// changed the various rbmXXX(W,a,b) constructors to take rvalue
// references
// 
// 86    1/28/11 14:43 Fseide
// further tidying-up, clean-up, moving-around, commenting as prep for
// CUDA transition
// 
// 85    1/28/11 11:41 Fseide
// new data type cachedmatrix as a first step to abstract out NUMA/CUDA
// stuff from rbmbase
// 
// 84    1/28/11 11:37 Fseide
// moved matrix-product functions out from rbmbase, in prep of CUDA
// version
// 
// 83    1/28/11 11:24 Fseide
// (removed some unused code)
// 
// 82    1/28/11 11:23 Fseide
// moved functions around in prep for modularization for CUDA
// 
// 81    1/28/11 11:11 Fseide
// removed residuals of ZMSIGM experiment
// 
// 80    1/28/11 11:09 Fseide
// removed copy construction and clone()
// 
// 79    1/28/11 10:54 Fseide
// removed cachedWt and all that depends on it (no longer needed, we clone
// and transpose on the fly)
// 
// 78    1/28/11 10:48 Fseide
// added enter/exitcomputation();
// deleted some old code related to old, frame-wise parallelization
// 
// 77    1/24/11 12:21p Fseide
// (added some #if-0'ed out debug code)
// 
// 76    1/19/11 16:34 Fseide
// (added comments to scaleandaddmatprod_numa() towards transition to a
// more standard GEMM call)
// 
// 75    1/19/11 10:05a Fseide
// added checks for matprod to check whether parallelized version is
// correct
// 
// 74    1/19/11 8:38a Fseide
// changed updatedeltas() from initializing da/db by move to initializing
// it by assignment (which requires prior allocation since this is the
// matrixbase type which cannot allocate)
// 
// 73    1/14/11 10:20p Fseide
// disabled the "speed-up" hacks, they don't seem to work just like they
// are now, something still wrong
// 
// 72    1/14/11 9:25p Fseide
// added the "optimizations" according to the "BP tricks" document (need
// to find the author and correct title!)
// 
// 71    1/14/11 6:03p Fseide
// removed a log message
// 
// 70    1/14/11 5:45p Fseide
// (cosmetic change to a log message)
// 
// 69    1/14/11 5:44p Fseide
// eliminated vt from updatedeltas() because we can now directly operate
// on the untransposed matrix
// 
// 68    1/14/11 5:36p Fseide
// scaleandaddmatprod_mtm_numa() renamed to scaleandaddmatprod_numa();
// it now implements Aistransposed flag
// 
// 67    1/14/11 5:01p Fseide
// (renamed a function--no longer needed in the future anyway)
// 
// 66    1/14/11 4:47p Fseide
// preparation for scaleandaddmatprod_numa() towards implicit
// transposition
// 
// 65    1/13/11 10:40a Fseide
// (added a diagnostics message to scaleandaddmatprod_numa())
// 
// 64    1/13/11 10:08a Fseide
// scaleandaddmatprod_numa() now parallelizes distribution of the
// input matrix
// 
// 63    1/12/11 12:30p Fseide
// towards more local parallelization of fprop/bprop
// 
// 62    1/12/11 10:48a Fseide
// some refactoring towards new parallelization of prop functions
// 
// 61    1/10/11 16:31 Fseide
// (added a comment)
// 
// 60    1/05/11 9:59p Fseide
// updatedeltas() now resizes again to account for the last block
// 
// 59    1/05/11 9:35p Fseide
// updatedeltas() now keeps its memory allocated (in a member variable)
// 
// 58    1/05/11 6:37p Fseide
// updatedeltas() now only copying portion of vt locally that is needed
// 
// 57    1/05/11 6:12p Fseide
// NUMA-optimized the model update--significant gain compared to
// non-optimized (NUMA-bad) version
// 
// 56    1/05/11 4:51p Fseide
// some tidying-up in prep for NUMA-parallelizing update function;
// backpropagationmodelupdate() no longer virtual (the same for all types)
// 
// 55    1/05/11 12:11p Fseide
// backpropagateprepare now operating striped for optimal NUMA performance
// 
// 54    1/05/11 8:37a Fseide
// copyfrom() no longer copies deltas, and only allocates cachedWt (not
// copying)
// 
// 53    1/04/11 10:20p Fseide
// changed needWt() to not doing transpose in parallel, due to new
// architecture
// 
// 52    1/04/11 9:45p Fseide
// new method copyfrom() to reclone without mem allocation (for NUMA)
// 
// 51    12/21/10 18:54 Fseide
// bug fix for top layer (which has no 'b')
// 
// 50    12/21/10 18:37 Fseide
// added experimental functionality to "split" a hidden layer by doubling
// its number of hidden nodes
// 
// 49    12/09/10 9:07p Fseide
// added several checknan() calls
// 
// 48    12/09/10 12:32 Fseide
// removed an assert() that was incorrect when running single-threaded
// 
// 47    12/08/10 3:24p Fseide
// added an overflow check (remove later)
// 
// 46    12/08/10 3:07p Fseide
// softmax() now using normalization
// 
// 45    12/06/10 15:23 Fseide
// removed 'negate' flag from updatedeltas() (was always 'false')
// 
// 44    11/30/10 1:11p Fseide
// initrandom() changed init val for a to 0 from -4 (later we probably
// want to distinguish bp and pt)
// 
// 43    11/30/10 11:22a Fseide
// switched to new implementation of backpropagationupdateshared() that
// shares code with pretraining
// 
// 42    11/30/10 9:12 Fseide
// pretrainingprepare() now separate from backpropagationprepare()
// (although doing the same)
// 
// 41    11/30/10 7:31a Fseide
// now using a little trick in pretrainingmodelupdate() for the negation
// 
// 40    11/30/10 7:01a Fseide
// (added a typecast for 64-bit correctness)
// 
// 39    11/29/10 17:00 Fseide
// several updates/fixes to updatedeltas()
// 
// 38    11/29/10 16:10 Fseide
// new virtual method type()
// 
// 37    11/29/10 15:42 Fseide
// added constructors to construct fresh RBMs from scratch, with random
// initialization
// 
// 36    11/29/10 15:06 Fseide
// (fixed a comment)
// 
// 35    11/29/10 13:17 Fseide
// added pretraining code, but untested so far
// 
// 34    11/26/10 17:12 Fseide
// (added a comment)
// 
// 33    11/26/10 16:30 Fseide
// (minor further change to the same function)
// 
// 32    11/26/10 16:15 Fseide
// (continued to refactor bp shared function, #if 0-ed out)
// 
// 31    11/26/10 16:06 Fseide
// (started to factor some bp update code for pretraining)
// 
// 30    11/25/10 17:04 Fseide
// using parallel matprod now for first (non-momentum) bp update
// 
// 29    11/25/10 15:07 Fseide
// backpropagateupdate() now implements momentum (to be tested)
// 
// 28    11/24/10 7:23 Fseide
// added functions for file I/O
// 
// 27    11/23/10 11:30a Fseide
// now using parallel_transpose() in update... no big difference
// 
// 26    11/23/10 11:22a Fseide
// rbmbase() copy constructor now copies cachedWt
// 
// 25    11/23/10 8:54 Fseide
// (added a comment)
// 
// 24    11/22/10 2:12p Fseide
// backpropagationstats() now calls mulbydsigm()
// 
// 23    11/22/10 13:36 Fseide
// removed a __forceinline as it tripped up the optimizer
// 
// 22    11/22/10 13:07 Fseide
// back prop update now operating in parallel --but slows down back-prop
// error??
// 
// 21    11/22/10 11:02a Fseide
// (added ability to switch back to single-threaded in
// parallel_transpose())
// 
// 20    11/22/10 10:50 Fseide
// backpropagationprepare() now calls parallel_transpose() (but it does
// not seem to help)
// 
// 19    11/19/10 19:11 Fseide
// (minor optimization)
// 
// 18    11/19/10 17:30 Fseide
// added a comment
// 
// 17    11/19/10 16:40 Fseide
// backpropagationupdateshared() changed to use the transpose() function
// 
// 16    11/19/10 16:07 Fseide
// added cachedWt for faster computation
// 
// 15    11/19/10 15:25 Fseide
// (added a comment)
// 
// 14    11/19/10 15:22 Fseide
// basic back-propagation training seems now complete (without momentum)
// 
// 13    11/19/10 12:48 Fseide
// fixed bug in backpropagationupdateshared()
// 
// 12    11/19/10 12:33 Fseide
// (documented a bug, not fixed yet)
// 
// 11    11/19/10 10:56 Fseide
// redesigned interface to back-propagation to avoid locks
// 
// 10    11/19/10 7:43 Fseide
// renamed 'toprbm' to 'perceptron' which is more yet not fully accurate
// 
// 9     11/19/10 7:28 Fseide
// (minor refactoring)
// 
// 8     11/19/10 7:18 Fseide
// (added 2 comments)
// 
// 7     11/18/10 17:00 Fseide
// back-propagation implemented (not tested)
// 
// 6     11/17/10 14:44 Fseide
// cleanup of backpropagationstats()
// 
// 5     11/17/10 13:02 Fseide
// implemented backwardprob();
// new method moveaccumulator() for parallelized training
// 
// 4     11/17/10 12:46 Fseide
// implemented explicit rbmbase copying constructor to allow for the
// non-assignable CCritSec object
// 
// 3     11/17/10 12:29 Fseide
// steps towards training (back-propagation for now)
// 
// 2     11/15/10 18:40 Fseide
// added the ability to clone a model, for use in NUMA-local computation
// 
// 1     11/12/10 11:38 Fseide
// RBM and DBN factored into separate header files

#if 0               //add by Hang Su to set aside code and conments
#endif
#pragma once


#include "ssematrix.h"          // for basic matrix type
#include "parallelrbmmatrix.h"  // for parallel accelerated matrix operations (NUMA, CUDA)
#include <string>
#include <stdexcept>
#include <stdlib.h>
#include <map>

namespace msra { namespace dbn {

// ===========================================================================
// matrix, vector types for use in the networks
// ===========================================================================

// model matrices that live in CUDA during computation
typedef rbmmodelmatrixbase<matrixbase> rbmmodelmatrix;
// TODO: fix the spelling error rmb->rbm --oops
typedef rbmmodelmatrixbase<matrixbase> rmbmodelmatrix;  // spelling error!
typedef rbmmodelmatrix rmbmodelvector;                  // spelling error!
typedef rbmmodelmatrix::cachedmatrix cachedmatrix;

// network state (input and activations) that lives in CUDA
// ... This is not implemented yet.
typedef rbmstatevectorsbase<matrixbase> rbmstatevectors;
typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;
typedef rbmstatevectors::lockforreading rbmstatevectorsrefreading;
typedef rbmstatevectors::lockforwriting rbmstatevectorsrefwriting;

enum regularizationtype
{
    regNone,        // no regularization
    regL2,          // + ||W-W_ref||_2^2
    regIsmoothing,  // I-smoothing: add ref model with a weight
    regKL,          // KL between ref posterior distributions and new model's posterior distributions
    regCL,          // change large weights only
    regCS,           // change small weights only
    regL2C          // maximum allowed column norm ||W_:j||_2^2
};

class Iannlayer;


// ===========================================================================
// modelupdateinfo -- parameters for SGD in its various flavors
// TODO: 'info' is not a good name; are these 'params'?
// ===========================================================================

struct modelupdateinfo
{
    // sparse weights
    float sparsethreshold;
    // regularization
    regularizationtype regtype;     // type of regularization
    float nochangeifaboveorbelow;   // for regCL and regCS
    const Iannlayer * preflayer;    // layer of reference model that corresponds to currently processed layer
    float L2weightpersample;        // contribution of L2 term per sample
    float L2maxcolnorm;
    // TODO: 'modelupdateinfo' is per layer, but options below are not per layer -> use a shared underlying structure for non-layer dependent settings; also put momentum there; LR could be layer-dependent
    // AdaGrad
    bool enableadagrad;             // enable AdaGrad code
    enum { onpartialsubgradient, onsubgradient, onrawgradient, onsmoothedgradient } adagradwhere; // apply AdaGrad to smoothed gradient rather than the raw one
    float adagradavdenom;           // manually set weight, e.g. 0.04
    size_t adagradT;                // time constant for AdaGrad computation (1st-order filter)
    class adagradstate_t * adagradstate;
    // MPI support
    mpiaggregator * mpiaggregator;  // or NULL
    size_t mpimasize;               // target size for model averaging; do local updates until this is #frames reached (0=disable)
    bool distributefixedcost;       // if true then do some fixed-cost operations on the aggregating node (stripe dimension) instead of afterwards (full dimension)
    // default constructor resets all to 'harmless'
    modelupdateinfo()
    {
        regtype = regNone;
        sparsethreshold = 0.0f;
        nochangeifaboveorbelow = 0.0f;
        preflayer = nullptr;
        L2weightpersample = 0.0f;
        L2maxcolnorm = 0.0f;
        enableadagrad = false;
        adagradwhere = onrawgradient;   // 'onsmoothedgradient' corresponds to the old code version, which works worst
        adagradavdenom = 0.0f;
        adagradT = 0;
        adagradstate = nullptr;
        mpiaggregator = nullptr;
        mpimasize = 0;
        distributefixedcost = false;
    }
};

// model-level state info for AdaGrad (that is, state that is shared across layers)
// The goal is to have an overall aggregated average denominator.
// The problem is that AdaGrad is so engrained in the current code logic that we cannot compute it separately but only as we go through layers.
// Workaround:
//  - early iterations use per-layer avdenom
//  - later iterations accumulate into here; once an average is accumulated, it will be used instead of per-layer avdenom
class adagradstate_t
{
    float totalavdenom;     // the averaged numerator, once we have one
    double avdenomsum;      // accumulator for averaging
    double avdenomcount;
public:
    adagradstate_t()
    {
        totalavdenom = 0.0f;
        initaccumulation();
    }
    // get the total av denom of the last round; return 'false' if none yet
    float gettotalavdenom() const
    {
        if (totalavdenom == 0.0f)    // nothing there yet
            throw std::logic_error ("gettotalavdenom: was called before anything was accumulated");
        return totalavdenom;
    }
private:
    // call this before going through layers
    void initaccumulation()
    {
        avdenomsum = 0.0;
        avdenomcount = 0.0;
    }
public:
    // accumulate an average
    void accumulate (double avdenom, size_t rows, size_t cols, size_t mbframes)
    {
        const double weight = rows * cols * (double) mbframes;  // average gets weighted by matrix size and frames
        avdenomsum += avdenom * weight;
        avdenomcount += weight;
    }
    // call this after going through layers
    void finishaccumulation()   // TODO: rename this
    {
        if (avdenomcount)                                       // we call this not at the start, so the count will still be 0
            totalavdenom = (float) (avdenomsum / avdenomcount); // once this is non-0, we will start returning it in gettotalavdenom()
        fprintf (stderr, "finishaccumulation: %.8f / %.8f -> %.8f x 1e-6\n", avdenomsum, avdenomcount, totalavdenom * 1e6);
        // get ready for next
        initaccumulation();
    }
};


// ===========================================================================
// parameters
// TODO: This could be a more generic library in a separate header.
// ===========================================================================

// value of one configuration parameter
class layerconfigval : std::string
{
public:
    layerconfigval (const std::string & val) : std::string (val) { }
    // it auto-casts to the common types
    // Note: This is meant to read out a parameter once to assign it, instead of over again.
    operator std::string () const { return *this; } // TODO: does not seem to work
    operator const char * () const { return c_str(); }
    operator double () const
    {
        char * ep;          // will be set to point to first character that failed parsing
        double value = strtod (c_str(), &ep);
        if (empty() || *ep != 0)
            throw std::runtime_error ("layerconfigval (double): invalid input string");
        return value;
    }
    operator float () const { return (float) (double) *this; }
    operator int () const
    {
        double val = (double) *this;
        int ival = (int) val;
        if (val != ival)
            throw runtime_error ("layerconfigval (int): integer argument expected");
        return ival;
    }
    operator size_t () const
    {
        int ival = (int) *this; // note: we don't really support the full size_t range; fix this if you care
        if (ival < 0)
            throw runtime_error ("layerconfigval (size_t): non-negative integer argument expected");
        return (size_t) ival;
    }
    operator bool () const
    {
        const auto & us = *this;
        if (us == "t" || us == "true" || us == "T" || us == "True" || us == "TRUE" || us == "1")
            return true;
        if (us == "f" || us == "false" || us == "F" || us == "False" || us == "FALSE" || us == "0" || us == "")
            return false;
        throw runtime_error ("layerconfigval (bool): boolean argument expected");
        // TODO: do we want to allow accept non-empty strings and non-0 numerical values as 'true'?
    }
};

// dictionary of parameters
class layerconfigparameters : std::map<std::string, layerconfigval>
{
    std::string configname;       // name of this configuration, e.g. for error messages
public:
    layerconfigparameters (const std::string & configname) : configname (configname) { }
    void insert (const std::string & name, const std::string & val)
    {
        auto res = std::map<std::string, layerconfigval>::insert (std::make_pair (name, val));
        if (!res.second)    // no insertion was made
            throw std::runtime_error ("layerconfigparameters: duplicate parameter definition for " + configname + ":" + name);
    }
    // dict(name,default): read out a value; if not given, use provided default value
    template<typename VALTYPE>
    const VALTYPE & operator() (const std::string & name, const VALTYPE & defaultvalue) const
    {
        auto iter = find (name);
        if (iter != end())
            return iter->second;
        else
            return defaultvalue;
    }
    // dict(name): read out a mandatory parameter value; if not given, use provided default value
    const layerconfigval & operator() (const std::string & name) const
    {
        auto iter = find (name);
        if (iter != end())
            return iter->second;
        else
            throw std::runtime_error ("layerconfigparameters: required parameter missing: " + configname + ":" + name);
    }
    // dump for debugging purposes
    void dump() const
    {
        for (auto iter = begin(); iter != end(); iter++)
            fprintf (stderr, "layerconfigparameters: %s:%s=%s\n", configname.c_str(), iter->first.c_str(), (const char *) iter->second);
    }
};

// ===========================================================================
// Iannlayer -- interface of all network layers
// ===========================================================================
class Iannlayer
{
private:
    Iannlayer (const Iannlayer &);
    void operator= (const Iannlayer &);

protected:
    Iannlayer() { }

public:
    virtual ~Iannlayer() { }
    virtual string type() const = 0;

    virtual void print() const = 0;
    virtual void print (FILE * f) const = 0;
    virtual void write (FILE * f) const = 0;
    virtual void write (HANDLE f) const = 0;
    virtual void dumplayer() const { }
    virtual pair<unsigned int,unsigned int> printvaluedistribution (const string & tag) const { return make_pair(0,0); }
    
    virtual const matrix & peekweightmatrix() const = 0;
    virtual const vector & peekbias() const = 0;

    virtual size_t vdim() const = 0;
    virtual size_t hdim() const = 0;  // in dtnn case this returns the overall hidden layer size (h1*h2)
    virtual std::vector<size_t> hdims() const { return std::vector<size_t> (1, hdim()); }  // return each individual hidden layer size (for dtnn)

    // copy all configuration and weights (realloc if needed)
    // Note: Do not forget to override this in downstream classes!
    virtual void copyfrom (const Iannlayer & other) = 0;
    virtual void mpiredistribute (const modelupdateinfo &) { throw std::logic_error ("mpiredistribute() not implemented for this type of layer"); }

    virtual void entercomputation (int type) = 0;
#ifdef MULTICUDA
    virtual void entercomputation (int type, size_t deviceid) = 0;
    virtual void entercomputation (int type, size_t deviceid, bool stripedmode, size_t devnum) = 0;
    virtual void exitcomputation (size_t deviceid) = 0;
    virtual void exitcomputation (size_t deviceid, bool stripedmode, size_t topdevicenum) = 0;
    virtual float getweightvalue(size_t c, size_t l) const = 0;
#endif
    virtual void exitcomputation() = 0;

    virtual void entermpiaggregation (std::vector<size_t> & mpistripebuffersizes, size_t bits) { throw std::logic_error ("entermpiaggregation() not implemented for this type of layer"); }
    virtual void exitmpiaggregation() { throw std::logic_error ("exitmpiaggregation() not implemented for this type of layer"); }

    virtual void doublenodes (bool out) { throw std::logic_error ("doublenodes() not implemented for this type of layer"); }

    virtual void validatedims() const = 0;
    //virtual void initrandom (unsigned int randomseed) = 0;
    //mask is added at the activation (i.e., before sigmoid and thus a large negative values means masked and 0 means not)
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & h, const bool linearonly = false, rbmstatevectorsref * pmask=NULL) const = 0;
    virtual void dropoutscaleweights (float factor) = 0;

    // methods required for Hessian free optimization
    // see the analogous methods in dbn.h for documentation
    // implementations can be found in rbmbase
    // for convenience, dummy methods are implemented here

    virtual void forwardprophessianvectorproduct(const rbmstatevectorsref &layerin, const rbmstatevectorsref &layerout, const rbmstatevectorsref &forwardstatisticsin, rbmstatevectorsref &forwardstatisticsout, bool zeroforwardstatisticsin) const
    { throw std::logic_error ("forwarding hessian vector product() not implemented for this type of layer"); }
    virtual void initcgfromzero (bool usepreconditioning, float nobservations, float lambda, float alpha) { throw std::logic_error ("initcgfromzero() not implemented for this type of layer"); }
    virtual void initcg (bool usepreconditioning, float nobservations, float lambda, float alpha) { throw std::logic_error ("initcg() not implemented for this type of layer"); }
    virtual void inithessianfree (size_t nofbacktrackingmodels) { throw std::logic_error ("inithessianfree() not implemented for this type of layer"); }
    virtual void scalecgiterate (float scalingfactor) { throw std::logic_error ("scalecgiterate() not implemented for this type of layer"); }
    virtual void setcgsearchdirection (rbmmodelmatrix &W, rbmmodelmatrix &a)  { throw std::logic_error ("setcgsearchdirection() not implemented for this type of layer"); }
    virtual float calculatecgresidualnorm (bool weighted) const  { throw std::logic_error ("calculatecgresidualnorm() not implemented for this type of layer"); }
    virtual float calculatepcgresidualnorm() const { throw std::logic_error ("calculatepcgresidualnorm() not implemented for this type of layer"); }
    virtual float calculatesquaredparameternorm (bool weighted) const  { throw std::logic_error ("calculatesquaredparameternorm() not implemented for this type of layer"); }
    virtual float calculatesquaredcgiteratenorm (bool weighted) const { throw std::logic_error ("calculatesquaredcgiteratenorm() not implemented for this type of layer"); }
    virtual float calculatesquaredcgsearchdirectionnorm (bool weighted) const { throw std::logic_error ("calculatesquaredcgsearchdirectionnorm() not implemented for this type of layer"); }
    virtual float calculatecgcurvatureproduct() const  { throw std::logic_error ("calculatecurvatureproduct() not implemented for this type of layer"); }
    virtual void updatecgiterate (float stepsize) { throw std::logic_error ("updatecgiterate() not implemented for this type of layer"); }
    virtual void updatecgresidual (float stepsize) { throw std::logic_error ("updatecgresidual() not implemented for this type of layer"); }
    virtual void solveforpcgresidual() { throw std::logic_error ("solveforpcgresidual() not implemented for this type of layer"); }
    virtual void updatecgsearchdirection (float stepsize) { throw std::logic_error ("updatecgsearchdirection() not implemented for this type of layer"); }
    virtual void updatepcgsearchdirection (float stepsize) { throw std::logic_error ("updatepcgsearchdirection() not implemented for this type of layer"); }
    virtual void setdummyhessianvectorproduct (float weight) { throw std::logic_error ("dummy hv product() not implemented for this type of layer"); }
    virtual void setdummygradient() { throw std::logic_error ("dummy gradient() not implemented for this type of layer"); }
    virtual float calculategradientcgiterateproduct (bool weighted) const { throw std::logic_error ("calculategradientcgiterateproduct gradient() not implemented for this type of layer"); }
    virtual float calculatecgresidualcgsearchdirectionproduct (bool weighted) const { throw std::logic_error ("calculatecgresidualcgsearchdirectionproduct gradient() not implemented for this type of layer"); }
    virtual void normalizegradient (size_t nobservations) { throw std::logic_error ("normalizegradient() not implemented for this type of layer"); }
    virtual void allocateaccumulators (bool usecgpreconditioning) { throw std::logic_error ("allocateaccumulator() not implemented for this type of layer"); }
    virtual void settoaccumulator (bool usecgpreconditioning) { throw std::logic_error ("settoaccumulator() not implemented for this type of layer"); }
    virtual float calculatesquaredgradientnorm (bool weighted) const { throw std::logic_error ("calculatesquaredgradientnorm() not implemented for this type of layer"); }
    virtual void adddampingterm (float lambda) { throw std::logic_error ("adddampingterm() not implemented for this type of layer"); }
    virtual void storecgiterate (size_t position) { throw std::logic_error ("storecgiterate() not implemented for this type of layer"); }
    virtual void settointermediateresult (size_t position, float stepsize) { throw std::logic_error ("settointermediateresult() not implemented for this type of layer"); }
    virtual void backupmodel() { throw std::logic_error ("backupmodel() not implemented for this type of layer"); }
    virtual void restoremodel() { throw std::logic_error ("restoremodel() not implemented for this type of layer"); }
    virtual void finalizecg (int cgiter, float cginitdecayingfactor) { throw std::logic_error ("finalizecg() not implemented for this type of layer"); }
    virtual float calculatecgresidualcgiterateproduct (bool weighted) const  { throw std::logic_error ("calculatecgresidualcgiterateproduct() not implemented for this type of layer"); }
    virtual float calculatecgiteratecgsearchdirectionproduct (bool weighted) const { throw std::logic_error ("calculatecgiteratecgsearchdirectionproduct() not implemented for this type of layer"); }
    virtual void collectgradient (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, bool isfirstbatch, bool usedoubleaccumulator)  { throw std::logic_error ("collecting gradient() not implemented for this type of layer"); }
    virtual void collectsquaredgradient (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, rbmstatevectorsref & vsquared, rbmstatevectorsref & ehxssquared, bool isfirstbatch, bool usedoubleaccumulator) { throw std::logic_error ("collecting squared gradient() not implemented for this type of layer"); }
    virtual void collecthessianvectorproduct (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, bool isfirstbatch, size_t nsecondorderframes)  { throw std::logic_error ("collecting hessian vector product() not implemented for this type of layer"); }

    virtual size_t svd (std::vector<std::vector<float>>&,float) { throw std::logic_error ("svd() not implemented for this type of layer"); }

#ifdef COMPACTTRAINER  // for compact trianer. [v-xieche]
    virtual msra::cuda::matrix & getcudaweight (size_t deviceid) = 0;
#ifdef STRIPEDTOPLAYER
    virtual msra::cuda::matrix  & stripedgetcudaweight (size_t devid, size_t devnum, msra::dbn::cudadistributedmatrix::cudastriping_t s) = 0;
    virtual msra::cuda::matrix &stripedgetcudabias (size_t deviceid,size_t devnum, msra::dbn::cudadistributedmatrix::cudastriping_t s) = 0;
    virtual void backpropagationmodelupdatestripedincuda (const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, modelupdateinfo & bpinfo, size_t deviceid, size_t devnumusedintoplayer = 2) = 0;
#endif
    virtual msra::cuda::matrix & getcudabias (size_t deviceid) = 0;
    
    virtual void backpropagationmodelupdateincuda (const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                                   float learningratepersample, double momentumpersample, bool resetmomentum, modelupdateinfo & bpinfo, size_t deviceid) = 0;
    virtual rmbmodelmatrix & getweight () = 0;
    virtual rmbmodelmatrix & getbias () = 0;
#endif

    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const = 0;
    virtual void backpropagationmodelupdate1 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v, const modelupdateinfo & bpinfo) = 0;
    virtual void backpropagationmodelupdate2 (const modelupdateinfo & bpinfo, bool mpimaisfirst, bool mpimaislast, float learningratepersample, double momentumpersample) = 0;
    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & ehxs_legacy,  const rbmstatevectorsref & v_legacy,
                                             float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo) = 0;

    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const { throw std::logic_error ("forwardpropdelta() not implemented for this type of layer"); }
    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const { throw std::logic_error ("forwardpropwithoutnonlinearity() not implemented for this type of layer"); }

    virtual shared_ptr<char> allocatetransferbuffer (size_t stripe, size_t size) { throw std::logic_error ("allocatetransferbuffer() not implemented for this type of layer"); }
    virtual void quantizeandfetchsubbatchstripe (size_t stripe, char * bufferbegin, size_t buffersize, size_t & submbframes) { throw std::logic_error ("quantizeandfetchsubbatchstripe() not implemented for this type of layer"); }
    virtual void syncfetchsubbatchstripe (size_t stripe) { throw std::logic_error ("syncfetchsubbatchstripe() not implemented for this type of layer"); }
    virtual void unquantizeandaggregatestripe (size_t ourstripe, size_t kfrom, const char * bufferbegin, size_t buffersize, bool isfirst, bool islast, size_t mbframes, const modelupdateinfo &, double momentumpersample, float learningratepersample) { throw std::logic_error ("unquantizeandaggregatestripe() not implemented for this type of layer"); }
    virtual void quantizeandassignaggregatedstripe (size_t ourstripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled) { throw std::logic_error ("quantizeandassignaggregatedstripe() not implemented for this type of layer"); }
    virtual void assignaggregatedstripe (size_t stripe, const char * bufferbegin, size_t buffersize) { throw std::logic_error ("assignaggregatedstripe() not implemented for this type of layer"); }
    virtual void syncassignaggregatedstripeandunquantize (size_t stripe, const char * bufferbegin, size_t buffersize, size_t aggmbframes, const modelupdateinfo &) { throw std::logic_error ("syncassignaggregatedstripeandunquantize() not implemented for this type of layer"); }
    virtual void mpiallreducegradient (const modelupdateinfo & bpinfo) { throw std::logic_error ("mpiallreducegradient() not implemented for this type of layer"); }

    // v1 and h1 below are CD 1 (only useful for RBM) on minibatch
    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const { throw std::logic_error ("pretrainingstats() not implemented for this type of layer"); }
    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningrate, double momentumpersample) { throw std::logic_error ("pretrainingmodelupdate() not implemented for this type of layer"); }

    virtual size_t getnumberofweightsets() const { throw std::logic_error ("getnumberofweightsets() not implemented for this type of layer"); }
    virtual pair<size_t, size_t> getweightsetdims (const size_t weightsetindex) { throw std::logic_error ("getweightsetdims() not implemented for this type of layer"); }

    virtual void blowup (const size_t blowupfactor) { throw std::logic_error ("blowup() not implemented for this type of layer"); }
    virtual void blowup (const size_t blowupfactor, const std::vector<size_t> & statemap) { throw std::logic_error ("blowup() not implemented for this type of layer"); }
    virtual void setlinearlayerweight (const matrix & adaptmatrix) { throw std::logic_error ("setlinearlayerweight() not implemented for this type of layer"); }

    virtual void setdeltas (const rmbmodelmatrix& otherdW, const rmbmodelmatrix& otherda) { };
    // TODO: we cannot make template funciton a virtual function. but we do want to have it virtual. do we really need template?
    template<class WTYPE, class ATYPE> void setweights (const WTYPE & newW, const ATYPE & newa, const size_t weightsetindex) { throw std::logic_error ("setweights() not implemented for this type of layer"); }
#ifdef STEEPERSIGMOID
    // function for output debug info and flatten sigmoid function. [v-xieche]
    virtual void multiplywith (float n) = 0;
    virtual float getweightvalue(size_t c, size_t l) const = 0;
    virtual float getbiasvalue(size_t m) const = 0;
#endif
};

// ===========================================================================
// rbmbase -- base class for all networks used here
// TODO: rename this. It is no longer limited to RBMs. layerbase? annlayer?
// ===========================================================================
enum nonlinearitykind_t     // we support multiple non-linearities
{
    linearkind,
    sigmoidkind,
    relukind,
    softmaxkind,
    leakyrootkind,          // hack
    softpluskind
};

static inline const char * nonlinearitykindtostring (nonlinearitykind_t nonlinearitykind)
{
    if (nonlinearitykind == linearkind)
        return "linearkind";
    else if (nonlinearitykind == sigmoidkind)
        return "sigmoidkind";
    else if (nonlinearitykind == relukind)
        return "relukind";
    else if (nonlinearitykind == softmaxkind)
        return "softmaxkind";
    else if (nonlinearitykind == leakyrootkind)
        return "leakyrootkind";
    else if (nonlinearitykind == softpluskind)
        return "softpluskind";
    else
        throw std::logic_error("nonlinearitykindstring: invalid nonlinearitykind value");
}

// abstract base class to allow unified operations on the network
class rbmbase : public Iannlayer
{
    rbmbase (const rbmbase &);
    void operator= (const rbmbase &);
protected:
    // model: E(v,h) = v'Wh + v'b + a'h ; p(v,h) = exp -E(v,h)
    // Note: Hinton's publications are not consistent in the use of a and b (which is which).
    rmbmodelmatrix W;   // for v'Wh
public: // make public as a hack for analysis
    rmbmodelvector a;   // for a'h
protected:
    rmbmodelvector b;   // for v'b  --this is unused by top level; we keep it here to keep everything together
    nonlinearitykind_t nonlinearitykind;  // which non-linearity?

    // ... I think the following are for multi-class fDLR, which never went anywhere
    size_t vdimnumroundup;  // for adaptation roundup  by Hang Su adaptation
    size_t hdimnumroundup;
    size_t numstream;       // for recording num of class  --TODO: what's this? Still needed? TODO: correct spelling

    // raw gradient of one minibatch
    // In case of deferred update, we accumulate into this.
    // Note: If 'raw_dmbframes' == 0 then the raw gradient content must be pretended to be 0 (we do not explicitly setzero() to save time)
    rmbmodelmatrix raw_dW;  // raw gradients before accumulated to momentum sum
    rmbmodelvector raw_da;
    rmbmodelvector raw_db;
    size_t raw_dmbframes;   // number of frames currently accumulated in the raw_ gradient, gets reset after it is consumed

    // MVN-SGD [Wiesler, 2014]
    rmbmodelvector mean;
    rmbmodelvector var;     // actually the diagonal of a matrix
    mutable rmbmodelvector vtemp, ehtemp;       // TODO: vtemp currently unused
    rmbmodelvector meanacc;
    rmbmodelvector varacc;
    size_t numacc;          // TODO: rename to mvnframes (and others likewise)

    // for faking double-buffering (a very special experiment):
    rmbmodelmatrix raw2_dW;
    rmbmodelvector raw2_da;
    size_t raw2_dmbframes;
    size_t rawframesseen;

    // for local-loop accumulation in data parallelism
    rmbmodelmatrix local_W;
    rmbmodelvector local_a;
    size_t local_mbframes;

    // momentum-smoothed gradient
    // TODO: remove scaling by momentum; instead scale by learning rate --TODO: is this still the case?? Otherwise remove this comment
    rmbmodelmatrix dW;  // derivative at last call to -update(); used for momentum
    rmbmodelvector da;  // Note that these deltas are scaled by 1/(1-momentum), for code simplicity.
    rmbmodelvector db;  // This must be corrected when adding these to the model parameters (multiply by (1-momentum)).

    // double precision accumulators        --TODO: what are these used for?? comment it
    rbmmatrixaccumulator accdW;
    rbmmatrixaccumulator accda;

    // Hessian free optimization statistics
    rmbmodelmatrix hessianvectorproductW;  // hessian vector product - matrix part
    rmbmodelvector hessianvectorproducta;  // hessian vector product - bias part
    // TODO avoid using pointers
    std::vector<rbmmodelmatrix*> cgintermediateresultsW;
    std::vector<rbmmodelmatrix*> cgintermediateresultsa;
    rbmmodelmatrix backupmodelW;
    rbmmodelmatrix backupmodela;
    // cg statististcs needed for hessian free optimization
    rbmmodelmatrix cgiterateW;
    rbmmodelmatrix cgiteratea;
    rbmmodelmatrix cgresidualW;
    rbmmodelmatrix cgresiduala;
    rbmmodelmatrix cgsearchdirectionW;
    rbmmodelmatrix cgsearchdirectiona;
     
    // preconditioned  cg statistics
    rbmmodelmatrix pcgresidualW;
    rbmmodelmatrix pcgresiduala;
    rbmmodelmatrix cgdiagonalpreconditionerW;
    rbmmodelmatrix cgdiagonalpreconditionera;
    rbmmodelmatrix dWsquared;
    rbmmodelmatrix dasquared;
    // double precision accumulators
    rbmmatrixaccumulator accdWsquared;
    rbmmatrixaccumulator accdasquared;

    // columnwise L2 norms for W
    rbmmodelmatrix norms;
    size_t normsmbcounter;      // maxnorm: we don't always normalize to save time

    static void malformed (std::string msg) { throw std::runtime_error ("rbmbase: invalid model file: " + msg); }
    void validatedims() const   // check if dimensions match
    {
        if (W.cols() != a.rows())
            malformed ("invalid model file--W matrix dimensions mismatch bias dimensions");
        if (b.rows() != 0 && b.rows() != W.rows())
            malformed ("invalid model file--W matrix dimensions mismatch rbm bias dimensions");
    }

    // -----------------------------------------------------------------------
    // constructor / destructor
    // -----------------------------------------------------------------------

    rbmbase() : nonlinearitykind (sigmoidkind) { }

    virtual ~rbmbase()
    {
        // HF clean-up; TODO: wrap these pointers
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            delete cgintermediateresultsW[i];   // TODO: wrap these pointers
            delete cgintermediateresultsa[i];
        }
    }

    // -----------------------------------------------------------------------
    // general accessors
    // -----------------------------------------------------------------------

    // get the dimensions
    size_t vdim() const { return W.rows(); }
    size_t hdim() const { return W.cols(); }

    // self-identification of the type of the model; used for saving
    virtual string type() const = 0;

    // -----------------------------------------------------------------------
    // CUDA support
    // -----------------------------------------------------------------------

    // do necessary preparations to start any computation with the model
    // 'type'can be:
    //  -2 -> Hessian free optimizer
    //  -1 -> backpropagation
    //  +1 -> pretraining
    //   0 -> evaluation
    // With CUDA, this loads the model into the CUDA RAM.
    // For the training modes, this also initializes accumulators to 0.
    // Note: We don't seem to distinguish between -1 and +1 at present.
    void entercomputation (int type)
    {
#if 1   // HACK
        // MVN-SGD
        // This is currently enabled by setting enablemvn to 1 or 2 during model loading; if 0 but loaded models have mean/var, then it remains enabled.
        // TODO: this must be done elsewhere, but where? Have a function enablemvn() that creates all these?
        int enablemvn = 0;
        if (enablemvn && mean.empty())                  // lazily create mean/var
        {
            mean.resize (W.rows(), 1);
            if (enablemvn < 2)                          // this disables variance normalization
                var.resize  (0, 1);
            else
                var.resize  (mean.rows(), 1);
            fprintf (stderr, "rbmbase: enabling %s normalization\n", var.empty() ? "mean" : "mean/var"), fflush (stderr);
            foreach_coord (i, j, mean) mean(i,j) = 0.0f;
            foreach_coord (i, j, var)  var(i,j)  = 1.0f;
        }
        if (enablemvn || !mean.empty())
        {
            mean.glimpse ("mean", false);
            var.glimpse ("var", false);
            fprintf (stderr, "rbmbase: accumulators for %s normalization present\n", var.empty() ? "mean" : "mean/var"), fflush (stderr);
        }
#endif
        W.entercomputation(); a.entercomputation(); b.entercomputation();
        // lazily allocate the gradient matrices
        if (type != 0)
        {
            raw_dW.resize (W.rows(), W.cols()); dW.resize (W.rows(), W.cols());
            raw_da.resize (a.rows(), a.cols()); da.resize (a.rows(), a.cols()); // (a.cols()==1, it's a vector)
            if (!b.empty() && type > 0)
            {
                raw_db.resize (b.rows(), b.cols());    
                db.resize (b.rows(), b.cols());
            }
#if 0       // enable this to get double-buffering in backpropagationmodelupdate2()
            raw2_dW.resize (W.rows(), W.cols());
            raw2_da.resize (a.rows(), a.cols());
#endif
            // probably there is much better way to allocate this container when necessary
            // at the same time making it avaliable for any layer type inheriting from rbmbase
            // I am no quite sure though so doing this way regardless it is really used or not
            norms.resize (W.cols(), 1); 

            // MVN-SGD
            // lazily allocate the gradient matrices
            vtemp.resize   (vdim(), 1);
            ehtemp.resize  (hdim(), 1);
            meanacc.resize (mean.rows(), mean.cols());  // (if not enabled, then these are all empty)
            varacc.resize  (var.rows(),  var.cols());
        }
        if (type == -2)
        {
            enterhessianfreeresize();
            enterhessianfreesync();
            enterhessianfreeinit();
        }
        raw_dW.entercomputation(); raw_da.entercomputation(); raw_db.entercomputation();
        raw2_dW.entercomputation(); raw2_da.entercomputation();
        if (!local_W.empty())
        {
            local_W.entercomputation(); local_a.entercomputation();
        }
        dW.entercomputation(); da.entercomputation(); db.entercomputation();
        // clear the accumulators at the beginning of a computation
        raw_dmbframes = 0; raw_dW.setzero(); raw_da.setzero(); raw_db.setzero();
        mean.entercomputation();    var.entercomputation();     // MVN-SGD (if not enabled, then all these are empty)
        meanacc.entercomputation(); varacc.entercomputation();
        vtemp.entercomputation();   ehtemp.entercomputation();
        numacc = 0; meanacc.setzero(); varacc.setzero();
        raw2_dmbframes = 0; raw2_dW.setzero(); raw2_da.setzero();
        rawframesseen = 0;
        local_mbframes = 0;
        dW.setzero(); da.setzero(); db.setzero();
        norms.entercomputation();
        normsmbcounter = 0;
        enteradagrad();
    }

    void lazyenterlocalloopcomputation()
    {
        if (!local_W.empty())
            return;
        local_W.resize (W.rows(), W.cols());
        local_a.resize (a.rows(), a.cols());
        local_W.entercomputation(); local_a.entercomputation();
    }

    // same do necessary finalization, e.g. in case of CUDA, copy updated models back to CPU RAM
    void exitcomputation()
    {
        W.exitcomputation(); a.exitcomputation(); b.exitcomputation();
        raw_dW.exitcomputation(); raw_da.exitcomputation(); raw_db.exitcomputation();
        raw2_dW.exitcomputation(); raw2_da.exitcomputation();
        if (!local_W.empty())
        {
            local_W.exitcomputation(); local_a.exitcomputation();
        }
        dW.exitcomputation(); da.exitcomputation(); db.exitcomputation();
        mean.exitcomputation();    var.exitcomputation();
        meanacc.exitcomputation(); varacc.exitcomputation();
        vtemp.exitcomputation();   ehtemp.exitcomputation();
        norms.exitcomputation();
        exitadagrad();
    }

    // -----------------------------------------------------------------------
    // reading and writing
    // -----------------------------------------------------------------------

    // network type is serialized to file with this structure
    struct networktypedesc_t
    {
        nonlinearitykind_t nonlinearitykind;
        // More fields can be added in the future without breaking file compat:
        // Just initialize them to default in the constructor to allow for reading old files with shorter structs.

        networktypedesc_t()
        {
            memset (this, 0, sizeof (*this));   // to allow memcmp() below
            nonlinearitykind = sigmoidkind;
        }
        bool operator!= (const networktypedesc_t & other) { return memcmp (this, &other, sizeof (*this)) != 0; }
        std::string tostring() const    // TODO: unify with nonlinearitykindtostring() which is nearly redundant to this
        {
            switch (nonlinearitykind)
            {
            case linearkind:    return "linear";
            case sigmoidkind:   return "sigmoid";
            case relukind:      return "relu";
            case leakyrootkind: return "leakyroot";
            case softmaxkind:   return "softmax";
            case softpluskind:  return "softplus";
            default:            throw std::logic_error ("tostring: invalid nonlinearitykind value");
            }
        }
    };

    template<typename FILEHANDLETYPE>
    rbmbase (FILEHANDLETYPE f)                          // constructor from file
    {
        // read optional type flags first
        // This is a little messy since it was not designed that way originally.
        string tag = fgetTag (f);
        if (tag == "BTYP")                              // network type descriptor
        {
            networktypedesc_t desc;
            size_t size = fgetint (f);
            if (size > sizeof (desc))
                throw runtime_error ("rbmbase: malformed BTYP item");
            freadOrDie (&desc, size, 1, f);
            fcheckTag (f, "ETYP");
            nonlinearitykind = desc.nonlinearitykind;   // copy over all fields
         //   fprintf (stderr, "rbmbase: reading model with non-linearity kind '%s'\n", desc.tostring().c_str());
            tag = fgetTag (f);                          // and advance the tag
        }
        else
        {
            nonlinearitykind = sigmoidkind;             // default if BTYP missing (compat with old files)
        }
        // another messy bit: mean/variance
        if (tag == "BMVN")
        {
            mean.read (f, "mean");
            var.read (f, "var");
            fcheckTag (f, "EMVN");
            tag = fgetTag (f);                          // and advance the tag
        }
        W.read (f, "W", tag);
        a.read (f, "a");
        b.read (f, "b");
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;
    }

    virtual void copyfrom (const Iannlayer & iother)
    {
        //const auto & other = dynamic_cast<const decltype(*this) &> (iother);
        const rbmbase & other = dynamic_cast<const rbmbase &> (iother);
        // copy over all fields that describe the model (but not accumulators and temps)
        nonlinearitykind = other.nonlinearitykind;
        W = other.W;    // first time will allocate the matrix
        a = other.a;
        b = other.b;
        vdimnumroundup = other.vdimnumroundup;
        hdimnumroundup = other.hdimnumroundup;
        numstream = other.numstream;
        mean = other.mean;
        var = other.var;
    }

    // redistribute the model parameters through MPI
    // The models must already have been set up and dimensioned correctly; here we only exchange the weight parameters.
    virtual void mpiredistribute (const modelupdateinfo & bpinfo)
    {
        auto & mpiaggregator = *bpinfo.mpiaggregator;
        mpiaggregator.redistribute (W.asvectorref());
        mpiaggregator.redistribute (a.asvectorref());
        if (!b.empty())
            mpiaggregator.redistribute (b.asvectorref());
    }

    // all-reduce the raw gradient through MPI
    // This is used for simple MPI variants such as model averaging (where we don't want any of the quantization or double-buffering stuff)
    virtual void mpiallreducegradient (const modelupdateinfo & bpinfo)
    {
        // all-reduce the raw gradient
        fprintf (stderr, "mpiallreducegradient: all-reducing gradient over %d frames\n", (int) raw_dmbframes);
        auto & mpiaggregator = *bpinfo.mpiaggregator;
        raw_dW.allreduce (mpiaggregator);
        raw_da.allreduce (mpiaggregator);
        if (!raw_db.empty())
            raw_db.allreduce (mpiaggregator);
        mpiaggregator.allreducescalar (raw_dmbframes);
        fprintf (stderr, "mpiallreducegradient: all-reduced gradient over total %d frames\n", (int) raw_dmbframes);
    }

    // write to file
    // This is virtual to allow networks to save network-type specific data, e.g. used for 'lineartransform'.
    // 'Overridden' reading is done in the constructor from FILE *.
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        networktypedesc_t desc, defaultdesc;
        desc.nonlinearitykind = nonlinearitykind;       // copy over the fields

        // fix preventing saving 'BTYP' tag for softmax layer (which is still a default type for output layer but defaultdesc is set to sigmoidkind). Required to keep backward compatibility with latgen
        // TODO: better check the actual object type
        if (nonlinearitykind == softmaxkind)
            defaultdesc.nonlinearitykind = softmaxkind;
        // BUGBUG: This is not working w.r.t. SVD models, which are 'linearkind', but we don't know here that this has been detected outside

        if (desc != defaultdesc)
        {
            fputTag (f, "BTYP");
            fputint (f, sizeof (desc));                 // save length for future extensibility
            fwriteOrDie (&desc, sizeof (desc), 1, f);
            fputTag (f, "ETYP");
        }
        if (!mean.empty() || !var.empty())              // for MVN-SGD
        {
            fputTag (f, "BMVN");
            mean.write (f, "mean");
            var.write (f, "var");
            fputTag (f, "EMVN");
        }
        W.write (f, "W");
        a.write (f, "a");
        b.write (f, "b");
    }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    // -----------------------------------------------------------------------
    // helper for constructors in derived classes
    // -----------------------------------------------------------------------

    // reset W, a, and b with random values
    void initrandom (unsigned int randomseed)
    {
        // for scale-preserving layers such as ReLU, we just set every 1/100-th, all others are set to 0
        // TODO: Should this also be done for linear bottleneck layers? Could be wrong if they feed a sigmoid layer (would skew the gradients).
        // Note: for sigmoid, the recommendation is r = 4 sqrt (6/(fan-in + fan-out))  [Glorot and Bengio (2010)]
        // Note: This does not work (makes ReLU training MUCH slower); so we just leave skipratio at 0 (no skipping).
        const float skipratio = 0;//(nonlinearitykind == relukind) ? 0.99f : 0.0f;  // set this many to zero (randomly)
        fprintf (stderr, "initrandom: skipping %.1f%% (%s)\n", skipratio * 100.0f, nonlinearitykindtostring (nonlinearitykind)), fflush (stderr);
        srand (randomseed);
        foreach_coord (i, j, W)
            if (skipratio != 0.0f && ::rand() < skipratio * RAND_MAX)   // (note: don't call rand() if no skipping, for back compat)
                W(i,j) = 0.0f;
            else
                W(i,j) = (::rand() * 0.1f / RAND_MAX) - 0.05f;
        foreach_row (i, a)
            a[i] = 0.0f;
            //a[i] = -4.0f;   // per recommendation in guideTR.pdf
        if (!b.empty())
            foreach_row (j, b)
                b[j] = 0.0f;
    }

    // set W to identity matrix
    void initidentity()
    {
        foreach_coord (i, j, W)
        {
            if(i == j)  W(i, j) = 1.0;
            else        W(i, j) = 0.0;
        }
    }

    // set bias(-es) to 0
    void initbiaszero()
    {
        foreach_row (i, a)
            a[i] = 0.0f;
        if (!b.empty())
            foreach_row (j, b)
                b[j] = 0.0f;
    }

    // initialize W from an adaptation matrix
    // TODO: why is this called initidentity?? Ugh!
    void initidentity (const matrix & adaptmatrix)
    {
        if (adaptmatrix.cols() != W.cols() || adaptmatrix.rows() != W.rows())
            throw runtime_error ("initidentity: intput adaptation matrix does not match with current matrix");
        foreach_coord (i, j, W)
            W(i,j) = adaptmatrix(i,j);
    }

public:

    // -----------------------------------------------------------------------
    // special-purpose accessors (it's a research project after all...)
    // -----------------------------------------------------------------------

    const matrix & peekweightmatrix() const { return W.peek(); }
    const vector & peekbias() const { return a.peek(); }

    // get the row = c, col = l value from the weight matrix [v-xieche]
    float getweightvalue(size_t c, size_t l) const { return W(c,l); }

    nonlinearitykind_t peeknonlinearitykind() const { return nonlinearitykind; }    // used to support old SVD code

    // -----------------------------------------------------------------------
    // main forward/back propagation functions
    // -----------------------------------------------------------------------

protected:

    mutable cachedmatrix cachedWs;
    mutable cachedmatrix cachedWts;
    mutable cachedmatrix cachedvs;
    mutable cachedmatrix cachedhs;
    mutable cachedmatrix cacheda1s;
    mutable cachedmatrix cachedb1s;
    mutable cachedmatrix cachedsearchdirectionW;
    mutable cachedmatrix cachedsearchdirectiona;
    mutable cachedmatrix cachedforwardstatisticsin;
    mutable cachedmatrix cachedforwardstatisticsout;

    // apply the weight matrix to v plus bias to get z (which is then the input to the non-linearity)
    // This function is shared across all types.
    void vtoz (const rbmstatevectorsref & v, rbmstatevectorsref & z) const
    {
        W.matprod_mtm (v, cachedWs, cachedvs, z, cachedhs, a, cacheda1s);     // z = W' v + a
    }

    void vtoz (const matrixstripe & v, matrixstripe & z, size_t i) const // same but only one column (on-demand LL eval in decoder)
    {
        W.matprod_col_mtm (v, z, a, i);    // z = W_i' v + a_i
    }

    void evtoeh (const rbmstatevectorsref & ev, rbmstatevectorsref & eh) const
    {
        W.matprod_mtm (ev, cachedWs, cachedvs, eh, cachedhs);     // eh = W' ev
    }

    mutable rbmstatevectors mmexchangebuffers;

    // apply weights in reverse direction (reconstruction, with bias)
    void htov (const rbmstatevectorsref & h, rbmstatevectorsref & v) const
    {
        assert (h.cols() == v.cols());
        W.matprod_mm (h, cachedWts, cachedhs, v, cachedvs, b, cachedb1s, mmexchangebuffers);     // v = W h + b
    }

    // apply weights to error signal in reverse direction for error back-propagation
    // Difference to htov() is that no bias is added as this deals with error signals.
    void ehtoev (const rbmstatevectorsref & eh, rbmstatevectorsref & ev) const
    {
        assert (eh.cols() == ev.cols());
        W.matprod_mm (eh, cachedWts, cachedhs, ev, cachedvs, mmexchangebuffers);   // v = W h
    }

    mutable rbmstatevectors softmaxbuffer;  // used for sub-minibatch model parallelism. Note: BUGBUG cannot handle sub-minibatches created in main.cpp passed on multiple threads, if we ever want to go there

    // forward propagation
    // This code is shared between Gaussian-Bernoulli and Bernoulli-Bernoulli networks as well as perceptron (softmax) and linearnetwork.
    // v and Eh are blocks of column vectors
    // For softmax, an optional mask can be added to the activation (i.e., before sigmoid and thus a large negative values means masked and 0 means not).
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly = false, rbmstatevectorsref * pmask = NULL) const
    {
        vtoz (v, Eh);   // z = W' v + a     --using Eh as a buffer for 'z'

        // mask
        if (pmask && nonlinearitykind == softmaxkind)
            Eh.addweighted (*pmask);
        else if (pmask)
            throw std::logic_error ("forwardprop: non-null mask is not implemented yet for non-softmax layer");

        //W.dump("weight");
        //Eh.dump("forward signal");
#ifdef NN_LR_TEST
		if (!linearonly) switch (nonlinearitykind)
		{
		case linearkind:  /* nothing */                 break;
		case sigmoidkind:   Eh.sigmoid();               break;
		case relukind:      Eh.setto0ifbelow(0.0f);    break;
		case leakyrootkind: throw std::logic_error("this function should be overloaded for this kind"); break;
		case softmaxkind:   Eh.sigmoid(); break;
		case softpluskind:  Eh.softplus();              break;
		}
#else
        if (!linearonly) switch (nonlinearitykind)
        {
        case linearkind:  /* nothing */                 break;
        case sigmoidkind:   Eh.sigmoid();               break;
        case relukind:      Eh.setto0ifbelow (0.0f);    break;
        case leakyrootkind: throw std::logic_error ("this function should be overloaded for this kind"); break;
        case softmaxkind:   Eh.softmax (softmaxbuffer); break;
        case softpluskind:  Eh.softplus();              break;
        }
#endif 
#ifdef SAMPLING_EXPERIMENT
        static unsigned int randomseed = 0;
        Eh.samplebinary (Eh, randomseed);
        randomseed++;
#endif
    }

    // forward propagates hessian vector product statistics
    // layerIn: input to layer
    // layerOut: output of layer
    // forwardstatisticsIn: statistics of previous layer
    // forwardstatisticsOut: resulting statistics
    virtual void forwardprophessianvectorproduct(const rbmstatevectorsref &layerin, const rbmstatevectorsref &layerout,        
                                                 const rbmstatevectorsref &forwardstatisticsin, rbmstatevectorsref &forwardstatisticsout, bool zeroforwardstatisticsin) const
    {
        // forwardstatisticsout = W^T forwardstatisticsin
        if (!zeroforwardstatisticsin)
            W.matprod_mtm(forwardstatisticsin, cachedWs, cachedforwardstatisticsin, forwardstatisticsout, cachedforwardstatisticsout);

        // forwardstatisticsout += cgsearchdirectionW^T * layerin + cgsearchdirectiona
        float weight = zeroforwardstatisticsin ? 0.0f : 1.0f;
        cgsearchdirectionW.matprod_mtm(layerin, cachedsearchdirectionW, cachedvs, forwardstatisticsout, cachedforwardstatisticsout, cgsearchdirectiona, cachedsearchdirectiona, weight);
        
        // forwardstatisticsout *= sigma' (componentwise)
        if (nonlinearitykind != sigmoidkind)
            throw runtime_error ("forwardprophessianvectorproduct: non-sigmoid units not implemented");
        forwardstatisticsout.mulbydsigm(layerout);
    }

    // backward error propagation: compute the error signal for a group of training frames (multiple columns)
    //  - in 'h' are the activation probabilities from the preceding forwardprop() step
    //  - in 'eh' is the error signal from layer above
    //  - out 'eh' is the error signal to be used for model update (updated in-place)
    //  - out 'ev' is the error signal to be passed on to layer below
    // This combines two steps of back-propagation, actually:
    //  - through the non-linearity
    //  - through the weight matrix
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        assert (eh.cols() == h.cols() && eh.rows() == h.rows());

        // compute 'dW' = [ dh/d(w(i,j)) ] and 'da' = [ dh/d(a[i]) ]
        // eh = hdesired - h
        // err = eh .* derivative of non-linearity
        // sigmoid:
        //   err = eh .* h .* (1 - h)
        // ReLUs:
        //   err = eh .* (h > 0)      // note: h = lru(z), i.e. > 0 <=> z > 0 (derivative of lru(z) is 1 if z > 0, 0 else)
        // update 'eh' in place for later use in accumulation

        // through the non-linearity: multiply by derivative
        // This is done in place because we need the very same product later in the model update.
        // Note that eh no longer corresponds to e^l in the paper after this operation.
        switch (nonlinearitykind)
        {
        case linearkind:    break;
        case sigmoidkind:   eh.mulbydsigm (h); break;
        case relukind:      eh.mulbydlru (h); break;
        case leakyrootkind: throw std::logic_error ("this function should be overloaded for this kind"); break;
        case softmaxkind:   break;    // TODO: test this
        case softpluskind:  eh.mulbydsoftplus (h); break;
        }
        //eh.dump("multiplied by derivative of sigmoid");

        // divided the log (us + epison) for exerting log function on hidden layer. [v-xieche]
#ifdef LOGINSIGMOID
        eh.divideaddsigmoid (h);
#endif
        // through the weight matrix: multiply by W
        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh  (eh is the updated one)
            //ev.dump("error signal");
    }

    // -----------------------------------------------------------------------
    // main update functions
    // -----------------------------------------------------------------------

    // NUMA temp storage for pushtorawgradient() and updategradient() below
    // This means one instance of the model can only train once at the same time. Makes sense.
    mutable vector sumhtmp;
    mutable vector sumvtmp;
    mutable matrix httmp;
    mutable cachedmatrix cachedhts;
    mutable cachedmatrix cachedvts;
    // and a temp for MVN-SGD
    mutable rbmstatevectors vtmp;

public:

    // compute the raw (un-smoothed, un-scaled) gradient: (h,v) -> raw_dW,a,b
    void pushtorawgradient (const rbmstatevectorsref & h, const rbmstatevectorsref & v, bool updateb,
                            rmbmodelmatrix & to_dW, rmbmodelvector & to_da, rmbmodelvector & to_db, size_t & to_dmbframes,
                            const modelupdateinfo * modelupdateparameters) const;
	
    // moved to the end of the file to enable access to types declared later --TODO: this is a hack for 'convolutional', we should get rid of this; rather use a virtual function

    // update the momentum-smoothed gradient accumulator from the raw gradients; also may do some regularization step here
    // dW <- dW * feedbackweight + raw_dW * inputweight
    // The caller chooses the weights to implement or not implement a unity-gain filter, apply learning rate etc.
    void updategradient (const float feedbackweight, float inputweight, bool updateb, const modelupdateinfo * modelupdateparameters)
    {
        if (raw_dmbframes == 0)
        {
            fprintf (stderr, "updategradient: called with an empty raw gradient (raw_dmbframes = 0)\n");
            inputweight = 0.0f;     // consume it as if it was zero (this saves us to call setzero() on it)
        }
        fprintf (stderr, "updategradient: consuming %d frames from raw gradient acc\n", (int) raw_dmbframes);

        // accumulate into momentum accumulator:
        // da <- da * feedbackweight + sum (h)      * inputweight
        // dW <- dW * feedbackweight + sum (v * h') * inputweight
        // where h and v are matrices with columns = frames.
        // to implement the momentum filter, the weights are meant to be:
        //  - feedbackweight = momentum
        //  - inputweight    = (1-momentum)
        da.addweighted (feedbackweight, raw_da, inputweight);
        dW.addweighted (feedbackweight, raw_dW, inputweight);
        if (updateb)
            db.addweighted(feedbackweight, raw_db, inputweight);

        // apply regularization that can be done to the deltas
        // Note: This may be buggy, as it interferes with momentum.
        if (modelupdateparameters && (modelupdateparameters->regtype == regCL || modelupdateparameters->regtype == regCS))
        {
            const float threshold = fabs (modelupdateparameters->nochangeifaboveorbelow);
            if (threshold > 0)
            {
                // TODO: interplay with momentum not clear
                if (modelupdateparameters->nochangeifaboveorbelow > 0) 
                    dW.setto0ifabsabove2 (W, threshold);
                else
                    dW.setto0ifabsbelow2 (W, threshold);
            }
        }

        // diagnostics for relu (and maxouts) --in what value range are the gradients?
#if 0
        if (nonlinearitykind == relukind /*|| nonlinearitykind == linearkind*/)
            fprintf (stderr, "updatedeltas: root of av sqr of dW = %.10f and da = %.10f\n", sqrt (dW.avsqr()), sqrt (da.avsqr()));
#endif
#if 0   // this is buggy; regularization should not modify the momentum accumulator
        else if (modelupdateparameters && modelupdateparameters->regtype == regL2)
        {
            const rbmbase & rbmlayer = dynamic_cast<const rbmbase &> (*modelupdateparameters->preflayer);
            const rmbmodelmatrix & Wref = rbmlayer.W;
            const rmbmodelmatrix & aref = rbmlayer.a;
            const float alpha = modelupdateparameters->alpha * v.cols() * inputweight;  //adjust it based on number of frames

            assert(W.rows() == Wref.rows());
            assert(W.cols() == Wref.cols());
            assert(a.rows() == aref.rows());
            assert(a.cols() == aref.cols());

            // dW += alpha * (Wref - Wcur)
            // TODO: alpha interplays with scaling of gradient --ensure it is correct
            dW.addweighted (1.0f, W, -alpha);
            dW.addweighted (1.0f, Wref, alpha);
            da.addweighted (1.0f, a, -alpha);
            da.addweighted (1.0f, aref, alpha);
        }
#endif

        // we have consumed the raw gradient
        raw_dmbframes = 0;
#if 0
        raw_dW.setzero(); raw_da.setzero(); raw_db.setzero();
#endif
    }

    // update AdaGrad accumulators
    // 'mbframes' is the number of frames in this minibatch
    void updateadagrad (const modelupdateinfo & modelupdateparameters, const rbmmodelmatrix & dW_in, const rbmmodelmatrix & da_in, size_t mbframes)
    {
        lazyinitadagrad();  // upon first call, this will allocate the AdaGrad accumulators and lazily call enteradagrad() on them
        const size_t Tframes = modelupdateparameters.adagradT;  // adagradT = 2 * 3600 * 100
        const float keepweight = (float) exp (-1.0 * mbframes / Tframes);
        // accumulate the square of the (batch) gradient
        if (mbframes > 0)   // (if 0 then it is allowed that the gradient matrix is not actually initialized; need to pretend it's zero)
        {
            adagraddWsqrsum.accumulatesqr (dW_in, keepweight);
            adagraddasqrsum.accumulatesqr (da_in, keepweight);
            if (!adagraddWsum.empty())
            {
                adagraddWsum.addweighted (keepweight, dW_in, 1.0f - keepweight);
                adagraddasum.addweighted (keepweight, da_in, 1.0f - keepweight);
            }
        }
        adagradframes = keepweight * adagradframes + (1.0f - keepweight) * mbframes;
        // keep some stats (these are used to control some stuff)
        adagradframespushed += mbframes;
        adagradsummands++;
    }

    // apply AdaGrad to gradient, put result into adagradW/a
    // Only if 'targetadagradavdenom' is not specified, i.e. passed as 0.0, we will use 'adagradstate' to accumulate the actual average. Otherwise it is unused.
    void applyadagrad (const rbmmodelmatrix & dW_in, const rbmmodelmatrix & da_in, size_t mbframes, float targetadagradavdenom/*0.0 to use actual*/, adagradstate_t & adagradstate/*unused when using targetadagradavdenom*/)
    {
        if (adagradframes == 0.0f)  // at start, we may have no gradient due to double-buffering--pretend the raw gradient is zero
        {
            adagraddW.setzero();
            adagradda.setzero();
            return;
        }
#if 0
        // layers seem to have fundamentally different gradient ranges
        //  - input layer is ~4 times larger
        //  - top layer is ~5 times smaller
        // Since we normalize by dividing by their range, those layers are implicitly treated rather differently.
        // So we manually add an adjustment weight for them to make the manually chosen weight more similar.
        // I have no idea whether that will help or rather defeat the purpose of AdaGrad... We will see!
        if (type() == "rbmgaussbernoulli")
            targetadagradavdenom *= 4.0f;                   // input layer: make 4 times stronger
        else if (type() == "perceptron")
            targetadagradavdenom /= 5.0f;                   // top layer: weigh down 5 times
#endif

        // compute the weighting
        // AdaGrad divides by the stddev over time of each weight gradient component.
        // I see e.g. average weight gradients of 0.000567.98638916 or 0.00155116772461.
        // Thus, AdaGrad changes their range dramatically, and thus the numeric meaning of our learning-rate parameter.
        // To keep known learning rates roughly meaningful, we compensate by multiplying with 'avdenom', for which we have two methods:
        //  - a given fixed value specified by the user, e.g. --adagradavdenom 1/400
        //  - the actual average over all components
        // The former method has shown to work better than the latter with a value of 0.0025.
        // The latter is expensive and can only operate at a delay of 1 minibatch since the computation is inside here, and this does not really help so I won't move it out.
        // If this method is used, then we do not take the average over all layers for the first 16 minibatches, and from then use the average from the previous minibatch, refreshed every 16 minibatches.
        // TODO: we don't need to compute the following when using a fixed target; we do now for debugging only
if (&adagraddW == &dW_in && targetadagradavdenom != 0)  // hack: for onpartialsubgradient, we temporarily compute the raw gradient in adagradW, which we also need for adagradientavdenom()
    fprintf (stderr, "applyadagrad: skipping computation of avdenom due to buffer over-use\n"); // when adagradientavdenom() is only for diagnostics, we can skip it
else if (&adagraddW == &dW_in)
    throw std::logic_error ("applyadagrad: cannot compute avdenom due to buffer over-use --fix this code if you need it\n");    // not for diagnostics: this combination is not supported
else
        if ((adagradsummands < 16) || (adagradsummands % 16 == 0))  // for first 16, we use our own av from this layer's weight matrix
        {
            avdenomdW = adagraddW.adagradientavdenom (adagraddWsqrsum, adagraddW/*denom temp*/, (float) adagradframes, adagraddWsum, adagraddWsum.empty() ? 0.0f : (float) adagradframes, mbframes);
            avdenomda = adagradda.adagradientavdenom (adagraddasqrsum, adagradda/*denom temp*/, (float) adagradframes, adagraddasum, adagraddWsum.empty() ? 0.0f : (float) adagradframes, mbframes);
            adagradstate.accumulate (avdenomdW, adagraddWsqrsum.rows(), adagraddWsqrsum.cols(), mbframes);
            adagradstate.accumulate (avdenomda, adagraddasqrsum.rows(), adagraddasqrsum.cols(), mbframes);
        }
        else if (adagradsummands > 16)    // once we have enough & are stable, we use the cross-layer average from a previous minibatch
            avdenomdW = adagradstate.gettotalavdenom();

        const float usedavdenom = (targetadagradavdenom != 0) ? targetadagradavdenom : avdenomdW;   // if we are given a fixed avdenom then we just use that

#if 0
        const float lrfudgefactor = 2.5f / 1.122f;      // I want behavior of 1024 (avdenom=1.122e-3) when target given as 2.5e-3
#else
        const float lrfudgefactor = 1.0f;//(type() == "perceptron") ? 0.66f : 1.8f;     // approximate old behavior for 4k frames
#endif

        // compute the adapted gradients
        // adagradient = gradient ./ (adagraddenom / avdenom)
        // where adagraddenom(i,j) = sqrt (sqracc(i,j) / numframes)     as a rolling average (low-pass)
        // and avdenom = normalization value to bring the adaptive gradient back into normal value ranges, such that learning rates remain meaningful
        // 'avdenom' can be chosen as average over all components (repeatedly updated) or a hand-specified constant. The latter seems to work better.
        adagraddW.adagradient (dW_in, adagraddWsqrsum, adagraddW/*denom temp, currently unused*/, (float) adagradframes, mbframes, lrfudgefactor * usedavdenom,                    -1.0f/*remove this*/);
        adagradda.adagradient (da_in, adagraddasqrsum, adagradda/*denom temp, currently unused*/, (float) adagradframes, mbframes, lrfudgefactor * usedavdenom/*borrow from dW!*/, -1.0f);
        //if (targetadagradavdenom != 0)                      // if we are given a fixed avdenom then we just use that
            fprintf (stderr, "applyadagrad: av. weights (%.5f eff. frames) for dW: %.2fe-3, da: %.2fe-3, using: %.2fe-3\n",
                             adagradframes, avdenomdW * 1e3, avdenomda * 1e3, usedavdenom * 1e3); // 1e3 for better readability
    }

    // the actual inner model-update function
    void addtomodel (float keepweight, const rmbmodelmatrix & dW, const rmbmodelmatrix & da, bool updateb, const rmbmodelmatrix & db, float learningratepersample)
    {
        // Description of MVN-SGD:
        //
        // We modify local objective to express this w.r.t. zero-mean input v1 and constant mean v0, where weights apply only to v1:
        //
        //          v = D v1 + v0        D = diagonal stddev matrix
        //  z1(W1,a1) = W1'v1 + a1       z1 does not mean zero-mean, it means the changed local objective
        //         W1 = D W
        //         a1 = a + W'v0         where W is not a parameter to be updated
        // 
        // We compute desired gradients (gW1, ga1) of this local objective function.
        // (Raw gradients are gW1 = e v1' = invD e (v-v0)' ; ga1 = e ; we apply momentum and AdaGrad to them.)
        // 
        // After model update, z1 computes as:
        //  z1(W1 + gW1, a1+ga1) = (W1 + gW1)'v1 + (a1 + ga1)
        //                       = (D W + gW1)'v1 + (W'v0 + a) + ga1
        //                       = (W + invD gW1)'D v1 + (W'v0 + a) + ga1
        //                       = (W + invD gW1)'D v1 + (W + invD gW1)'v0 - (W + invD gW1)'v0 + (W'v0 + a) + ga1
        //                       = (W + invD gW1)'(D v1 + v0)              - (W + invD gW1)'v0 + (W'v0 + a) + ga1
        //                       = (W + invD gW1)'(D v1 + v0)                   - invD gW1'v0          + a  + ga1
        //                       = (W + invD gW1)'D v + a + (ga1 - invD gW1'v0)
        //                       = z(W + invD gW1, a + (ga1 - invD gW1'v0))
        // 
        // I.e. the parameter update should be
        //    W <- W + invD gW1
        //    a <- a + (ga1 - invD gW1'v0)
        //
        // Note that gW1 already has a factor of invD compared to no MVN-SGD; so we really scale down the original gradient by the variance (D^2).
        // This makes sense: a D times larger v will lead to an invD larger W (by means of training) but D times larger gradient; i.e. relatively, the gradient is D^2 larger.
        // The above scales the gradient down by D^2, which will compensate this.
        //
        // So we need to apply the correction terms where gW1 and ga1 are the gradients computed for the modified local objective, which is what we have in our dW/da variables.
        // Thus, we need to change dW to invD dW and then subtract dW'mean from a.
        //
        // Hope: variance normalization may be useful for ReLUs...? Or will it just run away? Or will it simply not matter if it runs away? I.e. we can post-scale to redistribute.
        //       Variance-only scaling simply means to normalize the gradient by the stddev. This will do away with scale invariance in ReLUs indeed (I think).

        // MVN-SGD correction for weights
        if (!var.empty())
        {
fprintf (stderr, "addtomodel: doing varnorm for MVN-SGD\n");
            // scale dW by invD; that is, scale every row by 1/sqrt(var)--this is what meanvarnorm() does, actually, so we can reuse the kernel
            // UGH! doing this on a 'const' dW; pray that noone uses this afterwards...
            auto & gW1 = const_cast<rmbmodelmatrix &> (dW); // UGH!
            gW1.varnorm (var);
            // dW has been updated
        }

        // main parameter update
        W.addweighted (keepweight, dW, learningratepersample);
        a.addweighted (keepweight, da, learningratepersample);

        // MVN-SGD correction for bias
        // TODO: if we already patch 'W' in-place, we can also patch 'a' in-place
        if (!mean.empty())
        {
            if (updateb)
                throw std::logic_error ("addtomodel: MVN-SGD not implemented for RBM pre-training");    // (not sure if it even makes sense)
fprintf (stderr, "addtomodel: doing sgemm_mtm for MVN-SGD\n");
            a.sgemm_mtm (1.0f, dW, mean, -learningratepersample);
        }
        assert (!db.empty() || !updateb);
        if (updateb)
            b.addweighted (keepweight, db, learningratepersample);
    }

    // add gradient to model parameters; also do some regularization here
    // gradients are already
    //  - summed over the frames of the minibatch
    //  - momentum-smoothed
    // This also implements some regularization and AdaGrad.
    void addgradienttomodel (float learningratepersample, bool updateb, const rmbmodelmatrix & dW, const rmbmodelmatrix & da, const rmbmodelmatrix & db,/*TODO: rename dW,a,b to avoid shadowing*/
                             const modelupdateinfo * modelupdateparameters, size_t mbframes/*for L2 reg and AdaGrad*/)
    {
#if 0   // hack to skip the first N frames; used for tracking down the half-batch issue
        rawframesseen += mbframes;
        const size_t minrawframesseen = 86400; // 1% of SWBD data set
        if (rawframesseen <= minrawframesseen)
        {
            fprintf (stderr, "addgradienttomodel: too few frames (%d, min is %d) -> skipping\n", rawframesseen, minrawframesseen);
            return;
        }
#endif

        // W += dW * learning rate
        // a += da * learning rate
        // b += db * learning rate  if 'updateb'

        float currentmodeldiscount = 0.0f;
        float L2weight = 0.0f;
        if (modelupdateparameters && (modelupdateparameters->regtype == regL2 || modelupdateparameters->regtype == regIsmoothing))
        {
            // L2 regularization leads to adding an additional term to the overall objective, i.e. an additional term to the gradient.
            // That term is (reference model itself minus the current model), which is added to the normal BP gradient with a weight.
            // In the context of minibatches, we uniformly distribute its contribution over all minibatches (but using the respective current model).
            // The 'modelupdateparameters->alpha' specifies the weight of the contribution of the L2 term per frame to the gradient.

            // I-smoothing for MMI has a similar form, except that the current model is not subtracted.
            // Thoughts on setting the parameters:
            //  - for full-batch MMI, we'd add the previous model with a weight of tau, e.g. 50;
            //    i.e. a contribution of tau/T per frame (T=#frames in corpus).
            //  - mb SGD converges faster; I suggest to define an "effective" corpus size T_eff;
            //    i.e. the per-frame contribution would be tau/T_eff.
            //    E.g. if for the 111-million-frame SWBD 309h corpus, the T_eff=1,000,000 (~1/100 of the full corpus),
            //    alpha = tau/T_eff = 50e-6 = 0.000050  (per 1000-frame minibatch, that'd be 0.05, i.e. we add 5% of the ref model in each mb)
            // This never worked I think, so the comment above can be discarded. F-smoothing is what works.

            // model update with L2 regularization
            if (updateb) throw std::logic_error ("addgradienttomodel: L2 regularization not supported in pre-training (b update)");

            // weight for L2/Ismoothing term
            L2weight = modelupdateparameters->L2weightpersample * mbframes;
            // for L2 reg, we subtract the current model (for Ismoothing, we just add the old model as observations)
            if (modelupdateparameters->regtype == regL2)
                currentmodeldiscount = L2weight;
        }

        // add the gradient
        // When doing L2 regularization, we also subtract the current model.
        if (modelupdateparameters && modelupdateparameters->enableadagrad && modelupdateparameters->adagradwhere == modelupdateinfo::onsmoothedgradient)
        {
            if (adagradframes == 0.0f)
            {
                // we could also accumulate
                fprintf (stderr, "addgradienttomodel: skipping model update since AdaGrad gradient not estimated yet (0 frames)\n");
                return;
            }

            //const float targetadagradavdenom = (modelupdateparameters->adagradavdenom/32) ;//* learningratepersample; // we compute AdaGrad denom after learningratepersample is already applied, so need to factor it in here as well
            // TODO: adagradavdenom is specified 32 x too large, for compat with old code; get rid of this
            // Example: specified 0.08 -> 0.0025 * learningrate
            // It really just multiplies onto the learning rate
            //const float targetadagradavdenom = 0.0025; // old target param is this
            applyadagrad (dW, da, mbframes, modelupdateparameters->adagradavdenom, *modelupdateparameters->adagradstate);

            addtomodel (1.0f - currentmodeldiscount, adagraddW, adagradda, false/*updateb*/, db/*dummy*/, learningratepersample);
        }
        else    // this is the regular model update
        {
            addtomodel (1.0f - currentmodeldiscount, dW, da, updateb, db, learningratepersample);
        }

        // for L2 reg and Ismoothing, we need to add in a portion of the reference model
        if (L2weight > 0.0f)
        {
            const rbmbase & rbmlayer = dynamic_cast<const rbmbase &> (*modelupdateparameters->preflayer);   // get the reference model
            addtomodel (1.0f, rbmlayer.W, rbmlayer.a, updateb, rbmlayer.b, L2weight);   // (note: never tested for the 'updatebd' case) (note: this version never tested actually after refactoring)
        }
    }

    // map momentum from per-sample to mbsize
    static float scalemomentum (double momentumpersample, size_t mbsize)
    {
        if (momentumpersample > 0.0)
            return (float) exp (log (momentumpersample) * mbsize);
        else
            return 0.0f;
    }

    // model update, first step: set the raw gradients
    // This updates raw_d{W,a,b,mbframes}.
    // 'ehxs' is the error signal multiplied with the sigmoid' (except for linear fDLR layer).
    virtual void backpropagationmodelupdate1 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v, const modelupdateinfo & bpinfo)
    {
        // MVN-SGD mean/var estimation: gather all columns of v into an accumulator
        if (!mean.empty())
        {
            if (normsmbcounter < 8 || normsmbcounter % 16 == 0)
                v.glimpse ("v", true);

            v.meanvaracc (numacc > 0, meanacc, mean, varacc);
            numacc += v.cols();             // and remember how many frames we added

            if (normsmbcounter < 8 || normsmbcounter % 16 == 0)
            {
                meanacc.glimpse ("meanacc", true);
                if (!var.empty())
                    varacc.glimpse ("varacc", true);
                mean.glimpse ("mean", true);
                if (!var.empty())
                    var.glimpse ("var", true);
            }
        }

#if 0
        pushtorawgradient (ehxs, v, false/*updateb*/, /*operate on these:*/raw_dW, raw_da, raw_db, raw_dmbframes, &bpinfo);
#else
        // simple case
        if (!bpinfo.enableadagrad || bpinfo.adagradwhere != modelupdateinfo::onpartialsubgradient)
        {
            pushtorawgradient (ehxs, v, false/*updateb*/, /*operate on these:*/raw_dW, raw_da, raw_db, raw_dmbframes, &bpinfo);
            return;
        }

        // complex case: in AdaGrad 'onpartialsubgradient' mode, we compute AdaGrad on small chunks
        // This is done by computing the gradient itself in chunks and accumulating those, while also estimating the AdaGrad denominator on them.
        // This is VERY slow! Example:
        //  with:    finetune: completed epoch .41, av log pp = -8.47981 in 864000 frames (11.8% correct) [16384,1.2 min,17.5 kfps-] ###
        //  wihtout: finetune: completed epoch .41, av log pp = -2.34448 in 864000 frames (50.7% correct) [16384,0.7 min,38.5 kfps-] ###
        // It also doesn't really make a difference (well, above it did, MB size too high, but in real tests, it made no difference accuracy-wise).
        // The temporary gradients are computed inside adagraddW/a.
        // TODO: we can remove this code again I think.
fprintf (stderr, "backpropagationmodelupdate1: AdaGrad on partial sub gradient =====================================\n");
        lazyinitadagrad();  // upon first call, this will allocate the AdaGrad accumulators and lazily call enteradagrad() on them
        // in case of sub-partial AdaGrad, we compute it in pieces
        const size_t targetchunksize = 64;//ehxs.cols(); //256;                     // intended chunk size
        const size_t inmbsize = ehxs.cols();
        const size_t chunks = max ((inmbsize + targetchunksize/2) / targetchunksize, 1);   // rounded int division
        //const size_t chunksize = (inmbsize+chunks-1) / chunks;  // rounded-up int division
        for (size_t chunk = 0; chunk < chunks; chunk++)
        {
            // compute the gradient chunk by chunk
            const size_t ts = inmbsize * chunk     / chunks;
            const size_t te = inmbsize * (chunk+1) / chunks;
            // compute raw gradient into adagraddW/a (using it as a temp buffer)
            fprintf (stderr, "backpropagationmodelupdate1: computing partial-partial minibatch gradient over frames %d..%d\n", ts, te-1);
            rbmstatevectorsref chunk_ehxs (ehxs.stripe (ts, te-ts));
            rbmstatevectorsref chunk_v    (v.stripe (ts, te-ts));
            auto & tempdW = (raw_dmbframes > 0) ? adagraddW : raw_dW;   // for 0 we can use raw_dW; this will give us an AdaGrad avdenom value (which uses adagraddW as a buffer as well)
            auto & tempda = (raw_dmbframes > 0) ? adagradda : raw_da;
            size_t tempdmbframes = 0;
            pushtorawgradient (chunk_ehxs, chunk_v, false/*updateb*/, /*operate on these:*/tempdW, tempda, raw_db/*dummy*/, tempdmbframes/*=0, means don't add*/, &bpinfo);
            // apply adagrad
            updateadagrad (bpinfo, tempdW, tempda, tempdmbframes);   // accumulate AdaGrad running sums
            applyadagrad (tempdW, tempda, tempdmbframes, bpinfo.adagradavdenom, *bpinfo.adagradstate);   // result -> adagraddW/a
            // now add to raw_dW/a
            const float keepweight = (raw_dmbframes > 0) ? 1.0f : 0.0f; // if we already have frames in here, so accumulate
            if (raw_dmbframes)
                fprintf (stderr, "backpropagationmodelupdate1: *adding* partial-partial minibatch to raw gradient acc (current frames: %d)\n", (int) raw_dmbframes);
            raw_dW.addweighted (keepweight, adagraddW, 1.0f);
            raw_da.addweighted (keepweight, adagradda, 1.0f);
            raw_dmbframes += tempdmbframes;                          // this is the #frames we have accumulated in here up to now
        }
        fprintf (stderr, "backpropagationmodelupdate1: %d frames in raw gradient acc\n", (int) raw_dmbframes);
#endif
    }

    // 4DSGD D*SGD: distributed data-parallel deterministic double-buffered SGD :)
    // data parallelism design for async behavior:
    //  - quantization:
    //     - GPU is requested to asynchronously quantize stripes of data into a CPU buffer
    //        - all pieces are requested in the order in which we need to send them
    //        - the GPU function returns an opaque event handle to wait for (the function takes a key so that we can reuse the events // first try without)
    //     - GPU lib gets a new function for the CPU to wait on an event handle
    //  - data exchange:
    //     - MPI lib gets a function to do data exchange on the chunks
    //        - aggregation phase:
    //           - it gets passed an array of lambdas that return buffers to send
    //           - for each of those lambdas
    //              - the lambda will call the respective GPU event-wait functions and return the buffer to exchange
    //              - each returned buffer is then send asynchronously to the respective stripe owner
    //              - while concurrently we receive a stripe in our own role as a stripe owner
    //              - wait for reception to complete
    //              - dequantize and aggregate our stripe
    //              - wait for sending to complete  // needed?
    //           - now, each node has the stripe it owns completely aggregated
    //        - distribution phase:
    //           - this function also gets passed an array of lambdas to copy buffers to the GPU and dequantize them
    //           - for each other compute node
    //              - use MPI to send aggregated quantized stripe data to respective other compute node
    //              - while receiving a stripe from another compute node
    //              - wait for reception to complete
    //              - call send-to-GPU lambda // no event needed since next compute step will sync on "stream 0" i.e. wait until all ops are complete
    //              - wait for sending to complete  // needed?
    //           - now, each node has a copy of all aggregated stripes, and they are on their way to the GPU memory to be dequantized there and ready for use
    //        - note that each lambda corresponds to a full-stack stripe, while the GPU returns per-layer stripes
    //  - interaction with model parallelism:
    //     - model parallelism also stripes models
    //     - we align MPI stripes with GPU stripes such that one MPI stripe does not span multiple GPUs
    //       (this will require that the number of nodes is equal or larger than the number of GPUs)
    //     - (if that is a problem, the alternative is to use numnodes*numgpus stripes in the end)
    //     - this design should be "correct" but not necessarily efficient since the data-transfer unit usage of the GPUs has not been carefully aligned
    //       (the above simply assumes that there are two transfer units, which we don't actually have, and we don't know how NVidia will multiplex things)

    // doc on PPL tasks:
    //  - auto xxx = create_task ([capture by value] { xxx; return result; }); auto res = xxx.get(); // waits and gets result's value
    //  - passing lambdas to create_task(): http://msdn.microsoft.com/en-us/library/dd492427.aspx
    //  - concurrency::task::wait() to wait for completion, and get() to get the lambda's return value--yes!
    //  - task<int> t([]() { return 42; });

    // model update, second step, in case of data parallelism: exchange gradients across compute nodes

    // helper to allocate CUDA-suitable memory (for use in distributed data parallelism)
    virtual shared_ptr<char> allocatetransferbuffer (size_t stripe, size_t size)
    {
        // the buffer is shared; we just ask a random matrix to allocate it for us
        return raw_dW.allocatetransferbuffer (stripe, size);
    }

    // step 1: quantize a model stripe, write result into sub-range if the specified CPU-side buffer
    // We also return the number of frames accumulated in the current sub-batch.
    // This is called from the MPI aggregator, on the main thread.
    virtual void quantizeandfetchsubbatchstripe (size_t stripe, char * bufferbegin, size_t buffersize, size_t & submbframes)
    {
        submbframes = raw_dmbframes;        // pass back the number of frames
        raw_dW.quantizeandfetchsubbatchstripe (stripe, adagraddWsqrsum, adagradframes, bufferbegin, buffersize);
        raw_da.quantizeandfetchsubbatchstripe (stripe, adagraddasqrsum, adagradframes, bufferbegin, buffersize);
    }

    // step 2: wait for our stripe to be completely quantized into the CPU-side buffer
    // This is called from the MPI aggregator, on the *background* thread.
    virtual void syncfetchsubbatchstripe (size_t stripe)
    {
        raw_dW.syncfetchsubbatchstripe (stripe);
        raw_da.syncfetchsubbatchstripe (stripe);
    }

    // step 3: accumulate quantized stripes into the stripe that we own
    // 'accumulator' is managed inside the parameter matrix object.
    // 'isfirst' is the first time (used to reset)
    // 'islast' is the last time (used to fold in fixed cost operations here)
    virtual void unquantizeandaggregatestripe (size_t ourstripe, size_t kfrom, const char * bufferbegin, size_t buffersize, bool isfirst,
                                               bool islast, size_t mbframes, const modelupdateinfo & bpinfo, double momentumpersample, float learningratepersample)
    {
        // if 'islast', we optionally cut in fixed-cost steps (AdaGrad, momentum, learning-rate scaling)
        float adagradkeepweight = 0.0f, momentumkeepweight = 0.0f, learningratescaling = 1.0f;
        if (islast && bpinfo.distributefixedcost)
        {
            const size_t Tframes = bpinfo.adagradT;
            if (Tframes != 0)
                adagradkeepweight = (float) exp (-1.0 * mbframes / Tframes);
            momentumkeepweight = scalemomentum (momentumpersample, mbframes); // map momentum per frame to momentum per minibatch
            // setting either to non-zero will trigger the operation to be applied here
            learningratescaling = learningratepersample;    // also pre-scale the gradient with the learning rate
        }
        raw_dW.unquantizeandaggregatestripe (ourstripe, kfrom, bufferbegin, buffersize, isfirst, mbframes, adagradkeepweight, bpinfo.adagradavdenom, momentumkeepweight, learningratescaling);
        raw_da.unquantizeandaggregatestripe (ourstripe, kfrom, bufferbegin, buffersize, isfirst, mbframes, adagradkeepweight, bpinfo.adagradavdenom, momentumkeepweight, learningratescaling);
    }

    // step 4: quantize an aggregate stripe from the aggregation accumulator
    virtual void quantizeandassignaggregatedstripe (size_t stripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled)
    {
        raw_dW.quantizeandassignaggregatedstripe (stripe, bufferbegin, buffersize, reuserangescaled);
        raw_da.quantizeandassignaggregatedstripe (stripe, bufferbegin, buffersize, reuserangescaled);
    }

    // step 5: move back an aggregated stripe (which is in quantized form)
    // this runs on a bg thread and only kicks off the CPU-to-GPU transfer
    virtual void assignaggregatedstripe (size_t stripe, const char * bufferbegin, size_t buffersize)
    {
        raw_dW.assignaggregatedstripe (stripe, bufferbegin, buffersize);
        raw_da.assignaggregatedstripe (stripe, bufferbegin, buffersize);
    }

    // step 6: unquantize a model stripe, from sub-range if the specified CPU-side buffer
    // This is called from the MPI aggregator, on the main thread.
    // In 'distributefixedcost' mode, this will already update the model.
    virtual void syncassignaggregatedstripeandunquantize (size_t stripe, const char * bufferbegin, size_t buffersize, size_t aggmbframes,
                                                          const modelupdateinfo & bpinfo)
    {
        if (bpinfo.distributefixedcost) // in 'distributefixedcost' the gradient is the final one, so we can add it in right away
        {
            //fprintf (stderr, "syncassignaggregatedstripeandunquantize: straight update from unquantized raw gradient (%d frames)\n", aggmbframes);
            if (aggmbframes > 0)
            {
                raw_dW.syncassignaggregatedstripeandunquantize (stripe, bufferbegin, buffersize, &W/*add to here*/);
                raw_da.syncassignaggregatedstripeandunquantize (stripe, bufferbegin, buffersize, &a/*add to here*/);
            }
            raw_dmbframes = 0;      // already consumed
        }
        else
        {
            //fprintf (stderr, "syncassignaggregatedstripeandunquantize: updating raw_dmbframes from %d to %d\n", raw_dmbframes, aggmbframes);
            raw_dmbframes = aggmbframes;        // number of frames after aggregation
            // special case: if 'aggmbframes' is 0 then we don't require a buffer (we just set the gradient to zero instead of unquantizing anything)
            // This is needed for the first call in double-buffered mode, which has no data.
            if (raw_dmbframes == 0)
            {
                raw_dW.setzero();
                raw_da.setzero();
                return;
            }
            raw_dW.syncassignaggregatedstripeandunquantize (stripe, bufferbegin, buffersize, nullptr);
            raw_da.syncassignaggregatedstripeandunquantize (stripe, bufferbegin, buffersize, nullptr);
        }
    }

    // called after backpropagationmodelupdate1() but only if not deferred update; and done before the MPI exchange
    // We can do the AdaGrad application here.
    // We also do the hack for faking double-buffering without actual MPI here.
    // And the local loop for data parallelism.
    virtual void backpropagationmodelupdate2 (const modelupdateinfo & bpinfo, bool mpimaisfirst, bool mpimaislast, float learningratepersample, double momentumpersample)
    {
        // hack: implement double buffering without MPI (for direct comparison)
        if (!raw2_dW.empty())
        {
            fprintf (stderr, "backpropagationmodelupdate2: emulating double-buffering on raw gradient\n");
            ::swap (raw_dmbframes, raw2_dmbframes);
            raw2_dW.swap (raw_dW);
            raw2_da.swap (raw_da);
        }

        // update AdaGrad  --when operating on the raw gradient
        // This updates the raw gradient in-place with an AdaGrad-scaled one.
        if (bpinfo.enableadagrad && bpinfo.adagradwhere == modelupdateinfo::onsubgradient)
        {
            fprintf (stderr, "backpropagationmodelupdate2: AdaGrad on sub gradient =====================================\n");
            updateadagrad (bpinfo, raw_dW, raw_da, raw_dmbframes);   // accumulate AdaGrad running sums
            applyadagrad (raw_dW, raw_da, raw_dmbframes, bpinfo.adagradavdenom, *bpinfo.adagradstate);   // result -> adagraddW/a  --TODO: go in-place
            raw_dW.addweighted (0.0f, adagraddW, 1.0f);     // this is an assignment, actually  --TODO: later use a proper assign function, or even better, have updateadagrad() go directly
            raw_da.addweighted (0.0f, adagradda, 1.0f);     // or can we use swap()?
        }

        // local loop for data parallelism
        if (bpinfo.mpimasize)
        {
            fprintf (stderr, "backpropagationmodelupdate2: isfirst = %d, islast = %d\n", mpimaisfirst, mpimaislast);
            // aggregate into local-loop accumulator
            if (mpimaisfirst)
            {
                lazyenterlocalloopcomputation();
                local_W.addweighted (0.0f, W, 1.0f);        // keep a copy of our starting point so that at the end of the local loop we can know the difference
                local_a.addweighted (0.0f, a, 1.0f);
                local_mbframes = 0;
            }
            // Note: Or should we accumulate the momentum-smoothed gradient? That's the one that local-loop corrections are based on.
            // perform model update
            // This goes through momentum accumulation etc.
            local_mbframes += raw_dmbframes;
            fprintf (stderr, "backpropagationmodelupdate2: accumulating %d frames in local loop, %d so far\n", raw_dmbframes, local_mbframes);
            if (bpinfo.distributefixedcost)
                throw std::logic_error ("backpropagationmodelupdate2: model averaging is mutually exclusive with 'distributefixedcost' mode");
            backpropagationmodelupdate3 (*(rbmstatevectorsref*)nullptr, *(rbmstatevectorsref*)nullptr/*not used--ugh!*/, learningratepersample, momentumpersample, bpinfo);
            // last local update: set raw gradient to be the local-loop accumulator
            if (mpimaislast)
            {
                // W/a = locally updated model; local_W/a = model at start of local loop; raw_W/a = where the difference should go
                local_W.swap (W);       // restore model at start of the local loop; new model now in local_W/a
                local_a.swap (a);
                raw_dW.swap (local_W);  // raw gradient now contains new model (while local_W now contains dummy data)
                raw_da.swap (local_a);
                float K = bpinfo.mpiaggregator ? (float) bpinfo.mpiaggregator->nodes() : 1.0f;  // model averaging (we actually average the gradient, same thing)
                raw_dW.addweighted (1.0f/K, W, -1.0f/K);    // subtract starting model -> raw gradient is now the difference
                raw_da.addweighted (1.0f/K, a, -1.0f/K);    //div by K represents model averaging
                raw_dmbframes = local_mbframes;             // and this is the number of frames
                fprintf (stderr, "backpropagationmodelupdate2: local loop done, %d frames moved to raw gradient (with weight 1/%.2f)\n", raw_dmbframes, K);
            }
//fflush (stderr);
        }
    }

    // model update, third step:
    //  - accumulate raw gradient into smoothed one (momentum)
    //  - add smoothed gradient to model (the actual SGD model update)
    //  - maxnorm & the likes (used for ReLU)
    //  - sparsification
    // In 'distributefixedcost' mode, AdaGrad and momentum have been done already; so in that case, we just plain add the raw gradient to the model, nothing else.
    // This is the default implementation. 'linearnetwork' has its own, hence the virtual function.
    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & /*ehxs_legacy*/,  const rbmstatevectorsref & /*v_legacy*/,   // (<- these two are only used by old implementations)
                                              float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo)
    {
#if 0
        // learningrate adjustment of top layer for Relu tests (so far to no avail)
        if (type() == "perceptron")                     
            learningratepersample *= 2.0f; //1.0f / 2.0f                    // top layer: make 2 times stronger or 2 times weaker
#endif

        const size_t mbframes = raw_dmbframes;                              // frames in raw gradient

        if (!bpinfo.distributefixedcost/*done already*/)
        {
            // update AdaGrad  --when operating on the raw gradient
            if (bpinfo.enableadagrad && bpinfo.adagradwhere == modelupdateinfo::onrawgradient && !bpinfo.distributefixedcost/*done already*/)
            {
                fprintf (stderr, "backpropagationmodelupdate3: AdaGrad on raw gradient =====================================\n");
                updateadagrad (bpinfo, raw_dW, raw_da, mbframes);   // accumulate AdaGrad running sums
                applyadagrad (raw_dW, raw_da, mbframes, bpinfo.adagradavdenom, *bpinfo.adagradstate);   // result -> adagraddW/a  --TODO: go in-place
                raw_dW.addweighted (0.0f, adagraddW, 1.0f); // this is an assignment, actually  --TODO: later use a proper assign function, or even better, have updateadagrad() go directly
                raw_da.addweighted (0.0f, adagradda, 1.0f);
            }

            // add raw gradient to smoothed gradient; keep previous smoothed gradient as "momentum"
            // momentum is a 1st-order IIR low-pass, with unit gain
            // y(t+1) = momentum * y(t) + (1-momentum) * x(t)
            if (momentumpersample != 0.0f && !(bpinfo.enableadagrad && bpinfo.adagradwhere == modelupdateinfo::onsmoothedgradient) && !bpinfo.distributefixedcost/*done already*/)
            {
                const float momentum = scalemomentum (momentumpersample, mbframes); // map momentum per frame to momentum per minibatch
                updategradient (momentum/*keep weight*/, (1.0f - momentum)/*new data weight*/, false/*updateb*/, &bpinfo);
                // note: this ^^ resets raw_dmbframes to 0 to denote that it has consumed it
                //dW.dump("deltas for weight matrix");

                // update AdaGrad accumulator  --when operating on the smoothed gradient
                if (bpinfo.enableadagrad && bpinfo.adagradwhere == modelupdateinfo::onsmoothedgradient)
                    updateadagrad (bpinfo, dW, da, mbframes);

                addgradienttomodel (learningratepersample, false/*updateb*/, dW, da, db, &bpinfo, mbframes);
                //W.dump("updated weight matrix");
                //a.dump("updated bias matrix");
            }
            else    // short-cut version for no momentum--will bypass the momentum acc; this is assumed for local-loop process for data parallelism (momentum applied in local loop)
            {
                fprintf (stderr, "backpropagationmodelupdate3: straight update from raw gradient from %d frames\n", raw_dmbframes);
                addgradienttomodel (learningratepersample, false/*updateb*/, raw_dW, raw_da, raw_db, &bpinfo, raw_dmbframes);
                raw_dmbframes = 0;  // consumed
            }
        }
        else
            fprintf (stderr, "backpropagationmodelupdate3: no model update in 'distributefixedcost' mode (done elsewhere)\n");

        // MVN-SGD mean/var estimation: update mean/var with accumulator
        // BUGBUG: This happens after update() (pushtorawgradient()) (where mean/var were used to update v) but before update3() (where they are used to compensate).
        // BUGBUG: ^^ wrong
        // BUGBUG: Does this break the sequence? Also won't work when called from update1() in that special AdaGrad mode (which we can remove though)
        if (!mean.empty() && numacc > 0)
        {
            const size_t T = 24*3600*100;   // 24h of speech  --TODO: we could get this from a config parameter
            //const size_t T = 10*60*100;   // 10 min of speech, hoping to catch the explosion
            const float keepweight = (float) exp (-1.0 * numacc / T);
            const float inputscale = 1.0f / numacc; // divide frame sum by numacc to get average
            mean.addweighted (keepweight, meanacc, (1.0f - keepweight) * inputscale);
            if (!var.empty())
                var.addweighted (keepweight, varacc, (1.0f - keepweight) * inputscale);
            numacc = 0;                     // Consumed. (we don't clear the acc since numacc = 0 will notify update1() that there is nothing to add into)
        }

        // post model update operations (model fix-ups):

        // maxnorm (regL2C):  if necessary, scale down the weights according to the maxcolnorm
        if (bpinfo.regtype == regL2C && bpinfo.L2maxcolnorm != 0.0f)
        {
            if (bpinfo.L2maxcolnorm > 0.0f)         // compute the scale factors which satisfy L2maxcolnorm upper limit
            {
                // old (maxnorm) version; has several (potential) problems:
                //  - not scaling the biases
                //  - scaling the softmax layer as well--that will affect the LM factor etc., can't be right
                W.colwisenrm2 (norms, bpinfo.L2maxcolnorm);     // compute the scale factors which satisfy L2maxcolnorm upper limit
                W.scalecolwise (norms);
#if 0           // missing in original implementation
                a.scalerowwise (norms);     // bias must be scaled as well
#endif
            }
            // Frank's weird experimental alternative hack version trying to redistribute weights smoothly (enabled by negative --regparams parameter; it does not work as well for now)
            // This is hacky in that it assumes relunetwork*n:perceptron, won't work right with anything lese
            else //if (normsmbcounter < 8 || normsmbcounter % 64 == 0)            // do it only every 64th since it's not entirely cheap
            {
                static rbmbase * prev = nullptr;    // BAD HACK: we know we are called from bottom to top from dbn.h
                static size_t relulayers = 0;       // ReLU layers we scale (BAD HACK: this is not available inside here otherwise)
                static float targetnorm = 1.0f;     // goal: all layers' columns shall have this norm
                if (prev)
                {
                    fprintf (stderr, "addgradienttomodel: making good from prev (%d x %d)\n", prev->W.rows(), prev->W.cols());
                    W.unscalerowwise (prev->norms); // we make good for the scaling of our input later
                    prev = nullptr;
                }
                if (type() == "relunetwork")
                {
                    // initialize: we recover the previous 'targetnorm' from the very first layer
                    if (normsmbcounter == 0 && relulayers == 0)
                    {
                        W.colwisenrm2 (norms, -1.0f);     // compute (1/col norm * 1)
                        const float thisnorm = 1.0f / norms.absaverage();
                        const float extranormgrowth = sqrt (W.cols()/*out dim*/ / (float) W.rows()/*in dim*/);
                        targetnorm = thisnorm * sqrt (extranormgrowth);    // actual growth factor of vectors run through this layer
                        fprintf (stderr, "addgradienttomodel: ReLU normalization initialized to targetnorm %.3f\n", targetnorm);
                    }
                    // AWE (adaptive weight equalization)
                    //  - goal: for each layer, the gradient has the same relative weighting
                    //  - for now: each layer should preserve the length of its input vector  --what should its norm be?
                    //     - for square matrix, that approximately means column norm = const C (same C across all layers)
                    //     - for non-square, it means column norm C/sqrt(output dim / input dim)
                    //  - not considered so far:
                    //     - column norm C/sqrt(output dim / input dim) means gradient relatively 1/(.) too large
                    //     - impact of ReLU sparseness
                    //     - impact of dropout
                    // If matrix is not square then the norms will change with sqrt(output dim / input dim).
                    // E.g. going from 1024 to 2048, if all columns have norm 1, and the input has norm 1 as well, the output will have norm sqrt(2).
                    // Likewise, in back-prop, a 2048-dim error signal of norm 1 would be a 1024-dim vector of norm sqrt(2).
                    // Thus, we want our norm to be smaller by sqrt(2).
                    // TODO: Should our gradient also be smaller by sqrt(2), to be consistent? We could wing it somehow, I think.
                    const float extranormgrowth = sqrt (W.cols()/*out dim*/ / (float) W.rows()/*in dim*/);
                    const float thistargetnorm = targetnorm / sqrt (extranormgrowth);  // we want growth of 'targetnorm' but not of 'extranormgrowth'
                    W.colwisenrm2 (norms, -thistargetnorm);     // compute (1/col norm * thistargetnorm)
                    fprintf (stderr, "addgradienttomodel: av norm before scaling %.2f\n", thistargetnorm / norms.absaverage());
                    W.scalecolwise (norms);     // now the norm is 'targetnorm'
                    a.scalerowwise (norms);     // bias must be scaled as well
                    prev = this;                // next layer gets compensated for this change first
                    relulayers++;
                    fprintf (stderr, "addgradienttomodel: normalized this (%d x %d) to have norm %.3f (with growth factor %.2f)\n", prev->W.rows(), prev->W.cols(), thistargetnorm, extranormgrowth);
                }
                else if (type() == "perceptron")
                {
                    // at this point, all lower layers' column norms are 'targetnorm', while any scaling we applied has been pushed into the softmax layer
                    // We want some balance; that is, all including softmax should have that norm, so that gradients have the same impact.
                    // (do they? We measure the parameter norms, not the gradient norms...; at least the scaling on error and input of each layer is the same)
                    // measure actual growth factor due to softmax layer
                    W.colwisenrm2 (norms, -1.0f);                                                           // measure...
                    const float avsoftmaxnorm = 1.0f / norms.absaverage();                                  // ...the column norm of softmax layer
                    // The desired growth factor due to softmax layer is 'thistargetnorm' with
                    // thistargetnorm = newtargetnorm / extranormgrowth.
                    // But the actual is 'avsoftmaxnorm'.
                    // So we want to scale the softmax layer by (thistargetnorm/avsoftmaxnorm).
                    // I.e. the total un-scaling must aggregate to (thistargetnorm/avsoftmaxnorm).
                    // Thus, the totality of ReLU layers should be scaled by an additional (avsoftmaxnorm/thistargetnorm),
                    // i.e. each ReLU layer would be scaled by an additional (avsoftmaxnorm/thistargetnorm)^(1/#layers).
                    // In other words,
                    // newtargetnorm = targetnorm * (avsoftmaxnorm/thistargetnorm)^(1/#layers)
                    //  = targetnorm * (avsoftmaxnorm / newtargetnorm * extranormgrowth))^(1/#layers)
                    // newtargetnorm^#layers = targetnorm^#layers * avsoftmaxnorm / newtargetnorm * extranormgrowth
                    // newtargetnorm^(1+#layers) = targetnorm^#layers * avsoftmaxnorm * extranormgrowth
                    // newtargetnorm = (targetnorm^#layers * avsoftmaxnorm * extranormgrowth)^[1/(1+#layers)]
                    const float extranormgrowth = sqrt (W.cols()/*out dim*/ / (float) W.rows()/*in dim*/);
                    const float softmaxscaling = avsoftmaxnorm * sqrt (extranormgrowth);                           // actual scaling due to the softmax layer
                    const float totalscaling = pow (targetnorm, (float) relulayers) * softmaxscaling;       // actual current total scale incl. softmax
                    const float newtargetnorm = pow (totalscaling, 1.0f/(relulayers+1.0f));             // choose new target to evenly distribute this total scale
                    fprintf (stderr, "addgradienttomodel: softmax av norm before scaling %.3f (softmax); total scaling %.3f; target norm: %.3f -> %.3f\n",
                             avsoftmaxnorm, totalscaling, targetnorm, newtargetnorm);
                    targetnorm = newtargetnorm;
                    relulayers = 0;     // reset counter for next minibatch
                }
            }
            //W.dump("w2");
            // statistics
            //if (normsmbcounter < 8 || normsmbcounter % 64 == 0)            // do it only every 64th since it's not entirely cheap
            {
                rbmmodelmatrix tmpn1;
                tmpn1.resize(norms.rows(), norms.cols());  norms.getweightmatrix(tmpn1);
                size_t rows = tmpn1.rows();
                for (size_t i=0, j=0; i<rows; i++) {
                    if (tmpn1(i,0) != 1.0) {
                        fprintf(stderr, "addgradienttomodel: W : %d-th column scaled by factor %f (norm was %f) (%s, %d x %d)\n", i, tmpn1(i,0), bpinfo.L2maxcolnorm/tmpn1(i,0), type().c_str(), vdim(), hdim());
                        if (j++>10) break; //print only some first normalised numbers to get some intuition whats is going on with the weights
                    }
                }
            }
            normsmbcounter++;
        }

        // weight sparseness
        if (bpinfo.sparsethreshold > 0)                // make weights sparse
            sparsifyweights (bpinfo.sparsethreshold);
    }

    // -----------------------------------------------------------------------
    // printing (for diagnostics)
    // -----------------------------------------------------------------------

    void print() const
    {
        printmat(W);
        printmat(a);
        printmat(b);
    }

    void print (FILE *f) const
    {
        printmatfile(W,f);
        printmatfile(a,f);
        printmatfile(b,f);
    }

    // dump the layer element to stdout as mat format [v-xieche]
    void dumplayer() const
    {
        fprintf(stderr, "W:[ ");
        foreach_coord(i, j, W)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", W(i, j));
        }
        fprintf(stderr, ";]\n");

        fprintf(stderr, "a:[ ");
        foreach_coord(i, j, a)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", a(i, j));
        }
        fprintf(stderr, ";]\n");

        fprintf(stderr, "b:[ ");
        foreach_coord(i, j, b)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", b(i, j));
        }
        fprintf(stderr, ";]\n");
    }

    // print model stats
    // Returns a pair (total model params, total non-null model params).
    pair<unsigned int,unsigned int> printvaluedistribution (const string & tag) const
    {
        auto Wstats = msra::math::printmatvaluedistributionf (("W " + tag).c_str(), W);
        auto astats = msra::math::printmatvaluedistributionf (("a " + tag).c_str(), a);
        auto bstats = msra::math::printmatvaluedistributionf (("b " + tag).c_str(), b);
        return make_pair (Wstats.first + astats.first + bstats.first, Wstats.second + astats.second + bstats.second);
    }

    // -----------------------------------------------------------------------
    // code for implementing SVD
    // -----------------------------------------------------------------------

    // perform SVD
    //  - decompose W' matrix with SVD into two factors W' = M = U V' with reduced dimension
    //    W'v --> U V' v
    //    with W: (vdim x hdim) --> U: (hdim x bdim) and V': (bdim x vdim)
    //    returned as:
    //     W <- U': (bdim x hdim)
    //     V: (vdim x bdim)
    //  - this layer remains the second factor U'
    //    i.e. W <- U' (we remain a sigmoid layer)
    //  - return the first factor V, which will then be used to initialize a linear layer injected right below 'this'
    // TODO: verify the above description
    // TODO: pass proper matrix objects around, not matrices hiding in weird STL vector objects
    virtual size_t svd (std::vector<std::vector<float>> & V, float rank)
    {
        fprintf (stderr, "svd: start (%s, %d x %d)\n", type().c_str(), vdim(), hdim());
        size_t dim = W.svd (V/*buffer for Vt*/, rank);

        // variable V gets filled with V' --> transpose it to get V
        // TODO: cut to 'dim' here?
        size_t dimn = V.size();
        for (size_t j = 0; j < dimn; ++j)
            for (size_t k = 0; k < j; ++k)
                ::swap (V[j][k], V[k][j]);
        // V: (vdim x bdim)
        // note: returned still as full dimension; caller must drop columns beyond bdim

        //b.resize (dim, 1);        // TODO: what's b got to do here?? Shouldn't it be empty?  --TODO: delete this once confirmed
        b.resize (0, 0);            // no RBM for SVD models
        fprintf (stderr, "svd: done -> W'v = U V' v = (%d x %d) (%d x %d) v ; type of U: %s\n", hdim(), vdim(), dim, V.size(), type().c_str());
        return dim;
    }

    // -----------------------------------------------------------------------
    // dropout support
    // -----------------------------------------------------------------------

    // scale weight matrix; this is used to convert to/from mean-model representation used for dropout
    // note: call this inside enter/exitcomputation()
    // BUGBUG: current version will fail for explicit 'convertmodel' command due to missing entercomputation()
    virtual void dropoutscaleweights (float factor)
    {
        W.scale (factor);
    }

    // -----------------------------------------------------------------------
    // BEYOND THIS POINT is the more experimental, non-standard stuff
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // polarity flipping of sigmoids
    // Weirdo experiment by Frank to elicit seeming sparseness (most likely this is garbage):
    // What happens if we flip sigmoids from 0..1 to 1..0, changing the sparseness behaviour, while keeping the model equivalent?
    // To do that:
    //  - flip sign of input to sigmoid
    //  - compensate for that in receiving layer
    // YES, it's garbage--immediately diverges badly after 1 minibatch (but that mb has the same result, so code is OK).
    // Obviously, the gradient is much larger now (so many ones).
    // -----------------------------------------------------------------------

    // returns linear transform (mul, add) that one must now apply to undo the change
    void flippolarity (float & mul, float & add)
    {
        if (nonlinearitykind != sigmoidkind)    // only for actual sigmoids
        {
            mul = 1.0;
            add = 0.0;
            return;     // nothing done
        }
        foreach_coord (i, j, W)
            W(i,j) = -W(i,j);
        foreach_coord (i, j, a)
            a(i,j) = -a(i,j);
        // sigmoid outputs have now been flipped from 0..1 to 1..0
        // instructions to next layer how to undo this:
        mul = -1.0; // flip sign from 1..0 to -1..0
        add = 1.0;  // shift up down from -1..0 to 0..1 --> and we are back at the original values
    }

    // apply linear transform (mul, add) to input vectors
    // I.e. multiply this into (W,a)
    void applytransform (float mul, float add)
    {
        if (mul == 1.0f && add == 0.0f)  // that was easy :)
            return;
        // all inputs must be multiplied by 'mul'
        // vnew <- mul * v + add * ones
        // h = W' * vnew + a
        //   = W' * (mul * v + add * ones) + a
        //   = [W' * mul] * v + [W' * add * ones + a]
        // Wnew <- W * mul
        // anew <- W' * add * ones + a
        foreach_coord (i/*input dim*/, j/*output dim*/, W)
        {
            a[j] += W(i,j) * add;
            W(i,j) *= mul;
        }
    }

    // -----------------------------------------------------------------------
    // support for MPI-based data parallelism
    // TODO: find a more generic name for this; such as distributed cross-node aggregation?
    // -----------------------------------------------------------------------

    // link to MPI data-parallel gradient aggregation
    // Calling sequence is a little convoluted:
    //  - this is called after entercomputation() for all layers
    //  - it determines the patches of each stripe this layer, and how many bytes their quantized version would occupy in a CPU-side buffer (per stripe, aggregated over all layers)
    //  - it does NOT allocate the CPU-side buffer, since we don't know its aggregate size yet, nor GPU-side buffers
    //  - buffer allocation happens lazily upon first usage
    //  - all allocated here is torn down in exitmpiaggregation(), which is called before exitcomputation()
    void entermpiaggregation (std::vector<size_t> & mpistripebuffersizes, size_t bits)
    {
        raw_dW.entermpiaggregation (mpistripebuffersizes, bits);
        raw_da.entermpiaggregation (mpistripebuffersizes, bits);
    }

    // this tears down all MPI-related objects here
    void exitmpiaggregation()
    {
        raw_dW.exitmpiaggregation();
        raw_da.exitmpiaggregation();
    }

    // -----------------------------------------------------------------------
    // AdaGrad support
    // -----------------------------------------------------------------------

    // From Jeff Dean's talk:
    //  - learns a per-parameter learning rate
    //  - "accumulate the sum of the gradients squared"
    //  - "divide each update by the square root of that sum"
    // We have no easy access to individual frames' squares, so we will use the minibatch gradient instead.

    // TODO: move this back to data section
    // AdaGrad-related data
    rbmmodelmatrix adagraddWsqrsum; // Sum(dW_iter^2) over frames
    rbmmodelmatrix adagraddWsum; // Sum(dW_iter) over frames
    rbmmodelmatrix adagraddW;       // weighted dW
    rbmmodelmatrix adagraddasqrsum; // and likewise for da
    rbmmodelmatrix adagraddasum; // and likewise for da
    rbmmodelmatrix adagradda;
    float avdenomdW, avdenomda;     // average of avdenom; expensive to update, so updating this only every 16 calls except for first 16
    double adagradframes;           // number of frames in above sqrsum variables, with forgetting factor applied
    size_t adagradframespushed;     // processed frames (no forgetting factor)
    size_t adagradsummands;         // number of summands in above sqrsum variables --we do more for early iterations

    void enteradagrad()
    {
        // reset scalars already here so we can test them for non-zero
        avdenomdW = -1.0f;
        avdenomda = -1.0f;
        adagradframes = 0.0;
        adagradframespushed = 0;
        adagradsummands = 0;
        if (adagraddW.rows() == 0)          // if no AdaGrad then we are done
            return;

        adagraddW.entercomputation();       // this will allocate it CUDA-side
        adagraddWsqrsum.entercomputation();
        adagraddWsum.entercomputation();
        adagradda.entercomputation();
        adagraddasqrsum.entercomputation();
        adagraddasum.entercomputation();

        // reset accumulators
        adagraddWsqrsum.setzero();
        adagraddWsum.setzero();
        adagraddW.setzero();
        adagraddasqrsum.setzero();
        adagraddasum.setzero();
        adagradda.setzero();
    }

    // lazy initialization of AdaGrad; we also lazily enteradagrad() here if we just allocated it
    void lazyinitadagrad()
    {
        if (adagraddW.rows() != 0)
            return;

        fprintf (stderr, "creating AdaGrad storage\n");
        adagraddW.resize (dW.rows(), dW.cols());
        adagraddWsqrsum.resize (dW.rows(), dW.cols());
        adagradda.resize (da.rows(), da.cols());
        adagraddasqrsum.resize (da.rows(), da.cols());
#if 0   // enable this to create mean acc (so far only used to measure avdenom for diagnostics, but not in actual AdaGrad normalization, and seems problematic causing NaNs)
        adagraddWsum.resize (dW.rows(), dW.cols());
        adagraddasum.resize (da.rows(), da.cols());
#endif
        enteradagrad();
    }

    void exitadagrad()
    {
        if (adagraddW.rows() == 0)
            return;
        adagraddW.exitcomputation();
        adagraddWsqrsum.exitcomputation();
        adagraddWsum.exitcomputation();
        adagradda.exitcomputation();
        adagraddasqrsum.exitcomputation();
        adagraddasum.exitcomputation();
    }

    // -----------------------------------------------------------------------
    // functions for Hessian-free 2nd-order training
    // -----------------------------------------------------------------------

    // compute gradient from error signal and activations and add it to the accumulator
    void collectgradient (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, bool isfirstbatch, bool usedoubleaccumulator)
    {
        assert (ehxs.cols() == v.cols());  // cols = frames
        assert (!da.empty() && !dW.empty());
        
        float weight = isfirstbatch ? 0.0f : 1.0f;
        da.scaleandaddallcols (weight, ehxs, 1.0f, sumhtmp);
        dW.scaleandaddmatprod (weight, v, ehxs, 1.0f, httmp, cachedvts, cachedhts);
        if (usedoubleaccumulator)
        {
            accdW.accumulate(weight, dW, 1.0f);
            dW.setzero();
            accda.accumulate(weight, da, 1.0f);
            da.setzero();
        }
    }

    // compute gradient from error signal and activations and add it to the accumulator act, err, actsquared, errsquared, isfirstbatch, nsecondorderframes
    void collectsquaredgradient (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, rbmstatevectorsref & vsquared, rbmstatevectorsref & ehxssquared, bool isfirstbatch, bool usedoubleaccumulator)
    {
        assert (ehxs.cols() == v.cols());  // cols = frames
        assert (ehxssquared.cols() == vsquared.cols());  // cols = frames
        assert (!da.empty() && !dW.empty());
        
        // set squared elements
        vsquared.setsquare(v);
        ehxssquared.setsquare(ehxs);
        float weight = isfirstbatch ? 0.0f : 1.0f;
        dasquared.scaleandaddallcols (weight, ehxssquared, 1.0f, sumhtmp);
        dWsquared.scaleandaddmatprod (weight, vsquared, ehxssquared, 1.0f, httmp, cachedvts, cachedhts);
        if (usedoubleaccumulator)
        {
            accdWsquared.accumulate(weight, dWsquared, 1.0f);
            dWsquared.setzero();
            accdasquared.accumulate(weight, dasquared, 1.0f);
            dasquared.setzero();
        }
    }

    // compute hessian vector product from error signal and activations and add it to the accumulator
    void collecthessianvectorproduct (const rbmstatevectorsref & v, const rbmstatevectorsref & ehxs, bool isfirstbatch, size_t nsecondorderframes)
    {
        assert (ehxs.cols() == v.cols());  // cols = frames
        assert (!da.empty() && !dW.empty());
        float thisscale = isfirstbatch ? 0.0f : 1.0f;
        float otherscale = 1.0f / (float) nsecondorderframes;
        
        // bias vectors
        hessianvectorproducta.scaleandaddallcols (thisscale, ehxs, otherscale, sumhtmp);
        // the matrix
        hessianvectorproductW.scaleandaddmatprod (thisscale, v, ehxs, otherscale, httmp, cachedvts, cachedhts);
    }

    virtual void initcgfromzero(bool usepreconditioning, float nobservations, float lambda, float alpha)
    {
        cgiterateW.setzero();
        cgiteratea.setzero();
        cgresidualW.addweighted(0.0f, dW, -1.0f);
        cgresiduala.addweighted(0.0f, da, -1.0f);
        if (usepreconditioning)
        {
            setdiagonalpreconditioner(nobservations, lambda, alpha);
            pcgresidualW.elementwisedivision(cgresidualW, cgdiagonalpreconditionerW);
            pcgresiduala.elementwisedivision(cgresiduala, cgdiagonalpreconditionera);
            cgsearchdirectionW.addweighted(0.0f, pcgresidualW, -1.0f);
            cgsearchdirectiona.addweighted(0.0f, pcgresiduala, -1.0f);
        }
        else
        {
            cgsearchdirectionW.addweighted(0.0f, dW, 1.0f);
            cgsearchdirectiona.addweighted(0.0f, da, 1.0f);
        }
    }

    virtual void setdiagonalpreconditioner(float nobservations, float lambda, float alpha)
    {
        cgdiagonalpreconditionerW.setdiagonalpreconditioner(dWsquared, nobservations, lambda, alpha);
        cgdiagonalpreconditionera.setdiagonalpreconditioner(dasquared, nobservations, lambda, alpha);
    }
    
    virtual void initcg(bool usepreconditioning, float nobservations, float lambda, float alpha)
    {
        cgresidualW.addweighted(0.0f, hessianvectorproductW, 1.0f);
        cgresiduala.addweighted(0.0f, hessianvectorproducta, 1.0f);
        cgresidualW.addweighted(1.0f, dW, -1.0f);
        cgresiduala.addweighted(1.0f, da, -1.0f);
        if (usepreconditioning)
        {
            setdiagonalpreconditioner(nobservations, lambda, alpha);
            pcgresidualW.elementwisedivision(cgresidualW, cgdiagonalpreconditionerW);
            pcgresiduala.elementwisedivision(cgresiduala, cgdiagonalpreconditionera);
            cgsearchdirectionW.addweighted(0.0f, pcgresidualW, -1.0f);
            cgsearchdirectiona.addweighted(0.0f, pcgresiduala, -1.0f);
        }
        else
        {
            cgsearchdirectionW.addweighted(0.0f, dW, 1.0f);
            cgsearchdirectiona.addweighted(0.0f, da, 1.0f);
        }
    }

    virtual void inithessianfree(size_t nofbacktrackingmodels)
    {
        cgintermediateresultsW.resize(nofbacktrackingmodels,0);
        cgintermediateresultsa.resize(nofbacktrackingmodels,0);
        for (size_t i = 0; i < nofbacktrackingmodels; i++)
        {
            cgintermediateresultsW[i] = new rbmmodelmatrix();
            cgintermediateresultsa[i] = new rbmmodelmatrix();
            cgintermediateresultsW[i]->resize(W.rows(), W.cols());
            cgintermediateresultsa[i]->resize(a.rows(), a.cols());
        }
    }
    
    virtual void scalecgiterate(float scalingfactor)
    {
        cgiterateW.scale(scalingfactor);
        cgiteratea.scale(scalingfactor);
    }

    virtual void setcgsearchdirection(rbmmodelmatrix &W, rbmmodelmatrix &a) 
    {
         cgsearchdirectionW.addweighted(0.0f, W, 1.0f);
         cgsearchdirectiona.addweighted(0.0f, a, 1.0f);
    }

    virtual float calculatecgcurvatureproduct() const 
    {
        float result = 0.0f;
        result = cgsearchdirectionW.dot_mtm(hessianvectorproductW);
        result += cgsearchdirectiona.dot_mtm(hessianvectorproducta);
        return result;
    }

    virtual float calculatecgresidualnorm(bool weighted) const
    {
        float result = 0.0f;
        if (weighted)
        {
            result += cgresidualW.weighteddot_mtm(cgdiagonalpreconditionerW, cgresidualW);
            result += cgresiduala.weighteddot_mtm(cgdiagonalpreconditionera, cgresiduala);
        }
        else
        {
            result += cgresidualW.dot_mtm(cgresidualW);
            result += cgresiduala.dot_mtm(cgresiduala);
        }
        return result;
    }

    virtual float calculatepcgresidualnorm() const
    {
        float result = 0.0f;
        result += pcgresidualW.dot_mtm(cgresidualW);
        result += pcgresiduala.dot_mtm(cgresiduala);
        return result;
    }

    virtual float calculatesquaredcgiteratenorm(bool weighted) const
    {
       float result = 0.0f;
       if (weighted)
       {
           result += cgiterateW.weighteddot_mtm(cgdiagonalpreconditionerW, cgiterateW);
           result += cgiteratea.weighteddot_mtm(cgdiagonalpreconditionera, cgiteratea);
       }
       else
       {
           result += cgiterateW.dot_mtm(cgiterateW);
           result += cgiteratea.dot_mtm(cgiteratea);
       }
       return result;
    }

    virtual float calculatesquaredcgsearchdirectionnorm(bool weighted) const
    {
       float result = 0.0f;
       if (weighted)
       {
           result += cgsearchdirectionW.weighteddot_mtm(cgdiagonalpreconditionerW, cgsearchdirectionW);
           result += cgsearchdirectiona.weighteddot_mtm(cgdiagonalpreconditionera, cgsearchdirectiona);
       }
       else
       {
           result += cgsearchdirectionW.dot_mtm(cgsearchdirectionW);
           result += cgsearchdirectiona.dot_mtm(cgsearchdirectiona);
       }
       return result;
    }

    virtual float calculatesquaredparameternorm(bool weighted) const
    {
       float result = 0.0f;
       if (weighted)
       {
           result += W.weighteddot_mtm(cgdiagonalpreconditionerW, W);
           result += a.weighteddot_mtm(cgdiagonalpreconditionera, a);
       }
       else
       {
           result += W.dot_mtm(W);
           result += a.dot_mtm(a);
       }
       return result;
    }

    virtual void updatecgiterate(float stepsize)
    {
        cgiterateW.addweighted(1.0f, cgsearchdirectionW, stepsize);
        cgiteratea.addweighted(1.0f, cgsearchdirectiona, stepsize);
    }

    virtual void updatecgresidual(float stepsize)
    {
        cgresidualW.addweighted(1.0f, hessianvectorproductW, stepsize);
        cgresiduala.addweighted(1.0f, hessianvectorproducta, stepsize);
    }

    virtual void solveforpcgresidual()
    {
        pcgresidualW.elementwisedivision(cgresidualW, cgdiagonalpreconditionerW);
        pcgresiduala.elementwisedivision(cgresiduala, cgdiagonalpreconditionera);
    }

    virtual void updatecgsearchdirection(float stepsize)
    {
        cgsearchdirectionW.addweighted(stepsize, cgresidualW, -1.0f);
        cgsearchdirectiona.addweighted(stepsize, cgresiduala, -1.0f);
    }

    virtual void updatepcgsearchdirection(float stepsize)
    {
        cgsearchdirectionW.addweighted(stepsize, pcgresidualW, -1.0f);
        cgsearchdirectiona.addweighted(stepsize, pcgresiduala, -1.0f);
    }

    virtual float calculatecgresidualcgsearchdirectionproduct(bool weighted) const
    {
        float result = 0.0f;
        if (weighted)
        {
            result += cgresidualW.weighteddot_mtm(cgdiagonalpreconditionerW, cgiterateW);
            result += cgresiduala.weighteddot_mtm(cgdiagonalpreconditionera, cgiteratea);
        }
        else
        {
            result += cgresidualW.dot_mtm(cgiterateW);
            result += cgresiduala.dot_mtm(cgiteratea);
        }
        return result;
    }

    virtual void normalizegradient(size_t nobservations)
    {
        float factor = 1.0f / (float) nobservations;
        dW.scale(factor);
        da.scale(factor);
    }

    virtual void allocateaccumulators(bool usecgpreconditioning)
    {
        accdW.allocate(dW.rows(), dW.cols());
        accda.allocate(da.rows(), da.cols());
        if (usecgpreconditioning)
        {
            accdWsquared.allocate(dWsquared.rows(), dWsquared.cols());
            accdasquared.allocate(dasquared.rows(), dasquared.cols());
        }
    }

    virtual void settoaccumulator(bool usecgpreconditioning)
    {
        accdW.tomatrix(dW);
        accda.tomatrix(da);
        accdW.reset();
        accda.reset();
        if (usecgpreconditioning)
        {
            accdWsquared.tomatrix(dWsquared);
            accdasquared.tomatrix(dasquared);
            accdWsquared.reset();
            accdasquared.reset();
        }
    }

    virtual void setdummyhessianvectorproduct(float weight)
    {
        hessianvectorproductW.addweighted(0.0f, cgsearchdirectionW, weight);
        hessianvectorproducta.addweighted(0.0f, cgsearchdirectiona, weight);
    }

    virtual void setdummygradient()
    {
        dW.setvalue(1.0f);
        da.setvalue(1.0f);
    }

    virtual float calculatesquaredgradientnorm(bool weighted) const
    {
        float result = 0.0f;
        if (weighted)
        {
            result += dW.weighteddot_mtm(cgdiagonalpreconditionerW, dW);
            result += da.weighteddot_mtm(cgdiagonalpreconditionera, da);
        }
        else
        {
            result += dW.dot_mtm(dW);
            result += da.dot_mtm(da);
        }
        return result;
    }

    virtual float calculategradientcgiterateproduct(bool weighted) const 
    {
        float result = 0.0f;
        if (weighted)
        {
            result += dW.weighteddot_mtm(cgdiagonalpreconditionerW, cgiterateW);
            result += da.weighteddot_mtm(cgdiagonalpreconditionera, cgiteratea);
        }
        else
        {
            result += dW.dot_mtm(cgiterateW);
            result += da.dot_mtm(cgiteratea);
        }
        return result;
    }
    
    virtual void adddampingterm(float lambda)
    {
        hessianvectorproductW.addweighted(1.0f, cgsearchdirectionW, lambda);
        hessianvectorproducta.addweighted(1.0f, cgsearchdirectiona, lambda);
    }

    virtual void storecgiterate(size_t position)
    {
        assert(position < cgintermediateresultsW.size());
        assert(position < cgintermediateresultsa.size());
        cgintermediateresultsW[position]->addweighted(0.0f, cgiterateW, 1.0f);
        cgintermediateresultsa[position]->addweighted(0.0f, cgiteratea, 1.0f);
    }

    virtual void settointermediateresult(size_t position, float stepsize)
    {
        assert(position < cgintermediateresultsW.size());
        assert(position < cgintermediateresultsa.size());
        W.addweighted(0.0f, backupmodelW, 1.0f);
        a.addweighted(0.0f, backupmodela, 1.0f);
        W.addweighted(1.0f, *cgintermediateresultsW[position], stepsize);
        a.addweighted(1.0f, *cgintermediateresultsa[position], stepsize);
    }

    virtual void backupmodel()
    {
        backupmodelW.addweighted(0.0f, W, 1.0f);
        backupmodela.addweighted(0.0f, a, 1.0f);
    }

    virtual void restoremodel()
    {
        W.addweighted(0.0f, backupmodelW, 1.0f);
        a.addweighted(0.0f, backupmodela, 1.0f);
    }

    virtual void finalizecg(int cgiter, float cginitdecayingfactor)
    {
        dW.setzero();
        da.setzero();
        // iterate for new epoch is set to final iterate times decayingfactor (typically 0.95)
        if (cgiter == -1)
        {
            cgiterateW.scale(cginitdecayingfactor);
            cgiteratea.scale(cginitdecayingfactor);
        }
        else
        {
            cgiterateW.addweighted(0.0f, *cgintermediateresultsW[cgiter], cginitdecayingfactor);
            cgiteratea.addweighted(0.0f, *cgintermediateresultsa[cgiter], cginitdecayingfactor);
        }
        // should not be necessary, but to be sure, we set those to zero as well
        cgresidualW.setzero();
        cgresiduala.setzero();
        cgsearchdirectionW.setzero();
        cgsearchdirectiona.setzero();
        hessianvectorproductW.setzero();
        hessianvectorproducta.setzero();
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            cgintermediateresultsW[i]->setzero();
            cgintermediateresultsa[i]->setzero();
        }
        backupmodelW.setzero();
        backupmodela.setzero();
        cgdiagonalpreconditionerW.setzero();
        cgdiagonalpreconditionera.setzero();
    }

    virtual float calculatecgresidualcgiterateproduct(bool weighted) const 
    {
        float result = 0.0f;
        if (weighted)
        {
            result += cgresidualW.weighteddot_mtm(cgdiagonalpreconditionerW, cgiterateW);
            result += cgresiduala.weighteddot_mtm(cgdiagonalpreconditionera, cgiteratea);
        }
        else
        {
            result += cgresidualW.dot_mtm(cgiterateW);
            result += cgresiduala.dot_mtm(cgiteratea);
        }
        return result;
    }

    virtual float calculatecgiteratecgsearchdirectionproduct(bool weighted) const 
    {
        float result = 0.0f;
        if (weighted)
        {
            result += cgsearchdirectionW.weighteddot_mtm(cgdiagonalpreconditionerW, cgiterateW);
            result += cgsearchdirectiona.weighteddot_mtm(cgdiagonalpreconditionera, cgiteratea);
        }
        else
        {
            result += cgsearchdirectionW.dot_mtm(cgiterateW);
            result += cgsearchdirectiona.dot_mtm(cgiteratea);
        }
        return result;
    }

    // resizes all HF statistics
    void enterhessianfreeresize()
    {
        cgiterateW.resize(W.rows(), W.cols());
        cgiteratea.resize(a.rows(), a.cols());
        cgresidualW.resize(W.rows(), W.cols());
        cgresiduala.resize(a.rows(), a.cols());
        cgsearchdirectionW.resize(W.rows(), W.cols());
        cgsearchdirectiona.resize(a.rows(), a.cols());
        hessianvectorproductW.resize(W.rows(), W.cols());
        hessianvectorproducta.resize(a.rows(), a.cols());
        backupmodelW.resize(W.rows(), W.cols());
        backupmodela.resize(a.rows(), a.cols());
        pcgresidualW.resize(W.rows(), W.cols());
        pcgresiduala.resize(a.rows(), a.cols());
        cgdiagonalpreconditionerW.resize(W.rows(), W.cols());
        cgdiagonalpreconditionera.resize(a.rows(), a.cols());
        dWsquared.resize(W.rows(), W.cols());
        dasquared.resize(a.rows(), a.cols());
    }

    // syncs all HF statistics to CUDA
    void enterhessianfreesync()
    {
        cgiterateW.entercomputation();
        cgiteratea.entercomputation();
        cgresidualW.entercomputation();
        cgresiduala.entercomputation();
        cgsearchdirectionW.entercomputation();
        cgsearchdirectiona.entercomputation();
        hessianvectorproductW.entercomputation();
        hessianvectorproducta.entercomputation();
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            cgintermediateresultsW[i]->entercomputation();
            cgintermediateresultsa[i]->entercomputation();
        }
        backupmodelW.entercomputation();
        backupmodela.entercomputation();
        pcgresidualW.entercomputation();
        pcgresiduala.entercomputation();
        cgdiagonalpreconditionerW.entercomputation();
        cgdiagonalpreconditionera.entercomputation();
        dWsquared.entercomputation();
        dasquared.entercomputation();

    }

#ifdef MULTICUDA
    // syncs all HF statistics to CUDA
    void enterhessianfreesync(size_t deviceid)
    {
        cgiterateW.entercomputation(deviceid);
        cgiteratea.entercomputation(deviceid);
        cgresidualW.entercomputation(deviceid);
        cgresiduala.entercomputation(deviceid);
        cgsearchdirectionW.entercomputation(deviceid);
        cgsearchdirectiona.entercomputation(deviceid);
        hessianvectorproductW.entercomputation(deviceid);
        hessianvectorproducta.entercomputation(deviceid);
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            cgintermediateresultsW[i]->entercomputation(deviceid);
            cgintermediateresultsa[i]->entercomputation(deviceid);
        }
        backupmodelW.entercomputation(deviceid);
        backupmodela.entercomputation(deviceid);
        pcgresidualW.entercomputation(deviceid);
        pcgresiduala.entercomputation(deviceid);
        cgdiagonalpreconditionerW.entercomputation(deviceid);
        cgdiagonalpreconditionera.entercomputation(deviceid);
        dWsquared.entercomputation(deviceid);
        dasquared.entercomputation(deviceid);
    }

    // syncs all HF statistics to CUDA
    void enterhessianfreesync(std::vector<size_t> &deviceids)
    {
        cgiterateW.entercomputation(deviceids);
        cgiteratea.entercomputation(deviceids);
        cgresidualW.entercomputation(deviceids);
        cgresiduala.entercomputation(deviceids);
        cgsearchdirectionW.entercomputation(deviceids);
        cgsearchdirectiona.entercomputation(deviceids);
        hessianvectorproductW.entercomputation(deviceids);
        hessianvectorproducta.entercomputation(deviceids);
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            cgintermediateresultsW[i]->entercomputation(deviceids);
            cgintermediateresultsa[i]->entercomputation(deviceids);
        }
        backupmodelW.entercomputation(deviceids);
        backupmodela.entercomputation(deviceids);
        pcgresidualW.entercomputation(deviceids);
        pcgresiduala.entercomputation(deviceids);
        cgdiagonalpreconditionerW.entercomputation(deviceids);
        cgdiagonalpreconditionera.entercomputation(deviceids);
        dWsquared.entercomputation(deviceids);
        dasquared.entercomputation(deviceids);
    }
#endif

    // intializes all HF statistics with zero
    void enterhessianfreeinit()
    {
        cgiterateW.setzero();
        cgiteratea.setzero();
        cgresidualW.setzero();
        cgresiduala.setzero();
        cgsearchdirectionW.setzero();
        cgsearchdirectiona.setzero();
        hessianvectorproductW.setzero();
        hessianvectorproducta.setzero();
        for (size_t i = 0; i < cgintermediateresultsW.size(); i++)
        {
            cgintermediateresultsW[i]->setzero();
            cgintermediateresultsa[i]->setzero();
        }
        backupmodelW.setzero();
        backupmodela.setzero();
        pcgresidualW.setzero();
        pcgresiduala.setzero();
        cgdiagonalpreconditionerW.setzero();
        cgdiagonalpreconditionera.setzero();
        dWsquared.setzero();
        dasquared.setzero();
    }

#ifdef UNSEEN_COMPENSATION   // not used
    // used for experimental compensation of states unseen in the lattice
    void compensationupdate (const rbmstatevectorsref & vplusdv, const rbmstatevectorsref & etop, const float eps)        // [v-hansu]
    {
        const size_t mbframes = etop.cols(); assert (vplusdv.cols() == mbframes);
        dW.scaleandaddmatprod (1.0f /*feedbackweight*/, vplusdv, etop, eps, httmp, cachedvs, cachedhts);
        W.scaleandaddmatprod (1.0f /*feedbackweight*/, vplusdv, etop, eps, httmp, cachedvs, cachedhts);
    }
#endif

#if 0
    virtual void backpropagationmodelupdate (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, float sparsethreshold, const rbmbase & reflayer, float alpha)
    {
        if (deferupdate)
            throw std::logic_error ("backpropagationmodelupdate: deferupdate flag not implemented");
        const size_t mbsize = ehxs.cols(); assert (v.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

#if 1   //def NEWGRADIENTSCALING  // TODO: remove this #ifdef in the future
        const float gradientscaling = learningratepersample;    // learning rate is applied to gradients before momentum smoothing for consistency
#else   // old gradient scaling
        const float gradientscaling = 1.0f / (1.0f - momentum);                 // gradients are scaled by this
#endif
        const float inputweight = (1.0f - momentum) * gradientscaling;
        const float gradientweight = learningratepersample / gradientscaling;

        // compute the deltas; keep previous deltas as "momentum" (unless 'resetmomentum')
        // Note: smoothed gradients are scaled by 1/(1-momentum).
        updatedeltas (resetmomentum ? 0.0f : momentum, v, ehxs, inputweight, false/*updateb*/);

        // Note: (1-momentum) is to unscale the scaled smoothed gradients, see above.
        addgradienttomodel (gradientweight, false/*updateb*/, reflayer.W, reflayer.a, alpha);

        if (sparsethreshold > 0)                // make weights sparse
            sparsifyweights (sparsethreshold);
    }
#endif

    // -----------------------------------------------------------------------
    // accessors to the model parameters directly, for short-cut trainers
    // -----------------------------------------------------------------------

    rbmmodelmatrix & getdW() { return dW; }
    rbmmodelmatrix & getda() { return da; }
#ifdef COMPACTTRAINER  // get the weight matrix point in CUDA device. [v-xieche]
    msra::cuda::matrix & getcudaweight (size_t deviceid) { return W.getcudamatrix (deviceid); }
#ifdef STRIPEDTOPLAYER
    virtual msra::cuda::matrix & stripedgetcudaweight (size_t devid, size_t devnum, msra::dbn::cudadistributedmatrix::cudastriping_t s) { return W.stripedgetcudamatrix (devid, devnum, s); }
    msra::cuda::matrix & stripedgetcudabias (size_t devid, size_t devnum, msra::dbn::cudadistributedmatrix::cudastriping_t s) { return a.stripedgetcudamatrix (devid, devnum, s); }
#endif
    msra::cuda::matrix & getcudabias (size_t deviceid) { return a.getcudamatrix (deviceid); }
    msra::cuda::matrix & getcudadiffbias (size_t deviceid) { return da.getcudamatrix (deviceid); }
    msra::cuda::matrix & getcudadiffweight (size_t deviceid) { return dW.getcudamatrix (deviceid); }
    msra::cuda::matrix & getcudadiffb (size_t deviceid) { return db.getcudamatrix (deviceid); }
    rbmmodelmatrix & getweight() { return W; }
    rbmmodelmatrix & getbias() { return a; }

    // -----------------------------------------------------------------------
    // code for implementing model update operating on CUDA directly
    // -----------------------------------------------------------------------

#ifdef MULTICUDA  
    void entercomputation (int type, size_t deviceid)
    {
        W.entercomputation(deviceid); a.entercomputation(deviceid);
        b.entercomputation(deviceid);
        if (type != 0)
        {
            da.resize (a.rows(), a.cols()); // (a.cols()==1, it's a vector)
            if (!b.empty())
                db.resize (b.rows(), b.cols());
            dW.resize (W.rows(), W.cols());
        }
        if (type == -2)
        {
            enterhessianfreeresize();
            enterhessianfreesync(deviceid);
        }
        dW.entercomputation(deviceid); da.entercomputation(deviceid); db.entercomputation(deviceid);
        enteradagrad();
    }
    // hack for striped mode
    void entercomputation (int type, size_t deviceid, bool stripedflag, size_t topdevicenum)
    {
        if (!stripedflag)
            throw runtime_error ("entercomputation: should in striped flag when could this function!");
        std::vector<size_t> deviceids;
        deviceids.resize (topdevicenum);
        for (size_t i = 0; i < topdevicenum; i ++)
            deviceids[i] = deviceid + i;
        // deviceids[0] = deviceid;
        // deviceids[1] = (deviceid + 1) % msra::dbn::numcudadevices();
        W.entercomputation(deviceids); a.entercomputation(deviceids);
        b.entercomputation(deviceids);
        if (type != 0)
        {
            da.resize (a.rows(), a.cols()); // (a.cols()==1, it's a vector)
            if (!b.empty())
                db.resize (b.rows(), b.cols());
            dW.resize (W.rows(), W.cols());
        }
        if (type == -2)
        {
            enterhessianfreeresize();
            enterhessianfreesync(deviceids);
        }
        dW.entercomputation(deviceids); da.entercomputation(deviceids); db.entercomputation(deviceids);
        enteradagrad();
    }

    // hack for striped mode
    void exitcomputation (size_t deviceid, bool stripedflag, size_t topdevicenum)
    {
        if (! stripedflag)
            throw runtime_error ("entercomputation: should in striped flag when could this function!");
        std::vector<size_t> deviceids;
        deviceids.resize (topdevicenum);
        for (size_t i = 0; i < topdevicenum; i ++)
            deviceids[i] = deviceid + i;
        // deviceids[0] = deviceid;
        // deviceids[1] = (deviceid + 1) % msra::dbn::numcudadevices();
        W.exitcomputation(deviceids); a.exitcomputation(deviceids); b.exitcomputation(deviceids);
        dW.exitcomputation(deviceids); da.exitcomputation(deviceids); db.exitcomputation(deviceids);
        exitadagrad();
    }
    void exitcomputation (size_t deviceid)
    {
        W.exitcomputation(deviceid); a.exitcomputation(deviceid); b.exitcomputation(deviceid);
        dW.exitcomputation(deviceid); da.exitcomputation(deviceid); db.exitcomputation(deviceid);
        exitadagrad();
    }
#endif

    virtual void backpropagationmodelupdateincuda (const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                                   float learningratepersample, double momentumpersample, bool resetmomentum, /*const --fix this*/ modelupdateinfo & bpinfo, size_t deviceid)
    {
        const size_t mbsize = ehxs.cols(); assert (v.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

        const float gradientscaling = learningratepersample;    // learning rate is applied to gradients before momentum smoothing for consistency
        static bool f = false;
        if (!f)
        {
            f = true;
            fprintf (stderr, "backpropagationmodelupdate: new gradient scaling (by learning rate) enabled\n");
        }
        const float inputweight = (1.0f - momentum) * gradientscaling;
        const float gradientweight = gradientscaling<1e-30? 1:learningratepersample / gradientscaling;

        // compute the deltas; keep previous deltas as "momentum" (unless 'resetmomentum')
        // Note: smoothed gradients are scaled by 1/(1-momentum).
        auto &daref = da.getcudamatrix (deviceid);
        auto &dWref = dW.getcudamatrix (deviceid);
        auto &vref = v.getcudamatrix (deviceid);
        auto &ehxsref = ehxs.getcudamatrix (deviceid);

        daref.addrowsum (resetmomentum ? 0.0f : momentum, ehxsref, inputweight);
        dWref.gemm (resetmomentum ? 0.0f : momentum, vref, false, ehxsref, true, inputweight);

        auto &Wref = W.getcudamatrix (deviceid);
        auto &aref = a.getcudamatrix (deviceid);
        Wref.gems (1.0f, dWref, gradientweight);
        aref.gems (1.0f, daref, gradientweight);

        // Note: (1-momentum) is to unscale the scaled smoothed gradients, see above.
        if (bpinfo.sparsethreshold > 0)                // make weights sparse
            Wref.setto0ifabsbelow (bpinfo.sparsethreshold);
    }

#ifdef STRIPEDTOPLAYER
    virtual void backpropagationmodelupdatestripedincuda(const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                                         float learningratepersample, double momentumpersample, bool resetmomentum, modelupdateinfo & bpinfo, size_t deviceid, size_t devnumusedintoplayer = 2)
    {
        const size_t mbsize = ehxs.cols(); assert (v.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

        const float gradientscaling = learningratepersample;    // learning rate is applied to gradients before momentum smoothing for consistency
        static bool f = false;
        if (!f)
        {
            f = true;
            fprintf (stderr, "backpropagationmodelupdate: new gradient scaling (by learning rate) enabled\n");
        }
        const float inputweight = (1.0f - momentum) * gradientscaling;
        const float gradientweight = gradientscaling<1e-30? 1:learningratepersample / gradientscaling;

        // compute the deltas; keep previous deltas as "momentum" (unless 'resetmomentum')
        // Note: smoothed gradients are scaled by 1/(1-momentum).

        for (size_t devindex = deviceid; devindex < deviceid + devnumusedintoplayer; devindex ++)
        {
            size_t thisdevid = devindex % numcudadevices();
            auto &daref = da.stripedgetcudamatrix (thisdevid, devnumusedintoplayer, msra::dbn::cudadistributedmatrix::stripedwrtrows);
            auto &dWref = dW.stripedgetcudamatrix (thisdevid, devnumusedintoplayer, msra::dbn::cudadistributedmatrix::stripedwrtcols);
            auto &vref = v.getcudamatrix (thisdevid);
            auto &ehxsref = ehxs.stripedgetcudamatrix (thisdevid, devnumusedintoplayer, msra::dbn::cudadistributedmatrix::stripedwrtrows);

            daref.addrowsum (resetmomentum ? 0.0f : momentum, ehxsref, inputweight);
            dWref.gemm (resetmomentum ? 0.0f : momentum, vref, false, ehxsref, true, inputweight);

            auto &Wref = W.stripedgetcudamatrix (thisdevid, devnumusedintoplayer, msra::dbn::cudadistributedmatrix::stripedwrtcols);
            auto &aref = a.stripedgetcudamatrix (thisdevid, devnumusedintoplayer, msra::dbn::cudadistributedmatrix::stripedwrtrows);
            Wref.gems (1.0f, dWref, gradientweight);
            aref.gems (1.0f, daref, gradientweight);

            // Note: (1-momentum) is to unscale the scaled smoothed gradients, see above.
            if (bpinfo.sparsethreshold > 0)                // make weights sparse
                Wref.setto0ifabsbelow (bpinfo.sparsethreshold);
        }
    }
#endif
#endif
    // get weight matrix from W. [v-xieche]
    template <class AType> void getweightmatrix (AType & weightbuf) { W.getweightmatrix (weightbuf); }

    // assign weight matrix to W. [v-xieche]
    template <class AType> void assignweightmatrix (AType & weightbuf) { W.assignweightmatrix (weightbuf); }

    // get the m-th elemeent of the bias [v-xieche]
    float getbiasvalue(size_t m) const { return a[m]; }

    // -----------------------------------------------------------------------
    // support for miscellaneous hack experiments
    // -----------------------------------------------------------------------

#ifdef STEEPERSIGMOID
    // multiply the W with n. used for temp experiment to see what happend when sigmoid become steeper. [v-xieche]
    void multiplywith (float n)
    {
        W.multiplywith (n);
#ifdef SCALEBIASFORSS
        a.multiplywith (n);
#endif
    }
#endif

    virtual void setdeltas (const rmbmodelmatrix & otherdW, const rmbmodelmatrix & otherda)
    {
        assert (!da.empty() && !dW.empty());
        assert (otherdW.rows() == dW.rows());
        assert (otherdW.cols() == dW.cols());
        assert (otherda.rows() == da.rows());
        assert (otherda.cols() == da.cols());

        dW.addweighted (0.0f, otherdW, 1.0f);
        da.addweighted (0.0f, otherda, 1.0f);
    }

    // split (double-up) nodes
    // 'out' true means double the output nodes, else double the input nodes.
    // This is only defined for the forward direction. b is just updated in terms of dimension.
    // So far identical for all nodes. Change to virtual if not.
    void doublenodes (bool out)
    {
        srand ((unsigned int) W.rows());
        // double output nodes: perturb a little
        if (out)
        {
            // h = W' v + a
            const size_t hdim = W.cols();
            matrix newW (W.rows(), 2 * hdim);   // double out dim
            vector newa (2 * hdim);
            foreach_column (j, W)
            {
                // Note: 'eps' might have to depend on the actual scale of the column,
                // which impacts the slope of the sigmoid. Now I just try to make it 'small'.
                foreach_row (i, W)
                {
                    const float eps = ::rand() * 0.01f / RAND_MAX;
                    newW(i,j)        = W(i,j) + eps;
                    newW(i,j + hdim) = W(i,j) - eps;
                }
                newa[j]        = a[j];
                newa[j + hdim] = a[j];
            }
            W = std::move (newW);
            a = std::move (newa);
        }
        // double input nodes: half the weights
        // Weights are halved because each input now exists twice (except small perturbance)
        else
        {
            // v = W h + b
            const size_t vdim = W.rows();
            matrix newW (2 * vdim, W.cols());
            vector newb (2 * vdim);
            foreach_row (i, W)
            {
                foreach_column (j, W)
                {
                    newW(i,j)        = W(i,j) * 0.5f;
                    newW(i + vdim,j) = W(i,j) * 0.5f;
                }
                // b is only updated as a formality; but it has no meaning, don't use it
                if (!b.empty())
                {
                    newb[i]        = b[i];
                    newb[i + vdim] = b[i];
                }
            }
            W = std::move (newW);
            if (!b.empty())
                b = std::move (newb);
        }
    }

    virtual size_t getnumberofweightsets() const { return 1; }
    virtual pair<size_t, size_t> getweightsetdims(const size_t weightsetindex) const 
    {
        assert (weightsetindex < getnumberofweightsets());

        pair<size_t, size_t> p(W.rows(), W.cols());
        return p;
    }

    // set weights (this is to support hack experiments)
    template<class WTYPE, class ATYPE>
    void setweights (const WTYPE & newW, const ATYPE & newa, const size_t weightsetindex)
    {
        assert (newW.rows() == W.rows() && newW.cols() == W.cols());
        assert ((size_t) newa.size() == a.rows());
        assert (weightsetindex < getnumberofweightsets());

        foreach_coord (i, j, W)
            W(i,j) = (float) newW(i,j);
        foreach_index (i, newa)
            a[i] = (float) newa[i];
    }

    // force sparseness to parameters
    void sparsifyweights (float threshold) { W.setto0ifabsbelow (threshold); }
    void dumplayerpart() const                          //added by Hang Su adaptation
    {
        FILE *filetowrite = fopen ("dumpedlayer.txt" , "w");
        fprintfOrDie(filetowrite, "W: [ ");
        fflush(filetowrite);
        for (size_t j = 0; j < 1/*W.cols()*/; j++)
        {
            for (size_t i = 0; i < W.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", W(i, j));
            }
            fprintf(filetowrite, ";]\n");
            fflush(filetowrite);
        }
        for (size_t j = W.cols()/2; j < W.cols()/2+1; j++)
        {
            for (size_t i = 0; i < W.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", W(i, j));
            }
            fprintf(filetowrite, ";]\n");
            fflush(filetowrite);
        }
        fflush(filetowrite);
        fprintf(filetowrite, "a: [ ");
        for (size_t j = 0; j < a.cols(); j++)
        {
            for (size_t i = 0; i < a.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", a(i, j));
            }
            fprintf(filetowrite, ";]\n");
        }
        fflush(filetowrite);
        fclose(filetowrite);
    }
};

// ===========================================================================
// RBM -- implementation of a Restricted Boltzman Machine
// ===========================================================================

class rbm : public rbmbase
{
public:
#if 0
    rbm (matrix && pW, vector && pa, vector && pb)
    {
        W = std::move (pW);
        a = std::move (pa);
        b = std::move (pb);
    }
#endif
    rbm (size_t vdim, size_t hdim, nonlinearitykind_t nlkind, unsigned int randomseed)
    {
        fprintf (stderr, "rbm: instantiating layer with %s nonlinearity of dimension %d x %d\n", nonlinearitykindtostring (nlkind), vdim, hdim);
        nonlinearitykind = nlkind;      // overwrite the kind (rbmbase initializes it as sigmoidkind)
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        b.resize (vdim, 1);
        validatedims();
        initrandom (randomseed);
    }
    template<typename FILEHANDLETYPE>
    rbm (FILEHANDLETYPE f) : rbmbase (f) { validatedims(); }

protected:

    // perform CD-1 statistics for pretraining (mostly identical for Gauss and Bernoulli case)
    // Eh is the probability for binary h value being 1
    void pretrainingcd1 (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed, bool dovsigmoid) const
    {
        // randomly sample binary h according to distribution (binary)
        // We use h1 temporarily as a buffer (will be overwritten below).
        // This takes ~7% of the runtime.
        h1.samplebinary (Eh, randomseed);

        // reconstruct v = W h + b  (use probability per Section 3.2 of guideTR.pdf)
        htov (h1, v1);

        // and v <- sigmoid (v) except for first (Gaussian) layer
        if (nonlinearitykind != sigmoidkind)
            throw runtime_error ("pretrainingcd1: non-sigmoid units not implemented");
        if (dovsigmoid)
            v1.sigmoid();

        // compute output probabilities h = sigmoid (v' W + a)
        forwardprop (v1, h1);
    }


    /* steps for pretraining
    # steps: (see Section 2 in guideTR.pdf)
    #  - normal forward propagation
    #  - add the current h, v, and v h' to deltas (positive term in Eq. (5) in guideTR.pdf)
    #    also apply momentum to current deltas
    #  - randomly sample binary h according to distribution (binary)
    #  - reconstruct v = W h + a  (use probability per Section 3.2 of guideTR.pdf)
    #    and v <- sigmoid (v) except for first (Gaussian) layer
    #  - compute output probabilities h = sigmoid (v' W + b)
    #  - subtract the new h, v, and v h' from deltas (negative term)
    */
    // update models for pretraining
    // ('virtual' only because top layer does not implement this--it overrides this to throw)
    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample)
    {
        const size_t mbsize = v.cols(); assert (h.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

        // note that learning rate is applied to gradients before momentum smoothing for consistency with old code
        // TODO: fix this by just passing 'learningratepersample' to addgradienttomodel() instead (but fix this only when I have a chance to test it)
        const float inputweight = (1.0f - momentum) * learningratepersample;

        // compute the deltas; keep previous deltas as "momentum"

        // We compute this with a little trick:
        //  dX = (momentum * dX) + Xpos - Xneg
        //     = -((-momentum * dX) + Xneg) + Xpos

        //checknan (v1); checknan (h1);

        // the negative summand
        pushtorawgradient (h1, v1, true/*updateb*/, /*operate on these:*/raw_dW, raw_da, raw_db, raw_dmbframes, NULL);
        updategradient (-momentum, inputweight, true/*updateb*/, NULL);

        // the positive summand
        pushtorawgradient (h, v, true/*updateb*/, /*operate on these:*/raw_dW, raw_da, raw_db, raw_dmbframes, NULL);
        updategradient (-1.0f, inputweight, true/*updateb*/, NULL);

        // and update the model
        addgradienttomodel (1.0f/*learningratepersample, already applied above*/, true/*updateb*/, dW, da, db, NULL, 0);
    }

    // computation of deltavs for unseen state compensation
    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnormsbuf,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        assert (deltah.cols() == h.cols() && deltah.rows() == h.rows());
        assert (eh.cols() == h.cols()     && eh.rows() == h.rows());
        assert (deltav.empty() || (deltav.cols() == v.cols() && deltav.rows() == v.rows()));
        assert (deltah.cols() == v.cols());
        assert (vnormsbuf.rows() == 1 && vnormsbuf.cols() == deltah.cols()); // row vector

        // deltah = h .* (1-h) .* [deltaW' * v + W' * deltav + deltaa]
        //        = h .* (1-h) .* [h .* (1-h) .* eps * e * v' * v + W' * deltav + h .* (1-h) .* eps * e]
        // deltah = h .* (1-h) .* [h .* (1-h) .* eps * e * (v' * v + 1) + W' * deltav]
        // matrix-wise deltaH = H .* (1-H) .* [H .* (1-H) .* eps * E * (diag(V' * V) + I) + W' * deltaV]
        // where
        //  deltaW = eps * h .* (1-h) .* e * v'
        //  deltaa = eps * h .* (1-h) .* e
        // and eps chosen to be the same as in the actual model update

        const size_t mbsize = deltah.cols();
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change
        const float eps = (1.0f - momentum) * learningrateperframe;

        v.columnnormsquares (vnormsbuf);                // vnormsbuf <- diag(V' * V)    (stored as a vector)

        const float addconst = 1.0f;                    // set to 0 if not compensating for 'a'
        deltah.scaledcolumns (eh, vnormsbuf, addconst); // deltah <- eh * vnormsbuf = H .* (1-H) .* E * (diag(V' * V) + I)
        // note: errorbackprop() already factored H .* (1-H) into E, as an in-place update.

        deltah.scale (eps);                             // deltah <- eps * deltah = H .* (1-H) .* eps * E * (diag(V' * V) + I)

        if (!deltav.empty())                            // empty means pretend it is 0
            W.addmatprod_mtm (deltav, deltah);          // deltah += W' * deltav

        if (nonlinearitykind != sigmoidkind)
            throw runtime_error ("forwardpropdelta: non-sigmoid units not implemented");
        deltah.mulbydsigm (deltah);                     // deltah <- h .* (1-h) .* deltah

        return eps;
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
//        dumplayerpart();            // do a check
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        numstream = blowupfactor;
        vdimnumroundup = W.colstride() - wrowsori;
        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        foreach_coord(i,j,W)  Wbackup(i,j) = W(i,j);
        const size_t wrowblowed = (wrowsori + vdimnumroundup) * blowupfactor - vdimnumroundup;  // the last stream does not need round up
        const size_t wcolblowed = wcolsori * blowupfactor;
        W.resize(wrowblowed, wcolblowed);
        foreach_coord(i,j,W)  W(i,j) = 0;

        const size_t arowsori = a.rows();
        const size_t acolsori = a.cols();
        rmbmodelmatrix abackup;
        abackup.resize (arowsori , acolsori);
        foreach_coord(i,j,a)  abackup(i,j) = a(i,j);
        a.resize( arowsori * blowupfactor , acolsori );

        const size_t browsori = b.rows();
        const size_t bcolsori = b.cols();
        rmbmodelmatrix bbackup;
        bbackup.resize (browsori , bcolsori);
        foreach_coord(i,j,b)  bbackup(i,j) = b(i,j);
        b.resize( (browsori + vdimnumroundup) * blowupfactor - vdimnumroundup, bcolsori);
        foreach_coord(i,j,b)   b(i,j) = 0;

        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for (size_t i = 0 ; i < wrowsori; i++)
            {
                for (size_t j = 0 ; j < wcolsori; j++)
                    W(i + blockindex * (vdimnumroundup + wrowsori),j + blockindex * wcolsori) = Wbackup(i,j);
            }
            for (size_t i = 0; i < arowsori; i++ )
                a(i + blockindex * arowsori, 0) = abackup(i, 0);
            for (size_t i = 0; i < browsori; i++ )
                b(i + blockindex * browsori, 0) = bbackup(i, 0);
        }
        validatedims();
//        dumplayerpart();            //do a check
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: rbm layer shall not use this function with statemap");
    }
};

// ===========================================================================
// rbmgaussbernoulli -- RBM with continuous input (Gaussian)
// ===========================================================================

class rbmgaussbernoulli : public rbm
{
public:
    //rbmgaussbernoulli (matrix && W, vector && a, vector && b) : rbm (std::move (W), std::move (a), std::move (b)) {}
    rbmgaussbernoulli (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : rbm (vdim, hdim, sigmoidkind, randomseed) {}
    template<typename FILEHANDLETYPE>
    rbmgaussbernoulli (FILEHANDLETYPE f) : rbm (f) { }
    rbmgaussbernoulli (const HANDLE f) : rbm (f) {}
    //virtual rbmbase * clone() const { return new rbmgaussbernoulli (*this); }
    virtual string type() const { return "rbmgaussbernoulli"; }

    virtual void copyfrom (const Iannlayer & other) { rbm::copyfrom (other); }

    // perform CD-1 statistics for pretraining
    // Eh is the probability for binary h value being 1
    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        pretrainingcd1 (Eh, v1, h1, randomseed, false/*no sigmoid for v for Gauss*/);
    }

#if 0
    // TODO: these should be moved to the base class [fseide]
    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }
#endif

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }
};

// ===========================================================================
// rbmbernoullibernoulli -- RBM with binary input
// ===========================================================================

class rbmbernoullibernoulli : public rbm
{
public:
    //rbmbernoullibernoulli (matrix && W, vector && a, vector && b) : rbm (std::move (W), std::move (a), std::move (b)) {}
    rbmbernoullibernoulli (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : rbm (vdim, hdim, sigmoidkind, randomseed) {}
    template<typename FILEHANDLETYPE>
    rbmbernoullibernoulli (FILEHANDLETYPE f) : rbm (f) { }

    // construct as the first factor of an SVD-decomposed layer (the first one is the linear one)
    // TODO: This should be something different, a different model type. It seems broken to initialize it here.
    rbmbernoullibernoulli (const std::vector<std::vector<float>> &v, size_t dimm/*rows*/, size_t dimn/*cols*/)
      : rbm (dimm, dimn, linearkind/*first factor is linear*/, 0/*randomseed, irrelevant since we overwrite it*/)
    {
        W.setfrom (v);
        initbiaszero();
        b.resize (0, 0);    // not an RBM actually  --TODO: this whole thing should be of type 'linearnetwork', really
    }
    //virtual rbmbase * clone() const { return new rbmbernoullibernoulli (*this); }
    virtual string type() const { return "rbmbernoullibernoulli"; }

    virtual void copyfrom (const Iannlayer & other) { rbm::copyfrom (other); }

    // back-compat hack, used to read old-format IPE-compatible SVD file
    template<typename FILEHANDLETYPE> rbmbernoullibernoulli (FILEHANDLETYPE f, bool islinearoverride) : rbm (f)
    {
        if (islinearoverride)                       // HACK: for old IPE model files which knew they were linear from a hacked model type string at load time
        {
          //  fprintf (stderr, "rbmbernoullibernoulli: non-linearity overwritten to 'linearkind'\n");
            nonlinearitykind = linearkind;
        }
    }

    // back-compat hack, helper function used to write old-format IPE-compatible SVD file
    template<typename FILEHANDLETYPE> void writeassuminglinearkind (FILEHANDLETYPE f) const
    {
        W.write (f, "W"); a.write (f, "a"); b.write (f, "b");
    }

    // perform CD-1 statistics for pretraining
    // Eh is the probability for binary h value being 1
    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        pretrainingcd1 (Eh, v1, h1, randomseed, true/*sigmoid for v for Bernoulli*/);
    }

#if 0
    // TODO: these should be moved to the base class [fseide]
    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }
#endif

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }
};

// ===========================================================================
// noptnetwork -- network layer with a certain non-linearity for which we don't support pre-training
// Very simple class: An 'rbm' with nonlinearity overridden to the 'nonlinearitykind' template argument.
// ===========================================================================

template<nonlinearitykind_t NLKIND> class noptnetwork : public rbm
{
protected:
    // constructors: from scratch and from file  --these are protected since this is a base class that lacks the type() override
    noptnetwork (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : rbm (vdim, hdim, NLKIND, randomseed)
    {
        config.dump();                   // (for debugging; TODO: remove this once we see it working)
    }
    template<typename FILEHANDLETYPE> noptnetwork (FILEHANDLETYPE f) : rbm (f)
    {
        if (nonlinearitykind != NLKIND)
            throw std::logic_error ("noptnetwork: unexpectedly failed to read nonlinearitykind from file");
    }

    virtual void copyfrom (const Iannlayer & other) { rbm::copyfrom (other); }

    virtual void pretrainingstats (const rbmstatevectorsref &, rbmstatevectorsref &, rbmstatevectorsref &, unsigned int) const { throw std::logic_error ("pretrainingstats: not implemented for noptnetwork layers for now"); }
    virtual void pretrainingmodelupdate (const rbmstatevectorsref &, const rbmstatevectorsref &, rbmstatevectorsref &, rbmstatevectorsref &, float, double) { throw std::logic_error ("pretrainingmodelupdate: not implemented for noptnetwork layers for now"); }
};

// ===========================================================================
// relunetwork -- network layer with rectified linear non-linearity
// Even simpler class: A 'noptnetwork<relukind>'.
// ===========================================================================

class relunetwork : public noptnetwork<relukind>
{
public:
    relunetwork (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : noptnetwork (vdim, hdim, config, randomseed) { }
    template<typename FILEHANDLETYPE> relunetwork (FILEHANDLETYPE f) : noptnetwork (f) { }
protected:
    virtual string type() const { return "relunetwork"; }
};

// ===========================================================================
// softplusnetwork -- network layer with rectified linear non-linearity
// Even simpler class: A 'noptnetwork<softpluskind>'.
// ===========================================================================

class softplusnetwork : public noptnetwork<softpluskind>
{
public:
    softplusnetwork (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : noptnetwork (vdim, hdim, config, randomseed) { }
    template<typename FILEHANDLETYPE> softplusnetwork (FILEHANDLETYPE f) : noptnetwork (f) { }
protected:
    virtual string type() const { return "softplusnetwork"; }
};

// ===========================================================================
// leakyrootnetwork -- network layer with leaky-rectified root-like non-linearity
// Very simple class: An 'rbm' with nonlinearity overridden to leakyrootkind.
// ===========================================================================

// TODO: remove this, neither leaky nor root helps, and plain RELUs are covered by the base class already

class leakyrootnetwork : public rbm
{
    size_t rootorder;       // e.g. 5 for 5-th root; or 1 for leaky relu
    float leakiness;        // e.g. 0.01
public:
    // constructors: from scratch and from file
    leakyrootnetwork (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : rbm (vdim, hdim, leakyrootkind, randomseed)
    {
        config.dump();                      // (for debugging; TODO: remove this once we see it working)
        rootorder = config("rootorder");
        leakiness = config("leakiness");
    }
    template<typename FILEHANDLETYPE> leakyrootnetwork (FILEHANDLETYPE f) : rbm (f)
    {
        if (nonlinearitykind != leakyrootkind)
            throw std::logic_error ("leakyrootnetwork: unexpectedly failed to read nonlinearitykind from file");
        fcheckTag (f, "BLRN");
        rootorder = fgetint (f);
        leakiness = fgetfloat (f);
        fcheckTag (f, "ELRN");
    }

    // overridden so we can write extra information
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        rbmbase::write (f);
        fputTag (f, "BLRN");
        fputint (f, (int) rootorder);
        fputfloat (f, leakiness);
        fputTag (f, "ELRN");
    }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    virtual void copyfrom (const Iannlayer & iother)
    {
        throw std::logic_error ("copyfrom: untested"); 
        //const auto & other = dynamic_cast<const decltype(*this) &> (iother);
        const leakyrootnetwork & other = dynamic_cast<const leakyrootnetwork &> (iother);
        rbm::copyfrom (other);
        rootorder = other.rootorder;
        leakiness = other.leakiness;
    }

protected:

    virtual string type() const { return "leakyrootnetwork"; }

    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly = false, rbmstatevectorsref * pmask = NULL) const
    {
        vtoz (v, Eh);   // z = W' v + a

        if (pmask)
            throw std::logic_error ("forwardprop: non-null mask is not implemented yet for non-softmax layer");

        Eh.leakyroot (rootorder, leakiness);
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        assert (eh.cols() == h.cols() && eh.rows() == h.rows());

        // compute 'dW' = [ dh/d(w(i,j)) ] and 'da' = [ dh/d(a[i]) ]
        // err = eh .* derivative of non-linearity computed from its output value
        // update 'eh' in place for later use in accumulation

        // multiply by derivative
        eh.mulbydleakyroot (h, rootorder, leakiness);

        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh  (eh is the updated one)
            //ev.dump("error signal");
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: not implemented for leakyrootnetworks for now");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: not implemented for leakyrootnetworks for now");
    }

    // computation of deltavs for unseen state compensation [v-hansu]
    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        throw::logic_error ("forwardpropdelta: not implemented for leakyrootnetworks");
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: relunetwork layer shall not use this function with statemap");
    }

    virtual void blowup(const size_t blowupfactor)
    {
        throw::logic_error ("blowup: not implemented for leakyrootnetworks");
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented for leakyrootnetworks");
    }
};

// ===========================================================================
// perceptron -- for top layer
// A softmax classifier, atually (bad name--a perceptron has no non-linearity)
// ===========================================================================

class perceptron : public rbmbase
{
public:
#if 0
    perceptron (matrix && pW, vector && pa)
    {
        W = std::move (pW);
        a = std::move (pa);
        validatedims();
    }
#endif
    perceptron (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed)
    {
        nonlinearitykind = softmaxkind;     // overwrite the kind (rbmbase initializes it as sigmoidkind)
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        validatedims();
        initrandom (randomseed);
    }
    template<typename FILEHANDLETYPE>
    perceptron (FILEHANDLETYPE f) : rbmbase (f)
    {
        if (nonlinearitykind != softmaxkind)     // compat: old files do not store the nonlinearitykind
        {
          //  fprintf (stderr, "perceptron: nonlinearitykind (%d) not read from file, assuming legacy file format and set to softmaxkind (%d)\n", nonlinearitykind, softmaxkind);
            nonlinearitykind = softmaxkind;
        }
        validatedims();
    }

    virtual string type() const { return "perceptron"; }

    virtual void copyfrom (const Iannlayer & other) { rbmbase::copyfrom (other); }

    // only keep the top layer that falls into [startoutputid, endoutputid)
    // This is for multi-lingual learning.
    void shrink (const size_t startoutputid, const size_t endoutputid)
    {
        if (startoutputid<0 || endoutputid<0 || endoutputid<=startoutputid || endoutputid>hdim())
            malformed ("shrink: must have hdim >= endoutputid > startoutputid >=0");

        matrix oldW = W.peek();

        W.resize(vdim(), endoutputid - startoutputid);
        foreach_coord(i, j, W)
            W(i,j) = oldW(i, startoutputid+j);

        matrix olda = a.peek();

        a.resize(endoutputid - startoutputid, 1);
        foreach_row(i, a)
            a(i,0) = olda(startoutputid+i, 0);
    }

#if 0   // now done in base class
    // forward propagation
    // mask is added at the activation (i.e., before sigmoid and thus a large negative values means masked and 0 means not)
    // If not for the mask, we could just leave out this function; the base class knows the non-linearity anyway.
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Pu, const bool linearonly, rbmstatevectorsref * pmask = NULL) const
    {
        //W.dump("weight matrix");
        vtoz (v, Pu);       // h = W' v + a
        if (pmask)
            Pu.addweighted (*pmask);
        if (!linearonly)    // we do not need actual softmax when doing lattice-based sequence training
            Pu.softmax();
        //Pu.dump("softmax layer");
    }
#endif

    // the rest below is non-standard stuff

    // forward propagation
    void forwardpropwithoutbias (const rbmstatevectorsref & v, rbmstatevectorsref & Pu) const
    {
        evtoeh (v, Pu);       // h = W' v
    }

    // this computes only component i of the output vector
    // Special function supposed to be used for on-demand LL evaluation.
    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const
    {
        vtoz (v, u, i);   // u = w_i' v + a_i
    }

    // forward propagates hessian vector product statistics
    // layerIn: input to layer
    // layerOut: output of layer
    // forwardstatisticsIn: statistics of previous layer
    // forwardstatisticsOut: resulting statistics
    virtual void forwardprophessianvectorproduct(const rbmstatevectorsref &layerin, const rbmstatevectorsref &layerout,         
                                                 const rbmstatevectorsref &forwardstatisticsin, rbmstatevectorsref &forwardstatisticsout, bool zeroforwardstatisticsin) const
    {
        if (!zeroforwardstatisticsin)
            W.matprod_mtm(forwardstatisticsin, cachedWs, cachedforwardstatisticsin, forwardstatisticsout, cachedforwardstatisticsout);

        // forwardstatisticsout += gradientW^T * layerin + gradienta
        float weight = zeroforwardstatisticsin ? 0.0f : 1.0f;
        cgsearchdirectionW.matprod_mtm(layerin, cachedsearchdirectionW, cachedvs, forwardstatisticsout, cachedforwardstatisticsout, cgsearchdirectiona, cachedsearchdirectiona, weight);
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: cannot be called on top layer");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: cannot be called on top layer");
    }

#if 0   // (covered by base class--we can delete this)
    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        h;  // not used here
        // and for this type of model, 'eh' does not get modified
        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        // BUGBUG: why do we need to check for empty? The bottom level is never a perceptron!
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh
        //ev.dump("error propogation signal");
    }
#endif

    // computation of deltavs for unseen state compensation [v-hansu]
    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        throw::logic_error ("forwardpropdelta: perceptron's forwardpropdelta() shall not be called, please check");
    }

    virtual void blowup(const size_t blowupfactor)
    {
        throw std::logic_error ("blowup: perceptron layer shall not use this function without statemap");
    }

    void blowup (const size_t blowupfactor, const std::vector<size_t> & statemap)        // added by Hang Su adaptation, not completed, shall make use of state mapping
    {
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        vdimnumroundup = 0;
        numstream = blowupfactor;
        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        for( size_t j = 0 ; j < W.cols(); j++)
        {
            for ( size_t i = 0 ; i < W.rows(); i++)
            {
                Wbackup(i,j) = W(i,j);
            }
        }
        W.resize( wrowsori * blowupfactor , wcolsori);

        const size_t browsori = b.rows();
        const size_t bcolsori = b.cols();
        b.resize( browsori * blowupfactor , bcolsori);

        // there is no need to blow up "a"
        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for (size_t j = 0 ; j < wcolsori; j++)
            {
                for (size_t i = 0 ; i < wrowsori; i++)
                {
                    if( statemap[j] == blockindex )
                        W(i + blockindex * wrowsori,j) = Wbackup(i,j);
                    else
                        W(i + blockindex * wrowsori,j) = 0;
                }
            }
            for (size_t i = 0; i < browsori; i++)
                b(i + blockindex*browsori , 0) = 0;
        }
    }
};

// ===========================================================================
// mvn -- mean/variance normalization layer
// Based on Simon Wiesler's idea (ICASSP 2014), but doing it more directly.
// ===========================================================================

class mvn : public Iannlayer
{
    rmbmodelvector mean;
    rmbmodelvector var;         // actually the diagonal of a matrix
    size_t T;                   // time constant
    rmbmodelvector meanacc;
    rmbmodelvector varacc;
    size_t numacc;
public:
    void validatedims() const { }
    mvn (size_t vdim, size_t hdim, const layerconfigparameters & config)
    {
        if (vdim != hdim)
            throw std::runtime_error ("mvn: MVN layer must have identical input and output dimension");
        mean.resize (hdim, 1);
#if 1   // this disables variance normalization
        var.resize  (0, 1);
#else
        var.resize  (hdim, 1);
#endif
        fprintf (stderr, "mvn: created normalization layer of dim %d (%d for variance)\n", mean.rows(), var.rows());
        foreach_coord (i, j, mean) mean(i,j) = 0.0f;
        foreach_coord (i, j, var)  var(i,j)  = 1.0f;
        T = 24*3600*100;   // 24h of speech  --TODO: we could get this from a config parameter
    }
    virtual void copyfrom (const Iannlayer & iother) { auto & other = dynamic_cast<const mvn &> (iother); mean = other.mean; var = other.var; }

    // redistribute the model parameters through MPI
    // The models must already have been set up and dimensioned correctly; here we only exchange the weight parameters.
    virtual void mpiredistribute (const modelupdateinfo & bpinfo)
    {
        auto & mpiaggregator = *bpinfo.mpiaggregator;
        mpiaggregator.redistribute (mean.asvectorref());
        mpiaggregator.redistribute (var.asvectorref());
    }
    void entermpiaggregation (std::vector<size_t> & mpistripebuffersizes, size_t bits) { }  // dummy; we keep the accumulator local (hacky but OK for now)
    void exitmpiaggregation() { }
    virtual void quantizeandfetchsubbatchstripe (size_t stripe, char * bufferbegin, size_t buffersize, size_t & submbframes) { }
    virtual void syncfetchsubbatchstripe (size_t stripe) { }
    virtual void unquantizeandaggregatestripe (size_t ourstripe, size_t kfrom, const char * bufferbegin, size_t buffersize, bool isfirst, bool islast, size_t mbframes, const modelupdateinfo &, double momentumpersample, float learningratepersample) { }
    virtual void quantizeandassignaggregatedstripe (size_t ourstripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled) { }
    virtual void assignaggregatedstripe (size_t stripe, const char * bufferbegin, size_t buffersize) { }
    virtual void syncassignaggregatedstripeandunquantize (size_t stripe, const char * bufferbegin, size_t buffersize, size_t aggmbframes, const modelupdateinfo &) { }
    virtual void mpiallreducegradient (const modelupdateinfo & bpinfo) { }

    template<typename FILEHANDLETYPE> mvn (FILEHANDLETYPE f) { mean.read (f, "mean");
        mean.glimpse ("mean", false);
        var.read (f, "var"); var.glimpse ("var", false); T = (size_t) fgetint (f); }

    template<typename FILEHANDLETYPE> void dowrite (FILEHANDLETYPE f) const { mean.write (f, "mean"); var.write (f, "var"); fputint (f, (int) T); }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    virtual string type() const { return "mvn"; }

    virtual size_t vdim() const { return mean.rows(); }
    virtual size_t hdim() const { return mean.rows(); }

    virtual const matrix & peekweightmatrix() const { return var.peek(); }
    virtual const vector & peekbias() const { return mean.peek(); }

    // all sorts of functions that we don't need but that the interface requires to be implemented
    void print() const { printmat(mean); printmat(var); }
    void print (FILE *f) const { printmatfile(mean,f); printmatfile(var,f); }
    void dropoutscaleweights (float factor) { if (factor != 1.0f) throw std::logic_error ("mvn: MVN layer not compatible with dropout yet"); }

    void entercomputation (int type)
    {
        // lazily allocate the gradient matrices
        if (type != 0)
        {
            meanacc.resize (mean.rows(), mean.cols());
            varacc.resize  (var.rows(),  var.cols());
        }
        mean.entercomputation();    var.entercomputation();
        meanacc.entercomputation(); varacc.entercomputation();
        numacc = 0; meanacc.setzero(); varacc.setzero();
    }

    void exitcomputation()
    {
        mean.exitcomputation();    var.exitcomputation();
        meanacc.exitcomputation(); varacc.exitcomputation();
    }

    // forward propagation
    // Our job is to mean/var normalize the input vector.
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & vout, const bool /*linearonly*/, rbmstatevectorsref * /*pmask*/) const
    {
        // apply mean/variance normalization -> vout
        v.meanvarnorm (mean, true/*subtractmean*/, var, vout);
    }
    // back propagation just copies the error signal back
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & /*h*/, rbmstatevectorsref & /*out*/ ev) const
    {
        // this applies the variance scaling once again, but not the mean -> ev
        if (!var.empty())
            eh.meanvarnorm (mean/*dummy*/, false/*subtractmean*/, var, ev);
    }
    mutable vector sumhtmp;             // (buffer for NUMA version; I don't think we will ever run this)
    virtual void backpropagationmodelupdate1 (const rbmstatevectorsref & /*ehxs*/, const rbmstatevectorsref & v, const modelupdateinfo & bpinfo)
    {
        // gather all columns of v into an accumulator
        v.meanvaracc (numacc > 0, meanacc, mean, varacc);
        numacc += v.cols();             // and remember how many frames we added
#if 0   // diagnostics
        v.glimpse ("v", true);
        meanacc.glimpse ("meanacc", true);
        varacc.glimpse ("varacc", true);
        mean.glimpse ("mean", true);
        var.glimpse ("var", true);
#endif
    }
    virtual void backpropagationmodelupdate2 (const modelupdateinfo & bpinfo, bool mpimaisfirst, bool mpimaislast, float learningratepersample, double momentumpersample)
    {
        // update mean with accumulator
        // This function is called after the last frame of a minibatch (whereas update1() is potentially called multiple times in case of deferred update).
        if (numacc == 0)
            return;
        const float keepweight = (float) exp (-1.0 * numacc / T);
        const float inputscale = 1.0f / numacc; // divide frame sum by numacc to get average
        mean.addweighted (numacc > 0 ? keepweight : 0.0f, meanacc, (1.0f - keepweight) * inputscale);
        if (!var.empty())
            var.addweighted (numacc > 0 ? keepweight : 0.0f, varacc, (1.0f - keepweight) * inputscale);
        numacc = 0;                     // Consumed. (we don't clear the acc since numacc = 0 will notify update1() that there is nothing to add into)
    }
    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & ehxs_legacy,  const rbmstatevectorsref & v_legacy, float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo) { }
};


// ===========================================================================
// linearnetwork -- linear tranasform layer without non-linearity
// The class supports reduced-dimension mappings (block-diagonal structure)
// for use with neighbor-frame augmented input feature vectors:
//  - 'diagblocks' diagonal block matrices. All outside elements are 0.
//    This should be either 1 (no block-diag structure) or equal to the
//    number of neighbor frames used in the input feature vector.
//  - the diag blocks can be pooled or non-pooled ('poolblocks')
//    This is implemented in training, actual matrix contains copies.
// TODO:
//  - for non-pooled just use rbm with nonlinearitykind = linearkind
//  - for pooled we could use a convolutional network of appropriate dimensioning
// ===========================================================================

class linearnetwork : public rbmbase
{
    size_t diagblocks;          // e.g. 11 (must match neighbor expansion; 1 for full matrix)
    bool poolblocks;            // true -> blocks of neighbor frames are pooled
public:
    // TODO: get diagblocks and poolblocks from the config
    linearnetwork (size_t vdim, size_t hdim, const layerconfigparameters & config, size_t diagblocks, bool poolblocks)
        : diagblocks (diagblocks), poolblocks (poolblocks)
    {
        nonlinearitykind = linearkind;
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        validatedims();
        // initialize to identity (does nothing unless trained)
        initidentity();
        initbiaszero();
        fprintf (stderr, "linearnetwork: %d diagonal blocks for %d x %d network, %spooled\n", diagblocks, vdim, hdim, poolblocks ? "" : "not ");
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;
    }

    template<typename FILEHANDLETYPE>
    linearnetwork (FILEHANDLETYPE f) : rbmbase (f)
    {
        if (nonlinearitykind != linearkind)     // compat: old files do not store the nonlinearitykind
        {
            fprintf (stderr, "linearnetwork: nonlinearitykind not read from file, assuming legacy file format\n");
            nonlinearitykind = linearkind;
        }
        validatedims();
        fcheckTag (f, "BFLR");
        diagblocks = fgetint (f);
        poolblocks = fgetint (f) != 0;
        fcheckTag (f, "EFLR");
        numstream = W.cols() / W.rows();
        if ( numstream == 1 )
            hdimnumroundup = 0;
        else
            hdimnumroundup = (W.cols() - W.rows() * numstream) / (numstream - 1);
    }

    virtual string type() const { return "linearnetwork"; }

    // overridden so we can write extra information
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        rbmbase::write (f);
        fputTag (f, "BFLR");
        fputint (f, (int) diagblocks);
        fputint (f, poolblocks ? 1 : 0);
        fputTag (f, "EFLR");
    }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    virtual void copyfrom (const Iannlayer & iother)
    {
        //const auto & other = dynamic_cast<const decltype(*this) &> (iother);
        const linearnetwork & other = dynamic_cast<const linearnetwork &> (iother);
        rbmbase::copyfrom (other);
        diagblocks = other.diagblocks;
        poolblocks = other.poolblocks;
    }

#if 0   // handled by base class now
    // forward propagation
    // TODO: do we need this? The base function should do it.
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly=false, rbmstatevectorsref * pmask=NULL) const
    {
        if (pmask!=NULL) throw std::logic_error ("forwardprop: non-null mask is not implemented yet for linearnetwork layer");

        vtoz (v, Eh);   // h = W' v + a
    }
#endif

    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: linearnetwork does not support pre-training");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                    float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: linearnetwork does not support pre-training");
    }

    // linearnetwork needs special version of this to implement block-diagonal structure

    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                              float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo)
    {
        // first do the default update...
        rbmbase::backpropagationmodelupdate3 (ehxs, v, learningratepersample, momentumpersample, bpinfo);

        // ...and then post-process to enforce block-diagonal structure of matrix
        if (diagblocks > 1)
        {
            W.setblockdiagonal (diagblocks, poolblocks, numstream, hdimnumroundup, diagblocks == vdim()/*setidentity, indicate topsecond layer*/);
            a.setblockdiagonal (diagblocks, poolblocks, numstream, hdimnumroundup, false);
        }
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        // for a linear layer, there is no non-linearity to apply
        h; eh; ev;  // not used here

        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh
    }

    // computation of deltavs for unseen state compensation [v-hansu]
    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        // to be finished
        // will this function be called? possible, but the formula below shall be revised
        // deltah = h .* (1-h) .* [h .* (1-h) .* e * v' * v + W' * deltav]
        // deltaH = H .* (1-H) .* [H .* (1-H) .* E * diag(V' * V) + W' * deltaV]
        throw::logic_error ("forwardpropdelta: linearnetwork's forwardpropdelta is not finished, please check");
    }

    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        hdimnumroundup = W.colstride() - wrowsori;
        vdimnumroundup = 0;         //remember vdimroundup only record the vdim roundup, so it shall be set to 0;
        numstream = blowupfactor;
        const size_t arowsori = a.rows();
        const size_t acolsori = a.cols();

        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        foreach_coord(i, j, W)  Wbackup(i,j) = W(i,j);
        W.resize( wrowsori , (wcolsori + hdimnumroundup) * blowupfactor - hdimnumroundup);
        foreach_coord( i, j, W)    W(i, j) = 0;

        a.resize( (arowsori + hdimnumroundup) * blowupfactor - hdimnumroundup, acolsori );
        foreach_coord( i, j, a )    a(i, j) = 0;

        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for( size_t j = 0; j < wcolsori; j++)
                for( size_t i = 0; i < wrowsori; i++)
                    W(i, j + blockindex * (wcolsori + hdimnumroundup)) = Wbackup(i,j);
            for (size_t j = 0; j < acolsori; j++) 
                for (size_t i = 0; i < arowsori; i++)
                    a(i + blockindex * (arowsori + hdimnumroundup), j) = 0;
        }
        validatedims();
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: rbm layer shall not use this function with statemap");
    }

    virtual void setlinearlayerweight (const matrix & adaptmatrix)
    {
        fprintf(stderr,"setlinearlayerweight: initial adaptation matrix to GMM adaptation matrix");
        initidentity(adaptmatrix);
        initbiaszero();
    }
};

// ===========================================================================
// convolutionalBase -- base for the convolutional layers (convolution and maxpool)
// ===========================================================================

// handles all the convolutional parameters so we don't have to do it twice
class convolutionalBase : public rbmbase
{
protected:    
    mutable msra::cuda::convolutionParams params;

    convolutionalBase (const layerconfigparameters & config)
    {
        params.prevBands          = config("prevBands");
        params.prevKernels        = config("prevKernels");
        params.minibatchSize      = config("minibatchSize");
        params.bands              = config("bands");
        params.kernels            = config("kernels");
        params.poolSize           = config("poolSize");
        params.poolingBandShift   = config("poolingBandShift");
        params.filterSize         = config("filterSize");
        params.numFeatureSegments = config("numFeatureSegments");
    }

    template<typename FILEHANDLETYPE>
    convolutionalBase (FILEHANDLETYPE f) : rbmbase(f) { }

    void outputParams() const
    {
        fprintf (stderr, "prevBands %d, prevKernels %d, minibatchSize %d, bands %d, kernels %d, poolSize %d, poolingBandShift %d, filterSize %d, numFeatureSegments %d\n",
                 params.prevBands, params.prevKernels, params.minibatchSize, params.bands, params.kernels, params.poolSize, params.poolingBandShift, params.filterSize, params.numFeatureSegments);    
    }

    template<typename FILEHANDLETYPE>
    void readParams (FILEHANDLETYPE f)
    {
        params.prevBands          = fgetint (f);
        params.prevKernels        = fgetint (f);
        params.minibatchSize      = fgetint (f);
        params.bands              = fgetint (f);
        params.kernels            = fgetint (f);
        params.poolSize           = fgetint (f);
        params.poolingBandShift   = fgetint (f);
        params.filterSize         = fgetint (f);
        params.numFeatureSegments = fgetint (f);
    }

    template<typename FILEHANDLETYPE>
    void writeParams (FILEHANDLETYPE f) const
    {
        fputint (f, (int) params.prevBands);
        fputint (f, (int) params.prevKernels);
        fputint (f, (int) params.minibatchSize);
        fputint (f, (int) params.bands);
        fputint (f, (int) params.kernels);
        fputint (f, (int) params.poolSize);
        fputint (f, (int) params.poolingBandShift);
        fputint (f, (int) params.filterSize);
        fputint (f, (int) params.numFeatureSegments);
    }

public:
    // TODO: public?? or protected?
    const msra::cuda::convolutionParams & getParams() const { return params; }

    virtual void copyfrom (const Iannlayer & iother)
    {
        throw std::logic_error ("copyfrom: untested"); 
        //const auto & other = dynamic_cast<const decltype(*this) &> (iother);
        const convolutionalBase & other = dynamic_cast<const convolutionalBase &> (iother);
        rbmbase::copyfrom (other);
        params = other.params;
    }
};


// ===========================================================================
// convolutional -- convolutional network
// ===========================================================================

// TEMPORARY: This needs to be finished.

// We had to add an "energy" component to the dataset, since it didn't have 
// any energy component in the original data. which is why we have 41 for the prevBands, 
// one is an energy band. I believe these reference frequency bands.
// We just set energy to the same value for everything as I recall.

// prevKernels - this is some grouping of the bands to be processed together (45 in our default case). 
// After convolution and maxpooling have taken place we get back a different grouping of bands and kernels,
// which I believe is the point of convolution. It comes back as 20 bands and 84 kernels, which then feeds into the normal hidden layers of the DNN stack.

// poolSize is the size of the pool of outputs from the convolution layer that are considered to produce the maximum in the max-pool layer. 
// The maxPool layer takes it's input and reduces it by this factor, only returning the maximum value found in each pool.
 
// poolingBandShift, filtersize - as convolution takes a sample from the original data it shifts over by this many elements before taking the next sample. 
// I believe is sample is "filter" elements wide, and thus the pooling bands will overlap with the previous and next bands depending on the values of poolingBandShift and filterSize.
 
// numFeatureSegments - this is used to determine how many segments the original prevKernels parameter contains. I believe it was 3*15 = 45 in our default case. The 15 number doesn't appear here but can be derived by prevKernels/numFeatureSegments. It is used somewhere internally in the convolution layer code.

// how it works:
// Consider the input as a tensor with these three dimensions:
//  - time ("width")
//  - frequency band ("height")   --this assumes (and only makes sense for) a plain FBANK
//  - derivative order ("depth")
// A convolutional model applies a "filter" (=linear comb + sigmoid) only to sub-blocks of this tensor,
// of a given size, and those parameters are tied ("convolutional").
// Of course, for each there are many neurons covering the same input range in parallel.
// Typically, these would then be passed through a max-pooling layer next.
// This specific implementation can do the tying partially.
// How it is configured:
//  - filterSize = height of sub-block, e.g. 9
//    width and depth of sub-block are max, i.e. no sub-block taken there
//  - bands = frequency bands after filtering, e.g. 24 - 9 + 1
//  - kernels = number of different parallel parameter sets per frequency range, e.g. 256??
//  -> output dimension = bands * kernels
//  - prevKernels = numberof parallel parameter sets in input--if input is also convolutional (otherwise: 1)
// A max-pool layer is also convolutional, but instead of a linear transform + sigmoid, it simply takes the max of its inputs:
//  - poolingSize: equivalent of filterSize: height of sub-block to take the max over, e.g. 3
//    width and depth of the block are 1, since these dimensions are taken out by the convolutional layer
//  - poolingBandShift: ... TODO

// Arrangement of input and output data:

// Efficient computation:
//  - to share computation, ranges that share the same kernel are expanded into a sequence fo "frames" (columns)
//  - thus, we can just apply a matrix multiplication
//  - to be able to do that, we need to know the minibatch size, for memory allocation

// reverse engineering the code:
//  - output dimension hdim == params.bands * params.kernels
//  - input dimension vdim == params.prevKernels * (params.filterSize+1))


// CHECK/CORRECT THIS:
// The IBM approach (T. Sainath, ICASSP 2013) has two layers of conv+maxpool.
// The first uses the full time span, but band-ranges of 9, overlapping and spaced by 1.
// Three neighboring bands share tied parameters, and undergo max-pooling into a single output.
// E.g. a 24-dim feature vector has 24-9 = 15 shift positions; after max-pooling, those will be 3.
class convolutional : public convolutionalBase
{
    mutable rbmstatevectors reordered;  // OK to change this on const objects
    mutable rbmstatevectors deltaTranslated;  // OK to change this on const objects

#if 0
    void validatedims() const   // check if dimensions match
    {
        if (W.cols() != a.rows())
            malformed ("invalid model file--W matrix dimensions mismatch bias dimensions");
    }
#endif
public:
#if 0   // do we ever need this?
    convolutional (matrix && pW, vector && pa)
    {
        W = std::move (pW);
        a = std::move (pa);
        validatedims();
    }
#endif
    // TODO: merge convParams into c
    convolutional (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : convolutionalBase (config)
    {
        // BUGBUG: ugh, some redundancy here! Rather check whether the value is correct
        if (hdim != params.bands * params.kernels
            || vdim != params.prevKernels * (params.filterSize+1))
            throw std::runtime_error ("convolutional: specified dimensions do not match conv params (bands, kernels, prevKernels, filterSize)");
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        deltaTranslated.resize(params.minibatchSize, params.kernels * params.bands * params.poolSize);
        validatedims();
        initrandom (randomseed);

        fprintf (stderr, "convolutional: %d x %d network", vdim, hdim);
        outputParams();
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;
    }

    size_t vdim() const { return params.prevBands*params.prevKernels; }
    size_t hdim() const { return params.bands*params.kernels*params.poolSize; }

    template<typename FILEHANDLETYPE>
    convolutional (FILEHANDLETYPE f) : convolutionalBase (f)
    {
        validatedims();
        fcheckTag (f, "BCNN");
        readParams(f);
        fcheckTag (f, "ECNN");
        numstream = W.cols() / W.rows();
        if ( numstream == 1 )
            hdimnumroundup = 0;
        else
            hdimnumroundup = (W.cols() - W.rows() * numstream) / (numstream - 1);
    }

    virtual string type() const { return "convolutional"; }

    // get a reference to the reordered input
    rbmstatevectorsref getReorderedInput() const {return reordered.stripe(0, reordered.cols());}
    rbmstatevectorsref getDeltaTranslated() const {return deltaTranslated.stripe(0, deltaTranslated.cols());}

    // overridden so we can write extra information
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        rbmbase::write (f);
        fputTag (f, "BCNN");
        writeParams (f);
        fputTag (f, "ECNN");
    }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    virtual void copyfrom (const Iannlayer & other) { throw std::logic_error ("copyfrom: untested"); convolutionalBase::copyfrom (other); }

    // forward propagation
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly=false, rbmstatevectorsref * pmask=NULL) const
    {
        params.minibatchSize = v.cols();    // get the batch size from the number of columns on input
        //rbmstatevectors* nonconstMatrix = const_cast<rbmstatevectors*>(&reordered);
        reordered.resize(v.cols(), v.rows());
        rbmstatevectorsref u (reordered.stripe(0, v.rows()));

        //v.dump("convolution input");
        v.reorderForConvolutional(u, params);
        //u.dump("convolution reordered input");
        //W.dump("weight matrix");
        W.convolutionForward(u, Eh, a, params);
        //Eh.dump("convolution output");

        // TODO: move the non-linearity out from that kernel (activateCNN())
    }

    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        deltah; deltav; h; v; eh; vnorms; learningrateperframe; momentumpersample;
        throw std::logic_error ("forwardpropdelta: convolutional does not support forwardpropdelta");
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: convolutional does not support pre-training");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                    float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: convolutional does not support pre-training");
    }

    // we have our own version of this since we try to compute stuff more efficiently (is it?)
    virtual void backpropagationmodelupdate1 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v, const modelupdateinfo & bpinfo)
    {
        // first do the default update...
        // resize the matrix to the right size for the next operation
        deltaTranslated.resize (params.minibatchSize, this->hdim());

        // we misuse 'modelupdateinfo' to tell pushtorawgradient() that we are "convolutional" (this was a quick hack)
        modelupdateinfo * pbpinfo = const_cast<modelupdateinfo *>(&bpinfo);
        pbpinfo->nochangeifaboveorbelow = 0;  /*set to 0 for linear layer*/
        pbpinfo->preflayer = this;
        //W.dump("Convolution Weight before");

        //v.dump("convolution backprop input");
        rbmbase::backpropagationmodelupdate1 (ehxs, v, bpinfo);
        //W.dump("Convolution Weight after");
        //ehxs.dump("convolution backprop output");
        pbpinfo->preflayer = NULL;
    }

#if 0
    virtual void backpropagationmodelupdatedeleteme (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, const modelupdateinfo & bpinfo)
    {
        // first do the default update...
        // resize the matrix to the right size for the next operation
        deltaTranslated.resize (params.minibatchSize, this->hdim());

        modelupdateinfo *pbpinfo = const_cast<modelupdateinfo *>(&bpinfo);

        // overloading the modelupdateinfo, since it's already being sent in anyhow
        pbpinfo->nochangeifaboveorbelow = 0;  /*set to 0 for linear layer*/
        pbpinfo->preflayer = this;
        //W.dump("Convolution Weight before");

        //v.dump("convolution backprop input");
        rbmbase::backpropagationmodelupdate (ehxs, v, learningratepersample, momentumpersample, resetmomentum, deferupdate, bpinfo);
        //W.dump("Convolution Weight after");
        //ehxs.dump("convolution backprop output");
        pbpinfo->preflayer = NULL;
    }
#endif

    // backward error propagation
    // TODO: [fseide] question: does this not need to apply the sigmoid derivative?
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        // for a linear layer, there is nothing to do here
        // BUGBUG? Don't we need the non-linearity derivative here?
        h; eh; ev;  // not used here

        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        // TODO: Check the math whether this is actually correct once we ever use this as an intermediate rather than bottom layer.
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh
    }

#if 0
    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }
#endif

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        hdimnumroundup = W.colstride() - wrowsori;
        vdimnumroundup = 0;         //remember vdimroundup only record the vdim roundup, so it shall be set to 0;
        numstream = blowupfactor;
        const size_t arowsori = a.rows();
        const size_t acolsori = a.cols();

        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        foreach_coord(i, j, W)  Wbackup(i,j) = W(i,j);
        W.resize( wrowsori , (wcolsori + hdimnumroundup) * blowupfactor - hdimnumroundup);
        foreach_coord( i, j, W)    W(i, j) = 0;

        a.resize( (arowsori + hdimnumroundup) * blowupfactor - hdimnumroundup, acolsori );
        foreach_coord( i, j, a )    a(i, j) = 0;

        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for( size_t j = 0; j < wcolsori; j++)
                for( size_t i = 0; i < wrowsori; i++)
                    W(i, j + blockindex * (wcolsori + hdimnumroundup)) = Wbackup(i,j);
            for (size_t j = 0; j < acolsori; j++) 
                for (size_t i = 0; i < arowsori; i++)
                    a(i + blockindex * (arowsori + hdimnumroundup), j) = 0;
        }
        validatedims();
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: rbm layer shall not use this function with statemap");
    }

    virtual void setlinearlayerweight(const matrix & adaptmatrix)
    {
        fprintf(stderr,"setlinearlayerweight: initial adaptation matrix to GMM adaptation matrix");
        initidentity(adaptmatrix);
        initbiaszero();
    }
};

class maxpool : public convolutionalBase
{
    mutable rbmstatevectors maxIndex;   // holds the index of all the max entries in the input and how they map to the output matrix    
                                // we store them fror back propogation, should be integers, but will use floats for now
    
public:
    // TODO: merge convParams into c
    maxpool (size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed) : convolutionalBase (config)
    {
        //W.resize (vdim, hdim); - no weight matrix needed
        // a.resize (hdim, 1); 
        //validatedims();

        fprintf (stderr, "maxpool: %d x %d network", vdim, hdim);
        outputParams();
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;

        if (hdim != this->hdim() || vdim != this->vdim())
            throw std::runtime_error ("maxpool: specified dimensions do not match conv params (bands, kernels, poolSize)");
    }

    // get the dimensions
    size_t vdim() const { return params.bands * params.kernels * params.poolSize; }
    size_t hdim() const { return params.bands * params.kernels; }

    template<typename FILEHANDLETYPE>
    maxpool (FILEHANDLETYPE f) : convolutionalBase (f)
    {
        fcheckTag (f, "BMPN");
        readParams(f);
        fcheckTag (f, "EMPN");
        numstream = 1;
    }

    virtual string type() const { return "maxpool"; }

    // overridden so we can write extra information
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        rbmbase::write (f);
        fputTag (f, "BMPN");
        writeParams(f);
        fputTag (f, "EMPN");
    }

    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }

    virtual void copyfrom (const Iannlayer & other) { throw std::logic_error ("copyfrom: untested"); convolutionalBase::copyfrom (other); }

    // forward propagation
    
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly=false, rbmstatevectorsref * pmask=NULL) const
    {
        // make sure the dimensions of the input and output are correct
        assert(v.cols() == Eh.cols());
        assert(v.rows() == Eh.rows()*params.poolSize);

        params.minibatchSize = v.cols();    // get the batch size from the number of columns on input
        maxIndex.resize(hdim(), params.minibatchSize);

        //rbmstatevectors* nonconstMatrix = const_cast<rbmstatevectors*>(&maxIndex);
        rbmstatevectorsref u (maxIndex.stripe(0,maxIndex.cols()));
        //v.dump("maxpool forward input");
        v.maxpoolForward (Eh, u, params);    // take the max and remember its index as well
        //Eh.dump("maxpool forward output");
        //u.dump("maxpool maxIndex");
    }

    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        deltah; deltav; h; v; eh; vnorms; learningrateperframe; momentumpersample;
        throw std::logic_error ("forwardpropdelta: maxpool does not support forwardpropdelta");
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: maxpool does not support pre-training");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                    float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: maxpool does not support pre-training");
    }

    // no model updates necessary since we don't have a weight matrix for this layer
    virtual void backpropagationmodelupdate1 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v, const modelupdateinfo & bpinfo) { }

    virtual void backpropagationmodelupdate2 (const modelupdateinfo & bpinfo, bool mpimaisfirst, bool mpimaislast, float learningratepersample, double momentumpersample) { }

    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo) { }

    // backward error propagation
    // Note that a maxpool layer has no weights.
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        // make sure the dimensions of the input and output are correct
        assert(ev.cols() == eh.cols());
        assert(ev.rows() == eh.rows()*params.poolSize);

        //rbmstatevectors* nonconstMatrix = const_cast<rbmstatevectors*>(&maxIndex);
        rbmstatevectorsref u (maxIndex.stripe(0,maxIndex.cols()));
        //eh.dump("maxpool backprop input");
        //h.dump("maxpool 'h' input");
        //if (nonlinearitykind != sigmoidkind)
        //    throw runtime_error ("backpropagationstats: non-sigmoid units not implemented");

        switch (nonlinearitykind) 
        {
        case sigmoidkind:   eh.mulbydsigm (h); break;
        case relukind:      eh.mulbydlru (h); break;
        case leakyrootkind: throw std::logic_error ("this function should be overloaded for this kind"); break; 
#if 1
        case linearkind:    eh.mulbydmaxout (h); break; // this is hack for dropout to not propagate through zeroed units, otherwise it is linear
#else
        case linearkind:    /*do nothing*/ break;
#endif
        default:
            throw runtime_error (msra::strfun::strprintf ("backpropagationstats: units of type %d not implemented", nonlinearitykind));
        }

        //eh.dump("maxpool input post sigmoid derivative multiply");
        // eh is input, ev is output 
        eh.maxpoolBack(ev, u, params);        
        //ev.dump("maxpool backprop output");
    }

    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: not implemented");
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: not implemented");
    }

    virtual void setlinearlayerweight(const matrix & adaptmatrix)
    {
        throw std::logic_error ("setlinearlayerweight: not implemented");
    }
};


#if 1

// ===========================================================================
// maxoutnetwork -- affine transform + (optional nonlinearity) + max pooling on the top
// in this implementation we do pooling implicitly in the class rather than two layer variant, 
// also we use convolutionParams structure to reuse the maxpool code and avoid modyfing dropout code logic
// (which would be required in case of two layer implementation of maxout abstract as we drop activations of the maxpooled layers only). 
// Only poolSize and poolBandShift* attributes of convolutionParams are meaningful for nonconvoluted maxouts, 'kernels' and 'bands' 
// does not make too much sense but we still set these params to 'sensible' values as maxpool relies on them to compute pooling offsets
// *TODO: poolstriding not implemented yet
// ===========================================================================

// we inherit from maxpool, which is built on top of rbmbase (of which parametrs are not used by maxpool anyway) so we can
// easily reuse them and possibly try different nonlinearities in maxout configuration 
class maxoutnetwork : public maxpool
{
    bool   randomizePools;  //TODO: move to convolutionalBase? and implement activations shuffling before maxpolling
    mutable rbmstatevectors EhBase; //this is an 'intermediate' hidden space for acitvations vector, the visible layer activations are maxpooled from it
    mutable rbmstatevectors ehBase; //similar here, error signal coming from the layer above need to be pooled back to the 'intermediate' hidden space activation
    //mutable rbmmodelmatrix norms;  //column wise norms
    //float maxcolnorm; //normalize columns of W based on this parameter. TODO: that perhaps should be in the model, allowing to apply accross different layer types

public:
    // constructors: from scratch and from file
    maxoutnetwork (size_t vdim, size_t hdim, size_t poolSize, const layerconfigparameters & config, unsigned int randomseed) : maxpool (hdim*poolSize, hdim, config, randomseed)
    {
        config.dump();                      // (for debugging; TODO: remove this once we see it working)
        nonlinearitykind = linearkind;   // overwrite the kind (rbm initializes it as sigmoidkind), --TODO, pass through config

        //vdim, hdim are the one inferred from the command line arguments (after maxpooling), hidden dim for rbmbase is hdim*params.poolSize
        //alter kernels and bands so maxpooling is correct for nonconvoluted network and maxpool::vdim() and hdim() works correctly
        params.bands = hdim;
        params.kernels = 1;

        W.resize (vdim, hdim*params.poolSize);
        a.resize (hdim*params.poolSize, 1);
        norms.resize(W.cols(),1);
        validatedims ();
        initrandom (randomseed);

        //maxcolnorm = config("maxcolnorm",  0.0f); //maxcolnorm is not the member of convolutionalBase, so won't be in params
        /*maxcolnorm=2.5f;
        if (maxcolnorm>0.0) {
            norms.resize(W.cols(), 1);
            norms.entercomputation();
        }*/

        fprintf (stderr, "maxout: layer crated with W [%d x %d]\n", W.rows(), W.cols());
    }

    template<typename FILEHANDLETYPE> maxoutnetwork (FILEHANDLETYPE f) : maxpool(f)
    {
        //if (nonlinearitykind != linearkind) //[TODO] we could actually use any nonlinearities on top of pooled maxout activations
        //    throw std::logic_error ("maxoutnetwork: unexpectedly failed to read nonlinearitykind from file");
        /*maxcolnorm = 2.5f; //tmp, save this with file as may be reuired when learning was restarted
        if (maxcolnorm>0.0) {
           norms.resize(W.cols(), 1);
           norms.entercomputation();
        }*/
    }

    /*// overridden so we can write extra information
    template<typename FILEHANDLETYPE>
    void dowrite (FILEHANDLETYPE f) const
    {
        rbmbase::write (f);
        fputTag (f, "BMON");
        writeParams(f);
        fputTag (f, "EMON");
    }
    virtual void write (FILE * f) const { dowrite (f); }
    virtual void write (HANDLE f) const { dowrite (f); }*/

    virtual void copyfrom (const Iannlayer & other) { throw std::logic_error ("copyfrom: not yet implemented for this type of layer"); }

    // get the dimensions
    size_t vdim() const { return W.rows(); } //we hide intermediate maxpool input dimension and get the actual 'real' value
    size_t hdim() const { return params.bands * params.kernels; } //as in maxpool

protected:

    virtual string type() const { return "maxoutnetwork"; }

    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly = false, rbmstatevectorsref * pmask = NULL) const
    { 
        EhBase.resize (W.cols(), Eh.cols());
        rbmstatevectorsref EhBaseRef(EhBase.stripe(0, Eh.cols()));
        
        rbmbase::forwardprop(v, EhBaseRef, linearonly, pmask);
        maxpool::forwardprop(EhBaseRef, Eh, linearonly, pmask);

        v.dump("maxout: v ");
        EhBaseRef.dump("maxout: Eh before maxpool");
        Eh.dump("maxout: Eh after maxpool");
    }

    // we have our own version since we do some memory shuffling with ehxs   --TODO: is this reason correct? Or do we no longer need this override?
    virtual void backpropagationmodelupdate3 (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                              float learningratepersample, double momentumpersample, const modelupdateinfo & bpinfo)
    {
        if (bpinfo.distributefixedcost)
            throw std::logic_error ("backpropagationmodelupdate3 (maxoutnetwork): compatibility with 'distributefixedcost' mode to be verified (once you verify, please remove this error)");
        rbmstatevectorsref ehBaseRef (ehBase.stripe (0, ehxs.cols()));
        ehBaseRef.dump ("BackModUpd: ehBase");
        rbmbase::backpropagationmodelupdate3 (ehBaseRef, v, learningratepersample, momentumpersample, bpinfo);

#if 0   // maxnorm is implemented in the base now
        const float maxcolnorm = 0.0f;      // TODO: if this ever works & proves useful, make this a config parameter
        // column-wise regularization moved to addgradienttomodel
        // at this moment W keeps fully updated values, check whether norms are OK and renormalize them otherwise
        // W = W*diag(Wcolscale) where Wcolscale = maxcolnorm / (actcolnorm + eps) for each actcolnorm > maxcolnorm or 1 otherwise
        if (maxcolnorm>0.0f)
        {
            W.colwisenrm2(norms, maxcolnorm); //compute the scale factors which satisfy maxcolnorm upper limit
            //W.colwisenrm2(norms, 0.0f); //get norms instead of scales
            
#ifdef _DEBUG
            rbmmodelmatrix tmpn1;
            tmpn1.resize(norms.rows(), norms.cols());  norms.getweightmatrix(tmpn1);
            size_t rows = tmpn1.rows();
            for (size_t i=0, j=0; i<rows; i++) {
                if (tmpn1(i,0) != 1.0) {
                    fprintf(stderr, "maxout:backpropagationmodelupdate: W : %d-th column scaled by factor %f (norm was %f) \n", i, tmpn1(i,0), maxcolnorm/tmpn1(i,0));
                    if (j++>10) break; //print only some first normalised numbers to get some intuition whats is going on with the weights
                }
            }
#endif
            //norms.dump("norms");
            //W.dump("w");
            W.scalecolwise(norms);
            //W.dump("w2");
        }
#endif
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        ehBase.resize(W.cols(), ev.cols()); //this will run out of memory in case actually resizing, assuming here minibatch is constant within the epoch

        rbmstatevectorsref ehBaseRef(ehBase.stripe(0, ev.cols()));
        maxpool::backpropagationstats(eh, h, ehBaseRef);

        h.dump("maxout: h activations before pooling back");
        eh.dump("maxout: eh error singal before pooling back");
        ehBaseRef.dump("maxout: ehBase: error singal after pooling back");

        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (ehBaseRef, ev);  // ev = W eh
    }

    // TODO: apply here autoencoder like pretraining where we minimise ||v-v'||^2_2 and v' = W^T(poolback(dropout(maxpool(Wv+a))))+b, since we do maxpooling 
    // in the single iteration we only update a fraction of parameters, additionally, dropout will prevent from learning identity mapping
    // so dropping inputs as in denoising autoencoders may not be neccesary (there is already some evidence it harms speech accuracy)
    virtual void pretrainingstats (const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Eh; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: not implemented for maxoutnetwork for now");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample)
    {
        v; h; v1; h1; learningratepersample; momentumpersample;
        throw std::logic_error ("pretrainingmodelupdate: not implemented for maxoutnetwork for now");
    }

    // computation of deltavs for unseen state compensation [v-hansu]
    virtual float forwardpropdelta (rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h, 
                                    /*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
                                    const float learningrateperframe, const double momentumpersample) const
    {
        throw::logic_error ("forwardpropdelta: not implemented for maxoutnetwork");
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: maxoutnetwork layer shall not use this function with statemap");
    }

    virtual void blowup(const size_t blowupfactor)
    {
        throw::logic_error ("blowup: not implemented for maxoutnetwork");
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented for maxoutnetwork");
    }
};

#endif
    // compute the raw (un-smoothed, un-scaled) gradient: (eh,v) -> raw_dW,a,b
    // 'eh' is the error signal multiplied with the sigmoid' (except for linear fDLR layer).
    // In order to support 'deferred update', the raw gradient is actually accumulated.
    // The later function updategradient() will reset raw_dmbframes to 0 when it has consumed the raw gradient.
    // This function is passed the raw accumulators by reference since in the case of AdaGrad, it may apply it to a temp buffer first.
    void rbmbase::pushtorawgradient (const rbmstatevectorsref & eh, const rbmstatevectorsref & v, bool updateb,
                                     rmbmodelmatrix & to_dW, rmbmodelvector & to_da, rmbmodelvector & to_db, size_t & to_dmbframes,
                                     const modelupdateinfo * modelupdateparameters) const
    {
        assert (eh.cols() == v.cols());  // cols = frames
        assert (!da.empty() && !dW.empty());
        assert (!db.empty() || !updateb);

        const float keepweight = (to_dmbframes > 0) ? 1.0f : 0.0f;     // if we already have frames in here, so accumulate; only happens in case of 'deferred update'
        if (to_dmbframes)
            fprintf (stderr, "pushtorawgradient: *adding* partial minibatch to raw gradient acc (current frames: %d)\n", (int) to_dmbframes);

        if (eh.cols() == 0) // special fix for empty minibatch... ugh!
        {
            if (keepweight == 0.0f) // special case that is not handled correctly by various functions, so we pre-fix it here
            {
                to_da.setzero();
                to_dW.setzero();
                if (updateb)
                    to_db.setzero();
            }
            // TODO: fix those functions! addrowsum is already correct; so it seems gemm() is broken; that would be a bug in CUBLAS that I should report. (Also probably need to fix the convolutional variants.)
        }
        // convolutional models are represented differently
        // Note that neither Dong nor Adam remember how...
        else if (modelupdateparameters && modelupdateparameters->preflayer && modelupdateparameters->preflayer->type() == "convolutional")
        {   // ^^ TODO: a bad hack to communicate that we are convolutional --why not just check 'this'??
            if (!mean.empty())
                throw std::logic_error ("pushtorawgradient: MVN-SGD not implemented for convolutional networks");
            const convolutional * conv = ((const convolutional *) modelupdateparameters->preflayer);    // TODO: Shouldn't this be a dynamic_cast?
            const msra::cuda::convolutionParams & params = conv->getParams();
            rbmstatevectorsref reordered = conv->getReorderedInput();
            rbmstatevectorsref deltaTranslated = conv->getDeltaTranslated();
            to_da.scaleandaddallcolspool (keepweight, eh, 1, sumhtmp, params.poolSize, params.bands, params.kernels);
            to_dW.convolutionalScaleAndAddMatProduct (keepweight, reordered, const_cast<rbmstatevectorsref &>(v), eh, deltaTranslated, 1, params);
            if (updateb)
                throw std::logic_error ("pushtorawgradient: pre-training not implemented for convolutional networks");   // TODO: is it?
        }
        else
        {
            if (!mean.empty())  // MVN-SGD
            {
                // MVN-SGD [Wiesler, 2014]
                // This consists of two steps.
                //  - compute gradient on mean-normalized observation (here)
                //    UGH! We patch up 'v' in-place!! This will make it do the right thing.
                //    This computes gradients for a modified local objective that uses (v-v0).
                //  - (do all you want to the gradients, e.g. Adagrad, momentum, quantization)
                //  - map those gradients to the non-modified local objective
                // See addtomodel() for more documentation.
                fprintf (stderr, "pushtorawgradient: MVN-SGD enabled, patching 'v' in-place (%s, %d x %d)\n", type().c_str(), vdim(), hdim());
                v.meanvarnorm (mean, true/*subtractmean*/, var, const_cast<rbmstatevectorsref &> (v));  // v1 = (v-v0)/D, with v1 stored in place of v
                if (normsmbcounter < 8 || normsmbcounter % 16 == 0)
                    v.glimpse ("v1", true);
                // TODO: we could also do that to to_dW directly; first sum up all frames in eh -> ehsum, then
                //to_dW.scaleandaddmatprod (keepweight, mean, ehsum, -1.0f, httmp, cachedvts, cachedhts);
            }

            to_da.scaleandaddallcols (keepweight, eh, 1, sumhtmp);
            to_dW.scaleandaddmatprod (keepweight, v, eh, 1, httmp, cachedvts, cachedhts);
            if (updateb)
                to_db.scaleandaddallcols (keepweight, v, 1, sumvtmp);
        }
        // also record number of frames for this gradient (e.g. important for data parallelism, where we aggregate multiple)
        const size_t mbframes = eh.cols(); assert (v.cols() == mbframes);
        to_dmbframes += mbframes;      // this is the #frames we have accumulated in here up to now
        fprintf (stderr, "pushtorawgradient: %d frames in raw gradient acc\n", (int) to_dmbframes);
    }

};};
