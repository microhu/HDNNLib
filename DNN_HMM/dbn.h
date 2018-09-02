// dbn.h -- implementation of Hinton's Deep Belief Network
//
// F. Seide, Nov 2010 based on code provided by Yu Dong, MSR Speech Research Group
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/dbn.h $
// 
// 528   7/04/14 12:10 Fseide
// commented dimensions of SVD matrices
// 
// 527   7/04/14 12:01 Fseide
// moved w[] vector to inside RBM's svd() function, since it is not
// exposed or needed outside at all
// 
// 526   7/04/14 11:43 Fseide
// (forgot a mem alloc in svd())
// 
// 525   7/04/14 11:38 Fseide
// fixed lack of parallel_for() to go backwards
// 
// 524   7/04/14 11:27 Fseide
// SVD: moved transposition of V inside the layer's svd() function;
// SVD now uses parallel_for() over layers
// 
// 523   7/04/14 11:12 Fseide
// (comments)
// 
// 522   7/04/14 10:57 Fseide
// towards parallelizing SVD across layers
// 
// 521   7/03/14 16:34 Fseide
// printvaluedistribution() now prints average column length and average
// bias across layers;
// some comments
// 
// 520   6/16/14 16:42 Fseide
// setutterancedata() no longer syncs after transferring;
// added some log messages to track down sync/cuda-alloc points;
// pred(,) renamed to scaledLLs(,) and now declared outside of MB loop to
// reduce reallocs
// 
// 519   6/12/14 9:23p Fseide
// bug fix of an incorrect error check in converttosvdmodel(), which fail
// it for the perceptron layer
// 
// 518   6/11/14 6:51p Fseide
// bug fix: previous change incorrectly set the layer to linearkind,
// whereas that should be the other layer which was already set correctly
// (the original code was correct after all)
// 
// 517   6/11/14 4:20p Fseide
// further clean-up of the SVD mess, to be tested someday
// 
// 516   6/06/14 6:05p Fseide
// logging of layer creation
// 
// 515   6/06/14 4:08p Fseide
// softplusnetwork implemented
// 
// 514   6/04/14 10:18a Fseide
// removed the fflush() from the layer stats output, as it gets in the way
// of tracking progress
// 
// 513   5/29/14 12:12p Fseide
// dropoutscaling() now prints no message if it does nothing
// 
// 512   5/21/14 3:43p Fseide
// (spelling)
// 
// 511   5/20/14 3:27p Fseide
// add frame dropping function for SE training.
// 
// 510   5/20/14 10:11a Fseide
// reduced statscounter since it really slows sown time measurements
// 
// 509   5/16/14 10:06a Fseide
// new command "insertmvn" and new method insertmvnlayers()
// 
// 508   5/14/14 14:41 Fseide
// creation/loading functions now know about class mvn
// 
// 507   5/12/14 2:51p Fseide
// added two log messages for dropout
// 
// 506   5/12/14 14:09 Fseide
// cleanup of forwardprop() w.r.t. dropout pre-scaling (no longer done
// inside forwardprop());
// comments
// 
// 505   5/12/14 13:53 Fseide
// removed dropout pre-scaling everywhere incl. CUDA side
// 
// 504   5/12/14 13:50 Fseide
// removed prescaledropout flag from dbn.h as well
// 
// 503   5/12/14 13:36 Fseide
// added infrastructure to no longer require dropout models to be scaled
// later (they are now scaled upon exitcomputation(), and unscaled upon
// entercomputation())--note: still need to fix the actual forwardprop()
// function;
// GenerateMeanModel() renamed to dropoutscaling() (dbn.h) and
// dropoutscaleweights (rbm.h) since it is now also used to undo the
// change
// 
// 502   4/29/14 1:42p Fseide
// unquantizeandaggregatestripe() now also takes the learning rate;
// gradientfixups() now multiplies the gradient with an additional
// learning-rate parameter -> this way it can compute the final-final
// thing that gets added to the model;
// syncassignaggregatedstripeandunquantize() now takes bpinfo so it can
// know about 'distributefixedcost' mode
// 
// 501   4/28/14 3:20p Fseide
// (comment)
// 
// 500   4/28/14 2:40p Fseide
// implemented direct use of MPI_Allreduce() in non-quantized case (as one
// would prefer for model-averaging case);
// made some methods robust to empty matrix dimensions;
// new methods mpiallreducegradient(), mpihelper::allreducescalar(), and
// allreduce() for matrices
// 
// 499   4/27/14 16:58 Fseide
// qpackages now contain a header (struct mpistripeheader) that stores the
// number of frames of the gradient they represent;
// syncassignaggregatedstripeandunquantize() no longer takes the
// 'numstripes' argument (it was unused and is superfluous now)
// 
// 498   4/27/14 16:26 Fseide
// moved resetting of mpistripebuffersizes[] out from the init lambda to
// entercomputation(), which can now pre-init it to allocate a header
// 
// 497   4/27/14 16:11 Fseide
// new modelupdateinfo parameter 'distributefixedcost';
// unquantizeandaggregatestripe() now takes all parameters needed to
// distribute fixed cost (AdaGrad, momentum)
// 
// 496   4/27/14 15:34 Fseide
// (towards distributed fixed cost)
// 
// 495   4/27/14 15:03 Fseide
// data parallelism: first steps towards doing part of fixed cost inside
// the stripe operation
// 
// 494   4/09/14 18:16 Fseide
// (commented out log message)
// 
// 493   4/09/14 16:53 Fseide
// poor-man's implementation of dropout (that is, inefficient) without
// requiring model fix-up (--prescaledropout)
// 
// 492   4/09/14 16:02 Fseide
// dropout() now applied to input not output (i.e. refactored the loops,
// same behavior)
// 
// 491   4/09/14 8:55 Fseide
// (comments)
// 
// 490   4/08/14 18:07 Fseide
// forwardprop(): dropout no longer applied to input layer; and now prints
// layer statistics every 64 minibatches
// 
// 489   4/04/14 11:21a Fseide
// various calls to matrix::resize() now pass an explicit 'setzero' flag
// 
// 488   4/01/14 3:09p Fseide
// bug fix for interplay between AdaGrad (finishaccumulation()) and local
// loop
// 
// 487   3/28/14 7:01p Fseide
// mpimaframes now always interpreted as a MB count (no longer as #frames)
// 
// 486   3/20/14 6:58p Fseide
// mpimasize can now mean a relative factor instead of #frames
// 
// 485   3/20/14 5:57p Fseide
// (fixed a warning)
// 
// 484   3/20/14 5:55p Fseide
// bug fix: in local-loop mode, the final update should have no learning
// rate factor
// 
// 483   3/20/14 5:46p Fseide
// un-defined TIME_MODELUPDATE--why was that on, and what jobs did it
// affect??
// 
// 482   3/20/14 4:07p Fseide
// some refactoring and initial code for local loop for data parallelism
// 
// 481   3/20/14 3:06p Fseide
// towards Kaldi-style model averaging for data parallelism
// 
// 480   3/20/14 1:43p Fseide
// added optimized versions of col/row-wise scaling, but disabled for now
// until verified
// 
// 479   2/25/14 9:07p V-jiaxu
// fix bug in idim(). ivectormean.size() doesn't return 0 when
// ivectormean.cols() == 0.
// 
// 478   2/25/14 11:31a V-jiaxu
// add method idim() to get ivector dimension, fix augmentationextent()
// 
// 477   2/24/14 18:24 Fseide
// onpartialsubgradient mode now allows for computing the avdenom on the
// first chunk, to at least have some tracking
// 
// 476   2/24/14 15:56 Fseide
// bug fix in backpropagationmodelupdate(), moved the wrong piece of code
// just now!
// 
// 475   2/24/14 14:32 Fseide
// (removed constructor of 'model' without ivector params, only used in an
// unimportant piece of code, fixed that instead)
// 
// 474   2/24/14 14:27 Fseide
// moved up initialization of AdaGrad gradient before ...update2() so that
// we can do AdaGrad in there
// 
// 473   2/21/14 3:45p V-jiaxu
// save ivectormean/ivectorstd in model instead of ivecdim for ivector
// normalization
// 
// 472   2/21/14 13:50 Fseide
// backpropagationmodelupdate2() now takes 'bpinfo' like all others;
// implemented sub-gradient AdaGrad in there (to be tested)
// 
// 471   2/21/14 13:38 Fseide
// entercomputation() now takes a flag to en/disable double buffering;
// double buffering now gets automatically disabled when using AdaGrad
// since it does not play well with it (this is not a final answer)
// 
// 470   2/21/14 13:30 Fseide
// entermpiaggregation() now takes 'bits' as an explicit parameter
// 
// 469   2/21/14 9:54a V-jiaxu
// ivector normalization
// 
// 468   2/19/14 3:31p V-jiaxu
// To support new input feature (ivector), dimension mismatch
// (!layers.empty() && layers[0]->vdim() % mean.size() != 0) no longer
// throw an error 
// 
// 467   2/18/14 6:55p Fseide
// (logging)
// 
// 466   2/18/14 2:50p Fseide
// implemented multi-GPU support (model parallelism) for MPI (data
// parallelism), not tested yet but should at least work for single GPU;
// allocatetransferbuffer() now allocates on the correct GPU
// 
// 465   2/18/14 11:44a Fseide
// (removed a fflush(stderr))
// 
// 464   2/18/14 10:34a Fseide
// new class adagradstate_t to hold the cross-layer AdaGrad average
// implemented (passed around but not used yet)
// 
// 463   2/18/14 9:57a Fseide
// raw_dmbframes is now updated under the control of MPI aggregation. This
// is a preparation for separating out AdaGrad from ...update3().
// 
// 462   2/17/14 5:23p Fseide
// bug fix in the timing code for ...update3()
// 
// 461   2/17/14 2:09p Fseide
// backpropagationmodelupdate() now by default times
// backpropagationmodelupdate3() every 20-th MB (which implies a CUDA
// sync, so we don't want to do it every single time)
// 
// 460   2/14/14 6:29p Fseide
// updated the time measurement code (#ifdef-ed out) to include only the
// "fixed cost"
// 
// 459   2/14/14 10:54 Fseide
// added model variables--a dictionary of key/value pairs that is
// persisted in the model file itself, for carrying over iteration state
// 
// 458   2/13/14 15:21 Fseide
// bug fix for variable Kopt (#nodes in MPI mode): in MPI mode, model is
// now being sync'ed from node 0 to all others at start of each epoch, as
// to get all nodes the latest model even if they did not participate in
// previous epochs and thus did not get their model updates
// 
// 457   2/11/14 22:06 Fseide
// back to outputting timing info again to track down that strange glitch
// that kills us
// 
// 456   2/11/14 17:30 Fseide
// (renamed a variable for clarity)
// 
// 455   2/11/14 3:04p Fseide
// added equivalents to synchronize() that specifically wait after fetch()
// and assign(), implemented as stream syncs rather than global device
// syncs, hoping for better efficiency
// 
// 454   2/11/14 10:25 Fseide
// (renamed a variable for more clarity)
// 
// 453   2/07/14 14:52 Fseide
// unquantizeandaggregatestripe() now uses different buffers for each
// stream
// 
// 452   2/07/14 8:55 Fseide
// renamed quantizeaggregatedstripe() to
// quantizeandassignaggregatedstripe() and combined it with its subsequent
// call to assignaggregatedstripe(), for upcoming move to doing this on
// the GPU
// 
// 451   2/05/14 16:53 Fseide
// eliminated resetmomentum flag
// 
// 450   1/27/14 10:14 Fseide
// redefined currently unused function backpropagationmodelupdate2() to
// run after a deferred update but before aggregation, so that we can do
// AdaGrad scaling here.
// 
// 449   1/26/14 11:19 Fseide
// completely moved 'deferupdate' into dbn.h by just not calling aggregate
// and backpropagationmodelupdate3(), and rather have pushtorawgradient()
// and updategradient() control the state
// 
// 448   1/26/14 11:10 Fseide
// next step towards deferred update in raw gradient
// 
// 447   1/26/14 10:49 Fseide
// towards moving deferupdate to the raw gradient
// 
// 446   1/24/14 16:28 Fseide
// number of compute nodes is now dynamically chosen (for small mb sizes,
// less nodes may be faster)
// 
// 445   1/24/14 11:06 Fseide
// towards setting numframes correctly in ...update3()
// 
// 444   1/23/14 14:41 Fseide
// more instrumentation for timing
// 
// 443   1/23/14 11:18a Fseide
// added more instrumentation for time measurements
// 
// 442   1/21/14 4:15p Fseide
// (added to a message)
// 
// 441   1/21/14 11:49 Fseide
// bug fix in skipping the recomputation: forgot to scale to #nodes;
// 'accuracy' now set to 5 stddevs (before: 2), big difference for 16 bit
// in early iterations;
// bg thread disabled for now to get better-comparable log output
// 
// 440   1/20/14 17:04 Fseide
// (added a comment)
// 
// 439   1/20/14 15:04 Fseide
// (added a comment)
// 
// 438   1/18/14 22:10 Fseide
// double-buffering flag now owned by mpiaggregator
// 
// 437   1/18/14 21:45 Fseide
// disabled double-buffering for now
// 
// 436   1/17/14 15:23 Fseide
// enabled doublebuffering for MPI aggregation
// 
// 435   1/17/14 11:31 Fseide
// allocatetransferbuffer() implemented, now uses cudamatrix lib's
// newsharedtransferbuffer() when in CUDA mode
// 
// 434   1/17/14 11:23 Fseide
// GPU buffer allocation now owned by dbn.h, not mpiaggregator (and it
// will be pushed further)
// 
// 433   1/17/14 10:54 Fseide
// MPI: changed bufferbegin/end to bufferbegin/size
// 
// 432   1/17/14 9:21 Fseide
// streamlined enter/exitmpiaggregation() a little, such that on matrix
// level, those functions no longer know about the mpiaggregator
// 
// 431   1/14/14 18:35 Fseide
// now skipping second computerange() (instead, we reuse the range
// determined on our local stripe)--to be verified once we have multiple
// nodes
// 
// 430   1/10/14 16:51 Fseide
// syncassignaggregatedstripeandunquantize() now takes CPU-side buffer so
// that it can operate on the CPU without GPU
// 
// 429   1/10/14 11:01 Fseide
// renamed at/detachmpiaggregator() to enter/exitmpiaggregation()
// 
// 428   1/10/14 10:59 Fseide
// changed all lambdas passed to MPI aggregate() to not capture anything
// by reference (since they will run on a bg thread eventually)
// 
// 427   1/09/14 19:57 Fseide
// completed the MPI aggregation function--it compiles... now it must be
// tested!!
// 
// 426   1/09/14 18:53 Fseide
// partially implemented the second MPI exchange step, incl. buffer
// management
// 
// 425   1/09/14 17:53 Fseide
// began to implement the big fat MPI aggregator using the lambdas
// 
// 424   1/09/14 17:06 Fseide
// last lambda for MPI aggregation implemented, now on to the MPI
// aggregator function itself!
// 
// 423   1/09/14 16:55 Fseide
// quantizeaggregatedstripe() implemented
// 
// 422   1/09/14 15:55 Fseide
// added more lambdas for MPI aggregation
// 
// 421   1/09/14 15:06 Fseide
// towards implementing the lambdas that the MPI aggregator needs
// 
// 420   1/09/14 13:52 Fseide
// quantization now peruses the residual in-place (instead of first adding
// it explicitly to the raw gradient)--better separation of concerns
// (residual belongs to quantization);
// bug fix in quantizeandfetchqstripe(): forgot to apply the patch to the
// residual
// 
// 419   1/09/14 9:17 Fseide
// backpropagationmodelupdate2() now just takes the mpiaggregator directly
// (i.e it now knows that it is for this specific purpose)
// 
// 418   1/08/14 18:25 Fseide
// bug fix: it can now again write IPE-compatible 'linearkind' files that
// arise out of SVD (so we can use their older version to read them)
// 
// 417   1/08/14 8:53 Fseide
// (comment)
// 
// 416   1/08/14 8:52 Fseide
// (comments)
// 
// 415   1/03/14 17:26 Fseide
// initialization sequence of MPI stuff stratified, new methods
// entermpiaggregation() and exitmpiaggregation() in rbm.h, called by
// respective functions in dbn.h;
// moved logic to determine buffer size etc. from CUDA-side qstripe to a
// new CPU-side structure mpistripebufferref (since stripes may live on
// different GPUs, the GPU side cannot do this);
// stripes are now associated with a GPU (in theory--the actual
// determination of the GPU device is not implemented)
// 
// 414   1/03/14 16:01 Fseide
// towards MPI data-parallel gradient aggregation
// 
// 413   1/03/14 10:25 Fseide
// backpropagationmodelupdate() now performs the three sub-steps in
// separate loops, in preparation for whole-model gradient exchange for
// data parallelism
// 
// 412   12/20/13 7:18p V-haofu
// fake split all other versions of backpropagationmodelupdate into 3
// functions;
// for each layer call backpropagationmodelupdate1..3 instead;
// 
// 411   12/09/13 6:22p F-gli
// added commented out code for hidden layer activation stats
// 
// 410   9/29/13 17:16 Fseide
// refined TIME_MODELUPDATE
// 
// 409   9/29/13 17:09 Fseide
// added code for timing backprop
// 
// 408   9/29/13 16:57 Fseide
// input-layer prep now skips the forced sync before preparing the
// minibatch, saves ~8% time
// 
// 407   9/29/13 16:13 Fseide
// added diagnostic code to time the model-update step (disabled)
// 
// 406   9/28/13 20:52 Fseide
// added a comment about possible sync inefficiency with
// accumulatepriors()
// 
// 405   9/28/13 19:11 Fseide
// renamed the buffers that posteriorstats() needs into something more
// opaque
// 
// 404   9/27/13 18:45 Fseide
// model-parallel version of posteriorstats() implemented;
// changed argument order of posteriorstats() CUDA function to be more
// logical
// 
// 403   9/27/13 2:44p V-jiacli
// rename seterrorsignalandtrackingLL() to
// setautoencodererrorsignalandtrackll(), revised some comments
// 
// 402   9/25/13 20:01 Fseide
// added some explicit timing code into forwardprop()
// 
// 401   9/21/13 17:53 Fseide
// (renamed TIME_CUDA to TIME_INLAYER)
// 
// 400   9/20/13 19:02 Fseide
// (added debug code that is disabled)
// 
// 399   9/17/13 7:08p V-jiacli
// added the tracking LL part for auto-encoder
// 
// 398   9/16/13 3:20p Fseide
// disabled #define MULTICUDA and made it compile again (some code was not
// guarded)
// 
// 397   9/16/13 2:54p V-jiacli
// copied the targetfeat to GPU before calculating with it
// 
// 396   9/16/13 9:56a Fseide
// new methods flipsigmoids(), flippolarity() and applytransform() for
// some weird experiment aimed at understanding seeming sparseness
// 
// 395   9/15/13 10:24p V-jiacli
// revised seterrorsignal() for auto-encoder, may still have problem
// 
// 394   9/13/13 3:36p Fseide
// added seterrorsignal(), but wrong
// 
// 393   9/04/13 5:02p V-haofu
// clear all layers before allocation, this might avoid leak of memory
// 
// 392   8/28/13 10:55a Fseide
// model::copyfrom() completed: now clones a layer by creating it with
// dummy args and then calling copyfrom() on it
// 
// 391   8/27/13 8:22p Fseide
// new methods model::backto()/restorefrom() and
// Iannlayer::clone()/copyfrom() to support model backup for lookahead LR
// tuning (clone() and copyfrom() are actually not implemented yet for any
// class)
// 
// 390   8/23/13 1:06p Fseide
// new empty constructor that expects a call to load() before use
// 
// 389   8/07/13 4:09p T-paswie
// added and propagated evalcvmode  flag so forwardprop with dropout for
// cv set gives test-scenario accuracies (instead of dropping, activations
// are scaled by 1-dropoutrate)
// 
// 387   6/28/13 4:34p T-paswie
// added poolSize to maxout constructors, commeting out some debug
// messages
// 
// 386   6/28/13 3:50p T-paswie
// maxouts debugged, looks like are learning something. 
// 
// 385   6/27/13 8:43p T-paswie
// maxouts cont. - implemented, not tested + Frank's comment regarding
// convnolutional nets
// 
// 384   6/10/13 10:14 Fseide
// (fixed signed/unsigned warning)
// 
// 383   6/10/13 10:09 Fseide
// (fixed signed/unsigned warnings)
// 
// 382   6/07/13 20:37 Fseide
// added experimental generalization of recitifed linear units using an
// optional non-linearity (n-th root) and leakiness
// 
// 381   6/06/13 14:34 Fseide
// global model::convparams and its setter removed, convolution-related
// networks now get their config through the layerconfigparameters;
// layerconfigval type casts now implemented, i.e. it is fully functional;
// --convolutionparams is now mapped to the layerconfigparameters
// mechanism, from which 'convolutional' and 'maxpool' will now retrieve
// their parameters
// 
// 380   6/06/13 11:42 Fseide
// now passing prescaledropout flag through (but have not added actual
// cmd-line option yet)
// 
// 379   6/04/13 14:28 Fseide
// (comment)
// 
// 378   6/04/13 14:26 Fseide
// infrastructure for passing creation parameters to network
// instantiations (actual parsing and use to be implemented)
// 
// 377   6/03/13 21:34 Fseide
// renamed class rlu to relunetwork (we will use relu for a non-linearity
// layer for use with CNNs)
// 
// 376   6/03/13 20:39 Fseide
// write() and save() are 'const'--added the const modifier;
// save(path) had lost the checkmodel() call somehow, recovered;
// deleted FlushFileBuffers(h) call from save(h), since it is useless (it
// had no error check--that's what the one in the FILE* version is for)
// 
// 375   6/03/13 8:49 Fseide
// convolution parameters now transported through layercreationargs_t (so
// far as a global, not per-layer as it should be, and still passed to dbn
// as a global setting rather than to individual creation calls);
// grouped the command-line argument variables in prep for moving them
// into structs
// 
// 374   6/03/13 6:59 Fseide
// removed unnecessary layertypes[] arrays inside dbn and evaluator
// (redundant with layer->type());
// now using template for read() and write()
// 
// 373   6/02/13 9:23 Fseide
// class factory now understands "rlu" type
// 
// 372   6/02/13 8:56 Fseide
// new arg --randomize (arg passed through but so far ignored);
// save() call moved out from Do_SVD to main();
// renamed Do_SVD() to converttosvdmodel() to stay within our naming style
// 
// 371   6/02/13 7:25 Fseide
// infrastructure for storing non-linearity kind in the model file;
// removed some duplicate file-reading functions (FILE*, HANDLE) into
// function templates
// 
// 370   6/02/13 3:34 Fseide
// lots of code hygiene for SVD implementation, including rename 'flag'
// (in an interface!) to something meaningful and lots of formatting
// inconsistencies;
// 
// 369   4/09/13 8:53p V-hansu
// (fix some comments)
// 
// 368   4/08/13 8:13p V-hansu
// remove PRINT_UPDATE_STATISTICS and PRINT_GAMMAS_STATISTICS since they
// are no longer used
// 
// 367   4/04/13 10:04a Jianxue
// Add SVD decomposition
// 
// 366   3/20/13 10:58a F-gli
// new methods fdim(0 and augmentationextent()
// 
// 365   3/07/13 11:26a Fseide
// correction to previous check-in comment: It should have been:
// fixed TABs (there should be no TABs in our sources to ensure uniform
// indentation independent of IDE settings);
// changed a few sprintf() to sprintf_s() since that had caused a compiler
// warning for Debug Win32
// 
// 364   3/07/13 11:18a Fseide
// removed unnecessary #include of cudalattice.h
// 
// 363   2/08/13 3:53p F-gli
// [ganl] fix errors to make latgen build
// 
// 362   1/23/13 8:20p V-hansu
// add distribution reallocation for mmi in setgammas(), not activated.
// modify mmidiagnosis()
// 
// 361   1/09/13 5:09p V-hansu
// modify seterrorsignal() to get it work in CUDA mode
// 
// 360   1/09/13 4:54p V-hansu
// modify setgammas() to use cuda for hsmoothing
// 
// 359   1/09/13 4:07p V-hansu
// modify setgammas() to use cuda for computation
// 
// 358   1/06/13 11:07a V-hansu
// remove dropoutrate from setgammas()
// 
// 357   1/05/13 5:25p V-hansu
// rename Hcriteriaweight to hsmoothingweight
// 
// 356   1/05/13 11:15a V-hansu
// add hcriteriaweight to seterrorsignal for smbr with CE 
// 
// 355   1/03/13 8:51p Kaisheny
// Asynchronous SGD using data pipe.
// 
// 354   12/24/12 9:44p V-hansu
// add code for Hcriteria in setgammas()
// 
// 353   12/24/12 7:24p V-hansu
// change the location of code for updating weights of partial states
// 
// 352   12/19/12 10:45p Kaisheny
// Initial version of asynchronous stochastic gradient descent algorithm
// for distributed training of DNN. 
// 
// 351   12/07/12 5:23a Adame
// convolution/maxpool support (GPU only)
// --convolutionalParams flag to support convolution parameters
// --addEnergy flag to add energy to datasets (such as HVT)
// --asyncopy flag to enable asynccopy on multi-GPU setups with pipeline
// trainer
// zero out all arrays on creation (eliminate NANs)
// 
// 350   11/28/12 2:22a V-hansu
// print some stats in settopnsenones()
// 
// 349   11/27/12 6:54p V-hansu
// modify setgammas(), to get senone2update done
// 
// 348   11/27/12 6:28p V-hansu
// modify setgammas() to include senone2update, not debugged through
// 
// 347   11/27/12 3:30p V-hansu
// add method settopnsenones() and pass senone2keepmodelupdate to
// setgammas()
// 
// 346   11/23/12 8:54a Fseide
// (comments)
// 
// 345   11/20/12 16:23 Fseide
// (silly bug in fprintf() format string)
// 
// 344   11/20/12 16:08 Fseide
// fixed collatemixturepriors()
// 
// 343   11/20/12 13:44 Fseide
// lleval command now post-patches the priors since the trainer forgot to
// do that
// 
// 342   11/18/12 4:12p V-hansu
// print some message in unseenstatecompensation()
// 
// 341   11/17/12 7:55p V-hansu
// remove COMPENSATION_STATS but get it as as default.
// 
// 340   11/17/12 4:33p Fseide
// added/fixed comments on unseen-state compensation;
// fixed const correctness in unseenstatecompensation();
// added compensation for deltaa to unseenstatecompensation(), but current
// formula may not be correct
// 
// 339   11/16/12 10:34p V-hansu
// add code to unseenstatecompensation() to print statistics
// 
// 338   11/16/12 7:18p Fseide
// added comments
// 
// 337   11/16/12 7:17p Fseide
// unseenstatecompensation() implemented for new formula
// 
// 336   11/16/12 6:45p Fseide
// added formula to unseenstatecompensation(), but otherwise code has not
// been updated to latest formula
// 
// 335   11/16/12 5:48p Fseide
// forwardpropdelta() now returns its 'eps', which is then passed to
// compensationupdate()
// 
// 334   11/16/12 5:39p V-hansu
// change vnormsbuf into row vectors
// 
// 333   11/16/12 5:34p Fseide
// forwardpropdelta() now takes learning rate per frame and momentum per
// sample;
// so does unseenstatecompensaion(), but that is not fully implemented yet
// 
// 332   11/16/12 5:16p V-hansu
// disable UNSEEN_COMPENSATION
// 
// 331   11/16/12 5:15p V-hansu
// change layerstatenorms to vnormsbufs, and modify forwardpropdelta()
// 
// 330   11/16/12 11:02a V-hansu
// add eh into forwardpropdelta() since it will be used
// 
// 329   11/16/12 10:50a V-hansu
// disable UNSEEN_COMPENSATION
// 
// 328   11/16/12 10:50a V-hansu
// modify scalederrorE to layerstatenorm, and scaleE to vnorms, add method
// allocatestatevectors()
// 
// 327   11/15/12 7:35p V-hansu
// add another matrixref scaledE in forwardpropdelta() to save middle
// results
// 
// 326   11/14/12 7:40p V-hansu
// add UNSEEN_COMPENSATION, not enabled
// 
// 325   11/14/12 7:26p V-hansu
// add deltastate and forwardpropdelta(), not finished
// 
// 324   11/11/12 6:07p V-hansu
// remove verify code to unseenstatecompensation()
// 
// 323   11/11/12 5:17p V-hansu
// change unseenstatecompensation()
// 
// 322   11/10/12 5:20p V-hansu
// change some code in unseenstatecompensation()
// 
// 321   11/10/12 3:25p V-hansu
// remove startlayers i forwardprop, add forwardpropwithoutbias(), modify
// unseenstatecompensation(), but seems still not working
// 
// 320   11/10/12 3:41a V-hansu
// modify unseenstatecompensation()
// 
// 319   11/10/12 1:50a V-hansu
// unseenstatecompensation() finally debugged through...
// 
// 318   11/09/12 11:51p V-hansu
// add startlayer for forwardprop(), finish unseenstatecompensation(), not
// tested
// 
// 317   11/09/12 9:21p V-hansu
// add unseenstatecompensation (), now an empty function
// 
// 316   11/08/12 4:37p T-simonw
// add double precision methods
// documentation
// 
// 315   11/07/12 5:59p V-hansu
// add some code to mmidiagnosis()
// 
// 314   11/07/12 9:43a V-hansu
// add some comments to mmidiagnosis() and make it print to
// mmidiagnosis.log
// 
// 313   11/07/12 8:43a V-hansu
// add mmidiagnosis() to check gammas and ce posterior
// 
// 312   11/06/12 4:12p T-simonw
// move hessianfreeupdate from dbn.h to hessianfreetrainer.h
// 
// 311   11/05/12 1:37a V-hansu
// add dropout rate to refmodel
// 
// 310   11/05/12 12:32a V-hansu
// comment out #ifdef HF to make dbn build.
// make KL regularization for mmi training now, but not there is another
// bug relating to read-ahead thread in mmi reg mode
// 
// 309   11/03/12 15:04 Fseide
// put a HF-related function within #ifdef HF/#endif, since it broke the
// HAPI build
// 
// 308   11/02/12 11:09a Fseide
// splitlayer() now also splits the priors --this needs some cleanup!
// 
// 307   11/02/12 10:48a Fseide
// split() now works for output layer as well
// 
// 306   11/01/12 12:23p T-simonw
// hessianvectorproduct: featsource is unique_ptr again (instead of
// shared_ptr)
// 
// 305   11/01/12 9:24a Fseide
// fixed mixture models (forgot the weighing in my derivation...)
// 
// 304   10/31/12 10:01a T-simonw
// change in interface of trainer: entercomputation gets integer instead
// of boolean istoplayer
// add Hessian free optimization methods
// allow for less verbose logging
// 
// 303   10/21/12 10:00a V-hansu
// modify setgammas() to make build
// 
// 302   10/19/12 8:58p V-hansu
// add method setgammas for mmi error back
// 
// 301   10/17/12 5:00p Dongyu
// fixed several copy&paste errors in the dropout code. 
// 
// 300   10/15/12 5:43p Fseide
// summixturecomponents() moved into evaluator class
// 
// 299   10/14/12 1:39p Fseide
// bug fix in scattermixtureerrorsignals()--we were updating the wrong
// variable... :( copy-paste error
// 
// 298   10/13/12 8:33p Fseide
// added experimental functions for mixture experiments:
// summixturecomponents() and scattermixtureerrorsignals()
// 
// 297   10/12/12 4:47p T-simonw
// forwardprop: only divide by std deviation if it is non-zero (otherwise
// it is a constant feature)
// add iscomputing() method
// 
// 296   10/12/12 1:47p Dongyu
// added support of dropout training for DNN (frame level training only). 
// addes support to convert the model based on dropout rate used in the
// training and/or senone sections used in multilingual training.
// 
// 295   10/10/12 10:00a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new langauges.
// 
// 294   10/05/12 4:21p Fseide
// moved the hack for DPT out again from addlayer(), now done in
// trainlayer() directly
// 
// 293   10/05/12 2:47p Fseide
// addlayer() now has a "special" mode to detect creation of the top layer
// after DPT--in which case we will find a left-over softmax layer that
// must be repealed and replaced by a fresh hidden sigmoid layer
// 
// 292   10/05/12 1:11p Fseide
// renamed accumulator constructor parameter 'istoplayer' to 'bpmode', and
// it is no longer saved since it only affects construction
// 
// 291   10/04/12 5:38p Fseide
// create() now only infers that we want fDLR mode for square
// matrices--not nice, better have additional parameters for this, or an
// injectlinearlayer function that does this explicitly
// 
// 290   10/04/12 4:02p Fseide
// another silly error in addlayer()--getting old!
// 
// 289   10/04/12 3:35p Fseide
// addlayer() fix--forgot to save the layer pointer...
// 
// 288   10/04/12 1:56p Fseide
// addlayer() now obeys the layertype parameter (if not given, then fall
// back to old behaviour which will set up a default system)
// 
// 287   9/27/12 3:21p Fseide
// enablesse now activated only if no CUDA device present, otherwise using
// CUDA (to support the monster models)
// 
// 286   9/25/12 1:39p Fseide
// logLL() in nosoftmax mode now normalizes the log priors by their
// maximum, to keep values in a more normal range;
// logPuV() now supports nosoftmax mode
// 
// 285   9/25/12 11:56a Fseide
// logLL() now checks the posterior to be 0 --finally it happened (not
// known yet whether that model is just so good...)
// 
// 284   9/25/12 11:22a Fseide
// checkmodel() now checks the priors as well
// 
// 283   9/25/12 11:16a Fseide
// added a sanity check for NaNs and INF for models when loaded or saved
// 
// 282   9/23/12 8:03p Fseide
// initialization of bpinfo now lifted out into trainlayer(), to make it
// easier to add new options, such as AdaGrad
// 
// 281   9/23/12 7:20p Fseide
// (broken temp checkin since suddenly the machine became unusably slow)
// 
// 280   9/21/12 3:24p Fseide
// added nosoftmax mode, to speed up sequence training by bypassing the
// unnecessary expensive softmax() computation
// 
// 279   9/21/12 2:49p Fseide
// seterrorsignal() (assignment version) no longer copies element by
// element but calls assign(), which can directly copy to GPU side if in
// CUDA mode
// 
// 278   9/21/12 2:26p Fseide
// backpropagationstatsmmi2() renamed to seterrorsignalmmi();
// backpropagationstatssmbr2() renamed to seterrorsignal()
// 
// 277   9/21/12 2:17p Fseide
// renamed pretrainingstats2() to pretrainingstats()
// 
// 276   9/21/12 2:14p Fseide
// renamed backpropagationstats3() to errorbackprop()
// 
// 275   9/21/12 2:11p Fseide
// renamed backpropagationstats2() to seterrorsignal()
// 
// 274   9/21/12 2:06p Fseide
// renamed backpropagationorpretrainingstats1() to forwardprop()
// 
// 273   9/17/12 3:32p Fseide
// bug fix in backpropagationmodelupdate()--tested the model type by
// accessing the layertypes[] array, which is inconsistent after adding a
// layer (TODO: should we actually keep it at all?)
// 
// 272   9/08/12 6:40p V-hansu
// remove some PRINT lines and add a function backpropagationstatssmbr2
// for smbr error assignment
// 
// 271   9/03/12 9:08p V-hansu
// add a TODO comment
// 
// 270   9/02/12 6:19p Fseide
// implemented I-smoothing
// 
// 269   9/02/12 5:47p Fseide
// regularizationtype::alpha renamed to more concise L2weightpersample
// 
// 268   9/02/12 5:38p Fseide
// (minor cleanup in setting up the bpinfo struct)
// 
// 267   8/28/12 1:56p V-hansu
// add some codes to PRINT_GAMMAS_STATISTICS
// 
// 266   8/27/12 3:28p V-hansu
// add funcation dumpstatistics and change PRINT_MMI_STATISTICS to
// PRINT_UPDATE_STATISTICS
// 
// 265   8/27/12 7:44 Fseide
// (editorial; #if 0'ed out an unused function)
// 
// 264   8/26/12 6:32p V-hansu
// modify backpropagtionstatsmmi2 to make compatible with sMBR mode
// 
// 263   8/25/12 4:08p V-hansu
// update for sMBRmode (not tested yet)
// 
// 262   8/24/12 3:58p V-hansu
// modify some codes relating to PRINT_MMI_STATISTICS
// 
// 261   8/21/12 10:50p V-hansu
// add PRINT_MMI_STATISTICS to print statistics in
// backpropagationstatsmmi2
// 
// 260   8/20/12 8:06p V-hansu
// modify insertlayer to let layertypes change over operation
// 
// 259   8/17/12 9:39p V-hansu
// add a macro and some code to print statistics relating to mmi and
// framebased training
// 
// 258   8/16/12 8:20p V-hansu
// add a member (layertypes) to model class to instruct
// backpropagationmodelupdate
// 
// 257   8/15/12 10:16a V-hansu
// change some indentation
// 
// 256   8/15/12 10:15a V-hansu
// modify some indentation
// 
// 255   8/07/12 18:14 Fseide
// completed implementation of deferupdate flag (still to be tested)
// 
// 254   8/07/12 17:46 Fseide
// new option to backpropagationmodelupdate(): deferupdate, used to
// implement batches of batches, for an MMI experiment
// 
// 253   8/07/12 5:06p V-hansu
// remove INJECTTOPSECONDLAYER macro, modify insertlayer, combine
// injectbottomltlayer and injecttopsecondlayer into injectlinearlayer
// 
// 252   8/07/12 10:17 Fseide
// itrainer now has a virtual destructor
// 
// 251   8/07/12 9:55 Fseide
// moved itrainer here
// 
// 250   8/07/12 9:15 Fseide
// (fixed a few compiler warnings)
// 
// 249   7/23/12 10:53a V-hansu
// add macro INJECTTOPSECONDLAYER to do adaptation using top second layer
// 
// 248   7/19/12 11:46a Adame
// update copydata() with the correct parameter (rest of no-sync
// framework)
// 
// 247   7/19/12 11:43a V-hansu
// modified checkdimensions function
// 
// 246   7/06/12 9:16p V-hansu
// modify the way of checking dimensions
// 
// 245   7/05/12 8:00p V-hansu
// chang the interface of blow up to let it able to return roundup unit
// 
// 244   7/02/12 4:27p V-hansu
// add function setlinearlayerweight to use GMM adaptation matrix to
// initialize
// 
// 243   6/27/12 9:23p V-hansu
// add a print(FILE *f) function for debugging
// 
// 242   6/24/12 9:24p V-xieche
// switch code into a work point(an old version as well).
// 
// 241   6/22/12 2:12p V-hansu
// changed the interface of function blowup
// 
// 240   6/18/12 7:59p Adame
// pinned memory update
// 
// 239   6/08/12 8:36p V-xieche
// add a flag to decide to use async copy or sync copy. Need to improve it
// later.
// 
// 238   6/05/12 4:22p V-hansu
// add a blowup function to "model" class to do class-based adaptation
// 
// 237   6/02/12 8:21p V-xieche
// delete the code related to DELAYUPDATE and TIMESTATS.Not use them any
// more. 
// 
// 236   5/31/12 10:54p V-xieche
// fix a bug in exitcomputation function for more than 2 cuda devices on
// top layer
// 
// 235   5/13/12 11:00p V-xieche
// modify entercomputation for pipeline training to make it support more
// than 2 cuda devices.
// 
// 234   5/11/12 4:40p V-xieche
// remove class compacttrainer and complex trainer. do not need it anymore
// 
// 233   4/18/12 4:38p V-xieche
// clean up the code related to macro DELAYUPDATE
// 
// 232   4/18/12 4:00p V-xieche
// clean up all code related to target propagation.
// 
// 231   4/18/12 2:04p V-xieche
// clean up the code in block of #ifdef TARGETBP #endif 
// 
// 230   4/10/12 7:18p V-xieche
// add interface function to get mean, std, Pu and layers of model, used
// in class dbnfasttrain.
// 
// 229   4/07/12 8:31p V-xieche
// fix a bug for accumulate prior probability during training. And
// verified its correctness.
// 
// 228   4/07/12 2:13p V-xieche
// modify the code to make posterior in each cuda parallelly.
// 
// 227   4/07/12 2:05p V-xieche
// fix a bug when computing logpps and pps for striped top layer. Now
// logpps and pps are correct, but need to make them parallel computing in
// multi devices.
// 
// 226   4/06/12 6:25p V-xieche
// Add codes for posteriorstats function for striped top layer. not
// finished yet.
// 
// 225   4/05/12 9:50p V-xieche
// add code for accumulate prior and posteriorstats in striped toplayer
// pipeline training. not finished yet.
// 
// 224   4/04/12 10:07p V-xieche
// add some commend and delete some debug code and old code won't use
// anymore.
// 
// 223   4/03/12 8:30p V-xieche
// check in all code for supporting pipeline training. striped top layer
// in two cuda devices. need to add comment and rewrite it to make it easy
// to read.
// 
// 222   3/27/12 1:19a V-xieche
// Add codes for pipeline training with multi cuda devices. Need to add
// comments later.
// 
// 220   3/16/12 2:23a V-xieche
// delete some debug output
// 
// 219   3/16/12 2:12a V-xieche
// use fetch function to get thereference from cuda, the time used for
// training in compacttrainer is correct now.
// 
// 218   3/14/12 11:59p V-xieche
// no need copy operation for pipeline training anymore. and it is
// verified as correct; add macro TIMESTATS
// 
// 217   3/14/12 12:45a V-xieche
// Modified the code for pipeline training for compact trainer. It works
// now.
// 
// 216   3/12/12 10:17p V-xieche
// modify some bugs in pipeline training for compact trainer.
// 
// 215   3/11/12 9:46p V-xieche
// modify code for pipeline training for compact trainer.
// 
// 214   3/11/12 7:14p V-xieche
// add macro TIMESTATS to output time distribution.
// 
// 213   3/11/12 7:05p V-xieche
// add code for a compact trainer. make it run in CUDA directly.
// 
// 212   3/08/12 10:35p V-xieche
// add code to make forward and backward prop do in CUDA directly.
// verified the training is correct, while speed faster than previous.
// need to debug it.
// 
// 211   3/06/12 10:51p V-xieche
// add code for compact trainer. Not finished.
// 
// 210   3/05/12 9:10p V-xieche
// Add code for compact trainer to simplify the implmentation of DNN
// trainer(MACRO COMPACTTRAINER), to make it purely on CUDA and prepare
// for pipeline training in multiply CUDA device.
// 
// 209   2/26/12 8:46p V-xieche
// Add macro COPYINCUDA_FORDELAYUPDATE_V2, copy all data in CUDA device
// directly now.
// 
// 208   2/26/12 6:58p V-xieche
// Add codes for coping date between CUDA device.
// 
// 207   2/24/12 11:16p V-xieche
// Add code to assign value in CUDA directly for delayupdate training. not
// finished yet.
// 
// 206   2/23/12 5:47p V-xieche
// fix bugs exist in previous code for delay update mode.
// 
// 205   1/06/12 2:33p Fseide
// (fixed the last change)
// 
// 204   1/06/12 2:14p Fseide
// added a check and optional resize to setprior()
// 
// 203   1/05/12 7:34p Fseide
// fixed MMI error-signal update with interpolation
// 
// 202   1/04/12 5:42p Fseide
// bugfix in dropframes(), now only sub-samples top-level error vector
// (lower layers are still virgin and need no processing)
// 
// 201   1/04/12 4:59p Fseide
// new method dropframes()
// 
// 200   1/03/12 11:21a Dongyu
// add support to use simple layer type names. "sm" for softmax, "gb" for
// gaussianBernoulli, "bb" for BernoulliBernoulli, "tn" for tensor
// network, and "ln" for linear network.
// 
// 199   1/01/12 10:01p F-gli
// 
// 198   1/01/12 3:03p Fseide
// split backpropagationstats2() into backpropagationstats2() and
// backpropagationstats3(), where the former sets the top-level error
// signal, and the latter propagates it through;
// new method backpropagationstatsmmi2() to set top-level error signal
// from numer and denom lattice posteriors
// 
// 197   11/23/11 4:32p Dongyu
// modified rbmbase factory to iannlayer factory. support layers with more
// than one network parts.
// 
// 196   11/17/11 14:05 Fseide
// experimental: setdenominatorgammas() allows to interpolate with frame
// posteriors
// 
// 195   11/17/11 11:45a V-xieche
// modify a potential bug for steeper sigmoid training of deep
// model(layernum > 2).
// 
// 194   11/16/11 11:57p V-xieche
// Modify the code to implement swap function in delay update model
// correctly. After swap, layerstate and errorstate will empty. Need to
// change code where use them.
// 
// 193   11/16/11 8:01 Fseide
// new method setdenominatorgammas() for MMI update
// 
// 192   11/15/11 8:44p V-xieche
// fix a minor bug for delay update model. clear prevuids when entering a
// new epoch.
// 
// 191   11/14/11 4:03p V-xieche
// fix a bug, previous code ignored that the size of last minibatch in a
// data sweep maybe smaller than other minibatches. Also add code for
// recover model from flatter or steeper sigmoid to continue training. 
// 
// 190   11/12/11 12:29a V-xieche
// fix some bugs in delayupdate version 2. utilized the time delay to
// update each layer independently in the same time slot. seems correct
// from the first minibatchs, need to test it futher.
// 
// 189   11/11/11 16:01 Fseide
// factored forwardprop() out from logPuv()
// 
// 188   11/11/11 15:10 Fseide
// merged the two identical stats1 functions into one
// backpropagationorpretrainingstats1() --still need find a better name
// 
// 187   11/11/11 14:47 Fseide
// pretrainingstats() split into two functions, in prep of generalizing
// the structure a little w.r.t. the changes for MMI training
// 
// 186   11/11/11 14:37 Fseide
// split backpropagationstats() into two sub-functions
// 
// 185   11/08/11 8:06p V-xieche
// fix some bugs for delayupdate_v2, without runtime error now. still not
// complete. 
// 
// 184   11/07/11 11:13p V-xieche
// Fix a minor bug when do update delay in forwardprop. Not finished yet.
// 
// 183   11/06/11 10:12p V-xieche
// add some comment and code for delay update model. not complete yet.
// 
// 182   11/05/11 8:10p V-xieche
// add code for delay update model in code block DELAYUPDATE_V2
// 
// 181   11/03/11 15:18 Fseide
// momentum now passed down to network-update functions as momentum per
// sample, in prep for also taking the scaling out of the gradient
// 
// 180   11/01/11 11:36p V-xieche
// fix a minor bug for delay update model (considerate delay momentum as
// well)
// 
// 179   10/31/11 9:01p V-xieche
// add code for simple experiment of delay update models.
// 
// 178   10/28/11 14:51 Fseide
// formal change to use the new otherweight parameters in model update
// (but currently passing 1.0)
// 
// 177   10/28/11 13:36 Fseide
// changed 'momentum' to 'double' in prep of pushing in the scaling
// 
// 176   10/25/11 5:17p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 175   10/20/11 10:54a V-xieche
// modify the code for flat sigmoid in multi layer model
// 
// 174   10/18/11 9:06p V-xieche
// modify the code to implement a true steeper or flat sigmoid function.
// i.e. scale the bias as well
// 
// 173   10/11/11 3:22p V-xieche
// modify the code for setting output of hidden layer below specific value
// to zero.
// 
// 172   10/11/11 15:13 Fseide
// moved classes optimizedmlp and ondemandevaluator out from dbn.h to a
// separate header file (dnnruntime.h)
// 
// 171   10/11/11 12:11 Fseide
// (added some stats that we probably won't need anyway...)
// 
// 170   10/11/11 11:38a V-xieche
// add code for setto0ifbelow for the output of hidden layer.
// 
// 169   10/11/11 8:21 Fseide
// fixed a bunch of compiler warnings
// 
// 168   10/11/11 8:12 Fseide
// added percentile for quantization range;
// added experimental code for trying quantization strategies without
// implementing the data types
// 
// 167   10/10/11 10:36 Fseide
// added a hack to test quantization by quantizing the original matrix
// 
// 166   10/09/11 15:46 Fseide
// (added timing of DBN evaluations)
// 
// 165   10/09/11 11:42 Fseide
// now prints LL statistics
// 
// 164   10/08/11 15:26 Fseide
// SSE disabled (does not work as of now)
// 
// 163   10/08/11 14:19 Fseide
// (partially debugged and fixed the quantized version)
// 
// 162   10/08/11 10:32 Fseide
// (fixed minor compiler issues)
// 
// 161   10/08/11 10:17 Fseide
// added new class 'optimizedmlp' which implements quantized storage and
// SSE evaluation of the MLP (not tested yet; SSE not implemented yet)
// 
// 160   10/08/11 8:38 Fseide
// ondemandevaluator now moved out and compiles again
// 
// 159   10/08/11 8:32 Fseide
// ondemandevaluator moved out from dbn::model to standalone class, aiming
// at splitting it off into a separate source file
// 
// 158   10/06/11 5:17p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 157   9/28/11 10:05p V-xieche
// modify a minor bug for log(1+sigmoid(z)) in code. get the output of
// sigmoid before backpropagationstats
// 
// 156   9/26/11 8:42p V-xieche
// Add some codes for log(sigmoid + epison) experiment.
// 
// 155   9/22/11 9:23p V-xieche
// fix a bug for deeper sigmoid experiment.
// 
// 154   9/20/11 2:45p V-xieche
// a minor modification
// 
// 153   9/19/11 10:54p V-xieche
// delete some debug code.
// 
// 152   9/19/11 10:48p V-xieche
// Add some code for using steeper sigmoid function, simulating the hard
// decision function.
// 
// 151   8/23/11 7:59p V-xieche
// delete a temp debug line.
// 
// 150   8/23/11 7:57p V-xieche
// add margin-based training code for dbn according to Heigold's thesis.
// 
// 149   8/21/11 4:57p V-xieche
// add some code for target propagation version 5, it try to modify the
// weight according the normal BP algorithm, to see whether it works.
// 
// 148   8/18/11 11:03p V-xieche
// add code to set the target feature assign to weight vector.
// 
// 147   8/16/11 10:35p V-xieche
// add code for targetpropagation v4.
// 
// 146   8/15/11 10:58p V-xieche
// fix a minor bug when statistic the correct ratio for weight vector lies
// in decision region experiment
// 
// 145   8/15/11 10:30p V-xieche
// add code to statistic the ratio the top layer weight matrix lies in
// their class decision region
// 
// 144   8/09/11 7:41a Dongyu
// add support to output log p(u|v) (i.e., wighout div by prior). This is
// useful for some applications.
// 
// 143   8/02/11 12:30a V-xieche
// add the function for target propagation for b=w*h instead of h. i.e
// targetbpv2
// 
// 142   7/29/11 5:59p V-xieche
// move logmaxpp out of evaluator class. add some debug code to verify the
// target feature is correct. add code to verify target propagation could
// decrease the square error when updating bottom layer
// 
// 141   7/29/11 8:25a Dongyu
// Now it supports automatic learning rate adjustment based on cross
// validation set.
// 
// 140   7/28/11 8:51p V-xieche
// add getandupdatetargetfeatstats() function for get and update the
// target feature for that minibatch if neccessary
// 
// 139   7/28/11 2:33p V-xieche
// add some indicatioin modified by v-xieche
// 
// 138   7/27/11 9:22p V-xieche
// Add the code for target propagation(to be debugged). in the #ifdef
// TARGEBP #endif block. output the maxpp for each class in the corpus.
// modify the original BP function for the purpose of target propagation
// 
// 137   7/25/11 8:54p V-xieche
// modify some minor place such as checknotcoputing or TAB format..Add
// backpropagationstats_quan for the binarizing the output of hidden layer
// when updating top layer.
// 
// 136   7/23/11 5:13p V-xieche
// Add getweight and getbias function to get value from a specific
// location of a specific layer.
// 
// 135   7/20/11 1:08p V-xieche
// Add a function dumplayer to output a specific layer element in matlab
// format.
// 
// 134   7/13/11 21:09 Fseide
// removed realtime log messages
// 
// 133   7/13/11 19:01 Fseide
// on-demand mode enabled in ondemandevaluator
// 
// 132   7/13/11 17:42 Fseide
// ondemandevaluator: te is now hidden
// 
// 131   7/13/11 17:18 Fseide
// completed the refactoring into ondemandevaluator
// 
// 130   7/13/11 10:33 Fseide
// some refactoring to split realtimeevaluator into (1) frame buffering
// and (2) shared-state buffering, the latter being moved into new class
// ondemandevaluator
// 
// 129   7/12/11 19:38 Fseide
// bug fix in shiftleft()
// 
// 128   7/12/11 10:15 Fseide
// removed an unnecessary refreading in evaluateshared()
// 
// 127   7/12/11 9:31 Fseide
// added some debug code to logll()
// 
// 126   7/11/11 20:24 Fseide
// new methods for realtime recognition: evaluateshared(), logLL(),
// shiftleft()
// 
// 125   7/11/11 11:17a V-xieche
// Add the function for cheating experiment on hidden layer
// 
// 124   7/07/11 12:11 Fseide
// fixed the momentum bug in the refactoring that was the bug fix for
// 1-frame minibatches
// 
// 123   7/07/11 11:26a V-xieche
// modify in the hist statistic code, change  the point to matrix class
// for consistent and safe consideratioin
// 
// 122   7/07/11 10:14a V-xieche
// modify a bug use new but use free to delete
// 
// 121   7/06/11 11:29p V-xieche
// add some code in the #if 0 #endif block for the histgoram stats.
// 
// 120   7/06/11 13:52 Fseide
// pushed weighting of learning rate through to rbm
// 
// 119   7/06/11 13:44 Fseide
// moved momentumfiltergain also through from main.cpp to dbn.h, still all
// needs to be further moved into rbm.h
// 
// 118   7/06/11 13:35 Fseide
// learningrate is now passed on to DBN as learning rate / sample (but
// still with momentum gain)
// 
// 117   6/21/11 9:25p V-xieche
// set poolblocks to be true for generate the pooled-diag matrix
// 
// 116   6/21/11 10:04a V-xieche
// change in function insertlayer, the parameter previous is string, now
// to const string & to avoid the copy cost.
// 
// 115   6/20/11 12:32p V-xieche
// modify the insertlayer funtion don't accord to the previous code;
// change the char * to string type; modify a mis-spell "perceptron".
// 
// 114   6/20/11 7:23 Fseide
// factored network construction by type string out from dbn.h into a
// factory class in rbm.h where it belongs
// 
// 113   6/19/11 3:45p V-xieche
// Modify some minor think in the code and an else in case there is an
// exception.
// 
// 112   6/19/11 2:43p V-xieche
// Add the function insertlayer(), add a givin type network in a given
// layer. 
// mainly for the linearnetwork at the 0 layer for F'ML'L
// RTODO: modify the read and write model  function for linearnetwork.
// 
// 111   6/18/11 16:50 Fseide
// now able to read linearnetwork layers;
// towards injecting a new linearnetwork layer (not completed)
// 
// 110   6/18/11 16:36 Fseide
// backpropagationmodelupdate() now has new arg to restrict updates to a
// single layer;
// new method injectbottomltlayer() (so far empty, to be completed)
// 
// 109   6/17/11 11:08 Fseide
// (added a comment)
// 
// 108   6/13/11 9:48 Fseide
// new method hdim() to support bottleneck features
// 
// 107   6/13/11 9:31 Fseide
// removed an unnecessary and wrong assert() from eval::evaluate()
// 
// 106   6/12/11 18:51 Fseide
// extended evaluate() to be useful for bottleneck features;
// forwardprop() now takes a layer parameter to enable evaluation of only
// part of the stack for bottleneck features;
// [based on code by Nikko Strom]
// 
// 105   6/10/11 8:03 Fseide
// can now compile without Matlab I/O support--just don't #include
// "matio.h" before including this;
// added a missing #include <regex>
// 
// 104   5/23/11 11:17a Fseide
// trainlayer() now logs frame-correct rate as well on the epoch level
// 
// 103   5/10/11 7:41a Fseide
// (refined logging of stats)
// 
// 102   5/09/11 4:20p Fseide
// fixed printvaluedistribution() hack for pre-training stage
// 
// 101   5/09/11 15:24 Fseide
// added a hack to printvaluedistribution() to zero-normalize the biases
// for analysis
// 
// 100   4/18/11 8:55a Fseide
// llstats() now returns binary stats for Bernoulli-Bernoulli layer
// (before it was always Gaussian in keeping with the original Python
// script)
// 
// 99    3/23/11 4:00p Fseide
// new methods overridelayer() and evaluate()
// 
// 98    3/17/11 9:17p Fseide
// new overload for accumulatepriors() that performs forwardprop()
// 
// 97    3/05/11 8:30p Fseide
// printmatvaluedistribution() and checkmodel() now compute/print the
// overall number of non-null parameters in aggregate
// 
// 96    3/04/11 6:17a Dongyu
// added model weight distribution analysis and dumping functionality
// through the "checkmodel" switch
// 
// 95    3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 94    2/25/11 7:47p Fseide
// new method evaluator::synchronize() --call this if accurate time
// measurements are desired right before reading out the timer
// 
// 93    2/23/11 1:46p Fseide
// (cosmetic change)
// 
// 92    2/10/11 3:44p Fseide
// (cosmetic change to a log message)
// 
// 91    2/10/11 1:00p Fseide
// posteriorstats(): temp vectors changed from column to row vectors
// 
// 90    2/10/11 12:37p Fseide
// posteriorstats() factored into rbmstatevectorsref
// 
// 89    2/10/11 11:55a Fseide
// factored out uidsstripe()
// 
// 88    2/10/11 10:53a Fseide
// seterrorsignal() working now
// 
// 87    2/10/11 10:32a Fseide
// moved error-signal computation to rbmstatevectors, for future CUDA
// implementation
// 
// 86    2/10/11 10:03a Fseide
// accumulatepriors() now operates in CUDA  --no measurable time
// difference
// 
// 85    2/09/11 12:22a Fseide
// fixed the resize(0,0) of the lowest BP layer to resize(0,nfwd)
// 
// 84    2/08/11 2:19p Fseide
// commented out some checknan() calls
// 
// 83    2/07/11 9:31p Fseide
// moved llstats() into rbmstatevectorsref, to allow acceleration by CUDA
// 
// 82    2/07/11 7:20p Fseide
// added test code to find out that llstats() takes a lot of time!
// 
// 81    2/07/11 4:28p Fseide
// (minor cleanup)
// 
// 80    2/07/11 3:29p Fseide
// updated all code related to rbmstatevectors to follow the new locking
// scheme
// 
// 79    2/07/11 1:50p Fseide
// adapted to changes in rbmstatevectorsref creation
// 
// 78    2/06/11 3:18p Fseide
// (added a comment)
// 
// 77    2/05/11 9:29p Fseide
// added a comment
// 
// 76    2/05/11 8:22p Fseide
// added lock/unlock calls for rbmstatevectors, which may be CUDA-based
// 
// 75    2/05/11 6:58p Fseide
// added a few comments and assertions
// 
// 74    2/03/11 9:30p Fseide
// evaluator no longer cares whether the model is in 'computing' state or
// not at the time of instantiation
// 
// 73    2/02/11 10:46a Fseide
// reenabled the checknan() calls because we can use them again
// 
// 72    2/02/11 10:26a Fseide
// (previous check-in) matrixstripe changed to rbmstatevectorsref;
// removed unused transfer() and import() functions
// 
// 71    2/02/11 10:21a Fseide
// 
// 70    2/02/11 9:24a Fseide
// changed layer and error state to rbmstatevectors (not working yet
// because of striping... meh)
// 
// 69    1/28/11 16:37 Fseide
// bug fix: enter/exitcomputation() now actually set/clear the 'computing'
// flag...
// 
// 68    1/28/11 15:16 Fseide
// changed the various rbmXXX(W,a,b) constructors to take rvalue
// references
// 
// 67    1/28/11 14:52 Fseide
// added a flag 'computing' to ensure the state
// 
// 66    1/28/11 14:43 Fseide
// further tidying-up, clean-up, moving-around, commenting as prep for
// CUDA transition
// 
// 65    1/28/11 14:22 Fseide
// (some moving-around of code)
// 
// 64    1/28/11 11:08 Fseide
// removed copy construction
// 
// 63    1/28/11 10:48 Fseide
// added enter/exitcomputation();
// deleted some old code related to old, frame-wise parallelization
// 
// 62    1/21/11 16:08 Fseide
// pretraining/backpropagationupdate() now take the stripe, allowing for a
// static trainer object
// 
// 61    1/19/11 10:03a Fseide
// 
// 60    1/14/11 10:20p Fseide
// (minor fix in counting frame errors: >= instead of >)
// 
// 59    1/14/11 9:25p Fseide
// added the "optimizations" according to the "BP tricks" document (need
// to find the author and correct title!)
// 
// 58    1/05/11 3:57p Fseide
// changed 'evaluator' and 'accumulator' class to systematically work on
// [ts,te), not [0,te-ts), for any of its arguments (for NUMA efficiency)
// 
// 57    1/05/11 12:11p Fseide
// backpropagateprepare now operating striped for optimal NUMA performance
// 
// 56    1/04/11 9:45p Fseide
// new method copyfrom() to reclone without mem allocation (for NUMA)
// 
// 55    12/21/10 18:37 Fseide
// added experimental functionality to "split" a hidden layer by doubling
// its number of hidden nodes
// 
// 54    12/15/10 11:48 Fseide
// new method accumulatepriors()
// 
// 53    12/09/10 9:06p Fseide
// using more digits in llstats() so we can see a numeric error better
// 
// 52    12/08/10 3:24p Fseide
// posteriorstats() now limits the pp value before taking log, as to avoid
// taking log(0), which will screw up the entire epoch's reported LL
// 
// 51    12/07/10 8:17 Fseide
// fixed the comment in llstats()--the normalization we use does make
// sense after all! :)
// 
// 50    12/06/10 4:07p Fseide
// cross-entropy now normalized to be better readble (changes more
// visible), but we are not using this anyway as the global metric
// 
// 49    12/06/10 15:51 Fseide
// bug fix in pretraining_accumulate(): randomseed now also depends on
// epoch (before we decided the same binary random value across epochs)
// 
// 48    12/05/10 2:14p Fseide
// (added a log message)
// 
// 47    12/05/10 13:59 Fseide
// implemented limiting BP for only the top N layers
// 
// 46    12/03/10 12:03 Fseide
// new method getpriors()
// 
// 45    11/30/10 14:56 Fseide
// new method setprior()
// 
// 44    11/30/10 11:38 Fseide
// llstats() now computes the square error in all cases, per
// recommendation by Dong
// 
// 43    11/30/10 9:12 Fseide
// pretrainingprepare() now separate from backpropagationprepare()
// (although doing the same)
// 
// 42    11/30/10 8:41 Fseide
// new constructor to create an empty model only with mean and std;
// adapted read() and write() to handle empty models correctly
// 
// 41    11/30/10 7:43a Fseide
// (llstat now logs the per-slice value)
// 
// 40    11/30/10 7:30a Fseide
// llstats() now clipping Pv1 because I did observe 1.0
// 
// 39    11/30/10 6:42 Fseide
// new method llstats() to track progress of pretraining
// 
// 38    11/29/10 16:59 Fseide
// implemented transfer() for pre-training;
// shedlayers() now clears out Pu
// 
// 37    11/29/10 16:10 Fseide
// incompatible file-format change: now each layer carries a type tag;
// addlayer() now creates Pu for top layer
// 
// 36    11/29/10 15:43 Fseide
// implemented addlayer();
// new method shedlayers()
// 
// 35    11/29/10 13:07 Fseide
// pretraining implemented in accumulator and trainer (except for
// transfer() function)
// 
// 34    11/29/10 9:25 Fseide
// 'trainer' now knows what it is training
// 
// 33    11/29/10 9:15 Fseide
// partial infrastructure for pretraining
// 
// 32    11/26/10 17:52 Fseide
// (new method numlayers())
// 
// 31    11/26/10 17:13 Fseide
// starting with infrastructure of pretraining
// 
// 30    11/25/10 15:06 Fseide
// added 'momentum' parameter to backpropagateupdate()
// 
// 29    11/24/10 7:50 Fseide
// fixed some int/size_t correctness
// 
// 28    11/24/10 7:23 Fseide
// added functions for file I/O
// 
// 27    11/23/10 4:51p Fseide
// (minor tweak to a log message)
// 
// 26    11/23/10 16:17 Fseide
// posteriorstats() now prints frame accuracy instead of error
// 
// 25    11/23/10 15:59 Fseide
// new method posteriorstats()
// 
// 24    11/23/10 11:11a Fseide
// new method importlayerstate()
// 
// 23    11/23/10 10:56a Fseide
// new method transferlayerstate()
// 
// 22    11/23/10 9:54a Fseide
// removed the call to forwardprop() from backpropagateerror(), for
// experiments with NUMA efficiency (we may move it back later)
// 
// 21    11/23/10 8:52 Fseide
// trainer split into two instantes--one per-thread with local NUMA RAM,
// and one shared that gets the result (lwo bandwidth) copied into;
// backpropagateprepare() moved from trainer to model itself
// 
// 20    11/19/10 16:10 Fseide
// added backpropagateprepare()
// 
// 19    11/19/10 12:34 Fseide
// bug fix in backpropagateerror(): now correctly handles the empty
// errorstate[0]
// 
// 18    11/19/10 11:11 Fseide
// (fixed the sparseness hack--not used at the moment)
// 
// 17    11/19/10 10:58 Fseide
// fixed some assert()-related code
// 
// 16    11/19/10 10:56 Fseide
// redesigned interface to back-propagation to avoid locks
// 
// 15    11/19/10 8:24 Fseide
// (fixed an assert())
// 
// 14    11/19/10 7:43 Fseide
// renamed 'toprbm' to 'perceptron' which is more yet not fully accurate
// 
// 13    11/19/10 7:18 Fseide
// (cleaned up some comment)
// 
// 12    11/18/10 16:59 Fseide
// backprop implemented
// 
// 11    11/18/10 16:15 Fseide
// some refactorization from Puv to logPuv, renamed Puv to forwardprop()
// 
// 10    11/18/10 15:47 Fseide
// renamed 'eval' to 'evaluator' and 'train' to the more accurate
// 'accumulator'
// 
// 9     11/18/10 15:37 Fseide
// steps towards training
// 
// 8     11/16/10 11:22 Fseide
// (silly compiler warning due to debug code)
// 
// 7     11/16/10 11:02 Fseide
// disabled our speed-up hack
// 
// 6     11/15/10 8:45p Fseide
// oops, must not clamp to 0 the final emission densities...
// 
// 5     11/15/10 20:32 Fseide
// added small-activations hack
// 
// 4     11/15/10 7:05p Fseide
// fixed minor int/size_t incorrectness detected by the x64 compiler
// 
// 3     11/15/10 18:40 Fseide
// added the ability to clone a model, for use in NUMA-local computation
// 
// 2     11/12/10 12:17 Fseide
// fixed an assertion
// 
// 1     11/12/10 11:38 Fseide
// RBM and DBN factored into separate header files

#if 0                           // [v-hansu] separate comments and codes
#endif

#undef UNSEEN_COMPENSATION         // [v-hansu] compensation for unseen states in seq-discriminative mode
#pragma once

#include "rbm.h"
#include "dtnn.h"
#include <regex>
#include <queue>
#include <iostream>

namespace msra { namespace dbn {

// ===========================================================================
// annlayerfactory -- factory class to help with construction from a given type string
// TODO: we have amassed a lot of inconsistency here. Why why why...
// TODO: doesn't this belong into rbm.h? That's where Iannlayer is declared, as well as the type() strings.
// ===========================================================================
class annlayerfactory
{
public:
    // create and read a network, given a type string
    template<typename FILEHANDLETYPE>
    static inline Iannlayer * createfromfile (const string & type, FILEHANDLETYPE f)
    {
        if (type == "perceptron")
            return new perceptron (f);
        else if (type == "rbmbernoullibernoulli")
            return new rbmbernoullibernoulli (f);
        else if (type == "rbmisalinearbernoulli")   // hack to keep old IPG format--the first SVD layer is named this only in the file
        {
          //  fprintf (stderr, "createfromfile: special type identifier 'rbmisalinearbernoulli' detected, reading as 'rbmbernoullibernoulli' but overwriting non-linearity to 'linearkind'\n");
            return new rbmbernoullibernoulli (f, true/*special flag: it's a linear layer*/);
            // TODO: ^^ get rid of this name and change to 'linearnetwork'
        }
        else if (type == "rbmgaussbernoulli")
            return new rbmgaussbernoulli (f);
        else if (type == "convolutional")
            return new convolutional (f);
        else if (type == "maxpool")
            return new maxpool (f);
        else if (type == "relunetwork")
            return new relunetwork (f);
        else if (type == "softplusnetwork")
            return new softplusnetwork (f);
        else if (type == "leakyrootnetwork")
            return new leakyrootnetwork (f);
        else if (type == "linearnetwork")
            return new linearnetwork (f);
        else if (type == "dtnn")
            return new dtnn (f);
        else if (type == "maxoutnetwork")
            return new maxoutnetwork (f);
        else if (type == "mvn")
            return new mvn (f);
        else
            throw runtime_error ("createfromfile: invalid model type: " + type);
    }

    static Iannlayer * create (const string & type, size_t vdim, std::vector<size_t> hdims, const layerconfigparameters & config, unsigned int randomseed)
    {
        if (type == "dtnn" || type == "tn")
            return new dtnn (vdim, hdims[0], hdims[1], config, randomseed); 
        else
            return create (type, vdim, hdims[0], config, randomseed);
    }

    // TODO: why these duplicate names for layer types?
    static Iannlayer * create (const string & type, size_t vdim, size_t hdim, const layerconfigparameters & config, unsigned int randomseed)
    {
        fprintf (stderr, "create: creating %s layer of dim %d x %d\n", type.c_str(), vdim, hdim);
        if (type == "perceptron" || type == "softmax" || type == "sm" )  // add a perception layer 
            return new perceptron (vdim, hdim, config, randomseed);
        else if (type == "rbmbernoullibernoulli" || type == "bb")
            return new rbmbernoullibernoulli (vdim, hdim, config, randomseed);
        else if (type == "rbmgaussbernoulli" || type == "gb")
            return new rbmgaussbernoulli (vdim, hdim, config, randomseed);
        else if (type == "convolutional")
            return new convolutional (vdim, hdim, config, randomseed);
        else if (type == "maxpool")
            return new maxpool (vdim, hdim, config, randomseed);
        else if (type == "relunetwork")
            return new relunetwork (vdim, hdim, config, randomseed);
        else if (type == "softplusnetwork")
            return new softplusnetwork (vdim, hdim, config, randomseed);
        else if (type == "leakyrootnetwork")
            return new leakyrootnetwork (vdim, hdim, config, randomseed);
        else if (type == "linearnetwork" || type == "ln")
        {
            const bool inferredfDLRmode = (vdim == hdim);   // square matrix--guess that we do fDLR
            const size_t numblocks = inferredfDLRmode ? 11/*must match neighbor expansion*/ : 1;
            const bool poolblocks = inferredfDLRmode;
            // this is hacky, so better let user verify that he/she got what was desired
            // TODO: specify it in c
            fprintf (stderr, "create: inferred %d blocks in %s mode for 'linearnetwork' layer\n", numblocks, poolblocks ? "pooled" : "non-pooled");
            return new linearnetwork(vdim, hdim, config, numblocks, poolblocks); 
        }
        else if (type == "maxoutnetwork")
        {
            const size_t poolSize = config("poolSize");
            return new maxoutnetwork(vdim, hdim, poolSize, config, randomseed);
        }
        else if (type == "mvn")
            return new mvn (vdim, hdim, config);
        else
            throw runtime_error ("create: invalid model type: " + type);
    }
};

#if 1 // TODO: move it to a proper place.[v-xieche]
void copydata (msra::cuda::matrix &dst, msra::cuda::matrix &src, msra::math::ssematrix<msra::math::ssematrixbase> &bufmatrix, int copyFlags)
{
    dst.assign (src, &bufmatrix(0,0), bufmatrix.getcolstride(), false, copyFlags);
}
#endif

// state mapping for class based adaptation __added by Hang Su adaptation

// ===========================================================================
// a DBN model
// This is a stack of RBMs (it also holds mean/std for v norm and Pu for u norm).
// ===========================================================================
class model
{
    vector mean, std;                                 // input normalization (frame-level feature)
    vector ivectormean, ivectorstd;                   // input normalization (sentence-level feature)
    std::vector<unique_ptr<Iannlayer>> layers;        // the layers
    vector Pu;                                        // prior probs over u (empty until we have a supervised layer)
    bool computing;                                   // entercomputation() called?

    model & operator= (const model &);
    model (const model &);
#if 0
    // constructor that copies the whole structure (for NUMA efficiency)
    model (const model & other)
    {
        throw runtime_error ("model: copy constructor not supported");
        //copyfrom (other);
    }
#endif

    // assign
    // The model must already have been created with all its dimension information.
    void copyfrom (const model & other, bool create)
    {
        // clone all layers
        layers.resize (other.layers.size());
        foreach_index (i, other.layers)
        {
            const auto & otherlayer = *other.layers[i];
            if (!layers[i] || layers[i]->type() != otherlayer.type())
            {
                if (!create)
                    throw logic_error ("copyfrom: must copy into a pre-allocated model");
                // we clone a layer by instantiating it with dummy args and then doing a copyfrom()
                // Layers must be written in a way to handle this right.
                layerconfigparameters params ("dummy");   // dummy
                layers[i].reset (annlayerfactory::create (otherlayer.type(), 0, 0, params, 0)); // we assume that 0 dims can be overridden by copyfrom()
            }
            layers[i]->copyfrom (otherlayer);
        }
        // clone the other stuff
        mean = other.mean;
        std = other.std;
        ivectormean = other.ivectormean;
        ivectorstd = other.ivectorstd;
        Pu = other.Pu;
        // and the vars
dumpvars ("copyfrom (before)"); // for testing for now
        vars = other.vars;
dumpvars ("copyfrom (after)"); // for testing for now
    }

    static void malformed (string msg) { throw runtime_error ("dbnmodel: invalid model: " + msg); }

    // -----------------------------------------------------------------------
    // variables  --allows to store, e.g., iteration state in the model
    // -----------------------------------------------------------------------

    std::map<std::string,std::string> vars; // dictionary of variables e.g. iteration state
    void musthavevar (const char * var) const { if (!hasvar (var)) throw std::logic_error ("model: required variable not present: " + string (var)); }
public:
    // TODO: these don't handle size_t and double gracefully (they are just mapped to int and float)
    void setvar (const char * var, const char * val) { vars[var] = val; }
    void setvar (const char * var, const string & val) { setvar (var, val.c_str()); }
    void setvar (const char * var, int val) { setvar (var, msra::strfun::strprintf ("%d", val)); }
    void setvar (const char * var, size_t val) { setvar (var, msra::strfun::strprintf ("%d", (int) val)); }
    void setvar (const char * var, float val) { setvar (var, msra::strfun::strprintf ("%f", val)); }
    void setvar (const char * var, double val) { setvar (var, msra::strfun::strprintf ("%f", (float) val)); }
    bool hasvar (const char * var) const { return vars.find (var) != vars.end(); }
    const char * getvar (const char * var) const { musthavevar (var); return vars.find (var)->second.c_str(); }
    const char * getvar (const char * var, const char * deflt) const { return hasvar (var) ? getvar (var) : deflt; }
    string getvar (const char * var, const string deflt) const { return hasvar (var) ? getvar (var) : deflt; }
    int getintvar (const char * var) const { musthavevar (var); return msra::strfun::toint (vars.find (var)->second.c_str()); }
    int getvar (const char * var, int deflt) const { return hasvar (var) ? getintvar (var) : deflt; }
    size_t getvar (const char * var, size_t deflt) const { return hasvar (var) ? (size_t) getintvar (var) : deflt; }
    float getfloatvar (const char * var) const { musthavevar (var); return (float) msra::strfun::todouble (vars.find (var)->second.c_str()); }
    float getvar (const char * var, float deflt) const { return hasvar (var) ? getfloatvar (var) : deflt; }
    void dumpvars (const char * what) const { fprintf (stderr, "%s:\n----- model vars -----\n%s\n----------------------\n", what, serializevars().c_str()); }
private:
    string serializevars() const
    {
        // serialize all vars into a string
        //  - one line per entry, terminated by '\n' character
        //  - format: var=val
        string buf; buf.reserve (10000);
        for (auto iter = vars.begin(); iter != vars.end(); iter++)
        {
            if (!buf.empty())
                buf.append ("\n");
            buf.append (iter->first.c_str());
            buf.append ("=");
            if (iter->second.find ('\n') != string::npos)
                throw std::logic_error ("serializevars: variable values must not contain a newline character"); 
            buf.append (iter->second.c_str());
        }
        return buf;
    }
    void deserializevars (const string & buf)
    {
        vars.clear();
        if (buf.find ('=') == string::npos) // upwards compat with old files that only had a comment (assuming we never put a = in those comments)
        {
            setvar ("comment", buf);
            return;
        }
        auto lines = msra::strfun::split (buf, "\n");
        foreach_index (i, lines)
        {
            auto & line = lines[i];
            auto eqpos = line.find ('=');
            if (eqpos == string::npos)
                throw std::runtime_error ("deserializevars: malformed line: " + line);
            setvar (line.substr (0, eqpos).c_str(), line.substr (eqpos+1));
        }
    }
    template<typename FILEHANDLETYPE>
    void readvars (FILEHANDLETYPE f)
    {
        char buf[10000];
        fgetstring (f, buf);        // TODO: use a variable-size buffer here if needed
        deserializevars (buf);
        dumpvars ("readvars");
    }
    template<typename FILEHANDLETYPE>
    void writevars (FILEHANDLETYPE f) const
    {
        fputstring (f, serializevars().c_str());
        dumpvars ("writevars");
    }

    // -----------------------------------------------------------------------
    // reading and writing (private bits)
    // -----------------------------------------------------------------------

    template<typename FILEHANDLETYPE>
    void read (FILEHANDLETYPE f)
    {
        fcheckTag (f, "DBN\n");
        readvars (f);
        fcheckTag (f, "BDBN");
        int version = fgetint (f);
        if (version != 0) malformed ("unsupported version number");
        int numlayers = fgetint (f);

        mean.read (f, "gmean");
        std.read (f, "gstddev");
        if (mean.size() != std.size()) malformed ("inconsistent size of mean and std vector");

        std::string tag = fgetTag(f);
        if (tag == "BSLF")
        {
            // have sentence level feature (SLF) mean/std
            ivectormean.read(f, "ivectorgmean");
            ivectorstd.read(f, "ivectorgstddev");
            if (ivectormean.size() != ivectorstd.size()) malformed ("inconsistent size of mean and std vector");
            fcheckTag (f, "ESLF");
            fcheckTag (f, "BNET");
        }
        else
        {
            fcompareTag(tag, "BNET");
        }
        // read all layers
        layers.clear();
        layers.resize (numlayers);
        // fcheckTag (f, "BNET");
        foreach_index (i, layers)
        {
            char buf[100];
            layers[i].reset (annlayerfactory::createfromfile (fgetstring (f, buf), f));
            fprintf (stderr, "read: read %s layer of dim %d x %d\n", layers[i]->type().c_str(), layers[i]->vdim(), layers[i]->hdim());
        }
        fcheckTag (f, "ENET");
        if (!layers.empty() /*&& layers[0]->vdim() % mean.size() != 0*/) 
        {
            fprintf (stderr, "first-level weight dim (%d) and mean/std (%d)\n", layers[0]->vdim(), mean.size());
            // malformed (msra::strfun::strprintf ("inconsistent first-level weight dim (%d) and mean/std (%d) \n", layers[0]->vdim(), mean.size()));
        }

        // read Pu (priors) if we have a top layer
        if (!layers.empty())
        {
            size_t toplayer = layers.size() -1;
            if (layers[toplayer]->type() == "perceptron")   // ... not nice; can't we generalize this somehow? TODO: add virtual bool isoutputlayer() & allow it only on top
            {
                Pu.read (f, "Pu");
                if (Pu.size() != layers[toplayer]->hdim()) malformed ("inconsistent size of top-level weight matrix and priors");
            }
        }
        fcheckTag (f, "EDBN");
    }

    template<typename FILEHANDLETYPE>
    void write (FILEHANDLETYPE f, const string & comment) const
    {
        fputTag (f, "DBN\n");
        const_cast<model*>(this)->setvar ("comment", comment);  // (hacky compat mode: that old comment is now force-put into a var)
        writevars (f);
        fputTag (f, "BDBN");
        fputint (f, 0);                     // a version number
        fputint (f, (int) layers.size());   // number of layers
        mean.write (f, "gmean");
        std.write (f, "gstddev");
        if (idim() > 0)
        {
            fputTag (f, "BSLF");
            ivectormean.write(f, "ivectorgmean");
            ivectorstd.write(f, "ivectorgstddev");
            fputTag (f, "ESLF");
        }
        fputTag (f, "BNET");
        foreach_index (i, layers)
        {
            string type = layers[i]->type();
            // hack for SVD model (kept for compat with old IPG file format)
            auto svdlayer = dynamic_cast<const rbmbernoullibernoulli*> (layers[i].get());
            if (svdlayer && svdlayer->peeknonlinearitykind() == linearkind) // this is the first factor of an SVD
            {
                // we want to remain compatible with IPE's SVD format; their old code remembered the nonlinearitykind by patching the 'type' (ugh!)
                // Note that this will trigger for any 'linearkind' Bernoulli-Bernoulli model
                fprintf (stderr, "write: writing 'rbmbernoullibernoulli' layer as 'rbmisalinearbernoulli' to maintain compatibility with IPG\n");
                fputstring (f, "rbmisalinearbernoulli");
                svdlayer->writeassuminglinearkind (f);
                continue;
            }
            // end hack
            fputstring (f, type.c_str());
            layers[i]->write (f);
        }
        fputTag (f, "ENET");
        if (!layers.empty() && layers[layers.size() -1]->type() == "perceptron")
            Pu.write (f, "Pu"); // only write if the model is complete
        fputTag (f, "EDBN");
    }

    // helper for reading Matlab files: enumerate all names, find the highest numerical value
    template<typename DUMMY> static int gettoplayer (const DUMMY & inmats)
    {
        int toplayer = 0;
        for (auto iter = inmats.begin(); iter != inmats.end(); iter++)
        {
            const string & matname = iter->first;
            int i = atoi (matname.c_str());
            if (i > toplayer)
                toplayer = i;
        }
        return toplayer;
    }

    // load a DBN from two Matlab-5 formatted files
    void loadmatfile (const wstring & path)
    {
#ifndef HAS_MSRA_MATIO  // explicitly #include matio.h in parent source file(s) to get this function
        path;
        throw std::logic_error ("loadmatfile: Matlab support not compiled in. Do not #define NOMATLAB to get it.");
#else
        wstring mvppath = regex_replace (path, wregex (L"\\.mat$"), wstring (L".mvp"));
        // load the Matlab matrix file
        //  - first layer is Gaussian-Bernoulli
        //     - "0-hidBias"
        //     - "0-visBias"
        //     - "0-visToHid"
        //  - intermediate layers is Bernoulli-Bernoulli (N = layer):
        //     - "N-hidBias"
        //     - "N-visBias"
        //     - "N-visToHid"
        //  - final layer is a different type
        //     - "N-W"
        //     - "N-bias"
        msra::math::matio<matrix> MVP (mvppath);
        // load the mean/var/prior file
        //  - "m": mean (original feature dimension)
        //  - "s": mean (original feature dimension)
        //  - "priors": prior probs over u
        msra::math::matio<matrix> M (path);
        int toplayer = gettoplayer (M.getobjects());
        layers.resize (toplayer + 1);
        // preprocessing layer --features get normalized before being fed into it
        mean = MVP["m"];
        std = MVP["s"];
        if (mean.size() != std.size()) malformed ("inconsistent size of mean and std vector");
        // Gaussian-Bernoulli layer
        layers[0].reset (new rbmgaussbernoulli (std::move (M["0-visToHid"]), std::move (M["0-hidBias"]), std::move (M["0-visBias"])));
        if (layers[0]->vdim() % mean.size() != 0) malformed ("inconsistent first-level weight dim and mean/std");
        // intermediate Bernoulli-Bernoulli layers
        for (int i = 1; i < toplayer; i++)
        {
            string pfx = msra::strfun::strprintf ("%d-", i);
            layers[i].reset (new rbmbernoullibernoulli (std::move (M[pfx + "visToHid"]), std::move (M[pfx + "hidBias"]), std::move (M[pfx + "visBias"])));
        }
        // top layer
        string pfx = msra::strfun::strprintf ("%d-", toplayer);
        layers[toplayer].reset (new perceptron (std::move (M[pfx + "W"]), std::move (M[pfx + "bias"])));
        // priors
        Pu = MVP["priors"];
        if (Pu.size() != layers[toplayer]->hdim()) malformed ("inconsistent size of top-level weight matrix and priors");
        // prepare for computation --also checks dimensions
#endif // HAS_MSRA_MATIO
    }

    void checknotcomputing() const { if (computing) throw std::logic_error ("function called while in 'computing' state, forbidden"); }
    void checkcomputing() const { if (!computing) throw std::logic_error ("function called while not in 'computing' state, forbidden"); }

public:

    // -----------------------------------------------------------------------
    // constructors
    // -----------------------------------------------------------------------

    // Note: The public methods on this class are to be called outside of computation.
    // Once we start computing (entercomputation()), do not call any method in this object directly.

    // constructor for a fresh model  --we need to pass feature mean/std (and also ivector mean/std if used, otherwise pass an empty vector)
    model (const std::vector<float> & datamean, const std::vector<float> & datastd, const std::vector<float> & ivectodatarmean, const std::vector<float> & ivectordatastd) : mean (datamean), std (datastd), ivectormean (ivectodatarmean), ivectorstd (ivectordatastd), computing (false) {}

    // constructor that loads from input file
    model (const wstring & path) : computing (false) { load (path); }

    // partial constructor that expects load() to be called afterwards, to allow putting model into a class and initializing inside the constructor
    model() : computing (false) { }

    // -----------------------------------------------------------------------
    // reading and writing (public)
    // -----------------------------------------------------------------------

    void load (const HANDLE f)
    {
        if (f == INVALID_HANDLE_VALUE) return; 

        // read the model
        read (f);
#ifdef LOADSTEEPERSIGMOIDMODEL
        fprintf (stderr, "load : divided scale %.4f after laod model.[v-xieche]\n", AMPNUM);
        for (size_t i = 0; i < layers.size() - 1; i ++)
            layers[i]->multiplywith (float(1.0 / AMPNUM) );
#endif

    }

    void load (const wstring & path)
    {
        auto_file_ptr f = fopenOrDie (path, L"rbS");

        // check if Matlab 5 format --Dong's experiments stored models in this format
        string tag = fgetTag (f);
        if (tag == "MATL")
        {
            fclose (f);
            loadmatfile (path);
            return;
        }
        rewind (f);

        // read the model
        read<FILE*> (f);
#ifdef LOADSTEEPERSIGMOIDMODEL
        fprintf (stderr, "load : divided scale %.4f after laod model.[v-xieche]\n", AMPNUM);
        for (size_t i = 0; i < layers.size() - 1; i ++)
            layers[i]->multiplywith (float(1.0 / AMPNUM) );
#endif

        // sanity check for convenience
        checkmodel();
    }

    // serializing a snapshot to disk
    void save (const wstring & path, const string & comment) const
    {
        checknotcomputing();
#ifdef STEEPERSIGMOID   // for steeper sigmoid experiment. should save the multiply factor before write to mode file. [v-xieche]
        for (size_t i = 0;  i < layers.size() - 1; i ++)
            layers[i]->multiplywith (AMPNUM);
#endif
        auto_file_ptr f = fopenOrDie (path, L"wbS");
        write<FILE*> (f, comment);
        fflushOrDie (f);
#ifdef STEEPERSIGMOID   // for steeper sigmoid experiment. should save the multiply factor before write to mode file.[v-xieche]
        for (size_t i = 0;  i < layers.size() - 1; i ++)
            layers[i]->multiplywith (float(1.0 / AMPNUM) );
#endif

        // sanity check for convenience
        checkmodel();
    }

    void save (HANDLE f, const string & comment) const
    {
        checknotcomputing();
        write (f, comment);
    }

    // backup and restore
    void backupto (model & other) { other.copyfrom (*this, true); }
    void restorefrom (const model & other) { copyfrom (other, false); }

    // redistribute the model from the main node to all others
    void mpiredistribute (const std::vector<modelupdateinfo> & bpinfos)
    {
        fprintf (stderr, "mpiredistribute: distributing model parameters from main MPI node to all nodes\n");
        checknotcomputing();
        foreach_index (i, layers)
            layers[i]->mpiredistribute (bpinfos[i]);
        // clone the other stuff
        auto & mpiaggregator = *bpinfos[0].mpiaggregator;
        mpiaggregator.redistribute (mean.asvectorref());
        mpiaggregator.redistribute (std.asvectorref());
        mpiaggregator.redistribute (Pu.asvectorref());
        // and the vars (they may contain iteration state)
dumpvars ("mpiredistribute (before)");
        string currentvars = serializevars();   // (only those of the main node matter)
        mpiaggregator.redistributestring (currentvars);
        deserializevars (currentvars);
dumpvars ("mpiredistribute (after)");
    }

    // only keep the top layer that falls into [startoutputid  endoutputid)
    void shrinktoplayerandprior (const size_t startoutputid, const size_t endoutputid)
    {
        checknotcomputing();

        if (layers.empty() || layers[layers.size() -1]->type() != "perceptron")
            malformed ("shrinktoplayerandprior: only applies to the top softmax layer which does not exist in this model");

        if (startoutputid<0 || endoutputid<0 || endoutputid<=startoutputid)
            malformed ("shrinktoplayerandprior: endoutputid should > startoutputid >=0");

        //shrink prior
        vector oldPu = Pu;
        Pu.resize(endoutputid - startoutputid, 1, false);

        float sum=0.0;
        foreach_row(i, Pu)
            sum += oldPu(startoutputid+i, 0);

        foreach_row(i, Pu)
            Pu(i,0) = oldPu(startoutputid+i, 0)/sum;


        //shrink toplayer
        perceptron & toplayer = dynamic_cast<perceptron &>(*(layers[layers.size() -1]));
        toplayer.shrink(startoutputid, endoutputid);
    }

    // scale the weight matrix in order to generate the mean model for a model trained with dropout (or in order to undo it)
    // With dropout, W matrices come out too large by 1/(1-dropoutrate); this function is to undo this factor (or to redo when entering the epoch).
    void dropoutscaling (float factor)
    {
        checkcomputing();
        if (factor == 1.0f)
            return;
        fprintf (stderr, "dropoutscaling: scaling weight matrices by %.2f for layers with dropout input\n", factor), fflush (stderr);
        foreach_index (i, layers)
            if (shouldapplydropoutatinputof (i))            // true if inputs are dropped out, which will cause a detuned W matrix
                layers[i]->dropoutscaleweights (factor);
    }

    void exitdropouttraining  (float dropoutrate) { dropoutscaling (1.0f - dropoutrate); }          // turn drop-out updated model into decodable average model
    void enterdropouttraining (float dropoutrate) { dropoutscaling (1.0f / (1.0f - dropoutrate)); } // turn it back into the trainable representation

    bool iscomputing() const { return computing; }
    // dimensions
    size_t fdim() const throw() { return mean.size(); }                         // raw input feature dim
    size_t idim() const throw() { return ivectormean.cols() == 0 ? 0 : ivectormean.size(); }                  // raw input ivector dim
    size_t vdim() const throw() { return layers[0]->vdim(); }                   // input feature dim after augmentation of neighbors
    size_t udim() const throw() { return layers[layers.size()-1]->hdim(); }
    size_t hdim (size_t layer) const throw() { return layers[layer]->hdim(); }  // note: 0-based, e.g. first hidden layer dim = hdim[0]
    size_t numlayers() const throw() { return layers.size(); }  // number of networks
    size_t augmentationextent() const   // note: untested
    {
        size_t n = (vdim() - idim()) / fdim();
        if ((n-1) % 2 != 0 || fdim() * n != vdim() - idim())
            throw std::runtime_error("numneighbors: mismatching fdim() vs. vdim() - idim()");
        return (n-1) / 2;
    }

    // input: numsenone2keep
    // output: senone2update
    // senone2update records whether a senone shall be updated through the training
    void settopnsenones (std::vector<bool> & senone2update, const size_t numsenone2update)        // [v-hansu] this is a tmp solution
    {
        std::vector<pair<float,size_t>> priorwithindex (udim());
        foreach_index (i, priorwithindex)
        {
            priorwithindex[i].first = Pu[i];
            priorwithindex[i].second = i;
        }
        std::sort (priorwithindex.begin(), priorwithindex.end());
        senone2update.resize (udim());
        senone2update.assign (senone2update.size(), 0);
        double sumofPu = 0.0f;
        for (size_t i = 0; i < numsenone2update; i++)
        {
            senone2update[priorwithindex[priorwithindex.size() - 1 - i].second] = true;
            sumofPu += priorwithindex[priorwithindex.size() - 1 - i].first;
        }
        fprintf (stderr, "settopnsenones: sum of top %d senones' Pu is %f%%\n", numsenone2update, sumofPu*100);
#if 0   // print the distribution of Pu from biggest -> smallest
        size_t numstep = 30;
        size_t stepsize = udim() / numstep;
        double sumofPus = 0.0f;
        size_t countstep = 1;
        for (size_t i = 0; i < udim(); i++)
        {
            if ((i + 1) % stepsize == 0)
            {
                fprintf (stderr, "settopnsenones: top %d: %f%%\n", countstep * stepsize, sumofPus * 100);
                countstep++;
            }
            sumofPus += priorwithindex[priorwithindex.size() - 1 - i].first;
        }
        fprintf (stderr, "settopnsenones: top %d: %f%%\n", udim(), sumofPus * 100);
#endif
    }

    const std::string /*& fix this by changing to const char * */ layertype (size_t layer) const throw() { return layers[layer]->type(); }

    // for special purposes, we allow to get internal access (it's a research project)
    const Iannlayer & peeklayer (size_t i) const throw() { return *layers[i].get(); }

#ifdef MULTICUDA
    // is CUDA enabled?
    bool cudamode() const { return layers[0]->getweight().cudamode; }
#endif

    // check dimensions --used at load time during training to verify we have the config right
    void checkdimensions (const std::vector<size_t> & dims) const
    {
        if (dims.size() <= layers.size())
            malformed ("checkdimensions: too few layerdims expected compared to model");
        foreach_index (i, layers)
        {
            if (layers[i]->vdim() != dims[i] || layers[i]->hdim() != dims[i+1])
            {
                fprintf (stderr, "for layer i(%d): layers[i]->vdim(%d) != dims[i](%d) || layers[i]->hdim(%d) != dims[i+1](%d) ", i, layers[i]->vdim(), dims[i],layers[i]->hdim(), dims[i+1]);
                malformed ("checkdimensions: expected layerdims mismatching the actual model");
            }
        }
    }

    // create a layer
    // Called when pretraining/finetuning an additional layer.
    // Initializes W to (deterministic pseudo-) random numbers.
    // TODO: 'top' parameter is superceded by layertype (use "perceptron"), we should get rid of it (should be no longer used, actually)
    void addlayer (bool top, size_t vdim, size_t hdim, const wstring & layertype, const layerconfigparameters & config)
    {
        checknotcomputing();
        size_t layer = numlayers();
        layers.resize (layer +1);
        // create the layer
        unsigned int randomseed = (unsigned int) layer;
        if (layertype == L"dtnn" ||layertype == L"tn"  )
        {
            size_t h2dim = (size_t)sqrt((double)hdim);
            if (h2dim * h2dim != hdim)
                malformed ("addlayer: for dtnn layer the layer dim must be a square of a number");
            layers[layer].reset (new dtnn (vdim, h2dim, h2dim, config, randomseed));
        }
        else if (!layertype.empty())  // other layer, with specified type
            layers[layer].reset (annlayerfactory::create (strfun::utf8 (layertype), vdim, hdim, config, randomseed));
        // legacy: use defaults if type is not given
        // TODO: this is outdated since we have layertype, and since we have a default in main.cpp, we should not need this code branch anymore
        else if (top)
            layers[layer].reset (new perceptron (vdim, hdim, config, randomseed));
        else if (layer > 0)
            layers[layer].reset (new rbmbernoullibernoulli (vdim, hdim, config, randomseed));
        else
            layers[layer].reset (new rbmgaussbernoulli (vdim, hdim, config, randomseed));

        // for top layer, we also create the prior probabilities
        if (dynamic_cast<const perceptron *> (layers[layer].get()) != NULL)
        {
            Pu.resize (hdim, 1, false);
            foreach_index (i, Pu)
                Pu[i] = 1.0f / Pu.size();
        }
    }

    // enlarge the model to do classes based adaptation  --TODO: delete this and all related code
    void blowup (const size_t blowupfactor, const std::vector<size_t> & layerdims, const std::vector<size_t> & statemapping)    //added by Hang Su adaptation
    {
        for (size_t i = 0; i < layers.size() -1; i++)
        {
            if (layerdims[i+1] / layers[i]->hdim() != blowupfactor)
            {
                throw std::runtime_error ("modelblowup: layerdims does not match with blowupfactor.");
            }
        }
        for (size_t i = 0; i < layers.size() -1; i++)
            layers[i]->blowup(blowupfactor);
        layers[layers.size() - 1]->blowup(blowupfactor,statemapping);
    }

    void setlinearlayer (const msra::dbn::matrix & adaptmatrixinitpath)
    {
        layers[0]->setlinearlayerweight (adaptmatrixinitpath);
    }

#ifdef STEEPERSIGMOID   // for steeper sigmoid experiment. should save the multiply factor before write to mode file.[v-xieche]
    // get the w(i, j) value for specific layer, for experimental purposes [v-xieche]
    float getweight(size_t layer, size_t r, size_t c) const
    {
        checknotcomputing();
        assert (layer >=0 && layer <= layers.size());
        assert (r >= 0 && c >= 0 && r < layers[layer]->vdim() && c < layers[layer]->hdim());
        return layers[layer]->getweightvalue(r, c);
    }
    // get the a(m) value for specific layer, for experimental purposes [v-xieche]
    float getbias(size_t layer, size_t m) const
    {
        checknotcomputing();
        assert (layer >=0 && layer <= layers.size());
        assert (m >= 0 && m <= layers[layer]->hdim());
        return layers[layer]->getbiasvalue(m);
    }
#endif

    // perform SVD on the model in-place and write out the resulting model
    void converttosvdmodel (const std::vector<float> & ranks)
    {
        size_t numnewlayers = 0;
        size_t numlayers = this->numlayers();

        if (numlayers != ranks.size())
            throw std::runtime_error ("converttosvdmodel: number of layers mismatches rank dimensions array");

        for (size_t i = 0; i < numlayers; ++i)
            if (ranks[i])
                ++numnewlayers;

        std::vector<unique_ptr<Iannlayer>> newlayers;
        std::vector<string> newlayertypes;

        // we split each decomposed layer into two factors; the first in the existing layers[] array, and the second collected in V and then in newlayers[]
        newlayers.resize (numnewlayers);

        std::vector<size_t> dims (numlayers, 0);                        // [layer] bottleneck dimension of layer after SVD (0 if this layer was not decomposed)
        std::vector<std::vector<std::vector<float>>> Vs (numlayers);    // [layer][matrix indices] the first factor for each is remembered here, for later insertion as a linear layer

        Concurrency::parallel_for ((size_t) 0, numlayers, [&] (size_t ir)
        {
            const size_t i = numlayers - 1 - ir;    // loop backwards to get the largets (output) layer started first (parallel_for cannot step backwards--that's a big "meh!")
            if (ranks[i] == 0)
            {
                fprintf (stderr, "converttosvdmodel: no SVD on layer %d requested, skipping\n", i);
                return;
            }

            // buffer for the SVD decomposition
            auto & V = Vs[i];
            fprintf (stderr, "converttosvdmodel: applying SVD to layer %d\n", i);

            // TODO: why is this check necessary? E.g. can we do this with ReLUs?
            if (!dynamic_cast<rbmbernoullibernoulli *> (layers[i].get()) && !dynamic_cast<perceptron *> (layers[i].get()))
                throw std::runtime_error ("converttosvdmodel: attempted to decompose layer that is not of type 'rbmbernoullibernoulli' or 'perceptron'");

            // perform SVD decomposition W' = U V'
            // this will update this layer's weights W in-place to be the second factor (W <- U'), while returning the first factor V
            dims[i] = layers[i]->svd (V, ranks[i]);   // TODO: why not use the matrix type here?
            // V is a square matrix, but when we transfer it into the linear layer, we cut the dimension to match 'dims[i]'   --TODO: cut it inside svd()

            fprintf (stderr, "converttosvdmodel: keeping %d singular values for layer %d\n", dims[i], i);
        });

        // create the linear layer for the first factor (gets sorted in below)
        for (size_t i = 0, index = 0; i < numlayers; i++)
        {
            if (dims[i] == 0)
                continue;
            const auto & V = Vs[i];
            size_t dimn = V.size();
            // TODO: we should check whether the input is actually a rbmbernoullibernoulli, and throw if not --TODO: << I don't understand my own comment anymore
            fprintf (stderr, "converttosvdmodel: inserting first factor as new layer (rbmbernoullibernoulli/linearkind, %d x %d)\n", dimn, dims[i]);
            newlayers[index++].reset (new rbmbernoullibernoulli (V, (int) dimn/*vdim = rows*/, (int) dims[i]/*hdim = cols = bdim*/));    // this constructor overload constructs a linear layer --UGH! TODO: just use a linearnetwork!!
        }

        // shuffle the new layers into the current ones at the respective correct positions
        size_t newnumlayers = numlayers + numnewlayers; // TODO: better naming 'newnumlayers' vs. 'numnewlayers' --UGH!
        layers.resize (newnumlayers);

        // inject the original layers into the layers[] array (from newlayers[] where they were parked)
        for (int i = (int) newnumlayers-1, j = (int) numlayers-1, k = (int) numnewlayers-1; i >= 0; --j)
        {
            if (i != j)
                layers[i].swap(layers[j]);
            --i;

            if (ranks[j])
            {
                layers[i].swap(newlayers[k]);
                --i;
                --k;
            }
        }
    }

    // decide whether a layer can have dropout applied at its input or not
    // Currently we just do not apply it at the original input layer. This does not handle fDLR adaptation correctly, but then, don't use dropout here.
    // Insert more heuristics/rules here.
    static bool shouldapplydropoutatinputof (size_t layerindex)
    {
        return layerindex > 0;  // inputs to layer 0 should not get dropout
    }

private:
    // add some type layer in a given layer
    // TODO: Really only used for injectlinearlayer() and insertmvnlayers() for now. Tricky because of different init parameters.
    // TODO: move into annlayerfactory class
    void insertlayer (const string & type, size_t vdim, size_t hdim, size_t injectlocation, const layerconfigparameters & config)
    {
        checknotcomputing();
        unsigned int randomseed = (unsigned int)injectlocation;
        if (type == "linearnetwork")
        {
            // notice: linearnetwork don't initialize the W and a as a rand number between (0,1)
            // feature dim can be 52 or 39
            const size_t featuredimoriplp = 52;
            const size_t featuredimorihlda = 39;
            size_t expandfactor;                // expandfactor records the num of blocks in linear network
            // TODO: pass this in config
            if (hdim % featuredimoriplp == 0)
                expandfactor = hdim / featuredimoriplp;
            else if (hdim % featuredimorihlda == 0)
                expandfactor = hdim / featuredimorihlda;
            else
                expandfactor = hdim;
            unique_ptr<Iannlayer> newlayer(new linearnetwork (vdim, hdim, config, expandfactor, true));
            layers.insert(layers.begin() + injectlocation, move(newlayer)); 
        }
        else
        {
            unique_ptr<Iannlayer> newlayer(annlayerfactory::create (type, vdim, hdim, config, randomseed));
            layers.insert(layers.begin() + injectlocation, move(newlayer));
        }
    }
public:
    const vector& getmeanref () { return mean; }
    const vector& getstdref ()  { return std;  }
    const vector& getPuref  ()  { return Pu; }
    const std::vector<unique_ptr<Iannlayer>>& getlayersref() { return layers;}

    // insert a lineartransform layer at the bottom
    void injectlinearlayer (size_t injectlocation, const layerconfigparameters & c)
    {
        checknotcomputing();
        const size_t linearlayerdim = layers[injectlocation]->vdim();
        insertlayer ("linearnetwork", linearlayerdim, linearlayerdim, injectlocation, c);
    }
    // insert a layer of type 'mvn' between each two layers
    void insertmvnlayers()
    {
        layerconfigparameters emptyconfig ("");
        for (size_t i = layers.size() -1; i > 0; i--)
            insertlayer ("mvn", layers[i-1]->hdim(), layers[i]->vdim(), i, emptyconfig);
    }

    // recreate a layer from scratch (discard the old one)
    void reinitializelayer (size_t layer, const layerconfigparameters & config)
    {
        checknotcomputing();
        unsigned int randomseed = (unsigned int)layer;
        layers[layer].reset (annlayerfactory::create (layers[layer]->type(), layers[layer]->vdim(), layers[layer]->hdims(), config, randomseed));
    }

    // reduce number of layers --used for testing purposes only
    void shedlayers (size_t n)
    {
        checknotcomputing();
        if (n > layers.size())
            malformed ("shedlayers: trying to shed layers that don't exist");
        if (n == layers.size())
            return;
        fprintf (stderr, "shedlayers: reducing number of layers to %d\n", n);
        layers.resize (n);
        Pu.resize (0, 0, false);    // no top layer anymore --no priors
    }

    // implant the prior probability
    void setprior (const msra::dbn::vector & newPu)
    {
        checknotcomputing();
        if (numlayers() == 0 || layers[numlayers()-1]->type() != "perceptron")
            malformed ("setprior: attempted to set prior when no top layer available");
        if (newPu.size() != udim())
            malformed ("setprior: new prior vector of wrong dimension");
        Pu = newPu;
    }

    // duplicate priors to copies (for experimental multiple-mixture experiment)
    void collatemixturepriors (const size_t nummix)
    {
        const size_t numsenones = udim() / nummix;
        if (numsenones * nummix != udim())
            throw std::logic_error ("collatemixturepriors: inconsistent nummix/udim");

        for (size_t s = 0; s < numsenones; s++) for (size_t c = 1; c < nummix; c++)
        {
            const size_t coffset = c * numsenones;
            if (s < 10)
                fprintf (stderr, "collatemixturepriors: mix %d: %.6f -> %.6f (main)\n", (int) c, Pu[s + coffset], Pu[s]);
            Pu[s + coffset] = Pu[s];
        }
    }

    // function for more hacky experiments to set an entire layer to some value
    template<class WTYPE, class ATYPE>
    void overridelayer (size_t n, const WTYPE & W, const ATYPE & a, const size_t weightsetindex)
    {
        checknotcomputing();

        //we need to do conversion since template function setweight is not virtual
        if (layers[n]->type() == "dtnn")
        {
            dtnn & layer = dynamic_cast<dtnn &>(*layers[n]);
            layer.setweights (W, a, weightsetindex);
        }
        else
        {
            rbmbase & layer = dynamic_cast<rbmbase &>(*layers[n]);
            layer.setweights (W, a, weightsetindex);
        }
    }

    // for diagnostics we allow to read this out
    const msra::dbn::vector & getprior() const { return Pu; }

    // split a layer (used for an experiment in growing layers dimension-wise, not used)
    void splitlayer (size_t layer)
    {
        checknotcomputing();
        layers[layer-1]->doublenodes (true);
        if (layer < layers.size())
            layers[layer]->doublenodes (false);
        else    // top layer: need to "split" priors, which really means to fill the top half with dummy values
        {
            const size_t numsenones = Pu.size();
            vector newPu (numsenones * 2);
            foreach_index (s, Pu)
            {
                newPu[s] = Pu[s];
                newPu[s + numsenones] = 1.0f;   // leads to a small value; should never be used anyway
            }
            setprior (newPu);
        }
    }

    // flip polarity of sigmoid layers
    // See flippolarity() for explanation.
    void flipsigmoids()
    {
        for (size_t i = 0; i < layers.size() -1; i++)
        {
            auto rbmlayer = dynamic_cast<rbmbase *> (layers[i].get());
            auto nextlayer = dynamic_cast<rbmbase *> (layers[i+1].get()); // nextlayer does not strictly have to be rbmbase, but I don't want to pollute the Iannlayer interface...
            if (!rbmlayer || !nextlayer)
                continue;
            // it's a sigmoid layer, and the receiving one is as well
            float a, b; // what linear transform needs to be applied to inputs to make good
            rbmlayer->flippolarity (a, b);
            nextlayer->applytransform (a, b);  // make good for it
        }
    }

    void print()    // const?
    {
        checknotcomputing();

        fprintf (stderr, "\n@@@@@@@@@@@@@@@ Dump model weights @@@@@@@@@@@@@@@\n");

        printmat(mean);
        printmat(std);
        printmat(Pu);

        std::vector<unique_ptr<Iannlayer>>::iterator itr;
        size_t layer;
        for ( itr = layers.begin(), layer=0; itr != layers.end(); ++itr, ++layer )
        {
            fprintf (stderr, "\n###### layer %d ######\n", layer);
            (*itr)->print();
        }
    }

    void print (FILE *f)        // added by Hang Su
    {
        checknotcomputing();

        fprintf (f, "\n@@@@@@@@@@@@@@@ Dump model weights @@@@@@@@@@@@@@@\n");

        std::vector<unique_ptr<Iannlayer>>::iterator itr;
        size_t layer;
        for ( itr = layers.begin(), layer=0; layer < 2 && itr != layers.end(); ++itr, ++layer )
        {
            fprintf (f, "\n###### layer %d ######\n", layer);
            (*itr)->print(f);
        }
        itr = layers.end();
        itr--;
        fprintf (f, "\n###### layer %d ######\n", layer);
        (*itr)->print(f);
    }

    // dump a specific layer matrix to stdout [v-xieche]
    void dumplayer(size_t layer)
    {
        checknotcomputing();
        fprintf(stderr, "\n@@@@@@@@@@@  Dump matrix of layer %d @@@@@@@@@@@\n", layer);
        layers[layer]->dumplayer();
    }

    // sanity check on model
    // Currently checks for NaN and INF and warns if any found.
    void checkmodel() const
    {
        checknotcomputing();
        size_t totalnansinf = 0;
        size_t totalnumparams = 0;
        // layers
        foreach_index (i, layers)
        {
            const auto & layerI = layers[i].get();
            const auto * layer = dynamic_cast<const rbmbase *> (layerI);
            if (layer == NULL)  // oops, wrong type
                continue;
            const auto & W = layer->peekweightmatrix();
            const auto & a = layer->peekbias();
            totalnumparams += W.rows() * W.cols() + a.rows() * a.cols();
            const size_t nansinffoundinW = W.countnaninf();
            const size_t nansinffoundina = a.countnaninf();
            if (nansinffoundinW > 0 || nansinffoundina > 0)
                fprintf (stderr, "checkmodel: anomaly detected: %d NaNs or INF in W and %d in a in model layer %d\n", (int) nansinffoundinW, (int) nansinffoundina, (int) i);
            // TODO: check 'b' as well if so desired
            totalnansinf += nansinffoundinW + nansinffoundina;
        }
        // priors
        const size_t nansinffoundinPu = Pu.countnaninf();
        if (nansinffoundinPu > 0)
            fprintf (stderr, "checkmodel: anomaly detected: %d NaNs or INF in priors\n", (int) nansinffoundinPu);
        totalnumparams += Pu.rows() * Pu.cols();
        totalnansinf += nansinffoundinPu;
        //if (totalnans > 0)
            fprintf (stderr, "checkmodel: %d NaNs/INF found in %d parameters\n", (int) totalnansinf, (int) totalnumparams);
    }

    // print model distribution
    // This function can be fluid, for various purposes.
    // Returns a pair (total params, total non-null params) for printing overall statistics.
    pair<unsigned int,unsigned int> printvaluedistribution() // const
    {
        checknotcomputing();

#if 1   // statistics interesting for tracking down the linear-layer training problem
        std::vector<double> avweightnorms (layers.size(), 0.0), avbiases (layers.size(), 0.0);
        foreach_index (k, layers)
        {
            const auto & W = layers[k]->peekweightmatrix();
            const auto & a = layers[k]->peekbias();

            double weightnormsum = 0.0;
            foreach_column (j, W)
            {
                double sqrsum = 0.0;
                foreach_row (i, W)
                    sqrsum += W(i,j) * W(i,j);
                double norm = sqrt (sqrsum);
                weightnormsum += norm;
            }
            avweightnorms[k] = weightnormsum / W.cols();

            double biassum = 0.0;
            foreach_row (i, a)
                biassum += a[i];
            avbiases[k] = biassum / W.rows();
        }

        foreach_index (k, layers)
            fprintf (stderr, "\t%d", k);
        fprintf (stderr, "\n");
        fprintf (stderr, "avnorm");
        foreach_index (k, layers)
            fprintf (stderr, "\t%.2f", avweightnorms[k]);
        fprintf (stderr, "\n");
        fprintf (stderr, "avbias");
        foreach_index (k, layers)
            fprintf (stderr, "\t%.2f", avbiases[k]);
        fprintf (stderr, "\n");

        unsigned int totalparams = 0, totalnonnullparams = 0;
#endif

#if 1   // model weight distributions
        fprintf (stderr, "\n@@@@@@@@@@@@@@@ Dump model weight distribution information @@@@@@@@@@@@@@@\n");

#if 0   // hack to normalize bias of softmax level by log priors
        size_t toplayer = layers.size() -1;
        auto & bias = layers[toplayer]->a;
        std::vector<float> savedbias (bias.rows());
        foreach_row (i, bias)
            savedbias[i] = bias[i];
        if (!Pu.empty())
        {
            double sum = 0.0;
            foreach_row (i, bias)
            {
                bias[i] -= log (Pu[i]);
                sum += bias[i];
            }
            foreach_row (i, bias)
                bias[i] -= (float) sum / bias.rows();
        }
#endif

        foreach_index (layer, layers)
        {
            fprintf (stderr, "\n###### layer %d ######\n", layer);
            auto stats = layers[layer]->printvaluedistribution (msra::strfun::strprintf ("[%d]", layer));
            totalparams += stats.first;
            totalnonnullparams += stats.second;
        }

#if 0   // undo hack to normalize bias of softmax level by log priors
        foreach_row (i, bias)
            bias[i] = savedbias[i];
#endif
#endif
        return make_pair (totalparams, totalnonnullparams);
    }

    // do necessary preparations to start any computation with the model
    // This is expensive and intended to be for an entire epoch.
    // 'type'can be:
    //  -2 -> Hessian free optimization
    //  -1 -> backpropagation
    //  +1 -> pretraining
    //   0 -> evaluation
    // With CUDA, this loads the model into the CUDA RAM.
    void entercomputation (int type)
    {
        checknotcomputing();
        computing = true;
        foreach_index (j, layers)
            layers[j]->entercomputation (type);
    }
#ifdef  MULTICUDA  // take deviceid into consideration.
    void entercomputation (int type, size_t deviceid)
    {
        checknotcomputing ();
        computing = true;

        foreach_index (j, layers)
        {
            layers[j]->entercomputation (type, deviceid);
        }
    }

    void exitcomputation (size_t deviceid)
    {
        checkcomputing ();
        foreach_index (j, layers)
            layers[j]->exitcomputation (deviceid);
        computing = false;
    }
#endif
#ifdef MULTICUDA
    void entercomputation (int type, std::vector<size_t> &deviceids)
    {
        checknotcomputing ();
        computing = true;

        foreach_index (j, layers)
        {
            layers[j]->entercomputation (type, deviceids[j]);
        }
    }
    // hack for striped mode [v-xieche]
    void entercomputation (int type, std::vector<size_t> &deviceids, bool stripedintoplayer, size_t topdevicenum)
    {
        checknotcomputing ();
        computing = true;

        foreach_index (j, layers)
        {
            if (stripedintoplayer && j == (int) layers.size() - 1) // for top layer and in striped mode. [v-xieche]
                layers[j]->entercomputation (type, deviceids[j], true, topdevicenum);
            else
                layers[j]->entercomputation (type, deviceids[j]);
        }
    }

    void exitcomputation (std::vector<size_t> &deviceids)
    {
        checkcomputing ();
        foreach_index (j, layers)
            layers[j]->exitcomputation (deviceids[j]);
        computing = false;
    }
    void exitcomputation (std::vector<size_t> &deviceids, bool stripedintoplayer,size_t topdevicenum)
    {
        checkcomputing ();
        foreach_index (j, layers)
        {
            if (stripedintoplayer && j == (int) layers.size() - 1)
                layers[j]->exitcomputation (deviceids[j], true, topdevicenum);
            else
                layers[j]->exitcomputation (deviceids[j]);
        }
        computing = false;
    }
#endif

    // same do necessary finalization, e.g. in case of CUDA, copy updated models back to CPU RAM
    void exitcomputation()
    {
        checkcomputing();
        foreach_index (j, layers)
            layers[j]->exitcomputation();
        computing = false;
    }

    // link to MPI data-parallel gradient aggregation
    // Call this right after entercomputation().
    void entermpiaggregation (msra::dbn::mpiaggregator & mpiaggregator, size_t mbsizeparam, size_t bits, bool allowdoublebuffering)
    {
        mpiaggregator.entercomputation (mbsizeparam, allowdoublebuffering, bits, [&] (std::vector<size_t> & mpistripebuffersizes, size_t bits)
        {
            // entercomputation() will call back into this code to perform the stripe and buffer dimensioning
            // layer[]->entermpiaggregation() will allocate its buffers by bumping up the numbers in mpistripebuffersizes[]
            // The current value will act as an offset into the buffer.
            // The mpistripebuffersizes[] array has already been initialized to account for space for a header owned by the aggregate() function.
            // compute the stripe dimension for each per-node stripe, and 'allocate' buffer bytes by bumping up mpistripebuffersizes[node]
            foreach_index (j, layers)
                layers[j]->entermpiaggregation (mpistripebuffersizes, bits);
        });
    }

    // Call this right before exitcomputation().
    // (the last double-buffer thread may still be running; mpiaggregator.exitcomputation() will handle that)
    void exitmpiaggregation (msra::dbn::mpiaggregator & mpiaggregator)
    {
        mpiaggregator.exitcomputation ([&]()
        {
            // exitcomputation() will call back into this code to perform the finalization
            foreach_index (j, layers)
                layers[j]->exitmpiaggregation();
        });
    }

    // -----------------------------------------------------------------------
    // class for evaluation
    // We use this separate class in order to hold memory for intermediate state values.
    // Call methods of this class only inside entercomputation()/exitcomputation().
    // -----------------------------------------------------------------------

    class evaluator
    {
        void operator=(evaluator&);
    protected:
        std::vector<rbmstatevectors> layerstate;    // all layer inputs/outputs as rbmstatevectors
#if 0  // hack for getting activation histogram [v-xieche]
        msra::basetypes::matrix<size_t> hist;
        size_t nnode;
        size_t nhiddenlayers;
        size_t nresolution;  // histogram resolution, used as index
        //size_t ts, te;
#endif

        // these are just remembered from dbnmodel
        const vector & mean, & std;
        const vector & ivectormean, & ivectorstd;
        const vector & Pu;
        const std::vector<unique_ptr<Iannlayer>> & layers;   // the layers

        size_t vdim() const throw() { return layers[0]->vdim(); }
        size_t udim() const throw() { return layers[layers.size()-1]->hdim(); }
        size_t idim() const throw() { return ivectormean.cols() == 0 ? 0 : ivectormean.size(); }
        size_t nfwd() const throw() { return layerstate[0].cols(); }

        void alloclayerstate (std::vector<rbmstatevectors> & layerstate, size_t nfwd)
        {
            layerstate.resize (layers.size()+1);
            layerstate[0].resize (layers[0]->vdim(), nfwd);
            foreach_index (i, layers)
            {
                // layerstate[i] is input of layer i
                if (layerstate[i].rows() != layers[i]->vdim())
                    model::malformed ("dimension mismatch between layers");
                // layerstate[i+1] is for output of layer i = input of layer i+1
                layerstate[i+1].resize (layers[i]->hdim(), nfwd);
            }

#if 0    // hack for getting activation histogram [v-xieche]
            nnode = layers[1]->vdim();   // the first layer's hidden layer number.
            nhiddenlayers = layers.size() - 1;  // exclude first and top layer
            nresolution = 20;
            hist.resize(nresolution, nhiddenlayers);
            foreach_coord(i, j, hist)
                hist(i,j) = 0;
#endif
            assert (Pu.empty() || udim() == Pu.size());
        }

        // [v-hansu] allocate a row vector rather than matrix for each layer, for unseen compensation  --TODO: did not work; we should clean out this code
        void allocvnormbufs (std::vector<rbmstatevectors> & vnormbufs, size_t nfwd)
        {
            vnormbufs.resize (layers.size()+1);
            foreach_index (i, vnormbufs)
                vnormbufs[i].resize (1, nfwd);
        }

    public:
        // This creates vectors to store the intermediate state activations (incl. v and u).
        // Note: We allow to instantiate this inside and outside 'computing' mode.
        evaluator (const model & M, size_t nfwd)
            : layers (M.layers), mean (M.mean), std (M.std), Pu (M.Pu), ivectormean (M.ivectormean), ivectorstd (M.ivectorstd)
        {
            alloclayerstate (layerstate, nfwd);
            assert (nfwd == this->nfwd());
        }

#if 0   //hack for getting activation histogram [v-xieche]
        ~evaluator()
        {
            rbmstatevectorsref outlayer (layerstate[1].stripe (ts, te - ts));
            outlayer.lockforreadwrite();
            foreach_coord (i, j, outlayer)
            {
                size_t index = int (outlayer(i,j) * ngrid);
                if(index > ngrid-1)
                {
                    index = ngrid - 1;
                }
                else if(index < 0)
                {
                    index = 0;
                }
                hist(i, index) ++;
            }
            outlayer.unlock();

            for(size_t hline = 0; hline < ngrid; hline ++)
                fprintf(stderr, "%0.2f-%0.2f ", double(hline) / ngrid, double(hline + 1) / ngrid);
            fprintf(stderr, "\n");
            for(size_t nid = 0; nid < nnode; nid ++)
            {
                fprintf(stderr, "Nodeid %4d : ", nid);
                for(size_t gid = 0; gid < ngrid; gid ++)
                    fprintf(stderr, " %4d", hist(nid, gid));
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
#endif 

        // wait for completion of offloaded computation (CUDA)
        void synchronize()
        {
BEGINTIME1("dbn.h synchronize");
            if (!layerstate.empty())
                layerstate[0].synchronize();
ENDTIME1();
        }

    public:

        unsigned int dropoutrandomseed;
        void resetdropout (size_t seed) { dropoutrandomseed = (unsigned int) seed; }

        size_t statscounter;
        void resetstats() { statscounter = 0; }

        // forward propagation through network; result is returned as a reference
        // Only the sub-range [ts,te) of 'v' is used, and only that is updated in the layerstate and return value.
        // This function is called directly only by the training. For recognition, you want to call logPuv() below.
        // 'numlayerstoeval' and 'prenonlinearity' are special modes for bottleneck features and training experiments.
        // 'numlayerstoeval == 0' is allowed and means to only scale the input features to layerstate[0].
        // mask the senone section
        // Notes on dropout:
        //  - dropout is enabled by dropoutrate > 0
        //  - the resulting model will learn W matrices that are too large by 1/(1-dropoutrate)
        //  - exitdropouttraining() will make up for it by scaling W by (1-dropoutrate); call this before saving (do that right before exitcomputation())
        //  - when entering an epoch, call enterropouttraining() to scale W back to being too large
        template<class VMATRIX> const rbmstatevectors & forwardprop (const VMATRIX & v, size_t ts, size_t te, size_t numlayerstoeval, bool prenonlinearity, 
                                                                     const float dropoutrate = 0.0f, rbmstatevectorsref * pmask = NULL, bool evalcvmode = false)
        {
            bool updatetoplayer = false;  // judge whether updating toplayer now.
            assert (ts <= te && te <= v.cols());
            assert (te <= layerstate[0].cols());
            assert (v.rows() == vdim());

            // apply mean/std normalization
            // Note that mean/std are for a single frame, while v is a concatenation of
            // multiple neighbor frames.
            // TODO: We have a perf issue here. This CPU-side op takes 13..17 ms for 1014 frames, vs. 62 ms for forwardprop(); that's 25% (or ~8% or so of total).
            // But the write lock syncs BEFORE the matrix is being prepared.
            // Solution: a mode for ref writing: sync before or after (we can sync after without losing efficiency); just always use the same mode for a matrix.
            // TODO: why don't we scale on the GPU side? It'd reduce this to trivial computation cost.
//#define TIME_INLAYER
#ifdef TIME_INLAYER    // temporary for Frank's parallelization experiments; remove when done
            layerstate[0].synchronize();
            auto_timer copycost;
#endif
BEGINTIME1("forwardprop: compute inlayer");
            {
                rbmstatevectorsrefwriting inlayer (layerstate[0], ts, te-ts, false/*skip sync->CPU computes concurrently; instead sync after copying back*/);
BEGINTIME1("forwardprop: compute inlayer loop");
                size_t featdim = mean.size();               // features are augmented (speech frames with neighbors ; ivector)
                size_t augfeatdim = v.rows() - idim();      // speech-frame part goes up to 'augfeatdim'
                if (augfeatdim % featdim != 0)
                    throw std::logic_error ("forwardprop: inconsistent dimension");
                for (size_t t = ts; t < te; t++) foreach_row (i, v)
                {
                    if (i < augfeatdim)
                    {
                        size_t k = i % featdim;
                        inlayer(i,t-ts) = std[k] != 0.0f ? (v(i,t) - mean[k]) / std[k] : (v(i,t) - mean[k]);
                    }
                    else
                    {
                        size_t k = i - augfeatdim;
                        inlayer(i,t-ts) = ivectorstd[k] != 0.0f ? (v(i,t) - ivectormean[k]) / ivectorstd[k] : (v(i,t) - ivectormean[k]);
                    }
                }
ENDTIME1();
                // destructor of inlayer copies back the stripe to CUDA
                // since we passed 'false' to refwriting constructor, the destructor now will ensure copy op has completed to make sure we don't catch it next time (we won't actually; enough syncs inbetween!)
            }
ENDTIME1();
#ifdef TIME_INLAYER
            layerstate[0].synchronize();
            copycost.show ("inlayer setup complete");
#endif
            // determine top layer to compute --if passed SIZE_MAX then compute the full network (default)
            if (numlayerstoeval == SIZE_MAX)
                numlayerstoeval = layers.size();
            else if (numlayerstoeval == 100)   // for temporary test purpose for the binary function. [v-xieche]
            {
                numlayerstoeval = layers.size();
                updatetoplayer = true;
            }

//#define TIME_FWPROP
#ifdef TIME_FWPROP           // do some explicit time measurement, for optimizing model parallelism
            static double totaltime = 0.0;
            static size_t totalframes = 0;
            synchronize();
            auto_timer forwardtimer;
#endif

            // apply network
            for (size_t i = 0; i < numlayerstoeval; i++)
            {
                // compute the output of layer i
                // We use a rbmstatevectorsref stripe to compute only the requested sub-range [ts,te).
                rbmstatevectorsref inlayer  (layerstate[i].stripe   (ts, te - ts));
                rbmstatevectorsref outlayer (layerstate[i+1].stripe (ts, te - ts));
#ifdef STEEPERSIGMOID   // approximate the first layer to 0-1 decisionfusion. i.e. multiply the layer matrix x N, make the cure more steep. [v-xieche]
                float amp = float(AMPNUM);
                if (i != numlayerstoeval - 1) // first layer
                    layers[i]->multiplywith (amp);

#endif
                // apply dropout to inputs of this layer
                //if (i>0) inlayer.dump("forwardprop: prevlayer after dropout.");
                //outlayer.dump("forwardprop: outlayer before dropout.");
                if (dropoutrate > 0.0f && shouldapplydropoutatinputof (i))  // only apply drop out to the hidden neurons, and not the input layer
                {
                    static bool first = true;
                    if (first)
                    {
                        fprintf (stderr, "forwardprop: applying dropout at rate %.1f%%\n", dropoutrate * 100.0f), fflush (stderr);
                        first = false;
                    }
//fprintf (stderr, "forwardprop: dropout\n"), fflush (stderr);
                    // drop activations at a rate of 'dropoutrate'
                    // Special case: When doing CV LL evaluation, we are in a limbo state in that the model is off-scale but we need to apply it.
                    // We pretend fixing the model by scaling the activations instead.
                    if (!evalcvmode)                                        // regular dropout (forwardprop() during training)
                        inlayer.dropout (dropoutrate, dropoutrandomseed++);
                    else                                                    // we are in evalcvll()
                        inlayer.scale (1.0f - dropoutrate);
                }

                //printf("######------ Forward Prop, Layer %d ------######\n", i);
                if (i == numlayerstoeval - 1 && prenonlinearity)   // last layer: option to bypass non-linearity
                    layers[i]->forwardprop (inlayer, outlayer, true, pmask);
                else if (i == numlayerstoeval -1)
                    layers[i]->forwardprop (inlayer, outlayer, false, pmask);
                else
                    layers[i]->forwardprop (inlayer, outlayer);
                //printf("######------ End Forward Prop, Layer %d ------######\n", i);

                // test for the binarize the output of hidden layer when updating the toplayer[v-xieche]
                // This argument will be executed only when numlayerstoeval = 100, this is only when 
                // execute backpropagationstats_quan() function. otherwise updatetoplayer will be SIZE_MAX 
                // or less than 100 normally [v-xieche]
#if 0
                if (i == 0 && updatetoplayer)   // quantilize for the hidden layer.
                {
                    outlayer.binarize ();
                }
#endif

#ifdef STEEPERSIGMOID  // use steeper sigmoid function. [v-xieche]
                if (i != numlayerstoeval - 1)
                    layers[i]->multiplywith ((float) 1.0/amp);
#endif
#ifdef LOGINSIGMOID   // log(sigmoid(z) + epison) function for hidden layer.[v-xieche]
                if (i != numlayerstoeval -1) 
                {
                    // sprintf (stderr, "exerting log..\n");
                    outlayer.addepisonlog();
                }
#endif
#ifdef SPARSENESSOUTPUTOFHIDDENLAYER   // utilized the sparseness of output of hidden layer. [v-xieche]
                const float thresholdvalue = 0.1;
                if (i != numlayerstoeval - 1)
                    outlayer.setto0ifbelow (thresholdvalue);
#endif
            }
#ifdef TIME_FWPROP
            synchronize();
            totaltime += forwardtimer;
            totalframes += te - ts;
            forwardtimer.show ("time measurement for forwardprop");
            fprintf (stderr, "averaged forwardprop time: %.2f fps (total time %.3f s for %d frames)\n", totalframes / totaltime, totaltime, (int) totalframes);
            fflush (stderr);
#endif
            if (statscounter < 1/*6*/ || statscounter % 64 == 0)    // some statistics --this is super-expensive!!
            {
               // fprintf (stderr, "forwardprop: ---------- layer stats ---------\n");
                for (size_t k = 0; k <= numlayerstoeval; k++)
                {
                    // layer statistics
                    const rbmstatevectorsrefreading outlayer (layerstate[k], ts, te - ts);
                    double stdsum = 0.0;
                    double sum = 0.0;
                    double positive = 0.0;
                    double lower20th = 0.0;
                    double upper20th = 0.0;
                    foreach_column (t, outlayer)
                    {
                        // compute stats over one activation vector
                        double sqrsum = 0.0;
                        foreach_row (i, outlayer)
                        {
                            float v = outlayer(i,t);
                            sqrsum += v*v;
                            sum += v;
                            positive += (v > 0.0f);     // non-saturated-0, useful for ReLU
                            lower20th += (v < 0.05f);   // first 1/20 slot [0..0.05] assuming sigmoid
                            upper20th += (v >= 0.95f);  // last 1/20 slot [0.95..1]
                        }
                        sqrsum /= outlayer.rows();  // variance
                        stdsum += sqrt (sqrsum);    // sum over col stddevs
                    }
                    // averages
                    double avstd = stdsum / outlayer.cols();
                    double mean = sum / (outlayer.rows() * outlayer.cols());
                    double nonnull = positive / (outlayer.rows() * outlayer.cols());
                    double under05 = lower20th / (outlayer.rows() * outlayer.cols());
                    double above95 = upper20th / (outlayer.rows() * outlayer.cols());
                  //  fprintf (stderr, "forwardprop: layerstate[%d] av stddev = %.5f ; mean = %.5f ; active = %.1f%% ; [..0.05] = %.1f%% ; [0.95..] = %.1f%%\n",
                   //          k, avstd, mean, nonnull * 100.0, under05 * 100.0, above95 * 100.0);
                }
            //    fprintf (stderr, "forwardprop: --------------------------------\n");
            }
            statscounter++;
#if 0    // hack for getting activation histogram
            for(int l = 0; l < hist.cols(); l ++)
            {
                rbmstatevectorsref outlayer (layerstate[l + 1].stripe (ts, te - ts));
                outlayer.lockforreadwrite();
                foreach_coord (i, j, outlayer)
                {
                    //fprintf(stderr, "(%d, %d) = %f\n", i, j, outlayer(i,j));
                    size_t index = int (outlayer(i,j) * nresolution);
                    if(index > nresolution - 1)
                    {
                        index = nresolution - 1;
                    }
                    else if(index < 0)
                    {
                        index = 0;// how come to get here?
                    }
                    hist(index, l) ++;
                }
                outlayer.unlock();
                /*size_t sum = 0;
                fprintf(stderr, "for layer %d\n", l + 1);
                for(int i = 0; i < nresolution; i ++)
                {
                fprintf(stderr, "%d = %d\n", i, hist(l, i));
                sum += hist(l, i);
                }
                fprintf(stderr, "sum is %d\n", sum);*/
            }
#endif

            // result is now in top layer's layerstate
            const auto & toplayerstate = layerstate[numlayerstoeval];
            return toplayerstate;
        }

#if 0    // hack for getting activation histogram, passed hist out because we want count hist for all utterance
         // but current design utterance are clear after each utterance
        void getacticationhist(msra::basetypes::matrix<size_t> & matrix) const
        {
            matrix.resize(hist.rows(), hist.cols());
            foreach_coord(i, j, hist)
                matrix(i,j) = hist(i,j);
        }
#endif

#ifdef UNSEEN_COMPENSATION 
        // [v-hansu] for unseen compensation
        void forwardpropwithoutbias (size_t ts, size_t te, size_t thislayer)
        {
            rbmstatevectorsref inlayer  (layerstate[thislayer].stripe   (ts, te - ts));
            rbmstatevectorsref outlayer (layerstate[thislayer+1].stripe (ts, te - ts));
            (dynamic_cast<msra::dbn::perceptron *> (layers[thislayer].get()))->forwardpropwithoutbias (inlayer, outlayer);
        }
#endif
            
    protected:

        // transfer uids[] vector to CUDA-side 'float' vector and return the stripe in ready-to-use form
        mutable rbmstatevectors fuids;          // tmp to move uids reference to CUDA
        template<class UIDSVECTOR>
        rbmstatevectorsref uidsstripe (const UIDSVECTOR & uids, size_t ts, size_t te) const
        {
            fuids.resize (1, nfwd());
            {
                rbmstatevectorsrefwriting fuids_stripe (fuids, ts, te-ts);
                for (size_t t = ts; t < te; t++)
                    fuids_stripe (0,t-ts) = (float) uids[t];    // store in a 'float' row vector
                // destructor syncs back fuids
            }

            // return the stripe in desired format
            return fuids.stripe (ts, te-ts);
        }

        // transfer target to GPU-side in auto-encoder mode
        mutable rbmstatevectors targets;                // tmp to move target reference to CUDA
        template<class VECTOR>
        rbmstatevectorsref targetstripe (const VECTOR & target, size_t ts, size_t te) const
        {
            targets.resize (target.rows(), target.cols());
            {
                rbmstatevectorsrefwriting targets_writer (targets, ts, te-ts);
                for (size_t t = ts; t < te; t++) foreach_row (i, target)
                {
                    targets_writer(i, t-ts) = target (i, t);
                }
                // destructor syncs back targets
            }
            // return the stripe in desired format
            return targets.stripe (ts, te-ts);
        }

    public:

        // -------------------------------------------------------------------
        // helpers for training
        // -------------------------------------------------------------------

        // compute posterior of reference, as one way of convergence tracking
        // Returns av. log posterior. Also prints av. posterior and training-batch frame accuracy.
        // If 'nosoftmax' then the top layer is the linear output without normalization.
        mutable rbmstatevectors posteriorstatsbuffer1, posteriorstatsbuffer2, posteriorstatsbuffer3; // (devid,t) vector buffers for posterior statistics
        template<class UIDSVECTOR>
        std::pair<double, double> posteriorstats (const UIDSVECTOR & uids, size_t ts, size_t te, bool nosoftmax, int verbosity = 2) const
        {
#if 1
            // TODO: gpus should be named better since it applies also to the no-GPU case
            size_t gpus = numcudadevices();
            if (gpus == 0)
                gpus = 1;
            
            // space for intermediate (column-wise) results (for CUDA use)
            posteriorstatsbuffer1.resize (gpus, nfwd());    // one row for each GPU; one row in total for CPU mode
            posteriorstatsbuffer2.resize (gpus, nfwd());    // TODO: it would be more efficient to use a column per GPU
            posteriorstatsbuffer3.resize (gpus, nfwd());
            // inputs
            const auto   fu = uidsstripe (uids, ts, te);                    // ground truth in CUDA-compatible format
            const rbmstatevectorsref Pu (layerstate[layers.size()].stripe (ts, te-ts)); // actual probabilities

            // results
            double avlogpp;     // log posterior
            double avpp;        // posterior
            double avfcor;      // rate of frames correctly detected

            // compute it
            rbmstatevectorsref buf1 = posteriorstatsbuffer1.stripe (ts, te-ts);
            rbmstatevectorsref buf2 = posteriorstatsbuffer2.stripe (ts, te-ts);
            rbmstatevectorsref buf3 = posteriorstatsbuffer3.stripe (ts, te-ts);
            fu.posteriorstats (Pu, buf1, buf2, buf3, avlogpp, avpp, avfcor, nosoftmax);  
#else
            const rbmstatevectorsrefreading u (layerstate[layers.size()], ts, te-ts);
            // TODO: inefficient, this will copy once again --keep a local cache, or do it in backprop()
            checknan (u);
            double sumlogpp = 0.0;  // log posterior, summed up
            double sumpp = 0.0;     // posterior, summed up
            size_t sumfcor = 0;     // frames correctly detected
            for (size_t t = ts; t < te; t++)
            {
                size_t clsid = uids[t];
                double pp = u(clsid,t-ts);
                sumpp += pp;
                sumlogpp += log (max (pp, 0.000001));   // (avoid underflow if prob has been rounded to 0)
                // which is the max?
                size_t imax = clsid;
                foreach_row (i, u)
                    if (u(i,t-ts) >= u(imax,t-ts))
                        imax = i;
                if (imax == clsid)
                    sumfcor++;
            }
            const size_t n = te - ts;
            const double avlogpp = sumlogpp / n;
            const double avpp = sumpp / n;
            const double avfcor = sumfcor / (double) n;
#endif
            // we ony log av log pp
            if (verbosity >= 2)
                fprintf (stderr, "posteriorstats: avlogPP=%.2f  avPP=%.2f  frames correct=%.1f%%  in %d frames\n",
                    avlogpp, avpp, 100.0f * avfcor, (int) (te-ts));
#if 0   // hack
            const rbmstatevectorsrefreading vtop (layerstate[layers.size()-1], ts, te-ts);
            double lensum = 0.0;
            foreach_column (t, vtop)
            {
                double colsum = 0.0;
                foreach_row (i, vtop) colsum += vtop(i,t) * vtop(i,t);
                double len = sqrt (colsum);
                lensum += len;
            }
            double avlen = lensum / (te-ts);
            fprintf (stderr, "evaluate: av. len of top layer's %d-dim input vectors: %.8f\n", vtop.rows(), avlen);
#endif
            return std::make_pair (avlogpp, avfcor);
        }

        // keep a running sum over all P(u|v), for use as a prior
        // Note: This is imprecise, because the model is not final.
        mutable rbmmodelmatrix Pusums;      // CUDA temp for accumulating Pu
        mutable matrix Pusumstmp;           // CPU temp
        void accumulatepriors (std::vector<double> & Pusum, size_t & Pusumcount, size_t ts, size_t te) const
        {
            if (Pusums.empty()) // Pusums follows the model-parameter paradigm, so need to 'entercomputation'
            {
                Pusums.resize (udim(), 1);
                Pusums.entercomputation();
            }
            // sum up all columns in CUDA space
            const auto & u = layerstate[layers.size()];
            Pusums.scaleandaddallcols (0.0f, u.stripe (ts, te-ts), 1.0f, Pusumstmp);
            // now accumulate into CPU-side overall accumulator
            // Note: This performs a 'sync'. That's OK because this happens before model update, so it won't harm overlapping with CPU-side MB reading too much.
            //       However, we need to sync anyway for posteriorstats(), so maybe we can move it there. --TODO: try this
            Pusums.accumulate (Pusum);
            Pusumcount += (te-ts);
        }
        // accumulate from features (perform forwardprop)
        template<class VMATRIX>
        void accumulatepriors (const VMATRIX & v, std::vector<double> & Pusum, size_t & Pusumcount, size_t ts, size_t te)
        {
            // perform forwardprop if needed (we do in fixpriors())
            forwardprop (v, ts, te, SIZE_MAX, false);
            // now accumulate
            accumulatepriors (Pusum, Pusumcount, ts, te);
        }

        // -------------------------------------------------------------------
        // LL evaluation
        // -------------------------------------------------------------------

        // get overall log likelihood / p(v) based on a previously done forwardprop()
        // p(v|u) = p(u|v) * p(v) / p(u)
        // This computes p(u|v) / p(u) (since p(v) is a constant per frame)
        // 'v' is supposed to be the final feature vector (with neighbor frames or whatever)
        // If 'nosoftmax' then the value is already logarithmic.
        std::vector<float> logPuCache;
        template<class UMATRIX> void logLL (UMATRIX & Pugv, const bool divbyprior, const bool nosoftmax)
        {
            const rbmstatevectorsrefreading u (layerstate[layers.size()], 0, Pugv.cols());
            assert (u.rows() == udim() && u.cols() == Pugv.cols());

            assert (Pugv.cols() == u.cols());
            if (nosoftmax)                                  // get logLLs for reco, e.g. lattice rescoring  --top layer output has been left linear to avoid unnecessary exp()/log()
            {
                if (!divbyprior)
                    throw std::logic_error ("logLL: nosoftmax supposed to be used for LL eval, so divbyprior is expected");
                // cache the log of the Pu (log is expensive)
                logPuCache.resize (Pu.size());
                float maxLogPu = -1e30f;
                foreach_index (i, Pu)                       // TODO: can we do this more cleverly to cache this only once instead of each minibatch?
                {
                    logPuCache[i] = logf (Pu[i]);
                    if (logPuCache[i] > maxLogPu)
                        maxLogPu = logPuCache[i];
                }
                foreach_coord (i, j, Pugv)                  // TODO: don't we have some compount statement for this, using SSE?
                    Pugv(i,j) = u(i,j) - (logPuCache[i] - maxLogPu);
            }
            else if (divbyprior)                            // divide by prior
            {
                foreach_coord (i, j, Pugv)
                {
                    const float Pugvij = u(i,j) / Pu[i];
                    Pugv(i,j) = Pugvij > 1e-30f ? logf (Pugvij) : -1e30f;
                }
            }
            else
            {
                foreach_coord (i, j, Pugv)
                    Pugv(i,j) = u(i,j) > 1e-30f ? logf (u(i,j)) : -1e30f;
            }

        }

        // evaluate overall log likelihood / p(v)
        // TODO: rename to a correct name (it is NOT P(u|v))
        template<class VMATRIX, class UMATRIX> void logPuv (const VMATRIX & v, UMATRIX & Pugv, const bool divbyprior, bool nosoftmax)
        {
            assert (v.cols() == Pugv.cols());
            assert (Pugv.rows() == udim());
            assert (Pu.size() == udim());

            // perform forward propagation through network -> u
            forwardprop (v, 0, v.cols(), SIZE_MAX, nosoftmax);

            // convert to scaled likelihoods
            logLL (Pugv, divbyprior, nosoftmax);
        }

        // logPuv from previous forwardprop() pass

        // evaluate a layer's activations
        // 'v' is supposed to be the final feature vector (with neighbor frames or whatever)
        // The top layer's output activation values are placed in Eh.
        // Differs from logPuv in that no priors are applied and no log is taken.
        // Currently used in experimental ML initialization (top layer) and bottleneck features (intermediate layer).
        // 'layer' = SIZE_MAX means top layer. 'prenonlinearity' allows to bypass the sigmoid (for bottleneck features).
        template<class VMATRIX, class UMATRIX> void evaluate (const VMATRIX & v, UMATRIX & Eh, size_t atlayer/*or SIZE_MAX*/, bool prenonlinearity)
        {
            assert (v.cols() == Eh.cols());
            if (atlayer == layers.size()) assert (Eh.rows() == udim()); else assert (Eh.rows() == layerstate[atlayer].rows());

            // perform forward propagation through network -> u
            const rbmstatevectorsrefreading u (forwardprop (v, 0, v.cols(), atlayer, prenonlinearity), 0, v.cols());
            // u is layer activations[atlayer], i.e. it can be an intermediate result

            // copy out the result
            foreach_coord (i, j, Eh)
                Eh(i,j) = u(i,j);
        }
    };

    // -----------------------------------------------------------------------
    // class accmulator --for thread-local accumulation step
    // This is no longer used and can be merged with 'trainer'
    // -----------------------------------------------------------------------

    class accumulator : public evaluator
    {
    protected:
        rbmstatevectors v1, h1;                         // pre-training: updated v and h after 1 step of CD (top layer only)
        size_t firstbplayer;                            // first layer that gets updated by backpropagation
        std::vector<rbmstatevectors> errorstate;        // [firstbplayer..numlayers-1] back-propagated error vectors; note that the paper's e^L=errorstate[L+1]!
        std::vector<rbmstatevectors> deltastate;        // for unseenstates compensations [v-hansu] deltaH = H .* (1-H) .* (H .* (1-H) * E .* diag(V' * V) + W' * deltaV)
        std::vector<rbmstatevectors> vnormsbufs;        // for diag(V' * V), store it as a row vector
        rbmstatevectors keepsampleflags;                // [t] temp for sub-sampling frames (1.0 = keep; 0.0 = remove)
        rbmstatevectors maskmatrix;                     // to mask the output layer to support multi set of senones in a hacky way
    public:
        // This creates:
        //  - v and h for forward propagation
        //  - RBM pt (bpmode = false): vectors to store updated v and h
        //  - bp (bpmode = true): vectors to store the shared intermediate error values
        // ... TODO: rename 'nfwd' to 'T' or something like that
        accumulator (const model & M, size_t nfwd, bool bpmode, size_t finetunedlayers) : evaluator (M, nfwd), firstbplayer (0)
        {
            if (bpmode)
            {
                if (finetunedlayers > M.numlayers()) finetunedlayers = M.numlayers();
                firstbplayer = M.numlayers() - finetunedlayers;
                if (firstbplayer > 0)
                    fprintf (stderr, "accumulator: backpropagation limited to layers %d..%d\n", (int) firstbplayer, M.numlayers() -1);
                alloclayerstate (errorstate, nfwd);
                errorstate[firstbplayer].resize (0, nfwd);      // we don't want lowest level ... TODO: do this nicer
#ifdef UNSEEN_COMPENSATION
                alloclayerstate (deltastate, nfwd);
                deltastate[firstbplayer].resize (0, nfwd);
                allocvnormbufs (vnormsbufs, nfwd);
#endif
            }
            else
            {
                size_t toplayer = layers.size() -1;
                v1.resize (layers[toplayer]->vdim(), nfwd);
                h1.resize (layers[toplayer]->hdim(), nfwd);
            }
            keepsampleflags.resize (1, nfwd);
        }

        // unsupervised pre-training accumulation
        //  - input = columns of feature vectors
        // This operates on the time stripe [ts,te)
        void pretrainingstats (const matrixbase & v_in, size_t ts, size_t te, unsigned int randomseedframebase)
        {
            size_t toplayer = layers.size() -1;
            const auto & Ph = layerstate[toplayer +1];

            rbmstatevectorsref Ph_stripe (Ph.stripe (ts, te - ts));
            rbmstatevectorsref v1_stripe (v1.stripe (ts, te - ts));
            rbmstatevectorsref h1_stripe (h1.stripe (ts, te - ts));

            layers[toplayer]->pretrainingstats (Ph_stripe, v1_stripe, h1_stripe, randomseedframebase);
        }

        // compute log LL for reconstruction, for tracking pre-training
        // To make the av log LL more readable, the LL is normalized by the null hypothesis
        // of perfect reconstruction. That normalization only depends on the input data,
        // and it takes out a large constant offset of the number which does not add
        // value and makes small changes so much harder to see.
        // This function expects pretrainingstats() to have been called before.
        // TODO: This function is no longer reentrant, so what's the point of passing ts, te ?
        mutable rbmstatevectors glogllsums, logllsums; // [i] vector buffers for likelihood values for statistics
        double llstats (size_t ts, size_t te) const
        {
            const size_t toplayer = layers.size() -1;

            logllsums.resize  (layers[toplayer]->vdim(), 1);
            glogllsums.resize (layers[toplayer]->vdim(), 1);

            assert (layers[toplayer]->type() == "rbmgaussbernoulli" || layers[toplayer]->type() == "rbmbernoullibernoulli");
            const bool gaussian = (layers[toplayer]->type() == "rbmgaussbernoulli");

            double glogllsum = 0.0; // Gaussian (also for binary units, for diagnostics)
            double logllsum = 0.0;
            rbmstatevectorsref glogllsums_stripe (glogllsums.stripe (0, 1));
            rbmstatevectorsref logllsums_stripe  (logllsums.stripe (0, 1));
            layerstate[toplayer].stripe (ts, te-ts).llstats (v1.stripe (ts, te-ts), glogllsums_stripe, logllsums_stripe, glogllsum, logllsum);

            fprintf (stderr, "llstats: avlogLL=%.5f, av Gaussian logLL=%.5f\n", logllsum / (te - ts), glogllsum / (te - ts));

#if 1
            if (gaussian)
                return glogllsum / (te - ts);
            else
                return logllsum / (te - ts);
#else
            //return logllsum / (te - ts);    // av log LL per frame
            return glogllsum / (te - ts);    // returning Gaussian distance for now (although it seems more noisy)
#endif
        }

        // first stage of BP and pre-training accumulation, the forward propagation
        void forwardprop (const matrixbase & v, size_t ts, size_t te, float dropoutrate, bool nosoftmax = false, bool evalcvmode=false)
        {
            evaluator::forwardprop (v, ts, te, SIZE_MAX, nosoftmax, dropoutrate, NULL /*pmask*/, evalcvmode);
        }

        // we need to decide which section of the senones are related using uids
        // we then need to mask out the rest senones 
        template<class UIDSVECTOR, class INTVECTOR>
        void forwardpropwithmultisenonesets (const matrixbase & v, const UIDSVECTOR & uids, size_t ts, size_t te, float dropoutrate, const INTVECTOR &senoneboundaries, bool nosoftmax = false)
        {
            // generate mask based on the uids and senone boundaries
            float maskvalue = -1000;

            assert(senoneboundaries.size()>2 && (size_t)senoneboundaries[0]==0);
            size_t numsenoneboundaries = (size_t)senoneboundaries.size();
            if (maskmatrix.rows() != (size_t)senoneboundaries[numsenoneboundaries-1] ||
                maskmatrix.cols() !=te-ts)
            {
                maskmatrix.resize((size_t)senoneboundaries[numsenoneboundaries-1], te-ts);
            }

            rbmstatevectorsref mask(maskmatrix.stripe(0, maskmatrix.cols()));
            mask.lockforwriting();
            for (size_t j=0; j<mask.cols(); j++)
            {
                for (size_t i=1; i<numsenoneboundaries; i++)
                {
                    if ((size_t)uids[j] < (size_t)senoneboundaries[i] && (size_t)uids[j] >= (size_t)senoneboundaries[i-1]) // inside the segment
                    {
                        for (size_t k=(size_t)senoneboundaries[i-1]; k<(size_t)senoneboundaries[i]; k++)
                        {
                            mask(k,j)=0.0f;
                        }
                    }
                    else
                    {
                        for (size_t k=(size_t)senoneboundaries[i-1]; k<(size_t)senoneboundaries[i];k++)
                        {
                            mask(k,j)=maskvalue;
                        }
                    }
                }

            }
            mask.unlock();
            evaluator::forwardprop (v, ts, te, SIZE_MAX, nosoftmax, dropoutrate, &mask);
        }

#if 0
        // update the output posteriors externally
        // This was intended for MMI training, which we try to reflect in the name of the function.
        // Call this before backpropagationstats2(), which will be based on these gammas.
        // Currently not used.
        template<class UMATRIX> void setdenominatorgammas (const UMATRIX & pp, float keepweight)
        {
#if 0
            keepweight;
            // we replace the output layer's activations
            rbmstatevectorsrefwriting u (layerstate[layers.size()], 0, pp.cols());
            assert (u.rows() == pp.rows());

            // copy over the posteriors
            foreach_coord (s, t, pp)
                u(s,t) = pp(s,t);
#else
            // we interpolate the output layer's activations
            rbmstatevectorsref u (layerstate[layers.size()].stripe (0, pp.cols()));
            u.lockforreadwrite();
            assert (u.rows() == pp.rows());

            // copy over the posteriors
            foreach_coord (s, t, pp)
                u(s,t) = u(s,t) * keepweight + pp(s,t) * (1.0f - keepweight);
            u.unlock();
#endif
        }
#endif

        // update the top-level error signal from MMI training
        // Call this before errorbackprop(), which will be based on these gammas.
        template<class UMATRIX, class UIDSVECTOR> void seterrorsignalmmi (const UMATRIX & numgammas, const UMATRIX & dengammas, size_t ts, size_t te, float keepweight, const UIDSVECTOR & uids)
        {
            // we interpolate the output layer's activations
            const rbmstatevectorsrefreading Pu (layerstate[layers.size()], ts, te - ts);

            // TODO: why not use rbmstatevectorsrefwriting here?
            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));  // -> error goes here
            err.lockforwriting();

            // copy over the posteriors
            for (size_t t = ts; t < te; t++)
            {
                foreach_row (s, err)
                    err(s,t-ts) = numgammas(s,t-ts) - (dengammas(s,t-ts) * (1.0f - keepweight) + Pu(s,t-ts) * keepweight);
            }
            err.unlock();   // sync it back
        }

        // update the top-level error signal from a matrix  --for use with sMBR training, where the final error signal is prepared during lattice processing
        // Call this before errorbackprop(), which will be based on these gammas.
        template<class UMATRIX, class UIDSVECTOR> void seterrorsignal (UMATRIX & errorsignal, const UIDSVECTOR & uids, const size_t ts, const size_t te, const float hsmoothingweight)
        {
            rbmstatevectorsref err (errorstate[layers.size()].stripe (0, errorsignal.cols()));  // -> error goes here
            if (hsmoothingweight == 1.0f)       // no need for interpolation
                err.assign (errorsignal, true/*sync*/);
            else
            {
                const rbmstatevectorsref fu (uidsstripe (uids, ts, te)); 
                rbmstatevectorsref Pu (layerstate[layers.size()].stripe (0, errorsignal.cols()));
                err.assign (errorsignal, true);
                err.seterrorsignalhsmoothing (fu, Pu, err, hsmoothingweight, 1/*errorsettingmode 1 : smbr + ce*/, 1.0f/*framedropthresh disabled*/);
                // err.seterrorsignalhsmoothing (fu, Pu, err, hsmoothingweight, 2/*errorsettingmode 2 : smbr + fsmbr*/);
            }
        }

        // this analyze the gammas and ce pps of each minibatch
        template<class UIDSVECTOR, class UMATRIX> 
        void mmidiagnosis (const matrixbase & v, const UMATRIX & dengammas, msra::dbn::model::evaluator * prefevaluator, 
                           const UIDSVECTOR & uids, const size_t ts, const size_t te)
        {
            static size_t diagnosticround = 0;
            static FILE *f = fopen ("mmidiagnosis.log", "w");
            fprintf (f, "diagnosticround %d:\n", diagnosticround);
            const rbmstatevectorsrefreading refrefu (layerstate.back(), ts, te);
            
            const size_t numstates = refrefu.rows();
            const size_t numframes = te - ts;
            float avunseenmmisumppnonsil = 0;       // average of sum of ce pp of unseen states over minibatches without silence frames
            float avunseenmmimaxppnonsil = 0;       // average of max of ce pp of unseen states over minibatches without silence frames
            float avrightceppnonsil = 0;            // average of ce pp of the label state
            size_t numframesnonsil = 0;
            size_t numframescerightnonsil = 0;      // accuracy of mmi hypo (nosil)
            size_t numframesmmirightnonsil = 0;     // accuracy of ce hypo (nosil)
            const size_t unitsil2 = 7670;           // this is a hack function, and these numbers are for the 9304 model only 
            const size_t unitsil4 = 7671;
            const size_t unitsilst = 7672;
            if (numstates != 9304)
                throw::logic_error ("mmidiagnosis: this function only applies to 9304-state model");
            foreach_column (t, refrefu)
            {
                float unseensumpp = 0;
                size_t numunseen = 0;
                float unseenmaxpp = FLT_MIN;
                size_t unseenmaxid = 0;
                float maxcepp = FLT_MIN;
                size_t maxceid = 0;
                float maxmmipp = FLT_MIN;
                size_t maxmmiid = 0;
                foreach_row (s, refrefu)
                {
                    if (refrefu(s,t) > maxcepp)
                    {
                        maxcepp = refrefu(s,t);
                        maxceid = s;
                    }
                    if (dengammas(s,t) > maxmmipp)
                    {
                        maxmmipp = dengammas(s,t);
                        maxmmiid = s;
                    }
                    if (dengammas(s,t) == 0.0)
                    {
                        unseensumpp += refrefu (s,t);
                        if (refrefu (s,t) > unseenmaxpp)
                        {
                            unseenmaxpp = refrefu (s,t);
                            unseenmaxid = s;
                        }
                        numunseen++;
                    }
                }
                if (uids[t] != unitsil2 && uids[t] != unitsil2 && uids[t] != unitsilst)      // this indicates that label is not sil or sp
                {
                    numframesnonsil++;
                    avunseenmmisumppnonsil += unseensumpp;                      // actually it is sum here, will be average out of loop
                    avunseenmmimaxppnonsil += unseenmaxpp;
                    avrightceppnonsil += refrefu (uids[t],t);
                    if (uids[t] == maxceid)
                        numframescerightnonsil++;
                    if (uids[t] == maxmmiid)
                        numframesmmirightnonsil++;
                }
                fprintf (f, "frm %4d, uid %4d, ceppmax %3.4f%%, cePmax %3.4f%%, ceright %d, mmippmax %3.4f%%, mmiright %d, unseen sumpp %3.4f%%, unseen maxpp %3.4f%%, unseen maxP %3.4f%%, unseen pct %3.4f%%\n", // pct means percentage
                         t, uids[t], maxcepp * 100, Pu[maxceid] * 100, int (uids[t] == maxceid), maxmmipp * 100, int (uids[t] == maxmmiid), unseensumpp * 100, unseenmaxpp * 100, Pu[unseenmaxid] * 100, ((float)numunseen)/numstates * 100);
            }
            avunseenmmisumppnonsil /= numframesnonsil;
            avunseenmmimaxppnonsil /= numframesnonsil;
            avrightceppnonsil /= numframesnonsil;
            fprintf (f, "diagminibatch %d: nonsil pct %3.2f%%, mmihp acc %3.2f%%, cehp acc %3.2f%%, av unseen sumcepp %3.4f%%, av unseen maxcepp %3.4f%%, av ce of correct state %3.4f%%\n\n", 
                     diagnosticround, ((float)numframesnonsil)/numframes * 100, ((float)numframesmmirightnonsil)/numframesnonsil * 100, 
                     ((float)numframescerightnonsil)/numframesnonsil * 100, avunseenmmisumppnonsil * 100, avunseenmmimaxppnonsil * 100, avrightceppnonsil * 100);
            
            diagnosticround++;
            fflush(f);
        }

        rbmstatevectors fsenone2update;          // this is a hack [v-hansu]
        // set the top-layer error signal from gammas in CPU memory
        // TODO: This is incorrectly named. What's a better name? setmmierrorsignal()? In the sense that CE training is also MMI?
        template<class UIDSVECTOR, class UMATRIX>
        void setgammas (const matrixbase & v, const UIDSVECTOR & uids, const size_t ts, const size_t te, 
                        /*const --comment out for Hcriteriamode*/ UMATRIX & dengammas, 
                        msra::dbn::model::evaluator * prefevaluator, const float alpha, 
                        const std::vector<bool> & senone2update, const float hsmoothingweight, const float framedropthresh)
        {
            fprintf (stderr, "setgammas: start\n");
            rbmstatevectorsref Pu (layerstate[layers.size()].stripe (0, dengammas.cols()));
            rbmstatevectorsref err (errorstate[layers.size()].stripe (0, dengammas.cols()));
            const rbmstatevectorsref fu  (uidsstripe (uids, ts, te)); 
            // lazy init, if fsenone2update is not allocated while senone2update's size is not zero
            if (fsenone2update.cols() == 0 && senone2update.size() != 0)    // only update some of the states
            {
                fsenone2update.resize (1, senone2update.size());
                rbmstatevectorsrefwriting fsenone2updateref (fsenone2update, 0, senone2update.size());
                foreach_index (i, senone2update)
                    fsenone2updateref (0, i) = senone2update[i] ? 1.0f : 0.0f;
            }

#if 0       // distribution reallocation      // [v-hansu]
            // p_mmi(seenstates)  = p_mmi(seenstates) * sum(p_ce(seenstates))
            // p_mmi(unseenstates) = p_ce(unseenstates)
            Pu.lockforreadwrite();
            foreach_column (t, Pu)
            {
                double sumseenstatepp = 0;
                foreach_row (s, Pu)
                {
                    if (dengammas(s,t) != 0.0)
                        sumseenstatepp += Pu(s,t);
                }
                foreach_row (s, Pu)
                {
                    if (dengammas(s,t) != 0.0)
                        dengammas(s,t) *= (float) sumseenstatepp;
                    else
                        dengammas(s,t) = Pu(s,t);
                }
            }
            Pu.unlock();
#endif
            if (hsmoothingweight != 1.0f)               // need interpolation
            {
                fprintf (stderr, "setgammas: assigning error signal with smoothing\n");
                err.assign (dengammas, true);           // TODO: why not pass dengammas directly? Is it CPU only?
                err.seterrorsignalhsmoothing (fu, Pu, err, hsmoothingweight, 0/*errorsettingmode 0 : mmi + ce*/, framedropthresh);
            }
            else if (prefevaluator && alpha > 0)        // KL mode   --TODO: this and the next mode do not support frame dropping, not sure why
            {
                Pu.assign (dengammas, true/*sync*/);    // sync to make sure that upon return, dengammas(,) has been completely consumed and can be reused immediately
                const auto & refu = prefevaluator->forwardprop(v, ts, te, SIZE_MAX, false);
                const rbmstatevectorsref refPu  (refu.stripe (ts, te - ts));                          // actual probabilities
                err.seterrorsignalwithklreg (fu, Pu, refPu, alpha);
            }
            else
            {
                Pu.assign (dengammas, true/*sync*/);    // sync to make sure that upon return, dengammas(,) has been completely consumed and can be reused immediately
                rbmstatevectorsref fsenone2updateref (fsenone2update.stripe (0, fsenone2update.cols()));
                err.seterrorsignal (fu, Pu, fsenone2updateref);  // compute the error signal
            }
        }

        // helper to dump a matrix to a file
        template<class UMATRIX> void dumpstatistics (const UMATRIX & thismatrix, FILE *f)
        {
            printmatfile (thismatrix, f);
        }

        // helper to print a histogram over parameters in a matrix
        template<class UMATRIX> void dumphist (const UMATRIX & thismatrix, FILE *f, size_t numofbins)
        {
            std::vector<size_t> bins(numofbins, 0);
            double maxvalue = thismatrix(0, 0);
            double minvalue = thismatrix(0, 0);
            foreach_coord(i, j, thismatrix)
            {
                if (maxvalue < thismatrix(i, j))
                    maxvalue = thismatrix(i, j);
                if (minvalue > thismatrix(i, j))
                    minvalue = thismatrix(i, j);
            }
            foreach_coord(i, j, thismatrix)
            {
                size_t histindex = (size_t) ((thismatrix(i, j) - minvalue) / (maxvalue - minvalue) * numofbins);
                if (histindex == numofbins)
                    histindex--;
                bins[histindex]++;
            }
            for (size_t i = 0; i < numofbins; i++)
                fprintf(f, "%d\t", (int) bins[i]);
        }

        // supervised back-propagation accumulation  --set error signal
        //  - input = columns of feature vectors
        //  - target output (uids[ts..te-1]) = indices of supervised class ids corresponding to vectors
        // Compute the top error signal.
        // This function operates only on the column range [ts,te) and can be safely called from multiple threads for disjoint column ranges.
        // forwardprop() must be run before entering this function.
        template<class UIDSVECTOR>
        void seterrorsignal (const matrixbase & v, const UIDSVECTOR & uids, size_t ts, size_t te, msra::dbn::model::evaluator * prefevaluator, const float alpha)
        {
            const auto & u = layerstate[layers.size()];
            const rbmstatevectorsref fu  (uidsstripe (uids, ts, te));                       // reference
            const rbmstatevectorsref Pu  (u.stripe (ts, te - ts));                          // actual probabilities
            rbmstatevectorsref       err (errorstate[layers.size()].stripe (ts, te - ts));  // -> error goes here

            if (prefevaluator && alpha > 0)
            {   // TODO: make this compatible with MMI training
                const auto & refu = prefevaluator->forwardprop (v, ts, te, SIZE_MAX, false);
                const rbmstatevectorsref refPu  (refu.stripe (ts, te - ts));                          // actual probabilities
                err.seterrorsignalwithklreg (fu, Pu, refPu, alpha);
            }
            else
            {
                //printf("######------ SetErrorSignal ------######\n");
                rbmstatevectorsref fsenone2updateref (fsenone2update.stripe (0, 0));
                // NOTE: fsenone2updateref is a temp solution which indicates states to update [v-hansu]
                err.seterrorsignal (fu, Pu, fsenone2updateref);  // compute the error signal 
                //printf("######------ End SetErrorSignal ------######\n");
            }
        }

        // set error signal and track LL of reconstructed feature for auto-encoder (target is a numeric vector)
        rbmstatevectors glogllsumsofdae, logllsumsofdae;
        template<class VMATRIX>
        void setautoencodererrorsignalandtrackll (const VMATRIX & target, size_t ts, size_t te, double & avlogp)
        {
            const auto & u = layerstate[layers.size()];
            const rbmstatevectorsref uvals (u.stripe (ts, te - ts));                       // actual predicted values
            const rbmstatevectorsref tvals (targetstripe (target, ts, te));                // reference
            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));       // -> error goes here

            err.settodiff (tvals, uvals);  // err = target - actual (L2 norm criterion)

            // tracking the LL of reconstructed feature, we combine this here for efficiency (no need to copy again)

            double logllsum = 0.0;
            double glogllsum = 0.0; // for Gaussian
           
            logllsumsofdae.resize  (target.rows(), 1);
            glogllsumsofdae.resize (target.rows(), 1);           
            rbmstatevectorsref glogllsumsofdae_stripe (glogllsumsofdae.stripe (0, 1));
            rbmstatevectorsref logllsumsofdae_stripe  (logllsumsofdae.stripe (0, 1));
           
            uvals.llstats (tvals, glogllsumsofdae_stripe, logllsumsofdae_stripe, glogllsum, logllsum);
            avlogp = glogllsum / (te - ts);

            fprintf (stderr, "llstats: avlogLL=%.5f, av Gaussian logLL=%.5f\n", logllsum / (te - ts), avlogp);
        }

        // error signal for mixtures:
        //  - P(c|s,v) - P(c|v)
        //  - c is a global mixture index
        //  - P(c|s,v) is the mixture-conditional, delta_c_in_s P(c|v) / sum_c_in_s P(c|v)
        //  - P(c|v) is the softmax
        // implementation to tack it on top of existing system:
        //  - after global mix softmax, accumulate into state --first section
        //    -> gives us correct logPP and frame-acc estimate
        //  - but other sections keep their original softmax value!
        //  - in error signal, we have all information, but we need to reconstruct the first mix contribution
        // mixture layout:
        //  - output layer, e.g. 9304, duplicated #mix times
        //  - model (s,c) = entry s + #senones * c
        // combine all mixtures; store in lower range (0..#senones-1)
        // sum up the weighted-mixcomp posteriors of all #senones sections into first #senones section.
        // This yields the correct mixture posterior.
        // We also reset all others to 0, for compat with priors and frame-acc counting (this is hacky).
        // By resetting, those upper ones will never accidentally be the best state. Priors will just accumulate to 0.
        void summixturecomponents (const size_t nummix, size_t ts, size_t te)
        {
            const auto & outlayer = layers.back();
            const size_t numsenones = outlayer->hdim() / nummix;
            if (numsenones * nummix != outlayer->hdim())
                throw std::logic_error ("summixturecomponents: inconsistent nummix/udim");

            auto & u = layerstate[layers.size()];
            rbmstatevectorsref Pu  (u.stripe (ts, te - ts));  // actual probabilities
            Pu.lockforreadwrite();

            foreach_column (t, Pu) for (size_t s = 0; s < numsenones; s++)
            {
                for (size_t c = 1; c < nummix; c++)
                {
                    // sum block c into block c=0
                    const size_t coffset = c * numsenones;
                    Pu(s,t) += Pu(s + coffset, t);
                    // but keep original value for all other blocks
                }
            }

            Pu.unlock();
        }

#if 0
        // error signal for mixture case
        // We operate on the output of summixturecomponents(), which has summed up the mix comps into the first block in-place.
        template<class UIDSVECTOR>
        void setmixtureerrorsignal (const UIDSVECTOR & uids, size_t ts, size_t te, size_t nummix)
        {
            const auto & outlayer = layers.back();
            const_array_ref<size_t> fu (&uids[ts], te - ts);                                // reference
            const rbmstatevectorsrefreading Pu (layerstate[layers.size()], ts, te - ts);    // actual probabilities
            rbmstatevectorsrefwriting       err (errorstate[layers.size()], ts, te - ts);   // -> error goes here

            const size_t numsenones = outlayer->hdim() / nummix;
            if (numsenones * nummix != outlayer->hdim())
                throw std::logic_error ("setmixtureerrorsignal: inconsistent nummix/udim");

            // compute the error signal
            foreach_column (t, err) for (size_t s = 0; s < numsenones; s++)
            {
                bool iscorrects = (fu[t] == s); // true if s is the ground-truth state
                const float Ps = Pu(s,t);       // P(s|v) was already summed up in summixturecomponents()
                double Pc0 = Ps;                // P(c0|v) needs to be reconstructed as Ps - sum_c>0 Pc
                for (size_t c = 1; c < nummix; c++)
                {
                    const size_t coffset = c * numsenones;
                    const float Pc = Pu(s + coffset, t);
                    if (Pc > Ps)
                        fprintf (stderr, "setmixtureerrorsignal(%d,%d,%d): Pc > Ps (%.10f > %.10f)\n", (int) s, (int) t, (int) c, Pc, Ps);
                    // compute error signal
                    const float deltaterm = (iscorrects && Ps > 0.0f) ? (Pc / Ps) : 0.0f;
                    err(s + coffset, t) = deltaterm - Pc;
                    // reconstruct first mixture component
                    Pc0 -= Pc;
                }
                // and for block 0
                const float Pc = (float) Pc0;
                if (Pc > Ps)
                    fprintf (stderr, "setmixtureerrorsignal(%d,%d,%d): Pc > Ps (%.10f > %.10f)\n", (int) s, (int) t, (int) 0, Pc, Ps);
                const float deltaterm = (iscorrects && Ps > 0.0f) ? (Pc / Ps) : 0.0f;
                err(s,t) = deltaterm - Pc;
            }
        }
#endif

        // redistribute the error signal of first #senones rows to dup rows
        // This yields the correct gradient.
        void scattermixtureerrorsignals (const size_t nummix, size_t ts, size_t te)
        {
            const auto & outlayer = layers.back();
            const size_t numsenones = outlayer->hdim() / nummix;
            if (numsenones * nummix != outlayer->hdim())
                throw std::logic_error ("scattermixtureerrorsignals: inconsistent nummix/udim");

            const rbmstatevectorsrefreading Pu (layerstate[layers.size()], ts, te - ts);    // actual probabilities
            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));  // -> error goes here
            err.lockforreadwrite();

            foreach_column (t, err) for (size_t s = 0; s < numsenones; s++)
            {
                float Ps = Pu(s,t);       // P(s|v) was already summed up in summixturecomponents()
                double Pc0 = Ps;          // P(c0|v) needs to be reconstructed as Ps - sum_c>0 Pc
                for (size_t c = 1; c < nummix; c++)
                {
                    // copy block c=0 to block c and weight by state-conditioned mixture posterior
                    const size_t coffset = c * numsenones;
                    const float Pc = Pu(s + coffset, t);
                    if (Pc > Ps)
                        fprintf (stderr, "setmixtureerrorsignal(%d,%d,%d): Pc > Ps (%.10f > %.10f)\n", (int) s, (int) t, (int) c, Pc, Ps);
                    err(s + coffset, t) = err(s,t) * Pc / Ps;
                    // reconstruct first mixture component
                    Pc0 -= Pc;
                }
                // and block 0
                const size_t coffset = 0;
                const float Pc = (float) Pc0;
                if (Pc > Ps)
                    fprintf (stderr, "setmixtureerrorsignal(%d,%d,%d): Pc > Ps (%.10f > %.10f)\n", (int) s, (int) t, (int) 0, Pc, Ps);
                err(s + coffset, t) = err(s,t) * Pc / Ps;
            }

            err.unlock();
        }

        // drop frames
        // All fields are set up ready for back propagation; that is top-level error and all forward-prop information (layerstate[]).
        // TODO: This function is not clearly defined for ts != 0.
        size_t dropframes (const std::vector<bool> & framestodrop, size_t ts, size_t te)
        {
            size_t framesfortraining = te - ts;
            assert (framestodrop.size() == framesfortraining);
            foreach_index (i, framestodrop) if (framestodrop[i]) framesfortraining--;
            if (framesfortraining == te - ts)    // nothing to do
                return framesfortraining;

            {
                rbmstatevectorsrefwriting keepsamples (keepsampleflags, ts, te-ts);
                for (size_t t = ts; t < te; t++)
                    keepsamples(0,t) = framestodrop[t] ? 0.0f : 1.0f;
                // destructor syncs back keepsampleflags
            }

            rbmstatevectorsref keepsamples (keepsampleflags.stripe (ts, te - ts));
            foreach_index (i, layerstate)
            {
                rbmstatevectorsref h (layerstate[i].stripe (ts, te - ts));
                h.dropframes (keepsamples);
            }

            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));
            err.dropframes (keepsamples);

            return framesfortraining;
        }

        // propagate the error signal of all layers and frames for later use in accumulation
        //  - input = columns of feature vectors
        //  - target output (uids[ts..te-1]) = indices of supervised class ids corresponding to vectors
        // This function operates only on the column range [ts,te) and can be safely called from multiple threads for disjoint column ranges.
        // forwardprop() must be run before entering this function, and error signal must have been set as well.
        void errorbackprop (size_t ts, size_t te)
        {
//#define TIME_BACKPROP
#ifdef TIME_BACKPROP           // do some explicit time measurement, for optimizing model parallelism
            static double totaltime = 0.0;
            static size_t totalframes = 0;
            synchronize();
            auto_timer backproptimer;
#endif
            // back-propagate through the layers and accumulate
            for (size_t i = layers.size(); i > firstbplayer; i--)
            {
                rbmstatevectorsref h  (layerstate[i].stripe (ts, te - ts));

                rbmstatevectorsref eh (errorstate[i].stripe (ts, te - ts));   
                //checknan (eh);
                //checknan (h);
                rbmstatevectorsref ev (errorstate[i-1].stripe (ts, te - ts));   // note: empty for bottom layer

#ifdef LOGINSIGMOID    // calculate the sigmoid value from the log(epison + s(z)) [v-xieche]
                h.getorisigmoid();
#endif
                //printf("######------ BackPropogation, Layer %d ------######\n", i);
                layers[i-1]->backpropagationstats (eh, h, ev);
                //printf("######------ End BackPropogation, Layer %d ------######\n", i);
                //checknan (ev);
                // Note: eh has now been updated and is ready for accumulation.
                // ev has been output as an input for the next stage, where it will,
                // in turn, be updated for accumulation.
            }
#ifdef TIME_BACKPROP
            synchronize();
            totaltime += backproptimer;
            totalframes += te - ts;
            backproptimer.show ("time measurement for backprop");
            fprintf (stderr, "averaged backprop time: %.2f fps (total time %.3f s for %d frames)\n", totalframes / totaltime, totaltime, (int) totalframes);
            fflush (stderr);
#endif
        }

        // compute deltastate vectors for unseen compensation [v-hansu]
        // vector-wise deltah = h .* (1-h) .* [h .* (1-h) .* eps * e * (v' * v + 1) + W' * deltav]
        // matrix-wise deltaH = H .* (1-H) .* [H .* (1-H) .* eps * E * (diag(V' * V) + I) + W' * deltaV]
        // where
        //  deltaW = eps * h .* (1-h) .* e * v'
        //  deltaa = eps * h .* (1-h) .* e
        // and eps chosen to be the same as in the actual model update
        // vnomrs is vectors for storing diag(V' * V), deltav is the input, deltah is the final output.
        // Learning rate and momentum are used to compute the deltaW with the same step size that the final update would use.
        float forwardpropdelta (size_t ts, size_t te, const float learningrateperframe, const double momentumperframe)
        {
            // forwardprop deltas through the layers
            float eps = 0.0f;
            for (size_t i = firstbplayer + 1; i < layers.size(); i++)
            {
                rbmstatevectorsref h (layerstate[i].stripe (ts, te - ts));
                rbmstatevectorsref v (layerstate[i-1].stripe (ts, te - ts));
                rbmstatevectorsref deltah (deltastate[i].stripe (ts, te - ts));
                rbmstatevectorsref deltav (deltastate[i-1].stripe (ts, te - ts));   // note: empty for bottom layer, tested in forwardpropdelta()
                rbmstatevectorsref eh (errorstate[i].stripe (ts, te - ts));
                
                rbmstatevectorsref vnormsbuf (vnormsbufs[i].stripe (ts, te - ts));
                
                eps = layers[i-1]->forwardpropdelta (deltah, deltav, h, v, eh, vnormsbuf, learningrateperframe, momentumperframe);  // propagate deltav through deltaW and delta h
                // TODO: to do this nicely, we'd need to check whether 'eps' has changed--it shouldn't
            }
            return eps;
        }
        // compensate for unseen states w.r.t. top layer weights  [v-hansu]
        //  - input feat, ts, te, dengammas
        //  - output, none
        // We assume the deltav passed to us is the gradient * eps.
        // Correction term for e(s,t):
        // e(s,t) <- e(s,t) - (Ws' deltav(t) / eps) / (||v(t)||^2+1)
        // How we arrived there:
        //  - mb update will affect the top v, thus it will affect unscaled log LL value (w_s'v+a) for unseen states
        //  - criterion: (w_s'v+a) for unseen states shall remain the same before and after this mb update
        //  - method:
        //     - estimate how v changes (-> deltav) as a consequence of updating the model from a *single* frame (deltaW, deltaa)
        //     - correct w_s of unseen states by deltaw_s such that (w_s+deltaw_s)'(v+deltav) == w_s'v
        //     - this linear equation system is underdetermined -> use pseudo-inverse   --this is rather arbitrary; what does it mean?
        //     - represent deltaw_s indirectly in the error signal, so that it later gets picked up through the normal model update
        //     - note: deltaw_s underlies momentum, i.e. compensation will happen multiple times; we assume that deltav does not change too much over mbs
        // TODOs:
        //  - can we possibly accumulate across multiple minibatches, with momentum?
        //  - v+deltav is approximated as v in the correction equation
        template<class UMATRIX>
        void unseenstatecompensation (const size_t ts, const size_t te, const UMATRIX & dengammas, const float eps)
        {
            const size_t numlayers = layers.size();
            const size_t toplayer = numlayers - 1;

            const rbmstatevectorsref v (layerstate[toplayer].stripe (ts, te - ts));
            const rbmstatevectorsref deltav (deltastate[toplayer].stripe (ts, te - ts));
            rbmstatevectorsref deltau (deltastate[toplayer+1].stripe (ts, te - ts));    // used as a buffer (no state)
            rbmstatevectorsref vnormsbuf (vnormsbufs[toplayer].stripe (ts, te - ts));

            // matrix form, pretending all states are unseen (we just ignore the seen ones later, they are independent:
            // E <- E - (W' deltaV / eps) / diag(V' V + 1) .* unseenmask

            // denominator
            assert (vnormsbuf.rows() == 1 && vnormsbuf.cols() == v.cols());
            v.columnnormsquares (vnormsbuf);                // vnormsbuf <- diag(V' * V)    (||v(t)||^2, stored as a vector over t)

            // numerator
            auto * layer = dynamic_cast<perceptron*> (layers[toplayer].get());
            layer->forwardpropwithoutbias (deltav, deltau); // deltau(s,t) = Ws' deltav(t)

            // rest in CPU code
            // E <- E - deltau / eps / vnormsbuf * unseenmask
            rbmstatevectorsref e (errorstate[toplayer+1].stripe (ts, te-ts));
            e.lockforreadwrite();
            deltau.lockforreadwrite();      // deltau is not a const instance, so we can only use lockforreadwrite(), otherwise the operator() will not support
            vnormsbuf.lockforreadwrite();   // the same as about comment

            assert (e.rows() == deltau.rows() && e.cols() == deltau.cols());
            assert (vnormsbuf.rows() == 1 && vnormsbuf.cols() == deltau.cols());

            static size_t countminibatch = 0;       // used for printing stats
            const bool collectstats = (countminibatch <= 100) || (countminibatch % 100) == 0;   // print stats for first 100 minibatch and then every 100 minibatch
            float averagevnorm2p1 = 0.0f;
            float averagedeltaunorm1unseen = 0.0f;
            float averagedeltaunorm1seen = 0.0f;
            float averagedeltavnorm1 = 0.0f;
            float averrornorm1seen = 0.0f;
            float averrornorm1unseen = 0.0f;
            size_t numunseen = 0;
            size_t numseen = 0;
            if (collectstats)
            {
                deltav.lockforreading();
                foreach_coord (i, j, deltav)
                    averagedeltavnorm1 += abs(deltav(i,j));
                averagedeltavnorm1 /= deltav.rows() * deltav.cols() * eps;
                deltav.unlock();
            }

            foreach_column (t, e)
            {
                const float vnorm2p1 = vnormsbuf(0,t) + 1;  // ||v(t)||^2 + 1
                foreach_row (s, e)
                {
                    if (dengammas(s,t) != 0.0f)             // this means the state is seen
                    {
                        if (collectstats)
                        {
                            averagedeltaunorm1seen += abs(deltau(s,t));
                            averrornorm1seen += abs(e(s,t));
                            numseen++;
                        }
                        continue;                           // skip seen states (no correction term)
                    }
                    assert (e(s,t) == 0.0f);
                    e(s,t) -= deltau(s,t) / eps / vnorm2p1; // e(s,t) = e(s,t) - Ws' deltav / eps / (||v(t)||^2 + 1)
                    if (collectstats)
                    {
                        averagedeltaunorm1unseen += abs(deltau(s,t));
                        averrornorm1unseen += abs(e(s,t));
                        numunseen++;
                    }
                }
                if (collectstats)
                    averagevnorm2p1 += vnorm2p1;
            }
            if (collectstats)
            {
                averagevnorm2p1 /= e.cols();
                averagedeltaunorm1seen /= (numseen * eps);
                averagedeltaunorm1unseen /= (numunseen * eps);
                averrornorm1seen /= numseen;
                averrornorm1unseen /= numunseen;
                fprintf (stderr, "unseenstatecompensation: eps %f, averagevnorm2p1 %f, averagedeltaunorm1unseen/eps %f, averagedeltaunorm1seen/eps %f, averagedeltavnorm1/eps %f, averrornorm1seen %f, averrornorm1unseen %f\n", 
                                 eps, averagevnorm2p1, averagedeltaunorm1unseen, averagedeltaunorm1seen, averagedeltavnorm1, averrornorm1seen, averrornorm1unseen);
            }
            vnormsbuf.unlock();
            deltau.unlock();
            e.unlock();
            countminibatch++;
        }
    };

    // -----------------------------------------------------------------------
    // class trainer --for global model update
    // TODO: no longer needed to separate this from 'accumulator'
    // -----------------------------------------------------------------------

    // The 'trainer' is created per epoch, while multiple 'accumulator' are kept NUMA-locally.
    class trainer : public accumulator
    {
    public:
        // This creates vectors to store the shared intermediate state activations (incl. v and u).
        trainer (const model & M, size_t mbsize, bool bpmode, size_t finetunedlayers) : accumulator (M, mbsize, bpmode, finetunedlayers) { }

        // add virtual destructor in order to make trainer a polymoprphic type
        virtual ~trainer() {}

        // update the model based on layerstate[][] and errorstate[][]
        void pretrainingmodelupdate (size_t ts, size_t te, float learningratepersample, double momentumpersample)
        {
            size_t toplayer = layers.size() -1; // pre-training applies to the top layer only
            rbmstatevectorsref v (layerstate[toplayer].stripe (ts, te - ts));
            rbmstatevectorsref h (layerstate[toplayer+1].stripe (ts, te - ts));
            rbmstatevectorsref v1ref (v1.stripe (ts, te - ts));
            rbmstatevectorsref h1ref (h1.stripe (ts, te - ts));
            layers[toplayer]->pretrainingmodelupdate (v, h, v1ref, h1ref, learningratepersample, momentumpersample);
        }

        void aggregatedistributedgradient (const size_t begin, const size_t end/*layer range*/, mpiaggregator & mpiaggregator,
                                           const std::vector<modelupdateinfo> & bpinfos, double momentumpersample, float learningratepersample)
        {
            // note: begin/end is not fully supported by DSGD in that it will always exchange the full-size buffer; probably OK since begin/end here is used for special purposes only

            // for model averaging, we'd better not be quantizing (bits=32) and not use double buffering; then we should just use plain MPIAllReduce() here
            if (mpiaggregator.canusempiallreduce() && !bpinfos[0].distributefixedcost)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->mpiallreducegradient (bpinfos[i]);
                return;
            }

            // aggregator needs the following functions of 'stripe index' (which operate on whole stack) as lambdas:
            //  - allocatetransferbuffer                // allocate a suitable data transfer buffer of a given size
            //  - quantizeandfetchsubbatchstripe        // from source, typ. GPU
            //  - syncfetchsubbatchstripe               // wait for it to complete
            //  - unquantizeandaggregatestripe          // CPU (synchronous)
            //  - quantizeandassignaggregatedstripe     // CPU (synchronous)
            //  - assignaggregatedstripe
            //  - syncassignaggregatedstripeandunquantize
            // Note some will be passed to a background thread, so they cannot capture any locals (e.g. 'begin', 'end') by reference.

            // allocator
            auto allocatetransferbuffer = [this] (size_t stripe, size_t size)
            {
                // since the buffer is shared across layers, we ask a random layer to do this for us
                return layers[0]->allocatetransferbuffer (stripe, size);
            };

            // this will be called from the main thread
            auto quantizeandfetchsubbatchstripe = [this, begin, end] (size_t stripe, char * bufferbegin, size_t buffersize, size_t & submbframes)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->quantizeandfetchsubbatchstripe (stripe, bufferbegin, buffersize, submbframes);
            };

            // wait for our stripe to be completely quantized into the CPU-side buffer
            // this will be called from a bg thread
            auto syncfetchsubbatchstripe = [this, begin, end] (size_t stripe)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->syncfetchsubbatchstripe (stripe);
            };

            // aggregate a (received) sub-batch stripe into the aggregation accumulator
            // called for every peer node, as well as for our own stripe
            // upon last call, this might do additional fixed-cost operations such as AdaGrad and momentum (on owned stripe only)
            // 'stripe' is always the same and refers to the stripe we own
            // this will be called from a bg thread
            auto unquantizeandaggregatestripe = [this, begin, end, &bpinfos, momentumpersample, learningratepersample] (size_t ourstripe, size_t kfrom, const char * bufferbegin, size_t buffersize, bool isfirst,
                                                                                                                        bool islast, size_t mbframes)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->unquantizeandaggregatestripe (ourstripe, kfrom, bufferbegin, buffersize, isfirst, islast, mbframes, bpinfos[i], momentumpersample, learningratepersample);
            };

            // quantize an aggregate stripe from the aggregation accumulator, and also move it back already
            // called once, in prep for sharing it out with others
            // 'stripe' is the stripe we own
            // this will be called from a bg thread (but executes synchronously there)
            auto quantizeandassignaggregatedstripe = [this, begin, end] (size_t stripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->quantizeandassignaggregatedstripe (stripe, bufferbegin, buffersize, reuserangescaled);
            };

            // move back an aggregated stripe (which is in quantized form)
            // this will be called from a bg thread and only kicks off the CPU-to-GPU transfer
            auto assignaggregatedstripe = [this, begin, end] (size_t stripe, const char * bufferbegin, size_t buffersize)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->assignaggregatedstripe (stripe, bufferbegin, buffersize);
            };

            // note: due to multi-threading/double-buffering, this runs right after quantization on main thread
            auto syncassignaggregatedstripeandunquantize = [this, begin, end, &bpinfos] (size_t stripe, const char * bufferbegin, size_t buffersize, size_t aggmbframes)
            {
                for (size_t i = begin; i < end; i++)
                    layers[i]->syncassignaggregatedstripeandunquantize (stripe, bufferbegin, buffersize, aggmbframes, bpinfos[i]);
            };

            // kick off exchange process
            // When running double-buffered then this runs as a background thread that exchanges the current gradient
            // while immediately returning the previous gradient (one minibatch delay).
//synchronize();  // we are timing and want to subtract this function's time
BEGINTIME("mpiaggregator.aggregate");
            mpiaggregator.aggregate (allocatetransferbuffer,
                                     quantizeandfetchsubbatchstripe, syncfetchsubbatchstripe, unquantizeandaggregatestripe,
                                     quantizeandassignaggregatedstripe, assignaggregatedstripe, syncassignaggregatedstripeandunquantize);
ENDTIME();
        }

        size_t mpimaframes;     // we got this many frames locally; do local loop update only until we hit mpimasize, only then exchange (if enabled)

        void resetmpiaggregation() { mpimaframes = 0; }    // call this before entering

        // update the model based on layerstate[][] and errorstate[][]
        void backpropagationmodelupdate (size_t ts, size_t te, float learningratepersample, double momentumpersample, bool deferupdate,
                                         size_t restricttosinglelayer/*or SIZE_MAX*/, const std::vector<modelupdateinfo> & bpinfos)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            if (restricttosinglelayer != SIZE_MAX)
            {
                if (begin < restricttosinglelayer)
                    begin = restricttosinglelayer;
                if (end > restricttosinglelayer + 1)
                    end = restricttosinglelayer + 1;
                if (end <= begin)
                {
                    fprintf (stderr, "begin %d, end %d\n", begin, end);
                    throw std::runtime_error ("backpropagationmodelupdate: restricttosinglelayer conflicts with firstbplayer or is out of range");
                }
            }

            // set all layers' raw gradients
            // Note that some layer kinds do this in backpropagationmodelupdate3(), while backpropagationmodelupdate1() is a dummy.
            // This must only be implemented in backpropagationmodelupdate1() where we implement data parallelism.
            for (size_t i = begin; i < end; i++)
            {
                // update parameters of model i based on BP step
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe   (ts, te - ts));
                //checknan (err); checknan (act);
                layers[i]->backpropagationmodelupdate1 (err, act, bpinfos[i]);
            }

            // cross-layer AdaGrad state update
            if (bpinfos[0].adagradstate && bpinfos[0].enableadagrad && bpinfos[0].adagradwhere == modelupdateinfo::onpartialsubgradient)
                bpinfos[0].adagradstate->finishaccumulation();

            // if deferred then that's it
            if (deferupdate)
                return;

            // model averaging, Kaldi style (potentially also useful for ADMM):
            //  - back-up model                                                     --in update2()
            //  - repeat:
            //     - compute sub-gradient
            //     - model update (with local AdaGrad and local momentum)
            //  - restore model
            //  - sub-gradient <- new model - backed-up model
            //  - data exchange                                                     --as usual
            //  - model update (without AdaGrad or momentum)                        --in update3() as a mode
            // Kaldi [http://www.danielpovey.com/files/2014_icassp_dnn.pdf]:
            //  "After processing a specified amount of data (typically 300000 samples per machine, which can take 10
            //  20 minutes when using CPUs), each machine writes its model to disk and we average the model parameters."
            //  "We have found that the lack of convexity of the neural network objective function is simply not an issue. However,
            //  for this method to make fast progress we must use a higher learning rate than we would use if we were training on a single machine, and
            //  this can sometimes lead to parameter divergence or saturation of the sigmoidal units. Our parallel model training method tends to give a
            //  small degradation in WER when compared with training on a single machine, but we do it anyway because it is much faster"
            // Note: 300000 = 50 minutes of data; e.g. mbsize=4096, bpinfo.mpimasize = 300

            const bool dompima = (bpinfos[0].mpimasize != 0);
            bool mpimaisfirst = false;
            bool mpimaislast = false;
            if (dompima)        // local loop state control
            {
                mpimaisfirst = (mpimaframes == 0);                      // true -> reset local-loop accumulator
                //if (bpinfos[0].mpimasize < 500)
                mpimaframes++;      // < 500: interpret as number of minibatches rather than frames
                //else
                //    mpimaframes += te - ts;
                mpimaislast = (mpimaframes >= bpinfos[0].mpimasize);    // true -> move local-loop accumulator to raw gradient
                if (mpimaislast)
                    mpimaframes = 0;    // get ready for next call
            }

            // give it a chance to post-process after 'deferupdate' but before MPI exchange
            // This is intended for AdaGrad scaling of the raw gradient.
            // This is actually not used (didn't work), we can remove it again.
            // This also handles the local loop for data parallelism.
            for (size_t i = begin; i < end; i++)
                layers[i]->backpropagationmodelupdate2 (bpinfos[i], mpimaisfirst, mpimaislast, learningratepersample, momentumpersample);

            // cross-layer AdaGrad state update
            if (bpinfos[0].adagradstate && bpinfos[0].enableadagrad && bpinfos[0].adagradwhere == modelupdateinfo::onsubgradient)
                bpinfos[0].adagradstate->finishaccumulation();

            // if we are just doing local loop now then we are done
            if (dompima && !mpimaislast)
                return;     // we just did a local update, done

#undef TIME_MODELUPDATE    // this measures all steps that do not depend on MB size but on model dimension ("fixed" cost for data parallelism)
#ifdef TIME_MODELUPDATE     // temporary for Frank's parallelization experiments; remove when done
            static double totaltime = 0.0;
            static size_t totalframes = 0;
            static size_t totalmbs = 0;
            static size_t measurethiscounter = 0;
            bool measurethis = (measurethiscounter++ % 20 == 19);   // we do this every 20th MB, and leave it enabled always
            if (measurethis)
                synchronize();
            auto_timer updatetimer;
#endif

            // in case of data parallelism, exchange the gradients (replace raw sub-minibatch gradients with raw whole-minibatch gradients)
            // Note: in 'distributefixedcost' mode, this already applies AdaGrad and momentum, interwoven with data exchange, and updates the model, too.
            if (bpinfos[0].mpiaggregator)
                aggregatedistributedgradient (begin, end, *bpinfos[0].mpiaggregator, bpinfos, dompima ? 0.0 : momentumpersample, dompima ? 1.0f : learningratepersample);

            // update models with gradient smoothing (momentum)
            for (size_t i = begin; i < end; i++)
            {
                // update parameters of model i based on BP step
                // Note: 'err' and 'act' are not really used by the new MPI-compatible version; only by the old code.
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe   (ts, te - ts));
                layers[i]->backpropagationmodelupdate3 (err, act, dompima ? 1.0f : learningratepersample, dompima ? 0.0 : momentumpersample, bpinfos[i]);
            }

            // cross-layer AdaGrad state update
            // TODO: in double-buffering and 'distributefixedcost' mode, this is not really supported
            if (bpinfos[0].adagradstate && bpinfos[0].enableadagrad && bpinfos[0].adagradwhere >= modelupdateinfo::onrawgradient)
                bpinfos[0].adagradstate->finishaccumulation();

#ifdef TIME_MODELUPDATE
            if (measurethis)
            {
                synchronize();
                double thistime = updatetimer;
                totaltime += thistime;
                totalmbs++;
                totalframes += te - ts;
                //updatetimer.show ("time measurement for model update");
                fprintf (stderr, "backpropagationmodelupdate [timing]: averaged model update time (=fixed cost): %.1f ms/mb, av: %.1f ms/mb (over total time %.3f s, %d frames)\n",
                         thistime * 1000.0f,
                         totaltime / totalmbs * 1000.0f,
                         totaltime, (int) totalframes);
                //fflush (stderr);
            }
#endif
        }

        // update gradient based on layerstate[][] and errorstate[][]
        void collectgradient (size_t ts, size_t te, bool isfirstbatch, bool usedoubleaccumulator)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
            {
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe (ts, te - ts));
                layers[i]->collectgradient (act, err, isfirstbatch, usedoubleaccumulator); // computes minibatch gradient from activation and error signal and adds it to the accumulator
            }
        }

        // TODO move this to entercomputation ? but need to pass one more argument then
        // allocate double precision accumulators
        void allocateaccumulators(bool usecgpreconditioning)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
                layers[i]->allocateaccumulators(usecgpreconditioning);
        }

        // set values of (float) gradient to double precision accumulator values
        void settoaccumulator(bool usecgpreconditioning)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
                layers[i]->settoaccumulator(usecgpreconditioning);
        }

    };

    // trainer for Hessian free optimization (see Martens, ICML 2010)
    // note: instead of the Hessian, we use the Gauss-Newton matrix (as Martens)
    class hessianfreetrainer : public trainer 
    {
    
    protected:
        std::vector<rbmstatevectors> forwardstatistics;    // forward statistics of Hessian vector product
        std::vector<rbmstatevectors> layerstatesquared;    // squared layerstates, needed for preconditioning (TODO pc statistics could also be computed directly, thus saving some memory)
        std::vector<rbmstatevectors> errorstatesquared;    // squared errorstates, needed for preconditioning (TODO pc statistics could also be computed directly, thus saving some memory)
        bool initnextcgfromzero;

    public:
        hessianfreetrainer (const model & M, size_t mbsize, bool istoplayer, size_t finetunedlayers, size_t nofbacktrackingmodels, bool needssquaredstatistics) : 
          trainer(M, mbsize, istoplayer, finetunedlayers) , initnextcgfromzero(true)
        {
            // allocate memory for hessianfree optimizer statistics
            alloclayerstate (forwardstatistics, mbsize);
            if (needssquaredstatistics)
            {
                alloclayerstate (layerstatesquared, mbsize);
                alloclayerstate (errorstatesquared, mbsize);
            }
            forwardstatistics[0].resize (0, mbsize);
            for (size_t n = 0; n < layers.size(); n++)
                layers[n]->inithessianfree(nofbacktrackingmodels);
        }

        // update sum of squared gradients 
        void collectsquaredgradient(size_t ts, size_t te, bool isfirstbatch, bool usedoubleaccumulator)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
            {
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe (ts, te - ts));
                rbmstatevectorsref errsquared (errorstatesquared[i+1].stripe (ts, te - ts));
                rbmstatevectorsref actsquared (layerstatesquared[i].stripe (ts, te - ts));

                layers[i]->collectsquaredgradient(act, err, actsquared, errsquared, isfirstbatch, usedoubleaccumulator); 
            }
        }


        // forward propagates Hessian vector product statistics
        void forwardprophessianvectorproduct (size_t ts, size_t te)
        {
            for (size_t i = 0; i < layers.size(); i++)
            {
                rbmstatevectorsref layerin  (layerstate[i].stripe   (ts, te - ts));
                rbmstatevectorsref layerout (layerstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref forwardstatisticsin (forwardstatistics[i].stripe (ts, te - ts));
                rbmstatevectorsref forwardstatisticsout (forwardstatistics[i+1].stripe (ts, te - ts));
                
                bool zeroforwardstatisticsin = i == 0; // statistics of zeroth layer are always zero
                layers[i]->forwardprophessianvectorproduct(layerin, layerout, 
                    forwardstatisticsin, forwardstatisticsout, zeroforwardstatisticsin); 
            }
        }

        // sets error signal for computation of Hessian vector product
        // refers to Hessian of cross entropy training
        void sethessianvectorsignal (size_t ts, size_t te)
        {
            const auto & u = layerstate[layers.size()];
            const rbmstatevectorsref Pu  (u.stripe (ts, te - ts));
            const auto & f = forwardstatistics[layers.size()];
            const rbmstatevectorsref fstripe(f.stripe(ts, te - ts));
            
            rbmstatevectorsref signal (errorstate[layers.size()].stripe (ts, te - ts));
            signal.sethessianvectorsignal(Pu, fstripe);
        }

        // CG methods

        // initialize conjugate gradient statistics
        // begin with cg iterate zero
        void initcgfromzero(bool usepreconditioning, float nobservations, float lambda, float alpha)
        {
            for (size_t i = 0; i < layers.size(); i++)
                layers[i]->initcgfromzero(usepreconditioning, nobservations, lambda, alpha);
        }

        // initialize conjugate gradient statistics
        // begin with cg iterate different from zero
        // requires run of hessianvectorproduct before this method is called
        void initcg(bool usepreconditioning, float nobservations, float lambda, float alpha)
        {
            for (size_t i = 0; i < layers.size(); i++)
                layers[i]->initcg(usepreconditioning, nobservations, lambda, alpha);
        }
          
        // update hessian vector product based on layerstate[][] and errorstate[][]
        void collecthessianvectorproduct (size_t ts, size_t te, bool isfirstbatch, size_t nsecondorderframes)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
            {
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe (ts, te - ts));
                layers[i]->collecthessianvectorproduct (act, err, isfirstbatch, nsecondorderframes); // computes minibatch hessian vector product from activation and error signal and adds it to the accumulator
            }
        }

        // returns weighted inner product searchdirection' * Hessian * searchdirection
        // value is computed as searchdirection' * hessianvectorproduct
        // requires tha hessianvectorproduct is already computed
        float calculatecgcurvatureproduct() const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float cgcurvatureproduct = 0.0f;
            for (size_t i = begin; i < end; i++)
                cgcurvatureproduct += layers[i]->calculatecgcurvatureproduct();
            return cgcurvatureproduct;
        }

        // returns squared residual norm
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatecgresidualnorm(bool weighted) const
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            float residualnorm = 0.0f;
            for (size_t i = begin; i < end; i++)
                residualnorm += layers[i]->calculatecgresidualnorm(weighted);
            return residualnorm;
        }

        // returns squared residual norm, norm implied by preconditioner
        // calculation via cgresidual' * pcgresidual
        float calculatepcgresidualnorm() const
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            float residualnorm = 0.0f;
            for (size_t i = begin; i < end; i++)
                residualnorm += layers[i]->calculatepcgresidualnorm();
            return residualnorm;
        }

        // returns squared norm of cgiterate
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatesquaredcgiteratenorm(bool weighted) const
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatesquaredcgiteratenorm(weighted);
            return result;
        }

        // returns squared norm of cgsearchdirection
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatesquaredcgsearchdirectionnorm(bool weighted) const
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatesquaredcgsearchdirectionnorm(weighted);
            return result;
        }

        // returns squared norm of model
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatesquaredparameternorm(bool weighted) const
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatesquaredparameternorm(weighted);
            return result;
        }

        // updates cg iterate 
        // cgiterate += stepsize * cgsearchdirection
        void updatecgiterate(float stepsize)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->updatecgiterate(stepsize);
        }

        // updates cg residual 
        // cgresiudal += stepsize * hessianvectorproduct
        void updatecgresidual(float stepsize)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
                layers[i]->updatecgresidual(stepsize);
        }
        
        // solves M pcgresidual = cgresidual
        void solveforpcgresidual()
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->solveforpcgresidual();
        }

        // updates cgsearchdirection
        // cgsearchdirection *= stepsize
        // cgsearchdirection -= cgresidual
        void updatecgsearchdirection(float stepsize)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->updatecgsearchdirection(stepsize);
        }

        // updates cgsearchdirection
        // cgsearchdirection *= stepsize
        // cgsearchdirection -= pcgresidual
        void updatepcgsearchdirection(float stepsize)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
                layers[i]->updatepcgsearchdirection(stepsize);
        }

        // divides gradient by number of observations
        void normalizegradient(size_t nobservations) {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            
            for (size_t i = begin; i < end; i++)
                layers[i]->normalizegradient(nobservations);
        }

        // returns squared gradient norm
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatesquaredgradientnorm(bool weighted) const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
                
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatesquaredgradientnorm(weighted);
            return result;
        }

        // returns dot product of cgresidual and cgsearchdirection
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatecgresidualcgsearchdirectionproduct(bool weighted) const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatecgresidualcgsearchdirectionproduct(weighted);
            return result;
        }
        
        // returns dot product of cgresidual and cgiterate
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatecgresidualcgiterateproduct(bool weighted) const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
                
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatecgresidualcgiterateproduct(weighted);
            return result;
        }

        // returns dot product of gradient and cgiterate
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculategradientcgiterateproduct(bool weighted) const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculategradientcgiterateproduct(weighted);
            return result;
        }

        // returns dot product of cg iterate and cg search direction
        // if weighted: preconditioner-norm
        // else: Euclidean norm
        float calculatecgiteratecgsearchdirectionproduct(bool weighted) const 
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            float result = 0.0f;
            for (size_t i = begin; i < end; i++)
                result += layers[i]->calculatecgiteratecgsearchdirectionproduct(weighted);
            return result;
        }

        // scales cg iterate by scalingfactor
        void scalecgiterate(float scalingfactor)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->scalecgiterate(scalingfactor);
        }

        // sets cg searchdirections to Ws and as
        // only required for unit tests
        void setcgsearchdirections(std::vector<rbmmodelmatrix*> &Ws, std::vector<rbmmodelmatrix*> &as){
            assert (Ws.size() == layers.size());
            assert (as.size() == layers.size());
            for (size_t i = 0; i < layers.size(); i++)
                layers[i]->setcgsearchdirection(*(Ws[i]), *(as[i]));
        }

        // adds lambda * cgsearchdirection to Hessian vector product
        // corresponds to damping of Hessian by lambda * identity
        void adddampingterm(float lambda)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->adddampingterm(lambda);
        }

        // copies cgiterate to cgintermediateresults
        // required for CG backtracking and linsearch (see Martens)
        void storecgiterate(size_t position)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t n = begin; n < end; n++)
                layers[n]->storecgiterate(position);
        }

        // adds stepsize * intermediateresult to model
        // required for CG backtracking and linsearch (see Martens)
        void settointermediateresult(size_t position, float stepsize)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t n = begin; n < end; n++)
                layers[n]->settointermediateresult(position, stepsize);
        }

        // copies current model to backupmodel
        // backup is required when no cg iterate yields an improvement
        void backupmodel()
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t n = begin; n < end; n++)
                layers[n]->backupmodel();
        }

        // restores model from backupmodel
        // backup is required when no cg iterate yields an improvement
        void restoremodel()
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t n = begin; n < end; n++)
                layers[n]->restoremodel();
        }

        // resets all statistics to zero
        // except of the initial cg iterate, which is set to 
        // cgiterate = cginitdecayingfactor * cgintermediateresults[cgiter]
        void finalizecg(int cgiter, float cginitdecayingfactor)
        {
            initnextcgfromzero = cginitdecayingfactor == 0.0f;
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t n = begin; n < end; n++)
                layers[n]->finalizecg(cgiter, cginitdecayingfactor);
        }

        // returns whether we need to init CG from zero in the next step
        bool shallinitnextcgfromzero() 
        {
            return initnextcgfromzero; 
        }

        // sets dummy hessian vector product for debugging conjugate gradient
        void setdummyhessianvectorproduct()
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->setdummyhessianvectorproduct((float) (i+1));
        }

        // sets dummy gradient product for debugging conjugate gradient
        void setdummygradient()
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            for (size_t i = begin; i < end; i++)
                layers[i]->setdummygradient();
        }

    };

    // trainer for asynchronous SGD 
    class async_trainer: public trainer 
    {
    
    protected:
        bool gradients_pushed;    // for worker
        std::vector<rbmstatevectors> layerstatesquared;    // squared layerstates, needed for preconditioning (TODO pc statistics could also be computed directly, thus saving some memory)
        std::vector<rbmstatevectors> errorstatesquared;    // squared errorstates, needed for preconditioning (TODO pc statistics could also be computed directly, thus saving some memory)
		std::string contributing_worker_gradient_fn;                        // the gradient file from the contributing worker
		std::string contributing_worker_model_fn;                           // the model file to the contributing worker

    public:
        async_trainer (const model & M, size_t mbsize, bool istoplayer, size_t finetunedlayers) : 
          trainer(M, mbsize, istoplayer, finetunedlayers) 
        {
            contributing_worker_gradient_fn = "";
            contributing_worker_model_fn = "";
            gradients_pushed = false; 
          }

          bool is_gradient_pushed() { return gradients_pushed; }
          void reset_gradient_pushed() { gradients_pushed = false; }

          void set_worker_gradient_fn(std::string str) { contributing_worker_gradient_fn = str; }
          void set_worker_model_fn() 
          {
              assert(!contributing_worker_gradient_fn.empty());
              contributing_worker_model_fn = 	contributing_worker_gradient_fn + ".mdl";
          }
          std::string get_worker_gradient_fn()
          {
              return contributing_worker_gradient_fn;
          }
          std::string get_worker_model_fn()
          {
              return contributing_worker_model_fn;
          }

          void push_gradients(size_t restricttosinglelayer/*or SIZE_MAX*/, FILE *f = NULL)
          {
              size_t begin = firstbplayer;
              size_t end = layers.size();
              char   ctmp[64];

              if (f == NULL)
                  return;

              if (restricttosinglelayer != SIZE_MAX)
              {
                  if (begin < restricttosinglelayer)
                      begin = restricttosinglelayer;
                  if (end > restricttosinglelayer + 1)
                      end = restricttosinglelayer + 1;
                  if (end <= begin)
                  {
                      fprintf (stderr, "begin %d, end %d\n", begin, end);
                      throw std::runtime_error ("backpropagationmodelupdate: restricttosinglelayer conflicts with firstbplayer or is out of range");
                  }
              }

              try{
                  // apply to entire network top-down
                  for (size_t i = begin; i < end; i++)
                  {
                      rbmbase & layer = dynamic_cast<rbmbase &>(*layers[i]);
                      rmbmodelmatrix& dW = layer.getdW(); 
                      rmbmodelmatrix& da = layer.getda();

                      sprintf_s(ctmp, "dW%d\n", i);					
                      dW.exitcomputation();
                      dW.write(f, ctmp);
                      dW.entercomputation();

                      sprintf_s(ctmp, "da%d\n", i);
                      da.exitcomputation();
                      da.write(f, ctmp);
                      da.entercomputation();
                  }

                  gradients_pushed = true;
              }
              catch(...)
              {
                  fprintf (stderr, "error in push_gradients\n");
                  throw std::runtime_error ("push_gradients: error in writing to file"); 
              }
          }


          void push_gradients(size_t restricttosinglelayer/*or SIZE_MAX*/, const HANDLE f)
          {
              size_t begin = firstbplayer;
              size_t end = layers.size();
              char   ctmp[64];

              if (restricttosinglelayer != SIZE_MAX)
              {
                  if (begin < restricttosinglelayer)
                      begin = restricttosinglelayer;
                  if (end > restricttosinglelayer + 1)
                      end = restricttosinglelayer + 1;
                  if (end <= begin)
                  {
                      fprintf (stderr, "begin %d, end %d\n", begin, end);
                      throw std::runtime_error ("backpropagationmodelupdate: restricttosinglelayer conflicts with firstbplayer or is out of range");
                  }
              }

              try
              {
                  // apply to entire network top-down
                  for (size_t i = begin; i < end; i++)
                  {
                      rbmbase & layer = dynamic_cast<rbmbase &>(*layers[i]);
                      rmbmodelmatrix& dW = layer.getdW(); 
                      rmbmodelmatrix& da = layer.getda();

                      sprintf_s(ctmp, "dW%d\n", i);					
                      dW.exitcomputation();
                      dW.write(f, ctmp);
                      dW.entercomputation();

                      sprintf_s(ctmp, "da%d\n", i);
                      da.exitcomputation();
                      da.write(f, ctmp);
                      da.entercomputation();
                  }

                  gradients_pushed = true;
              }
              catch(...)
              {
                  fprintf (stderr, "error in push_gradients\n");
                  throw std::runtime_error ("push_gradients: error in writing to file"); 
              }
          }

          void read_gradients(HANDLE f = INVALID_HANDLE_VALUE)
          {
              size_t begin = firstbplayer;
              size_t end = layers.size();
              char   ctmp[64];

              if (f == INVALID_HANDLE_VALUE)
                  return;

              try
              {
                  // apply to entire network top-down
                  for (size_t i = begin; i < end; i++)
                  {
                      rmbmodelmatrix dW ;
                      rmbmodelmatrix da ;

                      sprintf_s(ctmp, "dW%d\n", i); 				
                      dW.read(f, ctmp);

                      sprintf_s(ctmp, "da%d\n", i);
                      da.read(f, ctmp);

                      layers[i]->setdeltas(dW, da); 
                  }
              }
              catch(...)
              {
                  cout << "error in push_gradients" << std::endl; 
                  throw std::runtime_error ("push_gradients: error in writing to file"); 
              }
          }

          void read_gradients(FILE *f = NULL)
          {
              size_t begin = firstbplayer;
              size_t end = layers.size();
              char   ctmp[64];

              if (f == NULL)
                  return;

              try
              {
                  // apply to entire network top-down
                  for (size_t i = begin; i < end; i++)
                  {
                      rmbmodelmatrix dW ;
                      rmbmodelmatrix da ;

                      sprintf_s(ctmp, "dW%d\n", i);					
                      dW.read(f, ctmp);

                      sprintf_s(ctmp, "da%d\n", i);
                      da.read(f, ctmp);

                      layers[i]->setdeltas(dW, da); 
                  }
              }
              catch(...)
              {
                  fprintf (stderr, "error in push_gradients\n");
                  throw std::runtime_error ("push_gradients: error in writing to file"); 
              }

          }

    };
};

// ===========================================================================
// interface to abstract trainer implementations (for now, only used with compact trainer and pipeline trainer)
// ===========================================================================

class itrainer 
{
public:             // define the interface functions for plain training and pipeline training.[v-xieche]
    virtual void entercomputation (msra::dbn::model& model, int entercompuationvalue) = 0;
    virtual void exitcomputation (msra::dbn::model &model) = 0;
    virtual void inittraining (size_t actualmbsize) = 0;        // initiate buffer will be used in the following training.
    virtual void getminibatch (std::vector<size_t> & uids, bool &flag_remove_pipeline, size_t & restrictupdatelayer) = 0;
    virtual void processminibatch (const msra::dbn::matrixstripe &v, const std::vector<size_t> & uids, size_t ts, size_t te, size_t startlayertoeval, size_t restrictupdatelayer,
                                   size_t numlayerstoeval, bool prenonlinearity /* false */, msra::dbn::model::evaluator * prefevaluator, const float alpha,
                                   float learningratepersample, double momentumpersample, 
                                   float sparsethreshold, size_t restricttosinglelayer/*or SIZE_MAX*/,
                                   const msra::dbn::regularizationtype regtype, const float regparam, unique_ptr<msra::dbn::model> & prefmodel, std::vector<double> &Pusum, size_t totalPucount) = 0;
    virtual void getmbstats (std::vector<double> &Pusum, const std::vector<size_t> &uids, size_t &totalPucount, size_t actualmbsize, double &logpsum, double &fcorsum, size_t &logpframes) = 0;
    virtual ~itrainer() { }
};

};};
