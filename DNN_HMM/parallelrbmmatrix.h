// parallelrbmmatrix.h -- parallelized implementation of matrix functions required for RBMs
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/parallelrbmmatrix.h $
// 
// 687   7/04/14 15:36 Fseide
// (added a comment on complexity of do_svd())
// 
// 686   7/04/14 14:41 Fseide
// (comment)
// 
// 685   7/04/14 14:39 Fseide
// sanitized do_svd() (comments, doc on dimensions, more meaningful
// parameter names)
// 
// 684   7/04/14 14:24 Fseide
// more clean-up and commenting of svd(), starting to make sense now
// 
// 683   7/04/14 13:53 Fseide
// SVD: moved 'w' buffer further up, and added more documentation on what
// the dimensions mean
// 
// 682   7/04/14 11:26 Fseide
// setfrom() actually can take a sub-dimension only--gngngn
// 
// 681   7/04/14 11:12 Fseide
// (comments)
// 
// 680   7/04/14 10:47 Fseide
// renamed setvalue (vector of vector) to setfrom() and added dimension
// checks (to be tested) and const correctness
// 
// 679   7/03/14 16:34 Fseide
// (editorial)
// 
// 678   7/01/14 17:35 Fseide
// (comment)
// 
// 677   7/01/14 16:34 Fseide
// (added debug code, disabled)
// 
// 676   6/26/14 5:25p Ganl
// fixed a warning
// 
// 675   6/25/14 18:39 Fseide
// ~mpihelper() no longer resets the MPI sub-communicator, as this can
// cause hangs during crashes, preventing it from restarting properly
// 
// 674   6/16/14 16:42 Fseide
// setutterancedata() no longer syncs after transferring;
// added some log messages to track down sync/cuda-alloc points;
// pred(,) renamed to scaledLLs(,) and now declared outside of MB loop to
// reduce reallocs
// 
// 673   6/13/14 5:29p Fseide
// changed log message in entermpiaggregation() to use new #defines
// 
// 672   6/12/14 6:25p Fseide
// commenting and some cleanup of SVD code
// 
// 671   6/12/14 2:11p Fseide
// entermpiaggregation() now prints a warning if we compile with using
// zero-mean in quantization
// 
// 670   6/11/14 4:20p Fseide
// further clean-up of the SVD mess, to be tested someday
// 
// 669   6/10/14 11:37 Fseide
// softplus() implemented for CPU mode, so that we can decode with
// Latgen/HAPI
// 
// 668   6/06/14 4:48p Fseide
// softplus implemented
// 
// 667   6/06/14 4:04p Fseide
// towards implementing the softplus non-linearity
// 
// 666   6/06/14 10:55a Fseide
// new method varnorm()
// 
// 665   6/03/14 9:39a Fseide
// bug fix in lazymakebuffer(): it scheduled its setzero() on the wrong
// CUDA stream, solved by synchronizing afterwards (still to be tested)
// 
// 664   5/30/14 5:56p Fseide
// changed a log message
// 
// 663   5/30/14 5:46p Fseide
// (uncommented a log message temporarily)
// 
// 662   5/28/14 10:03a Fseide
// very first epoch now never does MPI/data parallelism
// 
// 661   5/22/14 3:22p Fseide
// new method sgemm_mtm() for MVN-SGD
// 
// 660   5/20/14 3:24p Fseide
// seterrorsignalhsmoothing() now passes frame dropping thresh to error
// calculation funtion
// 
// 659   5/16/14 5:38p Fseide
// scaleandaddmatprod() and dot_mtm() can now take model matrices as well;
// some hacks to temporarily disable striping-mode checks for testing
// MVN-SGD (we clean this up once we find that it actually works)
// 
// 658   5/16/14 2:52p Fseide
// (removed a few glimpse() calls for MVN)
// 
// 657   5/15/14 7:24p Fseide
// variance normalization now implemented in class mvn;
// new method meanvaracc();
// meanvarnorm() now takes parameter whether to apply the mean or not
// 
// 656   5/15/14 5:29p Fseide
// added MPI to class mvn;
// glimpse() now works outside entercomputation()
// 
// 655   5/15/14 3:13p Fseide
// added a glimpse() call to meanvarnorm() in _DEBUG mode, looks right
// 
// 654   5/15/14 14:17 Fseide
// changed gems() to use applyonsubstreams(), to ensure compatibility with
// model parallelism;
// completed class mvn (not tested yet)
// 
// 653   5/14/14 15:19 Fseide
// added new method meanvarnorm(), not completed yet
// 
// 652   5/14/14 14:40 Fseide
// (spacing)
// 
// 651   5/12/14 4:26p Fseide
// setzero() now robust against being called on empty matrices that have
// not entered computation
// 
// 650   5/12/14 14:09 Fseide
// cleanup of forwardprop() w.r.t. dropout pre-scaling (no longer done
// inside forwardprop());
// comments
// 
// 649   5/12/14 13:53 Fseide
// removed dropout pre-scaling everywhere incl. CUDA side
// 
// 648   5/05/14 4:30p F-gli
// fixed bug of wrong colmax in softmax()
// 
// 647   4/30/14 11:17a Fseide
// bug fix: gradientfixups() now takes an optional qstripe to make it run
// on the associated CUDA stream, same as unquantization (without, it may
// run on the wrong stream and thus induce some non-deterministism);
// titied up some logging
// 
// 646   4/30/14 10:31a Fseide
// learning-rate scaling now works when done in gradientfixups(), but
// currently suboptimally due to beign on the wrong CUDA stream
// 
// 645   4/29/14 3:26p Fseide
// this version works but does not add gradient directly from unquantizing
// 
// 644   4/29/14 2:55p Fseide
// (comment)
// 
// 643   4/29/14 2:21p Fseide
// (fixed a log message and a typo)
// 
// 642   4/29/14 1:58p Fseide
// (comment)
// 
// 641   4/29/14 1:48p Fseide
// unquantizeandaggregatestripe() and gradientfixups() now check for
// learning rate != 1.0
// 
// 640   4/29/14 1:42p Fseide
// unquantizeandaggregatestripe() now also takes the learning rate;
// gradientfixups() now multiplies the gradient with an additional
// learning-rate parameter -> this way it can compute the final-final
// thing that gets added to the model;
// syncassignaggregatedstripeandunquantize() now takes bpinfo so it can
// know about 'distributefixedcost' mode
// 
// 639   4/28/14 4:06p Fseide
// 'distributefixedcost' mode enabling logic implemented & tested,
// arguments passed to unquantizeandaggregatestripe(), but actual
// math/kernel not there yet (and thus, the mode is not enabled in the
// checked-in version)
// 
// 638   4/28/14 2:40p Fseide
// implemented direct use of MPI_Allreduce() in non-quantized case (as one
// would prefer for model-averaging case);
// made some methods robust to empty matrix dimensions;
// new methods mpiallreducegradient(), mpihelper::allreducescalar(), and
// allreduce() for matrices
// 
// 637   4/28/14 10:31a Fseide
// using the true mbframes across nodes (no rounding) now debugged and
// tested
// 
// 636   4/27/14 16:58 Fseide
// qpackages now contain a header (struct mpistripeheader) that stores the
// number of frames of the gradient they represent;
// syncassignaggregatedstripeandunquantize() no longer takes the
// 'numstripes' argument (it was unused and is superfluous now)
// 
// 635   4/27/14 16:26 Fseide
// moved resetting of mpistripebuffersizes[] out from the init lambda to
// entercomputation(), which can now pre-init it to allocate a header
// 
// 634   4/27/14 16:11 Fseide
// new modelupdateinfo parameter 'distributefixedcost';
// unquantizeandaggregatestripe() now takes all parameters needed to
// distribute fixed cost (AdaGrad, momentum)
// 
// 633   4/27/14 15:34 Fseide
// (towards distributed fixed cost)
// 
// 632   4/27/14 15:07 Fseide
// (comments)
// 
// 631   4/27/14 15:03 Fseide
// data parallelism: first steps towards doing part of fixed cost inside
// the stripe operation
// 
// 630   4/09/14 18:03 Fseide
// fixed scale()
// 
// 629   4/09/14 16:54 Fseide
// removed an error check that is now outdated (dropout(), prescale flag)
// 
// 628   4/08/14 18:05 Fseide
// changed dropout() CUDA branch to use the applyonsubstreams() pattern
// 
// 627   4/06/14 20:49 Fseide
// Kopt now uses all nodes except for mbsize 56 (where it uses 1 node
// only)
// 
// 626   4/04/14 21:24 Fseide
// hardened posteriorstats() against weird numbers (I saw 1.#R, still
// don't know what that is!);
// new method absaverage()
// 
// 625   4/04/14 14:16 Fseide
// forceuseallnodes set to true
// 
// 624   3/28/14 7:01p Fseide
// constant 'quantizetobits' removed, it's now a cmd-line argument
// 
// 623   3/26/14 10:35a Fseide
// entermpiaggregation() no longer rounds row indices for SSE, since we
// only stripe by columns anyway so this is misguided now (and caused a
// bug when comining model and data parallelism)
// 
// 622   3/21/14 8:44a Fseide
// added a log message
// 
// 621   3/20/14 4:15p Fseide
// (removed that dummy constructor I just added since it does not work
// actually)
// 
// 620   3/20/14 4:07p Fseide
// some refactoring and initial code for local loop for data parallelism
// 
// 619   3/19/14 5:53p Fseide
// new methods (un)scalerowwise()
// 
// 618   3/19/14 11:19a Fseide
// towards model parallelism with ReLU
// 
// 617   2/28/14 2:41p Fseide
// MPI aggregator now distinguishes between doublebufferingallowed (now a
// general compile-time setting) and doublebufferingrequested (runtime
// setting, gated by doublebufferingallowed )
// 
// 616   2/27/14 21:00 Fseide
// (comments)
// 
// 615   2/27/14 18:51 Fseide
// new method swap() in rmbmodelmatrixbase and two underlying classes
// 
// 614   2/24/14 17:38 Fseide
// (comment)
// 
// 613   2/24/14 16:40 Fseide
// new method rbmstatevectorsrefbase::stripe()
// 
// 612   2/24/14 3:11p Zhijiey
// fix latgen build break when NOMPI is defined
// 
// 611   2/21/14 13:38 Fseide
// entercomputation() now takes a flag to en/disable double buffering;
// double buffering now gets automatically disabled when using AdaGrad
// since it does not play well with it (this is not a final answer)
// 
// 610   2/21/14 13:30 Fseide
// entermpiaggregation() now takes 'bits' as an explicit parameter
// 
// 609   2/20/14 4:27p Fseide
// adadenom() now takes a mean accumulator as well (currently only used
// for diagnostics, not in actual AdaGrad normalizations, assuming a fixed
// target is given);
// likewise, network layer now keeps a mean accumulator matching the sqr
// acc for AdaGrad;
// applyadagrad() now again computes avdenom even if it is not used (i.e.
// when fixed target is given), for diagnostics;
// 
// 608   2/19/14 15:53 Fseide
// (comments)
// 
// 607   2/18/14 7:11p Fseide
// hoped I fixed non-parallelized computation with model parallelism+MPI
// 
// 606   2/18/14 6:55p Fseide
// (logging)
// 
// 605   2/18/14 4:54p Fseide
// bug fix in entermpiaggregation(), mixed up rows() and cols()
// 
// 604   2/18/14 2:50p Fseide
// implemented multi-GPU support (model parallelism) for MPI (data
// parallelism), not tested yet but should at least work for single GPU;
// allocatetransferbuffer() now allocates on the correct GPU
// 
// 603   2/18/14 11:45a Fseide
// implemented sharing AdaGrad avdenom across all layers
// 
// 602   2/18/14 9:57a Fseide
// raw_dmbframes is now updated under the control of MPI aggregation. This
// is a preparation for separating out AdaGrad from ...update3().
// 
// 601   2/17/14 18:28 Fseide
// (comments)
// 
// 600   2/17/14 5:56p Fseide
// more aggressive parallelization
// 
// 599   2/17/14 5:28p Fseide
// implemented forcedfullysyncoperation option to allow for
// timemeasurements in MPI aggregate();
// disabledoublebuffering() renamed to forcefullysyncoperation()
// 
// 598   2/17/14 2:08p Fseide
// new method mpiaggregator::disabledoublebuffering() to support
// --mpiforcesync flag
// 
// 597   2/17/14 1:11p Fseide
// updated iss tracking
// 
// 596   2/14/14 19:22 Fseide
// reduced the parallelism a little, now that we fixed up the "fixed cost"
// part quite a bit
// 
// 595   2/14/14 6:18p Fseide
// added a log message to adagradient()
// 
// 594   2/14/14 4:05p Fseide
// new faster AdaGrad function that combines two: adagradientfromsqracc()
// 
// 593   2/14/14 13:45 Fseide
// started to refactor adagradient(), hoping for a potential speed-up
// 
// 592   2/14/14 10:54 Fseide
// added model variables--a dictionary of key/value pairs that is
// persisted in the model file itself, for carrying over iteration state
// 
// 591   2/13/14 21:06 Fseide
// merged the send() and recv() schedule loops
// 
// 590   2/13/14 20:44 Fseide
// added a log ping
// 
// 589   2/13/14 20:26 Fseide
// also made the aggregated-gradient exchange asynchronous (the initial
// sub-gradient exchange got 8% faster)
// 
// 588   2/13/14 19:59 Fseide
// split the aggregate redistribution also into async functions
// 
// 587   2/13/14 19:56 Fseide
// moved sendwaitall() after quantization, for a little more concurrency
// 
// 586   2/13/14 19:33 Fseide
// disabled logging for async send/recv
// 
// 585   2/13/14 19:27 Fseide
// more async process--receives are scheduled upfront, then after each
// send() we process whatever already came back
// 
// 584   2/13/14 18:52 Fseide
// replaced sendwait() loop by sendwaitall() which maps to MPI_Waitall()
// 
// 583   2/13/14 18:48 Fseide
// towards more async operation--now we wait for send completion at the
// end
// 
// 582   2/13/14 18:30 Fseide
// bug fix in asynchandle;
// added logging to new async process
// 
// 581   2/13/14 17:54 Fseide
// refactored towards more asynchronous send/recv (but still identical
// operation)
// 
// 580   2/13/14 15:21 Fseide
// bug fix for variable Kopt (#nodes in MPI mode): in MPI mode, model is
// now being sync'ed from node 0 to all others at start of each epoch, as
// to get all nodes the latest model even if they did not participate in
// previous epochs and thus did not get their model updates
// 
// 579   2/13/14 13:42 Fseide
// entercomputation(): changed the parameters of Kopt a little based on
// experiments
// 
// 578   2/11/14 18:31 Fseide
// adjusted the Kopt parameters to less parallelism
// 
// 577   2/11/14 17:31 Fseide
// entercomputation() now takes a full MB size, not a half-size
// 
// 576   2/11/14 17:07 Fseide
// reenabled Kopt computation
// 
// 575   2/11/14 17:03 Fseide
// reenabled to skip MPI aggregation entirely for K=1
// 
// 574   2/11/14 15:29 Fseide
// optimized CUDA matrix allocate() to never shrink, in order to avoid
// potential locks
// 
// 573   2/11/14 3:06p Fseide
// simplified unlock() for ref writing, sync case (previous check-in)
// 
// 572   2/11/14 3:04p Fseide
// added equivalents to synchronize() that specifically wait after fetch()
// and assign(), implemented as stream syncs rather than global device
// syncs, hoping for better efficiency
// 
// 571   2/11/14 14:10 Fseide
// annotated in log messages and comments whether a
// quantization/MPI-related function is async or blocking
// 
// 570   2/11/14 10:34 Fseide
// got node subsetting to work (I hope), by skipping pieces of code in
// enter/exitmpicomputation()
// 
// 569   2/10/14 11:29 Fseide
// reenabled bg thread
// 
// 568   2/08/14 14:24 Fseide
// bug fix: MPI aggregate now again works for K==1 (missed an assign)
// 
// 567   2/08/14 13:06 Fseide
// disabled bg thread for now, for time measurements
// 
// 566   2/07/14 19:00 Fseide
// moved 'timeme' class to cudamatrix.h so we can use it inside
// cudamatrix.cpp
// 
// 565   2/07/14 18:29 Fseide
// fixed definition of REQUANTNOCUDA, was reverse
// 
// 564   2/07/14 6:09p F-gli
// added dump hacks to make NOMPI build
// 
// 563   2/07/14 15:45 Fseide
// REQUANTNOCUDA now works, keeping it enabled (although it's not
// efficient as of yet)
// 
// 562   2/07/14 14:52 Fseide
// unquantizeandaggregatestripe() now uses different buffers for each
// stream
// 
// 561   2/07/14 14:14 Fseide
// (minor fix of previous check-in)
// 
// 560   2/07/14 14:05 Fseide
// completed the GPU code for requantization, but not giving the same
// result yet so I keep it disabled;
// quantizeandfetchqstripe() now supports the 'reuserangescaled' flag just
// like the CPU version
// 
// 559   2/07/14 13:22 Fseide
// bug fix in new peerbuffers[] initialization
// 
// 558   2/07/14 11:58 Fseide
// syncassignqstripeandunquantize() now takes an 'add' parameter;
// new flag 'prioritystream' to newqstripe(), but not used anywhere yet
// 
// 557   2/07/14 11:49 Fseide
// replaced the two peerbufferskN by peerbuffers[kfrom], so that we don't
// need to CPU-wait on GPU transfers in middle of transfer loop
// 
// 556   2/07/14 11:36 Fseide
// added GPU-related requant code, not active yet
// 
// 555   2/07/14 11:11 Fseide
// changed aggacc/resstripe from CPU-side matrices to CPU/GPU matrices, in
// prep for GPU-side requantization
// 
// 554   2/07/14 10:56 Fseide
// changed aggacc/recstripe from an array to a single value since it only
// ever is applied to our own stripe
// 
// 553   2/07/14 9:09 Fseide
// comments in prep of GPU requant
// 
// 552   2/07/14 8:55 Fseide
// renamed quantizeaggregatedstripe() to
// quantizeandassignaggregatedstripe() and combined it with its subsequent
// call to assignaggregatedstripe(), for upcoming move to doing this on
// the GPU
// 
// 551   2/06/14 18:44 Fseide
// (comments)
// 
// 550   2/06/14 18:16 Fseide
// fixed concurrent unquantization/sendrecv
// 
// 549   2/06/14 14:43 Fseide
// bg thread now enabled;
// bg thread is now kicked off after submitting the unquantization, to
// make sure the qaggstripe is not overwritten earlier (this works in
// conjunction with additional synchronization in the CUDA lib)
// 
// 548   2/06/14 9:52 Fseide
// disabled overlapped unquantization in mpi aggregate() because it has a
// bug (known, fix later once I figured out the multi-threading issue)
// 
// 547   2/05/14 19:26 Fseide
// adagradient() now accepts a fudge factor instead of a user-chosen
// target (not used yet)
// 
// 546   2/05/14 19:18 Fseide
// moved 'updateactualavdenom' flag out from parallelrbmmatrix
// 
// 545   2/05/14 18:46 Fseide
// shifted up the strange weight of targetavdenom by 32 out of the actual
// AdaGrad routines (we will eventually eliminate it)
// 
// 544   2/05/14 18:24 Fseide
// refactored the strange AdaGrad weighting code
// 
// 543   2/05/14 16:09 Fseide
// experimented with mbsizepernode (irrelevant since disabled)
// 
// 542   2/04/14 19:31 Fseide
// enabled double buffering but disabled the actual bg thread--behaves
// like not double-buffered? clearly a huge huge bug
// 
// 541   2/03/14 21:43 Fseide
// (comment)
// 
// 540   2/03/14 19:42 Fseide
// some (unelegant) renaming of CUDA quant sync functions for more clarity
// 
// 539   2/03/14 18:19 Fseide
// reenabled to recompute the mean at second quantization
// 
// 538   2/03/14 17:47 Fseide
// implemented AdaGrad-like pre-scaling for quantization, but does not
// work (disabled)
// 
// 537   2/03/14 16:19 Fseide
// (comment)
// 
// 536   2/03/14 16:09 Fseide
// changed AdaGrad implementation to isolate it out;
// minor fix of AdaGrad in interpreting old-style parameters (256 frames,
// now compatible with double-buffering=128 frames)
// 
// 535   2/03/14 11:32 Fseide
// (comment)
// 
// 534   2/02/14 22:19 Fseide
// AdaGrad now skips first update when double-buffering (which would use
// the fake zero gradient);
// check for first zero gradient moved from matrix function to rbm.h;
// forceaggregate=true for now so that we can test quant without data
// parallelism interference;
// disabled double-buffering for now since it does not play with AdaGrad,
// something wrong there
// 
// 533   1/31/14 0:31 Fseide
// (comments)
// 
// 532   1/30/14 19:42 Fseide
// now going back to not reducing the #nodes, since there is still some
// bug with it;
// better configurability of MPI aggregator (local 'const' variables
// rather than code changes)
// 
// 531   1/27/14 15:30 Fseide
// (comments)
// 
// 530   1/27/14 10:51 Fseide
// disabled using a separate communicator
// 
// 529   1/27/14 10:24 Fseide
// removed a lot of fprintf()s since I found the sub-group bug
// 
// 528   1/27/14 10:19 Fseide
// bug fix: needed to reset 'tag' to keep it in sync over varying #nodes
// 
// 527   1/26/14 20:04 Fseide
// (fixed a log message)
// 
// 526   1/26/14 19:38 Fseide
// unquant now overlaps with sendrecv() --didn't make much difference
// though
// 
// 525   1/26/14 5:27p F-gli
// added dummy MPI_Barrier function for NOMPI
// 
// 524   1/26/14 16:14 Fseide
// (comment)
// 
// 523   1/26/14 14:49 Fseide
// (comments)
// 
// 522   1/26/14 9:58 Fseide
// temporarily added an MPI barrier
// 
// 521   1/26/14 8:49 Fseide
// (added disabled debug code)
// 
// 520   1/24/14 19:02 Fseide
// disabled automatic scaling w.r.t. nodes due to a multi-threading bug
// 
// 519   1/24/14 17:39 Fseide
// (comment)
// 
// 518   1/24/14 17:01 Fseide
// MPI aggregation now short-cut if dynamically selected K is 1
// 
// 517   1/24/14 16:28 Fseide
// number of compute nodes is now dynamically chosen (for small mb sizes,
// less nodes may be faster)
// 
// 516   1/24/14 15:05 Fseide
// (shortened a message)
// 
// 515   1/24/14 11:44 Fseide
// (added an issue to track for 1-bit SGD)
// 
// 514   1/24/14 11:06 Fseide
// towards setting numframes correctly in ...update3()
// 
// 513   1/23/14 14:41 Fseide
// more instrumentation for timing
// 
// 512   1/23/14 11:48a F-gli
// fixed faked def for NOMPI case
// 
// 511   1/23/14 11:18a Fseide
// added more instrumentation for time measurements
// 
// 510   1/23/14 9:07 Fseide
// added functions for timing code
// 
// 509   1/22/14 19:42 Fseide
// temporarily added instrumentation for timing--the culprit is
// computerange()
// 
// 508   1/22/14 13:23 Fseide
// (further editing of comments)
// 
// 507   1/22/14 13:10 Fseide
// edited/updated the comments of the aggregate() functions
// 
// 506   1/22/14 10:19 Fseide
// (un)quantization now done on CUDA
// 
// 505   1/22/14 9:57 Fseide
// disabled QUANTNOCUDA, but separated a new UNQUANTNOCUDA which is kept
// active
// 
// 504   1/22/14 8:59 Fseide
// added a comment
// 
// 503   1/21/14 19:23 Fseide
// (comments)
// 
// 502   1/21/14 6:32p Fseide
// MPI allreduce() now clever enough to handle float, double, and int
// arguments
// 
// 501   1/21/14 5:52p Fseide
// redid the 'clever' tag, since the issue was caused by having another
// MPI_Allreduce() on the main thread, which is not allowed
// 
// 500   1/21/14 5:37p Fseide
// undid the attempt to be 'clever' for the 'tag' variable, now using the
// same tag for both directions again, otherwise it did not work
// 
// 499   1/21/14 5:24p Fseide
// added a 0.5 sec sleep per node to get a somewhat staggered startup
// 
// 498   1/21/14 5:21p Fseide
// added initial handshake of MPI functions to get stuff kicked off early
// 
// 497   1/21/14 4:54p Fseide
// bug fix in sendrecv() tag
// 
// 496   1/21/14 16:45 Fseide
// (comments)
// 
// 495   1/21/14 16:44 Fseide
// (comment)
// 
// 494   1/21/14 16:28 Fseide
// removed MPI getconfiguration(), use node() and nodes() instead
// 
// 493   1/21/14 16:27 Fseide
// made sendrecv() tag a class member and thus global
// 
// 492   1/21/14 4:12p Fseide
// lots of logging around bg thread termination/failure
// 
// 491   1/21/14 15:18 Fseide
// (added some log messages related to the end-of-epoch crash)
// 
// 490   1/21/14 12:39p Fseide
// reenabled bg thread for longer test
// 
// 489   1/21/14 12:24 Fseide
// disabled some debug output
// 
// 488   1/21/14 12:21 Fseide
// switched back to 1 bit (double-buffering is enabled, bg thread is not)
// 
// 487   1/21/14 11:54 Fseide
// (comment)
// 
// 486   1/21/14 11:53 Fseide
// added to the 1-bit 'iss tr'
// 
// 485   1/21/14 11:51 Fseide
// disabled optimization to not recompute the quantization range, as it
// may have unknown impact
// 
// 484   1/21/14 11:49 Fseide
// bug fix in skipping the recomputation: forgot to scale to #nodes;
// 'accuracy' now set to 5 stddevs (before: 2), big difference for 16 bit
// in early iterations;
// bg thread disabled for now to get better-comparable log output
// 
// 483   1/20/14 19:34 Fseide
// bits default changed to 16;
// bias layout now horizontally split, that is, only one stripe in node[0]
// 
// 482   1/20/14 19:16 Fseide
// added a message
// 
// 481   1/20/14 17:32 Fseide
// back to 1 bit
// 
// 480   1/20/14 17:21 Fseide
// (comments)
// 
// 479   1/20/14 17:06 Fseide
// reenabled double-buffering (bg thread)
// 
// 478   1/20/14 16:22 Fseide
// fixed a silly bug (wrong init of 'isfirst' flag in MPI aggregate())
// 
// 477   1/20/14 15:18 Fseide
// (refined a printf-debugging message)
// 
// 476   1/20/14 14:29 Fseide
// cleaned up some printf-debugging messages
// 
// 475   1/20/14 14:15 Fseide
// added lots of printf debugging messages for MPI parallelization
// 
// 474   1/20/14 11:44 Fseide
// syncassignaggregatedstripeandunquantize() now clears the very first raw
// gradient to 0 (this is needed to get identical results once we have >1
// node)
// 
// 473   1/20/14 11:25 Fseide
// (commented on a bug)
// 
// 472   1/20/14 11:21 Fseide
// JUSTDOUBLEBUFFER mode is now consistent with the bg thread (before, the
// first gradient was 0; now the first gradient is applied twice, whether
// that's correct or not)
// 
// 471   1/20/14 8:47 Fseide
// temporarily switched MPI mode to 32-bit quant (=no quant)
// 
// 470   1/18/14 22:30 Fseide
// a bit of tidy-up
// 
// 469   1/18/14 22:24 Fseide
// reenabled the thread
// 
// 468   1/18/14 22:19 Fseide
// reenabled step2and3()
// 
// 467   1/18/14 22:10 Fseide
// double-buffering flag now owned by mpiaggregator
// 
// 466   1/18/14 21:46 Fseide
// added a fake mode for double buffering on the 'float' level
// (JUSTDOUBLEBUFFER)
// 
// 465   1/18/14 20:01 Fseide
// temporarily disabled
//  - CUDA quantization (by defining QUANTNOCUDA)
//  - bg thread (now executing synchronously)
//  - step2and3 now does nothing except returning 'localbuffers'
// --> result is bad; we need to track down why
// 
// 464   1/18/14 19:47 Fseide
// added a comment
// 
// 463   1/18/14 19:07 Fseide
// fixed some incorrect indices in MPI interaction (no impact on 1 node
// simulations)
// 
// 462   1/18/14 18:21 Fseide
// (comments)
// 
// 461   1/17/14 17:42 Fseide
// (comment)
// 
// 460   1/17/14 16:40 Fseide
// (disabled bg thread by putting a wait() after it)
// 
// 459   1/17/14 16:22 Fseide
// MPI exitcomputation() now actually releases both GPU buffers
// 
// 458   1/17/14 15:58 Fseide
// added some debug messages
// 
// 457   1/17/14 15:43 Fseide
// (minor fix of last checkin)
// 
// 456   1/17/14 15:41 Fseide
// exitcomputation() now waits for thread to complete
// 
// 455   1/17/14 15:36 Fseide
// disabled the bg thread
// 
// 454   1/17/14 15:17 Fseide
// completed the double-buffered execution (with a bg thread), but not
// really tested yet;
// implemented a mini emulation of VS 2012's task class
// 
// 453   1/17/14 12:40 Fseide
// double-buffering prep done (now we still need the thread)
// 
// 452   1/17/14 12:36 Fseide
// first step towards double buffers
// 
// 451   1/17/14 12:23 Fseide
// changed qstripe to no longer remember the CPU-side buffer, but instead
// just a GPU-side buffer and associated events
// 
// 450   1/17/14 11:31 Fseide
// allocatetransferbuffer() implemented, now uses cudamatrix lib's
// newsharedtransferbuffer() when in CUDA mode
// 
// 449   1/17/14 11:23 Fseide
// GPU buffer allocation now owned by dbn.h, not mpiaggregator (and it
// will be pushed further)
// 
// 448   1/17/14 11:17 Fseide
// eliminated the second buffer in mpiaggregate() (it will come back for
// double-buffering though)
// 
// 447   1/17/14 11:02 Fseide
// now using fetchbuffer also to send things back to the GPU
// 
// 446   1/17/14 10:57 Fseide
// MPI: renamed begin() to data(), now we follow the vecetor interface
// 
// 445   1/17/14 10:54 Fseide
// MPI: changed bufferbegin/end to bufferbegin/size
// 
// 444   1/17/14 10:42 Fseide
// MPI now handles a missing msmpi.dll;
// refactored the buffering a little (more to come)
// 
// 443   1/17/14 9:28 Fseide
// renamed some data structures
// 
// 442   1/17/14 9:21 Fseide
// streamlined enter/exitmpiaggregation() a little, such that on matrix
// level, those functions no longer know about the mpiaggregator
// 
// 441   1/16/14 21:59 Fseide
// added design notes on going double-buffered
// 
// 440   1/16/14 9:08p Fseide
// moved a 10-second Sleep() from main.cpp into MPI code, now only
// executing it there (seems that was meant originally, probably put in by
// Jasha?)
// 
// 439   1/16/14 3:19p Fseide
// enabled MPI again
// 
// 438   1/16/14 12:12 Fseide
// added an fflush(stderr) to MPI and --stderr initialization
// 
// 437   1/16/14 11:18 Fseide
// changed mpigradientstripebuffer to use page-locked RAM
// 
// 436   1/15/14 19:05 Fseide
// (another missing fake MPI entry)
// 
// 435   1/15/14 19:02 Fseide
// 3 more missing fake MPI symbols added
// 
// 434   1/15/14 18:47 Fseide
// added a missing 'mpiconsts' fake MPI type (MPI_CHAR)
// 
// 433   1/14/14 18:35 Fseide
// now skipping second computerange() (instead, we reuse the range
// determined on our local stripe)--to be verified once we have multiple
// nodes
// 
// 432   1/14/14 10:12 Fseide
// cleaned up quantization-related CPU-side interfaces to assume a matrix
// patch to specify the rect dimension, rather than passing the dims
// around all the time (in prep for SSE-based quantization, since now we
// can enforce alignment)
// 
// 431   1/14/14 9:48 Fseide
// added ssematrix::patch;
// towards using this in (un)quantize() (for SSE-optimizing quantization)
// 
// 430   1/14/14 8:38 Fseide
// towards using patches for base matrix, for use in quantize()
// 
// 429   1/13/14 17:33 Fseide
// (added code to time unquantization)
// 
// 428   1/13/14 11:16 Fseide
// disabled QUANTNOCUDA (the code works)
// 
// 427   1/13/14 9:38 Fseide
// temporarily disabled GPU codepath for quantization, for testing the new
// int-loop based quantization code
// 
// 426   1/10/14 20:08 Fseide
// (comments)
// 
// 425   1/10/14 18:34 Fseide
// changed default quantization back to 1 bit
// 
// 424   1/10/14 16:51 Fseide
// syncassignaggregatedstripeandunquantize() now takes CPU-side buffer so
// that it can operate on the CPU without GPU
// 
// 423   1/10/14 16:05 Fseide
// (first bug fixes for MPI)
// 
// 422   1/10/14 11:48 Fseide
// CPU-side matrix quantization can now pass a patch region;
// agg residual now allocated per stripe, in order to match the dimensions
// and layout of agg accumulator
// 
// 421   1/10/14 11:22 Fseide
// moved MPI quantization residuals out of rbmbase into
// parallelrbmmatrix.h where they belong
// 
// 420   1/10/14 11:01 Fseide
// renamed at/detachmpiaggregator() to enter/exitmpiaggregation()
// 
// 419   1/10/14 10:59 Fseide
// changed all lambdas passed to MPI aggregate() to not capture anything
// by reference (since they will run on a bg thread eventually)
// 
// 418   1/10/14 10:48 Fseide
// (split quantizer into 3 classes)
// 
// 417   1/09/14 19:57 Fseide
// completed the MPI aggregation function--it compiles... now it must be
// tested!!
// 
// 416   1/09/14 19:11 Fseide
// implemented call to MPI_sendrecv() for first phase
// 
// 415   1/09/14 18:53 Fseide
// partially implemented the second MPI exchange step, incl. buffer
// management
// 
// 414   1/09/14 17:53 Fseide
// began to implement the big fat MPI aggregator using the lambdas
// 
// 413   1/09/14 17:06 Fseide
// last lambda for MPI aggregation implemented, now on to the MPI
// aggregator function itself!
// 
// 412   1/09/14 16:55 Fseide
// quantizeaggregatedstripe() implemented
// 
// 411   1/09/14 16:41 Fseide
// unquantizeandaggregatestripe() implemented
// 
// 410   1/09/14 15:55 Fseide
// added more lambdas for MPI aggregation
// 
// 409   1/09/14 15:06 Fseide
// towards implementing the lambdas that the MPI aggregator needs
// 
// 408   1/09/14 13:52 Fseide
// quantization now peruses the residual in-place (instead of first adding
// it explicitly to the raw gradient)--better separation of concerns
// (residual belongs to quantization);
// bug fix in quantizeandfetchqstripe(): forgot to apply the patch to the
// residual
// 
// 407   1/08/14 17:14 Fseide
// (comment)
// 
// 406   1/08/14 17:11 Fseide
// towards MPI and quantization (currently totally broken since MPI
// transfer is not cross-layer);
// mpiaggregate() now takes a second residual parameter
// 
// 405   1/08/14 15:01 Fseide
// added 'cudalock' in support of data parallelism, which accesses the GPU
// on a parallel thread--but it is already known that it is buggy
// 
// 404   1/08/14 11:00 Fseide
// (comments)
// 
// 403   1/08/14 10:34 Fseide
// mpistriperefs_dx and qstripes_dx moved to parallelrbmmatrix.h
// 
// 402   1/08/14 9:54 Fseide
// towards moving MPI data exchange from rbm.h to parallelrbmmatrix.h
// 
// 401   1/08/14 9:44 Fseide
// qstripe no longer knows about patch dimension and 'bits' parameter
// (none of its business)
// 
// 400   1/08/14 8:49 Fseide
// qstripe now does not take a buffer+offset but begin and end iterator;
// qstripe is now handed back from cudamatrix lib as a shared_ptr with
// custom deleter;
// MPI initialization sequence: entermpiaggregation() determines the
// needed stripe sizes and buffer offsets, while first use (within-layer)
// will lazily allocate (cross-layer) stripe buffers;
// split off mpihelper from mpiaggregator (mpihelper handles basic MPI
// interaction)
// 
// 399   1/06/14 9:45p V-haofu
// change the nucudadevices() condtion error
// 
// 398   1/03/14 17:26 Fseide
// initialization sequence of MPI stuff stratified, new methods
// entermpiaggregation() and exitmpiaggregation() in rbm.h, called by
// respective functions in dbn.h;
// moved logic to determine buffer size etc. from CUDA-side qstripe to a
// new CPU-side structure mpistripebufferref (since stripes may live on
// different GPUs, the GPU side cannot do this);
// stripes are now associated with a GPU (in theory--the actual
// determination of the GPU device is not implemented)
// 
// 397   1/03/14 16:01 Fseide
// towards MPI data-parallel gradient aggregation
// 
// 396   1/03/14 13:02 Fseide
// changed quantizeunquantize() to use quantizer class, for testing it
// 
// 395   1/03/14 12:19a V-haofu
// change the initialization of quantizer
// 
// 394   12/20/13 15:06 Fseide
// switched from quantizer class in rbm.h to msra::math::columnquantizer, which
// is the class shared with the GPU
// 
// 393   11/28/13 3:57p V-haofu
// modify the quantization for 1 bit case(use mean to unquantize the data)
// 
// 392   11/01/13 4:33p V-haofu
// implementation of quantization for data parallelization
// 
// 391   11/01/13 4:30p V-haofu
// 
// 390   10/16/13 5:43p V-haofu
// in mpi mode, we calculate and exchange sum of gradients(before momentum
// accumulation), then do momentum accumulation in each node
// 
// 389   9/29/13 19:11 Fseide
// (comment)
// 
// 388   9/29/13 18:33 Fseide
// bug fix: I had screwed up the loop order in matprod_mm()
// 
// 387   9/29/13 16:39 Fseide
// lockforwriting() now allows to disable the initial sync, speeds up
// input-layer preparation as it allows to compute it on the CPU
// concurrently (8% total speed-up)
// 
// 386   9/29/13 15:43 Fseide
// documented a bug in posteriorstats() and added a warning message when
// it is detected
// 
// 385   9/29/13 14:33 Fseide
// matprod_mm() now scheduling all ops per stream at once, which is needed
// for proper overapping
// 
// 384   9/29/13 14:15 Fseide
// first version with substream computation and communication that is
// correct... but not efficient due to wrong submission order
// 
// 383   9/29/13 13:19 Fseide
// matprod_mm() now takes a buffer variable for model parallelism
// 
// 382   9/29/13 9:05 Fseide
// enabled substream gemm() in matprod_mm() (but data exchange is not yet,
// so no speed-up)
// 
// 381   9/29/13 8:57 Fseide
// new method foreachsubbatchanddevice() to abstract the device loop, for
// sharing between matprod_mtm() and matprod_mm()
// 
// 380   9/28/13 20:57 Fseide
// (comment)
// 
// 379   9/28/13 20:54 Fseide
// mulbydsigm() now using substreams, in prep for substream version of
// matprod_mm()
// 
// 378   9/28/13 20:30 Fseide
// moved substream implementation of sigmoid() into a new function
// applyonsubstreams() which takes the operation as a lambda--next will be
// to use that for the derivative on the back-prop path
// 
// 377   9/28/13 20:09 Fseide
// (comment)
// 
// 376   9/28/13 19:48 Fseide
// (comments)
// 
// 375   9/28/13 19:36 Fseide
// tidied up the last posteriorstats() fix (removed the dummy args, made
// function arg names consistent, lots of comments)
// 
// 374   9/28/13 19:08 Fseide
// rewrote posteriorstats() on CUDA, since the old version did actually
// not work for multiple GPUs due to a design flaw
// 
// 373   9/28/13 10:33 Fseide
// (comment)
// 
// 372   9/28/13 10:27 Fseide
// (added a comment)
// 
// 371   9/28/13 10:17 Fseide
// CUDA posteriorstats() now returns the stats values directly by
// cudaMemcpy() inside that function (more efficient since only 3 floats)
// 
// 370   9/28/13 8:52 Fseide
// posteriorstats() now collating information on the GPUs into a single
// number
// 
// 369   9/28/13 7:21 Fseide
// (comment)
// 
// 368   9/27/13 18:58 Fseide
// added comments and a critcal BUGBUG to be fixed once I come back to
// office
// 
// 367   9/27/13 18:45 Fseide
// model-parallel version of posteriorstats() implemented;
// changed argument order of posteriorstats() CUDA function to be more
// logical
// 
// 366   9/27/13 18:05 Fseide
// merged stripedposteriorstats() and posteriorstats() (posteriorstats()
// can now run striped)
// 
// 365   9/27/13 17:18 Fseide
// seterrorsignal() now row-striped
// 
// 364   9/27/13 2:52p V-jiacli
// removed the error throw and added "const" for parameter in gems()
// settodiff()
// 
// 363   9/27/13 2:45p F-gli
// fixed a namespace issue when compiling in HVite
// 
// 362   9/27/13 14:39 Fseide
// some vectors functions use stripedwrtcols instead of stripedwrtrows,
// which is incorrect (thanks to Jiachen Li for finding this)--fixed for
// gemm() (following Jiachen) and commented for the other 4 cases
// (functions that are currently not used, so cannot test)
// 
// 361   9/26/13 17:16 Fseide
// added a comment: we no longer enforce submbs 1024 for K=2 GPUs, that is
// no longer applicable now that we have softmax fixed
// 
// 360   9/26/13 16:36 Fseide
// softmax() finished, now uses old code path again for single GPU (so we
// can be sure not to break anything, although it ends up being nearly
// identical to the new code path)
// 
// 359   9/26/13 16:18 Fseide
// removed all that experimental sync stuff from softmax(), it is
// performant now
// 
// 358   9/26/13 16:09 Fseide
// parallelized softmax() now working for 2 GPUs (and thus hopefully also
// for more)
// 
// 357   9/26/13 15:43 Fseide
// stripedsoftmaxstep1() now takes the full matrix, not just a stripe;
// softmax() currently not working for K>1;
// softmaxs0t() now returns the frame-wise partial sum not as a return
// value but by writing it into a location whose address is
// passed--otherwise it would not work (compiler bug?)
// 
// 356   9/25/13 20:02 Fseide
// (added comments)
// 
// 355   9/25/13 19:11 Fseide
// implemented proper model parallelism in softmax(), using the trick we
// already used in pipeline training
// 
// 354   9/25/13 17:51 Fseide
// towards model parallelism for softmax()--added the buffer
// 
// 353   9/24/13 17:38 Fseide
// added better rules for stripingconfig (for larger MB sizes)
// 
// 352   9/24/13 17:00 Fseide
// enabled substreams for sigmoid() and addtoallcols()--now we have a
// chain for forwardprop() except softmax
// 
// 351   9/24/13 13:44 Fseide
// matprod_mtm() now scheduling on sub-streams (but not working yet since
// sub-stream sync has a problem)
// 
// 350   9/24/13 10:34 Fseide
// (added another log message to TIME_MTM)
// 
// 349   9/23/13 16:45 Fseide
// added better timing measurement to matprod_mtm(), separating out
// transfer and computation explicitly and averaging over multiple
// 
// 348   9/23/13 15:17 Fseide
// (renamed TIME_CUDA to TIME_MTM inside matprod_mtm())
// 
// 347   9/23/13 13:25 Fseide
// (added comments)
// 
// 346   9/23/13 9:54 Fseide
// time measurement now prints accumulative results over many minibatches
// and all layers
// 
// 345   9/22/13 12:50 Fseide
// matprod_mtm() now logs its configuration (for reference)
// 
// 344   9/21/13 17:21 Fseide
// hard-coded a rule to choose sub-minibatch size for parallelized
// matprod_mtm() based on timing experiments (512 for K>2, else 1024,
// which means no overlap if mbsize=1024 as well)
// 
// 343   9/21/13 12:59 Fseide
// bug fix in makeinputrowstripingasync(), I indeed screwed up the loop
// order and caused dependencies between copy operations;
// matprod_mtm() now uses 512 instead of 256 frames in overlapped
// operation
// 
// 342   9/19/13 23:17 Fseide
// enabled overlapped processing for model parallelism (seems to work, and
// noone runs this anyway on multi-CUDA)
// 
// 341   9/19/13 19:17 Fseide
// (some commented-out debug code)
// 
// 340   9/19/13 13:09 Fseide
// fixed a bug in data exchange for model parallelism, now correct when
// copying synchronously
// 
// 339   9/19/13 11:06 Fseide
// refactored matprod_mtm() into sub-minibatches/overlapping (but still
// disabled)
// 
// 338   9/18/13 1:35p Fseide
// some steps forward, end-to-end code, but not working yet
// 
// 337   9/18/13 12:25p Fseide
// first intermediate version of matprod_mtm() that uses async send() (but
// then using a forced sync), seems to work, same accuracy and same
// runtime as before
// 
// 336   9/17/13 6:56p Fseide
// towards overlapped matprod_mtm(): new method
// makeinputrowstripingasync() (which will not stand like this finally,
// but contains the code bits)
// 
// 335   9/17/13 4:13p Fseide
// (removed some debugging code)
// 
// 334   9/17/13 12:35p Fseide
// (added a comment)
// 
// 333   9/16/13 8:52p Fseide
// added the design for overlapped processing in matprod_mtm() as a large
// comment
// 
// 332   9/16/13 6:22p Fseide
// matprod_mtm() with bias now using matprod_mtm() without (moved code up,
// verified, and removed old dup code)--tested
// 
// 331   9/16/13 6:10p Fseide
// matprod_mtm() with bias now uses the one without bias for the mat
// product
// 
// 330   9/16/13 6:07p Fseide
// towards unifying the two matprod_mtm() overloads
// 
// 329   9/16/13 3:34p Fseide
// disabled #define COMPACTRAINER and made it compile again
// 
// 328   9/16/13 3:20p Fseide
// disabled #define MULTICUDA and made it compile again (some code was not
// guarded)
// 
// 327   9/16/13 2:22p Fseide
// (completed re-factoring of matprod_mm())
// 
// 326   9/16/13 2:15p Fseide
// made the first matprod_mm the master copy of the matrix product, the
// second will be deleted after test
// 
// 325   9/16/13 2:14p Fseide
// refactored the two versions of matprod_mm() (about to remove code dup)
// 
// 324   9/15/13 10:28p V-haofu
// refactoring for adagrad LR adjustment of non-cuda part
// 
// 323   9/13/13 3:29p Fseide
// new methods in rbmstatevectors: gems() and settodiff()
// 
// 322   9/12/13 7:14p V-haofu
// fix a fail to build error of last change
// 
// 321   9/12/13 7:06p V-haofu
// continuing refactoring:cancle out the factor in last loop; disable the
// LRadjustment in main.cpp
// 
// 320   9/12/13 6:48p V-haofu
// continue refactoring to noncuda part: move the factor to where denomij
// is used
// 
// 319   9/12/13 6:43p V-haofu
// patch to last comment:move the Adagrad LR adjustment factor to where
// gradients and targetavdenom are used; keep cuda part unchanged.
// 
// 318   9/12/13 6:39p V-haofu
// move adagrad LRadjustment factor to where gradients are used(though
// failed to build due to lacking mbsize in interface)
// 
// 317   9/11/13 6:12p Fseide
// added comments for further tidy-up of AdaGrad
// 
// 316   9/11/13 6:00p Fseide
// (minor beautification)
// 
// 315   9/11/13 5:23p V-haofu
// fix a syntax bug for last change
// 
// 314   9/11/13 4:49p V-haofu
// refactoring of adagrad correction
// 
// 313   9/11/13 3:49p V-haofu
// delete unnecessary parentheses
// 
// 312   9/11/13 3:37p V-haofu
// add mbframes to the parameter list of function adagradientfromdenom
// 
// 311   9/11/13 11:58a V-haofu
// add formula for varG in adagrad
// 
// 310   9/11/13 10:48a Fseide
// tidied up adagradient(), removing the weird factor of sqrt(mbframes)
// (... and putting it back in where needed to maintain back compat, but
// at least now we know where it really goes, and thus where we shall
// remove it if we decide to break compat)
// 
// 309   9/10/13 7:02p Fseide
// towards understanding the weird AdaGrad scaling
// 
// 308   9/10/13 6:42p Fseide
// figured out AdaGrad compensation (assuming zero mean so far since we do
// not accumulate it)
// 
// 307   9/10/13 3:30p Fseide
// comment on AdaGrad
// 
// 306   9/10/13 3:14p Fseide
// (added a comment)
// 
// 305   9/10/13 1:56p Fseide
// toward untangling the AdaGrad implicit LR scaling
// 
// 304   9/10/13 1:48p V-haofu
// fix not compling problem caused by previous interface change
// 
// 303   9/10/13 10:15a V-haofu
// delete mbframes parameter in all accumulatesqr related functions
// 
// 302   9/09/13 5:06p V-haofu
// add comment for numframes in adagradient
// 
// 301   9/09/13 5:02p V-haofu
// us(i,j) not divided by mbsize to compute square of sum; divided it by
// mbsize where it is being used to assure functional unchanged.
// 
// 300   8/28/13 10:54a Fseide
// added an assignment operator to rbmmodelmatrixbase<> to support model
// backup
// 
// 299   8/07/13 4:46p T-paswie
// CPU version of colwisenrm
// 
// 298   7/09/13 1:27p Fseide
// added ability to compile parallelrbmmatrix.c without the MPI header and
// lib
// 
// 297   7/08/13 4:52p T-paswie
// some bugfixes to columnwise-based normalisation
// 
// 296   7/05/13 9:04p T-paswie
// colwise norms now propagated around the matrices code and used with
// maxouts
// 
// 295   6/10/13 10:09 Fseide
// (added a comment)
// 
// 294   6/08/13 15:43 Fseide
// bug fix in dleakyroot(): derivative should not get a sign change (inner
// derivative of -x undoes that);
// CPU version of leakyroot implemented (for debugging)
// 
// 293   6/07/13 21:50 Fseide
// (fixed a build break, sorry)
// 
// 292   6/07/13 21:48 Fseide
// leakyroot() and derivative implemented
// 
// 291   6/07/13 18:39 Fseide
// fixed striping issue with avsqr()
// 
// 290   6/07/13 18:15 Fseide
// new method avsqr();
// now prints diagnostics of gradient value range for relus, to check
// balance
// 
// 289   6/07/13 14:18 Fseide
// MPI mode now gets momentum right (subject to testing)
// 
// 288   6/07/13 14:07 Fseide
// implemented the MPI termination condition
// 
// 287   6/07/13 13:16 Fseide
// fixed MPI aggregate() w.r.t. 'checknotcomputing' issue (previous
// version used an accessor that was invalid while computing)
// 
// 286   6/07/13 13:08 Fseide
// dropou(): force-fail if prescale is set since current implemetnation is
// known to be incorrect
// 
// 285   6/07/13 11:20 Fseide
// new option --requirecuda to terminate the program quickly if no CUDA
// card is available;
// new option --mpi to activate MPI mode and perform sub-setting of the
// input utterances according to which part of the whole we are running
// 
// 284   6/06/13 20:20 Fseide
// added a type check to MPI aggregate();
// added missing prescale to CPU-side dropout() implementation;
// added --prescaledropout option, i.e. dropout pre-scaling should now be
// operational
// 
// 283   6/06/13 20:10 Fseide
// updatedeltas() now performs MPI-based aggregation if enabled (but no
// code yet to enable it);
// new method rbmmodelmatrixbase::mpiallreduce() to implement this
// 
// 282   6/06/13 17:57 Fseide
// mpiaggregator now has a destructor that finalizes MPI;
// mpiaggregator constructor now ensures that this is instantiated only
// once (to avoid multiple MPI_Init calls)
// 
// 281   6/06/13 17:36 Fseide
// better error handling for MPI aggregator
// 
// 280   6/06/13 17:25 Fseide
// MPI aggregate() implemented (but not tested, could be completely wrong)
// 
// 279   6/06/13 16:13 Fseide
// (minor addition to MPI aggregate())
// 
// 278   6/06/13 16:09 Fseide
// added infrastructure for MPI support, i.e. MPI libs and a first attempt
// at calling it (which compiles but failed to run due to lack of
// installed MPI environment)
// 
// 277   6/06/13 11:42 Fseide
// now passing prescaledropout flag through (but have not added actual
// cmd-line option yet)
// 
// 276   6/02/13 8:14 Fseide
// backpropagationstats() implemented for ReLU;
// mulbydlru() implemented
// 
// 275   6/02/13 7:25 Fseide
// infrastructure for storing non-linearity kind in the model file;
// removed some duplicate file-reading functions (FILE*, HANDLE) into
// function templates
// 
// 274   6/02/13 4:03 Fseide
// towards RLUs
// 
// 273   6/02/13 3:34 Fseide
// lots of code hygiene for SVD implementation, including rename 'flag'
// (in an interface!) to something meaningful and lots of formatting
// inconsistencies;
// 
// 272   4/04/13 10:12a Jianxue
// Add SVD decomposition.
// 
// 271   3/07/13 11:23a Fseide
// removed unnecessary #include of cudalattice.h
// 
// 270   1/09/13 5:10p V-hansu
// add a comment
// 
// 269   1/09/13 3:31p V-hansu
// add seterrorsignalhsmoothing() to prepare for CUDA based hsmoothing
// 
// 268   1/03/13 8:53p Kaisheny
// Asynchronous SGD using data pipe.
// 
// 267   12/17/12 1:53p Kaisheny
// Fixed CPU version for error propagation. The gradient for weight matrix
// should use the correct momentum method, which includes both the
// momentum and a scaling factor to the original gradient. In the old
// implemention, the scaling factor is not used by mistake for the
// weights, but is used for bias. The CUDA implementation is correct. 
// 
// 266   12/07/12 5:24a Adame
// convolution/maxpool support (GPU only)
// --convolutionalParams flag to support convolution parameters
// --addEnergy flag to add energy to datasets (such as HVT)
// --asyncopy flag to enable asynccopy on multi-GPU setups with pipeline
// trainer
// zero out all arrays on creation (eliminate NANs)
// 
// 265   11/27/12 6:29p V-hansu
// modify seterrorsignal() to make compile
// 
// 264   11/27/12 3:35p V-hansu
// add senone2keepmodelupdate() to seterrorsignal(), not used yet
// 
// 263   11/20/12 14:00 Fseide
// fixed int/size_t correctness in some HF functions for Win32 builds
// 
// 262   11/17/12 4:31p Fseide
// columnnormsquares() is now 'const' as it should be
// 
// 261   11/17/12 3:55p Fseide
// unseen-state compensation now covers 'a' as well
// 
// 260   11/16/12 5:37p Fseide
// columnnormsquares() and scaledcolumns() now expect the weight vector to
// be a row (single-row matrix)
// 
// 259   11/16/12 4:49p Fseide
// fixed a comment
// 
// 258   11/16/12 9:01a Fseide
// fixed a comment
// 
// 257   11/16/12 8:59a Fseide
// minor correction of scaledcolumns()
// 
// 256   11/16/12 3:53p V-hansu
// modify scaledcolumns() to make compiles, change back M to const once
// col() supports
// 
// 255   11/16/12 8:51a Fseide
// new method addmatprod_mtm()
// 
// 254   11/16/12 8:15a Fseide
// new method scaledcolumns()
// 
// 253   11/16/12 8:03a Fseide
// dotprodwrtcols() renamed to columnnormsquares(), minor edits
// 
// 252   11/16/12 2:43p V-hansu
// add method dotprodwrtcols()
// 
// 251   11/10/12 5:14p V-hansu
// change subtract() to addweightedrow()
// 
// 250   11/10/12 3:27p V-hansu
// modify glimpse() to make syncfromcuda() able to use
// 
// 249   11/10/12 1:49a V-hansu
// modify substract() to make it work for unseenstatecompensation()
// 
// 248   11/09/12 11:51p V-hansu
// add substract() similar to addweight()
// 
// 247   11/08/12 4:30p T-simonw
// add rbmmatrixaccumulator for accumulating statistics in double
// precision
// 
// 246   11/02/12 4:44p T-simonw
// catch exceptions thrown in unit-test
// 
// 245   11/02/12 4:32p T-simonw
// code formatting and documentation
// 
// 244   11/02/12 1:09p Fseide
// added comments after reverse-engineering my own AdaGrad implementation
// 
// 243   10/31/12 4:27p T-simonw
// added documentation
// setdiagonalpreconditioner, elementwisedivision: implemented NUMA mode
// added unit tests
// 
// 242   10/31/12 1:41p T-simonw
// replace the setvalue method, because the
// previous setvalue method only set a value if the matrix had exactly one
// colunmn
// the previous setvalue method has not been used anywhere
// 
// 241   10/31/12 10:30a T-simonw
// change cudamode to non-const boolean and add method setcudamode in
// order to allow for unit testing
// addweighted gets thisscale, also in NUMA mode
// add setsquare method
// add scale method
// add dot product method
// add setvalueincuda (because setvalue is NOT doing what the name
// suggests!)
// add Hessian free optimization method: sethessianvectorsignal,
// setdiagonalpreconditioner
// add thisweight to matprod_mtm
// add some very basic unit tests
// 
// 240   10/17/12 5:00p Dongyu
// fixed several copy&paste errors in the dropout code. 
// 
// 239   10/16/12 11:39a Fseide
// new methods dropout() and scale() to support Hinton's drop-out method
// 
// 238   10/12/12 1:48p Dongyu
// added support of dropout training for DNN (frame level training only). 
// addes support to convert the model based on dropout rate used in the
// training and/or senone sections used in multilingual training.
// 
// 237   10/10/12 10:02a Dongyu
// added support to train models that shares the same hidden layers but
// use different senone sets from different langauges. This allows us to
// train universal ASR with separate senonoes or use models trained using
// multiple languages to adapt to new langauges.
// 
// 236   10/09/12 6:44p Fseide
// changed rand() to ::rand() because there was some conflict
// 
// 235   9/27/12 12:30a V-hansu
// change setzero into setvalue
// 
// 234   9/25/12 11:57a Fseide
// (added some commented-out debug code)
// 
// 233   9/24/12 3:26p Fseide
// no longer passing numsummands to CUDA adadenom()
// 
// 232   9/24/12 3:00p Fseide
// AdaGrad adjustment now clipped to 10 x against the average of the
// respective parameter matrix/vector, only afterwards is it scaled to the
// user-specified target. This is to prevent clipping if the dynamics
// change.
// 
// 231   9/23/12 5:35p Fseide
// AdaGrad now displays the actual avdenom for diagnostics, although we
// hand-fix it
// 
// 230   9/21/12 5:49p T-simonw
// 
// 229   9/21/12 3:24p Fseide
// added nosoftmax mode, to speed up sequence training by bypassing the
// unnecessary expensive softmax() computation
// 
// 228   9/21/12 2:48p Fseide
// new method rbmstatevectorsrefbase::assign() that copies data directly
// to the target (CPU or GPU)
// 
// 227   9/21/12 8:10a Fseide
// new method msra::cuda::numcudadevices() inside cudamatrix.h, which
// determines the # devices but does not crash if CUDA DLL missing
// (returning 0 instead), this was factored out from
// msra::dbn::numcudadevices() so we can share it with lattice code;
// parallelstate() constructor now uses proper numcudadevices() function
// to determine whether CUDA is available (before just assumed it is,
// which was an early hack)
// 
// 226   9/20/12 5:42p Fseide
// (fixed TABs)
// 
// 225   9/20/12 4:38p Fseide
// AdaGrad now enforces to use the same av for the biases as for the W
// matrix (we basically ignore the bias in the avg)
// 
// 224   9/18/12 11:17a Fseide
// adagradient() now fully CUDA-based
// 
// 223   9/18/12 11:07a Fseide
// first step towards adagradient() on CUDA
// 
// 222   9/18/12 10:03a Fseide
// switched accumulatesqr() to CUDA mode
// 
// 221   9/18/12 10:02a Fseide
// changed AdaGrad to use a forgetting factor (time constant 2 hours of
// data)
// 
// 220   9/17/12 6:18p Fseide
// accumulatesqr() now available as CUDA code (but not yet tested)
// 
// 219   9/17/12 6:09p Fseide
// simplified some of the AdaGrad code (removed one function call) in prep
// for move to CUDA
// 
// 218   9/17/12 3:30p Fseide
// more steps towards AdaGrad
// 
// 217   9/16/12 5:43p Fseide
// new methods setzero(), accumulatesqr(), and adagradient()
// 
// 216   9/02/12 5:06p Fseide
// addweighted() now takes a 'thisscale' parameter, in prep for L2
// regularization
// 
// 215   8/31/12 9:56p F-gli
// added check if under cudamode when syncfromcuda, added const to
// caladagradweight()
// 
// 214   8/31/12 9:34p F-gli
// changed adagrad code according to Frank's comments
// 
// 213   8/31/12 4:57p F-gli
// checked in temp code about adagrad
// 
// 212   8/15/12 10:17a V-hansu
// change some indentation
// 
// 211   8/06/12 20:51 Fseide
// samplebinary() now works in-place
// 
// 210   7/23/12 10:57a V-hansu
// modify setblockdiagonal function to make it compatible with second top
// layer adaptation
// 
// 209   7/17/12 5:32p Adame
// Update for no-sync framework
// async copy fixes
// 
// 208   7/11/12 7:31p V-hansu
// modify setblockdiagonal to make it compatible with round up mode of
// adaptation
// 
// 207   7/06/12 9:15p V-hansu
// modify  setblockdiagonal to make it compatible with new adaptation
// method
// 
// 206   6/30/12 2:28p V-hansu
// add a function to get colstride
// 
// 205   6/08/12 9:32p V-xieche
// delete code related to delayupdate.
// 
// 204   5/27/12 3:37p V-xieche
// modify the funciton onedevicedim for striped mode, consider the
// situation when number of toplayer's nodes divide cuda devices used on
// toplayer is not an integer.
// 
// 203   5/15/12 8:40p V-xieche
// enable code defined by COMPACTTRAINER, MULTICUDA, OPTPIPELINETRAIN and
// STRIPEDTOPLAYER to make the code has all functions defined for pipeline
// training. remove these MACRO later.
// 
// 202   5/10/12 6:46p V-xieche
// add code to make the MACRO MULTICUDA be compatible plain BP training as
// well.
// 
// 200   5/08/12 9:49p V-xieche
// Add macro SIMPLIFYCODE used for cleanup the code.
// 
// 199   4/18/12 4:37p V-xieche
// clean up the code related to macro DELAYUPDATE
// 
// 198   4/18/12 4:01p V-xieche
// clean up all code related to target propagation.
// 
// 197   4/18/12 2:06p V-xieche
// clean up the code in block of #ifdef TARGETBP #endif 
// 
// 196   4/11/12 5:28p V-xieche
// add DBNFASTTRAIN_NORMALBPTRAIN and DBNFASTTRAIN_PIPELINETRAIN, to debug
// the normal BP training and pipeline training for class dbnfasttrain in
// dbnfasttrain.h
// 
// 195   4/10/12 7:17p V-xieche
// add a temp macro DBNFASTTRAIN, to test the new class in dbnfasttrain.h.
// need to delete it after verification.
// 
// 194   4/08/12 9:29p V-xieche
// modify code, use function instead of previous pointer.
// 
// 193   4/06/12 6:26p V-xieche
// Add codes for posteriorstats function for striped top layer. not
// finished yet.
// 
// 192   4/05/12 9:51p V-xieche
// add code for accumulate prior and posteriorstats in striped toplayer
// pipeline training. not finished yet.
// 
// 191   4/03/12 8:41p V-xieche
// check in all the code for pipeline training. stripe top layer on two
// devices. need to add comments and adjust the code make it easy to read.
//
// 190   4/01/12 2:10p Fseide
// adapted for new i0 argument to set error signal
// 
// 189   3/27/12 2:18a V-xieche
// delete a function don't use any more.
// 
// 188   3/27/12 1:18a V-xieche
// add code for pipeline training with multi cuda devices. Need to add
// comments later. 
// 
// 187   3/16/12 2:11a V-xieche
// use fetch function to get thereference from cuda, the time used for
// training in compacttrainer is correct now.
// 
// 186   3/14/12 12:45a V-xieche
// add the MACRO to output time stats in compact trainer.
// 
// 185   3/11/12 7:05p V-xieche
// add code for a compact trainer. make it run in CUDA directly.
// 
// 184   3/08/12 10:33p V-xieche
// add code to make forward and backward prop do in CUDA directly.
// verified the training is correct, while speed faster than previous.
// need to debug it.
// 
// 183   3/05/12 9:10p V-xieche
// Add code for compact trainer to simplify the implmentation of DNN
// trainer(MACRO COMPACTTRAINER), to make it purely on CUDA and prepare
// for pipeline training in multiply CUDA device.
// 
// 182   2/26/12 8:45p V-xieche
// Add macro COPYINCUDA_FORDELAYUPDATE_V2, copy all data in CUDA device
// directly now.
// 
// 181   2/26/12 6:57p V-xieche
// Add codes for copy date between CUDA.
// 
// 180   2/25/12 5:23p V-xieche
// modify code for copy data in CUDA device. not completed.
// 
// 179   2/24/12 11:16p V-xieche
// Add code to assign value in CUDA directly for delayupdate training. not
// finished yet.
// 
// 178   2/23/12 5:47p V-xieche
// fix bugs exist in previous code for delay update mode.
// 
// 177   1/05/12 7:34p Fseide
// (editorial)
// 
// 176   1/04/12 5:41p Fseide
// bug fix in dropframes(), now uses correct lock function
// 
// 175   1/04/12 4:59p Fseide
// new method dropframes()
// 
// 174   12/20/11 3:14p Dongyu
// move KhatriRaoProduct and reshapecolumnproduct to class
// rbmstatevectorsbase
// 
// 173   12/09/11 2:02p F-gli
// add comments
// 
// 172   12/09/11 2:01p F-gli
// share cudamatrix.h to current folder because latgen,
// TranscriptorService does not have cudamatrix project
// 
// 171   12/07/11 4:26p Dongyu
// fixed stripping errors in reshapecolumnproduct
// 
// 170   12/06/11 5:44p Dongyu
// fixed bugs in reshapecolumnproduct
// 
// 169   11/28/11 5:56p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 168   11/23/11 4:33p Dongyu
// add reshape and KhatriRaoProduct
// 
// 167   11/16/11 11:55p V-xieche
// add macro DELAYUPDATE_V2_SWAPTEST for swap on delay update model.
// 
// 166   11/15/11 8:46p V-xieche
// add swap function for acceleratedmatrixbase class. Swap date both in
// CPU and CUDA if at cudamode. need to test it.
// 
// 165   11/14/11 4:01p V-xieche
// add micro LOADSTEEPERSIGMOIDMODEL for training the model from
// intermediate model for steeper or flatter sigmoid model.
// 
// 164   11/05/11 8:11p V-xieche
// add code for delay update model in code block DELAYUPDATE_V2
// 
// 163   11/04/11 16:27 Fseide
// scaleandaddallcols() now implements 'otherweight' for CUDA mode (but
// not yet NUMA mode)
// 
// 162   11/04/11 14:22 Fseide
// (incorrect comment fixed)
// 
// 161   10/31/11 9:01p V-xieche
// add code for simple experiment of delay update models.
// 
// 160   10/28/11 15:37 Fseide
// (minor fix)
// 
// 159   10/28/11 15:35 Fseide
// towards allowing scaled update to parameters, for better handling of
// momentum
// 
// 158   10/28/11 8:23 Fseide
// fixed another embarrassing bug in the efficiency "fix" for
// scaleandaddmatprod_numa()
// 
// 157   10/27/11 19:35 Fseide
// fixed incorrect 'fix' of scaleandaddmatprod_numa() NUMA inefficiency
// 
// 156   10/26/11 11:24a Dongyu
// removed debugging code for regularized adaptation.
// 
// 155   10/25/11 5:18p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 154   10/18/11 9:07p V-xieche
// modify the code to implement a true steeper or flat sigmoid function.
// i.e. scale the bias as well
// 
// 153   10/11/11 3:34p V-xieche
// undefine SPARSENESSOUTPUTOFHIDDENLAYER. fix a minor argument.
// 
// 152   10/11/11 3:22p V-xieche
// modify the code for setting output of hidden layer below specific value
// to zero.
// 
// 151   10/11/11 12:09p V-xieche
// fix a minor bug for sparse experiment of hidden layer output
// 
// 150   10/11/11 11:38a V-xieche
// add code for setto0ifbelow for the output of hidden layer.
// 
// 149   10/11/11 8:24 Fseide
// fixed a compiler warning
// 
// 148   10/08/11 14:36 Fseide
// enabled NUMA fix
// 
// 147   10/08/11 10:22 Fseide
// fixed inefficieny of scaleandaddmatprod_numa() when not actually
// running in NUMA mode (to be tested);
// new special-purpose method peek()
// 
// 146   10/06/11 5:17p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 145   10/03/11 14:35 Fseide
// (fixed a compiler warning)
// 
// 144   9/29/11 10:05p V-xieche
// Add CLUSTERSTATE definition to cluster state by monophone.
// 
// 143   9/28/11 10:05p V-xieche
// 
// 142   9/26/11 8:43p V-xieche
// Add some codes for log(sigmoid + epison) experiment.
// 
// 141   9/22/11 9:24p V-xieche
// fix a bug for deeper sigmoid experiment.
// 
// 140   9/20/11 2:46p V-xieche
// fix a minor bug for steeper sigmoid experiment
// 
// 139   9/19/11 10:54p V-xieche
// delete some debug code.
// 
// 138   9/19/11 10:49p V-xieche
// get and set weight matrix for temp experiment.
// 
// 137   9/17/11 12:27p F-gli
// comment out one useless line to make it build
// 
// 136   8/24/11 9:06p V-xieche
// add some log infomation for adding margin term.
// 
// 135   8/23/11 7:57p V-xieche
// add margin-based training code for dbn according to Heigold's thesis.
// 
// 134   8/22/11 4:06p V-xieche
// add some code for target propagation v5
// 
// 133   8/21/11 4:57p V-xieche
// add some code for target propagation version 5, it try to modify the
// weight according the normal BP algorithm, to see whether it works.
// 
// 132   8/18/11 11:03p V-xieche
// add some comment and log information.
// 
// 131   8/17/11 10:29p V-xieche
// add some code for targetpropagation version 3 to use label as target
// feature
// 
// 130   8/16/11 10:36p V-xieche
// fix a minor bug
// 
// 129   8/16/11 10:34p V-xieche
// add target propagation version 4 code. which used to verify the valid
// of target propagation. It do the same thing as the normal
// backpropagation for 2kx1 model only updating the bottom layer.
// 
// 128   8/15/11 10:58p V-xieche
// fix a minor bug and add some code for statistic the correct ratio for
// weight vector lies in decision region experiment
// 
// 127   8/15/11 10:30p V-xieche
// add code to statistic the ratio the top layer weight matrix lies in
// their class decision region
// 
// 126   8/13/11 9:42p V-xieche
// fix a minor bug
// 
// 125   8/13/11 5:29p V-xieche
// Add target propagationv3 function for the experiment of B=M*h
// experiment, which considerate all b, not only the label class.
// 
// 124   8/02/11 10:48p V-xieche
// correct a commant
// 
// 123   8/02/11 12:49a V-xieche
// undefine TARGETBP for a check-in code
// 
// 122   8/02/11 12:47a V-xieche
// must add the minus direction, i.e times a -2
// 
// 121   8/02/11 12:31a V-xieche
// add function settargetbpv2errorsignal() to implement targetpropagation
// version2. b=w*h
// 
// 120   7/29/11 5:58p V-xieche
// add some debug code to verify the target feature is correct. add code
// to verify target propagation could decrease the square error when
// updating bottom layer
// 
// 119   7/29/11 12:07a V-xieche
// add  err.lockforreading and unlock function and modify 2 to -2
// according to formulas.
// 
// 118   7/28/11 8:38p V-xieche
// add updatetargetfeatstats function for get and update target feature
// 
// 117   7/27/11 9:23p V-xieche
// Add the code for target propagation(to be debugged). in the #ifdef
// TARGEBP #endif block. 
// 
// 116   7/26/11 8:51a V-xieche
// fix a bug in bianry function
// 
// 115   7/25/11 9:54p V-xieche
// fix a bug in binary function
// 
// 114   7/25/11 8:45p V-xieche
// fix the bug in binarize function and rename the original function to
// binarize
// 
// 113   7/25/11 1:15p V-xieche
// Modify the setvalue function considering CUDA mode
// 
// 112   7/25/11 10:15a V-xieche
// Modify quantization and setvalue with considerate the cuda model, add
// lockforreadwrite and unlock funtion in them.
// 
// 111   7/23/11 5:52p V-xieche
// Add a function for set a to a fixed value. Often for set bias to 0 for
// experiment purpose
// 
// 110   7/15/11 11:38 Fseide
// added a comment on potential perf improvement
// 
// 109   7/13/11 19:02 Fseide
// new method matprod_col_mtm() for supporting on-demand LL evaluation
// 
// 108   7/07/11 8:46a V-xieche
// Modify a bug exists in the first check in code
// 
// 107   7/06/11 11:29p V-xieche
// add some code in the #if 0 #endif block for the histgoram stats.
// 
// 106   6/30/11 1:46p V-xieche
// Modeify a bug for the Nopooled-Diag matrix adaption in the setdiagblock
// function
// 
// 105   6/23/11 10:49a V-xieche
// delete some unneccessary bracket in the setblockdiagonal function.
// also add an else to throw exception if it is not a square matrix or an
// array.
// 
// 104   6/22/11 5:11p V-xieche
// modify the setblockdiagonal function also support the array for the
// pool of a.
// 
// 103   6/21/11 9:25p V-xieche
// set poolblocks to be true for generate the pooled-diag matrix
// 
// 102   6/21/11 13:46 Fseide
// first step towards CUDA implementation of setblockdiagonal
// 
// 101   6/21/11 1:20p V-xieche
// Modify the part of pooled-diagonal matrix caulation in setblockdiagonal
// function, make it more general
// 
// 100   6/21/11 9:52a V-xieche
// modify the segblockdiagonal function avoiding some bound check and some
// additional  memory load.
// 
// 99    6/21/11 7:56 Fseide
// (added TODOs, fixed TAB/indentation)
// 
// 98    6/20/11 10:18p V-xieche
// implement the function setblockdiagonal for the diagonal matrix
// adaptation and pooled diagonal matrix adaptation.
// 
// 97    6/20/11 7:43 Fseide
// new method setblockdiagonal()
// 
// 96    6/10/11 15:46 Fseide
// (fixed a spelling error in a log message)
// 
// 95    6/10/11 8:04 Fseide
// (fixed a few compiler warnings about unused function arguments)
// 
// 94    5/17/11 1:57p Fseide
// (minor edit to disabled pruning mode in seterrorsignal())
// 
// 93    4/22/11 10:14 Fseide
// added experimental pruning to seterrorsignal()
// 
// 92    3/14/11 11:15 Fseide
// documented seterrorsignal()
// 
// 91    3/05/11 8:29p Fseide
// added a 'const' modifier to write()
// 
// 90    3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 89    2/26/11 6:03p Fseide
// minor fix in entercomputation(), now again working in CPU-only mode
// 
// 88    2/26/11 4:57p Fseide
// moved softmax() to GPU--reduces runtime by 1/3
// 
// 87    2/26/11 4:12p Fseide
// transited BP functions also to multi-GPU mode
// 
// 86    2/25/11 9:38p Fseide
// updated sumacrossdevices() to allow for parallel data transfer from
// different devices
// 
// 85    2/25/11 7:51p Fseide
// added explicit synchronization control to syncfromcuda()/synctocuda()
// 
// 84    2/25/11 6:04p Fseide
// changed synchronization--assign() and fetch() are now bulk-launched
// across devices, and synchronized after all have been kicked off, to
// allow for full parallelization (we don't know if it actually does it,
// though)
// 
// 83    2/25/11 5:41p Fseide
// bug fix in sumacrossdevices()
// 
// 82    2/25/11 10:03a Fseide
// sumacrossdevices() implemented, but still stuck due to gems()
// implementation
// 
// 81    2/24/11 11:16p Fseide
// (fixed a warning)
// 
// 80    2/24/11 11:15p Fseide
// (minor change of default 'simulateddevices' so we can test on the
// dual-Tesla machine before we really support 2 devices)
// 
// 79    2/24/11 11:00p Fseide
// (minor change to multi-device simulation)
// 
// 78    2/24/11 10:07p Fseide
// debugged and fixed syncto/fromcuda();
// added debugging facility to fake 2 cards (temporarily enabled)
// 
// 77    2/24/11 8:07p Fseide
// llstats() now parallelizes across multiple devices--completed
// parallelization of pretraining (except for sumacrossdevices())
// 
// 76    2/24/11 7:57p Fseide
// bug fix: cudadistributedmatrix::validstriping must be a reference to
// the master copy, it now is;
// new method sumacrossdevices() to reduce split matrix products
// 
// 75    2/24/11 6:06p Fseide
// few more steps towards multi-CUDA on the way
// 
// 74    2/24/11 3:00p Fseide
// added the on-the-fly view, but 'validstriping' not yet correctly
// supported in syncing--should we?
// 
// 73    2/24/11 9:33a Fseide
// removed viewedstriping in favor of an on-the-fly view
// 
// 72    2/23/11 6:08p Fseide
// steps towards multiple views for state vectors, still kind of messy
// 
// 71    2/23/11 4:04p Fseide
// misc. first-round bug fixes discovered during stepping-through;
// entercomputation() now sets the striping mode;
// checkcudastripingmode() now actually just checks rather than lazy
// initialization since that is no longer necessary
// 
// 70    2/23/11 1:51p Fseide
// finished change to cudadistributedmatrix, except for actual usage in
// math, where a compat mode was added for now
// 
// 69    2/19/11 16:52 Fseide
// baby coming--need to check in
// 
// 68    2/19/11 16:47 Fseide
// infrastructure for striped CUDA laid, but not completed yet as now all
// operations need to be updated (currently not compiling)
// 
// 67    2/17/11 18:13 Fseide
// now actually moves data to multiple GPU devices (but not used there
// yet)--not tested, could fail with assertions for old mode
// 
// 66    2/17/11 14:52 Fseide
// added more code towards multiple devices, not called yet
// 
// 65    2/16/11 16:49 Fseide
// (towards multi-GPUs)
// 
// 64    2/16/11 15:18 Fseide
// added design notes for multi-CUDA version
// 
// 63    2/15/11 16:28 Fseide
// rbmstatevectors() failed in non-CUDA mode;
// new method cudaptr() which returns a NULL in non-CUDA mode
// 
// 62    2/15/11 15:28 Fseide
// hascuda() now catches a DLL-load exception when calling
// getnumdevices(), i.e. we can run (in NUMA mode) if cudamatrix.dll or
// the CUDA DLLs are missing
// 
// 61    2/10/11 3:21p Fseide
// (fixed a variable spelling error in an assertion)
// 
// 60    2/10/11 1:54p Fseide
// switched to CUDA mode
// 
// 59    2/10/11 1:13p Fseide
// posteriorstats() change of logic
// 
// 58    2/10/11 12:59p Fseide
// posteriorstats() now uses vector mode, ready for CUDA version
// 
// 57    2/10/11 12:37p Fseide
// posteriorstats() factored into rbmstatevectorsref
// 
// 56    2/10/11 11:33a Fseide
// mulbydsigm() switched over to CUDA  --cuts over 15% runtime of
// backpropagationstats()
// 
// 55    2/10/11 11:18a Fseide
// seterrorsignal() switched to CUDA implementation
// 
// 54    2/10/11 10:53a Fseide
// seterrorsignal() working now
// 
// 53    2/10/11 10:32a Fseide
// moved error-signal computation to rbmstatevectors, for future CUDA
// implementation
// 
// 52    2/10/11 10:01a Fseide
// fetch() function gone;
// one unnecessary argument from scaleandaddallcols() gone;
// new method accumulate()
// 
// 51    2/09/11 10:10p Fseide
// added some #if-0'ed out experimental code (add +0.1 to the derivative)
// 
// 50    2/09/11 12:23a Fseide
// added some test code, but #if-ed out
// 
// 49    2/08/11 9:40p Fseide
// acceleratedmatrixbase::cudamatrix made private;
// acceleratedmatrixbase::operator msra::cuda::matrix*() changed to
// forcuda() & dealt with fallout;
// cachedmatrixbase no longer knows CUDA (getting ready for being folded
// into acceleratedmatrixbase)
// 
// 48    2/08/11 8:37p Fseide
// bug fix: alloccuda() should not not do anything if empty (that was an
// outdated condition)
// 
// 47    2/08/11 4:23p Fseide
// moved three resizeonce() calls from updatedeltas() to inside their NUMA
// counterparts (they are not used in CUDA, so no need to allocate them)
// 
// 46    2/08/11 2:50p Fseide
// made checkcudadims() const
// 
// 45    2/08/11 2:20p Fseide
// added code to verify runtime dimensions of CUDA matrix
// 
// 44    2/07/11 9:53p Fseide
// llstats() now uses CUDA implementation
// 
// 43    2/07/11 9:31p Fseide
// moved llstats() into rbmstatevectorsref, to allow acceleration by CUDA
// 
// 42    2/07/11 7:08p Fseide
// matprod_mt?m() now uses addtoallcolumns()
// 
// 41    2/07/11 6:52p Fseide
// samplebinary() now implemented in CUDA
// 
// 40    2/07/11 6:32p Fseide
// (moved up samplebinary())
// 
// 39    2/07/11 6:28p Fseide
// sigmoid() and addrowsum() now in CUDA
// 
// 38    2/07/11 5:24p Fseide
// rbmstatevector now going live in CUDA mode --data living in CUDA space;
// rbmmodelmatrixbase now implements operations in CUDA (so far not faster
// because too much happening on CPU side still)
// 
// 37    2/07/11 4:36p Fseide
// (removed two unused functions)
// 
// 36    2/07/11 4:29p Fseide
// hasnan() now requires locked mode
// 
// 35    2/07/11 4:11p Fseide
// bug fix: stripe() now longer returns a && but just the object itself
// (because it was just created inside);
// lock state changed to independent read and write state;
// lock state implemented in all rbmstatevectorsref functions, so they
// theoreticallt do work now even in case of CUDA
// 
// 34    2/07/11 3:25p Fseide
// moved the whole locking business from rbmstatevectorsbase to
// rbmstatevectorsrefbase
// 
// 33    2/07/11 2:26p Fseide
// rbmstatevectorsrefbase now derived from acceleratedmatrixbase, to
// reduce code duplication in managing CUDA stuff
// 
// 32    2/07/11 2:01p Fseide
// acceleratedmatrixbase moved down w.r.t. its template argument, which
// can now be matrix (with allocation) or matrixstriperef (no allocation)
// 
// 31    2/07/11 1:51p Fseide
// new method rbmstatevectors::stripe(), which required some reorganizing
// of things
// 
// 30    2/06/11 3:23p Fseide
// (minor cleanup)
// 
// 29    2/05/11 9:26p Fseide
// moved 'computing' state from acceleratedmatrixbase to derived class
// rbmmodelmatrixbase
// 
// 28    2/05/11 8:24p Fseide
// added mechanism for "locking" for direct access to the CPU-side
// rbmstatevectorsbase matrix, which takes care of moving from/to CUDA RAM
// 
// 27    2/05/11 7:00p Fseide
// factored out syncto/fromcuda();
// moved mulbydsigm() and samplebinary() to rbmstatevectorsref
// 
// 26    2/03/11 9:33p Fseide
// entercomputation() and fetch() now no longer fail if the matrix is
// empty (used to hit an assertion in matrix(i,j))
// 
// 25    2/02/11 11:29a Fseide
// added comments for next steps of CUDA transition
// 
// 24    2/02/11 11:23a Fseide
// moved sigmoid() and softmax() to here from original matrixbase class
// (which is now a mere typedef)
// 
// 23    2/02/11 11:14a Fseide
// rbmmodelmatrixbase now takes all state inputs as rbmmodelvectors;
// rbmmodelvectorsref fixed w.r.t. types for down-stream calls
// 
// 22    2/02/11 10:49a Fseide
// moved rbmstatevector before rbmmodelmatrix because the latter takes
// inputs of the type of the former
// 
// 21    2/02/11 10:47a Fseide
// added compat stub for hasnan()
// 
// 20    2/02/11 10:43a Fseide
// added some feed-through functions, to be replaced by abstracting the
// functions that call them here
// 
// 19    2/02/11 10:24a Fseide
// dummy implementations of rbmstatevectorsbase and rbmstatevectorsrefbase
// 
// 18    2/02/11 9:22a Fseide
// started new class rbmstatevectorsbase
// 
// 17    2/02/11 8:57a Fseide
// (added a comment)
// 
// 16    2/02/11 8:55a Fseide
// split rbmmodelmatrixbase out from acceleratedmatrixbase (we will later
// also have an rbmstatematrixbase)
// 
// 15    2/02/11 8:22a Fseide
// pushed some math ops on updatedeltas() down to acceleratedmatrix, for
// further CUDA optimization
// 
// 14    2/01/11 7:11p Fseide
// replaced addition of biases by a CPU-side function, because it leads to
// more accurate results (??)
// 
// 13    2/01/11 4:57p Fseide
// make_ones() compiles now--time to test!
// 
// 12    2/01/11 4:54p Fseide
// replaced addcol() by a dyadic matrix product--because cublas cannot do
// otherwise
// 
// 11    2/01/11 15:32 Fseide
// new CUDA method addcol for column-wise addition (to add bias)
// 
// 10    2/01/11 15:28 Fseide
// cuda version implemented
// 
// 9     2/01/11 15:00 Fseide
// matprod_m*m() functions now take one additional cache object for moving
// data to/from CUDA
// 
// 8     2/01/11 14:57 Fseide
// added stub if statements for cuda mode
// 
// 7     2/01/11 11:49a Fseide
// reactivated acceleratedmatrixbase::operator= (const) for use during
// computation state in updatedeltas()
// 
// 6     1/30/11 11:45p Fseide
// renamed numdevices() to getnumdevices()
// 
// 5     1/30/11 19:01 Fseide
// first steps towards CUDA mode--detect CUDA
// 
// 4     1/30/11 17:53 Fseide
// added #include "cudamatrix.h"
// 
// 3     1/30/11 16:37 Fseide
// added missing #pragma once
// 
// 2     1/30/11 16:33 Fseide
// acceleratedmatrixbase and cachedmatrixbase moved to parallelrbmmatrix.h
// 
// 1     1/30/11 16:30 Fseide
// parallelrbmmatrix.h added

#pragma once

#include "numahelpers.h"
#include "pplhelpers.h"
#include "cudamatrix.h"		// note: need to share cudamatrix.h to current folder because latgen, TranscriptorService does not have cudamatrix project
#include "matrixquantization.h"
#include <array>

// #define STEEPERSIGMOID // using a more steeper sigmoid function in hidden layer. [v-xieche]
// #define LOADSTEEPERSIGMOIDMODEL // continued to train steeper or flatter model, need to divide the scale before training when loading model.
// #define UPDATEWEIGHTFORSPSM  // when using steepersigmoid, updated weight matrix in hidden layer as well. i.e. multiply the scale on hidden layer.[v-xieche]
//#define SCALEBIASFORSS       // also add a scale on the bias in the hidden layer when using a steeper and flat sigmoid.[v-xieche]
//#define AMPNUM 0.7     // the scale of the sigmoid function [v-xieche]
// #define LOGINSIGMOID  // the log in output of sigmoid function in hidden layer. i.e. log (sigmoid(z) + epison). [v-xieche]
// #define EPISONFORLOG  0.5 // the epison used for log function, avoid of numeric problem.[v-xieche]
// #define CLUSTERSTATE   // cluster the monophone as a class, to analysis the histogram table. only used in mltrain model now[v-xieche]
// #define SPARSENESSOUTPUTOFHIDDENLAYER   // test the experiment of sparseness of output of hidden layer. [v-xieche]

// #define COMPACTTRAINER       //for a compact DNN trainer and for fast and pure train on CUDA. [v-xieche]
// #define PIPELINETRAIN        // implement the code for pipeline training. [v-xieche]
// #define MULTICUDA            // for multi cuda device and pipeline trianing on them. [v-xieche]
// #define OPTPIPELINETRAIN     // for top layer, Forward, Backward then Update. For other layers, Backward, Update, then Forward. [v-xieche]
// #define STRIPEDTOPLAYER      // striped top layer.[v-xieche]
// #define TIMESTATS            //statistical the time distribution in each part of each layer. [v-xieche]
// #define DEBUGINFO_PIPELINETRAIN // output the debug infomation for pipeline training [v-xieche]

static int do_svd (std::vector<std::vector<float>>&, int, int, std::vector<float>&, std::vector<std::vector<float>>&);

namespace msra { namespace dbn {

// ---------------------------------------------------------------------------
// class cudadistributedmatrix
// ---------------------------------------------------------------------------

// helper to get number of CUDA devices in a cached fashion
static size_t numcudadevices()
{
    static size_t cudadevices = SIZE_MAX;    // SIZE_MAX = unknown yet
    if (cudadevices == SIZE_MAX)
    {
        cudadevices = msra::cuda::numcudadevices();
        if (cudadevices == 0)
            fprintf (stderr, "numcudadevices: NUMA mode (no CUDA device found)\n");
        else
            fprintf (stderr, "numcudadevices: CUDA mode (%d CUDA devices found)\n", cudadevices);
#if 0       // for debugging we can pretend to have more (cuda lib will use mod operator to map to actual devices)
        const int simulatedevices = (cudadevices == 2) ? 1 : 2;	// 2 won't run; use 2 to debug on our 1-GPU machine
        if (simulatedevices != cudadevices)
        {
            fprintf (stderr, "numcudadevices: simulating %d devices instead of the real %d ones\n", simulatedevices, cudadevices);
            cudadevices = simulatedevices;
        }
#endif
    }
    return cudadevices;
}

static bool hascuda() { return numcudadevices() > 0; }

// notes on parallelization across N CUDA cards
//
// Key matrix operations (v and h are frames stacked into matrix columns):
//  - h = W' v + a  |  sigmoid          // forwardprop()
//  - v = W h + b   |  sigmoid          // CD, backpropagationstats()
//  - dW += v * h'                      // updatedeltas()
//
// Striping assumption:
//  - N vertical stripes of W (that's N horiontal stripes of W')
//  - N horizontal stripes of a
//  - N horizontal stripes of b
//  - accordingly for dW, da, db
//
// Consequence:
//  - v is needed full-copy format
//  - h is needed horizontal stripes
//  - as things propagate, h becomes v and vice versa, potentially requiring conversion
//
// Operation: h = W' v + a  |  sigmoid  // forwardprop(), i.e. used everywhere
//  - input: full copy of v       --N times distribution overhead
//     - full copy needed after filling from host
//     - at end of CD, v is in horizontal stripes
//  - on each card compute horizontal stripe of (W'v+a) -> horizontal stripe of h
//  - apply sigmoid to horizontal stripes of h in-place   --may need to push down sigmoid through interface
//  - output: horizontal stripes of h
//
// Operation: v = W h + b  |  sigmoid   // pre-training, BP
//  - input: horizontal stripes of h            --no overhead
//  - compute N partial results of W h
//  - aggregate partial results and b through binary merge
//     - move half of data to other half of GPUs
//     - half goes from upper to lower, other half goes from lower to upper
//     - then merge (fully N-way parallel)
//     - in last merge add b (CD version only)
//     - now v is in horizontal stripes
//     - total moving cost: (N/2 reads + N/2 writes) * log N * size of v   --cheaper than distribution of v
//  - then take sigmoid
//  - output: horizontal stripes of v
//  - conversion:
//  - htov version (CD; has b) -> need to convert to full copy for forwardprop() and updatedeltas()
//  - ehtoev version (BP; no b) -> this is already in the format needed for next layer of BP
//
// Operation: dW += v * h'              // updatedeltas()
//  - input: full copy of v
//  - input: horizontal stripes of h
//  - compute vertical stripes of dW, keep them separate
//  - both inputs happen to be in correct format in all use cases (v is h. stripes in CD but turned to copies for forwardprop())
//
// Additional operations:
//  - sigmoid
//     - seems always on horizontal stripes, i.e. no overhead
//  - random sampling of h
//     - input will be in horizontal stripes
//     - required right before W h, i.e. in horizontal stripes
//     - causes issue with rand() (not compatible)
//  - mulbydsigm
//     - operates on horizontal stripes of h and eh   --no overhead
//  - scaleandaddallcols  (in updatedeltas())
//     - operates on da and db
//     - horizontal stripes are suitable, would need h and v in h. stripes
//        - v will be in full copy, which is a superset of stripes
//  - softmax   --to be parallelized as well
//     - would need vertical stripes... an otherwise never needed format
//
// Overall approach:
//  - state vectors
//     - two formats
//        - a full copy   (for v)
//        - a horizontal stripe (for h, v (temp during CD), eh, ev)
//           - this is consistent with higher-level striping, which is in time dimension=columns
//     - allocate full memory, but only use stripe
//        - that's a waste! And can be big. Optimize later.
//     - keep a 'valid range' variable (full memory or only stripe) for assertions only
//  - models
//     - two formats
//        - a horizontal stripe (for a, b and deltas)
//        - a vertical stripe (for W and delta)
//     - store the sub-range,reorigined to 0; functions know hard-coded whether it is horizontal or vertical
//     - keep a variable to store the actual patch offset
//  - conversion
//     - conversion only happens at entry of forwardprop() (horizontal stripes -> full copy)
//     - updatedeltas() will find it in the right format
//     - all functions are now well-defined in their input/output formats and know what to do, only checks needed, no lazy conversion

// class to hold a matrix possibly distributed over multiple CUDA devices
// Its methods may only be called in cudamode.
class cudadistributedmatrix
{
    cudadistributedmatrix (const cudadistributedmatrix &); void operator= (const cudadistributedmatrix &);
#ifdef MULTICUDA
public:    // in order to use cudastriping type in dbn.h file. Not a good idea to modify it. TODO: fix it.[v-xieche]
#else
protected:
#endif
    bool cudamode;                        // true if has CUDA hardware, make this non-constant in order to enable unit testing

    enum cudastriping_t
    {
        invalidstriping = -1,  // not determined yet
        notstriped = 0,        // maintains a full copy on each device
        stripedwrtrows = 1,    // striped w.r.t. first coordinate (row index)
        stripedwrtcols = 2     // striped w.r.t. second coordinate (col index)
    };
private:
#ifdef MULTICUDA  // set the deviceid for cudadistributedmatrix to indicate which device it lies on. [v-xieche]
    size_t deviceid;
#endif
    // striping modes:
    //  - 'cudastriping' denotes what is stored on the CUDA side; e.g. 'stripedwrtrows' only even allocates rows, while 'notstriped' allocates the full matrix
    //  - 'validstriping' denotes which portion is valid
    //     - if cudastriping = validstriping then the CUDA-stored data is completely valid
    //     - if cudastriping = notstriped but validstriping = stripedwrtrows then the CUDA-side matrix is partially invalid (all values outside its own row)
    cudastriping_t cudastriping;                // how are the CUDA matrices striped
    cudastriping_t thisvalidstriping;           // 'notstriped' can be partially valid
    cudastriping_t & validstriping;             // we go through a reference so we get the right variable in a stripe
    size_t numrows, numcols;                    // overall dimensions of the underlying matrix
    std::vector<unique_ptr<msra::cuda::matrix> > cudamatrices;    // copies in CUDA space; validity not managed/checked in this class
protected:
    void swap (cudadistributedmatrix & other) throw()
    {
        ::swap (cudamode, other.cudamode);
        ::swap (cudastriping, other.cudastriping);
        ::swap (thisvalidstriping, other.thisvalidstriping);
        ::swap (numrows, other.numrows);
        ::swap (numcols, other.numcols);
        cudamatrices.swap (other.cudamatrices);
    }
private:
#if 0
    static size_t cudasubsetfirst (size_t subset, size_t subsets, size_t dim) { return subset * dim / subsets; }
    static size_t cudasubsetdim (size_t subset, size_t subsets, size_t dim) { return cudasubsetfirst (subset+1, subsets, dim) - cudasubsetfirst (subset, subsets, dim); }
    // starting coordinates of the stripe for a CUDA device, 0 if not striped in that dimension
    size_t cudafirstrow (size_t deviceid) const { checkdeviceid (deviceid); if (!rowstriped()) return 0; else return cudasubsetfirst (deviceid, numrowstripes(), numrows); }
    size_t cudafirstcol (size_t deviceid) const { checkdeviceid (deviceid); if (!colstriped()) return 0; else return cudasubsetfirst (deviceid, numcolstripes(), numcols); }
    bool rowstriped() const { checkcudastripingset(); return cudastriping == stripedwrtrows; }
    bool colstriped() const { checkcudastripingset(); return cudastriping == stripedwrtcols; }
    size_t numrowstripes() const { return rowstriped() ? numcudadevices() : 1; }
    size_t numcolstripes() const { return colstriped() ? numcudadevices() : 1; }
    size_t cudarows (size_t deviceid) const { checkdeviceid (deviceid); if (!rowstriped()) return numrows; else return cudasubsetdim (deviceid, numrowstripes(), numrows); }
    size_t cudacols (size_t deviceid) const { checkdeviceid (deviceid); if (!colstriped()) return numcols; else return cudasubsetdim (deviceid, numcolstripes(), numcols); }
#endif
    void checkcudastripingset() const { if (cudastriping == invalidstriping) throw std::logic_error ("checkcudastripingset: no CUDA striping set yet"); }
    void checkdeviceid (size_t deviceid) const { if (deviceid >= numcudadevices()) throw std::logic_error ("checkdeviceid: invalid CUDA device id"); }
    void alloccudamatrices() { if (cudamode) cudamatrices.resize (msra::dbn::numcudadevices());}
public:
#ifdef  MULTICUDA
    void setDeviceId (size_t devid)
    {
        deviceid = devid;
    }
    size_t getDeviceId () const
    {
        if (cudamatrices.size() == 1)
        {
            // should be plain BP training.[v-xieche]
            return 0;
        }
        return deviceid;
    }
#endif
    // needed for model matrices
    cudadistributedmatrix() : cudamode (hascuda()), cudastriping (invalidstriping), validstriping (thisvalidstriping), numrows (0), numcols (0) { alloccudamatrices(); }

    // construct from rvalue reference  --used when creating a stripe into an acceleratedmatrix
    cudadistributedmatrix (cudadistributedmatrix && other)
        : cudamode (hascuda()), cudastriping (other.cudastriping),
        thisvalidstriping ((&other.validstriping == &other.thisvalidstriping) ? other.validstriping : invalidstriping),
        validstriping ((&other.validstriping == &other.thisvalidstriping) ? thisvalidstriping : other.validstriping),   // keep external reference if it is one
        numrows (other.numrows), numcols (other.numcols), cudamatrices (std::move (other.cudamatrices))
    {
#ifdef MULTICUDA
        deviceid = other.deviceid;
#endif
        assert (cudamode == other.cudamode);
        // 'other' will be destructed right after this, so no value in resetting the scalar values in it; cudamatrices[] is already cleared
    }

    // constructor for a column stripe (standalone; pushed into an acceleratedmatrix object by move constructor above)
    cudadistributedmatrix (cudadistributedmatrix & other, size_t firstframe, size_t numframes)
        : cudamode (hascuda()), cudastriping (other.cudastriping), thisvalidstriping (invalidstriping),
        validstriping (other.validstriping),    // this is a reference--we keep the reference to the input one
        numrows (other.numrows), numcols (numframes)
    {
        assert (cudamode == other.cudamode);
        if (!cudamode)
            return;

        // copy over striping and allocate devices' matrices
        alloccudamatrices();

        // set up stripes
        if (cudastriping == stripedwrtcols)
            throw std::logic_error ("cudadistributedmatrix: cannot construct a column stripe from a column-striped distributed matrix");
        foreach_index (i, cudamatrices) // note: patch copies the device as well
            cudamatrices[i].reset (other.cudamatrices[i]->patch (0, other.cudamatrices[i]->rows(), firstframe, numframes + firstframe));
    }

#ifdef MULTICUDA // used to copy data from GPU to CPU. only need to part lies in deviceid device.[v-xieche]
    cudadistributedmatrix (cudadistributedmatrix & other, size_t firstframe, size_t numframes, size_t devid)
        : cudamode (hascuda()), cudastriping (other.cudastriping), thisvalidstriping (invalidstriping),
        validstriping (other.validstriping),    // this is a reference--we keep the reference to the input one
        numrows (other.numrows), numcols (numframes)
    {
        assert (cudamode == other.cudamode);
        if (!cudamode)
            return;

        // copy over striping and allocate devices' matrices
        alloccudamatrices();

        // set up stripes
        if (cudastriping == stripedwrtcols)
            throw std::logic_error ("cudadistributedmatrix: cannot construct a column stripe from a column-striped distributed matrix");
        cudamatrices[devid].reset (other.cudamatrices[devid]->patch (0, other.cudamatrices[devid]->rows(), firstframe, numframes + firstframe));
        deviceid = devid; // set the deviceid here as well. [v-xieche]
    }
    // hack for striped mode in top layer.
    cudadistributedmatrix (cudadistributedmatrix & other, size_t firstframe, size_t numframes, std::vector<size_t> devids)
        : cudamode (hascuda()), cudastriping (other.cudastriping), thisvalidstriping (invalidstriping),
        validstriping (other.validstriping),  numrows (other.numrows), numcols (numframes)
    {
        assert (cudamode == other.cudamode);
        if (!cudamode)
            return;

        // copy over striping and allocate devices' matrices
        alloccudamatrices();

        // set up stripes
        if (cudastriping == stripedwrtcols)
            throw std::logic_error ("cudadistributedmatrix: cannot construct a column stripe from a column-striped distributed matrix");
        foreach_index (i, devids)
        {
            cudamatrices[devids[i]].reset (other.cudamatrices[devids[i]]->patch (0, other.cudamatrices[devids[i]]->rows(), firstframe, numframes + firstframe));
        }
        deviceid = devids[0]; // set the deviceid here as well. [v-xieche]
    }
#endif

    // this method is only intended to be used for testing!
    void setcudamode(bool usecudamode)
    {
        cudamode = usecudamode;
    }

#ifdef COMPACTTRAINER // try to implement forward directly in CUDA device.[v-xieche]
    msra::cuda::matrix & getcudamatrix (size_t deviceid)
    {
        return * cudamatrices[deviceid].get();
    }
    msra::cuda::matrix & getcudamatrix (size_t deviceid) const
    {
        return * cudamatrices[deviceid].get();
    }

    msra::cuda::matrix & getcudamatrix (size_t deviceid, size_t nr, size_t nc)
    {
        return * cudamatrices[deviceid]->patch (0, nr, 0, nc);
    }
    msra::cuda::matrix & getcudamatrix (size_t deviceid, size_t nr, size_t nc) const
    {
        return * cudamatrices[deviceid]->patch (0, nr, 0, nc);
    }
#ifdef STRIPEDTOPLAYER
    // hack for striped mode on top layer.[v-xieche]
    msra::cuda::matrix & stripedgetcudamatrix (size_t deviceid, size_t devnum, cudastriping_t s, size_t numrows, size_t numcols)
    {
        assert (s != invalidstriping);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        // should we need it ??
        // if (s == cudastriping) 
        //    return * cudamatrices[deviceid]->patch (0, cudamatrices[deviceid]->rows(), 0, cudamatrices[deviceid]->cols());  // full view
        // view is sub-view  --base format must be 'notstriped'
        // assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        // devicedim (deviceid, s, fr, fc, nr, nc);
        devicedim (deviceid, s, fr, fc, nr, nc, devnum, numrows, numcols);
        return * cudamatrices[deviceid]->patch (fr, fr + nr, fc, fc + nc);
    }
    msra::cuda::matrix & stripedgetcudamatrix (size_t deviceid, size_t devnum, cudastriping_t s, size_t numrows, size_t numcols) const
    {
        assert (s != invalidstriping);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        if (s == cudastriping)
            return * cudamatrices[deviceid]->patch (0, cudamatrices[deviceid]->rows(), 0, cudamatrices[deviceid]->cols());  // full view
        // view is sub-view  --base format must be 'notstriped'
        assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        // devicedim (deviceid, s, fr, fc, nr, nc);
        devicedim (deviceid, s, fr, fc, nr, nc, devnum, numrows, numcols);
        return * cudamatrices[deviceid]->patch (fr, fr + nr, fc, fc + nc);
    }

    msra::cuda::matrix & stripedgetcudamatrix (size_t deviceid, size_t devnum, cudastriping_t s)
    {
        assert (s != invalidstriping);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        // should we need it ??
        // if (s == cudastriping) 
        //    return * cudamatrices[deviceid]->patch (0, cudamatrices[deviceid]->rows(), 0, cudamatrices[deviceid]->cols());  // full view
        // view is sub-view  --base format must be 'notstriped'
        // assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        // devicedim (deviceid, s, fr, fc, nr, nc);
        devicedim (deviceid, s, fr, fc, nr, nc, devnum);
        return * cudamatrices[deviceid]->patch (fr, fr + nr, fc, fc + nc);
    }

    msra::cuda::matrix & stripedgetcudamatrix (size_t deviceid, size_t devnum, cudastriping_t s) const
    {
        assert (s != invalidstriping);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        devicedim (deviceid, s, fr, fc, nr, nc, devnum);
        return * cudamatrices[deviceid]->patch (fr, fr + nr, fc, fc + nc);
    }
#endif
    void posteriorstatsincuda (cudadistributedmatrix &Pu, cudadistributedmatrix &sumlogpps, cudadistributedmatrix &sumpps, cudadistributedmatrix &sumfcors, double & avlogpp, double &avpp, double &avfcor)
    {
        // foreach_index (deviceid, cudamatrices)
        size_t deviceid = 0;
        {
            this->cudamatrices[deviceid]->posteriorstats (Pu.getcudamatrix (deviceid), sumlogpps.getcudamatrix (deviceid),
                sumpps.getcudamatrix (deviceid), sumfcors.getcudamatrix (deviceid), false/*nosoftmax, not supported here*/);
        }
        // need to add the vecsum function and statistic the correction ratio. [v-xieche]
    }
#ifdef COMPACTTRAINER
    void addrowsumincuda (size_t deviceid, const float thisscale, const cudadistributedmatrix &othercols, const float otherweight)
    {
        this->cudamatrices[deviceid]->addrowsum (thisscale, othercols.getcudamatrix(deviceid), otherweight);
    }

    void addrowsumpoolincuda (size_t deviceid, const float thisscale, const cudadistributedmatrix &othercols, const float otherweight, size_t poolSize, size_t bands, size_t kernels)
    {
        this->cudamatrices[deviceid]->addrowsumpool (thisscale, othercols.getcudamatrix(deviceid), otherweight, poolSize, bands, kernels);
    }
#endif
#endif

    // this can only be set once
    void setcudastriping (cudastriping_t s)
    {
        if (cudastriping != invalidstriping && cudastriping != s)
            throw std::logic_error ("setcudastriping: cannot change striping mode of a matrix once set");
        if (s == invalidstriping)
            throw std::logic_error ("setcudastriping: attempted to change striping mode to invalid");
        if (!cudamode)
            throw std::logic_error ("setcudastriping: cannot set striping mode if no CUDA device");
        if (cudamatrices.empty())
            throw std::logic_error ("setcudastriping: no CUDA device??");
        // TODO: strong exception guarantee: use local var + swap
        foreach_index (deviceid, cudamatrices)
        {
            cudamatrices[deviceid].reset (msra::cuda::newmatrix());
            cudamatrices[deviceid]->setdevice (deviceid);
        }
        cudastriping = s;
        validstriping = cudastriping;
    }

    // functions call this to verify striping
    void checkvalidstriping (cudastriping_t s) const
    {
        if (validstriping != s            && numcudadevices() != 1/*UGH! temp for MVN-SGD*/)
            throw std::logic_error ("checkvalidstriping: wrong striping mode or partially valid");
    }

    // verify that underlying striping is 's' and that it is valid
    void checkvalidcudastriping (cudastriping_t s) const
    {
        if (cudastriping != s            && numcudadevices() != 1/*UGH! temp for MVN-SGD*/)      // must have this type
            throw std::logic_error ("checkvalidcudastriping: wrong striping mode");
        checkvalidstriping (s);     // and be fully valid
    }
#if 0
private:
    // set how we want to view the data when using for input
    // Restricted by current validstriping, i.e. this does not upgrade the data itself, only the view.
    // This is reversible at no cost (i.e. continue to remember the validity of the entire thing).
    void setinputstriping (cudastriping_t s)
    {
        // check compatibility
        if (s == notstriped && validstriping != notstriped)
            throw std::logic_error ("setinputstriping: attempted to upgrade viewed striping mode beyond what we have");
        if (s != notstriped && validstriping != s && validstriping != notstriped)
            throw std::logic_error ("setinputstriping: attempted to downgrade viewed striping mode for mismatching valid striping mode");
        viewedstriping = s;
    }
public:
#endif

#ifdef STRIPEDTOPLAYER
    template<class MATRIX> void makeinputstriping_multicuda (MATRIX & buffer)
    {
        validstriping = stripedwrtrows;
        cudastriping = notstriped;
        syncfromcuda (buffer, true);          // copy to CPU space (this copies stripes)
        validstriping = notstriped;
        synctocuda (buffer, false);            // and copy back (this copies to all)
    }
    template<class MATRIX> void makeinputstriping_multicuda (MATRIX & buffer, std::vector<size_t> &devids)
    {
        validstriping = stripedwrtrows;
        cudastriping = notstriped;
        // should test it. [v-xieche]
        // syncfromcuda (buffer, true, devids);          // copy to CPU space (this copies stripes)
        syncfromcuda (buffer, false, devids);
        validstriping = notstriped;
        synctocuda (buffer, false, devids);            // and copy back (this copies to all)
    }
#endif

    // convert the striping mode if needed
    // Use for upgrading from row striping to full copy.
    // Do not use if you need a downgraded view only, use setinputstriping().
    template<class MATRIX> void makeinputstriping (cudastriping_t s, MATRIX & buffer)
    {
        // check compatibility
        if (s == notstriped && cudastriping != notstriped            && numcudadevices() != 1/*UGH! FIX THIS (tep for MVN-SGD)*/)
            throw std::logic_error ("makeinputstriping: attempted to upgrade valid striping mode for mismatching cudastriping mode");
        if (s != notstriped && cudastriping != s && cudastriping != notstriped)
            throw std::logic_error ("makeinputstriping: attempted to downgrade valid striping mode for mismatching cudastriping mode");
        // if upgrading then we actually do something
        const bool upgrading = (s == notstriped && validstriping != notstriped);
        const bool upgradingandneedtoactuallydosomething = upgrading && (numcudadevices() != 1);
        if (upgradingandneedtoactuallydosomething)
            syncfromcuda (buffer, true);            // copy to CPU space (this copies stripes)
        validstriping = s;
        if (upgradingandneedtoactuallydosomething)
            synctocuda (buffer, false);             // and copy back (this copies to all)
    }

    // config for striping (model parallelism)
    struct stripingconfig
    {
        size_t K;                       // number of devices to use
        size_t numframes;               // number of frames
        size_t submbframes;             // sub-minibatch size
        bool doasync;                   // do async at all (i.e. >1 GPU)?
        bool enableasynctransfers;      // use send()/receive() (rather than the old synchronous code)  --should always; TODO: remove this flag
        bool enablesubbatchcomputation; //

        template<typename VECTORS>
        stripingconfig (const VECTORS & vecs)
        {
            numframes = vecs.cols();
            K = vecs.numcudadevices();

            doasync = K > 1;                        // sub-mb computation makes compute itself slower, so only do it if we parallelize at all

            // on K20X, 256 leads to 1.4 x slow-down, and 512 to 1.2 x, of sgemm() performance
            if (!doasync)
                submbframes = numframes;            // no async: no sub-batching
            else
                submbframes = numframes / 2;        // we have perfect overlap across layers, so we can use the maximum possible, that is, half
#if 0       // overall, this does NOT help (2 GPUs, submbs 1024:512 = 26:31 fps)
            if (K == 2 && submbframes < 1024)       // for K=2, data exchange is so fast that its gain is eaten up by slower matrix product (less frames -> worse caching)
                submbframes = 1024;
            else if (submbframes < 512)
                submbframes = 512;
#endif

            // add '&& false to any of the below for debugging/diagnostics purposes
            enableasynctransfers = doasync;      // use send()/receive() (rather than the old synchronous code)
            //#define TIME_MTM
#ifdef TIME_MTM
            enablesubbatchcomputation = true;
#else
            enablesubbatchcomputation = doasync && submbframes < numframes; // use sub-mb computation (rather than the old full-batch code)
#endif

            // logging
            static bool configlogged = false;
            if (!configlogged)
            {
                fprintf (stderr, "stripingconfig: %d GPUs: async=%d, asynctransfers=%d, subbatchcomp=%d, submbframes=%d for numframes=%d\n",
                         K, (int) doasync, (int) enableasynctransfers, (int) enablesubbatchcomputation, (int) submbframes, (int) numframes);
                fflush (stderr);
                configlogged = true;
            }
        }

        // loop over sub-minibatches and execute body(); pass 'substream' for dependency tracking (dependencies exist only within a substream)
        template<typename FUNCTION>
        void foreachsubbatch (FUNCTION body) const
        {
            size_t te;
            for (size_t ts = 0; ts < numframes; ts = te)
            {
                te = ts + submbframes;      // frame range ts..te-1
                if (te > numframes)
                    te = numframes;
                const size_t dependentsubstream = ts/submbframes+1;
                body (ts, te, dependentsubstream);
            }
        }

        // loop over sub-minibatches and then devices and execute body(); pass 'substream' for dependency tracking (dependencies exist only within a substream)
        template<typename FUNCTION>
        void foreachsubbatchanddevice (FUNCTION body) const
        {
            size_t te;
            for (size_t ts = 0; ts < numframes; ts = te)
            {
                te = ts + submbframes;      // frame range ts..te-1
                if (te > numframes)
                    te = numframes;
                const size_t dependentsubstream = ts/submbframes+1;
                for (size_t deviceid = 0; deviceid < K; deviceid++)
                    body (ts, te, deviceid, dependentsubstream);
            }
        }
    };

    // exchange data for model parallelism
    // Input can be either notstriped (input layer) or stripedwrtrows (all other layers).
    // Output will be notstriped.
    // Returns true if any send() call was made; and false if no change was needed (hence, don't receive() either).
    bool makeinputrowstripingasync (const stripingconfig & sc)
    {
        if (validstriping == notstriped)    // data already distributed
            return false;

#ifdef _DEBUG
        for (size_t n = 0; n < cudamatrices.size(); n++)    // sync for logging
            cudamatrices[n]->synchronize();
        fprintf (stderr, "\n\n\nmakeinputrowstripingasync: STARTING\n\n\n\n");
        fflush (stderr);
#endif

        if (validstriping != stripedwrtrows)
            throw std::logic_error ("makeinputrowstripingasync: input must be stripedwrtrows or notstriped");
        if (cudastriping != notstriped)
            throw std::logic_error ("makeinputrowstripingasync: attempted to upgrade valid striping mode for mismatching cudastriping mode");
        // loop over sub-minibatches
        sc.foreachsubbatch ([&] (size_t ts, size_t te, size_t substream) -> void
        {
            for (size_t n = 1; n < cudamatrices.size(); n++)    // loop over device offsets (k to k+n)
            {
                foreach_index (todeviceid, cudamatrices)
                {
                    size_t fromdeviceid = (todeviceid + n) % cudamatrices.size();   // we avoid parallel access to the same device
                    size_t fr, nr;  // row range available on this device; col range (=time range) given by (ts,te)
                    onedevicedim (fromdeviceid, true, numrows, fr, nr);
                    // we now copy from 'fromdeviceid' to 'todeviceid' the rows (fr,nr) and columns (ts,te)
                    //fprintf (stderr, "makeinputrowstripingasync: sending rows (%d,%d) from %d to %d\n", fr, fr + nr, fromdeviceid, todeviceid);
                    msra::cuda::onsubstream sub (*cudamatrices[fromdeviceid], substream);    // send from this substream (transfer depends on this computation)
                    cudamatrices[fromdeviceid]->send (*cudamatrices[todeviceid].get(), fr, fr + nr, ts, te);
                }
                //fprintf (stderr, "sync\n");
                //synchronize();  // FOR TIMING MEASUREMENT --don't forget to comment out
            }
        });
        validstriping = notstriped;
#ifdef _DEBUG
        fprintf (stderr, "\nmakeinputrowstripingasync: exiting\n\n"); fflush (stderr);
#endif
        return true;
    }

    // sum up all device copies
    // Each device copy contains a partial matrix product of full dimension.
    // All devices' content needs to be summed up.
    template<class MATRIX> void sumacrossdevices (cudastriping_t s, MATRIX & buffer)
    {
        if (validstriping != notstriped)
            throw std::logic_error ("sumacrossdevices: can only be applied to 'notstriped' matrices");
        if (s == notstriped)
            throw std::logic_error ("sumacrossdevices: output format must be striped");
        if (numcudadevices() != 1)  // only one device: nothing to do
        {
            // TODO: Too bad, we cannot do without an additional second buffer.
            // This should be preallocated, but for now we do it locally here.

            // We process stripe by stripe and linearly add all other partial sums into the stripe's one.
            // Loop complexity: O((n-1)^2)    n=number of devices
            // Data complexity: O(n)

            // allocate a temp buffer in all of the devices and get a stripe view on each target
            std::vector<unique_ptr<msra::cuda::matrix> > ms (numcudadevices());
            std::vector<unique_ptr<msra::cuda::matrix> > targets (numcudadevices());
            for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
            {
                // get size of this stripe
                size_t frdummy, fcdummy, nr, nc;
                devicedim (targetdevid, s, frdummy, fcdummy, nr, nc);

                // allocate a CUDA-side matrix to move partial sums from other devices into
                unique_ptr<msra::cuda::matrix> & m = ms[targetdevid];
                m.reset (msra::cuda::newmatrix());
                m->setdevice (targetdevid);
                m->allocate (nr, nc);

                // get local (0-based) views on this stripe
                targets[targetdevid] = stripeforcudadevice (targetdevid, s);
            }
            // now accumulate all stripes
            // get stripe from device (targetdevid + relrevid
            for (size_t reldevid = 1; reldevid < numcudadevices(); reldevid++)
            {
                // for each stripe, accumulate from device reldevid devices away
                // Target stripes live in different devices as well.

                // first get the respective data to accumulate
                // These are all in different devices.
                for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
                {
                    // get patch coordinates of this stripe
                    size_t fr, fc, nr, nc;
                    devicedim (targetdevid, s, fr, fc, nr, nc);

                    const size_t sourcedevid = (targetdevid + reldevid) % numcudadevices();

                    // get the stripe into our local buffer variable
                    unique_ptr<msra::cuda::matrix> partial = stripeforcudadevice (cudamatrices[sourcedevid], targetdevid, s);
                    assert (buffer.rows() >= fr + nr && buffer.cols() >= fc + nc);
                    partial->fetch (0, nr, 0, nc, &buffer(fr,fc), buffer.getcolstride(), false);   // async
                }

                // accumulate stripe into device
                for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
                {
                    // get a local (0-based) view on this stripe
                    unique_ptr<msra::cuda::matrix> & target = targets[targetdevid];

                    // get patch coordinates of this stripe
                    size_t fr, fc, nr, nc;
                    devicedim (targetdevid, s, fr, fc, nr, nc);

                    const size_t sourcedevid = (targetdevid + reldevid) % numcudadevices();

                    // our local buffer variable in the target device
                    unique_ptr<msra::cuda::matrix> & m = ms[targetdevid];

                    // wait until incoming transfer is done
                    cudamatrices[sourcedevid]->synchronize();	// this is where it came from

                    // move it to target device
                    m->assign (0, nr, 0, nc, &buffer(fr,fc), buffer.getcolstride(), false);

                    // accumulate it up
                    target->gems (1.0f, *m, 1.0f);
                }
            }
            // TODO: does free() at the end cause a sync?
        }
        // our output is now in stripe format
        setoutputstriping (stripedwrtrows);
    }

    // notify of partial validity
    // Use this to set the result type of an operation.
    void setoutputstriping (cudastriping_t s)
    {
        // check compatibility
        if (s == notstriped && cudastriping != notstriped)
            throw std::logic_error ("setoutputstriping: attempted to upgrade valid striping mode for mismatching cudastriping mode");
        if (s != notstriped && cudastriping != s && cudastriping != notstriped)
            throw std::logic_error ("setoutputstriping: attempted to downgrade valid striping mode for mismatching cudastriping mode");
        validstriping = s;
    }

    // for all per-element operations, mode must match and be non-disjunct
    void checkmatchingdisjunctcudastriping (const cudadistributedmatrix & othercols) const
    {
        if (cudastriping != othercols.cudastriping)
            throw std::logic_error ("checkmatchingdisjunctcudastriping: mismatching striping modes");
        checkdisjunctcudastriping();
    }

    // check if non-overlapping striping
    void checkdisjunctcudastriping() const
    {
        checkcudastripingset();
        if (cudastriping == notstriped && numcudadevices() > 1) // single device is OK as a compat mode; later remove that condition
            throw std::logic_error ("checkdisjunctcudastriping: an operation was used that is invalid for overlapping striping");
    }

    // allocate all parts  --note: empty matrix possible  --TODO: is resize() possible?
    void alloccuda (size_t n, size_t m)
    {
        checkcudastripingset();
        // TODO: exception guarantee?
        numrows = n;
        numcols = m;
        foreach_index (deviceid, cudamatrices)
        {
            size_t fr, fc, nr, nc;  // coordinates in CPU-side matrix
            devicedim (deviceid, cudastriping, fr, fc, nr, nc);
            cudamatrices[deviceid]->allocate (nr, nc);
        }
    }
#ifdef MULTICUDA
    void alloccuda (size_t n, size_t m, size_t deviceid)
    {
        numrows = n;
        numcols = m;
        cudamatrices[deviceid]->allocate (n, m);
    }
#endif
    // determine the coordinate range of a stripe; or full if not striped
    void onedevicedim (const size_t deviceid, const bool isstriped, const size_t dim, size_t & first, size_t & subdim) const
    {
        if (isstriped)
        {
            const size_t n = numcudadevices();
            first = dim * deviceid / n;
            const size_t next = dim * (deviceid+1) / n;
            if (next > dim)
                throw std::logic_error ("onedevicedim: deviceid out of range");
#if 0       // (makes no difference when dims are multiples of 4, as they usually are in old setups)
            // we must ensure compat with CPU-side matrix patches to allow combining multi-GPU and data parallelism
            // BUGBUG: Seems actually not needed since we now stripe data-parallel exchanges by columns only; while having this HACK issue. So don't do it.
            if (dim >= 100)     // HACK: such small dimensions have a different meaning (one row per GPU in seterrorsignals() and related functions--TODO: stratify this!!)
            {                   //       ^^ I think the right way is to enforce this for columns only, like the CPU-side matrix does; but we don't know in this function, and need to check for edge cases
                matrix::alignpatchindex (first, dim);
                matrix::alignpatchindex (next,  dim);
            }
#endif
            subdim = next - first;
        }
        else
        {
            first = 0;
            subdim = dim;
        }
    }
    // determine the patch coordinates into the full matrix for a given device and striping mode
    void devicedim (size_t deviceid, cudastriping_t s, size_t & fr, size_t & fc, size_t & nr, size_t & nc) const
    {
        onedevicedim (deviceid, s == stripedwrtrows, numrows, fr, nr);
        onedevicedim (deviceid, s == stripedwrtcols, numcols, fc, nc);
    }
#ifdef STRIPEDTOPLAYER
    // determine the coordinate range of a stripe; or full if not striped
    void onedevicedim (const size_t devid, const bool isstriped, const size_t dim, size_t & first, size_t & subdim, size_t numdevice) const
    {
        if (isstriped)
        {
            const size_t n = numdevice;
            size_t thisdevid = (devid >= deviceid) ? (devid - deviceid) : (devid + n - deviceid); // get the correct index here.
            assert (thisdevid < n);
            first =  (size_t)(dim*1.0f*thisdevid/n);
            const size_t next = (size_t) (dim*1.0f*(thisdevid + 1)/n);
            // first = dim * thisdevid / n;		// need to consider the situation of dim/n is not an interger.[v-xieche]
            // const size_t next = dim * (thisdevid +1) / n;
            if (next > dim)
                throw std::logic_error ("onedevicedim: deviceid out of range");
            subdim = next - first;
        }
        else
        {
            first = 0;
            subdim = dim;
        }
    }
    void devicedim (size_t devid, cudastriping_t s, size_t & fr, size_t & fc, size_t & nr, size_t & nc, size_t numdevice, size_t numrows, size_t numcols) const
    {
        onedevicedim (devid, s == stripedwrtrows, numrows, fr, nr, numdevice);
        onedevicedim (devid, s == stripedwrtcols, numcols, fc, nc, numdevice);
    }
    void devicedim (size_t devid, cudastriping_t s, size_t & fr, size_t & fc, size_t & nr, size_t & nc, size_t numdevice) const
    {
        onedevicedim (devid, s == stripedwrtrows, numrows, fr, nr, numdevice);
        onedevicedim (devid, s == stripedwrtcols, numcols, fc, nc, numdevice);
    }
#endif
    // synchronize all devices after kicking off asynchronous data transfers with multiple
    void synchronize() const
    {
        foreach_index (deviceid, cudamatrices)
            cudamatrices[deviceid]->synchronize();
    }
    // sync the last fetch() call
    void syncfetch() const
    {
        foreach_index (deviceid, cudamatrices)
            cudamatrices[deviceid]->syncfetch();
    }
    // sync the last assign() call
    void syncassign() const
    {
        foreach_index (deviceid, cudamatrices)
            cudamatrices[deviceid]->syncassign();
    }

    void synchronize (size_t deviceid)
    {
        cudamatrices[deviceid]->synchronize();
    }

    void synchronize (std::vector<size_t> & deviceids) const
    {
        foreach_index (i, deviceids)
            cudamatrices[deviceids[i]]->synchronize();
    }

    // copy a matrix to the distributed setup
    // This may copy in stripes or make full copies.
    // This operates on the valid striping.
    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous)
    {
        checkcudastripingset();
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        foreach_index (deviceid, cudamatrices)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceid, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceid, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceid]->assign (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*dst*/, &m(vfr,vfc), m.getcolstride()/*src*/, false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            syncsynctocuda();
    }
    // do synctocuda()'s transfer sync, to complete a previous async transfer
    void syncsynctocuda() const
    {
        syncassign();
    }
#ifdef STRIPEDTOPLAYER
    template<class MATRIX> void syncfromcuda (MATRIX & m, bool synchronous) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        if (validstriping == notstriped)
            cudamatrices[0]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        // copy from all devices
        else foreach_index (devid, cudamatrices)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix

            devicedim (devid, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (devid, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
            {
                size_t thisdevid = (deviceid + devid) % cudamatrices.size();
                cudamatrices[thisdevid]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*src*/, const_cast<float*> (&m(vfr,vfc)), m.getcolstride()/*dst*/, false);
            }
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            syncfetch();
    }
#else
    // copy from devices
    // If 'notstriped' we assume all copies are identical and just copy the first.
    // This operates on the valid striping.
    template<class MATRIX> void syncfromcuda (MATRIX & m, bool synchronous) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        if (validstriping == notstriped)
        {
            if (!m.empty())
                cudamatrices[0]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        }
        // copy from all devices
        else foreach_index (deviceid, cudamatrices)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceid, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceid, validstriping, vfr, vfc, vnr, vnc);
//fprintf (stderr, "syncfromcuda [%d]: %d x %d stored portion %d %d %d %d, valid portion %d %d %d %d\n", deviceid, numrows, numcols, cfr, cfc, cnr, cnc, vfr, vfc, vnr, vnc); fflush (stderr);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceid]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*src*/, const_cast<float*> (&m(vfr,vfc)), m.getcolstride()/*dst*/, false);
        }
        // wait until all transfers have completed (the fetch() calls above are all asynchronous)
        if (synchronous)
            syncsyncfromcuda();
    }
    // do synctocuda()'s transfer sync, to complete a previous async transfer
    void syncsyncfromcuda() const
    {
        syncfetch();
    }
#endif

    template<class MATRIX> void syncfromcuda (MATRIX &m, bool synchronous, std::vector<size_t> & deviceids) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        if (validstriping == notstriped)
        {
            fprintf (stderr, "Could touch to here ? debug point[v-xieche]!\n");
            cudamatrices[deviceids[0]]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        }
        // copy from all devices
        else foreach_index (i, deviceids)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceids[i], cudastriping, cfr, cfc, cnr, cnc, deviceids.size());
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceids[i], validstriping, vfr, vfc, vnr, vnc, deviceids.size());
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceids[i]]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*src*/, const_cast<float*> (&m(vfr,vfc)), m.getcolstride()/*dst*/, false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
#ifdef MULTICUDA
        if (synchronous)
            synchronize (deviceids);
#else
        if (synchronous)
            syncfetch();
#endif
    }
#ifdef STRIPEDTOPLAYER  // currently used for exitcomputation and accumulate prior. [v-xieche]
    template<class MATRIX> void syncfromcuda (MATRIX &m, bool synchronous, std::vector<size_t> &deviceids, cudastriping_t s) const
    {
        foreach_index (i, deviceids)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceids[i], s, cfr, cfc, cnr, cnc, deviceids.size());
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceids[i], validstriping, vfr, vfc, vnr, vnc, deviceids.size());
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceids[i]]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*src*/, const_cast<float*> (&m(vfr,vfc)), m.getcolstride()/*dst*/, false);
        }
        if (synchronous)
            synchronize (deviceids);
    }
#endif
    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous, std::vector<size_t> &deviceids)
    {
        checkcudastripingset();
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        foreach_index (i, deviceids)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (i, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (i, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceids[i]]->assign (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc/*dst*/, &m(vfr,vfc), m.getcolstride()/*src*/, false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            syncassign();
    }

    // used for compacttrainer, we don't need striped type and each cuda device keeps a full copy data.
    template<class MATRIX> void syncfromcuda (MATRIX & m, bool synchronous, size_t deviceid) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        cudamatrices[deviceid]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        // wait until all transfers have completed (we hope they are in parallel)
        cudamatrices[deviceid]->syncfetch();
    }

    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous, size_t deviceid)
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        if (numrows > 0 && numcols > 0)
            cudamatrices[deviceid]->assign (0, numrows, 0, numcols, &m(0,0), m.getcolstride()/*src*/, false);
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            synchronize (deviceid);
    }

    // access to the stripes
    size_t numcudadevices() const { checkcudastripingset(); return cudamatrices.size(); }

    // CUDA matrix/matrix stripe for a specific device (if 'notstriped' then these are overlapping)
    // TODO: this should return a patch if our view is different from the base type
    // ... two functions: one operating on the valid view, and one operating on a sub-view if requested (returning a unique_ptr with a patch--done!)
    // get stripe for specific device in its base form (must be fully valid)
    // Takes a cudastriping_t which is checked but will not trigger any transform.
    msra::cuda::matrix &       forcudadevice (size_t deviceid, cudastriping_t s)       { checkvalidcudastriping (s); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }
    const msra::cuda::matrix & forcudadevice (size_t deviceid, cudastriping_t s) const { checkvalidcudastriping (s); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }

    // no striping parameter --view in its base form (must be fully valid). Used for models, where all that matters is consistent striping.
    msra::cuda::matrix &       forcudadevice (size_t deviceid)       { checkvalidstriping (cudastriping); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }
    const msra::cuda::matrix & forcudadevice (size_t deviceid) const { checkvalidstriping (cudastriping); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }

    // get stripe view for specific device, where the base type may be 'notstriped'
    // These return a newly created patch of the underlying matrix. It works for all combinations, although really intended for viewing a 'notstriped' as a striped matrix.
    //msra::cuda::matrix * makeselfpatch (msra::cuda::matrix * m) { return m->patch (0, m->rows(), 0, m->cols()); }
    unique_ptr<msra::cuda::matrix> stripeforcudadevice (unique_ptr<msra::cuda::matrix> & m, size_t deviceid, cudastriping_t s)
    {
        assert (s != invalidstriping);
        checkdeviceid (deviceid);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        if (s == cudastriping)
            return unique_ptr<msra::cuda::matrix> (m->patch (0, m->rows(), 0, m->cols()));  // full view
        // view is sub-view  --base format must be 'notstriped'
        assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        devicedim (deviceid, s, fr, fc, nr, nc);
        return unique_ptr<msra::cuda::matrix> (m->patch (fr, fr + nr, fc, fc + nc));
    }
#ifdef STRIPEDTOPLAYER // hack for striped mode in toplayer. [v-xieche]
    unique_ptr<msra::cuda::matrix> stripeforcudadevice (unique_ptr<msra::cuda::matrix> & m, size_t deviceid, cudastriping_t s, size_t numdevice)
    {
        assert (s != invalidstriping);
        checkdeviceid (deviceid);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        if (s == cudastriping)
            return unique_ptr<msra::cuda::matrix> (m->patch (0, m->rows(), 0, m->cols()));  // full view
        // view is sub-view  --base format must be 'notstriped'
        assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        devicedim (deviceid, s, fr, fc, nr, nc, numdevice);
        return unique_ptr<msra::cuda::matrix> (m->patch (fr, fr + nr, fc, fc + nc));
    }
#endif

    unique_ptr<msra::cuda::matrix> stripeforcudadevice (size_t deviceid, cudastriping_t s)
    {
        return stripeforcudadevice (cudamatrices[deviceid], deviceid, s);
    }
    const unique_ptr<msra::cuda::matrix> stripeforcudadevice (size_t deviceid, cudastriping_t s) const { return const_cast<cudadistributedmatrix*> (this)->stripeforcudadevice (deviceid, s); }

    // compat mode  --delete, then see what fails and fix it
    //msra::cuda::matrix &       forcuda()       { checkcudastripingset(); if (numcudadevices() != 1) throw std::runtime_error ("forcuda: compat mode only allowed for a single GPU"); return *cudamatrices[0].get(); }
    //const msra::cuda::matrix & forcuda() const { checkcudastripingset(); if (numcudadevices() != 1) throw std::runtime_error ("forcuda: compat mode only allowed for a single GPU"); return *cudamatrices[0].get(); }

    // (diagnostics only)
    std::string cudastripingtostr (bool wantvalid/*as opposed to cudastriping*/) const
    {
        switch (wantvalid ? validstriping : cudastriping)
        {
        case invalidstriping: return "invalidstriping";
        case notstriped: return "notstriped";
        case stripedwrtrows: return "stripedwrtrows";
        case stripedwrtcols: return "stripedwrtcols";
        default: return "(cudastripingtostr: invalid striping value--oops?)";
        }
    }
};


// ---------------------------------------------------------------------------
// class mpiaggregator
// ---------------------------------------------------------------------------

// helper class to aggregate accumulators across machines using MPI
// Basic operation: reduce(sum) + redistribute, all in-place.
// The MPI DLL is marked delay-loaded, and thus needs to be installed only if this class is instantiated.

#ifndef NOMPI                               // set this to compile without MPI support
#include "mpi.h"                            // for MPI support
#else                                       // fake emulation so we can compile without MPI (this will simulate a 1-proc environment)
enum mpiconsts { MPI_SUCCESS, MPI_ERR_INTERN, MPI_STATUS_IGNORE, MPI_MAX_ERROR_STRING, MPI_IN_PLACE, MPI_DOUBLE, MPI_FLOAT, MPI_CHAR, MPI_INT, MPI_LONG_LONG_INT, MPI_UNSIGNED, MPI_SUM, MPI_COMM_NULL, MPI_COMM_WORLD, MPI_THREAD_SERIALIZED, MPI_REQUEST_NULL, MPI_STATUSES_IGNORE, MPI_UNDEFINED };
typedef int MPI_Request;
typedef int MPI_Datatype;
typedef int MPI_Comm;
static inline int MPI_Error_string(int, char*, int*) { return 0; }
static inline int MPI_Abort(int,int) { abort(); return MPI_SUCCESS; }
static inline int MPI_Init(int*,char*** argv) { return MPI_SUCCESS; }
static inline int MPI_Init_thread (int*, char*** argv, int, int*) { return MPI_SUCCESS; }
static inline int MPI_Barrier (int) {return MPI_SUCCESS; }
static inline int MPI_Comm_rank(int, int* rank) { *rank = 0; return MPI_SUCCESS; }
static inline int MPI_Comm_size(int, int* size) { *size = 1; return MPI_SUCCESS; }
static inline int MPI_Allreduce(int/*void**/,void*,int,int,int,int) { return MPI_SUCCESS; }
static inline int MPI_Sendrecv(void*,int,int,int,int,void*,int,int,int,int,int,int/*void**/) { return MPI_SUCCESS; }
// TODO: comm_split
static inline int MPI_Comm_free (MPI_Comm*);
static inline int MPI_Finalize() { return MPI_SUCCESS; }
static inline int MPI_Wait(int*,int) { return MPI_SUCCESS; }
static inline int MPI_Isend(void*,int,int,int,int,int,int*) { return MPI_SUCCESS; }
static inline int MPI_Irecv(void*,int,int,int,int,int,int*) { return MPI_SUCCESS; }
static inline int MPI_Bcast(void*,int,int,int,int) { return MPI_SUCCESS; }
static inline int MPI_Waitall(int,void*,int) { return MPI_SUCCESS; }
static inline int MPI_Waitany(int,void*,void*,int) { return MPI_SUCCESS; }
static inline int MPI_Testany(int,void*,void*,void*,int) { return MPI_SUCCESS; }
#endif

struct mpifail : public std::string { mpifail (const string & what) : std::string (what) {} };
static int operator|| (int rc, const mpifail & what)
{
    //fprintf (stderr, "%s returned MPI status %d\n", what.c_str(), rc); fflush (stderr);
    if (rc == MPI_SUCCESS)
        return rc;
    fprintf (stderr, "%s, MPI error %d\n", what.c_str(), rc); fflush (stderr);
    if (rc != MPI_ERR_INTERN)       // (special case: we use that code to indicate a missing msmpi.dll...)
    {
        char errbuf[MPI_MAX_ERROR_STRING + 1] = { 0 };
        int len;
        MPI_Error_string (rc, &errbuf[0], &len);
        fprintf (stderr, "%s, MPI error %d: %s\n", what.c_str(), rc, errbuf); fflush (stderr);
        // we abort through this, so that the MPI system gets the memo
        MPI_Abort (MPI_COMM_WORLD, rc);
        // TODO: or does that only signal an issue, and we should still terminate ourselves?
        // BUGBUG: We'd also need to Abort through the other sub-set communicator
    }
    throw std::runtime_error (what);
}

class mpihelper
{
    int ourrank;            // we are this process ...
    int mpinodes;           // ...out of this many
    size_t nodesinuse;      // actually using this many
    MPI_Comm currentcomm;   // MPI communicator that reflects the current subset selection
    int MPI_Init_DL()       // MPI_Init() with delay-loading the msmpi.dll (possibly causing a failure if missing; we want to catch that)
    {
        //Sleep (10000);                          // (not sure why this is needed, but Jasha added this and I moved it here)
        __try
        {
            int argc = 0; char**argv = NULL;    // TODO: do we need these?
            int provided;
            return MPI_Init_thread (&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
        }
        __except (1/*EXCEPTION_EXECUTE_HANDLER, see excpt.h--not using constant to avoid Windows header in here*/)
        {
            fprintf (stderr, "mpihelper: msmpi.dll missing\n");
            return MPI_ERR_INTERN;
        }
    }
public:
    mpihelper() : currentcomm (MPI_COMM_WORLD)
    {
        static bool inited = false;
        if (inited)
            throw std::logic_error ("mpihelper: this is a singleton class that can only be instantiated once per process");
        inited = true;                      // MPI must be initialized exactly once
        fprintf (stderr, "mpihelper: initializing MPI\n"); fflush (stderr);
        try
        {
            MPI_Init_DL() || mpifail ("mpiaggregator: MPI_Init");
            MPI_Comm_rank (MPI_COMM_WORLD, &ourrank);
            MPI_Comm_size (MPI_COMM_WORLD, &mpinodes);
        }
        catch (...)
        {
#define FAKEMPI
#ifndef FAKEMPI
            throw;
#else       // for debugging, we can simulate it without actually having MPI installed
            ourrank = 0;
            mpinodes = 1;
            fprintf (stderr, "mpihelper: MPI_Init failed; faking MPI mode on one node (for debugging purposes only)\n");
#endif
        }
        nodesinuse = mpinodes;
        requestnodes ("mpihelper");                     // by default we use all of them

        if (mpinodes > 1)
            fprintf (stderr, "mpihelper: we are cog %d in a gearbox of %d\n", (int) ourrank, (int) mpinodes);
        else
            fprintf (stderr, "mpihelper: only one MPI process: MPI operation will be boring\n");
         fflush (stderr);
         // do an initial handshake, for the fun of it
         ping ("mpihelper");
         // stagger the jobs just a little to get a sort-of deterministic order e.g. in GPU allocation when running on one machine
         ::Sleep ((DWORD) (500 * node()));  // continue 0.5 seconds apart
    }
    // Note: we don't clear the sub-communication here although we should, because in case of a crash, this prevents the EXE from terminating.
    // It's OK since this class is a singleton anyway that gets instantiated exactly once at program startup.
    ~mpihelper() { fprintf (stderr, "~mpihelper\n"); fflush (stderr); /*requestnodes ("~mpihelper");*//*clear sub-comm*/ MPI_Finalize(); }
    // ping each other
    void ping (const char * msg) const
    {
         //fprintf (stderr, "ping [%s]: entering\n", msg); fflush (stderr);
#undef USE2NDCOMM
#ifndef USE2NDCOMM
         if (nodes() != mpinodes)
         {
             fprintf (stderr, "ping [%s]: cannot be applied to subset (%d) of nodes, skipping\n", msg, (int) nodes()); fflush (stderr);
             return;
         }
#endif
         std::array<int,1> handshake;
         handshake[0] = 1;
         fprintf (stderr, "ping [%s]: %d nodes pinging each other\n", msg, (int) nodes()); fflush (stderr);
         allreduce (handshake);
         fprintf (stderr, "ping [%s]: all %d nodes responded\n", msg, handshake[0]); fflush (stderr);
         //fprintf (stderr, "ping [%s]: exiting\n", msg); fflush (stderr);
    }

    // declare how many MPI nodes should be used
    // We may decide to not use all if the communication overhead would be net-negative.
    // This number can be dynamically adjusted within a job, e.g. when the minibatch size changes.
    // The effect of selecting a subset is to create a new communicator that reflects the subset.
    // This way, we can use both the subset in the non-idle nodes as well as the final MPI_Allreduce to sync up on MPI_COMM_WORLD, without interference.
    // ... I hope! NO, not working. And, according to Jeff Baxter, it shouldn't be necessary.
    bool forceuseallnodes() const { return false; }  // enable to forbid using a subset of nodes, for testing purposes
    void requestnodes (const char * msg, size_t requestednodes = SIZE_MAX/*default: all*/)
    {
        //fprintf (stderr, "requestnodes [%s,%d]: entering\n", msg, (int) requestednodes); fflush (stderr);
        if (forceuseallnodes() && requestednodes < SIZE_MAX)
        {
            requestednodes = SIZE_MAX;
            fprintf (stderr, "requestnodes: being forced to always use all nodes despite not being optimal\n");
        }
        //fprintf (stderr, "requestnodes: currentcomm is initially %x\n", (int) currentcomm); fflush (stderr);
        //fprintf (stderr, "requestnodes: was asked to use %d out of %d MPI nodes\n", (int) requestednodes, mpinodes); fflush (stderr);
        ping ("requestnodes (before change)");
        // undo current split
#ifdef USE2NDCOMM
        if (currentcomm != MPI_COMM_WORLD/*no subset*/ && currentcomm != MPI_COMM_NULL/*idle nodes*/)
        {
            fprintf (stderr, "requestnodes: MPI_Comm_free %x\n", (int) currentcomm); fflush (stderr);
            MPI_Comm_free (&currentcomm) || mpifail ("requestnodes: MPI_Comm_free");    // will leave MPI_COMM_NULL here
        }
#endif
        currentcomm = MPI_COMM_WORLD;       // reset to MPI_COMM_WORLD
        //fprintf (stderr, "requestnodes: currentcomm is %x\n", (int) currentcomm); fflush (stderr);
        // create a new split (unless all nodes were requested)
        if (requestednodes < mpinodes)
        {
#ifdef USE2NDCOMM
            fprintf (stderr, "requestnodes: MPI_Comm_split %d\n", (node() < requestednodes) ? 1 : MPI_UNDEFINED); fflush (stderr);
            MPI_Comm_split (communicator(), (node() < requestednodes) ? 1 : MPI_UNDEFINED, 0, &currentcomm) || mpifail ("requestnodes: MPI_Comm_split");
            fprintf (stderr, "requestnodes: MPI_Comm_split -> %x\n", (int) currentcomm); fflush (stderr);
#endif
        }
        else    // leave currentcomm as MPI_COMM_WORLD
            requestednodes = mpinodes;      // and clip to #nodes
        nodesinuse = requestednodes;
        fprintf (stderr, "requestnodes [%s]: using %d out of %d MPI nodes (%d requested); we (%d) are %s\n",
                 msg, nodesinuse, mpinodes, (int) requestednodes,
                 node(), isidle() ? "out (idle)" : "in (participating)");
        fflush (stderr);
        //fprintf (stderr, "requestnodes: currentcomm is %x, finally\n", (int) currentcomm); fflush (stderr);
        ping ("requestnodes (after change)");
        //fprintf (stderr, "requestnodes [%s,%d -> %d]: exiting\n", msg, (int) requestednodes, (int) nodes()); fflush (stderr);
    }
    // get the communicator that reflects the selected nodes
    MPI_Comm communicator() const { return currentcomm; }
    size_t nodes() const { return nodesinuse; }
    size_t node() const { return ourrank; }
    size_t ismainnode() const { return ourrank == 0; }          // we are the chosen one--do extra stuff like saving the model to disk
    bool isidle() const { return node() >= nodes(); }           // user had requested to not use this many nodes
    bool usingallnodes() const { return nodes() == mpinodes; }  // all nodes participate (used to check whether we can use MPI_Allreduce directly)

    // -----------------------------------------------------------------------
    // data-exchange functions (wrappers around MPI functions)
    // -----------------------------------------------------------------------

    // helpers to determine the MPI_Datatype of a pointer
    static MPI_Datatype getdatatype (char *)   { return MPI_CHAR; }
    static MPI_Datatype getdatatype (int *)    { return MPI_INT; }
    static MPI_Datatype getdatatype (float *)  { return MPI_FLOAT; }
    static MPI_Datatype getdatatype (double *) { return MPI_DOUBLE; }
    static MPI_Datatype getdatatype (size_t *) { return sizeof (size_t) == 4 ? MPI_UNSIGNED : MPI_LONG_LONG_INT; }

    // allreduce of a vector
    template<typename VECTORLIKEOBJECT>
    void allreduce (VECTORLIKEOBJECT & accumulator) const
    {
        auto * dataptr = accumulator.data();
        size_t totalnumelements = accumulator.size();
        // use MPI to compute the sum over all elements in (dataptr, totalnumelements) and redistribute to all nodes
        //fprintf (stderr, "allreduce: all-reducing matrix with %d elements\n", (int) totalnumelements); fflush (stderr);
        //fprintf (stderr, "allreduce:MPI_Allreduce\n"); fflush (stderr);
        if (nodes() > 1 && communicator() != MPI_COMM_NULL)
            MPI_Allreduce (MPI_IN_PLACE, dataptr, (int) totalnumelements, getdatatype (dataptr), MPI_SUM, communicator()) || mpifail ("allreduce: MPI_Allreduce");
        //fprintf (stderr, "allreduce: all-reduce done\n"); fflush (stderr);
    }
    // allreduce of a scalar
    template<typename T>
    void allreducescalar (T & val)
    {
        struct scalarasvectorref_t { T * p; scalarasvectorref_t (T & r) : p(&r) {} T * data() const { return p; } size_t size() const { return 1; } } r (val); // wraps 'val' as a VECTORLIKEOBJECT
        allreduce (r);
    }

    // redistribute a vector from main node to all others
    template<typename VECTORLIKEOBJECT>
    void redistribute (VECTORLIKEOBJECT & data) const
    {
        ping ("redistribute");
        auto * dataptr = data.data();
        size_t totalnumelements = data.size();
        // use MPI to send over all elements from the main node
        fprintf (stderr, "redistribute: redistributing matrix with %d elements %s this node\n", (int) totalnumelements, ismainnode() ? "from" : "to"); fflush (stderr);
        MPI_Bcast (dataptr, (int) totalnumelements, getdatatype (dataptr), 0/*send from this node*/, communicator()) || mpifail ("redistribute: MPI_Bcast");
    }

    // redistribute a variable-length string
    void redistributestring (std::string & str) const
    {
        ping ("redistribute (string)");
        // first transmit the size of the string
        std::array<int,1> len;
        len[0] = (int) str.size();
        redistribute (len);
        // then the string --we transmit it as a char vector
        std::vector<char> buf (str.begin(), str.end()); // copy to a char vector
        buf.resize (len[0]);                            // this will keep the main node's string at correct length, while extending or shrinking others, which is OK because those get overwritten
        redistribute (buf);                             // exchange as a char vector
        str.assign (buf.begin(), buf.end());            // and convert back to string
    }

    // send a buffer to 'tonode' while receiving a buffer from 'fromnode'
    template<typename BUFFER1, typename BUFFER2>
    void sendrecv (const BUFFER1 & fetchbuffer, size_t tonode,
                   BUFFER2 & recvsubbuffer, size_t fromnode)
    {
        //fprintf (stderr, "@@sendrecv [%d]: sending %d bytes to %d while receiving %d bytes from %d\n", (int) node(), (int) fetchbuffer.size(), (int) tonode, (int) recvsubbuffer.size(), (int) fromnode); fflush (stderr);
        MPI_Sendrecv (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,   (int) (nodes()*nodes() + tonode),
                      recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode, (int) (nodes()*nodes() + node()),
                      communicator(), MPI_STATUS_IGNORE) || mpifail ("sendrecv: MPI_Sendrecv");
    }

    // asynchronous send and receive
    // Call this, then do other stuff, and then call sencrevbwait() to finish it off (you must call it).
    std::vector<MPI_Request> sreq;  // lazily grown
    std::vector<MPI_Request> rreq;
    MPI_Request * getrequest (std::vector<MPI_Request> & req, size_t handle)
    {
        //fprintf (stderr, "@@getrequest [%c]: %d\n", &req == &sreq ? 's' : 'r', handle); fflush (stderr);
        if (handle >= req.size())                       // grow the handle array
            req.resize (handle +1, MPI_REQUEST_NULL);
        //if (req[handle] != MPI_REQUEST_NULL)            // sanity check
        //    fprintf (stderr, "@@getrequest: orphaned async send or recv operation %d\n", handle); fflush (stderr);
        if (req[handle] != MPI_REQUEST_NULL)            // sanity check
            throw std::logic_error ("getrequest: orphaned async send or recv operation");
        return &req[handle];        // MPI functions want a pointer
    }

    template<typename BUFFER>
    void sendasync (const BUFFER & fetchbuffer, size_t tonode, size_t asynchandle)
    {
        //fprintf (stderr, "@@sendasync: %d bytes to %d with handle %d and tag %d\n", fetchbuffer.size(), tonode, asynchandle, (int) (asynchandle * nodes() + tonode)); fflush (stderr);
        MPI_Isend (const_cast<char*> (fetchbuffer.data())/*header file const bug*/, (int) fetchbuffer.size(),   MPI_CHAR, (int) tonode,
                   (int) (asynchandle * nodes() + tonode), communicator(), getrequest (sreq, asynchandle)) || mpifail ("sendrecv: MPI_Isend");
    }

    template<typename BUFFER>
    void recvasync (BUFFER & recvsubbuffer, size_t fromnode, size_t asynchandle)
    {
        //fprintf (stderr, "@@recvasync: %d bytes from %d with handle %d and tag %d\n", recvsubbuffer.size(), fromnode, asynchandle, (int) (asynchandle * nodes() + node())); fflush (stderr);
        MPI_Irecv (recvsubbuffer.data(),                                            (int) recvsubbuffer.size(), MPI_CHAR, (int) fromnode,
                   (int) (asynchandle * nodes() + node()), communicator(), getrequest (rreq, asynchandle)) || mpifail ("sendrecv: MPI_Irecv");
    }
#if 0
    void waitrequest (std::vector<MPI_Request> & req, size_t handle)
    {
fprintf (stderr, "@@waitrequest [%c]: %d\n", &req == &sreq ? 's' : 'r', handle); fflush (stderr);
if (req[handle] == MPI_REQUEST_NULL)            // sanity check
    fprintf (stderr, "waitrequest: waiting for unused handle %d\n", handle); fflush (stderr);
        if (req[handle] == MPI_REQUEST_NULL)            // sanity check
            throw std::logic_error ("waitrequest: waiting for unused handle");
        MPI_Wait (&req[handle], MPI_STATUS_IGNORE) || mpifail ("waitreq: MPI_Wait");
        req[handle] = MPI_REQUEST_NULL;                 // for sanity check
fprintf (stderr, "@@waitrequest [%c]: %d done\n", &req == &sreq ? 's' : 'r', handle); fflush (stderr);
    }
    void sendwait (size_t asynchandle) { waitrequest (sreq, asynchandle); }
    void recvwait (size_t asynchandle) { waitrequest (rreq, asynchandle); }
#endif

    void sendwaitall()          // wait for all pending send requests to complete
    {
        auto & req = sreq;
        //fprintf (stderr, "@@sendwaitall\n"); fflush (stderr);
        MPI_Waitall ((int) req.size(), req.data(), MPI_STATUSES_IGNORE) || mpifail ("sendwaitall: MPI_Waitall");
        //fprintf (stderr, "@@sendwaitall: done\n"); fflush (stderr);
    }
    bool recvwaitany (bool blocking, size_t & handle)   // get zero or one pending receive request
    {
        auto & req = rreq;
        int i, f;
        if (blocking)   // blocking: last one will return MPI_UNDEFINED
            MPI_Waitany ((int) req.size(), req.data(), &i, MPI_STATUS_IGNORE) || mpifail ("recvwaitany: MPI_Waitany");
        else            // non-blocking: if none it will return MPI_UNDEFINED
            MPI_Testany ((int) req.size(), req.data(), &i, &f, MPI_STATUS_IGNORE) || mpifail ("recvwaitany: MPI_Testany");
        //fprintf (stderr, "@@recvwaitany [%sblocking]: got %d\n", blocking? "" : "non-", i); fflush (stderr);
        if (i == MPI_UNDEFINED)
            return false;
        handle = i;
        return true;
    }

#if 0
    // older joint functions, soon no longer to be used
    template<typename BUFFER1, typename BUFFER2>
    void sendrecvasync (const BUFFER1 & fetchbuffer, size_t tonode,
                        BUFFER2 & recvsubbuffer, size_t fromnode, size_t asynchandle)
    {
        sendasync (fetchbuffer,   tonode,   asynchandle);
        recvasync (recvsubbuffer, fromnode, asynchandle);
    }

    // call this once for every sendrecvasync() call
    void sendrecvwait (size_t asynchandle)
    {
BEGINTIME("& sendrecvwait");
        sendwait (asynchandle);
        recvwait (asynchandle);
ENDTIME();
    }
#endif
};

class mpiaggregator : public mpihelper
{
    // general configuration of MPI aggregation
    static bool doublebufferingallowed() { return true; }   // generally allow or disallow double-buffering
    bool doublebufferingrequested;  // enable double-buffering for this epoch
    bool quantizationrequested;     // bits<32 for this epoch
    bool forceaggregate;            // force to use aggregate function even for K=1 (for testing/debugging)
    size_t parallelizablembsize;    // do not parallelize for MB size below this
    size_t subbatchsize;            // #frames that can be computed during available variable-cost time  --TODO: make this a parameter, as it depends on the model
public:
    mpiaggregator()
    {
        doublebufferingrequested = true;    // double-buffering is requested for this epoch?
        quantizationrequested = true;       // bits<32 for this epoch
        forceaggregate = true;              // force to use aggregate function even for K=1 (for consistency when choosing MB size; also for testing/debugging)
        forcedfullysyncoperation = false;   // (debugging facility)
        parallelizablembsize = 1000;        // 1472 is inefficient if parallelized
        // measured for AdaGrad 40M-param model:  --note: this is BS
        /*
         * 1024 (1 process): 122-40 ms (fixed):89+40 ms (variable) (where 89 ms is fwbw for 1024 frames, rawgr another 40 ms, rest is fixed)
         * 2880: minibatch time = 227..244 ms bounded by I/O -> it's the comm time
         * compute time for N/2 frames = 82 + 129*(N/2)/1024/K
         * communication time for N/2 frames = 240
         * kfps = N/2/max{compute time, comm time} = N/2 / max { 82 + 129*(N/2)/1024/K, 240 }
         * -> K = 129*(N/2)/1024/(240-82) = N  / 2540  but empirically for 5824, 8 nodes are best, i.e. ~N/750
         */
        subbatchsize = 256;              // optimum #frames per node  --TODO: make this a parameter, as it depends on the model
        // actually measured times for the 40M-parameter SWBD model:
        //  - raw communication time: 2*4.5 = 9 ms (on XXX nodes)  --TODO: get more details
        //  - fixed-cost time: 18 ms (40M model, AdaGrad enabled)  --TODO: this does not count quantization; should be another ~18 ms
        //  - variable-cost times (for 18 ms fixed cost):  --TODO: this does not include quantization
        //    sub-batch size	sub-batch time	var cost	var cost/frame	e2e speed   --TODO: untabify these lines--WTF, VS does not do it!
        //    256		~59 ms		40 ms		1/6400 ms	4338 fps    --25% penalty on var cost, 50% overall
        //    512		~89 ms		70 ms		1/7300 ms	5753 fps
        //    1024		~143 ms		125 ms		1/8192 ms	7160 fps
        //    2048		~260 ms		240 ms		1/8500 ms	7876 fps
        //    4096		~490 ms		470 ms		1/8700 ms	8359 fps
        //    8192		~955 ms		935 ms		1/8800 ms	8578 fps
        //    16384 [defer]	~1870 ms	1850 ms		1/8900 ms	8761 fps
    }

    // querying the configuration (we need to prepare stuff outside actually)
    bool isdistributed() const { return nodes() > 1 || forceaggregate; }
    bool isdoublebuffered() const { return isdistributed() && doublebufferingrequested && doublebufferingallowed(); }
    bool canusempiallreduce() const { return !quantizationrequested && !isdoublebuffered() && usingallnodes(); }   // can we use MPI_Allreduce directly?

    // we configure the #bits here so we have all in one place (although this is used inside rbmmodelmatrix functions, where it really belongs)

    // -----------------------------------------------------------------------
    // debugging helpers to enforce synchronized operation
    // -----------------------------------------------------------------------

private:
    bool forcedfullysyncoperation;  // debugging/timing: disable both bg thread and GPU overlap
    std::unique_ptr<msra::cuda::matrix> dummycudamatrix; // for being able to sync the first device
public:
    // force to not use bg thread nor GPU overlapped processing
    void forcefullysyncoperation() { forcedfullysyncoperation = true; fprintf (stderr, "forcefullysyncoperation: disabling bg thread and overlapped GPU processing for MPI aggregation\n"); fflush (stderr); }
    // helper to force to synchronize the GPU
    void gpubarrier()
    {
        if (!dummycudamatrix)
            dummycudamatrix.reset (msra::cuda::newmatrix());
        dummycudamatrix->synchronize();
        dummycudamatrix->syncfetch();       // for good measure, not sure if non-blocking streams are covered by a device sync
        dummycudamatrix->syncassign();
    }

    // -----------------------------------------------------------------------
    // simple aggregation off CPU-side matrix objects
    // This is currently not used.
    // -----------------------------------------------------------------------

    // call this before aggregate()
    void startingnewepoch()
    {
        fprintf (stderr, "mpiaggregator: entering a new epoch\n");
        // (not much to do at this point in time, but in the future, we may reset delay lines etc.)
    }

    // call this to test if all nodes still have data
    // If the answer is 'no' then all nodes are to stop right there.
    // Do not call this again in the same epoch. All nodes must return from this as 'false' exactly once per epoch.
    // (We will leave some data on the table for this epoch, but not significantly so, and the data will be used in the future.)
    bool done (bool wearedone) const
    {
        fprintf (stderr, "done: entered\n");
        int numready = wearedone ? 0 : 1;   // number of nodes that are still ready (still have data)--first ourselves
        if (nodes() > 1)
            MPI_Allreduce (MPI_IN_PLACE, &numready, 1, MPI_INT, MPI_SUM, communicator()) || mpifail ("aggregate: MPI_Allreduce");
        fprintf (stderr, "done: total numready=%d\n", numready);
        // now 'numready' is the count of nodes that got 'wearedone = false' fed
        bool weshallstop = numready < (int) nodes();     // not all have data left: let's all call it a day, this epoch is over
        if (weshallstop)
            fprintf (stderr, "mpiaggregator: at least one node is out of data (that node is %s), so this epoch is over\n", wearedone ? "us" : "not us");
        return weshallstop;
    }

    // -----------------------------------------------------------------------
    // striped aggregation support
    // -----------------------------------------------------------------------

    struct mpistripeheader
    {
        size_t mbframes;                // #frames that constitutes the gradient in this qpackage
    };

    class mpistripebuffer               // buffer for qpackage of entire cross-model stripe
    {
        size_t rsize;
        std::shared_ptr<char> buffer;   // qpackages go here; this is page-locked memory for efficient GPU interaction
        void validate () const { if (!buffer) throw std::logic_error ("mpistripebuffer: buffer used before it was allocated"); }
    public:
        mpistripebuffer() : rsize (0) { }
        void init (std::shared_ptr<char> && b, size_t s) { buffer = b; rsize = s; }
        size_t size() const { return rsize; }
        char *       data()       { validate(); return buffer.get(); }
        const char * data() const { validate(); return buffer.get(); }
        mpistripeheader &       header()       { return *(mpistripeheader *) data(); }  // header is in the first bytes
        const mpistripeheader & header() const { return *(mpistripeheader *) data(); }
    };

    template<typename RETVAL>
    class task : public Concurrency::task_group // a simple emulation of VS 2013's task class (I only want get())
    {
        RETVAL retval;
    public:
        template<typename THREADFN>
        void run (THREADFN & f)
        {
#if 1       // disable to test double-buffering without an actual bg thread
            task_group::run ([=]()
            {
                //fprintf (stderr, "task::run: entering thread func\n"); fflush (stderr);
                // once it is complete, we write the return value to 'retval'; and wait() will return so that our get() method gets that value safely
                try
                {
BEGINTIME("steps2and3 end-to-end (on bg thread) ###");
                    retval = std::move (f());
ENDTIME();
                }
                catch (const exception & e) // TODO: I've never seen these trigger; test once that they get propagated out of the thread, then delete this
                {
                    fprintf (stderr, "task::run: thread func failed with exception ('%s')\n", e.what()); fflush (stderr);
                    throw;
                }
                catch (...)
                {
                    fprintf (stderr, "task::run: thread func failed with exception (unknown type)\n"); fflush (stderr);
                    throw;
                }
                //fprintf (stderr, "task::run: exited thread func\n"); fflush (stderr);
            });
#if 0       // enable to force immediate completion (for debugging/time measurements)
            fprintf (stderr, "task::run: forcing completion immediately\n");
            wait(); // force completion immediately
#endif
#else       // not really using the bg thread (but pretend we do w.r.t. logic)
BEGINTIME("steps2and3 end-to-end (on MAIN (!) thread) ###");
            retval = std::move (f());
ENDTIME();
#endif
        }
        RETVAL && get()
        {
            //fprintf (stderr, "task::get: beginning to wait for thread func to exit\n"); fflush (stderr);
            try
            {
                wait();
            }
            catch (const exception & e) // (see comment in run() on this)
            {
                fprintf (stderr, "task::get: thread wait returned an exception ('%s')\n", e.what()); fflush (stderr);
                throw;
            }
            catch (...)
            {
                fprintf (stderr, "task::get: thread wait returned an exception (unknown type)\n"); fflush (stderr);
                throw;
            }
            //fprintf (stderr, "task::get: completed to wait for thread func to exit\n"); fflush (stderr);
            return std::move (retval);
        }
        task() : task_group() {}
    };

    std::vector<size_t> mpistripebuffersizes;               // sizes of shared buffer (one size per node); set by entercomputation(), constant after that
    std::vector<mpistripebuffer> idlelocalbuffers;          // shared stripe buffers for communicating with our own node (quantized matrices go here)
    std::vector<std::vector<char>> peerbuffers;             // [kfrom] for receiving quantized data from other MPI nodes (all have dimension of stripe for 'k')
    task<std::vector<mpistripebuffer>> bgthread;            // background thread for aggregate function

    // note: passing 0 for mbsize and 32 for bits will switch to non-distributed mode while keeping the machinery; used for initial epoch
    template<typename F1>
    void entercomputation (size_t mbsize, bool withdoublebuffering, size_t bits, F1 init)
    {
        doublebufferingrequested = withdoublebuffering;     // remember the setting; for AdaGrad, we currently disable it
        quantizationrequested = (bits < 32);                // for model averaging we'd not want to quantize since error feedback would happen too rarely

        const bool nondistributed = mbsize == 0 && !withdoublebuffering && bits == 32;
        if (nondistributed)
            fprintf (stderr, "entercomputation: MPI aggregation disabled (supposed to be in very first epoch)\n"), fflush (stderr);
        else
            fprintf (stderr, "entercomputation: double buffering %srequested, quantization %srequested\n", doublebufferingrequested ? "" : "not ", quantizationrequested ? "" : "not "), fflush (stderr);

        // first make sure all nodes are still hanging in there
        requestnodes ("entercomputation (entering)" /*all*/);
        ping ("entercomputation");

        // compute how many nodes we can really make good use of
        // TODO: we can also incorporate other knowledge such as #parameters etc., but probably best to leave that to a user-experimentable parameter
        size_t Kopt;
#if 1
        forceaggregate = !nondistributed;                   // in regular operation, we force to aggregate for consistency
        Kopt = nondistributed ? 1 : SIZE_MAX;               // (nondistributed mode means one node)
#else
        if (mbsize < parallelizablembsize)
            Kopt = 1;                                           // note: this will disable double buffering and hence our whole MPI aggregation process incl. quantization
        else
            Kopt = (mbsize + subbatchsize -1) / subbatchsize;   // our guess for optimal number of nodes, cf. my formula
#endif
        // for model parallelism we must have a least as many stripes as GPUs since stripes cannot span multiple GPUs
        if ((Kopt != 1 || forceaggregate) && Kopt < numcudadevices())
            Kopt = numcudadevices();
        fprintf (stderr, "entercomputation: estimated optimal # MPI nodes = %d (for minibatch size of %d)\n", (int) Kopt, (int) mbsize); fflush (stderr);
        requestnodes ("entercomputation (selection)", Kopt);                                // ask for this many (we get at most as many MPI nodes as the user has started the job with)
        if (nodes() < numcudadevices() && isdistributed())
            throw std::runtime_error ("entercomputation: must have at least as many MPI nodes as model-parallel GPUs");
        // this will set nodes() to the number of nodes we actually got granted
        mpistripebuffersizes.clear();
        if (isidle())
            return; // idle--nothing to do

        // ask our caller (all layers) to dimension itself
        // For each node, it should determine its stripe dimension(s) and fill in mpistripebuffersizes[node] to say how many bytes each stripe's buffer needs.
        mpistripebuffersizes.resize (nodes());

        // we ourselves first reserve space for a header
        foreach_index (k, mpistripebuffersizes)     // initialize with 0
            mpistripebuffersizes[k] = sizeof (mpistripeheader);
        // init() will now bump up the value by the bytes needed by the quantized packages
        init (mpistripebuffersizes, bits);

        // buffer for receiving our owned stripe from peers
        const size_t k = node();
        peerbuffers.resize (nodes());
        foreach_index (kfrom, peerbuffers)
            peerbuffers[kfrom].resize (mpistripebuffersizes[k]);    // note: index='kfrom', dimension is for 'k'
    }

    template<typename F1>
    void exitcomputation (F1 finish)
    {
        fprintf (stderr, "exitcomputation: entering\n"); fflush (stderr);
        if (!isidle())
        {
            // stop any potentially ongoing double-buffering bg thread
            fprintf (stderr, "exitcomputation: waiting for bg thread\n"); fflush (stderr);
            auto lastretval = bgthread.get();
            //fprintf (stderr, "numl=%d\n", lastretval.size());
            lastretval.clear();         // explicitly clear it now  --this will release the GPU buffer, with all sync'ing etc.
            fprintf (stderr, "exitcomputation: bg thread done, freeing memory\n"); fflush (stderr);

            // free the shared buffers
            // The shared buffer's destructor knows to wait until potentially ongoing data transfers have completed.
            //fprintf (stderr, "numi=%d\n", idlelocalbuffers.size());
            idlelocalbuffers.clear();   // this is the GPU one; its destructor knows to wait
            //fprintf (stderr, "numi2=%d\n", idlelocalbuffers.size());
            peerbuffers.clear();            // this is our own one which only lives inside the thread function, thread is already stopped here

            // ask our caller to clean itself up
            fprintf (stderr, "exitcomputation: cleaning up layers\n"); fflush (stderr);
            finish();

            // reset ourselves (after each epoch; note that we don't get destructed; this brings us back into clean=empty state)
            fprintf (stderr, "exitcomputation: freeing up shared buffers\n"); fflush (stderr);
            mpistripebuffersizes.clear();
        }

        // reactivate all nodes and make sure all nodes are still hanging in there
        requestnodes("exitcomputation (reset at end)" /*all*/);
        ping ("exitcomputation");
        fprintf (stderr, "exitcomputation: done\n"); fflush (stderr);
    }

    // aggregate() --the big fat aggregator
    // This implement an "all-reduce" operation with these features:
    //  - double buffering;
    //  - concurrent GPU operation (through callbacks that are supposed to use async GPU calls);
    //  - concurrent data transfer through MPI (through a background thread).
    // Explanation of the concurrency model (example: 4 stripes, we are node 1 of 0..3):
    //  - symbols:
    //     - Nk = node k
    //     - Sk = sub-batch k stripe k as computed by node Nk
    //     - Gk = gradient stripe k as aggregated inside node Nk
    //  - timing:
    //       GPU-to-CPU Sk                    k=...   ||--2-|--3-|--0-|--1-|    |    |    |    ||--2-|--3-...
    //       MPI_Isend Sk to Nk                       ||idle|--2-|--3-|--0-|    |    |    |    ||idle|
    //       MPI_Irecv S1 from Nk                     ||idle|--0-|--3-|--2-|    |    |    |    ||idle|
    //       MPI_Isend G1 to Nk                       ||    |    |    |    |--2-|--3-|--0-|idle||
    //       MPI_Irecv Gk from Nk                     ||    |    |    |    |--0-|--3-|--2-|idle||
    //       CPU-to-GPU                   ...--3-|--2-||    |    |    |    |--1-|--0-|--3-|--2-||
    //       GPU compute                  |-setgrad-|q||u|----fprop---|----bprop---|-setgrad-|q||u|----fprop...
    //       GPU transfer buffer 1 in use           |---------------------------------------------|
    //       GPU transfer buffer 2 in use ---------------|                                   |------------
    //       peerbuffer in use                              |--------------|
    //       GPU subbuffer in use                   |----------------------|                 |------------
    //       GPU aggbuffer in use         ---------------|                 |----------------------|
    // ...TODO: a better approach would be to get a parallel buffer (CPU,GPU) and have mpiaggregate flip that one; now it's brittle
    //  - notes:
    //     - each 'Isend Sk' is preceded by a wait for GPU-to-CPU to complete
    //     - each 'Irecv S1' is followed by unquant and accumulate
    //     - last GPU-to-CPU is also followed by unquant and accumulate
    //     - first 'Isend G1' is preceded by quant
    //     - each 'Irecv Gk' is followed by unquant before kicking off CPU-to-GPU
    //     - 'q' (quant) kicks off GPU-to-CPU transfer for all Sn in the shown order
    //     - 'u' (unquant) is preceded by waiting for all CPU-to-GPU to complete
    // Buffer usage for double-buffering:
    //  - on the CPU, we need only one peer buffer since there is only one thread at a time, and it is not used beyond the lifetime of the thread lambda (middle steps)
    //  - on the GPU, we use two buffers (fetch/assign); --TODO: we may need to enforce more sync
    //     - in addition to each 'completed' event, add a 'ready' event for each GPU buffer  --easy! can be handled inside fetchxxx() and assignxxx()
    //     - ...NO! Not working; those buffers are used beyond the data transfer since we actually work with their content on the CPU & reuse the computerange() output.
    //       -> so best flip two equivalent buffers
    //     - GPU buffer begins with 'q' and ends with 'u', so can be handled locally inside the GPU code
    //  - we only need one sub-batch residual (GPU, managed by the main thread/0-stream) and one aggregated residual (CPU, not used beyond lifetime of thread lambda)
    //     - note: the GPU-side residual occupies a full model size
    // TODO: a test mode that simulates multiple nodes in one thread, to test the multi-level quantization stuff
    template<typename F1, typename F2, typename F3, typename F4, typename F5, typename F6, typename F7>
    void aggregate (
        F1 & allocatetransferbuffer,                    // (size_t size)
        F2 & quantizeandfetchsubbatchstripe,            // (size_t stripe, char * bufferbegin, size_t buffersize)
        F3 & syncfetchsubbatchstripe,                   // (size_t stripe)
        F4 & unquantizeandaggregatestripe,              // (size_t stripe, const char * bufferbegin, size_t buffersize, bool isfirst, bool islast)
        F5 & quantizeandassignaggregatedstripe,         // (size_t stripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled)
        F6 & assignaggregatedstripe,                    // (size_t stripe, const char * bufferbegin, size_t buffersize)
        F7 & syncassignaggregatedstripeandunquantize)   // (size_t stripe, size_t numstripes, const char * bufferbegin, size_t buffersize)
    {
        // if dynamic selection of #nodes has decided that we only want one node, then we skip this whole thing
        // TODO: for consistency, we should also check for #bits here (we could still force quantization even when using only 1 compute node)
        if (!isdistributed())
            return;

        const size_t k = node();
        const size_t K = nodes();

        // on how double buffering is realized with the 'localbuffers' variable and the bg thread:
        //  - only one buffer (localbuffers[] array) stored, in 'idlelocalbuffers'
        //  - we "check out" the buffer (take ownership) and run GPU quant on it
        //  - wait for current background thread; this will return its 'localbuffers' at time of when it was started (we don't wait if no bg thread)
        //     - bg thread's task is now idle again
        //  - start a new background thread; pass it in the current 'localbuffers' that we checked out
        //     - now the bg thread takes ownership of this set of 'localbuffers'
        //  - park the returned buffer in 'idlelocalbuffers' until next call into this function
        //  - (at exitmpicomputation(), wait for the thread to complete if any)

        // "check out" the buffer to operate on, that is, take ownership off our object into a local variable here
        // When double-buffering, there is a second buffer which is "floating" and in the hands (=local variable) of the bg thread; from where we will get it back when that thread is done.
        std::vector<mpistripebuffer> localbuffers;
        idlelocalbuffers.swap (localbuffers);   // check out the buffer (and push an empty one back)
        if (localbuffers.empty())               // lazy initialization of buffer
        {
            fprintf (stderr, "aggregate: lazy initialization of %d local data-transfer buffers (first size: %d bytes)\n", (int) mpistripebuffersizes.size(), (int) mpistripebuffersizes[0]); fflush (stderr);
            localbuffers.resize (mpistripebuffersizes.size());
            foreach_index (k, localbuffers)
                localbuffers[k].init (allocatetransferbuffer (k, mpistripebuffersizes[k]), mpistripebuffersizes[k]);
        }
        // We now have a "floating" buffer set 'localbuffers', and no 'idlelocalbuffers'.
        // Once the buffer set is done with, it will be deposited back into 'idlelocalbuffers'.

        if (forcedfullysyncoperation)       // force ongoing GPU ops to finish, before time measurement
            gpubarrier();
BEGINTIME("aggregate steps");

        // the all-reduce operation consists of these main steps:
        //  - step 1: kick off quantize and GPU-to-CPU transfer in required order (CPU only mode: just quantize)
        //  - step 2: exchange stripes with peers (MPI); aggregate the stripe we own (CPU); optionally perform part of fixed cost operations here
        //  - step 3: requantize (CPU) and redistribute across peers (MPI) and back to our own GPU
        //  - step 4: wait for all GPU transfers/unquantize ops to complete

        // step 1: kick off quantize and GPU-to-CPU transfer in required order (CPU only mode: just quantize)
        // We need to quantize and pull all stripes in the right order from the GPU so that they are ready for transfer.
BEGINTIME("[async] quantizeandfetchsubbatchstripe (all)");
        for (size_t i = 1; i <= K; i++)                     // GPU-to-CPU Sk |--2-|--3-|--0-|--1-|
        {
            size_t k1 = (k + i) % K;                        // stripe to transfer (we must do it in this precise ordering)
//BEGINTIME("[async] one quantizeandfetchsubbatchstripe");
            quantizeandfetchsubbatchstripe (k1, localbuffers[k1].data(), localbuffers[k1].size(), localbuffers[k1].header().mbframes/*out*/);
//ENDTIME();
        }
        fprintf (stderr, "aggregate: data exchange of %d frames between %d nodes\n", (int) localbuffers[0].header().mbframes, (int) K);

        if (forcedfullysyncoperation)
            gpubarrier();
ENDTIME();

        // steps 2 and 3 three run in the bg thread, so we put them into yet another lambds (make sure to capture by value, not reference)
        auto steps2and3 = [=] ()-> std::vector<mpistripebuffer>     // note: lambda makes a copy of the 'localbuffers' vector, but each buffer is a shared_ptr
        {
            const size_t K = nodes();                               // (for some reason these cannot be captured, compiler error?)
            // step 2: send and receive; aggregate
            //  - we computed a sub-batch gradient:
            //    -> send our own sub-batch gradient's stripes to all peers
            //  - we own accumulation of a stripe:
            //    -> receive sub-batch stripe we own from all peers
            //  - owned stripe == node id
            // first schedule all recvs()  --these will not return anything yet until sends() are scheduled, which we do next
            for (size_t i = 1; i < K; i++)                  // MPI_Isend Sk to Nk |idle|--2-|--3-|--0-| ; MPI_Irecv S1 from Nk |idle|--0-|--3-|--2-|
            {
                size_t kfrom = (k + K - i) % K;             // node to receive another sub-batch stripe k from (for the stripe we own, that is stripe k)
BEGINTIME("& [async] recvasync sub-batch");
                recvasync (peerbuffers[kfrom], kfrom/*node 'kfrom' is sending stripe 'k' at this point*/, i/*async op handle*/);
ENDTIME();
            }
            // now do the sends(), and interleaved with that, the posts to the GPU for unquant
            // TODO: should the high-pri unquant run after the low-pri quant? should we condition them on each other?
            bool isfirst = true;                            // for first one, we will reset the accumulator
            size_t aggmbframes = 0;                         // total mbframes in this sub-minibatch
            for (size_t i = 1; i < K; i++)                  // MPI_Isend Sk to Nk |idle|--2-|--3-|--0-| ; MPI_Irecv S1 from Nk |idle|--0-|--3-|--2-|
            {
                const size_t kto = (k + i) % K;                   // stripe to transfer (we must do it in the precise ordering), also node to transfer to
                const size_t kfrom = (k + K - i) % K;             // node to receive another sub-batch stripe k from (for the stripe we own, that is stripe k)

                // wait until we have received our own stripe that we want to send out
                // These have to happen in the same order as we requested them above.
BEGINTIME("& syncfetchsubbatchstripe (wait for GPU) (blocking)");
                syncfetchsubbatchstripe (kto);
ENDTIME();

                // send our sub-batch stripe 'kto' to the receiver node, all the while we
                // receive stripe k for sub-batch 'kfrom' from peer 'kfrom'
                // The stripe we receive has always the same stripe index (k) but comes from different nodes=different sub-batches.
                const size_t kto_from = (kfrom + i) % K;          // stripe to transfer (we must do it in the precise ordering), also node to transfer to
                assert (kto_from == k);                     // make sure we receive the correct stripe (actually easy to see symbolically that this is true)
string msg = msra::strfun::strprintf ("& [async] sendasync sub-batch of %d bytes", localbuffers[kto].size());
BEGINTIME(msg.c_str());
                sendasync (localbuffers[kto],  kto, i/*async op handle*/);
ENDTIME();
                // process all recvs() that have completed
                // If all goes well, there should be none for i==1, then one for each except for the last where we will block and shall get 2.
                // If, however, we have some laggard, then we will not be blocked by it until we got all others. In particular, our sends() will not be blocked.
                // In that case, however, our carefully chosen communication structure (one in one out stream at any give time) is broken.
                // Hopefully that won't make a huge difference, and only happen rarely yet work better than blocking earlier.
                const bool afterlastsendweblock = (i == K-1);              // if we did the last send then we shall wait for all pending receives
                size_t irecv;       // 'i' for the completed recv request if any
                while (recvwaitany (afterlastsendweblock/*blocking*/, irecv))
                {
                    const size_t kfrompending = (k + K - irecv) % K;            // node to receive another sub-batch stripe k from (for the stripe we own, that is stripe k)
BEGINTIME("& [async] unquantizeandaggregatestripe (GPU)");
                    // accumulate the received buffer
                    // If we do this on the GPU, then this will just schedule GPU operations (fast).
                    const auto & peerheader_kfrompending = *(mpistripeheader*)peerbuffers[kfrompending].data();
                    aggmbframes += peerheader_kfrompending.mbframes;            // number of frames in the gradient in this package
                    unquantizeandaggregatestripe (k, kfrompending, peerbuffers[kfrompending].data(), peerbuffers[kfrompending].size(), isfirst, false/*islast*/, 0/*mbframes for islast*/);
                    isfirst = false;                                            // from now on we will actually add stuff
ENDTIME();
                }
            }

            // we scheduled to receive our own stripe last since we don't need to pass it on; we still need to unquantize and aggregate it, though, like all others
BEGINTIME("& syncfetchsubbatchstripe (ourselves) (blocking)");
            syncfetchsubbatchstripe (k);
ENDTIME();
BEGINTIME("& [async] unquantizeandaggregatestripe, last (GPU ourselves)");
            if (forcedfullysyncoperation)
                gpubarrier();
            // TODO: we will cut in most of fixed-cost operations at this point; so pass on the information whether it's the last
            //       We wouldn't even want to send the last stripe (our own) to the CPU (maybe we must anyway for double buffering).
            aggmbframes += localbuffers[k].header().mbframes;
            unquantizeandaggregatestripe (k, k, localbuffers[k].data(), localbuffers[k].size(), isfirst, true/*islast*/, aggmbframes);
            if (forcedfullysyncoperation)
                gpubarrier();
ENDTIME();
            if (!isdoublebuffered())
                fprintf (stderr, "aggregate: data exchanged of total %d frames from %d nodes\n", (int) aggmbframes, (int) K);
            // now we have our own stripe k aggregated over all nodes, i.e. for the full minibatch as we need it

            // note: at this point, there is no GPU transfer ongoing; we can safely reuse those buffers now

            // step 3: requantize and redistribute
            //  - we have a full aggregate version of the stripe we own (k)
            //    -> send it to all other peers
            //  - receive all other stripes from the respective master peers
            //    -> and async-pass it on to our GPU after it was received (we immediately pass our own aggregate buffer back while we receive the next one)
            // For effiency, we do not recompute the quantization range from the aggregated (since that's expensive), but rather use the one from our own original sub-batch stripe.
            // We had transferred our own stripe into 'fetchbuffer' (of stripe [k]), and we use the same to send the aggregate back; so the quant ranges are already there.
            // ^^ DISABLED FOR NOW since impact yet to be verified.
            //    ^^ actually enabled in fact, made little difference for epoch 14+, need to test earlier behavior
            // TODO: When doing most of fixed cost here, we must reenable it (and can probably get rid of the option; at least leave it under the control of that function! No need to pass it in)
BEGINTIME("& quantizeandassignaggregatedstripe (GPU) (blocking)");
            if (forcedfullysyncoperation)
                gpubarrier();
            auto & localbuffers_k = const_cast<mpistripebuffer&>(localbuffers[k]);/*fix this*/
            // TODO: ^^ compiler thinks that localbuffers[] is 'const'--why? compiler error?
            localbuffers_k.header().mbframes = aggmbframes;    // this is the number of frames the aggregate gradient contains
            quantizeandassignaggregatedstripe (k, localbuffers_k.data(), localbuffers_k.size(), 0*   K/*reuserangescaled*/); // first quantize our owned stripe (which we want to send to all peers)
            if (forcedfullysyncoperation)
                gpubarrier();
ENDTIME();
            // Note: if this ^^ is done on a GPU, then this function performs a CPU-sync, i.e. the CPU-side buffer will contain the quantized stripe ready for use.

            // clean out all pending send() operations so that we can use the send buffer (localbuffers[]) again
            // Note that above we quantize into locabuffers[k] before sendwaitall(), but that's OK because buffer [k] is the one that we don't send to peers (there is no send pending for it).
            // By doing the sendwaitall() after quantize, we get some more concurrency (quantize is a blocking op that takes notable time).
            sendwaitall();

            // schedule the sends() and recvs() (these are all async, so this loop is quick)
            // TODO: makes no sense to schedule separately, combine. Seems slower, so is something wrong with the waitany protocol? Or the shift of sentwaitall()?
            for (size_t i = 1; i < K; i++)                      // MPI_Isend G1 to Nk |--2-|--3-|--0-|idle| ; MPI_Irecv Gk from Nk |--0-|--3-|--2-|idle| ; CPU-to-GPU |--1-|--0-|--3-|--2-|
            {
                size_t kto   = (k + i) % K;                     // node to transfer our own aggregate stripe k to
                size_t kfrom = (k + K - i) % K;                 // aggregate stripe to receive, also node to receive it from
                // send our aggregate quantized stripe k (in 'localbuffers[k]') to node 'kto', all the while we
                // receive the aggregate stripe 'kfrom' from node 'kfrom' (that node owns aggregation of that stripe)
                assert (kfrom != k);
BEGINTIME("& recvasync aggregate");
                recvasync (const_cast<mpistripebuffer&>(localbuffers[kfrom])/*fix this*/, kfrom, i/*async handle*/);
                // TODO: ^^ why is localbuffers[] 'const'??
ENDTIME();
string msg = msra::strfun::strprintf ("& sendasync aggregate of %d bytes", localbuffers[k].size());
BEGINTIME(msg.c_str());
                sendasync (localbuffers[k], kto/*is expecting stripe k*/, i/*async handle*/);
ENDTIME();
            }
            // and receive all in the order in which it comes in, and pass it on to the GPU
            size_t irecv;       // 'i' for the completed recv request if any
            while (recvwaitany (true/*blocking*/, irecv))
            {
                const size_t kfrom = (k + K - irecv) % K;                 // aggregate stripe to receive, also node to receive it from
                // we now have a fully aggregated stripe 'kfrom' available: pass it to ourselves (GPU) (note: this just initiates the data transfer to the CPU and returns immediately)
BEGINTIME("& [async] assignaggregatedstripe");
                assignaggregatedstripe (kfrom, localbuffers[kfrom].data(), localbuffers[kfrom].size());
ENDTIME();
            }
            // finally clean out all pending send() operations so that we can use the send buffer (localbuffers[]) again
            sendwaitall();
            return std::move(localbuffers);
        };
        // this is the end of what runs in the bg thread
        // note: 'localbuffers' has just been captured by lambda creation above

        // run steps 2 and 3 if not double-buffering; otherwise just get previous result (thread gets kicked off again later)
        // We must do this dance to keep the order of GPU submissions clean--unquant must be submitted before the next buffer gets transferred there.
BEGINTIME("steps2and3 wait (=time wasted from too slow bg thread) (blocking)");
        if (isdoublebuffered())                                 // wait for completion of bg thread of last call to this function and get its return value (=localbuffers variable at time of thread start)
            localbuffers = std::move (bgthread.get());
        else                                                    // no double buffering: we just run steps 2 and 3 right here
            localbuffers = std::move (steps2and3());
ENDTIME(); // steps2and3

        // step 4: wait for all GPU transfers/unquantize ops to complete (order is irrelevant, they all must finish)
        // This is 'u', i.e. happens right after 'q' but on previous operation
BEGINTIME("[async] syncassignaggregatedstripeandunquantize");    // should be 0
        for (size_t i = 0; i < K; i++)                          // (buffers were sent in this order to the ourselves, in case unquant (fast) overlaps with receiving)
        {
            size_t kfrom = (k + K - i) % K;                     // aggregate stripe to receive, also node we received it from
BEGINTIME("[async] one syncassignaggregatedstripeandunquantize");
            //if (!localbuffers.empty())                          // (first time it's empty in case of double-buffering)
            //    fprintf (stderr, "aggregate: unquantized gradient from %d frames on stripe %d\n", (int) localbuffers[kfrom].header().mbframes, kfrom);
            if (!localbuffers.empty())                          // (first time it's empty in case of double-buffering)
                syncassignaggregatedstripeandunquantize (kfrom, localbuffers[kfrom].data(), localbuffers[kfrom].size(), localbuffers[kfrom].header().mbframes);
            else
                syncassignaggregatedstripeandunquantize (kfrom, nullptr/*we have no buffer yet*/, 0, 0/*aggmbframes*/);
ENDTIME();
        }

        if (forcedfullysyncoperation)
            gpubarrier();
ENDTIME(); // unquant

        // and give the buffer back (buffer is idle now)
        idlelocalbuffers = std::move (localbuffers);

        // we still need to execute steps 2 and 3 if we use a the bg thread
        // We do that here as to make sure that the unquantize ops get scheduled to the GPU before anything on the bg thread
        // Note that the 'localbuffers' variable for use in it has been captured at lambda creation time; that's the one the thread will operate on.
BEGINTIME("[async] steps2and3 launch");
        if (isdoublebuffered())
            bgthread.run (steps2and3);

        if (forcedfullysyncoperation)
            gpubarrier();
ENDTIME(); // steps2and3
ENDTIME(); // aggregate steps
    }
};


// ---------------------------------------------------------------------------
// class acceleratedmatrixbase
// ---------------------------------------------------------------------------

// a matrix wrapper for accelerated matrix computation
//  - CUDA support:
//    - provides access to an array of CUDA matrices, one per device; can be a sub-stripe or full
//    - provides functions for moving data back and forth; they know to handle stripes
//    - provides a function to view a full copy as a stripe (it's a superset)
//    - restricts access to underlying CPU-side functions to classes that derive from this
//    - no caching or state tracking in this class, i.e. caller must avoid redundant data moves
//  - NUMA support:
//     - TODO: absorb cachedmatrix class into here entirely
//  - template argument can be matrix (with allocation) or matrixstriperef (no allocation)
template<class matrixbase> class acceleratedmatrixbase : protected matrixbase, public cudadistributedmatrix
{
protected:
    // move data to/from CUDA if in CUDA mode
    void alloccuda()
    {
        if (cudamode)               // allocate matrix memory in CUDA space (note: can be an empty matrix)
            cudadistributedmatrix::alloccuda (rows(), cols());
    }
#ifdef MULTICUDA
    void alloccuda (size_t deviceid)
    {
        if (cudamode)
            cudadistributedmatrix::alloccuda (rows(), cols(), deviceid);
    }
#endif
    friend class cudadistributedmatrix; // needs to access operator() and getcolstride() for syncto/fromcuda()
    void synctocuda (bool synchronously)
    {
        if (cudamode)
        {
            alloccuda();            // dimensions may have changed
            cudadistributedmatrix::synctocuda (*this, synchronously);
        }
    }
#ifdef  MULTICUDA
    void synctocuda (bool synchronously, size_t deviceid)
    {
        if (cudamode)
        {
            alloccuda(deviceid);
            cudadistributedmatrix::synctocuda (*this, synchronously, deviceid);
        }
    }
    void syncfromcuda (bool synchronously, size_t deviceid) const
    {
        if (cudamode)
            cudadistributedmatrix::syncfromcuda (*this, synchronously, deviceid);
    }

    // hack for striped mode in top layer[v-xieche]
    void syncfromcuda (bool synchronously, std::vector<size_t> &deviceids)
    {
        if (cudamode)
            cudadistributedmatrix::syncfromcuda (*this, synchronously, deviceids);
    }
    // used only when exit computation. [v-xieche]
    void syncfromcuda (bool synchronously, std::vector<size_t> &deviceids, cudastriping_t s)
    {
        if (cudamode)
            cudadistributedmatrix::syncfromcuda (*this, synchronously, deviceids, s);
    }

#endif
    void syncfromcuda (bool synchronously) const       // consider CPU-side copy mutable
    {
        if (cudamode)
            cudadistributedmatrix::syncfromcuda (*this, synchronously);
    }

public:
    // (see cudadistributedmatrix::makeinputstriping() for description)
    void makeinputstriping (cudastriping_t s) { cudadistributedmatrix::makeinputstriping (s, *this); }    // we pass our CPU-side matrix as the buffer
protected:
#ifdef STRIPEDTOPLAYER
public:  // temp code, need to be modified later![v-xieche]
    void makeinputstriping_multicuda () {cudadistributedmatrix::makeinputstriping_multicuda (*this);}
    void makeinputstriping_multicuda (std::vector<size_t> &devids) {cudadistributedmatrix::makeinputstriping_multicuda (*this, devids);}
protected:
#endif
    // merge partial sums (see cudadistributedmatrix for description)
    void sumacrossdevices (cudastriping_t s) { cudadistributedmatrix::sumacrossdevices (s, *this); }    // we pass our CPU-side matrix as the buffer

    // default constructor used in model matrix
    acceleratedmatrixbase() {}

    // used during construction only
    void resize (size_t n, size_t m)
    {
        matrix::resize (n, m);  // (only resizes CPU-side object)
        alloccuda();            // allocate CUDA-side object if in CUDA mode
    }
#ifdef MULTICUDA  // specify the cuda device when there are multi cuda device [v-xieche]
    void resize (size_t n, size_t m, size_t deviceid)
    {
        matrix::resize (n, m);
        alloccuda (deviceid);
        setDeviceId (deviceid);
    }
    // used for striped top layer only. [v-xieche]
    void resize (size_t n, size_t m, std::vector<size_t> deviceids)
    {
        matrix::resize (n, m);
        foreach_index (i, deviceids)  // keeps a full copy now. modity it later. [v-xieche]
            alloccuda (deviceids[i]);
        setDeviceId (deviceids[0]); // set deviceid with smallest device;
    }
#endif

    // move constructor used with stripe()
    acceleratedmatrixbase (acceleratedmatrixbase && other) : matrixbase (std::move (other)), cudadistributedmatrix (std::move (other)) {}

    // constructor for use with stripe()
    acceleratedmatrixbase (matrixbase && otherm, cudadistributedmatrix && otherc)
        : matrixbase (std::move (otherm)), cudadistributedmatrix (std::move (otherc)) {}

    // swap
    void swap (acceleratedmatrixbase & other) throw()
    {
        matrix::swap (other);
        cudadistributedmatrix::swap (other);
    }

    // assign to GPU directly without a CPU-side copy
    // Caller must do CPU-side assignment in its own control.
    // Caller must ensure correct size as well.
    template<typename MATRIX>
    void assigntocuda (const MATRIX & other, bool synchronously)
    {
        if (!cudamode)
            throw std::logic_error ("assigntocuda: must be called if not in CUDA mode");
        // no resize() since this may happen inside a stripe
        cudadistributedmatrix::synctocuda (other, synchronously);
    }

#if 0
    // constructor for use with template argument 'matrix'
    acceleratedmatrixbase() : cudamode (hascuda()), cudastriping (invalidstriping)
    {
        // allocate our copy (the base data structure) in CUDA space
        if (cudamode)
        {
            cudamatrix.reset (msra::cuda::newmatrix());
            cudamatrix->setdevice (0);  // currently we support only one device

            // FOR NOW: no striping supported. Remove this line once we actually set the striping mode.
            //setcudastriping (notstriped);
        }
    }

    // access to the CUDA matrix
    msra::cuda::matrix &       forcuda()       { checkcudamode(); checkcudadims(); return *cudamatrix.get(); }
    const msra::cuda::matrix & forcuda() const { checkcudamode(); checkcudadims(); return *cudamatrix.get(); }

    msra::cuda::matrix *       forcudaptr()       { checkcudadims(); return cudamatrix.get(); } // returns NULL if no CUDA mode
    const msra::cuda::matrix * forcudaptr() const { checkcudadims(); return cudamatrix.get(); }

    // ensure CUDA dimensions
    void checkcudadims() const
    {
        if (cudamode)
        {
            assert (rows() == cudamatrix->rows() && cols() == cudamatrix->cols());
            // make it a runtime check
            if (rows() != cudamatrix->rows() || cols() != cudamatrix->cols())
                throw std::logic_error ("acceleratedmatrixbase: CPU-side and CUDA-side matrix dimensions are not the same");
        }
    }
#endif
public:
    size_t rows() const { return matrixbase::rows(); }
    size_t cols() const { return matrixbase::cols(); }
    size_t colstride() const { return matrixbase::getcolstride(); }
    void reshape(const size_t newrows, const size_t newcols) { matrixbase::reshape(newrows, newcols);};
    bool empty() const { return matrixbase::empty(); }

    // used for secondary CUDA buffers: ensure matrix is large enough
    void ensuresize (size_t n/*rows*/, size_t m/*cols*/)
    {
        if (n > rows() || m > cols())
            resize (n, m);
    }

    // debug function to dump the value of the matrix out
    // Only dumps the first and last 3 elements in each dimension.
    // Note that this interferes in that it syncs the matrix back (fromcuda=true). If this changes results, then we also learned something.
    void glimpse (const char * name, bool fromcuda) const
    {
        fprintf (stderr, "### glimpsing at %s:     %d devices, valid=%s\n", name, fromcuda ? numcudadevices() : 0, cudastripingtostr (true).c_str());
        if (fromcuda)
            syncfromcuda (true);
        const matrix & us = *this;
        us.glimpse();
    }
};


// ---------------------------------------------------------------------------
// class rbmstatevectorsrefbase
// ---------------------------------------------------------------------------

// a reference RBM state vectors (including a sub-range) that reside in a rbmstatevectorsbase object
//  - implements all high-level operations needed inside the RBM class
//     - CUDA mode: these objects live entirely on the CUDA side; all operations happen inside CUDA
//     - NUMA mode: optimized parallelized (multi-threaded) matrix product implemented in this class
// TODO: this is a non-CUDA 'compatibility' implementation that exposes the underlying non-accelerated stripe
// TODO: move ALL computation to CUDA (!!)
// TODO: Also ensure that calls to fornuma() are only used when entering actual CPU-side computation.
// TODO: All other fornuma() calls (which are in CUDA mode) must be replaced by CUDA-side computation.
// TODO: Then we are done with CUDA-side computation.
template<class matrixbase> class rbmstatevectorsrefbase : public acceleratedmatrixbase<msra::math::ssematrixstriperef<matrixbase>>
{
    typedef acceleratedmatrixbase<msra::math::ssematrixstriperef<matrixbase>> acceleratedmatrix;
    rbmstatevectorsrefbase(){}

    mutable bool lockedforreading, lockedforwriting;
    static void failwithbadstate() { throw std::logic_error ("checklockstate: a rbmstatevectorsbase function was called in wrong state"); }
    static void checklockstate (bool lockstate) { if (!lockstate) failwithbadstate(); }
    static void checknotlockstate (bool lockstate) { if (lockstate) failwithbadstate(); }
    void checklockedforwriting() const { checklockstate (lockedforwriting); }
    void checknotlockedforwriting() const { checknotlockstate (lockedforwriting); }
    void checklockedforreading() const { checklockstate (lockedforreading); }
    void checklocked() const { checklockstate (lockedforreading || lockedforwriting); }
    void checkunlocked() const { checknotlockstate (lockedforreading || lockedforwriting); }
protected:
#if 1  // just for target propagation to use data in cuda device [v-xieche]
public:
#endif
    // lock/unlock for direct access of CPU-side ssematrix object
    // This is used to control the moving of data from/to CUDA.
    // Call lock() to be allowed to access operator(), and unlock() when done.
    void lockforreading() const // lockstate is mutable  --this allows this call on 'const' objects
    {
        checkunlocked();
        lockedforreading = true;
#ifdef MULTICUDA
        syncfromcuda (true, getDeviceId());
#else
        syncfromcuda (true);
#endif
    }

    // this <- this * thisscale + other * otherweight
    // If 'thisscale' is 0, the vector can be virgin memory.
    void gems (const float thisscale, const rbmstatevectorsrefbase & other, const float otherweight)
    {
        if (cudamode)
        {
#if 1
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, other, [thisscale, otherweight] (msra::cuda::matrix & us, msra::cuda::matrix & other) // do it on substreams for optimal model parallelism efficiency
            {
                us.gems (thisscale, other, otherweight);      // perform the sigmoid-derivate multiplication operation
            });
#else
            setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                 this->stripeforcudadevice (deviceid, stripedwrtrows)->gems (thisscale, *other.stripeforcudadevice (deviceid, stripedwrtrows), otherweight);
#endif
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            acceleratedmatrixbase::addweighted (thisscale, other, otherweight);
            checknan (us);
            us.unlock();
        }
    }

    // this <- a - b
    void settodiff (const rbmstatevectorsrefbase & a, const rbmstatevectorsrefbase & b)
    {
        gems (0.0f, a, 1.0f);       // this <- a
        gems (1.0f, b, -1.0f);      // this -= b
        //  yeah, this is not super-efficient, but can optimize later if needed
    }

    void addweighted (rbmstatevectorsrefbase & other, float otherweight = 1.0f)
    {
        // This function is broken for multi-GPU currently, but also not really used. So let's disable it for now & see if anyone ever hits it.
        throw logic_error ("addweighted: TO BE FIXED AND TESTED");   // remove this once this code has been tested once
#if 1   // TODO: to be tested
        gems (1.0f, other, otherweight);
#else
        if (cudamode)
        {
            // BUGBUG: This should be striped w.r.t. rows, not cols. It is only called from forwardprop() on Ph. Will be fixed automatically if we ever switch to just using gems().
            setoutputstriping (stripedwrtcols);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                 this->stripeforcudadevice (deviceid, stripedwrtcols)->addweighted (*other.stripeforcudadevice (deviceid, stripedwrtcols), otherweight);
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            acceleratedmatrixbase::addweighted (1.0f, other, otherweight);
            checknan (us);
            us.unlock();
        }
#endif
    }

    // [v-hansu] attention this use stripedwrtrows
    void addweightedrow (const rbmstatevectorsrefbase &other, const float weight)
    {
        if (cudamode)
        {
            setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                 this->stripeforcudadevice (deviceid, stripedwrtrows)->addweighted (*other.stripeforcudadevice (deviceid, stripedwrtrows), weight);
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            acceleratedmatrixbase::addweighted (1.0f, other, -1.0f);
            checknan (us);
            us.unlock();
        }
    }

#ifdef MULTICUDA
    // used for reading from one specified device. [v-xieche]
    void lockforreading (size_t deviceid) const
    {
        checkunlocked ();
        lockedforreading = true;
        syncfromcuda (true, deviceid);
    }
    size_t getcolstride() const { return matrixbase::getcolstride ();}
#endif
    void lockforwriting (bool sync = true)
    {
        checkunlocked();
        lockedforwriting = true;
        // make sure there is no copy-back action (synctocuda()) ongoing that we could interfere with
        // Default should be sync=true. If false, do a manual sync after unlock, to make sure we never catch this in-flight.
        if (sync)
            syncsynctocuda();
        // we will create this object --so no need to copy it from CUDA
    }
#ifdef  MULTICUDA
    void lockforwriting (size_t devid)
    {
        checkunlocked();
        lockedforwriting = true;
#ifndef ASYNCCOPY //removing sync - no-sync framework should be used for sync
        synchronize (getDeviceId()); 
#endif
    }
    // used for reading or writing on one specified device.[v-xieche]
    void unlock(size_t devid)
    {
        checklocked();
        if (lockedforwriting)       // if locked for writing then need to copy data to CUDA
        {
            synctocuda (false, devid);
        }
        lockedforreading = false;
        lockedforwriting = false;
    }
    //Actually do nothing seem it is used by const objective.[v-xieche]
    void unlock(size_t devid) const
    {
        checklockedforreading();
        checknotlockedforwriting();
        lockedforreading = false;
    }
#endif
    void lockforreadwrite()
    {
        lockforreading();
        lockedforwriting = true;    // causes it to sync back at the end
    }

    void unlock (bool sync = false)
    {
        checklocked();
        if (lockedforwriting)       // if locked for writing then need to copy data to CUDA
#ifdef MULTICUDA  // if we have multi cuda,we only hope to write data back to one cuda [v-xieche]
            synctocuda (sync, getDeviceId());
#else
            synctocuda (sync);
#endif
        lockedforreading = false;
        lockedforwriting = false;
    }

    // assign a matrix from CPU RAM  --don't lock it
    // TODO: rename this; as it misleads users into assigning from GPU to GPU
    template<class UMATRIX>
    void assign (const UMATRIX & other, bool sync)
    {
        checkunlocked();
        if (cudamode)
            acceleratedmatrixbase::assigntocuda (other, sync);
        else
            matrixbase::assign (other);  // just copy it locally
    }
#if 0
protected:
#endif

    void unlock() const
    {
        checklockedforreading();
        checknotlockedforwriting();
        lockedforreading = false;
    }
public:
    // to allow construction from rbmstatevectors::stripe()
    rbmstatevectorsrefbase (rbmstatevectorsrefbase && other) : acceleratedmatrix (std::move (other)), lockedforreading (false), lockedforwriting (false) { other.checkunlocked(); }

    // constructor for rbmstatevectors::stripe()
    // note: input matrix may be empty -> returns an empty matrix (will fail if ever accessed)
    rbmstatevectorsrefbase (msra::math::ssematrixstriperef<matrixbase> && stripem, cudadistributedmatrix && stripec)
        : acceleratedmatrix (std::move (stripem), std::move (stripec)), lockedforreading (false), lockedforwriting (false) { }

    // get underlying CPU-side matrixbase object, for NUMA operation (don't use with CUDA mode)
    msra::math::ssematrixstriperef<matrixbase> &       fornuma()       { checkunlocked(); return *this; }
    const msra::math::ssematrixstriperef<matrixbase> & fornuma() const { checkunlocked(); return *this; }

    // temp compat functions
    msra::math::ssematrixstriperef<matrixbase> &       fromcuda()       { checkunlocked(); checkcudadims(); syncfromcuda(); return *this; }
    const msra::math::ssematrixstriperef<matrixbase> & fromcuda() const { checkunlocked(); checkcudadims(); syncfromcuda(); return *this; }
    void tocuda() { checkunlocked(); checkcudadims(); synctocuda(); }

    // get underlying CUDA object
    //msra::cuda::matrix &       forcuda()       { checkunlocked(); return acceleratedmatrix::forcuda(); }
    //const msra::cuda::matrix & forcuda() const { checkunlocked(); return acceleratedmatrix::forcuda(); }

    // get a stripe without lock, for creating sub-views on a ref
    rbmstatevectorsrefbase stripe (size_t firstframe, size_t numframes) const
    {
        checkunlocked();
        const matrixbase & thismatrixbase = *this;  // we need to both down-slice and un-const it
        const cudadistributedmatrix & thiscudadistributedmatrix = *this;
        auto stripem = msra::math::ssematrixstriperef<matrixbase> (const_cast<matrixbase &> (thismatrixbase), firstframe, numframes);
        auto stripec = cudadistributedmatrix (const_cast<cudadistributedmatrix &> (thiscudadistributedmatrix), firstframe, numframes);
        return rbmstatevectorsrefbase (std::move (stripem), std::move (stripec));
    }

    // (see cudadistributedmatrix::makeinputstriping() for description)
    void makeinputstriping (cudastriping_t s)  { acceleratedmatrixbase::makeinputstriping (s); }
    void makeinputstriping (cudastriping_t s) const
    {
        // TODO: decide what is const and what is mutable...
        rbmstatevectorsrefbase * us = const_cast<rbmstatevectorsrefbase*> (this);
        us->makeinputstriping (s);
    }
#ifdef STRIPEDTOPLAYER
    void makeinputstriping_multicuda ()  { acceleratedmatrixbase::makeinputstriping_multicuda (); }
    void makeinputstriping_multicuda () const
    {
        // TODO: decide what is const and what is mutable...
        rbmstatevectorsrefbase * us = const_cast<rbmstatevectorsrefbase*> (this);
        us->makeinputstriping_multicuda ();
    }

    void makeinputstriping_multicuda (std::vector<size_t> &devids) 
    { 
        acceleratedmatrixbase::makeinputstriping_multicuda (devids); 
    }
    void makeinputstriping_multicuda (std::vector<size_t> &devids) const
    {
        // TODO: decide what is const and what is mutable...
        rbmstatevectorsrefbase * us = const_cast<rbmstatevectorsrefbase*> (this);
        us->makeinputstriping_multicuda (devids);
    }

#endif
    // merge partial sums
    void sumacrossdevices (cudastriping_t s) { acceleratedmatrixbase::sumacrossdevices (s); }

    // get underlying NUMA-cached object
    // TODO

    float &       operator() (size_t i, size_t j)       { checklockedforwriting(); return acceleratedmatrixbase::operator() (i, j); }
    const float & operator() (size_t i, size_t j) const { checklocked(); return acceleratedmatrixbase::operator() (i, j); }

    // operations
    bool hasnan (const char * name) const { checklocked(); return acceleratedmatrix::hasnan (name); }

    // perform a simple map operation in a sub-stream fashion (for overlapped processing in model parallelism)
    // TODO: can this be moved into a base class?
    template<typename FUNCTION>
    void applyonsubstreams (const stripingconfig & sc, FUNCTION op)
    {
        // process stripe by stripe
        setoutputstriping (stripedwrtrows);
        if (sc.enablesubbatchcomputation)
        {
            sc.foreachsubbatch ([&] (size_t ts, size_t te, size_t substream) -> void
            {
                for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                {
                    auto zstripe = stripeforcudadevice (deviceid, stripedwrtrows);
                    unique_ptr<msra::cuda::matrix> zstripesub (zstripe->patch (0, zstripe->rows(), ts, te));  // operate on a sub-range of frames
                    msra::cuda::onsubstream sub (*zstripesub.get(), substream);  // compute on this substream
                    op (*zstripesub);
                }
            });
        }
        else
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                op (*stripeforcudadevice (deviceid, stripedwrtrows));
    }
    // same for one argument ('zip' operation?)  --TODO: I hope we can do that much nicer, maybe with a nested lambda?
    template<typename FUNCTION, class OTHERVECTORS>
    void applyonsubstreams (const stripingconfig & sc, const OTHERVECTORS & other, FUNCTION op)
    {
        // process stripe by stripe
        setoutputstriping (stripedwrtrows);
        if (sc.enablesubbatchcomputation)
        {
            sc.foreachsubbatch ([&] (size_t ts, size_t te, size_t substream) -> void
            {
                for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                {
                    auto ostripe = const_cast<OTHERVECTORS &> (other).stripeforcudadevice (deviceid, stripedwrtrows);
                    auto zstripe = stripeforcudadevice (deviceid, stripedwrtrows);
                    unique_ptr<msra::cuda::matrix> ostripesub (ostripe->patch (0, ostripe->rows(), ts, te));  // operate on a sub-range of frames
                    unique_ptr<msra::cuda::matrix> zstripesub (zstripe->patch (0, zstripe->rows(), ts, te));  // operate on a sub-range of frames
                    msra::cuda::onsubstream sub (*zstripesub.get(), substream);  // compute on this substream
                    op (*zstripesub, *ostripesub);
                }
            });
        }
        else
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                op (*stripeforcudadevice (deviceid, stripedwrtrows), *other.stripeforcudadevice (deviceid, stripedwrtrows));
    }

    // this = sigmoid (this)
    void sigmoid()
    {
        if (cudamode)
        {
#undef PEEKCUDA    // for debugging
#ifdef PEEKCUDA
            fromcuda();
#endif
//#define TIME_CUDA     // used for model-parallelization experiments; remove this when those are done
#ifdef TIME_CUDA
            synchronize();
            auto_timer copycost;
#endif
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, [] (msra::cuda::matrix & us)
            {
                us.sigmoid();               // perform the sigmoid operation
            });
#ifdef TIME_CUDA
            synchronize();
            copycost.show ("sigmoid complete");
#endif
#ifdef PEEKCUDA
            fromcuda();
#endif
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_coord (i, j, us)
            {
                float exponent = us(i,j);
                if (exponent < -30.0f)
                    us(i,j) = 0.0f;
                else
                    us(i,j) = 1.0f / (1.0f + expf (-exponent));
#if 0
                if (_isnan (us(i,j)) || !_finite (us(i,j)))
                    fprintf (stderr, "sigmoid: NaN/INF detected for (%d,%d)\n", i, j);
#endif
            }
            checknan (us);
            us.unlock();
        }
    }

    // this = componentwise softplus (this)
    // softplus(x) = log(1+e^x)
    void softplus()
    {
        if (cudamode)
        {
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, [] (msra::cuda::matrix & us)
            {
                us.softplus();               // perform the sigmoid operation
            });
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_coord (i, j, us)
            {
                const float x = us(i,j);
                float f;
                if (x > 15.0f)                          // softplus(15) = 15.000000305902273713720485897019 = 15 in 'float' precision (exp(15) = 3.2 M, no overflow yet)
                    f = x;                              // this avoids overflows
                else
                    f = logf (1 + expf (x));
                us(i,j) = f;                            // in-place
            }
            checknan (us);
            us.unlock();
        }
    }

    // elementwise square 
    // this = a**2 (elementwise)
    // TODO implement striping
    void setsquare (const rbmstatevectorsrefbase & other)
    {
        if (cudamode)
        {
            // process stripe by stripe
            setoutputstriping (stripedwrtrows);
            other.makeinputstriping(notstriped);
            makeinputstriping(notstriped);
            // TODO hack !! there must be a way to do this on must be a way to do this un multiple GPUs, but how access the stripe of other?
            msra::cuda::matrix & stripeout = forcudadevice(0, notstriped);
            const msra::cuda::matrix & stripein = other.forcudadevice(0, notstriped);
            stripeout.elementwisesquare(stripein);
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            other.lockforreading();
            foreach_coord (i, j, us)
            {
                us(i,j) = other(i,j) * other(i,j);
            }
            us.unlock();
            other.unlock();
        }
    }

    // bianarize activation vectors using 0.5 as the threshold [v-xieche]
    void binarize()
    {
        auto & us = *this;
        us.lockforreadwrite ();
        fprintf (stderr, "binarizing the output of hidden layer when updating top layer...\n");
        foreach_coord (i, j, us)
        {
            if (us(i, j) > 0.5)  us(i, j) = 1.0;
            else us(i, j) = 0.0;
        }
        us.unlock ();
    }

    // used for (a) experiments with sparseness and (b) RLUs
    void setto0ifbelow (const float value)
    {
        if (cudamode)
        {
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, [value] (msra::cuda::matrix & us)
            {
                us.setto0ifbelow (value);
            });
#if 0   // old version, delete this
            // BUGBUG: should be stripedwrtrows
            setoutputstriping (stripedwrtcols);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                 this->stripeforcudadevice (deviceid, stripedwrtcols)->setto0ifbelow (value);
#endif
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            size_t counter = 0;
            size_t hitnum = 0;
            foreach_coord (i, j, us)
            {
                counter ++;
                if (us(i,j) < value) 
                {
                    us(i,j) = 0;
                    hitnum ++;
                }
            }
            float ratio = float(100.0) * hitnum / counter;
         //   fprintf (stderr, "setto0ifbelow: %.2f%% set to zero\n", ratio);  
            us.unlock ();
        }
    }

    void setto0ifabsbelow2 (rbmstatevectorsrefbase & ref, float threshold)
    {
        if (cudamode)
        {
            // BUGBUG: should be stripedwrtrows
            setoutputstriping (stripedwrtcols);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                 this->stripeforcudadevice (deviceid, stripedwrtcols)->setto0ifabsbelow2 (*ref.stripeforcudadevice (deviceid, stripedwrtcols), threshold);
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            ref.lockforreading();
            acceleratedmatrix::setto0ifabsbelow2(ref, threshold);
            checknan (us);
            ref.unlock();
            us.unlock();
        }
    }

    // add epsilon in the output of sigmoid, then exert log funtion on it. log (us + epison). [v-xieche]
    void addepisonlog()  
    {
        auto & us = *this;
        us.lockforreadwrite ();
        foreach_coord (i, j, us)
            us (i, j) = (float) log (us(i,j) + EPISONFORLOG);
        us.unlock ();
    }

    void getorisigmoid()  // calculate the sigmoid function from log ( epison + s(z) ). [v-xieche]
    {
        auto & us = *this;
        us.lockforreadwrite ();
        foreach_coord (i, j, us)
            us(i, j) = (float) (exp(us(i, j)) - EPISONFORLOG);
        us.unlock();
    }

    // special function for RBM pre-training: sample binary values according to probability
    // P is the probability that the bit should be 1.
    void samplebinary (const rbmstatevectorsrefbase & P, unsigned int randomseed)
    {
        if (cudamode)
        {
            // for now: do it on all devices for identical results (otherwise too trick to deal with randomseed)
            P.makeinputstriping (notstriped);
            // process stripe by stripe
            setoutputstriping (notstriped);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                forcudadevice (deviceid, notstriped).samplebinary (P.forcudadevice (deviceid, notstriped), randomseed);
            setoutputstriping (stripedwrtrows); // downgrade (this would come out of a parallelized version)
#if 0       // test code  --must be identical
            const auto & res = *this;
            P.lockforreading();
            res.lockforreading();
            foreach_column (t, res)
            {
                srand (randomseed + (unsigned int) t);  // (note: srand() is thread-safe)
                //fprintf (stderr, "samplebinary: seed = %d\n", randomseed + (unsigned int) t);
                foreach_row (j, res)
                {
                    float randval = rand() / (float) RAND_MAX;
                    float bit = randval < P(j,t) ? 1.0f : 0.0f;
                    if (res(j,t) != bit)
                        fprintf (stderr, "MISMATCH at %d,%d\n", j, t);
                }
            }
            res.unlock();
            P.unlock();
#endif
        }
        else
        {
            auto & res = *this;
            if (&P != &res)
            {
                P.lockforreading();
                res.lockforwriting();
            }
            else
                res.lockforreadwrite();     // in-place
            foreach_column (t, res)
            {
                srand (randomseed + (unsigned int) t);  // (note: srand() is thread-safe)
                //fprintf (stderr, "samplebinary: seed = %d\n", randomseed + (unsigned int) t);
                foreach_row (i, res)
                {
                    float randval = ::rand() / (float) RAND_MAX;
                    float bit = randval < P(i,t) ? 1.0f : 0.0f;
                    res(i,t) = bit;
                }
            }
            res.unlock();
            if (&P != &res) // (otherwise both are the same, we read-write lock only once)
                P.unlock();
        }
    }

private:

    double colvecsum()  // sum over all elements of a column vector
    {
        const rbmstatevectorsrefbase & us = *this;
        assert (us.cols() == 1);
        us.lockforreading();
        double sum = 0.0;
        foreach_row (i, us)
            sum += us[i];
        us.unlock();
        return sum;
    }
    double rowvecsum()  // sum over all elements of a row vector
    {
        const rbmstatevectorsrefbase & us = *this;
        assert (us.rows() == 1);
        us.lockforreading();
        double sum = 0.0;
        foreach_column (j, us)
            sum += us(0,j);
        us.unlock();
        return sum;
    }
    double allelementssum()  // sum over all elements of a row vector
    {
        const rbmstatevectorsrefbase & us = *this;
        us.lockforreading();
        double sum = 0.0;
        foreach_coord (i, j, us)
            sum += us(i,j);
        us.unlock();
        return sum;
    }

    size_t numpositive() // count number of elements > 0 in a matrix
    {
        const rbmstatevectorsrefbase & us = *this;
        us.lockforreading();
        size_t sum = 0;
        foreach_coord (i, j, us)
            if (us(i,j) > 0)
                sum++;
        us.unlock();
        return sum;
    }

public:

    // statistics for pre-training
    // Measures reconstruction likelihood of 'this' against 'v1' (v1=reconstructed)
    // This uses temp vectors glogllsums and logllsums to compute it independently for each node.
    // We compute both in all cases because Dong's original metric was the Gaussian one in either case.
    // TODO: Reduce to one metric, and take a 'gaussian' flag.
    // Then it sums up the values. This is to allow the main computation to run in CUDA space.
    // It did otherwise consume the same amount of time as all the remaining computation!
    // TODO: change return values from -sum to av-
    void llstats (const rbmstatevectorsrefbase & v1, rbmstatevectorsrefbase & glogllsums, rbmstatevectorsrefbase & logllsums, double & /*out*/glogllsum, double & /*out*/logllsum) const
    {
        const rbmstatevectorsrefbase & v = *this;

        assert (v.rows() == v1.rows() && v.cols() == v1.cols());
        assert (glogllsums.rows() == v.rows() && glogllsums.cols() == 1);
        assert (logllsums.rows() == v.rows() && logllsums.cols() == 1);

        // first compute it per node, summing up over frames
        // (This strange split is done to allow for CUDA acceleration.)
        if (cudamode)
        {
            // process stripe by stripe
            glogllsums.setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                v.stripeforcudadevice (deviceid, stripedwrtrows)->llstats (*v1.stripeforcudadevice (deviceid, stripedwrtrows), *glogllsums.stripeforcudadevice (deviceid, stripedwrtrows), true);
            logllsums.setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                v.stripeforcudadevice (deviceid, stripedwrtrows)->llstats (*v1.stripeforcudadevice (deviceid, stripedwrtrows), *logllsums.stripeforcudadevice (deviceid, stripedwrtrows), false);
        }
        else
        {
            v.lockforreading();
            v1.lockforreading();
            logllsums.lockforwriting();
            glogllsums.lockforwriting();

            checknan (v);
            checknan (v1);

            foreach_column (t, v)
            {
                // gaussian: we compute the Gaussian metric also for binary units, as it seems to be common...
                foreach_row (i, v)
                {
                    if (t == 0)
                        glogllsums[i] = 0.0;
                    double diff = v(i,t) - v1(i,t);
                    double glogll = -0.5 * diff * diff;         // note that we assume unit variance
                    // We normalize against the 'perfect reconstruction' hypothesis (diff == 0)
                    // thus the Gaussian normalization factor (1/sqrt (2.0 * M_PI)) cancels out.
                    glogllsums[i] += (float) glogll;
                }
                //fprintf (stderr, "llstat: glogll[%3d] = %7.4f\t   ->\t%7.4f\t%7.4f\t%7.4f\t  vs.\t%7.4f\t%7.4f\t%7.4f ...\n", t, glogll, v(0,t), v(1,t), v(2,t), v1(0,t), v1(1,t), v1(2,t));

                // binary: expected log prob of reconstruction, expectation over input data
                foreach_row (i, v)
                {
                    if (t == 0)
                        logllsums[i] = 0.0;
                    double Pv = v(i,t);     // prob of v being 1
                    if (Pv < 0.000001) Pv = 0.000001;   // to be sure (not observed)
                    if (Pv > 0.999999) Pv = 0.999999;
                    double Pv1 = v1(i,t);   // prob of v1 being 1
                    if (Pv1 < 0.000001) Pv1 = 0.000001;   // we do see 1.0
                    if (Pv1 > 0.999999) Pv1 = 0.999999;
                    double logll = Pv * log (Pv1) + (1 - Pv) * log (1 - Pv1);
                    // normalize against perfect reconstruction hypothesis for better readability
                    logll -= Pv * log (Pv) + (1 - Pv) * log (1 - Pv);
                    logllsums(i,0) += (float) logll;
                }
            }
            glogllsums.unlock();
            logllsums.unlock();
            v.unlock();
            v1.unlock();
        }

        // compute the sum
        logllsum = logllsums.colvecsum();
        glogllsum = glogllsums.colvecsum(); // Gaussian (also for binary units, for diagnostics)
    }

    // special function for backprop: multiply error vector by derivative of sigmoid function.
    //   err = eh .* h .* (1 - h)
    //   h .* (1 - h) = derivative of sigmoid
    // where err and eh are the same variable (updated in place)
    // We leverage that the derivative can be computed from values of the sigmoid function in 'sigm' cheaply.
    void mulbydsigm (const rbmstatevectorsrefbase & sigm)
    {
        if (cudamode)
        {
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, sigm/*other*/, [] (msra::cuda::matrix & us, msra::cuda::matrix & other) // do it on substreams for optimal model parallelism efficiency
            {
                us.mulbydsigm (other);      // perform the sigmoid-derivate multiplication operation
            });
        }
        else
        {
            auto & eh = *this;
            eh.lockforreadwrite();
            sigm.lockforreading();
            // suggestion by Fahlman (cf. Quickprop algorithm): add 0.1 to derivative to avoid flat tails
            // This does not seem to converge faster, but rather more slowly. Maybe I need to let it run longer.
#if 0
            foreach_coord (i, t, eh)
                eh(i,t) *= (0.1f + 0.9f * sigm(i,t) * (1.0f - sigm(i,t)));
#else
            foreach_coord (i, t, eh)
                eh(i,t) *= sigm(i,t) * (1.0f - sigm(i,t));
#endif
            sigm.unlock();
            eh.unlock();
        }
    }

    // special function for backprop: multiply error vector by derivative of ReLU function.
    //   relu(z) = z if z > 0, 0 else
    //   drelu(z)/dz = 1 if z > 0, 0 if z < 0, and undefined if z = 0
    // z > 0 can be tested by testing h = relu(z) > 0 which is what we do
    // i.e.
    //   err = eh .* (h > 0)
    // where err and eh are the same variable ('this', updated in place)
    void mulbydlru (const rbmstatevectorsrefbase & reluz)
    {
        if (cudamode)
        {
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, reluz/*other*/, [] (msra::cuda::matrix & us, msra::cuda::matrix & other) // do it on substreams for optimal model parallelism efficiency
            {
                us.mulbydlru (other);
            });
#if 0   // old, delete this
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtrows)->mulbydlru (*reluz.stripeforcudadevice (deviceid, stripedwrtrows));
#endif
        }
        else
        {
            auto & eh = *this;
            eh.lockforreadwrite();
            reluz.lockforreading();
            foreach_coord (i, t, eh)
                if (reluz(i,t) <= 0.0f)  // err = eh .* (h > 0) in place
                    eh(i,t) = 0.0f;
            reluz.unlock();
            eh.unlock();
        }
    }

    // special function for backprop: multiply error vector by derivative of softplus function.
    // We leverage that the derivative can be computed from values of the softplus function in 'softp' cheaply.
    void mulbydsoftplus (const rbmstatevectorsrefbase & softp)
    {
        if (cudamode)
        {
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, softp/*other*/, [] (msra::cuda::matrix & us, msra::cuda::matrix & other) // do it on substreams for optimal model parallelism efficiency
            {
                us.mulbydsoftplus (other);
            });
        }
        else
            throw std::logic_error ("mulbydsoftplus: not implemented yet in CPU mode");
    }

    // special function for backprop: multiply error vector by derivative of ReLU function.
    //   relu(z) = z^1/rootorder) if z > 0, -that * leakinessratio else
    //   drelu(z)/dz = 1 if z > 0, 0 if z < 0, and undefined if z = 0
    // i.e.
    //   err = eh .* XXXX
    // where err and eh are the same variable ('this', updated in place)
    // f(x) = (x+1)^1/r for x > 0; else that*(-leakiness)

    // these two are code dup from the CUDA kernel--TODO: eliminate code dup somehow
    float leakyroot (float x, size_t rootorder, float leakiness)
    {
        float ax = fabs (x);
        float y = pow (ax + 1, 1.0f/rootorder) - 1;
        if (x <= 0.0f)
            y *= -leakiness;
        return y;
    }
    float dleakyroot (float y, size_t rootorder, float leakiness)
    {
        float ay = y;           // reconstruct the positive y (before applying sign)
        if (y <= 0.0f)
            ay /= -leakiness;
        // derivative of f(x) = (x+1)^1/r is
        // df/dx = 1/r * (x+1)^(1/r-1)
#if 0
        float x = pow (ay + 1, rootorder) - 1;   // now we know the x, and it is non-negative; we can compute the derivative now
        float d = pow (x + 1, 1.0f/rootorder - 1.0f) / rootorder;
#else
        float d = pow (ay + 1, 1.0f - rootorder) / rootorder;
#endif
        if (y <= 0.0f)
            d *= leakiness;
        return d;
    }

    void leakyroot (size_t rootorder, float leakiness)
    {
        if (cudamode)
        {
            // BUGBUG: change this to the same code structure as mulbydsigm()
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
#ifdef _DEBUG
            this->syncfromcuda(true);   // debug
#endif
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtrows)->leakyroot (rootorder, leakiness);
#ifdef _DEBUG
            this->syncfromcuda(true);   // debug
#endif
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_coord (i, t, us)
                us(i,t) = leakyroot (us(i,t), rootorder, leakiness);
            checknan (us);
            us.unlock();
        }
    }

    void mulbydleakyroot (const rbmstatevectorsrefbase & lruvals, size_t rootorder, float leakiness)
    {
        if (cudamode)
        {
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
#ifdef _DEBUG
            this->syncfromcuda(true);   // debug
#endif
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtrows)->mulbydleakyroot (*lruvals.stripeforcudadevice (deviceid, stripedwrtrows), rootorder, leakiness);
#ifdef _DEBUG
            this->syncfromcuda(true);   // debug
#endif
        }
        else
        {
            auto & eh = *this;
            eh.lockforreadwrite();
            lruvals.lockforreading();
            foreach_coord (i, t, eh)
                eh(i,t) = eh(i,t) * dleakyroot (lruvals(i,t), rootorder, leakiness);  // err = eh .* derivative
            lruvals.unlock();
            eh.unlock();
        }
    }

    // special function for backprop: multiply error vector by derivative of maxout (which is 1.0)
    // but that means for dropout SGD error singals flows through zeroed units, need to test whether that's
    // desired, hence this silly function
    void mulbydmaxout (const rbmstatevectorsrefbase & h)
    {
        if (cudamode)
        {
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtrows)->mulbydmaxout (*h.stripeforcudadevice (deviceid, stripedwrtrows));
        }
        else
        {
            auto & eh = *this;
            eh.lockforreadwrite();
            h.lockforreading();

            foreach_coord (i, t, eh)
                if ( h(i,t) == 0 ) eh(i,t) = 0.0f;

            h.unlock();
            eh.unlock();
        }
    }

    // norms = diag (this' * this)
    // Result stored as a column vector rather than diagonal matrix.
    void columnnormsquares (rbmstatevectorsrefbase & norms) const
    {
        const auto & us = *this;
        assert (norms.cols() == us.cols() && norms.rows() == 1);
#if 0
        if (cudamode)
        {
            this->setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                throw::logic_error ("dotprodwrtcols: cuda verstion to be finished.");
        }
#endif
        {
            us.lockforreading();
            norms.lockforwriting();

            foreach_column (t, norms)
                matrix::dotprod (col(t), col(t), norms(0,t)); // norms[t] = ||colt||^2

            norms.unlock();
            us.unlock();
        }
    }

    // this = M * [diag (weightvector) + addconst * I]
    // weightvector is stored as a row vector, i.e. a matrix of 1 row.
    void scaledcolumns (/*const*/ rbmstatevectorsrefbase & M, const rbmstatevectorsrefbase & weightvector, const float addconst)
    {
        auto & us = *this;
        assert (M.cols() == weightvector.cols() && weightvector.rows() == 1);
        assert (us.cols() == M.cols() && us.rows() == M.rows());

        // if (cudamode) ... else
        {
            us.lockforwriting();
            M.lockforreading();
            weightvector.lockforreading();

            foreach_column (t, weightvector)
            {
                auto ust = us.col(t);
                const auto Mt = M.col(t);
                ust.addweighted (0.0f, Mt, weightvector(0,t) + addconst);   // us(.,)t = M(.,t) * [weightvector[t] + addconst]
            }

            weightvector.unlock();
            M.unlock();
            us.unlock();
        }
    }

    void KhatriRaoProduct(const rbmstatevectorsrefbase & m1, const rbmstatevectorsrefbase & m2)
    {
        assert(m1.cols() == m2.cols() && cols() == m1.cols());

        if (cudamode)
        {
            // BUGBUG: should be stripedwrtrows
            this->setoutputstriping (stripedwrtcols);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtcols)->KhatriRaoProduct (*m1.stripeforcudadevice (deviceid, stripedwrtrows), *m2.stripeforcudadevice (deviceid, stripedwrtrows));
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            m1.lockforreading();
            m2.lockforreading();

            matrixbase::KhatriRaoProduct(m1, m2);

            m1.unlock();
            m2.unlock();
            us.unlock();
        }
    }

    void reorderForConvolutional (rbmstatevectorsrefbase & to, const msra::cuda::convolutionParams &convParams) const
    {
        if (!cudamode)
            throw runtime_error ("reorderForConvolutional: must be called in cuda mode");
        const_cast<rbmstatevectorsrefbase*>(this)->setoutputstriping (notstriped);
        // process stripe by stripe
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            this->forcudadevice (deviceid).reorder(to.forcudadevice(deviceid), convParams);
    }

    void maxpoolForward (rbmstatevectorsrefbase & out, rbmstatevectorsrefbase & maxIndex, const msra::cuda::convolutionParams &convParams) const
    {
        if (!cudamode)
            throw runtime_error ("maxpoolForward: must be called in cuda mode");
        const_cast<rbmstatevectorsrefbase*>(this)->setoutputstriping (notstriped);
        // process stripe by stripe
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            const_cast<rbmstatevectorsrefbase*>(this)->forcudadevice (deviceid).maxpoolForward(out.forcudadevice(deviceid), maxIndex.forcudadevice(deviceid), convParams);
    }
    void maxpoolBack (rbmstatevectorsrefbase & out, const rbmstatevectorsrefbase & maxIndex, const msra::cuda::convolutionParams &convParams) const
    {
        if (!cudamode)
            throw runtime_error ("maxpoolBack: must be called in cuda mode");
        const_cast<rbmstatevectorsrefbase*>(this)->setoutputstriping (notstriped);
        // process stripe by stripe
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            const_cast<rbmstatevectorsrefbase*>(this)->forcudadevice (deviceid).maxpoolBack(out.forcudadevice(deviceid), maxIndex.forcudadevice(deviceid), convParams);        
    }

    // dump a matrix, done from CUDA side via Printf()
    void dump (char *name) const
    {
    
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
    		const_cast<rbmstatevectorsrefbase*>(this)->forcudadevice (deviceid).dump(name); 	   
        //auto & hnew = *this;
        //hnew.lockforreading();
        //printmatf(name, hnew);
        //hnew.unlock();
    }
    
    //   this = reshape each column of eh from (K1xK2,1) to (K1, K2) and times each column of h (K2, frames).
    //   the output is a (K1, frames) matrix
    //   eh can be transposed.
    //   used for tensor DNN
    void reshapecolumnproduct (const rbmstatevectorsrefbase & eh, const rbmstatevectorsrefbase & h, const bool isehtransposed)
    {
        assert(eh.cols() == h.cols() && cols() == h.cols());
        assert (eh.rows() == h.rows() * rows());

        if (cudamode)
        {
            // BUGBUG: should be stripedwrtrows, no?
            this->setoutputstriping (stripedwrtcols);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtcols)->reshapecolumnproduct (*eh.stripeforcudadevice (deviceid, stripedwrtrows), *h.stripeforcudadevice (deviceid, stripedwrtrows), isehtransposed);
        }
        else
        {
            auto & hnew = *this;
            hnew.lockforreadwrite();
            eh.lockforreading();
            h.lockforreading();

            matrixbase::reshapecolumnproduct(eh, h, isehtransposed);

            h.unlock();
            eh.unlock();
            hnew.unlock();
        }
    }

    // function: us = us / log (h + sigmoid) [v-xieche]
    void divideaddsigmoid (const rbmstatevectorsrefbase & sigm)
    {
        auto & us = *this;
        us.lockforreadwrite ();
        sigm.lockforreading ();
        foreach_coord (i, t, us)   
            us (i,t) /= (float)  (EPISONFORLOG + sigm(i, t));   // it now be s(z) + epison, don't need to add log again.
        us.unlock ();
        sigm.unlock ();
    }

#ifdef STRIPEDTOPLAYER 
    void softmax (std::vector<size_t> &devids)
    {
        if (!cudamode)
            throw runtime_error ("softmax: must be called in cuda mode");
        makeinputstriping_multicuda (devids);
        for (size_t i = 0; i < devids.size(); i++)
            this->forcudadevice (devids[i], notstriped).softmax();
    }
#endif
    // this = softmax (this)
    // Softmax is amazingly expensive, compared to matrix product. It is important that this is optimized correctly.
    void softmax (acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & softmaxbuffer)
    {
        if (cudamode)
        {
            // The last output of matprod_mtm() is row-striped. For optimal model parallelism,
            // we compute each row stripe.
            // This leads to incorrect normalization. Thus, in addition, the CUDA softmax function returns for each stripe
            // the (log of) the sum of all its numerators (for each frame). This way we can post-correct:
            //  - compute total sum of all its numerators
            //  - correct each stripe by (* its sum / total sum)
            // We need extra storage of one per frame per device, called 'softmaxbuffer.'
            // Note that for efficiency reasons (and I think some code in the past could not copy the matrix otherwise),
            // we store it as one column per device, where frames are rows (this is against our usual concention).
            // I.e. it is a col-striped matrix of (#frames, #devices) elements.

            stripingconfig sc (*this);
            if (sc.K > 1)
            {
                makeinputstriping (stripedwrtrows);     // (should already be)
                setoutputstriping (stripedwrtrows);     // we compute it for sub-vectors
                softmaxbuffer.ensuresize (cols(), numcudadevices());    // one column per GPU (frames are rows not cols--this is for efficiency and for being able to transfer the matrix)
                softmaxbuffer.setoutputstriping (notstriped);
                // compute partial softmax() for each stripe (sub-vector), one per GPU
                for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                    stripeforcudadevice (deviceid, stripedwrtrows)->stripedsoftmaxstep1 (softmaxbuffer.forcudadevice (deviceid, notstriped), deviceid);
                softmaxbuffer.setoutputstriping (stripedwrtcols);       // step 1 has actually only filled in columns, we set it not before because we still pass the whole matrix, somewhat violating our usual patterns
                if (sc.K > 1)   // if >1 GPU, values are only locally normalized; correct this now
                {
                    // exchange all those vectors
                    softmaxbuffer.makeinputstriping (notstriped);       // synchronous; that's OK since it's not a lot of data (mbsize 1024 -> 4096 bytes per GPU)
                    // fix up partial softmax() for each stripe (sub-vector)
                    for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                        stripeforcudadevice (deviceid, stripedwrtrows)->stripedsoftmaxstep2 (softmaxbuffer.forcudadevice (deviceid, notstriped), deviceid);
                }
                //makeinputstriping (notstriped);   // for timing tests: this will force a full output-dim exchange across all GPUs, so we can measure the saving from not doing it here
            }
            else
            {
#ifndef STRIPEDTOPLAYER
                makeinputstriping (notstriped); // this is VERY expensive if #devices > 1, hence the optimization above
#else
                makeinputstriping_multicuda();
#endif
                // compute it on each device in full
                for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                    this->forcudadevice (deviceid, notstriped).softmax();
            }
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_column (j, us)
            {
                // find max (to bring exp() into comfortable range)
                float colmax = -999999.9f;  // adagrad will make this value to big negtive number, e.g. -100
                foreach_row (i, us)
                {
                    float usij = us(i,j);
                    if (usij > colmax)
                        colmax = usij;
                }
                // sum
                // we divide by exp (colmax), which will cancel out when we normalize below
                double sum = 0.0;
                foreach_row (i, us)
                {
                    float usexp = exp (us(i,j)-colmax);
#if 0
                    if (_isnan (usexp) || !_finite (usexp))
                        fprintf (stderr, "softmax: NaN/INF detected for (%d,%d)\n", i, j);
#endif
                    us(i,j) = usexp;
                    sum += usexp;
                }
                // normalize
                float sumf = float (sum);
                foreach_row (i, us)
                {
                    us(i,j) /= sumf;
                }
            }
            checknan (us);
            us.unlock();
        }
    }

#ifdef STRIPEDTOPLAYER
    void seterrorsignal (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu, std::vector<size_t> &devids)
    {
        float pruningbeam = 0.0f;   // off the best (not considering correct state)

        if (cudamode && pruningbeam == 0.0f)
        {
            // This is a little tricky. Because setbackpropagationerrorsignal() compares the row index with uids[],
            // this breaks with striped rows. As a simple solution, we compute identical error signals in all devices.
            // TODO: Do this more efficiently (just pass a base index).

#ifdef STRIPEDTOPLAYER
            Pu.makeinputstriping_multicuda (devids);
#else
            Pu.makeinputstriping (notstriped);
#endif
            this->setoutputstriping (notstriped);
            // process stripe by stripe
            for (size_t i = 0; i < devids.size(); i++)
                this->forcudadevice (devids[i], notstriped).setbackpropagationerrorsignal (uids.forcudadevice (devids[i], notstriped), Pu.forcudadevice (devids[i], notstriped), 0);
        }
        else
        {
            throw runtime_error ("seterrorsignal: should call seterrorsignal function with striped in cuda mode!");
        }
    }
#endif

    // this = delta((Pu(i,t)==uids[t]) - Pu
    // What this is: (log softmax)'_j(z) = delta(s(t)==j)-softmax_j(z)
    // This is the error of the top layer, the signal being back-propagated.
    void seterrorsignal (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu, const rbmstatevectorsrefbase & senone2update)
    {
        // experimental pruning mode (disabled by default)  --note: pruning currently hurts bigtime. Need to normalize?
        float pruningbeam = 0.0f;   // off the best (not considering correct state)

        if (cudamode && pruningbeam == 0.0f)
        {
#ifdef STRIPEDTOPLAYER
            Pu.makeinputstriping_multicuda ();
#else
            Pu.makeinputstriping (stripedwrtrows);
#endif
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                size_t s0, nr;      // row range available on this device
                onedevicedim (deviceid, true, rows(), s0, nr);
                // note: uids[t] and senone2update[s] are really single-row vectors
                // TODO: senone2update[] really should better be a column vector--more memory efficient, and s is usually a row index
                this->stripeforcudadevice (deviceid, stripedwrtrows)
                    ->setbackpropagationerrorsignal (uids.forcudadevice (deviceid, notstriped),                     // (0,t) ground truth label sequence
                                                     *Pu.stripeforcudadevice (deviceid, stripedwrtrows), s0,  // (s-s0,t) posteriors (row stripe, starting with senone s0)
                                                     senone2update.forcudadevice (deviceid, notstriped));           // (0,s) not striped: it's a global config
            }
        }
        else
        {
            auto & us = *this;
            assert (cols() == uids.cols() && cols() == Pu.cols());
            assert (rows() == Pu.rows() && uids.rows() == 1);

            us.lockforwriting();
            uids.lockforreading();
            Pu.lockforreading();

            size_t numpruned = 0;
            foreach_column (t, us)
            {
                // pruning: we determine the maximum Pu(i,t), not considering the correct state.
                float maxPu_t = 0.0f;
                if (pruningbeam > 0)
                {
                    foreach_row (i, us)
                    {
                        const size_t uid = (size_t) uids(0,t);
                        if (i == uid)   // we only consider competitors
                            continue;
                        if (Pu(i,t) > maxPu_t)
                            maxPu_t = Pu(i,t);
                    }
                }

                // set the target weights for this frame
                // Each row of W will be nudged towards v(t) by the target weight.
                // For the correct state, the target weight is 1.0-P, otherwise -P, i.e.
                // for incorrect states, rows will be gently nudged away from v(t).
                // Setting the target weight to 0 will leave the row of W unchanged.
                foreach_row (i, us)
                {
                    const size_t uid = (size_t) uids(0,t);
                    if ((float) uid != uids(0,t))
                        throw std::runtime_error ("seterrorsignal: uids not integer!");
                    const float utarget_it = (i == uid) ? 1.0f : 0.0f;
                    us(i,t) = utarget_it - Pu(i,t);

                    // do the pruning
                    // We always nudge towards the correct state, but exclude low-scoring competitors.
                    if (i != uid && Pu(i,t) < maxPu_t * pruningbeam)
                    {
                        us(i,t) = 0.0f; // setting it to 0 will exclude it from having an effect
                        numpruned++;
                    }
                }
            }

            if (numpruned > 0)
                fprintf (stderr, "seterrorsignal: %d out of %d pruned (%.2f%%) (pruningbeam param = %.2f)\n",
                numpruned, us.rows()*us.cols(), 100.0 * numpruned / (us.rows()*us.cols()), pruningbeam);

            Pu.unlock();
            uids.unlock();
            checknan (us);
            us.unlock();
            //us.dump("set error signal");
        }
    }

    // this = (1-alpha)* delta((Pu(i,t)==uids[t]) + alpha*refPu(i,t) - Pu
    // Same as seterrorsignal(), but replacing 1/0 reference by linear interpolation of 1/0 reference and the posterior from the reference model.
    // I.e. keep it a little similar to the reference model.
    void seterrorsignalwithklreg (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu, const rbmstatevectorsrefbase & refPu, const float alpha)
    {
        // experimental pruning mode (disabled by default)  --note: pruning currently hurts bigtime. Need to normalize?
        float pruningbeam = 0.0f;   // off the best (not considering correct state)

        if (cudamode && pruningbeam == 0.0f)
        {
            // This is a little tricky. Because setbackpropagationerrorsignal() compares the row index with uids[],
            // this breaks with striped rows. As a simple solution, we compute identical error signals in all devices.
            // TODO: implement the speed-up of seterrorsignal() here as well, or, rather, merge the two into a single function
            Pu.makeinputstriping (notstriped);
            refPu.makeinputstriping (notstriped);
            this->setoutputstriping (notstriped);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, notstriped).setbackpropagationerrorsignalwithklreg (uids.forcudadevice (deviceid, notstriped), Pu.forcudadevice (deviceid, notstriped), 
                                                                                                   refPu.forcudadevice (deviceid, notstriped), alpha);
        }
        else
        {
            auto & us = *this;
            assert (cols() == uids.cols() && cols() == Pu.cols());
            assert (rows() == Pu.rows() && uids.rows() == 1);

            us.lockforwriting();
            uids.lockforreading();
            Pu.lockforreading();
            refPu.lockforreading();

            size_t numpruned = 0;
            const float oneminusalpha=1-alpha;
            foreach_column (t, us)
            {
                // pruning: we determine the maximum Pu(i,t), not considering the correct state.
                float maxPu_t = 0.0f;
                if (pruningbeam > 0)
                {
                    foreach_row (i, us)
                    {
                        const size_t uid = (size_t) uids(0,t);
                        if (i == uid)   // we only consider competitors
                            continue;
                        if (Pu(i,t) > maxPu_t)
                            maxPu_t = Pu(i,t);
                    }
                }

                // set the target weights for this frame
                // Each row of W will be nudged towards v(t) by the target weight.
                // For the correct state, the target weight is 1.0-P, otherwise -P, i.e.
                // for incorrect states, rows will be gently nudged away from v(t).
                // Setting the target weight to 0 will leave the row of W unchanged.
                foreach_row (i, us)
                {
                    const size_t uid = (size_t) uids(0,t);
                    if ((float) uid != uids(0,t))
                        throw std::runtime_error ("seterrorsignal: uids not integer!");
                    float utarget_it = (i == uid) ? 1.0f : 0.0f;
                    if (alpha>0) utarget_it = oneminusalpha*utarget_it + alpha*refPu(i,t);
                    us(i,t) = utarget_it - Pu(i,t);

                    // do the pruning
                    // We always nudge towards the correct state, but exclude low-scoring competitors.
                    if (i != uid && Pu(i,t) < maxPu_t * pruningbeam)
                    {
                        us(i,t) = 0.0f; // setting it to 0 will exclude it from having an effect
                        numpruned++;
                    }
                }
            }

            if (numpruned > 0)
                fprintf (stderr, "seterrorsignal: %d out of %d pruned (%.2f%%) (pruningbeam param = %.2f)\n",
                numpruned, us.rows()*us.cols(), 100.0 * numpruned / (us.rows()*us.cols()), pruningbeam);

            refPu.unlock();
            Pu.unlock();
            uids.unlock();
            checknan (us);
            us.unlock();
        }
    }

    // set error according to hsmoothingweight and errorsettingmode
    //  - uids: ground truth
    //  - Pu: CE frame posterior
    //  - refmat: latice frame posterior or error, depending on errorsettingmode
    // errorsettingmode 0 : refmat is gammas, this(s,t) = (1-hsmoothingweight) * ((s==uids[t]) - Pu(s,t)) + hsmoothingweight * ((s==uids[t]) - gammas(s,t))
    //                                                  = (s==uids[t]) - ((1-hsmoothingweight) * Pu(s,t) + hsmoothingweight * gammas(s,t))
    // errorsettingmode 1 : refmat is errors, this(s,t) = (1-hsmoothingweight) * ((s==uids[t]) - Pu(s,t)) + hsmoothingweight * errors(s,t)
    // errorsettingmode 2 : refmat is errors, this(s,t) = (1-hsmoothingweight) * ((s==uids[t]) - Pu(s,t)) * Pu(s,t) + hsmoothingweight * errors(s,t)
    // zhaorui pass frame dropping thresh to error calculation funtion
    void seterrorsignalhsmoothing (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu, const rbmstatevectorsrefbase & refmat,
                                   const float hsmoothingweight, const size_t errorsettingmode, const float framedropthresh)
    {
        if (cudamode)
        {
            fprintf (stderr, "seterrorsignalhsmoothing: making input striping not striped\n");
            Pu.makeinputstriping (notstriped);
            refmat.makeinputstriping (notstriped);
            fprintf (stderr, "seterrorsignalhsmoothing: done making input striping not striped\n");
            this->setoutputstriping (notstriped);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, notstriped).setbackpropagationerrorsignalhsmoothing (uids.forcudadevice (deviceid, notstriped), Pu.forcudadevice (deviceid, notstriped), refmat.forcudadevice (deviceid, notstriped),
                                                                                                    hsmoothingweight, errorsettingmode,framedropthresh);
            fprintf (stderr, "seterrorsignalhsmoothing: done setting the error signal\n");
        }
        else
            throw std::logic_error ("seterrorsignalhsmoothing not finished for NUMA mode");
    }

    // set to Hessian vector signal, depending on the the output of the softmax layer Pu and the forwarded Hessian vector statistics finalforwardstatistics
    // this = (diag(Pu) - Pu * Pu') * finalworwardstatistics
    //      = (finalworwardstatistics .* Pu) (elementwise) - dot(Pu, finalworwardstatistics) * Pu
    void sethessianvectorsignal (const rbmstatevectorsrefbase & Pu, const rbmstatevectorsrefbase & finalforwardstatistics)
    {
        if (cudamode)
        {
            // TODO think about multi GPU usage later ...
            assert(numcudadevices() == 1);
            Pu.makeinputstriping (notstriped);
            finalforwardstatistics.makeinputstriping(notstriped);
            this->setoutputstriping (notstriped);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, notstriped).sethessianvectorsignal (Pu.forcudadevice (deviceid, notstriped), finalforwardstatistics.forcudadevice (deviceid, notstriped));
        }
        else
        {
            auto & us = *this;
            assert (rows() == Pu.rows());
            Pu.lockforreading();
            us.lockforwriting();
            
            foreach_column (t, us)
            {
                // TODO use dot product method here                
                float dotproduct = 0.0f;
                foreach_row (i, us)
                {
                    dotproduct += Pu(i,t) * finalforwardstatistics(i,t);
                }
                foreach_row (i,us)
                {
                    us(i,t) = (finalforwardstatistics(i,t) - dotproduct) * Pu(i,t);
                }
            }

            Pu.unlock();
            us.unlock();
        }
    }


    // drop frames by consolidating surviving frames at the start of the slice
    // A frame is dropped if keepsampleflags[t] is 0.0.
    void dropframes (const rbmstatevectorsrefbase & keepsampleflags)
    {
#if 0
        if (cudamode && pruningbeam == 0.0f)
        {
            keepsampleflags.makeinputstriping (notstriped);
            this->setoutputstriping (notstriped);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, notstriped).dropframes (keepsampleflags.forcudadevice (deviceid, notstriped));
        }
        else
#endif
        {
            auto & us = *this;
            assert (cols() == keepsampleflags.cols() && keepsampleflags.rows() == 1);

            us.lockforreadwrite();
            keepsampleflags.lockforreading();

            size_t t1 = 0;
            foreach_column (t, us)
            {
                if (keepsampleflags(0,t) == 0.0f)
                    continue;
                if (t1 < t)
                    us.col(t1).assign (us.col(t));
                t1++;
            }

            keepsampleflags.unlock();
            us.unlock();
        }
    }

#ifdef STRIPEDTOPLAYER // try to implement the code for posteriorstats in striped mode. [v-xieche]
    double rowvecsum(size_t devid)  // sum over all elements of a row vector
    {
        const rbmstatevectorsrefbase & us = *this;
        assert (us.rows() == 1);
        us.lockforreading (devid);
        double sum = 0.0;
        foreach_column (j, us)
            sum += us(0,j);
        us.unlock(devid);
        return sum;
    }

    size_t numpositive(size_t devid) // count number of elements > 0 in a matrix
    {
        const rbmstatevectorsrefbase & us = *this;
        us.lockforreading(devid);
        size_t sum = 0;
        foreach_coord (i, j, us)
            if (us(i,j) > 0)
                sum++;
        us.unlock(devid);
        return sum;
    }
    void stripedposteriorstatssumvectors (rbmstatevectorsrefbase & logpps, rbmstatevectorsrefbase & pps, rbmstatevectorsrefbase & fcors, double & /*out*/avlogpp, double & /*out*/avpp, double & /*out*/avfcor, std::vector<size_t> &devids) const
    {
        if (!cudamode)
        {
            throw runtime_error ("can't call stripedposteriorstats in non-cuda mode.");
        }
        const auto & uids = *this;
        const size_t n = uids.cols();
        // aut &pu = Pu.

    }
#endif

    // statistics for tracking progress of backpropagation
    // called as: fu.posteriorstats (Pu, buf1, buf2, buf3, avlogpp, avpp, avfcor);
    // Measures av. log PP, av. PP, and av. frames correct against ground truth ('this' as a row vector of indices).
    // This uses temp row vectors to compute it independently for each time point (using CUDA), and then sums up the result.
    // If 'nosoftmax', then Pu has not had softmax() applied to it. We can still count errors, but not return meaningful pp values.
    // BUGBUG: This still produces some unseen log pp values.
    static inline bool isweird (double val) { return *(1+(int*) &val) == 0x7fffffff; }  // 7ff_16 is used to represent inf (if M=0) and NaNs (if M!=0),
    void posteriorstats (const rbmstatevectorsrefbase & Pu,
                         rbmstatevectorsrefbase & logpps, rbmstatevectorsrefbase & pps, rbmstatevectorsrefbase & maxlogpps, // buffers
                         double & /*out*/avlogpp, double & /*out*/avpp, double & /*out*/avfcor, bool nosoftmax) const
    {
        const auto & uids = *this;
        const size_t n = uids.cols();
        assert (n == Pu.cols());
        if (n == 0)     // special case which may arise for data parallelism
        {
            avlogpp = avpp = avfcor = 0.0;
            return;
        }

        if (cudamode)
        {
            // we first aggregate over row stripes, and finally merge the results
            Pu.makeinputstriping (stripedwrtrows);
            if (logpps.rows() != numcudadevices() || pps.rows() != numcudadevices() || maxlogpps.rows() != numcudadevices())
                throw logic_error ("posteriorstats: buffers must have been allocated as one row per GPU");
            logpps.setoutputstriping (stripedwrtrows); // must have been allocated as one row per GPU
            pps.setoutputstriping (stripedwrtrows);
            maxlogpps.setoutputstriping (stripedwrtrows);
#ifdef MULTICUDA
            logpps.setDeviceId (getDeviceId());
            pps.setDeviceId (getDeviceId());
            maxlogpps.setDeviceId (getDeviceId());
            // BUGBUG: this function is no longer correct for MULTICUDA
            throw logic_error ("posteriorstats: MULTICUDA mode has not been updated in a total rewrite of this function");
#endif
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                size_t s0, nr;      // row range available on this device
                onedevicedim (deviceid, true, Pu.rows(), s0, nr);
                this->forcudadevice (deviceid, notstriped)                                              // 'this' is the ground truth sequence
                    .posteriorstats (*Pu.stripeforcudadevice (deviceid, stripedwrtrows), s0, nosoftmax, // Pu is the posteriors (a row stripe, first row represents senone s0)
                                     *logpps.stripeforcudadevice (deviceid, stripedwrtrows),            // buffers (one row per GPU)
                                     *pps.stripeforcudadevice (deviceid, stripedwrtrows),
                                     *maxlogpps.stripeforcudadevice (deviceid, stripedwrtrows));
            }
            const auto & clogpps = logpps;  // (stupid check requires (,) to operate on a 'const' object when locked for reading)
            const auto & cpps = pps;
            const auto & cmaxlogpps = maxlogpps;
            clogpps.lockforreading();
            cpps.lockforreading();
            cmaxlogpps.lockforreading();
            avlogpp = 0.0;
            avpp = 0.0;
            avfcor = 0.0;
//fprintf (stderr, "posteriorstats: version with NaN detector\n");
            foreach_column (t, clogpps)                     // process all frames
            {
                // compute log pp and pp
                float logpp = -1e30f;
                float pp = 0.0f;
                float maxlogpp = -1e30f;
                foreach_row (dev, clogpps)                  // loop over GPUs
                {
                    if (_isnan (clogpps(dev,t)) || isweird (clogpps(dev,t)) || isweird (cpps(dev,t)))            // WORKAROUND: for unknown reasons, we sometimes get a NaN here (last observed for ReLUs with too large MB size)
                        fprintf (stderr, "posteriorstats: log PP is NaN for frame %d on GPU %d\n", t, dev), fflush (stderr);
                    else if (clogpps(dev,t) >= logpp)
                    {
                        logpp = clogpps(dev,t);             // ground-truth state log PP --only one of the row stripes (GPUs) will have != -1e30 here
                        pp = cpps(dev,t);                   // ground-truth state PP (not really used anymore except for being logged)
                    }
                    if (_isnan (cmaxlogpps(dev,t)) || isweird (cmaxlogpps(dev,t)))
                        fprintf (stderr, "posteriorstats: max log PP is NaN for frame %d on GPU %d\n", t, dev), fflush (stderr);
                    else if (cmaxlogpps(dev,t) > maxlogpp)  // max state log PP over all senones --if equal to ground-truth state, then it's a correct classification
                        maxlogpp = cmaxlogpps(dev,t);
                }
                if (logpp == -1e30f)                        // WORKAROUND: none found? we seem to get this sometimes, unclear why; at least warn about it
                    fprintf (stderr, "posteriorstats: BUGBUG: no log PP value for frame %d\n", t), fflush (stderr); // I observe that when it happens, t is always a multiple of 32
                else
                {
                    avlogpp += logpp / n;
                    avpp += pp / n;
                    // count correct frames
                    if (logpp >= maxlogpp)  // maximum-scoring state is same as ground-truth state: correct frame
                        avfcor += 1.0 / n;
                }
            }
            // gotta get this nasty 1.#R--what the hell is that!!
//fprintf (stderr, "posteriorstats: avpp = %.2f, in hex 0x%08x%08x\n", avpp, *(int*) &avpp, *(1+(int*) &avpp)), fflush (stderr);
            maxlogpps.unlock();
            pps.unlock();
            logpps.unlock();
        }
        else
        {
            // This is a weirdly inefficient implemetenation that emulates some old CUDA version. Make it nicer if you care.
            uids.lockforreading();
            Pu.lockforreading();
            logpps.lockforwriting();
            pps.lockforwriting();
            maxlogpps.lockforwriting();

            assert (Pu.cols() == n);
            assert (uids.rows() == 1);
            assert (logpps.cols() == n && pps.cols() == n && maxlogpps.cols() == n);
            assert (logpps.rows() == 1 && pps.rows() == 1 && maxlogpps.rows() == 1);
            checknan (Pu);
            const float zero = nosoftmax ? -1e30f : 0.0f;
            foreach_column (t, uids)
            {
                const size_t clsid = (size_t) uids(0,t);
                assert ((float) clsid == uids(0,t));
                const float pp = Pu(clsid,t);
                pps(0,t) = nosoftmax ? 0.0f : pp;                               // nosoftmax: we don't have the pps; return 0
                logpps(0,t) = nosoftmax ? pp : logf (max (pp, 0.000001f));      // (avoid underflow if prob has been rounded to 0)
                // which is the max?
                maxlogpps(0,t) = pp;            // non-null indicates assumption that it is correct
                foreach_row (i, Pu)
                    if (i != clsid && Pu(i,t) >= pp)
                        maxlogpps(0,t) = zero;  // assumption was wrong
            }
            // get aggregates over all frames
            float totallogpp = 0.0f;
            float totalpp = 0.0f;
            size_t totalfcors = 0;
            const size_t numcols = Pu.cols();
            for (size_t t = 0; t < numcols; t++)
            {
                totallogpp += logpps(0,t);
                totalpp += pps(0,t);
                if (maxlogpps(0,t) > zero)  // ground-truth's prob survived as best: it is correctly decided
                    totalfcors++;
            }

            maxlogpps.unlock();
            pps.unlock();
            logpps.unlock();
            Pu.unlock();
            uids.unlock();

            avlogpp = totallogpp / n;
            avpp = totalpp / n;
            avfcor = (double) totalfcors / (double) n;
        }
    }

    // dropout() --randomly set X% of values in this matrix to 0
    void dropout (float factor, unsigned int randomseed)
    {
        if (cudamode)
        {
#if 1       // seems to work
fprintf (stderr, "dropout: %.2f  %d\n", factor, (int) randomseed);
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, [factor, randomseed] (msra::cuda::matrix & us)
            {
                us.dropout (factor, randomseed);
            });
#else
            // UNTESTED --remove this comment once tested
            // for now: do it on all devices for identical results (otherwise too tricky to deal with randomseed)
            makeinputstriping (notstriped);
            // process stripe by stripe
            setoutputstriping (notstriped);xyz
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                forcudadevice (deviceid, notstriped).dropout (factor, prescale, randomseed);
            setoutputstriping (stripedwrtrows); // downgrade (this would come out of a parallelized version)
#endif
        }
        else
        {
            // UNTESTED --remove this comment once tested
            auto & us = *this;
            us.lockforreadwrite();     // in-place
            foreach_row (i, us)
            {
                srand (randomseed + (unsigned int) i);  // (note: srand() is thread-safe)
                foreach_column (j, us)
                {
                    float randval = ::rand() / (float) RAND_MAX;
                    if (randval < factor)
                        us(i,j) = 0.0f;
                }
            }
            us.unlock();
        }
    }

    // scale() --multiply a matrix in place with a factor
    void scale (float factor)
    {
        if (cudamode)
        {
#if 1       // just scale in whatever form it is (if we use applyonsubstreams() but call scale() out of sequence, which we do for dropout pre-scale, it will trigger some error check)
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                stripeforcudadevice (deviceid, stripedwrtrows)->scale (factor);
#else
            stripingconfig sc (*this);      // configuration for model parallelism is determined in here
            applyonsubstreams (sc, [factor] (msra::cuda::matrix & us)
            {
                us.scale (factor);
            });
#endif
        }
        else
        {
            // UNTESTED --remove this comment once tested
            auto & us = *this;
            us.lockforreadwrite();     // in-place
            us.scale (factor);
            us.unlock();
        }
    }

    // meanvarnorm() --perform mean/variance normalization on every frame
    //  out(i,j) = (this(i,j) - mean(i)) / diagvar(i)
    // Result gets written to a different vector.
    void meanvarnorm (const acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & mean, bool subtractmean, const acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & diagvar, rbmstatevectorsrefbase & out) const
    {
        if (cudamode)
        {
            //mean.makeinputstriping ();    ...this may be broken w.r.t. dimensions
            // TODO: use applyonsubstreams()?
            out.setoutputstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                stripeforcudadevice (deviceid, stripedwrtrows)->meanvarnorm (*mean.stripeforcudadevice (deviceid, stripedwrtrows), subtractmean,
                                                                             *diagvar.stripeforcudadevice (deviceid, stripedwrtrows),
                                                                             *out.stripeforcudadevice (deviceid, stripedwrtrows));
#ifdef _DEBUG
            this->glimpse ("in", true);
            mean.glimpse ("mean", true);
            diagvar.glimpse ("diagvar", true);
            out.glimpse ("out", true);
#endif
//                        hstripesub->addtoallcolumns (*astripesub.get());
        }
        else
        {
            throw std::logic_error ("meanvarnorm: not implemented for CPU mode");
        }
    }

    // accumulate all frames in 'this' into meanacc and varacc
    // If 'add' then add to existing content.
    void meanvaracc (bool add, acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & meanacc,
                     const acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & mean, acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & varacc) const
    {
        // compute the row sum
        if (cudamode)
        {
            //mean.makeinputstriping ();    ...this may be broken w.r.t. dimensions
            // TODO: use applyonsubstreams()?
            meanacc.setoutputstriping (stripedwrtrows);
            varacc.setoutputstriping  (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                stripeforcudadevice (deviceid, stripedwrtrows)->meanvaracc (add,
                                                                            *meanacc.stripeforcudadevice (deviceid, stripedwrtrows),
                                                                            *mean.stripeforcudadevice (deviceid, stripedwrtrows),
                                                                            *varacc.stripeforcudadevice (deviceid, stripedwrtrows));
        }
        else    // non-CUDA version
        {
            throw std::logic_error ("meanvaracc: not implemented for NUMA mode");
        }
    }
};


// ---------------------------------------------------------------------------
// class rbmstatevectorsbase
// ---------------------------------------------------------------------------

// matrix to hold RBM state vectors (input and activations; multiple vectors in time sequence)
//  - CUDA mode: stored entirely in CUDA memory, no outside computation allowed (except set input and get posteriors)
//  - DBN-level operations are implemented on this
//  - RBM-internal operations are implemented through rbmstatevectorsrefbase
//     - user get a ref first
//     - then execute operation on that object
// TODO: lock mechanism should return the ref
template<class matrixbase> class rbmstatevectorsbase : public acceleratedmatrixbase<msra::math::ssematrix<matrixbase>>
{
    typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;
public:
    // get a stripe without lock, for passing stripes from DBN to RBM
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes)
    {
        auto stripem = msra::math::ssematrixstriperef<matrixbase> (*this, firstframe, numframes);
        auto stripec = cudadistributedmatrix (*this, firstframe, numframes);
        return rbmstatevectorsref (std::move (stripem), std::move (stripec));
    }
#ifdef MULTICUDA  // for multi devices, we should copy the data from the required device. [v-xieche]
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes, size_t devid)
    {
        auto stripem = msra::math::ssematrixstriperef<matrixbase> (*this, firstframe, numframes);
        auto stripec = cudadistributedmatrix (*this, firstframe, numframes, devid);
        return rbmstatevectorsref (std::move (stripem), std::move (stripec));
    }
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes, size_t devid) const
    {
        auto res = const_cast<rbmstatevectorsbase*> (this)->stripe (firstframe, numframes, devid);
#ifdef MULTICUDA
        res.setDeviceId (devid);
#endif
        return res;
    }
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes, std::vector<size_t> devids)
    {
        auto stripem = msra::math::ssematrixstriperef<matrixbase> (*this, firstframe, numframes);
        auto stripec = cudadistributedmatrix (*this, firstframe, numframes, devids);
        return rbmstatevectorsref (std::move (stripem), std::move (stripec));
    }
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes, std::vector<size_t> devids) const
    {
        auto res = const_cast<rbmstatevectorsbase*> (this)->stripe (firstframe, numframes, devids);
        return res;
    }
#endif
    /*const*/ rbmstatevectorsref stripe (size_t firstframe, size_t numframes) const
    {
        auto res = const_cast<rbmstatevectorsbase*> (this)->stripe (firstframe, numframes);
        return res;
    }

    // get a stripe with a lock, for initializing and getting results out in DBN
    class lockforwriting : public rbmstatevectorsrefbase<matrixbase>
    {
		bool syncatend;	// set if instantiated with 'sync'=false; then, we ensure not to overwrite things by post-syncing rather than pre-syncing
    public:
        lockforwriting (rbmstatevectorsbase & m, size_t firstframe, size_t numframes, bool sync = true) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes)), syncatend (!sync) { rbmstatevectorsrefbase::lockforwriting (sync); }
#ifdef MULTICUDA  // specify one device to lockforwriting.[v-xieche]
        lockforwriting (rbmstatevectorsbase & m, size_t firstframe, size_t numframes, size_t devid) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes, devid)), syncatend (false) { setDeviceId(devid); rbmstatevectorsrefbase::lockforwriting(devid); }
#endif
		~lockforwriting() { unlock (syncatend); }
    };
    class lockforreading : public rbmstatevectorsrefbase<matrixbase>
    {
    public:
        lockforreading (const rbmstatevectorsbase & m, size_t firstframe, size_t numframes) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes)) {/*setDeviceId(m.getDeviceId());*/  rbmstatevectorsrefbase::lockforreading(); }
#ifdef MULTICUDA // specify one device to lockforreading [v-xieche]
        lockforreading (const rbmstatevectorsbase & m, size_t firstframe, size_t numframes, size_t devid) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes, devid)) { setDeviceId(devid); rbmstatevectorsrefbase::lockforreading(devid); }
#endif 
        ~lockforreading() { unlock(); }
    };

    // these two functions need to be there to allow vector<>::resize() and the likes, but we may never call those
    rbmstatevectorsbase (const rbmstatevectorsbase & other)   // will not copy, just construct acceleratedmatrixbase empty
    {
        if (!empty())
            throw std::logic_error ("rbmstatevectorsbase: cannot assign");
        if (cudamode)
        {
            other.checkvalidstriping (notstriped);
            setcudastriping (notstriped);
        }
    }
    void operator= (rbmstatevectorsbase &&) { throw std::logic_error ("rbmstatevectorsbase: cannot assign"); }

    // needed for the layerstate/errorstate vectors' initial resize()
    rbmstatevectorsbase() { if (cudamode) setcudastriping (notstriped); }

    // use during construction only
    void resize (size_t n, size_t m) { acceleratedmatrixbase::resize (n, m); }
#ifdef MULTICUDA
    void resize (size_t n, size_t m, size_t devid) {acceleratedmatrixbase::resize (n, m, devid); }

    // only used for striped top layer for now[v-xieche]
    void resize (size_t n, size_t m, std::vector<size_t> &devids) {acceleratedmatrixbase::resize (n, m, devids);}
#endif
};


// ---------------------------------------------------------------------------
// class cachedmatrixbase
// ---------------------------------------------------------------------------

// temporary storage for matrices used in accelerated computation
//  - NUMA mode: NUMA-node local copies
//  - CUDA mode: no longer used for this
// Although it is temporary, we keep the memory around across calls
// (that's what 'cached' in the name is supposed to indicate).
// TODO: better name... bufferedmatrix?
// Note: the CUDA side of this is not in here. Can we merge this with acceleratedmatrixbase?? Do it when I run non-CUDA again.
template<class matrixbase> class cachedmatrixbase   // TODO: move this inside acceleratedmatrixbase
{
    typedef msra::math::ssematrix<matrixbase> matrix;
    // NUMA version
    std::vector<matrix> numacopies;  // [numanodeid]
#if 0
    // CUDA version
    // we currently only support one device
    unique_ptr<msra::cuda::matrix> cudamatrix;  // CUDA-side copy/copies of the data
#endif
public:
    // NUMA version only
    // note: these are only supposed to be called from acceleratedmatrixbase
    // TODO: we should really make this a class inside there
    void allocate_numa (size_t n, size_t m)
    {
        // NUMA version
        size_t numnodes = msra::numa::getnumnodes();    // NUMA nodes
        numacopies.resize (numnodes);
        msra::numa::foreach_node_single_threaded ([&]()
        {
            size_t numanode = msra::numa::getcurrentnode();
            numacopies[numanode].resizeonce (n, m);
        });
    }
    // NUMA version only  --can this be abstracted better?
    matrix & operator[] (size_t numanodeid) { return numacopies[numanodeid]; }
#if 0
    // CUDA version only
    // note: needs more thinking to make it work with multiple devices
    void lazyinit_cuda()
    {
        if (!cudamatrix)
            cudamatrix.reset (msra::cuda::newmatrix());
    }
    void allocate_cuda (const matrixbase & us)  // allocate
    {
        lazyinit_cuda();
        cudamatrix->allocate (us.rows(), us.cols());
    }
    void assign_cuda (const matrixbase & other)
    {
        allocate_cuda (other);  // TODO: inefficient once we go multiple devices due to threading--make assign() allocate
        cudamatrix->assign (0, other.rows(), 0, other.cols(), &other(0,0), other.getcolstride());
    }
    void fetch_cuda (matrixbase & other) const  // call allocate_cuda() before CUDA op to allocate
    {
        assert (other.rows() == cudamatrix->rows() && other.cols() == cudamatrix->cols());
        cudamatrix->fetch (0, other.rows(), 0, other.cols(), &other(0,0), other.getcolstride());
    }
#endif
#if 0
    // There is no way in cublas to add a vector to all columns, so we need to do a matrix product... meh!
    // This function makes a row of ones.
    void makeones_cuda (const size_t m/*number of columns*/)
    {
        lazyinit_cuda();
        if (cudamatrix->rows() == m)
            return;
        cudamatrix->allocate (1, m);            // row vector
        const size_t colstride = 4;             // assume alignment to 4
        std::vector<float> ones (m * colstride, 1.0f);  // column vector
        cudamatrix->assign (0, 1, 0, m, &ones[0], colstride);
    }
#endif
#if 0
    operator msra::cuda::matrix & () const { return *cudamatrix.get(); }
    msra::cuda::matrix * operator-> () const { return cudamatrix.get(); }
#endif
};


// ---------------------------------------------------------------------------
// class rbmmodelmatrixbase
// ---------------------------------------------------------------------------

// matrix to hold RBM model parameters
//  - purpose: in CUDA mode, hold models in CUDA RAM except outside computation (that is, during model load/save)
//     - 'computing' state controls whether data currently lives in CPU or CUDA side
//     - provides underlying CUDA storage if in CUDA mode
//     - future: will handle multiple CUDA devices
//  - provides access to a controlled subset of matrix operations
//  - has all high-level operations needed for manipulating models
//     - CUDA mode: execution using CUDA copies wherever possible
//     - NUMA mode: optimized parallelized (multi-threaded) matrix product implemented in this class
template<class matrixbase> class rbmmodelmatrixbase : public acceleratedmatrixbase<msra::math::ssematrix<matrixbase>>
{
    typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;

    bool computing;                             // entercomputation() called <=> data lives in CUDA side?
    void checknotcomputing() const { if (computing) throw std::logic_error ("acceleratedmatrixbase: function called while in 'computing' state, forbidden"); }
    void checkcomputing() const { if (!computing) throw std::logic_error ("acceleratedmatrixbase: function called while not in 'computing' state, forbidden"); }
public:
    typedef cachedmatrixbase<matrixbase> cachedmatrix;  // TODO: move cachedmatrix inside here

    rbmmodelmatrixbase() : computing (false) {}
    rbmmodelmatrixbase (size_t rows, size_t cols) : computing (false) { resize (rows, cols); }
    ~rbmmodelmatrixbase() { if (computing) exitcomputation(); } // force GPU ops to complete before we free memory

    const matrix::vectorref<float> asvectorref()
    {
        checknotcomputing();
        return matrix::asvectorref();
    }

    // controlling 'computing' state
    void entercomputation()
    {
        checknotcomputing();
        // TODO: some badly encapsulated knowledge here
        if (cudamode)
        {
            if (rows() == 1 || cols() == 1)     // vectors--aha, must be the bias vectors, hence row-striped
                setcudastriping (stripedwrtrows);
            else                                // not a vector--aha, must be the weight matrix, hence col-striped
                setcudastriping (stripedwrtcols);
        }
        synctocuda (false);

        computing = true;           // matrix now owned by CUDA space; our CPU copy can only be used for reading dimensions etc.
    }
#ifdef  MULTICUDA
    void entercomputation (size_t deviceid)
    {
        checknotcomputing ();
        if (cudamode)
        {
            if (rows() == 1 || cols() == 1)     // vectors--aha, must be the bias vectors, hence row-striped
                setcudastriping (stripedwrtrows);
            else                                // not a vector--aha, must be the weight matrix, hence col-striped
                setcudastriping (stripedwrtcols);
        }
        cudadistributedmatrix::setDeviceId (deviceid);  // set deviceid here.
        synctocuda (false, deviceid);
        computing = true;
    }
    //hack for striped mode, distribute the model to multi devices
    void entercomputation (std::vector<size_t> &deviceids)
    {
        checknotcomputing ();
        if (cudamode)
        {
            if (rows() == 1 || cols() == 1)     // vectors--aha, must be the bias vectors, hence row-striped
                setcudastriping (stripedwrtrows);
            else                                // not a vector--aha, must be the weight matrix, hence col-striped
                setcudastriping (stripedwrtcols);
        }
        cudadistributedmatrix::setDeviceId (deviceids[0]);  // set the smallest deviceid.
        foreach_index (i, deviceids)  // copy to each device now, need to modify later.[v-xieche]
            synctocuda (false, deviceids[i]);
        computing = true;
    }
    void exitcomputation (size_t deviceid)
    {
        checkcomputing ();
        syncfromcuda (true, deviceid);
        computing = false;
    }
    // hack for striped mode in top layer
    void exitcomputation (std::vector<size_t> &deviceids)
    {
        checkcomputing ();
#ifdef STRIPEDTOPLAYER  // to make thing works first, set the cudastriping to be notstriped. [v-xieche]
        syncfromcuda (true, deviceids, notstriped);
#else
        syncfromcuda (true, deviceids);
#endif
        computing = false;
    }
#endif

    void exitcomputation()
    {
        checkcomputing();
        syncfromcuda (true);        // claim back matrix from CUDA space
        computing = false;
    }

    // outside computation
    void operator= (matrix && other) { checknotcomputing(); matrix::operator= (std::move (other)); }
    void operator= (const rbmmodelmatrixbase & other) { checknotcomputing(); matrix::operator= (other); }
    // operator= used in doublenodes() and model loading (all move semantics)
    // outside computation
    void resize (size_t n, size_t m) { checknotcomputing(); matrix::resize (n, m); }    // CPU-side (CUDA in entercomputation()); used by constructors only
    template<typename FILEHANDLETYPE>
    void read  (FILEHANDLETYPE f, const char * name, const std::string & begintag = std::string()) { checknotcomputing(); matrix::read (f, name, begintag); }
    template<typename FILEHANDLETYPE>
    void write (FILEHANDLETYPE f, const char * name) const { checknotcomputing(); matrix::write (f, name); }
    float &       operator() (size_t i, size_t j)       {  checknotcomputing(); return matrix::operator() (i, j); }  // doublenodes()
    const float & operator() (size_t i, size_t j) const {  checknotcomputing(); return matrix::operator() (i, j); }  // initrandom(), doublenodes();
    float &       operator[] (size_t i)       { checknotcomputing(); return matrix::operator[] (i); }  // doublenodes()
    const float & operator[] (size_t i) const { checknotcomputing(); return matrix::operator[] (i); }  // initrandom(), doublenodes();

    // special-purpose access, e.g. used to for model quantization
    const matrix & peek() const { checknotcomputing(); return (const matrix &) *this; }

    // swap
    void swap (rbmmodelmatrixbase & other)
    {
        checkcomputing(); other.checkcomputing();
        acceleratedmatrixbase::swap (other);
    }

    template<typename E>
    class datawithsize { E * p; size_t s; public: datawithsize (E * p, size_t s) : p(p), s(s) {} E * data() const { return p; } size_t size() const { return s; } };

    // MPI support
    void mpiallreducesum (const mpiaggregator & agg)
    {
        checkcomputing();
        if (cudamode)
            syncfromcuda (true);    // bring it into CPU space
        // BUGBUG: we need to pass data() since aggregate inside will checknotcomputing
        datawithsize<float> vec (&matrix::operator()(0,0), matrix::getcolstride() * matrix::cols());
        agg.aggregate (vec);      // MPI all-reduce
        if (cudamode)
            synctocuda (false);
    }

    // diagnostics: compute av over all elements squared
    float avsqr() const
    {
        float sqrsum = 0.0f;
        checkcomputing();
        if (!cudamode)
            //sqrsum += nrm2();
            fprintf (stderr, "avsqr: WARNING: not implemented, returning 0. Fix if you need it.");  // diagnostics only, so don't crash for now
        else
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                sqrsum += this->forcudadevice (deviceid).nrm2();
        return sqrsum / (cols() * rows());
    }

    // during computation
    // For these, we must be carefully choose what is where (CUDA or CPU memory). Should only be in one place at a time if at all possible (avoid cached copies).
    // For all public methods below, arguments passed in with a 'cachedmatrix' object are unique to this frame
    // and therefore must be moved to CUDA each time. The 'cachedmatrix' objects will hold pre-allocted memory for that.
    // All outputs are also frame-unique (unless same as input, and in CUDA memory).
    // Note that scaleandaddmatprod_numa, however, exists in two usage scenarios, one with A and one with C in CUDA memory.

    // this = this * thisscale + rowsum(othercols) * otherweight, 'othercols' is state memory i.e. temp per frame
    // The rowsum is computed into othertowsumtmp, which must have been allocated to correct size already.
    // If scale==0, we know to just assign.
    void scaleandaddallcols (const float thisscale, const rbmstatevectorsref & othercols, const float otherweight, msra::math::ssematrix<matrixbase> & otherrowsumtmp)  // used by updatedeltas()
    {
        // compute the row sum
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
#ifdef COMPACTTRAINER // not striped for compacttrainer. [v-xieche]
#ifdef MULTICUDA
            size_t deviceid = getDeviceId ();
#else
            size_t deviceid = 0; // need to modify later for multi-cuda
#endif		
            addrowsumincuda (deviceid, thisscale, othercols, otherweight);
#else
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, stripedwrtrows).addrowsum (thisscale, *othercols.stripeforcudadevice (deviceid, stripedwrtrows), otherweight);
#endif
        }
        else    // non-CUDA version
        {
            //if (otherweight != 1.0f) throw logic_error ("scaleandaddallcols: cannot yet scale the summand--implement this");   // TODO: implement this
            otherrowsumtmp.resizeonce (othercols.rows(), 1);
            othercols.fornuma().rowsum (otherrowsumtmp, otherweight);
            const matrixbase & other = otherrowsumtmp;  // the vector to add
            if (thisscale == 0.0f)  // this is an assignment (original content may be invalid, e.g. NaN)
                matrixbase::assign (other);
            else
                matrix::scaleandadd (thisscale, other);
        }
    }

    msra::math::ssematrix<matrixbase> &       fornuma()       { /*checkunlocked();*/ return *this; }
    const msra::math::ssematrix<matrixbase> & fornuma() const { /*checkunlocked();*/ return *this; }

    void computecolmean (rbmmodelmatrixbase & colmeanvector) const
    {
        checkcomputing();
        //if (cudamode)
        //{
        //    // TODO
        //}
        //else
        {
            if (cudamode)   // TODO: remove this part once CUDA code is there
                syncfromcuda (true);
            this->fornuma().colmean (colmeanvector);
            if (cudamode)
                colmeanvector.synctocuda (false);
        }
    }

    void computecolstddev(rbmmodelmatrixbase & colstddevvector, rbmmodelmatrixbase & colmeanvector) const
    {
        checkcomputing();
        //if (cudamode)
        //{
        //    // TODO
        //}
        //else
        {
            if (cudamode)   // TODO: remove this part once CUDA code is there
            {
                syncfromcuda (true);
                colmeanvector.syncfromcuda (true);
            }
            this->fornuma().colstddev (colstddevvector, colmeanvector);
            if (cudamode)
                colstddevvector.synctocuda (false);
        }
    }

    void scale(float factor)
    {
        checkcomputing();
        if (cudamode)
        {
            // determine striping mode based on matrix dimensions
            // TODO: can we not just *not* pass the 'stripingtype'?
            cudastriping_t stripingtype = cols() == 1 ? stripedwrtrows : stripedwrtcols;
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                msra::cuda::matrix &cudamatrix = forcudadevice(deviceid, stripingtype);
                cudamatrix.scale(factor);
            }
        }
        else{
            matrix::scale(factor);
        }
    }

    void scaleandaddallcolspool (const float thisscale, const rbmstatevectorsref & othercols, const float otherweight, msra::math::ssematrix<matrixbase> & otherrowsumtmp, size_t poolSize, size_t bands, size_t kernels)  // used by updatedeltas()
    {
        // compute the row sum
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
#ifdef COMPACTTRAINER // not striped for compacttrainer. [v-xieche]
#ifdef MULTICUDA
            size_t deviceid = getDeviceId ();
#else
            size_t deviceid = 0; // need to modify later for multi-cuda
#endif		
            addrowsumpoolincuda (deviceid, thisscale, othercols, otherweight, poolSize, bands, kernels);
#else
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, stripedwrtrows).addrowsum (thisscale, *othercols.stripeforcudadevice (deviceid, stripedwrtrows), otherweight);
#endif
        }
        else    // non-CUDA version
        {
            //if (otherweight != 1.0f) throw logic_error ("scaleandaddallcols: cannot yet scale the summand--implement this");   // TODO: implement this
            otherrowsumtmp.resizeonce (othercols.rows(), 1);
            othercols.fornuma().rowsum (otherrowsumtmp, otherweight);
            const matrixbase & other = otherrowsumtmp;  // the vector to add
            if (thisscale == 0.0f)  // this is an assignment (original content may be invalid, e.g. NaN)
                matrixbase::assign (other);
            else
                matrix::scaleandadd (thisscale, other);
        }
    }

    // accumulator += this, where other lives in CPU space
    // Actually it seems this makes no measurable runtime difference at all.
    template<class VECTOR> void accumulate (VECTOR & accumulator) // const
    {
        checkcomputing();
#ifndef MULTICUDA
        syncfromcuda (true);     // bring it into CPU space
#else
        syncfromcuda (true, getDeviceId());
#endif
        assert (accumulator.size() == rows() && cols() == 1);
        foreach_index (i, accumulator)
            accumulator[i] += matrix::operator() (i,0);
    }
#ifdef STRIPEDTOPLAYER  // implement the accumualte prior function.[v-xieche]
    template<class VECTOR> void accumulate_multicuda (VECTOR & accumulator, std::vector<size_t> &devids) 
    {
        checkcomputing ();
        syncfromcuda (true, devids, notstriped); 
        foreach_index (i, accumulator)
        {
            accumulator[i] += matrix::operator() (i,0);
        }
    }
#endif

    // this = this * thisscale + other * otherweight, both in accelerated memory
    // This is used for model update.
    void addweighted (float thisscale, const rbmmodelmatrixbase & other, float otherweight)         // adddeltas()
    {
        checkcomputing();
        if (cudamode)
        {
//syncfromcuda (true);
//other.syncfromcuda (true);
            checkmatchingdisjunctcudastriping (other);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).gems (thisscale, other.forcudadevice (deviceid), otherweight);
//syncfromcuda (true);

//syncfromcuda(true);
//other.syncfromcuda(true);
        }
        else
        {
            matrix::addweighted (thisscale, other, otherweight);
        }
    }

    // sets matrix to diagonal preconditioner derived from gradientsquared
    // this = (gradientsquared / nobservations + lambda)^alpha (elementwise)
    // TODO don't use special purpose method for this
    virtual void setdiagonalpreconditioner(const rbmmodelmatrixbase & gradientsquared, float nobservations, float lambda, float alpha)
    {
        assert(gradientsquared.rows() == rows());
        assert(gradientsquared.cols() == cols());
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setdiagonalpreconditioner(gradientsquared.forcudadevice (deviceid), nobservations, lambda, alpha);
        }
        else
            matrix::setdiagonalpreconditioner(gradientsquared, nobservations, lambda, alpha);
    }

    // elementwise division of a by b
    // this = a / b (elementwise)
    virtual void elementwisedivision(const rbmmodelmatrixbase &a, const rbmmodelmatrixbase &b)
    {
        assert(a.rows() == rows());
        assert(a.cols() == cols());
        assert(b.rows() == rows());
        assert(b.cols() == cols());
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).elementwisedivision(a.forcudadevice (deviceid), b.forcudadevice (deviceid));
        }
        else
            matrix::elementwisedivision(a,b);
    }

    // clear a matrix
    // in NUMA mode, setzero could be implemented by using memset
    // therefore it makes sense to have a different implementation for setzero and setvalue
    void setzero()
    {
        if (empty())    // sometimes called on empty matrices that have not entered computation since they were empty
            return;
        checkcomputing();
        if (cudamode)
        {
#if 0
            addweighted (0.0f, *this, 0.0f);
#else
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setvalue (0.0f);
#endif
        }
        else
            matrix::setzero();
    }

#if 0
    void initialize()  // Jian added for svd decomposition
    {
        matrix::setzero();
    }
#endif

    // -----------------------------------------------------------------------
    // AdaGrad
    // -----------------------------------------------------------------------

    // this = keepweight * this + (1-keepweight) * other .^ 2
    // This is for use in AdaGrad. We compute square of sum (with low-path filter) to approximate sum of square, but need to compensate before using it.

    void accumulatesqr (const rbmmodelmatrixbase & other, float keepweight)
    {
        checkcomputing();

        if (cudamode)
        {
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).accumulatesqr (other.forcudadevice (deviceid), keepweight);
        }
        else
        {
            matrix & us = *this;
            assert (us.cols() == other.cols() && us.rows() == other.rows());
            if (cudamode)   // TODO: remove this part once CUDA code is there
            {
                syncfromcuda (true);
                other.syncfromcuda (true);
            }
            const matrix & othermatrix = other;
            foreach_coord (i, j, us)
                us(i,j) = keepweight * us(i,j) + (1.0f - keepweight) * othermatrix(i,j) * othermatrix(i,j);      // low-pass filter of the sqr
            if (cudamode)
                synctocuda (false);
        }
    }

    // compute asum/#elem
    float absaverage()
    {
        checkcomputing();
        if (cudamode)
        {
            float sum = 0.0f;
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                sum += forcudadevice (deviceid).asum();
            return empty() ? 0.0f : sum / (cols() * rows());
        }
        else
            throw std::logic_error ("absaverage: not implemented");
    }

    // compute avdenom = av over all adagraddenom(i,j)    --to stay in the same range as before
    float adagradientavdenom (const rbmmodelmatrixbase & sqracc, rbmmodelmatrixbase & denombuf, float numframes/*summed frames in 'gradient'*/,
                              const rbmmodelmatrixbase & meanacc, float meannumframes,
                              size_t mbframes)
    {
        checkcomputing();
        if (cudamode)
        {
            // We temporarily save denom(,) in denombuf(,) which may be aliased to 'this'. It will later be overwritten by the adapted gradient.
            // TODO: clean up the meanacc business; either do it nice or delete it
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                denombuf.forcudadevice (deviceid).adadenom (sqracc.forcudadevice (deviceid), numframes, meanacc.forcudadevice (deviceid), meannumframes, mbframes);
            // BUGBUG?: is asum() correct for stripes?
            float sumdenom = 0.0;
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                sumdenom += denombuf.forcudadevice (deviceid).asum();
            float actualavdenom = sumdenom / (sqracc.rows() * sqracc.cols());         // average denom
            fprintf (stderr, "adagradient: actualavdenom updated to %.2f x 1e-6\n", actualavdenom * 1e6);
            return actualavdenom;
        }
        else
        {
            // TODO: this is nasty code dup with adagradient(); need to do some factoring, like on the CUDA side of things
            matrix & us = *this;
            matrix & denom = *this; // (note: may alias to 'us')

            assert (us.cols() == denom.cols()    && us.rows() == denom.rows());
            assert (us.cols() == sqracc.cols()   && us.rows() == sqracc.rows());

            // get data back from CUDA
            if (cudamode)
                sqracc.syncfromcuda (true);
            const matrix & sqraccmatrix = sqracc;

            // compute denom(,) and avdenom
            // We temporarily save denom(,) in denom(,), which may alias to 'us'. It will later be overwritten by the adapted gradient.
            // If too few summands then we just use 1 instead.
            double sumdenom = 0.0;
            foreach_coord (i, j, sqraccmatrix)
            {
                // AdaGrad works by normalizing each gradient component g by its sqrt (sum g^2). Specifically,
                //   g_AdaGrad(i,j) = g(i,j) * targetav / clip (sqrt (av2(i,j)))
                // where
                //   av2(i,j) = sum_n sum_t g_nt(i,j)^2 / NT
                // (Dirty secret: both g_nt and target have been multiplied with the learning rate; it thus cancels out. Ugh.)
                // Here, we express the frame index as _nt where n = minibatch index and t = frame index within the minibatch;
                // and N = number of minibatches and T = number frames per minibatch (total of N * T frames).
                // Beyond this point, everything shall be read as per-component, so we drop (i,j) for better readability.
                // Since we do not have access to individual gradient frames (they are never computed), what we actually accumulate is:
                //   acc0 = N * T
                //   acc1 = sum_n sum_t g_nt            // not yet actually computed, assuming 0
                //   acc2 = sum_n (sum_t g_nt)^2        // note: matrix is called sqraccmatrix
                const float acc0 = numframes;
                const float acc1 = 0.0f;                // (TODO: actually compute  this)
                const float acc2 = sqraccmatrix(i,j);
                const float T = (float) mbframes;
                // (In reality we compute a weighted sum (that momentum-like formula amounts to exactly that), but that will not change anything.)
                // We assume that the g_nt are Gaussian and ergodic.
                // Hence, acc0..2 represent the statistics of the sum of T independent processes. Let's call it G_n.
                //   G_n = sum_t g_nt
                //   meanG = T * mean
                //   varG  = T * var
                // Compute mean and var of G:
                //   meanG = acc1 / N
                //   varG = acc2 / N - meanG ^ 2 = acc2 / N - (acc1 / N) ^ 2
                // Now we can convert that into mean and var of g:
                //   mean = meanG / T = acc1 / acc0
                //   var  = varG / T = acc2 / acc0 - mean * mean * T 
                const float mean = acc1 / acc0;
                const float var = acc2 / acc0 - mean * mean * T;
                // Now we express av2 by the mean and var:
                //   av2 = sum_n sum_t g_nt^2 / NT = mean ^ 2 + var
                const float av2 = mean * mean + var;
                float denomij = sqrt (av2);         // clip happens later
                denom(i,j) = denomij;               // denominator value before clipping --note: 'denom' may alias to 'this'
                sumdenom += denomij;                // sum all the para in W matrixW
            }
            return (float) (sumdenom / (sqraccmatrix.rows() * sqraccmatrix.cols()));       // overwrite average denom
        }
    }

    // this = gradient / (adagraddenom / avdenom)
    // where adagraddenom(i,j) = sqrt (sqracc(i,j) / numframes)
    // and avdenom = av over all adagraddenom(i,j)    --to stay in the same range as before
    // where numframes is the (fractional) number of frames that has been accumulated in sqracc
    // Returns avdenom for diagnostics.
    // If too few summands then do not apply this; just return the unmodified gradient.
    // 'targetavdenom' is the av denom we shall assume (consider it the overall average) [pre-multipled by learningrateperframe to be consistent with dW/da].
    // 'actualavdenom' is the actual one for this matrix, and because it's expensive, not always updated.
    // It is only used for controlling the 10 x adjustment range, and for tracking.
    // TODO: outdated comment: Note that it is OK if 'this' aliases either 'denombuf' (used for smoothed-gradient AG) or 'gradient' (raw-gradient AG; in-place update raw gradient before quantization--not used/not working).
    void adagradient (const rbmmodelmatrixbase & gradient, const rbmmodelmatrixbase & sqracc, rbmmodelmatrixbase & denombuf_unused, float numframes/*summed frames in 'gradient'*/,
                      size_t mbframes, const float actualavdenom, const float targetavdenom/*manually chosen target*/)
    {
        if (actualavdenom == 0.0f)              // special guard against a perfect-0 gradient--we get that in double-buffered training  --TODO: guard against it outside!
            return;
        checkcomputing();
        if (cudamode)
        {
            // apply adagrad
#if 1       // new code; note: does not support targetavdenom, but it would be trivial to put it back in
            const float lrfudgefactor = -targetavdenom; // currently communicated like this for compat reasons
            if (lrfudgefactor <= 0.0f)
                throw std::runtime_error ("adagradient: this version cannot handle target values");
            // this = gradient ./ (denom / avdenom)
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).adagradientfromsqracc (gradient.forcudadevice (deviceid), sqracc.forcudadevice (deviceid), numframes, actualavdenom, lrfudgefactor);
#else       // old code which had more memory accesses
            // Note that we duplicate some computation with the above; but since we don't do the above too often, it's OK
            // TODO: merge these into a single function
            if (!updateactualavdenom)
                for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                    denombuf.forcudadevice (deviceid).adadenom (sqracc.forcudadevice (deviceid), numframes, mbframes);

            // this = gradient ./ (denom / avdenom)  --note: denom stored in 'this' for the moment
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).adagradientfromdenom (denombuf.forcudadevice (deviceid), gradient.forcudadevice (deviceid), actualavdenom, targetavdenom, mbframes);
#endif
        }
        else
        {
            matrix & us = *this;
            matrix & denom = *this; // (note: may alias to 'us')

            assert (us.cols() == gradient.cols() && us.rows() == gradient.rows());
            assert (us.cols() == denom.cols()    && us.rows() == denom.rows());
            assert (us.cols() == sqracc.cols()   && us.rows() == sqracc.rows());

            // get data back from CUDA
            if (cudamode)
                sqracc.syncfromcuda (true);
            const matrix & sqraccmatrix = sqracc;

            // compute denom(,) and avdenom
            // We temporarily save denom(,) in denom(,), which may alias to 'us'. It will later be overwritten by the adapted gradient.
            // If too few summands then we just use 1 instead.
            double sumdenom = 0.0;
            foreach_coord (i, j, sqraccmatrix)
            {
                // AdaGrad works by normalizing each gradient component g by its sqrt (sum g^2). Specifically,
                //   g_AdaGrad(i,j) = g(i,j) * targetav / clip (sqrt (av2(i,j)))
                // where
                //   av2(i,j) = sum_n sum_t g_nt(i,j)^2 / NT
                // (Dirty secret: both g_nt and target have been multiplied with the learning rate; it thus cancels out. Ugh.)
                // Here, we express the frame index as _nt where n = minibatch index and t = frame index within the minibatch;
                // and N = number of minibatches and T = number frames per minibatch (total of N * T frames).
                // Beyond this point, everything shall be read as per-component, so we drop (i,j) for better readability.
                // Since we do not have access to individual gradient frames (they are never computed), what we actually accumulate is:
                //   acc0 = N * T
                //   acc1 = sum_n sum_t g_nt            // not yet actually computed, assuming 0
                //   acc2 = sum_n (sum_t g_nt)^2        // note: matrix is called sqraccmatrix
                const float acc0 = numframes;
                const float acc1 = 0.0f;                // (TODO: actually compute  this)
                const float acc2 = sqraccmatrix(i,j);
                const float T = (float) mbframes;
                // (In reality we compute a weighted sum (that momentum-like formula amounts to exactly that), but that will not change anything.)
                // We assume that the g_nt are Gaussian and ergodic.
                // Hence, acc0..2 represent the statistics of the sum of T independent processes. Let's call it G_n.
                //   G_n = sum_t g_nt
                //   meanG = T * mean
                //   varG  = T * var
                // Compute mean and var of G:
                //   meanG = acc1 / N
                //   varG = acc2 / N - meanG ^ 2 = acc2 / N - (acc1 / N) ^ 2
                // Now we can convert that into mean and var of g:
                //   mean = meanG / T = acc1 / acc0
                //   var  = varG / T = acc2 / acc0 - mean * mean * T 
                const float mean = acc1 / acc0;
                const float var = acc2 / acc0 - mean * mean * T;
                // Now we express av2 by the mean and var:
                //   av2 = sum_n sum_t g_nt^2 / NT = mean ^ 2 + var
                const float av2 = mean * mean + var;
                float denomij = sqrt (av2);         // clip happens later
                denom(i,j) = denomij;               // denominator value before clipping --note: 'denom' may alias to 'this'
                sumdenom += denomij;                // sum all the para in W matrixW
            }

            // this = gradient ./ (denom / avdenom)  --note: denom stored in 'this' for the moment
            if (cudamode)
                gradient.syncfromcuda (true);
            const matrix & gradientmatrix = gradient;
            foreach_coord (i, j, gradientmatrix)
            {
                const float denomij = denom(i,j);                 // (denominator value before clipping; 'denom' may alias to 'this')
                // scale the gradient
                // We limit the scaling to 10 x the original gradient against the average of this parameter matrix.
                // After limiting, we then scale further to reach the desired factor (targetavdenom).
                // TODO: Thus, targetavdenom scaling is identical to scaling the learning rate. We could remove one.
                float weight;
                if (denomij == 0.0f)                            // special case: denom = 0 (zero denominator) -> use average
                    weight = 1.0f;
                else
                {
                    weight = actualavdenom / denomij;           // we weight the gradient with avdenom / denomij
                    if (weight > 10.0f)                         // clip the weight somewhat  --I saw outliers up to 100k+
                        weight = 10.0f;
                    else if (weight < 0.01f)                    // TODO: is this lower clipping bounds of any value?
                        weight = 0.01f;
                }
                if (targetavdenom > 0.0f)       // if requested then scale to a hand-tuned factor instead
                    weight *= targetavdenom / actualavdenom;
                else                            // if negative then we can pass a scaling factor
                    weight *= (-targetavdenom);
                // Note: targetavdenom has been pre-multiplied with learningrateperframe, just as dW and da.
                // compatibilty with old code:
#if 1           // backwards compatibility with an old bug
                if (targetavdenom > 0.0f && mbframes <= 256)
                    weight *= 2;            // led to different interpretation of parameter for mb size 256 vs. 1024
#endif
                us(i,j) = gradientmatrix(i,j) * weight; // (note: possible in-place update; us(,) may alias denom(,) or gradientmatrix(,))
            }

            // move resulting gradients to CUDA side
            if (cudamode)
                synctocuda (false);
        }
    }

    // -----------------------------------------------------------------------
    // MPI data parallelism support (to be called through callbacks from mpiaggregator)
    // -----------------------------------------------------------------------

    // ISSUE TRACKER for 1-bit SGD    --TODO: check that these are still meaningful; delete the old ones
    //  - current plan:
    //      - MPI thread: enforce using the same thread, to avoid CUDA's overhead for attaching to a new thread
    //  - 1-bit quantization:
    //     - zero mean works better --> tie both sides to guarantee symmetry (threshold is off otherwise) --> works better for later iterations
    //     - quantization tying makes a difference (early: bad; late: good) --> store levels explicitly
    //  - small unexplained non-determinism in frame acc--wtf? GPU-related? happens in MB la tuning, starting with 2048
    //  - reenable 'recomputerange' trick and optimize
    //  - check 'accuracy' factor for smaller #bits (so far only tested for 16 bits)
    //  - test of 0 frames in a sub-batch --don't skip the batch; rather make code resilient to 0 frames
    //  - somehow exchange actual mbsize, for correct momentum computation
    //  - support multiple GPUs (model parallelism)
    // old resolved issues:
    //  - GPU buffers must be page-locked  --FIXED
    //  - BUG: 'altlayout' not handling missing bits (underfull 32-bit words) correctly  --FIXED
    //  - actually run with MPI; get a test environment  --DONE, works
    //  - parallel thread for MPI interaction  --DONE, works
    //  - quantization even to 16 bit, 1 node, no db, causes 4 points loss! goes away with accuracy=5, leaving it at that
    //  - 2 nodes does not work at all except for quant=32; caused by recomputerange trick which forgot to scale with K
    //  - crash at end of epoch inside bg thread, seems it did not wait for it to terminate  --FIXED (accidentally removed the termination condition)
    //  - we stupidly compute full fw/bw on entire batch  --FIXED
    //  - memory consumption very high--a bug somewhere?  --it's the 24k frame buffer --FIXED (reduced to 12)
    //  - bias-vector split--vert vs. horiz--seems to make a small difference :(  --OK: because of different quant ranges
    //  - QUANTNOCUDA does not work  --DONE, actually does work
    //  - numframes is not correct, also leads to bad upates in deferred mode  --FIXED (deferred mode)
    //  - fix that time-remaining estimate  --DONE
    //  - deferred-update broken; needs to be done on raw gradient!! and no exchange before final  --FIXED
    //  - if we disable aggregate() because of K=1, we should also not divide the mbsize by 2  --DONE
    //     - also move /= 2 into processminibatches()  --DONE
    //     - also undo that factor in update3() for momentum  --DONE
    //  - do AdaGrad before quantization on raw gradient? maybe helps with quantization  --does not work
    //  - disable logpp exchange across nodes, or at least do it very rarely or maybe only at the end  --DONE
    //  - IMPORTANT: make 1-bit quantization symmetric  --DONE
    //  - larger MB size seems to not converge well with data parallelism--quantization? --> try AdaGrad pre quantization?  --FIXED
    //  - deferred update may or may not work --> leave disabled for now  --FIXED (resetmomentum bug)
    //  - CRITICAL: deferred update seems not working (crash for 64k/2, while 24k/2 was OK)  --DONE (resetmomentum bug)
    //  - AdaGrad goes bad for double-buffering --> disable for now  -DONE, but don't remember how (some init issue?)
    //  - CRITICAL: sub-set of nodes broken (leads to weird frame accuracies on last node; new? node)  --DONE (forgot to redist model)
    //  - check 1-bit quant  --running and seemingly working, not sure how well  --DONE (but large MBs seem to screw up)
    //  - optimize for 4k on non-network setup, 8 nodes (msrascr010)  --DONE; cuda event sync bug found, now as expected
    //  - AdaGrad: use average over all layers, maybe getting closer to targetavdenom, and definitely more correct  --DONE but targetavdenom works best
    //  - model parallelism; should help to cut fixed cost  --yay!  --DONE, and indeed it helps!

    // simplistic MPI_Allreduce() for matrices (used in unquantized case, e.g. model averaging)
    void allreduce (const mpiaggregator & mpiaggregator)
    {
        if (cudamode)
            syncfromcuda (true);
        mpiaggregator.allreduce (matrix::asvectorref());
        if (cudamode)
            synctocuda (false);
    }

private:

    // we need a bunch of helper objects; these are only instantiated if this matrix participates in MPI data parallelism

    // a qstripe in CPU RAM (there is also one in the GPU)
    // This stores:
    //  - the patch dimensions
    //  - the device
    //  - a byte range into a buffer to store the quantized data in (the actual data pointer is not kept here)
    struct mpistripebufferref
    {
        void operator= (const mpistripebufferref &);
        mpistripebufferref (const mpistripebufferref &);

        size_t begin, end;                          // this stripe's quantized package occupies this byte range in 'cpubuffer'

        size_t bits;
        size_t deviceid;                            // GPU device on which this lives
        size_t i0, i1, j0, j1;                      // stripe patch dimensions --for multi-GPUs, these are *relative* to the GPU-specific patch
                                                    // ^^ BUGBUG: This may preclude CPU-side emulations while running with GPU; better make them absolute, and adjust later
        matrixbase       patch (matrix & m) const       { if (deviceid != 0) throw std::logic_error ("patch: incompatible with multi-GPU"); return m.patch (i0, i1, j0, j1); }   // get our patch from a matrix
        const matrixbase patch (const matrix & m) const { if (deviceid != 0) throw std::logic_error ("patch: incompatible with multi-GPU"); return m.patch (i0, i1, j0, j1); }

        size_t buffersize() const { return msra::math::matrixquantizer::buffersize (bits, i1-i0, j1-j0); }  // TODO: needed?

        // mpistripebuffersize keeps track of the size of a given stripe
        mpistripebufferref (size_t bits, size_t i0, size_t i1, size_t j0, size_t j1, size_t deviceid, size_t & mpistripebuffersize) :
        bits (bits), i0 (i0), i1 (i1), j0 (j0), j1 (j1), deviceid (deviceid)
        {
            // allocate byte range in CPU-side buffer
            begin = mpistripebuffersize;
            mpistripebuffersize += msra::math::matrixquantizer::buffersize (bits, i1-i0, j1-j0);    // TODO: use buffersize() (local)?
            end = mpistripebuffersize;

            // size the CPU buffer accordingly (called should start with an empty buffer)
            if (end - begin != buffersize())    // TODO: this test is kind of redundant now...
                throw std::logic_error ("mpistripebufferref: incorrect allocation size");
        }
    };

    // sub-batch stripes to hold quantized stripes on both CPU and GPU --these only contain offsets, actual buffer must be supplied in each call to allow for double-buffering
    // The GPU buffer (qstripe) does hold a GPU-side buffer for quantization and then shipping it to the CPU (the CPU-side buffer is passed in to allow for double-buffering).
    // ... TODO: This is a little risky; better flip the GPU buffer as well. Works because we use two different buffers, one for quant and one for unquant; they never overlap. Brittle.
    std::vector<std::shared_ptr<mpistripebufferref>> mpistriperefs; // [stripe] CPU-side sub-batch stripes (does not contain actual buffer *, only offsets, for double-buffering)
    std::vector<std::shared_ptr<msra::cuda::qstripe>> qsubstripes;  // [stripe] GPU-side sub-batch stripes (cudamode only)

    // aggregation
    // This stripe is only modified inside the aggregation phase.
    // If we use a background thread (for double buffering) then these vars are accessed only in that thread function. Only one such function can exist at any time.
    std::shared_ptr<rbmmodelmatrixbase<matrixbase>> aggaccstripe;       // stripe-aggregation accumulator for our stripe (matrix with (0,0) at top left corner of the stripe)
    std::shared_ptr<rbmmodelmatrixbase<matrixbase>> aggresstripe;       // residuals for aggregated stripe (same layout as aggacctripe)
    std::vector<std::shared_ptr<msra::cuda::qstripe>> qaccstripes;      // [kfrom] GPU-side stripe for performing aggregation of our owned stripe (cudamode only)
    std::shared_ptr<msra::cuda::qstripe> qresstripe;                    // GPU-side stripe for requantization and back-transfer of our owned stripe (cudamode only)

    // accumulators for fixed-cost operations done in a distributed fasgion during aggregation
    std::shared_ptr<rbmmodelmatrixbase<matrixbase>> aggadagradsqrstripe;// accumulator for local per-stripe AdaGrad ('distributefixedcost' mode)
    double aggadagradsqrframes;                                         // #frames accumulated
    std::shared_ptr<rbmmodelmatrixbase<matrixbase>> aggsmoothedstripe;  // accumulator for local per-stripe momentum ('distributefixedcost' mode)

    // sub-batch stripe for sending the aggregate quantized blob to the GPU
    // This holds one GPU-side buffer; for the CPU-side, it only holds buffer offsets as to allow for passing in double buffers.
    std::vector<std::shared_ptr<msra::cuda::qstripe>> qaggstripes;  // [stripe] GPU-side aggregated (accumulated) stripes (cudamode only)

    // This is used only on main thread to carry over the residual of the first quantization.
    shared_ptr<rbmmodelmatrixbase<matrixbase>> psubresidual;        // residual on sub-batch (must be a ptr since this is the same type as 'this')

public:

    // determine the stripe dimensions (patch size, quantized buffer size) of this matrix and their place in the cross-model stripe buffer
    // The buffer size is allocated by increasing the respective mpistripebuffersizes[node] value. The mpiaggregator will later use that as the shared buffer size for each node's stripe.
    // The number of stripes may differ across epochs, so this function must allow for changing the allocation.
    void entermpiaggregation (std::vector<size_t> & mpistripebuffersizes, size_t bits)
    {
        checkcomputing();       // must be called inside enter/exitcomputation()
#ifdef ZERO_THRESHOLD_FOR_1BIT     // force 1-bit quant to threshold against 0 rather than the midpoint between lower and upper
         fprintf (stderr, "entermpiaggregation: WARNING: quantizer is compiled to assume zero threshold for 1-bit quantization\n"), fflush (stderr);
#elif defined (ASSUME_ZERO_MEAN_1BIT)
         fprintf (stderr, "entermpiaggregation: WARNING: quantizer is compiled to assume zero mean for quantization range\n"), fflush (stderr);
#else
         fprintf (stderr, "entermpiaggregation: note: quantizer compiled to use actual column mean\n"), fflush (stderr);
#endif

        // This is called for each layer after entercomputation() but before doing anything
        // Our job is to allocate the qsubstripes and connect them to the buffers in the localbuffers.

        // data layout:
        //  - dimensions:
        //     - MPI compute nodes (data parallelism)
        //     - layers
        //     - multiple GPU devices (model parallelism)
        //  - there is one buffer for each compute node, which contains
        //    for each layer:
        //     - dW and da as qsubstripes (one qstripe per column)
        //  - in case of multiple GPUs (model parallelism), qsubstripes are aligned with a single GPU,
        //    i.e. the MPI node id (=MPI stripe id) implies one GPU

        // create holders for stripes (the actual buffers are not known at this point since we are still collecting their size)
        const size_t numnodes = mpistripebuffersizes.size();
        mpistriperefs.resize (numnodes);            // stripe refs for this matrix: dimensions, byte range in cross-model stripe
        if (cudamode)                               // initialize cudamode-only objects
        {
            qsubstripes.resize (numnodes);          // pointers will be null-initialized now and populated later upon first use
            qaggstripes.resize (numnodes);          // dito.
            qaccstripes.resize (numnodes);
        }

        // allocate the residuals and entercomputation() for the first one (which lives on the GPU)
        psubresidual = make_shared<rbmmodelmatrixbase<matrixbase>> (rows(), cols());
        psubresidual->entercomputation();
        psubresidual->setzero();

        // determine each stripe's patch region and required storage size for this weight matrix
        // mpistripebuffersizes[mpinode] is an in/out that gets aggregated across layers (starting with 0)
        const size_t numdevices = cudamode ? numcudadevices() : 1;      // model parallelism within a data-parallel sub-batch
        for (size_t deviceid = 0; deviceid < numdevices; deviceid++)    // enumerate GPUs, then assign sub-range of stripes to each
        {
            // sub-range of stripes on this GPU
            // For example 5 nodes with 2 devices -> qstripes (0..1) are on GPU #0, and (2..4) on GPU #1.
            // The sub-ranges are identical for all matrices, such that each stripe is uniquely assigned to only one GPU.
            // There must be at least as many nodes as devices.
            if (numnodes < numdevices && numnodes != 1/*this is a bit shaky--we set up garbage knowing we won't use it...*/)
                throw std::logic_error ("entermpiaggregation: less nodes than GPU devices not supported");
            const size_t nodesbegin = numnodes * deviceid / numdevices;
            const size_t nodesend = numnodes * (deviceid+1) / numdevices;
            // get a CUDA patch
            size_t r = cudamode ? forcudadevice (deviceid).rows() : rows();
            size_t c = cudamode ? forcudadevice (deviceid).cols() : cols();
            // we now evenly distribute those node stripes over this device
            for (size_t mpinode = nodesbegin; mpinode < nodesend; mpinode++)
            {
                // get our patch dimension (local coordinates w.r.t. the multi-GPU stripe)
                size_t i0, i1, j0, j1;
                i0 = 0;
                i1 = r;
                j0 = c * (mpinode - nodesbegin)    / (nodesend - nodesbegin);   // equally distribute columns over the node range for this device
                j1 = c * (mpinode - nodesbegin +1) / (nodesend - nodesbegin);
#if 0           // This conflicts with model parallelism.
                // make it SSE-conformant
                i0 = alignrowpatchindex (i0);
                i1 = alignrowpatchindex (i1);
                // Note: bias stripe is row-striped and may thus not be SSE-aligned for multi-GPU  --TODO: really? How can it operate then?
#endif

                // remember it; this also determines the required buffer size and 'allocates' it by bumping up 'mpistripebuffersizes[node]'
                size_t & mpistripebuffersize = mpistripebuffersizes[mpinode];   // required cross-model stripe buffer size is recored here
                size_t begin = mpistripebuffersize;
                mpistriperefs[mpinode] = make_shared<mpistripebufferref> (bits, i0, i1, j0, j1, deviceid, mpistripebuffersize/*in/out*/);
                fprintf (stderr, "entermpiaggregation: configuring for %d-bits quantization, %d bytes buffer at offset %d, (%d,%d)..(%d,%d) on dev %d\n",
                         (int) bits, (int) (mpistripebuffersize - begin), (int) begin, i0, j0, i1, j1, deviceid);
            }
            fflush (stderr);
        }
    }

    // this tears down all MPI-related objects related to this weight matrix
    void exitmpiaggregation()
    {
        fprintf (stderr, "exitmpiaggregation: entering\n"); fflush (stderr);
        checkcomputing();       // must be called inside enter/exitcomputation()

        // clear the stripe refs
        // Note: destruction of GPU-side stripes will wait for GPU ops to complete.
        aggaccstripe.reset();           // lazily allocated in unquantizeandaggregatestripe()
        aggadagradsqrstripe.reset();    // lazily allocated in unquantizeandaggregatestripe()
        aggsmoothedstripe.reset();      // lazily allocated in unquantizeandaggregatestripe()
        qaccstripes.clear();            // lazily allocated in unquantizeandaggregatestripe()
        aggresstripe.reset();           // lazily allocated in quantizeandassignaggregatedstripe()
        qresstripe.reset();             // lazily allocated in quantizeandassignaggregatedstripe()
        qaggstripes.clear();            // lazily allocated in assignaggregatedstripe()
        qsubstripes.clear();            // lazily allocated in quantizeandfetchsubbatchstripe()
        mpistriperefs.clear();          // from entermpiaggregation()

        // take down our GPU-side helper matrices
        psubresidual.reset();
        fprintf (stderr, "exitmpiaggregation: done\n"); fflush (stderr);
    }

    // helper to allocate CUDA-suitable memory (for use in distributed data parallelism)
    // This buffer is shared across many layers and many weight matrices. Only one of them (arbitrary choice) will be asked to allocate it.
    // Note that this returns a shared_ptr with custom deleter, so it will do the right thing for both CUDA and non-CUDA.
    shared_ptr<char> allocatetransferbuffer (size_t stripe, size_t size)
    {
        if (cudamode)
            return forcudadevice(mpistriperefs[stripe]->deviceid).newsharedtransferbuffer (size); // this is going to be page-locked memory, which gives significantly more efficient transfer speeds
        else
            return shared_ptr<char> (new char[size], [](char*p){delete[]p;});
    }

    // (un-)scale this matrix in-place for quantization
    // 'unscale' = true means apply the inverse
    // We use the AdaGrad accumulator for scaling.
    // This does not work and is not used. It can probably be deleted.
    void prescaleforquantization (const rbmmodelmatrixbase<matrixbase> & adagraddsqrsum, const double adagradframes, size_t i0, size_t i1, size_t j0, size_t j1, bool unscale)
    {
        if (adagradframes == 0.0)   // AdaGrad not enabled or nothing accumulated yet
            return;

        // BUGBUG: use the proper device (this is a prototype only)
        if (cudamode)
        {
            size_t deviceid = 0;
            unique_ptr<msra::cuda::matrix> thispatch (forcudadevice(deviceid).patch (i0, i1, j0, j1));
            unique_ptr<msra::cuda::matrix const> sqrsumpath (adagraddsqrsum.forcudadevice(deviceid).patch (i0, i1, j0, j1));
            thispatch->prescaleforquantization (*sqrsumpath, adagradframes, unscale);
        }
        else
            throw std::logic_error ("prescaleforquantization: not implemented for CPU yet");
    }

#undef JUSTDOUBLEBUFFER    // test mode for testing double-buffering without any quantization
#ifdef JUSTDOUBLEBUFFER
    shared_ptr<matrix> plast;
#endif
    // step 1: quantize a stripe into CPU-side buffer sub-range (full-size buffer = bufferbegin/end)
    // This is called in the sequence that the MPI aggregator wants it to be, once per stripe.
    // To support GPU processing, this is an async function, call syncfetchsubbatchstripe() to wait for completion.
    // The subresidual carries over the quantization error across calls (its current value gets added to matrix to quantize, and it gets assigned the new error).
    // This function is asynchronous (non-blocking) in CUDA mode.
    void quantizeandfetchsubbatchstripe (size_t stripe, const rbmmodelmatrixbase<matrixbase> & adagraddsqrsum, const double adagradframes, char * bufferbegin, size_t buffersize)     // TODO: should be 'const', no?
    {
#ifdef JUSTDOUBLEBUFFER
        // swap with residual = previous one
        // 'plast' keeps the new raw gradient
        // previous raw gradient comes out from there
        // -> raw gradient gets delayed by 1
        {
            matrix & us = *this;
            if (cudamode)
                syncfromcuda (true);
            if (!plast)         // first time
                plast = make_shared<matrix> (rows(), cols());
            matrix & res = *plast;
            foreach_coord (i, j, us)
                ::swap (us(i,j), res(i,j));
            if (cudamode)
                synctocuda (false);
            return;
        }
#endif
        const auto & br = *mpistriperefs[stripe];
        if (br.end > buffersize)
            throw std::logic_error ("quantizeandfetchsubbatchstripe: unexpected mismatching buffer size");
        
#if 0   // does not work :(
        // scale for better quantization (using AdaGrad accumulator)
        // TODO: will fail for double-buffering!! So this is a prototype only.
        prescaleforquantization (adagraddsqrsum, adagradframes, br.i0, br.i1, br.j0, br.j1, false/*undo*/);
#endif

        auto & subresidual = *psubresidual;
#undef QUANTNOCUDA     // for testing the shared quantization code by using its CPU codepath
#ifndef QUANTNOCUDA
        if (cudamode)
        {
            // lazy init associated GPU stripes
            if (!qsubstripes[stripe])
                qsubstripes[stripe] = forcudadevice(br.deviceid).newqstripe (br.end - br.begin, false/*highpri*/);
            // perform quantization
            size_t deviceid = mpistriperefs[stripe]->deviceid;      // (model parallelism: GPU that this stripe resides on)
            auto & us = forcudadevice(deviceid);
            // note: this is kicking off an async process; someone needs to wait for completion
            //assert (qsubstripes[stripe]->size() == br.end - br.begin);
            us.quantizeandfetchqstripe (subresidual.forcudadevice(deviceid), br.i0, br.i1, br.j0, br.j1, qsubstripes[stripe].get(), bufferbegin + br.begin, br.bits, subresidual.forcudadevice(deviceid)/*in-place*/, 0); // -> sub-batch qbuffer
        }
        else
#endif
        {
            if (cudamode)
            {
                syncfromcuda (true);
                subresidual.syncfromcuda (true);
            }

            /*const*/ matrix & us = *this;
            matrix & res = subresidual;
            auto uspatch = br.patch (us);
            auto respatch = br.patch (res);
#if 0
if (rows() == 1100)
{
    fprintf (stderr, "quantizeandfetchsubbatchstripe: stripe being quantized[%d]=\n", (int) stripe);
    uspatch.glimpse();
}
#endif
            msra::math::matrixquantizer::quantize (uspatch, respatch, bufferbegin + br.begin, bufferbegin + br.end, br.bits, respatch, false/*allowaltlayout*/, 0/*reuserangescaled=no*/);
#if 0       // for bits==32, it must be the same  --delete this code later
            foreach_coord (i, j, respatch)
                if (respatch(i,j) != 0)
                    throw std::logic_error ("oops?");
            msra::math::matrixquantizer::unquantize (bufferbegin + br.begin, bufferbegin + br.end, br.bits, respatch, false/*add*/, false/*allowaltlayout*/);
            foreach_coord (i, j, respatch)
                if (respatch(i,j) != uspatch(i,j))
                    throw std::logic_error ("oops?");
            respatch.setzero();
#endif

            if (cudamode)
                subresidual.synctocuda (false);
        }
    }

    // step 2: wait for our stripe to be completely quantized into the CPU-side buffer
    // This is called in the sequence that the MPI aggregator has asked us to quantize.
    // The actual data transfer is in flight. This function waits for completion, so that the buffer content may be perused.
    // This function is *synchronous* (blocking) in CUDA mode by its nature.
    void syncfetchsubbatchstripe (size_t stripe)
    {
#ifdef JUSTDOUBLEBUFFER
        return;
#endif
#ifndef QUANTNOCUDA
        if (cudamode)
            forcudadevice(mpistriperefs[stripe]->deviceid).cpuneedsfetchqstripe (qsubstripes[stripe].get());
        // if we are not in cudamode, computation is synchronous on the CPU and has finished already
#endif
    }

private:
    // helper to create a stripe buffer
    // Note that these buffers are for use by the priority CUDA stream, so we cannot just schedule a setsero() here on the main stream (would run at the wrong time).
    bool lazymakebuffer (const mpistripebufferref & br, std::shared_ptr<rbmmodelmatrixbase<matrixbase>> & buffer, const char * what)
    {
        if (buffer)
            return false;
        fprintf (stderr, "unquantizeandaggregatestripe: creating %s stripe, range (%d,%d)..(%d,%d)\n", what, br.i0, br.j0, br.i1, br.j1), fflush (stderr);
        if (numcudadevices() == 1)
            buffer = make_shared<rbmmodelmatrixbase<matrixbase>> (br.i1 - br.i0, br.j1 - br.j0);
        else    // ugly: a rbmmodelmatrix is distributed over all GPUs, and we can't tell it to allocate only on the one we use, so we must over-allocate
            buffer = make_shared<rbmmodelmatrixbase<matrixbase>> (rows(), cols());
        buffer->entercomputation();
        buffer->setzero();
        buffer->synchronize();  // we are running on a background stream, so make sure this gets done before we use it
        return true;
    }
public:

    // step 3: accumulate quantized stripes into the stripe that we own
    // The data is on the CPU side currently, in the passed buffer.
    // We maintain the accumulator ourselves.
    // 'isfirst' is set in the first call (when accumulation starts)
    // If 'adagradkeepweight' or 'momentumkeepweight' or 'learningratescaling' are set, we do AdaGrad and/or momentum and/or scaling right here on the stripe (note: only pass this to the last call of an aggregation)
    // This function is asynchronous (non-blocking) in CUDA mode.
    void unquantizeandaggregatestripe (size_t ourstripe, size_t kfrom, const char * bufferbegin, size_t buffersize, bool isfirst,
                                       size_t mbframes, float adagradkeepweight, float targetadagradavdenom, float momentumkeepweight, float learningratescaling)
    {
#ifdef JUSTDOUBLEBUFFER
        return;
#endif

        const auto & br = *mpistriperefs[ourstripe];
        if (br.end > buffersize)
            throw std::logic_error ("unquantizeandaggregatestripe: unexpected mismatching buffer size");

        // lazily allocate the stripe accumulator
        lazymakebuffer (br, aggaccstripe, "accumulator");

        const bool hasfixedcoststep = (adagradkeepweight != 0.0f || momentumkeepweight != 0.0f || learningratescaling != 1.0f);
#undef REQUANTNOCUDA
#ifndef REQUANTNOCUDA
        if (cudamode)
        {
            // the unquant/aggregate/requant process runs on the GPU in a high-priority stream that preempts the main computation thread/NULL stream
            // lazy init associated GPU stripe
#if 0       // hack to try disabling the high-pri stream, to see whether it is the culprit
            // YAK! This makes accuracy tank. I.e. there is something seriously wrong here.
            static bool f = false;
            if (!f)
            {
                f = true;
                fprintf (stderr, "HACK: disabling high-pri stream\n"), fflush (stderr);
            }
            if (!qaccstripes[kfrom])
                qaccstripes[kfrom] = forcudadevice(br.deviceid).newqstripe (br.end - br.begin, false/*high-pri stream*/);
#else
            if (!qaccstripes[kfrom])
                qaccstripes[kfrom] = forcudadevice(br.deviceid).newqstripe (br.end - br.begin, true/*high-pri stream*/);
#endif
            auto & accumulator = aggaccstripe->forcudadevice(br.deviceid);
            // transfer the stripe to the GPU
            // TODO: ^^ if we unquantize on the GPU, then this one is already there, no need to transfer it back  --or is it? check whether that's correct in case of double buffering
            accumulator.assignqstripe (qaccstripes[kfrom].get(), bufferbegin + br.begin);
            // and unquantize
            accumulator.syncassignqstripeandunquantize (qaccstripes[kfrom].get(), br.bits, 0, br.i1 - br.i0, 0, br.j1 - br.j0, !isfirst/*add*/);
            // do fixed cost operations (AdaGrad, momentum) right here  --this makes them variable-cost
            if (hasfixedcoststep)
            {
                fprintf (stderr, "unquantizeandaggregatestripe: applying fixed-cost step in here, AdaGrad scaling %.6f, momentum %.6f, learning rate %.6f (%d x %d)\n", targetadagradavdenom, momentumkeepweight, learningratescaling, rows(), cols());
                if (lazymakebuffer (br, aggadagradsqrstripe, "AdaGrad sqr accumulator"))
                    aggadagradsqrframes = 0.0;  // also reset the counter
                lazymakebuffer (br, aggsmoothedstripe, "momentum-smoothed gradient");
                auto & adagradsqracc    = aggadagradsqrstripe->forcudadevice(br.deviceid);
                auto & smoothedgradient = aggsmoothedstripe->forcudadevice(br.deviceid);
                // accumulation count for AdaGrad
                aggadagradsqrframes = adagradkeepweight * aggadagradsqrframes + (1.0f - adagradkeepweight) * mbframes;
                // perform the steps
                const float targetadagradavdenom_x_sqrtadagradsqrframes = targetadagradavdenom * (float) sqrt (aggadagradsqrframes);
                accumulator.gradientfixups (qaccstripes[kfrom].get(), adagradsqracc, adagradkeepweight, targetadagradavdenom_x_sqrtadagradsqrframes, momentumkeepweight, smoothedgradient, learningratescaling);
                // now the aggregate gradient is the final one we want to add to the model
                // TODO: above is suboptimal: we write the gradient to both 'accumulator' and 'smoothedgradient'; avoidable with more logic (buffer swapping)
                // TODO (2nd step): merge both functions into one (maybe not that critical since it's only on a sub-stripe)
            }
            // these all ^^ are just submitting GPU work without waiting
        }
        else
#endif
        {
            // we operate on this accumulator
            // The accumulator is a matrix of the dimension of the stripe, with (0,0) at the top left corner of the stripe.
            matrix & accumulator = *aggaccstripe;
#if 0
if (rows() == 1100)
{
    fprintf (stderr, "unquantizeandaggregatestripe: stripe partial accumulator[%d] before accumulation=\n", (int) ourstripe);
    accumulator.glimpse();
}
#endif

            // unquantize
            msra::math::matrixquantizer::unquantize (bufferbegin + br.begin, bufferbegin + br.end, br.bits, accumulator, !isfirst/*add*/, true);
#if 0
if (rows() == 1100)
{
    fprintf (stderr, "unquantizeandaggregatestripe: stripe partial accumulator[%d]=\n", (int) ourstripe);
    accumulator.glimpse();
}
#endif
            if (hasfixedcoststep)
                throw std::logic_error ("unquantizeandaggregatestripe: distributing fixed cost not implemented for CPU");
        }
    }

    // step 4: quantize an aggregate stripe from the aggregation accumulator
    // Also send it back already, being the first stripe (we can't just call assignaggregatedstripe() because that one assigns from CPU RAM).
    // If 'reuserangescaled' is set, then the buffer must contain quantization ranges; pecifically when the same buffer that was used to send our GPU-side stripe to the CPU.
    // This function is *synchronous* (blocking) in CUDA mode since it waits for receiving the quantized stripe in CPU RAM.
    void quantizeandassignaggregatedstripe (size_t ourstripe, char * bufferbegin, size_t buffersize, size_t reuserangescaled)
    {
#ifdef JUSTDOUBLEBUFFER
        return;
#endif

        const auto & br = *mpistriperefs[ourstripe];
        if (br.end > buffersize)
            throw std::logic_error ("quantizeandassignaggregatedstripe: unexpected mismatching buffer size");

        // we operate on this accumulator
        // The accumulator is a matrix of the dimension of the stripe, with (0,0) at the top left corner of the stripe.
        if (!aggaccstripe)
            throw std::logic_error ("quantizeandassignaggregatedstripe: unallocated accumulator?");
        // lazily allocate the stripe quantization residuals
        if (!aggresstripe)  // TODO: use that lazy....() function above
        {
            if (numcudadevices() == 1)
                aggresstripe = make_shared<rbmmodelmatrixbase<matrixbase>> (br.i1 - br.i0, br.j1 - br.j0);
            else    // ugly: a rbmmodelmatrix is distributed over all GPUs, and we can't tell it to allocate only on the one we use, so we must over-allocate
                aggresstripe = make_shared<rbmmodelmatrixbase<matrixbase>> (rows(), cols());
            aggresstripe->entercomputation();
            aggresstripe->setzero();
        }

#ifndef REQUANTNOCUDA
        if (cudamode)
        {
            // lazy init associated GPU stripe
            if (!qresstripe)
                qresstripe = forcudadevice(br.deviceid).newqstripe (br.end - br.begin, true/*high-pri stream*/);
            auto & accumulator = aggaccstripe->forcudadevice(br.deviceid);
            auto & residual    = aggresstripe->forcudadevice(br.deviceid);
            // perform quantization
            accumulator.quantizeandfetchqstripe (residual, 0, br.i1 - br.i0, 0, br.j1 - br.j0, qresstripe.get(), bufferbegin + br.begin, br.bits, residual/*in-place*/, reuserangescaled);
            // CPU-wait for back-transfer to complete
            accumulator.cpuneedsfetchqstripe (qresstripe.get());    // note: this is synchronous/blocking
            // the stripe now exists on both GPU and CPU, but in the wrong qstripe --so we send it back once again :(
            // TODO: optimize this, e.g. use the same stripe in both cases (problem: the highpri flag)
            assignaggregatedstripe (ourstripe, bufferbegin, buffersize);
        }
        else
#endif
        {
            // we operate on this accumulator
            // Both accumulator and residual are matrices of the dimension of the stripe, with (0,0) at the top left corner of the stripe.
            matrix & accumulator = *aggaccstripe;
            matrix & residual    = *aggresstripe;

            // quantize
#if 0
            if (rows() == 1100)
            {
                fprintf (stderr, "quantizeandassignaggregatedstripe: stripe accumulator[%d]=\n", (int) ourstripe);
                accumulator.glimpse();
            }
#endif
            msra::math::matrixquantizer::quantize (accumulator, residual, bufferbegin + br.begin, bufferbegin + br.end, br.bits, residual, true/*allowaltlayout*/, reuserangescaled);

            // and already send it to the GPU
            assignaggregatedstripe (ourstripe, bufferbegin, buffersize);
        }
    }

    // step 5: move back an aggregated stripe (which is in quantized form)
    // this runs on a bg thread and only kicks off the CPU-to-GPU transfer
    // This function is asynchronous (non-blocking) in CUDA mode.
    void assignaggregatedstripe (size_t stripe, const char * bufferbegin, size_t buffersize)
    {
        //fprintf (stderr, "assignaggregatedstripe: bufferbegin = 0x%08x (stripe %d)\n", bufferbegin, stripe);
#ifdef JUSTDOUBLEBUFFER
        return;
#endif
#undef UNQUANTNOCUDA     // for testing the shared quantization code by using its CPU codepath
#ifndef UNQUANTNOCUDA
        if (cudamode)
        {
            const auto & br = *mpistriperefs[stripe];
            if (br.end > buffersize)
                throw std::logic_error ("assignaggregatedstripe: unexpected mismatching buffer size");
            // lazy init associated GPU stripes
            if (!qaggstripes[stripe])
                qaggstripes[stripe] = forcudadevice(br.deviceid).newqstripe (br.end - br.begin, false/*highpri*/);
            //assert (qaggstripes[stripe]->size() == br.end - br.begin);
            forcudadevice(mpistriperefs[stripe]->deviceid).assignqstripe (qaggstripes[stripe].get(), bufferbegin + br.begin);
            // Note: We don't submit the unquantization at this point, since GPU is likely still busy with main computation.
        }
        // if we are not in cudamode, there is nothing to sync, data is already at the right place
#endif
    }

    // step 6: unquantize a stripe from a CPU-side buffer sub-range (full-size buffer = bufferbegin/end)
    // This is called in the sequence that the MPI aggregator has the data and has sent it off to us already.
    // The actual data transfer is already in flight. This function waits for completion and then performs unquantization.
    // This function is asynchronous (non-blocking) in CUDA mode.
    // If 'addto' given then add to *addto, else write to 'this' (used when received gradient is already the final one and can be added directly to the model parameters)
    void syncassignaggregatedstripeandunquantize (size_t stripe, const char * bufferbegin, size_t buffersize, rbmmodelmatrixbase * addto/*or null*/)
    {
        //fprintf (stderr, "syncassignaggregatedstripeandunquantize: bufferbegin = 0x%08x (stripe %d)\n", bufferbegin, stripe);
#ifdef JUSTDOUBLEBUFFER
        return;
#endif
        if (!bufferbegin)   // (some old version allowed this, but no longer)
            throw std::logic_error ("syncassignaggregatedstripeandunquantize: must not be called with NULL buffer (no longer valid)");
        const auto & br = *mpistriperefs[stripe];
        const bool add = (addto != nullptr);    // in this case, result gets added to *addto rather than overwriting *this
#ifndef UNQUANTNOCUDA
        if (cudamode)
        {
            // perform quantization
            //size_t deviceid = mpistriperefs[stripe]->deviceid;      // (model parallelism: GPU that this stripe resides on)
            auto & us = addto ? *addto : *this;
            // note: this is kicking off an async process; someone needs to wait for completion
            if (!qaggstripes[stripe])
                throw std::logic_error ("syncassignaggregatedstripeandunquantize: unallocated qaggstripes?");
            us.forcudadevice(mpistriperefs[stripe]->deviceid).syncassignqstripeandunquantize (qaggstripes[stripe].get(), br.bits, br.i0, br.i1, br.j0, br.j1, add);
//syncfromcuda (true);
//sin(1.0f);
        }
        else
#endif
        {
            matrix & us = addto ? *addto : *this;
            if (br.end > buffersize)
                throw std::logic_error ("syncassignaggregatedstripeandunquantize: unexpected mismatching buffer size");
            auto uspatch = br.patch (us);
            msra::math::matrixquantizer::unquantize (bufferbegin + br.begin, bufferbegin + br.end, br.bits, uspatch, add, false/*allowaltlayout*/);
            if (cudamode)
                synctocuda (false);
#if 0
if (rows() == 1100)
{
    fprintf (stderr, "syncassignaggregatedstripeandunquantize: final aggregate stripe[%d]=\n", (int) stripe);
    uspatch.glimpse();
    glimpse("syncassignaggregatedstripeandunquantize: unquantized raw gradient=", false);
}
#endif
        }

#if 0   // does not work :(
        // scale for better quantization (using AdaGrad accumulator)
        // TODO: will fail for double-buffering!! So this is a prototype only.
        prescaleforquantization (adagraddsqrsum, adagradframes, br.i0, br.i1, br.j0, br.j1, true/*undo*/);
#endif
    }

#if 0
    // test function for quantization: quantize, then unquantize, while remembering the residual
    // THIS IS CURRENTLY UNTESTED and shall be deleted
    // TODO: merge this into mpiaggregate(), then DELETE THIS
    void quantizeunquantize (rbmmodelmatrixbase & residual, int Nbits, float accuracy)
    {
        checkcomputing();
        assert(rows() == residual.rows() && cols() == residual.cols());
        //if (cudamode)
        //{
        //    // TODO
        //}
        //else
        {
            matrix & us = *this;
            if (cudamode)
                syncfromcuda (true);
            matrix & residual1 = residual;
            size_t columndatasize = (us.rows()*Nbits + 31) / 32 * 4;    // (dup from msra::cuda::qstripe::columndatasize()...)
            std::vector<char> buffer (columndatasize);
            foreach_column (j, us)
            {
                float lower, upper;
                msra::math::columnquantizer::computerange (us, j, Nbits, accuracy, lower, upper);
                size_t ldNbits = msra::math::columnquantizer::ld (Nbits);
                msra::math::columnquantizer q (ldNbits, lower, upper);
                q.quantize (us, j, buffer.data(), residual1);
                q.unquantize (us, j, buffer.data());
            }
            if (cudamode)
            {
                synctocuda (false);
                residual.synctocuda (false);
            }
        }
    }
#endif

    // set the value to zero if less than threshold
    // This is used for model update
    void setto0ifabsbelow (float threshold) 
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setto0ifabsbelow (threshold);
        }
        else
            matrix::setto0ifabsbelow (threshold);
    }

    void setto0ifabsbelow2 (rbmmodelmatrixbase &  ref, float threshold) 
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setto0ifabsbelow2 (ref.forcudadevice (deviceid), threshold);
        }
        else
            matrix::setto0ifabsbelow2 (ref, threshold);
    }

    void setto0ifabsabove2 (rbmmodelmatrixbase &  ref, float threshold) 
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setto0ifabsabove2 (ref.forcudadevice (deviceid), threshold);
        }
        else
            matrix::setto0ifabsabove2 (ref, threshold);
    }

    void KhatriRaoProduct(const rbmstatevectorsref & m1, const rbmstatevectorsref & m2)
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).KhatriRaoProduct (m1.forcudadevice (deviceid), m2.forcudadevice (deviceid));
        }
        else
            matrix::KhatriRaoProduct(m1, m2);
    }

    // 'this' is the weight matrix
    void convolutionForward(rbmstatevectorsref & in, rbmstatevectorsref & out, const rbmmodelmatrixbase & bias, const msra::cuda::convolutionParams &convParams) const
    {
        if (!cudamode)
            throw runtime_error ("convolutionForward: must be called in cuda mode");
        in.setoutputstriping (notstriped);
        // process stripe by stripe
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            in.forcudadevice (deviceid).convolutionForward(out.forcudadevice(deviceid), this->forcudadevice(deviceid), bias.forcudadevice(deviceid), convParams);
    }

    // 'this' pointer is dw
    void computeCnnDeltaW(const rbmstatevectorsref & deltaM, const rbmstatevectorsref & vM, rbmstatevectorsref & deltatM, rbmstatevectorsref & vtM, float thisscale, float vhscale, const msra::cuda::convolutionParams &convParams)
    {
        if (!cudamode)
            throw runtime_error ("computeCnnDeltaW: must be called in cuda mode");
        // process stripe by stripe
        const_cast<rbmstatevectorsref &>(deltaM).setoutputstriping(notstriped);
        //this->setoutputstriping (notstriped);
        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            this->forcudadevice (deviceid).computeCnnDeltaW(deltaM.forcudadevice (deviceid), vM.forcudadevice(deviceid), deltatM.forcudadevice(deviceid), vtM.forcudadevice(deviceid), thisscale, vhscale, convParams);
    }

    void reshapecolumnproduct (const rbmstatevectorsref & eh, const rbmstatevectorsref & h, const bool isehtransposed)
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).reshapecolumnproduct (eh.forcudadevice (deviceid), h.forcudadevice (deviceid), isehtransposed);
        }
        else
            matrix::reshapecolumnproduct(eh, h, isehtransposed);
    }

    // sets all entries of matrix to value
    void setvalue (float value)
    {
        checkcomputing();
        if (cudamode)
        {
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid).setvalue(value);
        }
        else
            matrix::setvalue(value);
    }

    // set from a vector of column vectors
    // Note that the dimensions do not need to match; it is OK to copy only a sub-range. SVD does that. Yak!
    void setfrom (const std::vector<std::vector<float>> & v)
    {
        checknotcomputing();
        foreach_column (i, *this)
            if (v.size() < cols() || v[i].size() < rows())
                throw std::logic_error ("setfrom: input dimensions don't match");
            else
                memcpy (&col(i)[0], v[i].data(), rows() * sizeof(float));
    }

    // get weight matrix from W. [v-xieche]
    template <class AType> void getweightmatrix (AType & weightbuf)
    {
        if (cudamode)
            syncfromcuda (true);     // bring it into CPU space
        matrix & us = * this;
        foreach_coord (i, j, us)
            weightbuf (i,j) = us (i, j);
        if (cudamode)
            synctocuda (false);     // move it back to GPU
    }

    // assign weight matrix from W. [v-xieche]
    template <class AType> void assignweightmatrix (AType & weightbuf)
    {
        if (cudamode)
            syncfromcuda (true);     // bring it into CPU space
        matrix & us = * this;
        foreach_coord (i, j, us)
            us (i, j) = (float)weightbuf (i,j);
        if (cudamode)
            synctocuda (false);     // move it back to GPU
    }


    // multiply the W with n. used for temp experiment to see what happend when sigmoid become steeper. [v-xieche]
    void multiplywith (float n)
    {
        matrix & us = *this;
        if (cudamode && computing)   // if not computing, shoulbd be writen to model file
            syncfromcuda (true);
        foreach_coord (i, j, us)
            us (i, j) *= n;
        if (cudamode && computing)
            synctocuda (false);
    }


    // set a matrix to a block-diagonal structure, by setting off-elements to 0
    // If 'poolblocks' then each block gets replaced by the average over all blocks.
    // This is intended to support input-layer transforms when inputs are augmented by neighbor frames.
    void setblockdiagonal (size_t diagblocks, bool poolblocks, const size_t &/*why a reference if const??*/ numofclasses, const size_t & roundupunit, const bool setidentity)     // modified by Hang Su adaptation
    {
        checkcomputing();
        if (diagblocks == 1)        // nothing to do
            return;
#if 0
        if (cudamode)
        {
            // BUGBUG: THIS IS BROKEN for multi-devices--we have no access to the other blocks!
            if (numcudadevices() > 1)
                throw runtime_error ("setblockdiagonal: currently does not support multiple devices--bummer");

            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                size_t fr, fc, nr, nc;  // coordinates in CPU-side matrix
                devicedim (deviceid, cudastriping, fr, fc, nr, nc);
                const size_t firstcol = fc;
                this->forcudadevice (deviceid, stripedwrtcols).patchasblockdiagonal (diagblocks, poolblocks, firstcol);
            }
        }
        else
#else
        {
            if (cudamode)
                syncfromcuda (true);    // CUDA mode: matrix is on GPU, so bring it into CPU space
            fprintf (stderr, "setblockdigonal : diagblocks = %d, poolblock = %d\n", diagblocks, poolblocks);
            matrix & us = *this;
            // TO BE TESTED
            //All info is available in diagblocks and us.cols()/us.rows()

            if (setidentity)
            {
                foreach_coord(i,j,us)
                {
                    if ( i == j )
                        us(i,j) = 1;
                    else
                        us(i,j) = 0;
                }
            }
            else
            {
                const size_t uscols = (us.rows() + roundupunit) * numofclasses - roundupunit;     // the relationship between us.cols() and us.rows() -roundupunit because the last class is not rounded up
                if(us.cols() !=  uscols && us.cols() != 1)       throw std::logic_error ("setblockdiagonal:  the size of matrix is not a correct matrix for linear network or an array ");
                if(us.cols() == uscols)  // execute it only for rounded up adaptation matrix
                {
                    size_t feadim = us.rows() / diagblocks;      // it should be 39 in the normal conditioin
                    size_t classdim = us.cols() / numofclasses;
                    if(us.rows() != diagblocks * feadim )    throw std::logic_error ("setblockgiagonal: the row of matrix can't divided by diagblocks");
                    for (size_t j = 0; j < classdim - roundupunit; j++)         // set elements that are not in block to zeros
                    {
                        for (size_t i = 0; i < us.rows(); i++)
                        {
                            if(size_t(i / feadim) != size_t(j / feadim))
                            {
                                for ( size_t k = 0; k < numofclasses; k++)
                                    us(i, j + classdim *k) = 0;
                            }
                        }
                    }
                    for (size_t j = 0; j < roundupunit; j++)                    // set round up units in the matrix to zeros
                    {
                        for (size_t i = 0; i < us.rows(); i++)
                        {
                            for (size_t k = 0; k < numofclasses - 1; k++)       // -1 because the last class is not blowed up
                            {
                                us(i, j + us.rows() + classdim *k) = 0;
                            }
                        }
                    }
                    if(poolblocks)				//need to calculate the average from the blocks. 
                    {
                        for (size_t classid = 0; classid < numofclasses; classid++)     //modify adaptation matrix for each class 
                        {
                            for(size_t i = 0; i < feadim; i ++) for(size_t j = 0; j < feadim; j ++)  // first calculate the sum of the point corresponding at every position.
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(i, j + classdim * classid) += us(k*feadim + i, k*feadim + j + classdim * classid);
                            for(size_t i = 0; i < feadim; i ++)  for(size_t j = 0; j < feadim; j ++)  // assign W as the average of the blocks.
                            {
                                us(i, j + classdim * classid) = us(i, j + classdim * classid) / diagblocks;
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(k*feadim + i, k*feadim + j + classdim * classid) = us(i, j + classdim * classid);
                            }
                        }
                    }
                }
                else if(us.cols() == 1)   // for a
                {
                    size_t classdim = (us.rows() + roundupunit)/ numofclasses;
                    size_t feadim = ((us.rows() + roundupunit)/ numofclasses - roundupunit) /  diagblocks;      // it should be 39 in the normal conditioin
                    if((us.rows() + roundupunit)/ numofclasses - roundupunit != diagblocks * feadim )    throw std::logic_error ("setblockgiagonal: the row of matrix can't divided by diagblocks");
                    for (size_t classid = 0; classid < numofclasses - 1; classid ++)        // set roundup units to zeros
                    {
                        for (size_t i = 0; i < roundupunit; i++)
                            us(i + classdim - roundupunit + classid * classdim, 0) = 0;
                    }
                    if(poolblocks)
                    {
                        for (size_t classid = 0; classid < numofclasses; classid ++)
                        {
                            for(size_t i = 0; i < feadim; i ++)
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(i + classid * classdim, 0) += us(k*feadim + i + classid * classdim, 0);
                            for(size_t i = 0; i < feadim; i ++)
                            {
                                us(i + classid * classdim, 0) = us(i + classid * classdim, 0) / diagblocks;
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(k*feadim + i + classid * classdim, 0) = us(i + classid * classdim, 0);
                            }
                        }
                    }
                }
                else  throw std::logic_error ("setblockdiagonal : The input matrix is not a square or an array. can't be processed");

            }
            if (cudamode)
                synctocuda (false);     // move it back to GPU
        }
#endif
    }


    // modified version of matproduct for convolutional model
    // matrix product dW = dW * scale + v h', dW = 'this'
    // v and h are per-frame unique and thus live in CUDA space.
    void convolutionalScaleAndAddMatProduct(float thisscale, const rbmstatevectorsref & v, rbmstatevectorsref & vt, const rbmstatevectorsref & h, rbmstatevectorsref & ht, const float vhscale, const msra::cuda::convolutionParams & params)
    {
        checkcomputing();
        if (!cudamode)
            throw runtime_error ("convolutionalScaleAndAddMatProduct: must be called in cuda mode");
        
        computeCnnDeltaW(h, v, ht, vt, thisscale, vhscale, params);
    }

    // dump a matrix from CUDA side using CUDA printf
    void dump (char *name) const
    {
        if (!cudamode)
            throw runtime_error ("convolutionalScaleAndAddMatProduct: must be called in cuda mode");

        for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            this->forcudadevice (deviceid).dump(name);
        //syncfromcuda(true);        
        //printmatf(name, *this);
        
    }

    // varnorm() --perform variance normalization of every row of this matrix
    //  out(i,j) = (this(i,j) - mean(i)) / diagvar(i)
    // Result gets written to a different vector.
    void varnorm (const acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & diagvar)
    {
        if (cudamode)
        {
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                forcudadevice (deviceid).meanvarnorm (diagvar.forcudadevice (deviceid)/*dummy*/, false,
                                                      diagvar.forcudadevice (deviceid),
                                                      forcudadevice (deviceid)/*result written in-place*/);
        }
        else
        {
            throw std::logic_error ("varnorm: not implemented for CPU mode");
        }
    }
    
    // NUMA-localized matrix product dW = dW * scale + v h', dW = 'this'
    // h is transposed locally here into httmp. httmp must have been allocated at correct dimensions already.
    // v and h are per-frame unique and thus live in CUDA space.
    template<class vtype, class htype>  // rbmstatevectorsref or model matrix
    void scaleandaddmatprod (float thisscale, const vtype & v, const htype & h, const float vhscale,
                             msra::math::ssematrix<matrixbase> & httmp, cachedmatrix & cachedvs, cachedmatrix & cachedhts)
    {   // used in updatedeltas(): dW.scaleandaddmatprod_numa (momentum, v, ht, cachedvts, cachedhts);
        checkcomputing();
        if (cudamode)
        {
            //  - input: full copy of v
            //  - input: horizontal stripes of h
            //  - compute vertical stripes of dW, keep them separate
            // process stripe by stripe
            const_cast<vtype&> (v).makeinputstriping (notstriped);  // TODO: clean this up!
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice (deviceid, stripedwrtcols).gemm (thisscale, v.forcudadevice (deviceid, notstriped), false, *h.stripeforcudadevice (deviceid, stripedwrtrows), true/*h'*/, vhscale);
        }
        else
        {
            // if (vhscale != 1.0f) throw logic_error ("scaleandaddallcols: cannot yet scale the summand--implement this");   // TODO: implement this
            // transpose h -> httmp
            httmp.resizeonce (h.cols(), h.rows());
            h.fornuma().transpose (httmp);
            const matrixbase & ht = httmp;  // 'h' no longer used below, only 'httmp'
            scaleandaddmatprod_numa (thisscale, v.fornuma(), false, ht, cachedvs, cachedhts, *this, vhscale);
        }
    }

    // special-purpose function for on-demand LL evaluation
    // This computes only one row, row 'i', of the result matrix.
    // It assumes that the copy of the model parameters in CPU RAM are valid.
    void matprod_col_mtm (const msra::math::ssematrixstriperef<matrixbase> & v, msra::math::ssematrixstriperef<matrixbase> & h, const rbmmodelmatrixbase & a, size_t i) const
    {
        assert (h.rows() == 1); // only one result row
        checkcomputing();

        // h = us_i' v + a_i
        auto wi = matrixstripe (const_cast<matrixbase &> ((matrixbase &) *this), i, 1);
        h.matprod_mtm (wi, v);
        foreach_column (j,h)
            h(0,j) += ((const matrixbase &)a)[i];
    }

#ifdef MULTICUDA
    void matprod_mtm (const cudadistributedmatrix & v, cudadistributedmatrix & h, const cudadistributedmatrix & a) const
    {
        checkcomputing ();
        if (cudamode)
        {
            // it seems to modify the stripe method, should don't need it
            // v.makeinputstriping (notstriped);       // redistribute if needed
            h.setoutputstriping (stripedwrtrows);   // result are stripes although we have full allocation
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                h.stripeforcudadevice (deviceid, stripedwrtrows)->gemm (0.0f, this->forcudadevice (deviceid, stripedwrtcols),
                true, *v.stripeforcudadevice (deviceid, notstriped), false, 1.0f);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                h.stripeforcudadevice (deviceid, stripedwrtrows)->addtoallcolumns (a.forcudadevice (deviceid, stripedwrtrows));
        }
    }
#endif

    // NUMA-localized matrix product h = W' v , W' is the transpose of 'this', a is added to all columns
    // 'this' is model matrix in CUDA RAM.
    // v is frame-unique input, and h frame-unique output.
    void matprod_mtm (const rbmstatevectorsref & v, cachedmatrix & cachedWs, cachedmatrix & cachedvs, rbmstatevectorsref & h, cachedmatrix &/*cachedhs*/, const float weight = 0.0f) const
    {
        checkcomputing();
        if (cudamode)
        {
#ifdef PEEKCUDA
            v.fromcuda();
            this->syncfromcuda();
#endif

            // -- improving parallelism by sub-minibatches (overlapped operation):
            // We use row striping, i.e.
            //  - output per GPU is a partial vector that needs to be reassembled across GPUs
            //  - input can be either a full vector (broadcast) or a partial vector
            // If input is a partial vector then this operation needs to be done:
            //  - each GPU sends its partial vector to all other K-1 GPUs
            //  - each receiving GPU must wait until it has received all parts
            // Since for output layer, we don't need to reassemble in the same way (there's a trick), we instead reassemble upon the input (lazily).
            // Overlapped operation:
            //  - general ideas:
            //     - split computation into sub-minibatches without data dependency between them (makes life easy)
            //     - when entering, all data is either computed, or by queueing data transfers after the ongoing computations, they will be when data transfer commences
            //     - when exiting, we leave the last computation ongoing
            //     - for each sub-minibatch, GPUs pull their inputs and launch the computation; at the end, there are ongoing computations but no transfers in-flight
            //  - pre-condition: no data transfer in flight; computation may be ongoing
            //  - for all sub-minibatches:
            //     - for each GPU:
            //        - queue parallel transfers of partial vectors of sub-minibatch from all other K-1 GPUs
            //          (the data may be being produced by ongoing compute ops, so need to queue w.r.t. source GPU so that it starts after all computation has finished)
            //          Queued transfers are identified as (source device, destination device, sub-minibatch index)
            //          This can be done inside makeinputstriping() if we extend it to take a range.
            //  - i.e. we schedule all transfers, since we can (they only depend on the last computation in each respective GPU)
            //  - for all sub-minibatches:
            //     - for each GPU:
            //        - wait for all needed incoming transfers (source device, destination device, sub-minibatch index)
            //          What's coming in are partial vectors that are being assembled. Now the data is ready to be consumed.
            //          Note that this must be done after all transfers have been scheduled, otherwise those transfers will wait on the reading-side wait operation...
            //        - launch gemm on those partial vectors
            //          Note that within the current matprod_mtm(), noone depends on the result of this gemm() call.
            //  - post-condition: no data transfer in flight; computation may be ongoing
            //  - the queue will have this form:
            //     - (send)*
            //     - (wait, gemm)*   i.e. gemm() will work off the transferred vectors as fast as it can
            // E.g., for M=256 out of 1024 and fully parallel computation/transfer of equal duration, it will take 5/4 of the time (speed-up K * 4/5)

            // -- further improvement by cross-layer dependencies
            // Focus: forwardprop() chain with simple sigmoid()
            //  - now:
            //     - each GPU computes on the NULL stream
            //     - copy streams are allocated to not sync with NULL stream
            //     - thus, anything but copying syncs with NULL stream
            //  - approach: use one sub-stream for each sub-minibatch
            //     - in gemm(), add columns, and sigmoid()
            //     - this will give us optimal overlap all the way to the top, and we can use submbsize = 512 while being optimal
            //     - any function that is not sub-stream aware will use NULL stream which syncs against all substreams -> fully compatible and safe
            //  - new matrix function setsubstream() and associated onsubstream class
            //  - BUGBUG: stream sync in send() does not work;

            stripingconfig sc (h);      // configuration for model parallelism is determined in here

#ifdef TIME_MTM
            static double totaltime = 0.0;
            static double totalmacs = 0.0;
            static double totalcopytime = 0.0;
            static double totalbytes = 0.0;
            v.synchronize();
            h.synchronize();
            synchronize();
            auto_timer copycost;
#endif

            // perform the needed format conversions
            bool needreceive = false;                   // 'true' if send()s were launched that require receive()
            if (sc.enableasynctransfers)                // launch all transfer operations, in a meaningful order
                needreceive |= const_cast<rbmstatevectorsref &> (v).makeinputrowstripingasync (sc);    // if this returns false then no send() was launched (data already in the right format, as in first layer)
            else    // do it synchronously (inefficient; except zero cost for K == 1)
                v.makeinputstriping (notstriped);       // redistribute if needed (reassemble partial vectors into complete ones and distribute)

#ifdef TIME_MTM // force sync between copy and compute, so we can separate the measurement
            if (needreceive)   // interleaved receving and computation
            {
                // process sub-batch by sub-batch
                size_t te;
                for (size_t ts = 0; ts < sc.numframes; ts = te)
                {
                    te = ts + submbframes;      // frame range ts..te-1
                    if (te > sc.numframes)
                        te = sc.numframes;
                    //fprintf (stderr, "matprod_mtm receive() for frames %d..%d\n", ts, te); fflush (stderr);
                    for (size_t todeviceid = 0; todeviceid < K; todeviceid++)    // we are computing the stripe in 'todeviceid'
                    {
                        // first synchronize v against all required blocks
                        {
                            const size_t numrows = v.rows();
                            for (size_t n = 1; n < K; n++)
                            {
                                size_t fromdeviceid = (todeviceid + n) % K;   // we avoid parallel access to the same device
                                size_t fr, nr;  // row range available on this device; col range (=time range) given by (ts,te)
                                onedevicedim (fromdeviceid, true, numrows, fr, nr);
                                // synchronize stream until all data is received
                                const_cast<rbmstatevectorsref &> (v).forcudadevice (todeviceid).receive (const_cast<rbmstatevectorsref &> (v).forcudadevice (fromdeviceid), fr, fr + nr, ts, te);
            totalbytes += sizeof (float) * nr * (te - ts);
                            }
                        }
                    }
                }
            }
            needreceive = false;
            v.synchronize();
            totalcopytime += copycost;
#endif

            // TODO: rename h to z
            // perform matrix product stripe by stripe
            h.setoutputstriping (stripedwrtrows);       // result are stripes although we have full allocation
            if (needreceive || sc.enablesubbatchcomputation)   // interleaved receving and computation
            {
                sc.foreachsubbatchanddevice ([&] (size_t ts, size_t te, size_t todeviceid, size_t substream)
                {
                    // we are computing the stripe in 'todeviceid'
                    // first synchronize v against all required blocks
                    if (needreceive)
                    {
                        const size_t numrows = v.rows();
                        for (size_t n = 1; n < sc.K; n++)
                        {
                            size_t fromdeviceid = (todeviceid + n) % sc.K;   // we avoid parallel access to the same device
                            size_t fr, nr;  // row range available on this device; col range (=time range) given by (ts,te)
                            onedevicedim (fromdeviceid, true, numrows, fr, nr);
                            // synchronize stream until all data is received
                            msra::cuda::onsubstream sub (const_cast<rbmstatevectorsref &> (v).forcudadevice (todeviceid), substream);    // receive for this substream
                            const_cast<rbmstatevectorsref &> (v).forcudadevice (todeviceid).receive (const_cast<rbmstatevectorsref &> (v).forcudadevice (fromdeviceid), fr, fr + nr, ts, te);
                        }
                    }
                    if (sc.enablesubbatchcomputation)
                    {
                        // now we can launch the computation
                        auto hstripe = h.stripeforcudadevice (todeviceid, stripedwrtrows);
                        auto vstripe = const_cast<rbmstatevectorsref &> (v).stripeforcudadevice (todeviceid, notstriped);
                        unique_ptr<msra::cuda::matrix> hstripesub (hstripe->patch (0, hstripe->rows(), ts, te));  // operate on a sub-range of frames
                        unique_ptr<msra::cuda::matrix> vstripesub (vstripe->patch (0, vstripe->rows(), ts, te));
                        msra::cuda::onsubstream sub (*hstripesub.get(), substream);  // compute on this substream
                        hstripesub->gemm (weight, this->forcudadevice (todeviceid, stripedwrtcols), true, *vstripesub.get(), false, 1.0f);
                    }
                });
            }

#ifdef TIME_MTM
            v.synchronize();
            h.synchronize();
            synchronize();
            totaltime += copycost;
            totalmacs += sc.numframes * this->cols() * this->rows();
            copycost.show ("copy complete");
            // weird: somehow this exlicit sync makes it faster (in case of doing a full batch anyway)
            fprintf (stderr, "copy time: %.2f GB/s (total time %.3f s for %.2f MB)\n", totalbytes / totalcopytime / 1e9, totalcopytime, totalbytes / 1e6);
            fprintf (stderr, "async time: %.2f G ma/s (total time %.3f s for %.2f G macs)\n", totalmacs / (totaltime - totalcopytime) / 1e9, totaltime - totalcopytime, totalmacs / 1e9);
            fprintf (stderr, "matprod time incl. copy: %.2f G ma/s (total time %.3f s for %.2f G macs)\n", totalmacs / totaltime / 1e9, totaltime, totalmacs / 1e9);
#endif

            // when doing async computation, it has been woven into the loop; if not, we do it at once here for the entire matrix
            if (!sc.enablesubbatchcomputation)
                for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                    h.stripeforcudadevice (deviceid, stripedwrtrows)->gemm (weight, this->forcudadevice (deviceid, stripedwrtcols), true, *v.stripeforcudadevice (deviceid, notstriped), false, 1.0f);
            // done

#ifdef TIME_MTM
            h.synchronize();
            copycost.show ("sgemm complete");
#endif

#ifdef PEEKCUDA
            h.fromcuda();
            a.syncfromcuda();
            h.fromcuda();
#endif
        }
        else
        {
            matprod_mtm_numa (*this, v.fornuma(), cachedWs, cachedvs, h.fornuma(), weight);
        }
    }

    // matrix product h = h * weight + W' v + a, W' is the transpose of 'this', a is added to all columns
    // 'this' and 'a' are model parameters in CUDA RAM.
    // v is frame-unique input, and h frame-unique output.
    // used in vtoz(): W.matprod_mtm_numa (v, cachedWs, cachedvs, h, a);     // h = W' v + a
    void matprod_mtm (const rbmstatevectorsref & v, cachedmatrix & cachedWs, cachedmatrix & cachedvs, rbmstatevectorsref & h, cachedmatrix & cachedhs/*unused*/, const rbmmodelmatrixbase & a, cachedmatrix &/*cacheda1s*/, const float weight = 0.0f) const
    {
        // matrix product
        matprod_mtm (v, cachedWs, cachedvs, h, cachedhs, weight);

        // add bias
        if (cudamode)
        {
#if 1
#ifdef PEEKCUDA
            h.fromcuda();
            a.syncfromcuda();
#endif
#ifdef TIME_CUDA
            h.synchronize();
            auto_timer copycost;
#endif
            stripingconfig sc (h);      // configuration for model parallelism is determined in here
            const auto stripedwrtrowsmode = stripedwrtrows;   // compiler bug: constant 'stripedwrtrows' unknown inside lambda
            if (sc.enablesubbatchcomputation)
            {
                sc.foreachsubbatch ([&] (size_t ts, size_t te, size_t substream) -> void
                {
                    for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                    {
                        auto hstripe = h.stripeforcudadevice (deviceid, stripedwrtrowsmode);
                        const msra::cuda::matrix & astripe = a.forcudadevice (deviceid, stripedwrtrowsmode);
                        unique_ptr<msra::cuda::matrix> hstripesub (hstripe->patch (0, hstripe->rows(), ts, te));  // operate on a sub-range of frames
                        unique_ptr<msra::cuda::matrix> astripesub (const_cast<msra::cuda::matrix&>(astripe).patch (0, astripe.rows(), 0, 1));
                        msra::cuda::onsubstream sub (*hstripesub.get(), substream);  // compute on this substream
                        hstripesub->addtoallcolumns (*astripesub.get());
                    }
                });
            }
            else
                for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                    h.stripeforcudadevice (deviceid, stripedwrtrows)->addtoallcolumns (a.forcudadevice (deviceid, stripedwrtrows));
#ifdef TIME_CUDA
            h.synchronize();
            copycost.show ("addtoallcolumns complete");
#endif
#ifdef PEEKCUDA
            h.fromcuda();
#endif
#else       // do this CPU-side
            // This is mariginally slower (4-5% overall or so) but WAY more accurate.
            // The CUDA version below has significant numerical differences after only few minibatches.
            // It is not clear why that would be.
            // Solution: Write our own CUDA kernel to do this.
            //cachedhs.fetch_cuda (h.fornuma());
            const_cast<rbmmodelmatrixbase&>(a).fetch();  // current value of 'a'
            //h.fornuma() += a;
            h.fromcuda() += a;
            h.tocuda();
#endif
        }
        else
        {
            h.fornuma() += a;
        }
    }

    // add matrix product to a vector: h = h + W' v
    // W' is the transpose of 'this'
    // v is frame-unique input, and h frame-unique output.
    // (This is not really used currently; was for forwardpropdelta() which was used in one sequence-training experiment.)
    void addmatprod_mtm (const rbmstatevectorsref & v, rbmstatevectorsref & h) const
    {
        checkcomputing();
        if (cudamode)
        {
            // needed format conversions
            v.makeinputstriping (notstriped);       // redistribute if needed
            // process stripe by stripe
            h.setoutputstriping (stripedwrtrows);   // result are stripes although we have full allocation
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                h.stripeforcudadevice (deviceid, stripedwrtrows)->gemm (1.0f, this->forcudadevice (deviceid, stripedwrtcols), true, *v.stripeforcudadevice (deviceid, notstriped), false, 1.0f);
        }
        else
        {
            throw std::logic_error ("matprod_mtm: implement this");
        }
    }

    // add matrix product to a matrix: C = thisscale * C + otherscale * A' B
    void sgemm_mtm (float Cscale, const rbmmodelmatrixbase & A, /*const*/ rbmmodelmatrixbase & B, float ABweight)
    {
        checkcomputing();
        if (cudamode)
        {
            // needed format conversions
            B.makeinputstriping (notstriped);       // redistribute if needed
            // TODO: use the applyxxx function
            // process stripe by stripe
            setoutputstriping (stripedwrtrows);     // result are stripes
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                stripeforcudadevice (deviceid, stripedwrtrows)->gemm (Cscale, A.forcudadevice (deviceid, stripedwrtcols), true, *B.stripeforcudadevice (deviceid, notstriped), false, ABweight);
        }
        else
        {
            throw std::logic_error ("sgemm: implement this");
        }
    }


#ifdef MULTICUDA
    void matprod_mm (const cudadistributedmatrix & eh, cudadistributedmatrix & ev, const float vscale=0.0f) const
    {
        if (cudamode)
        {
            // same structure as matprod_mm below  --that's the master copy, see there for comments
            ev.setoutputstriping (notstriped);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                ev.forcudadevice (deviceid).gemm (vscale, this->forcudadevice (deviceid, stripedwrtcols), false, *eh.stripeforcudadevice (deviceid, stripedwrtrows), false, 1.0f);
            ev.sumacrossdevices (stripedwrtrows); // how to do this ???!!!
        }
    }
#endif

    // ev = W eh, W = 'this' is not transposed
    // used in ehtoev() (back-propagation): W.matprod_mm_numa (eh, cachedWts, cachedhs, ev);   // v = W h
    // also used as a sub-function of matprod_mm() with bias below, hence we use the variable names from there
    void matprod_mm (const rbmstatevectorsref & h, cachedmatrix & cachedWts, cachedmatrix & cachedhs, rbmstatevectorsref & v, cachedmatrix &/*cachedevs*/,
                     acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & gpuexchangebuffer,
                     const float vscale = 0.0f) const
    {
        checkcomputing();
        if (cudamode)
        {
            // multi-GPU:
            //  - input: row stripes of h
            //  - input: col stripes of W
            //  - output: full copy of v
            //  - compute vertical stripes of dW, keep them separate

            h.makeinputstriping (stripedwrtrows);   // (should be in this format)
            v.setoutputstriping (notstriped);       // result are partial sums --naming may be confusing: not striped indeed (full dim) but different and incomplete (partial)

            // For multi-GPU setups, every GPU...
            //  - gemm() computes: full-dim partial sum of v that is based on the GPU's stripes of W and h
            //  - needs: aggregation of the above for its row stripe (of 1/K dimension)
            // Thus:
            //  - every GPU must receive its respective 1/K-dim stripe from every other GPU; K-1 of them
            //  - and add it to its own stripe in-place (*after* it has transferred out its partial sum to all others)
            //  - and thus must happen interleaved with the big matrix product of the other half minibatch
            // Approach:
            //  - keep buffers to hold the partial sums from all other K-1 GPUs
            //  - dimension is (K-1 GPUs x 1/k dimension)
            //  - schedule transfer first, sync (to prepare for in-place summation)
            //  - schedule summations (substream gems(); right before the matrix product that consumes it),
            //    i.e. onto the respective stream
            //  - this will also allow us to encapsulate the send()/receive() pair inside this function
            //  - the memory size is ~(K-1)/K the memory of the activations themselves, so bearable
            // The above must happen in an overlapped fashion with gemm().
            // Buffers:
            //  - each GPU needs K-1 buffers of the height of the row stripe it is supposed to hold at the end
            //  - for simplicity with out objects, we create one buffer that is (K-1) x wider, and use patches
            //  - buffers are declared 'notstriped' but really are different on each GPU

            // process stripe by stripe, in an overlapped fashion

            stripingconfig sc (h);      // configuration for model parallelism is determined in here
//#define TIME_TM
#ifdef TIME_TM
            static double totaltime = 0.0;
            static double totalmacs = 0.0;
            static double totalcopytime = 0.0;
            static double totalbytes = 0.0;
            v.synchronize();
            h.synchronize();
            synchronize();
            auto_timer copycost;
            sc.enablesubbatchcomputation = true;    // we want to time overlapped operation
#endif
//sc.enableasynctransfers = false;    // we want to time overlapped operation
//sc.enablesubbatchcomputation = false;    // we want to time overlapped operation
            if (sc.enableasynctransfers || sc.enablesubbatchcomputation)
            {
                const size_t numrows = v.rows();
                const size_t numframes = v.cols();
                // allocate buffers
                const size_t reqbufrows = (numrows + sc.K-1) / sc.K;    // maximum row height
                const size_t reqbufcols = numframes * sc.K;               // we horizontally stack all buffers --note: wasting one for current GPU; only need K-1
                gpuexchangebuffer.ensuresize (reqbufrows, reqbufcols);

                // schedule all subminibatches for each computation device
                // We schedule the full sequence for each subminibatch in one go, to get the dependencies right.
                sc.foreachsubbatch ([&] (size_t ts, size_t te, size_t substream) -> void
                {
                    // launch the computation
                    for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                    {
                        auto hstripe = const_cast<rbmstatevectorsref &> (h).stripeforcudadevice (deviceid, stripedwrtrows);
                        auto vstripe = v.stripeforcudadevice (deviceid, notstriped);        // each device computes a full-dimension partial sum
                        unique_ptr<msra::cuda::matrix> hstripesub (hstripe->patch (0, hstripe->rows(), ts, te));  // operate on a sub-range of frames
                        unique_ptr<msra::cuda::matrix> vstripesub (vstripe->patch (0, vstripe->rows(), ts, te));
                        msra::cuda::onsubstream sub (*vstripesub.get(), substream);  // compute on this substream
                        vstripesub->gemm (vscale, this->forcudadevice (deviceid, stripedwrtcols), false, *hstripesub.get(), false, 1.0f);
                    }
                    // Now the subminibatches of each 'v' contain partial sums of full dimension.
                    // But we need full sums of partial dimension.
                    // That's what the following accomplishes.

                    //fprintf (stderr, "###\n"); fflush (stderr);

                    for (size_t step = 1; step <= 3; step++)    // 3 steps that operate on the same sub-matrices
                    {
                        // step 1: schedule all data transfers from current device to all others
                        //         In cross-bar fashion (first all send to next GPU, then all send to next-next etc.).
                        // step 2: schedule substream to wait until all outgoing transfers are done, so that we can overwrite v in-place with summations
                        // step 3: add transferred chunk to v

                        for (size_t n = 1; n < sc.K; n++)
                        {
                            for (size_t deviceid = 0; deviceid < sc.K; deviceid++)
                            {
                                // 'deviceid' is the "current" device we (1) transfer from and (2) compute on
                                // all comments below imply "on current subminibatch"
                                size_t todeviceid = (deviceid + n) % sc.K;              // we avoid parallel access to the same device
                                // we need to copy rows (fr,fr+nr) from current device to todevice's buffer set aside for receiving from current device
                                // source --remember we got a full-dim vector, but target only wants part of that
                                auto & vsrc = v.forcudadevice (deviceid);               // the full-dim CUDA matrix on source device
                                size_t fr, fc, nr, nc;
                                v.devicedim (todeviceid, stripedwrtrows, fr, fc, nr, nc); // get the row substream our target needs from us
                                unique_ptr<msra::cuda::matrix> vsrcstripe (vsrc.patch (fr, fr + nr, fc, fc + nc));  // living on src GPU, the stripe that the target needs
                                // target --we copy into a temp buffer on the target
                                auto & tgtbuffer = gpuexchangebuffer.forcudadevice (todeviceid);    // CUDA buffer on target device to send data to
                                const size_t tbegin = numframes * deviceid;             // each source has its own buffer in the target; they are stacked col-wise next to each other in the big buffer
                                const size_t tend = tbegin + numframes;
                                unique_ptr<msra::cuda::matrix> vtgtbuf (tgtbuffer.patch (0, vsrcstripe->rows(), tbegin, tend)); // lives on 'todeviceid'
                                if (step == 1)              // now send
                                {
                                    msra::cuda::onsubstream sub (*vsrcstripe, substream);   // configure it to condition the send() on this substream (transfer depends on this computation)
                                    vsrcstripe->send (*vtgtbuf, 0, nr, ts, te);
                                }
                                else if (step == 2)         // sync against completion of send (don't do that before all sends have been submitted, otherwise they may wait on each other...)
                                {
                                    msra::cuda::onsubstream sub (*vtgtbuf, substream);   // configure it to condition subsequent computation on this substream to depend on this send()
                                    vtgtbuf->receive (*vsrcstripe, 0, nr, ts, te);
                                }
                                else if (step == 3)         // accumulate --this runs after all gemms, which is fine because the GPU is not free anyway (nothing lost)
                                {
                                    // from 'deviceid' to 'todeviceid'
                                    // Data has been received, it now lives on 'todeviceid' inside 'vtgtbuf.' We can now safely add stuff in-place to 'v.'
                                    // We need to add 'vtgtbuf' to the respective row-stripe of 'v' on this device, limited to ts..te.
                                    auto vtgtstripe = v.stripeforcudadevice (todeviceid, stripedwrtrows);   // target: lives on 'todeviceid', the row stripe that expected there after matprod_mm()
                                    unique_ptr<msra::cuda::matrix> vtgtstripesub (vtgtstripe->patch (0, vtgtstripe->rows(), ts, te));   // limit to sub-minibatch
                                    unique_ptr<msra::cuda::matrix> vtgtbufsub (vtgtbuf->patch (0, vtgtbuf->rows(), ts, te));        // source: this is thedata that came from 'deviceid' for this row stripe
                                    msra::cuda::onsubstream sub (*vtgtstripesub, substream);   // configure it to run gems() on this substream
                                    vtgtstripesub->gems (1.0f, *vtgtbufsub, 1.0f);
                                }
                            }
                        }
                    }
                });
                v.setoutputstriping (stripedwrtrows);       // done: we are now in the right format
            }
            else        // non-overlapped version (old code, also use for 1 GPU)
            {
                for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                    v.forcudadevice (deviceid).gemm (vscale, this->forcudadevice (deviceid, stripedwrtcols), false, *h.stripeforcudadevice (deviceid, stripedwrtrows), false, 1.0f);
                v.sumacrossdevices (stripedwrtrows/*target striping (the only choice)*/);
            }

#ifdef TIME_TM
            v.synchronize();
            h.synchronize();
            synchronize();
            totaltime += copycost;
            totalmacs += sc.numframes * this->cols() * this->rows();
            copycost.show ("copy complete");
            // weird: somehow this exlicit sync makes it faster (in case of doing a full batch anyway)
            fprintf (stderr, "copy time: %.2f GB/s (total time %.3f s for %.2f MB)\n", totalbytes / totalcopytime / 1e9, totalcopytime, totalbytes / 1e6);
            fprintf (stderr, "async time: %.2f G ma/s (total time %.3f s for %.2f G macs)\n", totalmacs / (totaltime - totalcopytime) / 1e9, totaltime - totalcopytime, totalmacs / 1e9);
            fprintf (stderr, "matprod time incl. copy: %.2f G ma/s (total time %.3f s for %.2f G macs)\n", totalmacs / totaltime / 1e9, totaltime, totalmacs / 1e9);
#endif
        }
        else
        {
            matprod_mm_numa (*this, h.fornuma(), cachedWts, cachedhs, v.fornuma(), vscale);
        }
    }

    // NUMA-localized matrix product v = W h + b, W = 'this' not transposed, b is added to all columns
    // used in htov() (RBM pretraining): W.matprod_mm_numa (h, cachedWts, cachedhs, v, b);     // v = W h + b
    void matprod_mm (const rbmstatevectorsref & h, cachedmatrix & cachedWts, cachedmatrix & cachedhs, rbmstatevectorsref & v, cachedmatrix & cachedvs/*unused*/,
                     const rbmmodelmatrixbase & b, cachedmatrix &/*cachedb1s*/,
                     acceleratedmatrixbase<msra::math::ssematrix<matrixbase>> & gpuexchangebuffers,
                     const float vscale=0.0f) const
    {
        // v = W h  (without bias, which comes next)
        matprod_mm (h, cachedWts, cachedhs, v, cachedvs, gpuexchangebuffers, vscale);
        // add bias
        if (cudamode)
        {
            v.checkvalidstriping (stripedwrtrows);
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                v.stripeforcudadevice (deviceid, stripedwrtrows)->addtoallcolumns (b.forcudadevice (deviceid, stripedwrtrows));
        }
        else
        {
            v.fornuma() += b;
        }
    }

    // inner product of this and a
    // result = sum_i,j this(i,j) * a(i,j)
    template<class atype>   // rbmmodelmatrixbase or vector
    float dot_mtm (const atype & a) const
    {   
        float result = 0.0f;
        checkcomputing();
        if (cudamode)
        {
            // distinguish between striping types based on dimensions
            // vectors are assumed to be row striped
            // matrices are assumed to be col striped
            cudastriping_t stripingtype = a.cols() == 1 ? stripedwrtrows : stripedwrtcols;
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                const msra::cuda::matrix & cudamatrix1 = forcudadevice(deviceid, stripingtype);
                const msra::cuda::matrix & cudamatrix2 = a.forcudadevice(deviceid, stripingtype);
                result += cudamatrix1.dot(cudamatrix2);
            }
        }
        else
        {
#if 1   
            throw std::logic_error ("dot_mtm: non-CUDA version not implemented for rbmstatevectorsref; fix this if you need it");
#else
            result = matrix::dotprod (amatrix);
#endif
        }
        return result;
    }

    // weighted inner product
    // result = sum_i,j this(i,j) * weightingmatrix(i,j) * a(i,j)
    float weighteddot_mtm (const rbmmodelmatrixbase & weightingmatrix, const rbmmodelmatrixbase & a) const
    {   
        float result = 0.0f;
        checkcomputing();
        if (cudamode)
        {
            // distinguish between striping types based on dimensions
            // vectors are assumed to be row striped
            // matrices are assumed to be col striped
            cudastriping_t stripingtype = a.cols() == 1 ? stripedwrtrows : stripedwrtcols;
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
            {
                const msra::cuda::matrix & thisstripe = forcudadevice(deviceid, stripingtype);
                const msra::cuda::matrix & astripe = a.forcudadevice(deviceid, stripingtype);
                const msra::cuda::matrix & weightingmatrixstripe = weightingmatrix.forcudadevice(deviceid, stripingtype);
                result += thisstripe.weighteddot(weightingmatrixstripe, astripe);
            }
        }
        else
            result = matrix::weighteddot(weightingmatrix, a);    
        return result;
    }

    // add this to double precision accumulator
    // use in gpu mode only
    void addtoaccumulator(float accumulatorscale, float updatescale, msra::cuda::matrixaccumulator &accumulator)
    {
        // TODO generalize to striped matrices
        if (cudamode)
        {
            assert(numcudadevices() == 1);
            cudastriping_t stripingtype = cols() == 1 ? stripedwrtrows : stripedwrtcols;
            accumulator.accumulate(accumulatorscale, this->forcudadevice(0, stripingtype), updatescale);
        }
        else
            throw std::logic_error("This method should not be called in NUMA mode!");
    }

    // add this to double precision accumulator
    // use in NUMA mode only
    void addtoaccumulator(float accumulatorscale, float updatescale, msra::math::doublematrix &accumulator)
    {
        // TODO generalize to striped matrices
        if (cudamode)
            throw std::logic_error("This method should not be called in Cuda mode!");
        else
            accumulator.addfloat(accumulatorscale, *this, updatescale);
            
    }

    // set values to that of double precision gpu matrix
    // use in cuda mode only
    void setfrommatrixaccumulator(const msra::cuda::matrixaccumulator &accumulator)
    {
        // TODO generalize to striped matrices
        if (cudamode)
        {
            cudastriping_t stripingtype = cols() == 1 ? stripedwrtrows : stripedwrtcols;
            accumulator.tomatrix(this->forcudadevice(0, stripingtype));
        }
        else
            throw std::logic_error("This method should not be called in NUMA mode!");
    }
    
    // set values to that of double precision cpu matrix
    // use in NUMA mode only
    void setfrommatrixaccumulator(const msra::math::doublematrix &accumulator)
    {
        // TODO generalize to striped matrices
        if (cudamode)
            throw std::logic_error("This method should not be called in Cuda mode!");
        else
            accumulator.tomatrix(*this);
    }

    // perform an SVD decomposition of this matrix
    // this' --> this * V'
    // where 'this' gets updated in-place, while V is returned.
    // Returns the bottleneck dimension after cut-off.
    // The updated 'this' already gets cut w.r.t. Eigenvalue cut-off, while V is not, it expected that the caller shall drop columns of V (rows of Vt) beyond 'dim'.
    // V is returned in transposed form, indicated by the name 'Vt'.
    size_t svd (std::vector<std::vector<float>> & Vt, float rank)
    {
        // this code is based on an SVD function that does not use our matrix lib
        // so much of what's done below is copying stuff around

        // M <- this' = W'      where this = W matrix of a network layer. Note the transposition here (unless 'needtotranspose', see below).

        size_t dimn = rows();       // n = rows is input dimension,  aka vdim
        size_t dimm = cols();       // m = cols is output dimension, aka hdim
        // Note: we decompose W' and not W, therefore m and n refer to W' (with transposition), that is m=rows of W' and n=cols of W'

        // special case: if W' reduces the dimension rather than growing it (atypical), we transpose it for SVD and later transpose back, because the inner SVD function requires m >= n
        const bool needtotranspose = dimm < dimn;   // layer output dim < input dim?
        if (needtotranspose)
            ::swap (dimm, dimn);

        std::vector<std::vector<float>> M (dimm);   // M[m][n]
        for (size_t i = 0; i < dimm; ++i)
        {
            M[i].resize (dimn);
            if (!needtotranspose)   // TODO: memcpy() for a single SVD decomp preparation?? premature optimization!!
                memcpy (&M[i][0], &(col(i)[0]), dimn * sizeof (float));       // M <- W'
            else
                for (size_t j = 0; j < dimn; ++j)
                    M[i][j] = col(j)[i];      // M <- W; note: col(j)[i] = (*this)(i,j)
        }

        // now decompose M = U V'

        std::vector<float> w (dimn);    // result buffer for Eigenvalues
        Vt.resize (dimn);               // result buffer for the second factor matrix (square at this point, before we reduce the Eigenvalues)
        foreach_index (i, Vt) Vt[i].resize (dimn);

        // perform SVD: M = U w V'  with diagonal matrix w (stored as a vector)
        //  - U: (m x n)
        //  - w: (n x n)  diagonal, stored as vector
        //  - V: (n x n)
        auto success = do_svd (M, (int) dimm, (int) dimn, w, Vt);
        if (!success)
            throw std::runtime_error ("svd: do_svd() failed");
        auto & U = M;           // U is returned in place of M

        // now U, w, and Vt have been filled in such that W' = this' = U w V'

        // determine the dimension cut-off
        // 'rank' can have two meanings:
        //  - < 1: use as cut-off against accumulative Eigenvalues
        //  - > 1: use as dimension directly
        //  - (0: this layer should have been skipped in the first place)
        size_t dim;
        if (rank > 0 && rank < 1.0)
        {
            float sum = 0.0, accum = 0.0;
            size_t i;
            for (i = 0; i < dimn; ++i) sum += w[i];
            for (i = 0; (i < dimn) && (accum < sum*rank); ++i) accum += w[i];

            dim = i;
        }
        else if (rank < dimn && rank > 0)
            dim = (size_t)rank;
        else
            dim = dimn;         // does this code path make sense? It's no reduction. 

        if (dim % 8 != 0)       // round up to multiple of 8, to ensure good alignment
            dim = (dim/8+1)*8;

        // eliminate diagonal matrix (w) by distributing it equally (sqrt) over both factor matrices
        for (size_t j = 0; j < dimn; ++j)
        {
            float sqrtw, tempv;
            if (w[j] > 0)
            {
                sqrtw = sqrt (w[j]);
                tempv = sqrtw;
            }
            else        // negative Eigenvalue: keep sign on one of the two
            {
                sqrtw = sqrt (-w[j]);
                tempv = -sqrtw;
            }

            for (size_t i = 0; i < dimm; ++i)
                U[i][j] *= sqrtw;

            for (size_t i = 0; i < dimn; ++i)
                Vt[i][j] *= tempv;
        }

        // replace "this" by U' (i.e. U' is returned in-place, while V is returned separately)
        if (!needtotranspose)                               // (W not reducing dimension, that is, the typical case)
        {
            resize (dim/*less*/, dimm/*same as before*/);   // dimensionality reduction for 'this' happens here
            for(size_t i=0; i<dimm; ++i)                    // this <- U'
                memcpy (&(col(i)[0]), &U[i][0], dim * sizeof (float));
        }
        else    // if transposed, we must swap both matrices as well
        {
            resize (dim, dimn);
            for(size_t i=0; i<dimn; ++i)
                memcpy (&(col(i)[0]), &Vt[i][0], dim * sizeof (float));

            Vt.resize (dimm);
            for(size_t i=0; i<dimm; ++i)
            {
                Vt[i].resize (dimm);
                memcpy (&Vt[i][0], &U[i][0], dim * sizeof (float));
            }
        }

        // return dimension to caller since caller must still reduce V to the first 'dim' columns
        return dim;
    }

    // temp for maxouts, --TODO: generalize by dot product : this = this*diag(x) using, for example, cublasSdgmm
    void colwisenrm2(rbmmodelmatrixbase &norms, const float maxcolnorm)
    {   
        checkcomputing();
        if (cudamode) 
        {
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++) 
                this->forcudadevice(deviceid).colwisenrm2(norms.forcudadevice(deviceid), maxcolnorm);
                //norms.forcudadevice(deviceid).setvalue(0.0f);
        }
        else
        {
            throw std::logic_error("colwisenrm2: cpu ver not yet implemented, sorry");
        }
    }

    void scalecolwise (const rbmmodelmatrixbase & factors)
    {
        checkcomputing();
        if (cudamode) 
        {
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice(deviceid).scalecolwise (factors.forcudadevice(deviceid));
        }
        else
        {
            throw std::logic_error("scalecolwise: not yet implemented, sorry");
        }
    }

    void scalerowwise (const rbmmodelmatrixbase & factors)
    {
        checkcomputing();
        if (cudamode) 
        {
            if (numcudadevices() != 1)
                throw std::logic_error ("scalerowwise: not yet implemented for model parallelism, sorry");
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice(deviceid).scalerowwise (factors.forcudadevice(deviceid));
        }
        else
        {
            throw std::logic_error("scalerowwise: not yet implemented, sorry");
        }
    }

    void unscalerowwise (const rbmmodelmatrixbase & factors)
    {
        checkcomputing();
        if (cudamode) 
        {
            if (numcudadevices() != 1)
                throw std::logic_error ("unscalerowwise: not yet implemented for model parallelism, sorry");
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->forcudadevice(deviceid).unscalerowwise (factors.forcudadevice(deviceid));
        }
        else
        {
            throw std::logic_error("unscalerowwise: not yet implemented, sorry");
        }
    }

private:
    typedef msra::math::ssematrixstriperef<matrixbase> matrixstripe;

    // NUMA-localized matrix product C = C * scale + A B
    // where A is passed as A' (i.e. we compute At' B) if 'Aistransposed' (which is faster, so do it if you can)
    // Uses class-local NUMA-local memory that is kept allocated for efficiency.
    static void scaleandaddmatprod_numa (float thisscale, const matrixbase & A, bool Aistransposed, const matrixbase & B,
                                         cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C, const float otherweight = 1.0f)
    {
        const size_t Atrows = Aistransposed ? A.cols() : A.rows();
        const size_t Atcols = Aistransposed ? A.rows() : A.cols();
#if 0
        fprintf (stderr, "scaleandaddmatprod_numa [%.2f]: (%4d x %4d) * (%4d x %4d) -> (%4d x %4d) // %.3f MFlops, %.3f total MB\n",
            thisscale,
            Atrows, Atcols, B.rows(), B.cols(), Atrows, B.cols(),
            1e-6 * Atrows * Atcols * B.cols(), (Atrows * Atcols + B.rows() * B.cols()) / 1024.0 / 1024.0);
#endif
        // we do NUMA-local copies if necessary (i.e. if we are running with >1 NUMA node)
        const bool donuma = (msra::numa::getnumnodes() > 1) && (msra::parallel::get_cores() > 1);
#if 1   // print a message  --remove this once this seems to be working
        {
            static bool f = false;
            if (!f)
            {
                fprintf (stderr, "scaleandaddmatprod_numa: donuma = %s\n", donuma ? "true" : "false");
                f = true;
            }
        }
#endif
        // ensure memory is allocated as required
        if (donuma)
            cachedBs.allocate_numa (B.rows(), B.cols());
        if (donuma || !Aistransposed)
            cachedAts.allocate_numa (Atcols, Atrows);   // cachedAts also used if A is not transposed yet
        // copy B into all cachedBs[]
        if (donuma)
        {
            msra::numa::parallel_for_on_each_numa_node (true, [&] (size_t numanode, size_t i, size_t n)
            {
                matrixbase & cachedB = cachedBs[numanode];
                cachedB.assign (B, i, n);
            });
        }
        // perform product--row stripes of A will be copied locally if >1 NUMA node
        msra::parallel::foreach_index_block (Atrows, Atrows, 4, [&] (size_t i0, size_t i1)
        {
            const size_t numanode = msra::numa::getcurrentnode();
            // get the cached B
            const matrixbase & cachedB = donuma ? cachedBs[numanode] : B;
            //cachedB.checkequal (B);
            // copy over row stripe of A that belongs to this loop iteration
            matrixbase & cachedAt = (donuma || !Aistransposed) ? cachedAts[numanode] : const_cast<matrixbase &> (A);
            if (!Aistransposed) // copy and transpose a row stripe
            {
                // transpose a row stripe of A into a col stripe of At
                A.transposerows (cachedAt, i0, i1);
            }
            else if (donuma)    // copy a row stripe --it's in column form, so copy the range of columns
            {
                // rows [i0,i1) are columns [i0,i1) of A', so we can use a matrixstripe
                const matrixstripe src (const_cast<matrixbase &> (A), i0, i1 - i0);    // (it is 'const')
                matrixstripe dst (cachedAt, i0, i1 - i0);
                // TODO: This point is hit with 40% prob when randomly breaking into the debugger.
                //  --> need to avoid the copy if not NUMA
                dst.assign (src);  // only copy the bits that are needed
            }
            // perform operation from NUMA-local copies
#if 0
            if (i0 == 0)
                if (Aistransposed)
                    fprintf (stderr, "scaleandaddmatprod_numa: slice' (%4d x %4d) x (%4d x %4d)\n", i1, cachedAt.rows(), cachedB.rows(), cachedB.cols());
                else
                    fprintf (stderr, "scaleandaddmatprod_numa: slice  (%4d x %4d) x (%4d x %4d)\n", i1, cachedAt.cols(), cachedB.rows(), cachedB.cols());
#endif
            C.scaleandaddmatprod_mtm (thisscale, cachedAt, i0, i1, cachedB, otherweight);
        });
#if 0   // check that we copied A right   --OUTDATED for Aistransposed == false
        for (size_t k = 1; k < numnodes; k++)
            foreach_coord (i, j, cachedAts[0])
            cachedAts[0](i,j) += cachedAts[k](i,j);
        A.checkequal (cachedAts[0]);
#endif
    }

    // NUMA-localized matrix product C = C * weight + A B, A is transposed
    // see scaleandaddmatprod_numa
    static void matprod_mtm_numa (const matrixbase & At, const matrixbase & B, cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C, float weight = 0.0f)
    {
        scaleandaddmatprod_numa (weight, At, true, B, cachedAts, cachedBs, C);
#if 0   // check that distributed mat prod is correct
        assert (At.rows() == B.rows());
        for (size_t i = 0; i < At.cols(); i++) for (size_t j = 0; j < B.cols(); j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < At.rows(); k++)
                sum += At(k,i) * B(k,j);
            if (!equalenough (sum, C(i,j)))
            {
                fprintf (stderr, "matprod_mtm_numa: mismatch %.10f vs. %.10f\n", sum, C(i,j));
                return;
            }
        }
#endif
    }

    // NUMA-localized matrix product C = A B, A is not transposed
    // see scaleandaddmatprod_numa
    static void matprod_mm_numa (const matrixbase & A, const matrixbase & B, cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C, const float vscale = 0.0f)
    {
        scaleandaddmatprod_numa (vscale, A, false, B, cachedAts, cachedBs, C);
#if 0   // check that distributed mat prod is correct
        assert (A.cols() == B.rows());
        for (size_t i = 0; i < A.rows(); i++) for (size_t j = 0; j < B.cols(); j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols(); k++)
                sum += A(i,k) * B(k,j);
            if (!equalenough (sum, C(i,j)))
            {
                fprintf (stderr, "matprod_mm_numa: mismatch %.10f vs. %.10f\n", sum, C(i,j));
                return;
            }
        }
#endif
    }

    public:
    static void test()
    {
        printf("###########################\nTESTING RBMMODELMATRIX\n");
        try
        {
            testdot();
            testweighteddot();
            testaddweighted();
            testelementwisedivision();
            testscale();
        }
        catch (const exception & e)
        {
            std::cerr << "parallelrbmmatrix: unit test failed! " << e.what() << std::endl;
        }
    }
    
    static void testdot()
    {
        printf("DOT ...\n");
        for (int testit = 0; testit < 14; testit++)
        {
            bool cudamode = true;
            size_t dim1 = 0, dim2 = 1;
            switch(testit)
            {
                case 0: dim1 = 1; break;
                case 1: dim1 = 37; dim2 = 35; break;
                case 2: dim1 = 5000; break;
                case 3: dim1 = 32; break;
                case 4: dim1 = 31; break;
                case 5: dim1 = 31; dim2 = 31; break;
                case 6: dim1 = 32; dim2 = 32; break;
                case 7: dim1 = 1; cudamode = false; break;
                case 8: dim1 = 37; dim2 = 35; cudamode = false; break;
                case 9: dim1 = 5000; cudamode = false; break;
                case 10: dim1 = 32; cudamode = false; break;
                case 11: dim1 = 31; cudamode = false; break;
                case 12: dim1 = 31; dim2 = 31; cudamode = false; break;
                case 13: dim1 = 32; dim2 = 32; cudamode = false; break;
            }
            rbmmodelmatrix a;
            rbmmodelmatrix b;
            a.setcudamode(cudamode);
            b.setcudamode(cudamode);
            a.resize(dim1, dim2);
            b.resize(dim1, dim2);
            float checkval = 0.0f;
            for (size_t i = 0; i < dim1; i++)
            {
                for (size_t j = 0; j < dim2; j++)
                {
                    float val1 = (float) ((i+3)*(j+2) % 5);
                    float val2 = (float) ((i+1)*(j+3) % 6 - 3);
                    a(i,j) = val1;
                    b(i,j) = val2;
                    checkval += val1 * val2;
                }
            }
            a.entercomputation();
            b.entercomputation();
        
            float r = a.dot_mtm(b);
            printf("dot product %d (rbmmodelmatrix):\t%f\n", testit, r);
            printf("dot product %d (testvalue)     :\t%f\n", testit, checkval);
            if (r != checkval)
                throw std::logic_error("unit test failed!");
        }

        printf("... PASSED \n");
        printf("###########################\n");        
    }

    static void testweighteddot()
    {
        printf("WEIGHTED DOT ...\n");
        for (int testit = 0; testit < 14; testit++)
        {
            bool cudamode = true;
            size_t dim1 = 0, dim2 = 1;
            switch(testit)
            {
                case 0: dim1 = 1; break;
                case 1: dim1 = 37; dim2 = 35; break;
                case 2: dim1 = 5000; break;
                case 3: dim1 = 32; break;
                case 4: dim1 = 31; break;
                case 5: dim1 = 31; dim2 = 31; break;
                case 6: dim1 = 32; dim2 = 32; break;
                case 7: dim1 = 1; cudamode = false; break;
                case 8: dim1 = 37; dim2 = 35; cudamode = false; break;
                case 9: dim1 = 5000; cudamode = false; break;
                case 10: dim1 = 32; cudamode = false; break;
                case 11: dim1 = 31; cudamode = false; break;
                case 12: dim1 = 31; dim2 = 31; cudamode = false; break;
                case 13: dim1 = 32; dim2 = 32; cudamode = false; break;
            }
            rbmmodelmatrix a;
            rbmmodelmatrix b;
            rbmmodelmatrix M;
            rbmmodelmatrix Id;
            a.setcudamode(cudamode);
            b.setcudamode(cudamode);
            M.setcudamode(cudamode);
            Id.setcudamode(cudamode);
            a.resize(dim1, dim2);
            b.resize(dim1, dim2);
            M.resize(dim1, dim2);
            Id.resize(dim1, dim2);
            float checkval = 0.0f;
            float checkvalid = 0.0f;
            for (size_t i = 0; i < dim1; i++)
            {
                for (size_t j = 0; j < dim2;j++)
                {
                    float val1 = (float) ((i+3)*(j+2) % 5 - 1);
                    float val2 = (float) ((i+1)*(j+3) % 3);
                    float val3 = (float) ((i+1)*(j+3) % 6 - 3);
                    a(i,j) = val1;
                    b(i,j) = val2;
                    M(i,j) = val3;
                    Id(i,j) = 1.0f;
                    checkval += (float) val1 * val2 * val3;
                    checkvalid += (float) val1 * val2;
                }
            }
            a.entercomputation();
            b.entercomputation();
            M.entercomputation();
            Id.entercomputation();
            float r1 = a.weighteddot_mtm(M, b);
            float r2 = a.weighteddot_mtm(Id, b);
            float r3 = a.dot_mtm(b);
            printf("weighted dot product %d (rbmmodelmatrix):\t%f\n", testit, r1);
            printf("weighted dot product %d (testvalue)     :\t%f\n", testit, checkval);
            if (r1 != checkval)
                throw std::logic_error("unit test failed!");
            
            printf("weighted dot product %d (rbmmodelmatrix W=Id):\t%f\n", testit, r2);
            printf("weighted dot product %d (testvalue W=Id)     :\t%f\n", testit, checkvalid);
            if (r2 != checkvalid)
                throw std::logic_error("unit test failed!");
            printf("weighted dot product %d (rbmmodelmatrix W=Id                    ):\t%f\n", testit, r2);
            printf("weighted dot product %d (rbmmodelmatrix W=Id, dotprod_mtm method):\t%f\n", testit, r3);
            if (r2 != r3)
                throw std::logic_error("unit test failed!");
        }
        printf("... PASSED \n");
        printf("###########################\n");        
    }

    static void testscale()
    {
        printf("SCALE SQUARE ...\n");
        for (int testit = 0; testit < 14; testit++)
        {
            bool cudamode = true;
            size_t dim1 = 1, dim2 = 1;
            float scalingfactor = 0.1f;
            switch(testit)
            {
                case 0: dim1 = 1; break;
                case 1: dim1 = 37; dim2 = 35; break;
                case 2: dim1 = 5000; break;
                case 3: dim1 = 32; break;
                case 4: dim1 = 31; break;
                case 5: dim1 = 31; dim2 = 31; break;
                case 6: dim1 = 32; dim2 = 32; break;
                case 7: dim1 = 1; cudamode = false; break;
                case 8: dim1 = 37; dim2 = 35; cudamode = false; break;
                case 9: dim1 = 5000; cudamode = false; break;
                case 10: dim1 = 32; cudamode = false; break;
                case 11: dim1 = 31; cudamode = false; break;
                case 12: dim1 = 31; dim2 = 31; cudamode = false; break;
                case 13: dim1 = 32; dim2 = 32; cudamode = false; break;
            }
            rbmmodelmatrix a;
            rbmmodelmatrix c;
            a.setcudamode(cudamode);
            a.resize(dim1, dim2);
            c.resize(dim1, dim2);
            float checkval = 0.0f;
            for (size_t i = 0; i < dim1; i++)
            {
                for (size_t j = 0; j < dim2; j++)
                {
                    float val = (float) (i+1)*(j+1);
                    a(i,j) = val;
                    c(i,j) = val * scalingfactor;
                }
            }
            a.entercomputation();
            a.scale(scalingfactor);
            a.exitcomputation();
            printf("scale(0,0) %d (rbmmodelmatrix):\t%f\n", testit, a(0,0));
            printf("scale(0,0) %d (testvalue)     :\t%f\n", testit, c(0,0));
            for (size_t i = 0; i < dim1; i++)
                for (size_t j = 0; j < dim2; j++)
                    if (c(i,j) != a(i,j))
                        throw std::logic_error("unit test failed");
        }
        printf("... PASSED \n");
        printf("###########################\n");        
    }

    static void testaddweighted()
    {
        printf("ADDWEIGHTED ...\n");
        for (int testit = 0; testit < 4; testit++)
        {
            size_t dim1 = 0, dim2 = 0;
            float thisscale = 1.0f;
            float otherscale = 0.5f;
            rbmmodelmatrix a;
            rbmmodelmatrix b;
            rbmmodelmatrix c;
            switch(testit)
            {
                case 0: dim1 = 1, dim2 = 1; break;
                case 1: dim1 = 37, dim2 = 1; break;
                case 2: dim1 = 37, dim2 = 45; break;
                case 3: dim1 = 37, dim2 = 45; thisscale = 0.0f; break;
            }
            a.resize(dim1,dim2);
            b.resize(dim1,dim2);
            c.resize(dim1,dim2);
            for (size_t i = 0; i < dim1;i++)
            {
                for (size_t j = 0; j < dim2; j++)
                {
                    float val1 = (float) (i+1)*(j+2);
                    float val2 = (float) (i-5)*(j+4);
                    a(i,j) = val1;
                    b(i,j) = val2;
                    c(i,j) = thisscale * val1 + otherscale * val2;
                }
            }
            a.entercomputation();
            b.entercomputation();
        
            a.addweighted(thisscale, b, otherscale);
            a.exitcomputation();
            printf("addweighted(%d,%d) %d (rbmmodelmatrix):\t%f\n", testit, 0,0, a(0,0));
            printf("addweighted(%d,%d) %d (testvalue)     :\t%f\n", testit, 0,0, c(0,0));
            for (size_t i = 0; i < dim1; i++)
                for (size_t j = 0; j < dim2; j++)
                    if (c(i,j) != a(i,j))
                        throw std::logic_error("unit test failed");
        }
        printf("... PASSED \n");
        printf("###########################\n");        
    }

    static void testelementwisedivision()
    {
        printf("ELEMENTWISEDIVISION ...\n");
        for (int testit = 0; testit < 6; testit++)
        {
            bool cudamode = true;
            size_t dim1 = 0, dim2 = 0;
            rbmmodelmatrix a;
            rbmmodelmatrix b;
            rbmmodelmatrix c;
            rbmmodelmatrix d;
            switch(testit)
            {
                case 0: dim1 = 1, dim2 = 1; break;
                case 1: dim1 = 37, dim2 = 1; break;
                case 2: dim1 = 37, dim2 = 45; break;
                case 3: dim1 = 1, dim2 = 1; cudamode = false; break;
                case 4: dim1 = 37, dim2 = 1; cudamode = false; break;
                case 5: dim1 = 37, dim2 = 45; cudamode = false; break;

            }
            a.resize(dim1,dim2);
            b.resize(dim1,dim2);
            c.resize(dim1,dim2);
            d.resize(dim1,dim2);
            for (size_t i = 0; i < dim1;i++)
            {
                for (size_t j = 0; j < dim2; j++)
                {
                    float val1 = (float) 2*(i+1)*(j+2);
                    float val2 = (float) ((i+1)*(j+2) % 2);
                    a(i,j) = val1;
                    b(i,j) = val2;
                    c(i,j) = a(i,j) / b(i,j);
                }
            }
            a.entercomputation();
            b.entercomputation();
            d.entercomputation();

            d.elementwisedivision(a, b);
            d.exitcomputation();
            printf("addweighted(%d,%d) %d (rbmmodelmatrix):\t%f\n", testit, 0,0, d(0,0));
            printf("addweighted(%d,%d) %d (testvalue)     :\t%f\n", testit, 0,0, c(0,0));
            for (size_t i = 0; i < dim1; i++)
                for (size_t j = 0; j < dim2; j++)
                    if (c(i,j) != d(i,j))
                        throw std::logic_error("unit test failed");
        }
        printf("... PASSED \n");
        printf("###########################\n");        
    }

public:
#if 0   // for debugging
    static bool equalenough (float a, float b)
    {
        float adiff = fabs (a - b);
        float absa = fabs (a);
        if (absa > 1e-8)
        {
            float relerr = adiff / absa;
            return adiff < 1e-4 || relerr < 1e-3;
        }
        else
            return adiff < 1e-5;    // 0 --can't divide
    }
#endif
};

// double precision accumulator, handling NUMA and Cuda mode
class rbmmatrixaccumulator 
{

protected:
    const bool cudamode;
    size_t nrows;
    size_t ncols;
    msra::math::doublematrix doublecpumatrix;
    std::unique_ptr<msra::cuda::matrixaccumulator> doublecudamatrix;
public:
    rbmmatrixaccumulator() :
      cudamode(hascuda()),
      nrows(0),
      ncols(0)
      { }

    virtual ~rbmmatrixaccumulator() { }
    
    virtual void allocate(size_t n, size_t m)
    {
        nrows = n;
        ncols = m;
        if (cudamode)
        {
            doublecudamatrix.reset(msra::cuda::newmatrixaccumulator());
            doublecudamatrix->allocate(n,m);
            doublecudamatrix->reset();
        }
        else
        {
            doublecpumatrix.allocate(n,m);
            doublecpumatrix.reset();
        }
    }

    virtual void reset()
    {
        if (cudamode)
            doublecudamatrix->reset();
        else
            doublecpumatrix.reset();
    }

    template<class matrixbase> void accumulate(float thisscale, rbmmodelmatrixbase<typename matrixbase> &other, float otherweight)
    {
        if (cudamode)
            other.addtoaccumulator(thisscale, otherweight, *doublecudamatrix);
        else
            other.addtoaccumulator(thisscale, otherweight, doublecpumatrix);
    }

    template<class matrixbase> void tomatrix(rbmmodelmatrixbase<matrixbase>& other) const
    {
        if (cudamode)
            other.setfrommatrixaccumulator(*doublecudamatrix);
        else
            other.setfrommatrixaccumulator(doublecpumatrix);
    }

    size_t rows() { return nrows; }
    size_t cols() { return ncols; }
};

};};

static double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return result;
}

// SVD decomposition
// perform SVD: M = U * w * V'
//  - dimensions:
//     - M: (rows x cols)   with M = W' and rows >= cols
//     - U: (rows x cols)
//     - w: (cols x cols)   diagonal, stored as vector
//     - Vt: (cols x cols)
//  - U is returned in M
//  - w is diagonal (stored as a vector)
//  - V is returned in Vt in transposed form (as the name indicates)
// TODO: why not use the matrix type?? This really does not belong into this source code. Once it uses matrix types, we can put this into ssematrix.h.
// Complexity [http://en.wikipedia.org/wiki/Singular_value_decomposition]:
//  - two-step procedure. In the first step, the matrix is reduced to a bidiagonal matrix. This takes O(mn^2) floating-point operations (flops), assuming that m >= n.
//  - The second step is to compute the SVD of the bidiagonal matrix. This step can only be done with an iterative method (as with eigenvalue algorithms).
//    However, in practice it suffices to compute the SVD up to a certain precision, like the machine epsilon. If this precision is considered constant,
//    then the second step takes O(n) iterations, each costing O(n) flops.
//  - Thus, the first step is more expensive, and the overall cost is O(mn^2) flops (Trefethen & Bau III 1997, Lecture 31).
//  --> O(rows * cols^2) where for output layer, #rows of W' = #senones and #cols of W' = #hidden nodes; so proportional to #senones (given fixed hidden dimension)
//      E.g. udim = 32k, typical hdim = 3k => top layer is 10 x as expensive as hidden layers.
static int do_svd (std::vector<std::vector<float>> &M, int rows, int cols, std::vector<float> &w, std::vector<std::vector<float>> &Vt)
{
    int flag;
    const int iterations = 50;
    double f, h, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
  
    if (rows < cols)    // 'rows' is also the number of Eigenvalues of M; ill-defined if rows > cols
    {
        fprintf (stderr, "do_svd: number of rows must be > number of cols\n");
        return 0;
    }

    std::vector<double> sinvalue;
    sinvalue.resize(cols);

    fprintf (stderr, "do_svd: left and right hand reduction of %d columns\n", cols);
    for (int i = 0; i < cols; ++i) 
    {
        /* left-hand reduction */
        int index = i + 1;
        sinvalue[i] = scale * g;
        g = scale = 0.0;
        double s = 0.0;
        if (i < rows) 
        {
            for (int j = i; j < rows; ++j) 
                scale += fabs((double)M[j][i]);

            if (scale) 
            {
                for (int j = i; j < rows; ++j) 
                {
                    M[j][i] = (float)((double)M[j][i]/scale);
                    s += ((double)M[j][i] * (double)M[j][i]);
                }

                f = (double)M[i][i];
				if(f>=0) g = -sqrt(s);
				else g = sqrt(s);

                h = f * g - s;
                M[i][i] = (float)(f - g);
                if (i != cols - 1) 
                {
                    for (int j = index; j < cols; ++j) 
                    {
                        s = 0.0;
                        for (int k = i; k < rows; ++k) s += ((double)M[k][i] * (double)M[k][j]);

                        f = s / h;
                        for (int k = i; k < rows; ++k) M[k][j] += (float)(f * (double)M[k][i]);
                    }
                }
                for (int j = i; j < rows; ++j) 
                    M[j][i] = (float)((double)M[j][i]*scale);
            }
        }
        w[i] = (float)(scale * g);
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < rows && i != cols - 1) 
        {
            for (int j = index; j < cols; j++) scale += fabs((double)M[i][j]);

            if (scale) 
            {
                for (int j = index; j < cols; ++j) 
                {
                    M[i][j] = (float)((double)M[i][j]/scale);
                    s += ((double)M[i][j] * (double)M[i][j]);
                }

                f = (double)M[i][index];
                if (f >= 0) g = -sqrt(s);
                else        g = sqrt(s);

                h = f * g - s;
                M[i][index] = (float)(f - g);

                for (int j = index; j < cols; ++j) 
                    sinvalue[j] = (double)M[i][j] / h;

                if (i != rows - 1) 
                {
                    for (int j = index; j < rows; ++j) 
                    {
						s = 0.0;
                        for (int k = index; k < cols; ++k) 
                            s += ((double)M[j][k] * (double)M[i][k]);

                        for (int k = index; k < cols; ++k) 
                            M[j][k] += (float)(s * sinvalue[k]);
                    }
                }

                for (int j = index; j < cols; ++j) 
                    M[i][j] = (float)((double)M[i][j]*scale);
            }
        }
        anorm = max(anorm, (fabs((double)w[i]) + fabs(sinvalue[i])));
    }
  
    /* accumulate the right-hand transformation */
    fprintf (stderr, "do_svd: accumulating the right-hand transformation\n");
    for (int i = cols - 1; i >= 0; --i) 
    {
        if (i < cols - 1) 
        {
            if (g) 
            {
                for (int j = i+1; j < cols; ++j)
                    Vt[j][i] = (float)(((double)M[i][j] / (double)M[i][i+1]) / g);

                /* double division to avoid underflow */
                for (int j = i+1; j < cols; ++j) 
                {
                    double s = 0.0;
                    for (int k = i+1; k < cols; ++k) s += ((double)M[i][k] * (double)Vt[k][j]);

                    for (int k = i+1; k < cols; ++k) Vt[k][j] += (float)(s * (double)Vt[k][i]);
                }
            }
            for (int j = i+1; j < cols; ++j) Vt[i][j] = Vt[j][i] = 0.0;
        }
        Vt[i][i] = 1.0;
        g = sinvalue[i];
    }
  
    /* accumulate the left-hand transformation */
    fprintf (stderr, "do_svd: accumulating the left-hand transformation\n");
    for (int i = cols - 1; i >= 0; --i) 
    {
        int index = i + 1;
        g = (double)w[i];
        if (i < cols - 1) 
            for (int j = index; j < cols; ++j) 
                M[i][j] = 0.0;
        if (g) 
        {
            g = 1.0 / g;
            if (i != cols - 1) 
            {
                for (int j = index; j < cols; ++j) 
                {
					double s = 0.0;
                    for (int k = index; k < rows; ++k) 
                        s += ((double)M[k][i] * (double)M[k][j]);

                    f = (s / (double)M[i][i]) * g;
                    for (int k = i; k < rows; ++k) 
                        M[k][j] += (float)(f * (double)M[k][i]);
                }
            }
            for (int j = i; j < rows; ++j) 
                M[j][i] = (float)((double)M[j][i]*g);
        }
        else 
        {
            for (int j = i; j < rows; ++j) 
                M[j][i] = 0.0;
        }
        ++M[i][i];      // TODO: M[][] is not an integer; is this += 1.0? Then better write that
    }

    /* diagonalize the bidiagonal form */
    fprintf (stderr, "do_svd: diagonalizing the bidiagonal form\n");
    for (int k = cols - 1; k >= 0; --k) 
    {
        for (int iter = 0; iter < iterations; ++iter) 
        {                         
            flag = 1;

            int index = k, pos=0;

            for (; index >= 0; index--) 
            {        
                pos = index - 1;
                if (fabs(sinvalue[index]) + anorm == anorm) 
                {
                    flag = 0;
                    break;
                }
                if (fabs((double)w[pos]) + anorm == anorm) 
                    break;
            }
            if (flag) 
            {
                double c = 0.0, s = 1.0;
                for (int i = index; i <= k; i++) 
                {
                    f = s * sinvalue[i];
                    if (fabs(f) + anorm != anorm) 
                    {
                        g = (double)w[i];
                        h = PYTHAG(f, g);
                        w[i] = (float)h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (int j = 0; j < rows; j++) 
                        {
                            y = (double)M[j][pos];
                            z = (double)M[j][i];
                            M[j][pos] = (float)(y * c + z * s);
                            M[j][i] = (float)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w[k];
            if (index == k) 
            {     
                if (z < 0.0) 
                { 
                    w[k] = (float)(-z);
                    for (int j = 0; j < cols; j++) Vt[j][k] = (-Vt[j][k]);
                }
                break;
            }
            if (iter >= iterations)
            {
                fprintf (stderr, "do_svd: didn't converge\n");
                return 0;
            }
    
            x = (double)w[index];
            pos = k - 1;
            y = (double)w[pos];
            g = sinvalue[pos];
            h = sinvalue[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            double temp = g;
            if(f<0) temp = -temp;
            f = ((x - z) * (x + z) + h * ((y / (f + temp)) - h)) / x;
          
            /* next QR transformation */
            double c = 1.0, s = 1.0;
            for (int j = index; j <= pos; j++) 
            {
                int ii = j + 1;
                g = sinvalue[ii];
                y = (double)w[ii];
                h = s * g;
                g = c * g;
                z = PYTHAG(f, h);
                sinvalue[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (int jj = 0; jj < cols; jj++) 
                {
                    x = (double)Vt[jj][j];
                    z = (double)Vt[jj][ii];
                    Vt[jj][j] = (float)(x * c + z * s);
                    Vt[jj][ii] = (float)(z * c - x * s);
                }
                z = PYTHAG(f, h);
                w[j] = (float)z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (int jj = 0; jj < rows; jj++) 
                {
                    y = (double)M[jj][j];
                    z = (double)M[jj][ii];
                    M[jj][j] = (float)(y * c + z * s);
                    M[jj][ii] = (float)(z * c - y * s);
                }
            }
            sinvalue[index] = 0.0;
            sinvalue[k] = f;
            w[k] = (float)x;
        }
    }

    return 1;
}
