#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "utterancesource.h"        // minibatch sources
#include "readaheadsource.h"

#pragma once

// This file currently contains only the configuration mechanism for hessianfreetraining

namespace msra { namespace dbn {

    // configuration info for hessianfree training
    struct hfinfo 
    {
        // hessian vector product settings
        size_t secondordermbsize;               // mbsize for calculating Hessian vector product
        size_t nsecondorderminibatches;         // number of mbatches for calculating Hessian vector product
        // basic conjugate gradient settings
        size_t mincgiters;                      // minimal number of CG iterations (for Martens termination criterion)
        size_t maxcgiters;                      // maximal number of CG iterations
        float cgtolerance;
        bool superlinearcgtolerance;            // use additional test that guarantees superlinear convergence (guarantee holds only for true Hessian)
        bool quadraticcgtolerance;              // use additional test that guarantees quadratic convergence (guarantee holds only for true Hessian)
        float decreasingcgtolerancefactor;
        // damping settings
        bool adaptlambda;                       // adapt damping factor as in Martens
        float initlambda;                       // initial value for damping factor
        float maxlambda;                        // maximal damping factor (prevents too small steps)
        bool usefinaliterateforadaptinglambda;  // in Martens algorithm, the best iterate is used, using the best iterate works indeed better, but then we need the backtracking
        // backtracking & linesearch
        bool docgbacktracking;                  // do backtracking over the CG iterates as in Martens paper
        bool stopbacktrackingatfirstimprovement;// stop backtracking as soon as we found an improvement over the last sweep
        float backtrackingbase;                 // models are stored on each iter with iteration = ceil(backtrackingbase^j)
        size_t maxlinesearches;                 // maximal number of line searches (set to zero for switching off line searches)
        bool optoncv;                           // calculate all likelihoods on cross validation set (reqiuired for damping, backtracking, line search)
        float llfraction;                       // use only a fraction of the data for calculating likelihoods (does not work if batchmode & !optoncv)
        // preconditioning
        bool cgpreconditioning;                 // use preconditioned CG, preconditioner is of the form (sum_t gradient_t^2 + lambda)^alpha
        float preconditioningalpha;             
        float preconditioninglambda;
        // cg initialization
        float cginitdecayingfactor;             // in Martens paper: 0.95, set to zero if you want CG to start from zero always
        bool cginitfinaliterate;                // initialize next CG with final iterate from last sweep (as in Martens), if false, the best iterate is used
        bool cginitifnegativeonly;              // only init CG with final iterate, if this gives a better starting point in terms of the CG objective function
        // cg trust region options
        bool usetrustregion;                    // experimental: use trust region algorithm
        float inittrustregionradius;            // experimental: initial trust region, set to -1 for choosing this automatically, based on the model norm
        bool adaptlambdabasedontrustregion;     // experimental: adapt lambda based on trust region
        bool exitcgwhenleavingtr;               // experimental: exit CG when leaving trust region
        size_t maxcgruns;                       // experimental: maximal number of CG runs (with different lambda), needs to be 1 for Martens algorithm
        // stochastic mode
        bool stochasticmode;                    // use stochastic algorithm, i.e. optimization on large minibatches
        size_t largeminibatchsize;              // size of large minibatches, must be a multiple of the minibatchsize
        // else
        int verbosity;                          // possible values: 0,1,2
        bool usedoubleaccumulator;              // use double precision accumulator for gradient and squared gradient

        hfinfo() :
            secondordermbsize(2048),
            nsecondorderminibatches(3),
            mincgiters(1),
            maxcgiters(50),
            cgtolerance(0.0005f),
            superlinearcgtolerance(false),
            quadraticcgtolerance(false),
            decreasingcgtolerancefactor(0.0f),
            adaptlambda(true),
            initlambda(0.1f),
            maxlambda(std::numeric_limits<float>::infinity()),
            usefinaliterateforadaptinglambda(false),
            docgbacktracking(true),
            stopbacktrackingatfirstimprovement(false),
            backtrackingbase(1.3f),
            maxlinesearches(5),
            optoncv(false),
            llfraction(1.0f),
            cgpreconditioning(true),
            preconditioningalpha(0.75f),
            preconditioninglambda(-1.0f),
            cginitdecayingfactor(0.95f),
            cginitfinaliterate(true),
            cginitifnegativeonly(false),
            usetrustregion(false),
            inittrustregionradius(-1.0f),
            adaptlambdabasedontrustregion(false),
            exitcgwhenleavingtr(true),
            maxcgruns(1),
            stochasticmode(false),
            largeminibatchsize(2048),
            verbosity(1),
            usedoubleaccumulator(false)
        { }
    
        bool readfromfile(std::wstring &filename)
        {
            ifstream file(filename.c_str());
            std::string line;
            std::string identifier;
            float value;
            
            try
            {
                while(file.good())
                {
                    std::getline(file, line);
                    if (line[0] != '#')
                    {
                        std::stringstream sline(line);
                        sline >> identifier;
                        sline >> value;
                        parse(identifier, value);
                    }
                }
            }
            catch(exception &e)
            {
                std::cerr << "error in Hessian free configuration file: " << e.what() << std::endl;
                return false;
            }
            return true;
        }

        void parse(const std::string &identifier, float value)
        {
            if (identifier == "secondordermbsize")
            {
                if (value <= 0)
                    throw std::runtime_error("Minibatch size of cg iterations must be strictly positive! Exiting ..");
                secondordermbsize = (size_t) value;
            }
            else if (identifier == "mincgiters")
            {
                if  (value < 0)
                    throw std::runtime_error("Minimal number of cg iterations must be positive! Exiting ..");
                mincgiters = (size_t) value;
            }
            else if (identifier == "maxcgiters")
            {
                if (value < 0)
                    throw std::runtime_error("Maximal number of cg iterations must be positive! Exiting ..");
                maxcgiters = (size_t) value;
            }
            else if (identifier == "nsecondorderminibatches")
            {
                if (value <= 0)
                    throw std::runtime_error("Number of minibatches size for calculating Hessian vector product must be strictly positive! Exiting ..");
                nsecondorderminibatches = (size_t) value;
            }
            else if (identifier == "adaptlambda")
                adaptlambda = value != 0.0f;
            else if (identifier == "initlambda")
            {
                if (value < 0)
                    throw std::runtime_error("Damping factor must be positive! Exiting ..");
                initlambda = value;
            }
            else if (identifier == "maxlambda")
            {
                if (value < 0)
                    throw std::runtime_error("Maximal damping factor must be positive! Exiting ..");
                maxlambda = value;
            }
            else if (identifier == "cgtolerance")
            {
                if (value < 0)
                    throw std::runtime_error("Cg tolerance must be positive! Exiting ..");
                cgtolerance = value;
            }
            else if (identifier == "superlinearcgtolerance")
                superlinearcgtolerance = value != 0.0f;
            else if (identifier == "quadraticcgtolerance")
                quadraticcgtolerance = value != 0.0f;
            else if (identifier == "decreasingcgtolerancefactor")
            {
                if (value < 0)
                    throw std::runtime_error("decreasingcgtolerancefactor base must be positive! Exiting ..");
                decreasingcgtolerancefactor = value;
            }
            else if (identifier == "docgbacktracking")
                docgbacktracking = value != 0.0f;
            else if (identifier == "stopbacktrackingatfirstimprovement")
                stopbacktrackingatfirstimprovement = value != 0.0f;
            else if (identifier == "backtrackingbase")
            {
                if (value <= 1)
                    throw std::runtime_error("backtracking base must be greater than one! Exiting ..");
                backtrackingbase = value;
            }
            else if (identifier == "maxlinesearches")
            {
                if (value <= 1)
                    throw std::runtime_error("max linesearches must be greater than one! Exiting ..");
                maxlinesearches = (size_t ) value;
            }
            else if (identifier == "optoncv")
            {
                optoncv = value != 0.0f;
            }
            else if (identifier == "cgpreconditioning")
            {
                cgpreconditioning = value != 0.0f;
            }
            else if (identifier == "preconditioningalpha")
            {
                if (value <= 0)
                    throw std::runtime_error("preconditioningalpha must be positive! Exiting ..");
                preconditioningalpha = value;
            }
            else if (identifier == "preconditioninglambda")
            {
                if (value <= 0)
                    throw std::runtime_error("preconditioninglambda must be positive! Exiting ..");
                preconditioninglambda = value;
            }
            else if (identifier == "cginitdecayingfactor")
            {
                if (value < 0)
                    throw std::runtime_error("cginitdecayingfactor must be positive! Exiting ..");
                cginitdecayingfactor = value;
            }
            else if (identifier == "cginitfinaliterate")
                cginitfinaliterate = value != 0.0f;
            else if (identifier == "cginitifnegativeonly")
                cginitifnegativeonly = value != 0.0f;
            else if (identifier == "usefinaliterateforadaptinglambda")
                usefinaliterateforadaptinglambda = value != 0.0f;
            else if (identifier == "inittrustregionradius")
            {
                if (value < 0)
                    throw std::runtime_error("inittrustregionradius must be non-negative! Exiting ..");
                inittrustregionradius = value;
            }
            else if (identifier == "llfraction")
            {
                if (value <= 0 || value > 1.0)
                    throw std::runtime_error("llfraction must be in (0,1] ! Exiting ..");
                llfraction = value;
            }
            else if (identifier == "usetrustregion")
                usetrustregion = value != 0.0f;
            else if (identifier == "exitcgwhenleavingtr")
                exitcgwhenleavingtr = value != 0.0f;
            else if (identifier == "adaptlambdabasedontrustregion")
                adaptlambdabasedontrustregion = value != 0.0f;
            else if (identifier == "stochasticmode")
                stochasticmode = value != 0.0f;
            else if (identifier == "maxcgruns")
            {
                if (value <= 1)
                    throw std::runtime_error("maxcgruns must be greater than one");
                maxcgruns = (size_t) value;
            }
            else if (identifier == "largeminibatchsize")
            {
                if (value <= 1)
                    throw std::runtime_error("largeminibatchsize must be larger than one!");
                largeminibatchsize = (size_t) value;
            }
            else if (identifier == "verbosity")
                verbosity = (int) value;
            else if (identifier == "usedoubleaccumulator")
                usedoubleaccumulator = value != 0;
            else 
                throw std::runtime_error(msra::strfun::strprintf("can not parse identifier: %s", identifier.c_str()));
            
        }

        // show configuration
        void show()
        {
            std::cerr << "HESSIAN FREE CONFIGURATION" << std::endl;
            std::cerr << "secondordermbsize\t" << secondordermbsize << std::endl;
            std::cerr << "mincgiters\t" << mincgiters << std::endl;
            std::cerr << "maxcgiters\t" << maxcgiters << std::endl;
            std::cerr << "nsecondorderminibatches\t" << nsecondorderminibatches << std::endl;
            std::cerr << "adaptlambda\t" << adaptlambda << std::endl;
            std::cerr << "initlambda\t" << initlambda << std::endl;
            std::cerr << "maxlambda\t" << maxlambda << std::endl;
            std::cerr << "cgtolerance\t" << cgtolerance << std::endl;
            std::cerr << "superlinearcgtolerance\t" << superlinearcgtolerance << std::endl;
            std::cerr << "quadraticcgtolerance\t" << quadraticcgtolerance << std::endl;
            std::cerr << "decreasingcgtolerancefactor\t" << decreasingcgtolerancefactor << std::endl;
            std::cerr << "docgbacktracking\t" << docgbacktracking << std::endl;
            std::cerr << "stopbacktrackingatfirstimprovement\t" << stopbacktrackingatfirstimprovement << std::endl;
            std::cerr << "backtrackingbase\t" << backtrackingbase << std::endl;
            std::cerr << "optoncv\t" << optoncv << std::endl;
            std::cerr << "cgpreconditioning\t" << cgpreconditioning << std::endl;
            std::cerr << "preconditioningalpha\t" << preconditioningalpha << std::endl;
            std::cerr << "preconditioninglambda\t" << preconditioninglambda << std::endl;
            std::cerr << "cginitdecayingfactor\t" << cginitdecayingfactor << std::endl;
            std::cerr << "cginitfinaliterate\t" << cginitfinaliterate << std::endl;
            std::cerr << "cginitifnegativeonly\t" << cginitifnegativeonly << std::endl;
            std::cerr << "usefinaliterateforadaptinglambda\t" << usefinaliterateforadaptinglambda << std::endl;
            std::cerr << "llfraction\t" << llfraction << std::endl;
            std::cerr << "usetrustregion\t" << usetrustregion << std::endl;
            std::cerr << "inittrustregionradius\t" << inittrustregionradius << std::endl;
            std::cerr << "adaptlambdabasedontrustregion\t" << adaptlambdabasedontrustregion << std::endl;
            std::cerr << "maxcgreruns\t" << maxcgruns << std::endl;
            std::cerr << "exitcgwhenleavingtr" << exitcgwhenleavingtr << std::endl;
            std::cerr << "stochasticmode\t" << stochasticmode << std::endl;
            std::cerr << "largeminibatchsize\t" << largeminibatchsize << std::endl;
            std::cerr << "verbosity\t" << verbosity << std::endl;
            std::cerr << "usedoubleaccumulator\t" << usedoubleaccumulator << std::endl;
        }

        // returns the maximal number of models that can be stored for cg backtracking (depending on backtrackingbase & maxcgiters)
        size_t nofbacktrackingmodels() const
        {
            if (!docgbacktracking)
                return 1;
            size_t result = 0;
            size_t indexofnextstoredmodel = 1;
            for (size_t cgiter = 1; cgiter < maxcgiters + 1; cgiter++)
            {
                if (cgiter == indexofnextstoredmodel)
                {
                    indexofnextstoredmodel = (size_t) std::ceil(indexofnextstoredmodel * backtrackingbase);
                    result++;
                }
            }
            // always store the last iteration
            if (maxcgiters > 1)
                result++;
            return result;
        }
    };

    
    // evaluate log likelihood
    // TODO generalize method to other framesources
    // only use fraction information if start and end is not explicitly passed
    static double evalll (msra::dbn::model::trainer &trainer, msra::dbn::minibatchutterancesource & cvfeatsource, msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence>& cvlabels, const size_t epoch, const size_t mbsizeparam, float dropoutrate, bool prescaledropout, 
                            double &accuracy, int verbosity = 2, float fraction = 1.0, size_t startframe = 0, size_t endframe = 0)
    {
        const char * operation = "evalll";

        const size_t epochframes = endframe != 0 ? endframe - startframe + 1 : cvfeatsource.gettotalframes();
        const size_t epochstartframe = startframe != 0 ? startframe : epoch * epochframes;
        // only use fraction information if start and end is not explicitly passed
        const size_t epochendframe  = endframe != 0 ? epochstartframe + epochframes : epochstartframe + (size_t) std::ceil(fraction * epochframes);

        msra::dbn::matrix feat; // storage for minibatch data
        std::vector<size_t> uids;
		//zhaorui realign for reference path
		std::vector<size_t> phoneboundary;
        // accumulation of overall metric
        double logpsum = 0.0;
        double fcorsum = 0.0;   // frames correct rate
        size_t logpframes = 0;

        // do the minibatches
        size_t mbsize = mbsizeparam; 
        for (size_t mbstartframe = epochstartframe; mbstartframe < epochendframe; /*mbstartframe += framesadvanced; at end of loop*/)
        {
            msra::basetypes::auto_timer mbtimer;     // end-to-end timer per epoch

            // process one mini-batch (accumulation and update)
            const size_t requestedframes = min (mbsize, epochendframe - mbstartframe);    // (< mbsize at end)
            std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> latticesdummy;
            std::vector<const_array_ref<msra::lattices::lattice::htkmlfwordsequence::word>> transcriptsdummy;
            
            msra::dbn::matrix targetfeatdummy;
            size_t framesadvanced;

            //zhaorui realign for reference path
            cvfeatsource.getbatch (mbstartframe, requestedframes, 0, 1/*note: not supporting MPI for now*/, framesadvanced, feat, targetfeatdummy, uids, transcriptsdummy, latticesdummy, phoneboundary);
            if (targetfeatdummy.rows() != 0)
                throw std::runtime_error ("evalll(hessianfreetrainer.h): training with target features not supported");

            const size_t framesinblock = feat.cols();   // it may still return less

            trainer.forwardprop (feat, 0, framesinblock, dropoutrate, prescaledropout);

            auto stats /* (logp, fcor) */ = trainer.posteriorstats (uids, 0, framesinblock, false/*nosoftmax --not supported here*/, verbosity);
            trainer.synchronize();    // complete offloaded computation, for accurate time measurement
            logpsum += stats[0] * framesinblock;
            fcorsum += stats[1] * framesinblock;
            logpframes += framesinblock;

            double framedur = 0.01;// FIX THIS sampperiods[0] * 1e-7;    // (assume all equal)
            double filedur = framesadvanced * framedur;
            double cpudur = mbtimer;
            double timeperframe = cpudur / framesadvanced;
            if (verbosity >= 2)
                fprintf (stderr, "%s [%d..%d (%.1f%%)]: cpu %.2fs vs. file %.2fs -> %.2f ms/frame, %.1f min/ep (%.1f left)\n\n",
                        operation, mbstartframe, mbstartframe + framesadvanced -1,
                        100.0 * (mbstartframe + framesadvanced - epochstartframe) / (epochendframe - epochstartframe),
                        cpudur, filedur, timeperframe * 1000.0,
                        timeperframe * (epochendframe - epochstartframe) / 60.0, timeperframe * (epochendframe - mbstartframe) / 60.0);

            // advance to next global frame index
            mbstartframe += framesadvanced;
        }

        // done with accelerated computation
        if (verbosity >= 1)
            fprintf (stderr, "\n%s: completed epoch %d, av log %s = %.2f in %d frames (%.1f%% correct)\n", operation, (int) epoch,
                    "pp", logpsum / logpframes, logpframes, fcorsum / logpframes * 100.0);

        accuracy = fcorsum / logpframes * 100.0;
        return logpsum / logpframes;
    }

    // perform a linesearch with hessianfree trainer
    // reduce step size until likelihood decreases
    static void linesearch(msra::dbn::model::hessianfreetrainer &hftrainer, float factor, double cvmax, size_t cvargmax, double cvacc, size_t epoch, float fraction, size_t llevalstartframe, size_t llevalendframe,
        size_t mbsize, msra::dbn::minibatchutterancesource &featuresource, msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> &labels,
        const msra::dbn::hfinfo *hessianfreeinfo, size_t &linesearchiter, double &newcvavgll, double &newcvavgacc, float &stepsize)
    {
        double linesearchprevll = cvmax;
        double linesearchll = linesearchprevll;
        double linesearchprevacc = cvacc;
        double linesearchacc = linesearchprevacc;
        float linesearchstepsize = 1.0f;
        while (linesearchll >= linesearchprevll && linesearchiter < hessianfreeinfo->maxlinesearches)
        {
            linesearchstepsize *= factor;
            std::cerr << "line search: iter\t" << linesearchiter << " (step size: " << linesearchstepsize << " )" << std::endl;
            hftrainer.settointermediateresult(cvargmax, linesearchstepsize);
            linesearchprevll = linesearchll;
            linesearchprevacc = linesearchacc;
            linesearchll = evalll(hftrainer, featuresource, labels, epoch, mbsize, 0.0f, false, linesearchacc, 1, fraction, llevalstartframe, llevalendframe);
            linesearchiter++;
        }
        if (linesearchprevll > cvmax)
        {
            newcvavgll = linesearchprevll;
            newcvavgacc = linesearchprevacc;
            stepsize = linesearchstepsize / factor;
        }
        else
        {
            newcvavgll = cvmax;
            newcvavgacc = cvacc;
            stepsize = 1.0f;
        }
    }
    
    // calculate Hessian vector product: hessianvectorproduct = B * cgsearchdirection
    // use the same data in each CG iteration
    // data should be part of that used for the gradient evaluation
    static void hessianvectorproduct(msra::dbn::model::hessianfreetrainer &trainer, float hessianfreelambda, size_t epoch, size_t startframe, size_t endframe,  
        msra::dbn::minibatchsource &featuresource, msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> &labels,
        const msra::dbn::hfinfo *hessianfreeinfo)
    {
        msra::dbn::minibatchiterator mbiter (featuresource, epoch, startframe, endframe, hessianfreeinfo->secondordermbsize, 0, 1, 1);
        for (size_t mbidx = 0; mbidx < hessianfreeinfo->nsecondorderminibatches; mbidx++)
        {
            // get minibatch data
            const msra::dbn::matrix & feat = mbiter.frames();
            std::vector<size_t> & uids = mbiter.labels();
            const size_t mbstartframe = mbiter.currentmbstartframe();
            const size_t actualmbsize = feat.cols();   // it may still return less if at end of sweep
            assert (actualmbsize == mbiter.currentmbframes());
            assert (actualmbsize == hessianfreeinfo->secondordermbsize);

            // calculate Hessian vector product on minibatch
            // conventional forwardprop
            trainer.forwardprop(feat, 0, actualmbsize, 0.0f, false);
            // modifiefied forwardprop for Hessian vector statistics
            trainer.forwardprophessianvectorproduct(0, actualmbsize);
            // modifiefied sethessianvectorsignal for Hessian vector statistics
            // depends on the objective function
            // here: use cross-entropy criterion
            trainer.sethessianvectorsignal(0, actualmbsize);
            // conventional backpropagation (but starting from different error signal)
            trainer.errorbackprop(0, actualmbsize);
                
            // collect statistics
            bool isfirstbatch = mbidx == 0;
            trainer.collecthessianvectorproduct (0, actualmbsize, isfirstbatch, hessianfreeinfo->secondordermbsize * hessianfreeinfo->nsecondorderminibatches);
            // get next intermediate batch
            mbiter++;
        }
        // add damping term to Hessian vector product
        trainer.adddampingterm(hessianfreelambda);
    }

    // run conjugate gradient
    static void hessianfreecg(msra::dbn::model::hessianfreetrainer &trainer, const hfinfo *hessianfreeinfo, bool isdebugging, float hessianfreelambda, float hessianfreetrustregionradius, 
        size_t epoch, size_t hvproductstartframe, size_t hvproductendframe, float cgresidualtolerancesuperlinear, float cgresidualtolerancequadratic,
        std::unique_ptr<msra::dbn::minibatchsource> &featsource, msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> &labels, 
        size_t &nstoredmodels, std::vector<float> &cgobjectivefunctionvalues, std::vector<size_t> &backtrackingiterationindices, size_t &numberofcgiters,
        bool &isattrustregionboundary)
    {
        bool exitfromcg = false;
        size_t indexofnextstoredmodel = 1;
        for (size_t cgiter = 1; cgiter < hessianfreeinfo->maxcgiters + 1; cgiter++)
        {
            std::cerr << "\nCG ITERATION " << cgiter << std::endl;    
            if (isdebugging)
            {
                std::cerr << "WARNING: USING DEBUGGING HV PRODUCT!!" << std::endl;
                trainer.setdummyhessianvectorproduct();
            }
            else
                hessianvectorproduct(trainer, hessianfreelambda, epoch, hvproductstartframe, hvproductendframe, *featsource, labels, hessianfreeinfo);

            // calculate stepsize
            // cgstepsize = cgresidualnormsquared / cgcurvatureproduct
            // cgresidualnormsquared = cgresidual' * cgresidual
            // cgcurvatureproduct = cgsearchdirection' * B * cgsearchdirection ( = cgsearchdirection' * hessianvectorproduct)
            float cgcurvatureproduct = trainer.calculatecgcurvatureproduct();
            float cgresidualnormsquared = hessianfreeinfo->cgpreconditioning ? trainer.calculatepcgresidualnorm() : trainer.calculatecgresidualnorm(false);
            float cgstepsize = cgresidualnormsquared/ cgcurvatureproduct;
                
            std::cerr << "cg " << cgiter << ": cg curvature product:\t" << cgcurvatureproduct << std::endl;
            std::cerr << "cg " << cgiter << ": step size:\t" << cgstepsize << std::endl;

            // check curvature product for non-positivity, must not happen for damped Gauss-Newton matrix (at least with exact arithmetic)
            if (cgcurvatureproduct <= 0.0f)
            {
                std::cerr << "WARNING: direction of non-positive curvature encountered:\t" << cgcurvatureproduct ;
                std::cerr << "cg " << cgiter << ": Exit from conjugate gradient." << std::endl; 
                // store final model
                if (nstoredmodels == 0 || cgiter - 1 > backtrackingiterationindices[nstoredmodels - 1])
                {
                    trainer.storecgiterate(nstoredmodels);
                    backtrackingiterationindices[nstoredmodels] = cgiter - 1;
                    nstoredmodels++;
                    break;
                }
            }

            // update cgiterate
            // cgiterate += stepsize * cgsearchdirection
            trainer.updatecgiterate(cgstepsize);
            numberofcgiters++;

            // damped-HF: norm of iterates is only needed for debugging/ logging
            // when starting from zero, norm must be increasing
            // norm can be used for stopping criterion based on trust region 
            // always use Euclidean norm (not the norm implied by the preconditioner)
            float cgiteratenrmsquared = trainer.calculatesquaredcgiteratenorm(false);
            float cgiteratenrm =  std::sqrt(cgiteratenrmsquared);
            std::cerr << "cg " << cgiter << ": iterate norm:\t" << cgiteratenrm << std::endl;

            // if using TR approach: go until boundary of TR
            if (hessianfreeinfo->usetrustregion && cgiteratenrm >= hessianfreetrustregionradius && hessianfreeinfo->exitcgwhenleavingtr)
            {
                std::cerr << "trust region: trust region left, only going until boundary of trust region" << std::endl;
                // revert previous step
                trainer.updatecgiterate(-cgstepsize);
                // calculate step size until trust region boundary
                float cgiteratenrmsquared = trainer.calculatesquaredcgiteratenorm(false);
                float cgsearchdirectionsquarednorm = trainer.calculatesquaredcgsearchdirectionnorm(false);
                float scalarproduct = trainer.calculatecgiteratecgsearchdirectionproduct(false);
                float ph = scalarproduct / cgsearchdirectionsquarednorm;
                float q = (cgiteratenrmsquared - hessianfreetrustregionradius * hessianfreetrustregionradius) / cgsearchdirectionsquarednorm;
                assert(ph * ph - q >= 0);
                float cgboundarystepsize = -ph + std::sqrt(ph * ph - q);
                // new (restricted) step
                trainer.updatecgiterate(cgboundarystepsize);
                isattrustregionboundary = true;
                exitfromcg = true;
            }
                
            // store model for backtracking
            if (hessianfreeinfo->docgbacktracking && (cgiter == indexofnextstoredmodel || cgiter == hessianfreeinfo->maxcgiters))
            {
                indexofnextstoredmodel = (size_t) std::ceil(indexofnextstoredmodel * hessianfreeinfo->backtrackingbase);
                trainer.storecgiterate(nstoredmodels);
                backtrackingiterationindices[nstoredmodels] = cgiter;
                nstoredmodels++;
            }
                
            // update cgresidual
            // cgresidual += stepsize + hessianvectorproduct
            trainer.updatecgresidual(cgstepsize);
            
            // solve for pc residual
            if (hessianfreeinfo->cgpreconditioning)
                trainer.solveforpcgresidual();
                
            float cgnewresidualnormsquared = hessianfreeinfo->cgpreconditioning ? trainer.calculatepcgresidualnorm() : trainer.calculatecgresidualnorm(false);
            float cgnewresidualnorm = std::sqrt(cgnewresidualnormsquared);
            fprintf(stderr, "cg %d: residual norm:\t%f\n", cgiter, cgnewresidualnorm);
                
            // update cgsearchdirection
            if (hessianfreeinfo->cgpreconditioning)
                trainer.updatepcgsearchdirection(cgnewresidualnormsquared/cgresidualnormsquared);
            else
                trainer.updatecgsearchdirection(cgnewresidualnormsquared/cgresidualnormsquared);

            // calculate new cg objective (for stopping criterion and monotonicity check)
            float obj1 = trainer.calculatecgresidualcgiterateproduct(false);
            float obj2 = trainer.calculategradientcgiterateproduct(false);
            cgobjectivefunctionvalues[cgiter] = 0.5f*(obj1 - obj2);
            float cgobjectivefunctiondecrease = cgobjectivefunctionvalues[cgiter] - cgobjectivefunctionvalues[cgiter - 1];
            std::cerr << "cg " << cgiter << ": objective:\t" << cgobjectivefunctionvalues[cgiter] << " = 0.5*( " << obj1 << " + " << obj2 << ")" << std::endl;
            std::cerr << "cg " << cgiter << ": objective decrease:\t" << cgobjectivefunctiondecrease << std::endl;
                
            // cg termination check
            // looking maxpast steps back for checking termination conditions of conjugate gradient
            size_t maxpast = max(hessianfreeinfo->mincgiters, cgiter/10);            
            if (cgnewresidualnormsquared == 0.0f || cgresidualnormsquared == 0.0f) // always stop if residual is zero
            {
                std::cerr << "cg " << cgiter << ": residual is zero." << std::endl;
                exitfromcg = true;
            }
            else if (cgiter == hessianfreeinfo->maxcgiters) // stop if maximum of cg iters is reached
            {
                std::cerr << "cg " << cgiter << ": maximal number of CG iterations performed." << std::endl;
                exitfromcg = true;
            }
            else if (cgiter >= hessianfreeinfo->mincgiters) // check described in Martens paper based on CG objective function
            {
                assert (cgiter >= maxpast);
                if (cgobjectivefunctionvalues[cgiter] < 0)
                {
                    float relativechange = (cgobjectivefunctionvalues[cgiter] - cgobjectivefunctionvalues[cgiter - maxpast])/ cgobjectivefunctionvalues[cgiter];
                    if (relativechange <= maxpast * hessianfreeinfo->cgtolerance){
                        std::cerr << "average relative change in cg objective is below threshold: " << relativechange / maxpast << std::endl;
                        std::cerr << "... and cg objective is negative." << std::endl;
                        exitfromcg = true;
                    }
                }
                else 
                {
                    std::cerr << "cgobjective is still greater than zero, continue iterating" << std::endl;
                }
                if (exitfromcg && (hessianfreeinfo->superlinearcgtolerance || hessianfreeinfo->quadraticcgtolerance)) // conventional termination check based on residual norm (see Nocedal)
                {
                    float euclideanresidualnorm = hessianfreeinfo->cgpreconditioning ? std::sqrt(trainer.calculatecgresidualnorm(false)) : cgnewresidualnorm;
                    if (hessianfreeinfo->superlinearcgtolerance)
                        exitfromcg = exitfromcg && (euclideanresidualnorm < cgresidualtolerancesuperlinear);
                    else if (hessianfreeinfo->quadraticcgtolerance)
                        exitfromcg = exitfromcg && (euclideanresidualnorm < cgresidualtolerancequadratic);
                    if (exitfromcg)
                        std::cerr << "termination check based on gradient norm is true too" << std::endl;
                    else
                        std::cerr << "but termination based on gradient norm is false, continue iterating" << std::endl;
                }
            }
            if (exitfromcg) // exit from cg
            {
                std::cerr << "Exit from CG" << std::endl;
                if (nstoredmodels == 0 || cgiter > backtrackingiterationindices[nstoredmodels - 1])
                {
                    trainer.storecgiterate(nstoredmodels);
                    backtrackingiterationindices[nstoredmodels] = cgiter;
                    nstoredmodels++;
                }
                break;
            }
        }
    }

    // perform update step of HF optimization
    static void hessianfreeupdate(msra::dbn::model::hessianfreetrainer &trainer, size_t numframesforgradientcalculation, const hfinfo *hessianfreeinfo, size_t epoch, 
        size_t hvproductstartframe, size_t hvproductendframe, size_t llevalstartframe, size_t llevalendframe, size_t mbsizeallocation, 
        std::unique_ptr<msra::dbn::minibatchsource> &featsource, msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> &labels, 
        minibatchutterancesource &cvfeatsource,  msra::asr::htkmlfreader<msra::asr::htkmlfentry,msra::lattices::lattice::htkmlfwordsequence> &cvlabels,
	float &hessianfreelambda, float &hessianfreetrustregionradius, double &cvavgll, double &cvavgacc, size_t &totalnumberofcgbacktrackings, size_t &totalnumberoflinesearches)
    {
        std::cerr << "\nHESSIAN FREE UPDATE STEP" << std::endl;
        std::cerr << "gradient has been collected on " <<  numframesforgradientcalculation << " frames" << std::endl;
            
        // CG initialization
        std::cerr << "initializing CG" << std::endl;
            
        // backup model, required for linesearch
        trainer.backupmodel();
        // set gradient to double accumulator
        if (hessianfreeinfo->usedoubleaccumulator)
        {
            trainer.settoaccumulator(hessianfreeinfo->cgpreconditioning);
        }

        // normalize gradient by number of observations
        trainer.normalizegradient(numframesforgradientcalculation);
        // variables for managing CG backtracking
        
        size_t nstoredmodels = 0;
        std::vector<size_t> backtrackingiterationindices(hessianfreeinfo->nofbacktrackingmodels());
        // CG objective function values
        std::vector<float> cgobjectivefunctionvalues(hessianfreeinfo->maxcgiters + 1);
        // for debugging of CG
        bool isdebugging = false;
        if (isdebugging)
            trainer.setdummygradient();
            
        // initialization of CG: either with zero or with final iterate from last CG run
        bool initcgfromzero = hessianfreeinfo->cginitdecayingfactor == 0.0f || trainer.shallinitnextcgfromzero() || isdebugging;
        float cginitlambda = hessianfreeinfo->preconditioninglambda != -1.0f ? hessianfreeinfo->preconditioninglambda : hessianfreelambda;
        if (initcgfromzero)
	{
            std::cerr << "cg init: initializing cg from zero" << std::endl;
            trainer.initcgfromzero(hessianfreeinfo->cgpreconditioning, (float) numframesforgradientcalculation, cginitlambda, hessianfreeinfo->preconditioningalpha);
            cgobjectivefunctionvalues[0] = 0.0f;
            std::cerr << "cg init: cg objective:\t" << cgobjectivefunctionvalues[0] << " = 0.5*( " << 0.0f << " + " << 0.0f << ")" << std::endl;
        }
        else
        {
            hessianvectorproduct(trainer, hessianfreelambda, epoch, hvproductstartframe, hvproductendframe, *featsource, labels, hessianfreeinfo);
            trainer.initcg(hessianfreeinfo->cgpreconditioning, (float) numframesforgradientcalculation, hessianfreelambda, hessianfreeinfo->preconditioningalpha);
            float obj1 = trainer.calculatecgresidualcgiterateproduct(false);
            float obj2 = trainer.calculategradientcgiterateproduct(false);
            cgobjectivefunctionvalues[0] = 0.5f*(obj1 - obj2);
            std::cerr << "cg init: cg objective:\t" << cgobjectivefunctionvalues[0] << " = 0.5*( " << obj1 << " + " << obj2 << ")" << std::endl;
            if (hessianfreeinfo->cginitifnegativeonly && cgobjectivefunctionvalues[0] >= 0)
            {
                std::cerr << "cg init: re-initializing cg with zero because of positive cg objective" << std::endl;
                trainer.initcgfromzero(hessianfreeinfo->cgpreconditioning, (float) numframesforgradientcalculation, cginitlambda, hessianfreeinfo->preconditioningalpha);
                cgobjectivefunctionvalues[0] = 0.0f;
                initcgfromzero = true;
            }
        }

        // parameternorm (always use Euclidean norm), only needed for logging or TR approach
        float parameternorm = std::sqrt(trainer.calculatesquaredparameternorm(false));
        float gradientnorm = std::sqrt(trainer.calculatesquaredgradientnorm(false));
            
        std::cerr << "cg init: norm of model is " << parameternorm << std::endl;
        std::cerr << "cg init: norm of gradient is " << gradientnorm << std::endl;

        // CG tolerances based on norm of gradient (see Nocedal, Theorem 7.2)
        float cgresidualtolerancesuperlinear = hessianfreeinfo->decreasingcgtolerancefactor * min(0.5f, std::sqrt(gradientnorm)) * gradientnorm;
        float cgresidualtolerancequadratic = hessianfreeinfo->decreasingcgtolerancefactor * min(0.5f, gradientnorm) * gradientnorm; 

        // trust region
        // automatic setting of initial trust region radius
        if (hessianfreetrustregionradius == -1.0f)
        {
            hessianfreetrustregionradius = 0.01f * parameternorm;
            std::cerr << "trust region: automatic setting of initial value to\t" << hessianfreetrustregionradius << std::endl;
        }
        else if (hessianfreetrustregionradius >= 0.5f * parameternorm)
        {
            std::cerr << "trust region: restricting trust region radius to 0.5 * model norm" << std::endl;
            hessianfreetrustregionradius = 0.5f * parameternorm;
        }
        // only for experimental TR algorithm
        size_t lambdareruns = 0;
        bool isattrustregionboundary = false;
        if (!initcgfromzero && hessianfreeinfo->usetrustregion)
        {
            float cgiteratenorm = std::sqrt(trainer.calculatesquaredcgiteratenorm(hessianfreeinfo->cgpreconditioning));
            if (cgiteratenorm > hessianfreetrustregionradius)
            {
                std::cerr << "cg init: re-initializing cg with zero because initialization is out of trust region" << std::endl;
                trainer.initcgfromzero(hessianfreeinfo->cgpreconditioning, (float) numframesforgradientcalculation, cginitlambda, hessianfreeinfo->preconditioningalpha);
                cgobjectivefunctionvalues[0] = 0.0f;
                initcgfromzero = true;
            }
        }

        // conjugate gradient algorithm
        // in Martens algorithm: only one cgrun
        size_t numberofcgiters = 0;
        // includes multiple runs of cg
        size_t totalnumberofcgiters = 0;
        size_t cgrun = 0;
        for (;cgrun < hessianfreeinfo->maxcgruns;cgrun++)
        {
            if (cgrun > 0)
            {
                float cginitlambda = hessianfreeinfo->preconditioninglambda != -1.0f ? hessianfreeinfo->preconditioninglambda : hessianfreelambda;
                trainer.initcgfromzero(hessianfreeinfo->cgpreconditioning, (float) numframesforgradientcalculation, cginitlambda, hessianfreeinfo->preconditioningalpha);
                cgobjectivefunctionvalues[0] = 0.0f;
                hessianfreelambda *= 1.5f;
                nstoredmodels = 0;
                std::cerr << "rerunning cg with adapted lambda:\t" << hessianfreelambda << std::endl;
            }
            hessianfreecg(trainer, hessianfreeinfo, isdebugging, hessianfreelambda, hessianfreetrustregionradius, epoch, hvproductstartframe, hvproductendframe, cgresidualtolerancesuperlinear, cgresidualtolerancequadratic,
                featsource, labels, nstoredmodels, cgobjectivefunctionvalues, backtrackingiterationindices, numberofcgiters, isattrustregionboundary);
            totalnumberofcgiters += numberofcgiters;
            if (!isattrustregionboundary)
                break;
        }


        // perform iteration backtracking
        // test models beginning with final CG iterate, backtrack until no improvement in one step (unless hessianfreeinfo->stopbacktrackingatfirstimprovement is set)
        // requires evaluation of log-likelihood of the data => expensive!
        std::cerr << "backtracking: backtracking over " << nstoredmodels << " models " << std::endl;
        std::vector<double> cvlls(nstoredmodels, -std::numeric_limits<double>::infinity());
        std::vector<double> cvaccuracies(nstoredmodels, 0.0);
        size_t cvargmax = nstoredmodels;
        double cvmax = -std::numeric_limits<double>::infinity();
        std::cerr << "backtracking: previous model ll: " << cvavgll << std::endl;
        int ncgbacktrackings = 0;
        for (int backtrackindex = (int) nstoredmodels - 1; backtrackindex >= 0; backtrackindex--)
        {
            ncgbacktrackings++;
            totalnumberofcgbacktrackings++;
            trainer.settointermediateresult((size_t) backtrackindex, 1.0f);
            if (hessianfreeinfo->optoncv)
                cvlls[backtrackindex] = evalll(trainer, cvfeatsource, cvlabels, 0, mbsizeallocation, 0.0f, false, cvaccuracies[backtrackindex], 0, hessianfreeinfo->llfraction);
            else
                cvlls[backtrackindex] = evalll(trainer, *dynamic_cast<minibatchutterancesource*>(featsource.get()), labels, epoch, mbsizeallocation, 0.0f, false, cvaccuracies[backtrackindex], 1, 1.0f, llevalstartframe, llevalendframe);
            fprintf(stderr, "backtracking iteration %d (cg iteration %d): avg ll is\t%.7f (\tavg accuracy %.7f)\n", backtrackindex, backtrackingiterationindices[backtrackindex], cvlls[backtrackindex], cvaccuracies[backtrackindex]);
            if (cvlls[backtrackindex] > cvmax)
            {
                cvmax = cvlls[backtrackindex];
                cvargmax = backtrackindex;
            }
            if ((size_t) backtrackindex < nstoredmodels - 1 && cvlls[backtrackindex] < cvlls[backtrackindex + 1] && cvlls[backtrackindex + 1] > cvavgll)
            {
                fprintf(stderr, "backtracking: no further improvement, stopping backtracking\n");
                break;
            }
            if (hessianfreeinfo->stopbacktrackingatfirstimprovement && cvlls[backtrackindex] > cvavgll)
            {
                fprintf(stderr, "backtracking: improvement found, stopping backtracking\n");
                break;
            }
        }
        fprintf (stderr, "backtracking epoch %d: best cg iteration (of %d-%d):\t%d\t, ll\t%.7f\t, accuracy\t%.7f\n", epoch, numberofcgiters, totalnumberofcgiters, backtrackingiterationindices[cvargmax], cvlls[cvargmax], cvaccuracies[cvargmax]);
        double newcvavgll = cvlls[cvargmax];
        double newcvavgacc = cvaccuracies[cvargmax];
        
        // backtracking linesearch
        // only perform linesearch when no improvement has been obtained until now
        // this differs from Kingsbury's implementation, but seems to make more sense and is corresponds to the algorithm in [Nocedal & Yuan, Combining Trust Region and Line Search Technique]
        size_t linesearchiter = 0;
        float stepsize = 0.0f;
        if (newcvavgll > cvavgll) 
        {
            std::cerr << "backtracking: resuming from iteration " << backtrackingiterationindices[cvargmax] << std::endl;
            stepsize = 1.0f;
        }
        else
        {
            std::cerr << "line search: performing backtracking linesearch" << std::endl;
            if (hessianfreeinfo->optoncv)
                linesearch(trainer, 0.5f, cvmax, cvargmax, cvaccuracies[cvargmax], 0, hessianfreeinfo->llfraction, 0, 0, mbsizeallocation, cvfeatsource, cvlabels, hessianfreeinfo, 
                    linesearchiter, newcvavgll, newcvavgacc, stepsize);
            else
                linesearch(trainer, 0.5f, cvmax, cvargmax, cvaccuracies[cvargmax], epoch, 1.0f, llevalstartframe, llevalendframe, mbsizeallocation, *dynamic_cast<minibatchutterancesource*>(featsource.get()), labels, hessianfreeinfo, 
                    linesearchiter, newcvavgll, newcvavgacc, stepsize);
            if (stepsize == 1.0f)
            {
                std::cerr << "line search: did not yield any improvement" << std::endl;
                stepsize = 0.0f;
            }
            else if (stepsize != 1.0f && newcvavgll > cvavgll)
                std::cerr << "line search: resuming from model with stepsize " << stepsize << std::endl;
            else
            {
                std::cerr << "line search: improved over cg result, but still worse than previous model" << std::endl;
                stepsize = 0.0f;
            }
            totalnumberoflinesearches += linesearchiter;
        }
        if (stepsize != 0.0f)
            trainer.settointermediateresult(cvargmax, stepsize);
        else
        {
            std::cerr << "reverting step" << std::endl;
            trainer.restoremodel();
            newcvavgll = cvavgll;
            newcvavgacc = cvavgacc;
        }
            
        // determine new damping term
        // do not include line search model in expected improvement
        // in Martens paper: cg backtracking model is used, here: option for using final cg iterate (but this seems to perform worse)
        if (hessianfreeinfo->adaptlambda)
        {
            double expectedimprovement = hessianfreeinfo->usefinaliterateforadaptinglambda ? - cgobjectivefunctionvalues[backtrackingiterationindices[nstoredmodels - 1]] 
                : -cgobjectivefunctionvalues[backtrackingiterationindices[cvargmax]];
            double actualimprovement = hessianfreeinfo->usefinaliterateforadaptinglambda ? cvlls[nstoredmodels - 1] - cvavgll: cvlls[cvargmax]- cvavgll;
            double objectivecgmodelratio = actualimprovement / expectedimprovement;
            fprintf(stderr, "damping factor: actual improvement:\t%.7f\n", actualimprovement);
            fprintf(stderr, "damping factor: expected improvement:\t%.7f\n", expectedimprovement);
            fprintf(stderr, "damping factor: ratio is :\t%.7f\n", objectivecgmodelratio);
            if (objectivecgmodelratio < 0.25 && hessianfreelambda < hessianfreeinfo->maxlambda)
            {
                hessianfreelambda *= 1.5f;
                hessianfreelambda = min(hessianfreelambda, hessianfreeinfo->maxlambda);
                std::cerr << "damping factor: increasing lambda to\t" << hessianfreelambda << std::endl;
            }
            else if (objectivecgmodelratio > 0.75)
            {
                hessianfreelambda *=  2.0f/3.0f;
                std::cerr << "damping factor: decreasing lambda to\t" << hessianfreelambda << std::endl;
            }
            else
                std::cerr << "damping factor: keeping lambda at\t" << hessianfreelambda << std::endl;
        }
        // adapt trust region as in Nocedal Algorithm 4.1
        if (hessianfreeinfo->usetrustregion) 
        { 
            // for TR it doesn't make sense to use the values from cg backtracking
            double expectedimprovement = -cgobjectivefunctionvalues[backtrackingiterationindices[nstoredmodels - 1]];
            double actualimprovement = cvlls[nstoredmodels - 1] - cvavgll;
            double objectivecgmodelratio = actualimprovement / expectedimprovement;

            fprintf(stderr, "trust region: actual improvement (without cg backtracking):\t%.7f\n", actualimprovement);
            fprintf(stderr, "trust region: expected improvement:\t%.7f\n", expectedimprovement);
            fprintf(stderr, "trust region: ratio is :\t%.7f\n", objectivecgmodelratio);
            if (objectivecgmodelratio < 0.25)
            {
                hessianfreetrustregionradius *= 0.25f;
                std::cerr << "trust region: decreasing radius to\t" << hessianfreetrustregionradius << std::endl;
            }
            else if (objectivecgmodelratio > 0.75 && isattrustregionboundary) // only adapt if at trust region boundary
            {
                hessianfreetrustregionradius *=  2.0f;
                std::cerr << "trust region: increasing radius to\t" << hessianfreetrustregionradius << std::endl;
            }
            else
                std::cerr << "trust region: keeping radius at\t" << hessianfreetrustregionradius << std::endl;
            // experimental adaptation of lambda based on TR
            if (hessianfreeinfo->adaptlambdabasedontrustregion)
            {
                if (isattrustregionboundary)
                {
                    hessianfreelambda *= 1.5f;
                    std::cerr << "damping factor: increasing lambda to\t" << hessianfreelambda << std::endl;
                }
                else if (objectivecgmodelratio > 0.75)
                {
                    hessianfreelambda *= 2.0f/3.0f;
                    std::cerr << "damping factor: decreasing lambda to\t" << hessianfreelambda << std::endl;
                }
                else
                    std::cerr << "damping factor: keeping lambda at\t" << hessianfreelambda << std::endl;
            }
        }

        fprintf(stderr, "FINISHED EPOCH %d: av ll %.7f\t, av acc %.7f\n", epoch, newcvavgll, newcvavgacc);
        fprintf(stderr, "#cg iterations:\t%d%\n", numberofcgiters);
        fprintf(stderr, "#stored models:\t%d%\n", nstoredmodels);
        fprintf(stderr, "#cg backtracking steps:\t%d%\n", ncgbacktrackings);
        fprintf(stderr, "#line search steps:\t%d%\n", linesearchiter);
        fprintf(stderr, "total number of cg backtracking steps:\t%d%\n", totalnumberofcgbacktrackings);
        fprintf(stderr, "total number of line search steps:\t%d%\n", totalnumberoflinesearches);

        // reset gradient & other statistics
        // initialize iterate of next epoch with last cgiterate * cginitmomentum (only if the cg-backtracking model gave an improvement)
        if (cvmax > cvavgll && hessianfreeinfo->cginitdecayingfactor != 0.0f)
        {
            if (hessianfreeinfo->cginitfinaliterate)
            {
                std::cerr << "finalize: initialize cg iterate of next epoch with last cg iterate" << std::endl;
                trainer.finalizecg(-1, hessianfreeinfo->cginitdecayingfactor);
            }
            else
            {
                std::cerr << "finalize: initialize cg iterate of next epoch with best cg iterate" << std::endl;
                trainer.finalizecg((int) cvargmax, hessianfreeinfo->cginitdecayingfactor);
            }
        }
        else
        {
            std::cerr << "finalize: initialize cg iterate of next epoch with zero" << std::endl;
            trainer.finalizecg(-1, 0.0f);
        }
        cvavgll = newcvavgll;
        cvavgacc = newcvavgacc;
        fprintf(stderr, "###############\n");
    }

}

}
