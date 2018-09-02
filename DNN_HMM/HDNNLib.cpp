#include "HDNNLib.h"
#include "decode\decode.h"
size_t ppl_cores = 1;   // for pplhelpers.h
// BUGBUG: somehow it does not match ppl_cores and msra::parallel::ppl_cores; clean this up
#pragma warning (disable: 4100)         // needed to build--not sure why warning=error is not enabled in the DBN project??
#include "dbn.h"
#include "numahelpers.h"
int msra::numa::node_override = -1;     // for numahelpers.h
#include "pplhelpers.h"
size_t msra::parallel::ppl_cores = 1;   // for pplhelpers.h
#include "htkfeatio.h"
#include "commonFunc.h"
// static variables
static msra::dbn::model *s_pDNNModel = NULL;

static map<wstring, wstring> state2Ph;
static map<wstring, int> state2Index;
static map<wstring, vector<wstring>> ph2States;

static bool divbyprior = false;

// functions
static std::map<wstring, int> State2IndexLoad(const wstring& szStateList)
{
	auto_file_ptr fp = fopenOrDie(szStateList, L"r");
	map<wstring, int> stateList;
	for (int i = 0; !feof(fp); i++)
	{
		WSTRING st = split(fgetlinew(fp), L' ')[0];
		stateList[st] = i;
	}
	return stateList;
}

static std::map<wstring, wstring> State2PhoneLoad(const wstring& szStateList)
{
	auto_file_ptr fp = fopenOrDie(szStateList, L"r");
	map<wstring, wstring> stateList;
	for (int i = 0; !feof(fp); i++)
	{
		vector<wstring> items = split(fgetlinew(fp), L' ');
		stateList[items[0]] = items[1];
	}
	return stateList;
}

static std::map<wstring, vector<wstring>> Ph2StatesLoad(const wstring& szStateList)
{
	auto_file_ptr fp = fopenOrDie(szStateList, L"r");
	map<wstring, vector<wstring>> ph2States;
	for (int i = 0; !feof(fp); i++)
	{
		vector<wstring> items=split(fgetlinew(fp), L' ');
		if (items.size() != 2)
		{
			cout << "error format of the statelsit" << endl;
			return map<wstring, vector<wstring>>();
		}
		else
		{
			if (ph2States.count(items[1]))
			{
				if (find(ph2States[items[1]].begin(), ph2States[items[1]].end(), items[0]) == ph2States[items[1]].end())
					ph2States[items[1]].push_back(items[0]);
				else
					wcout << L"dumplicate statelist " << items[0] << endl;
			}
			else
				ph2States[items[1]].push_back(items[0]);
		}
	}
	return ph2States;
}

void ComputeLikelihoods(const msra::dbn::model *dnnModel,  msra::dbn::matrix & feat, msra::dbn::matrix & loglls, const bool &_divbyprior=true)
{
	// frames number for each block computing by GPU, this will reduce GPU memory allocation
	// but now we fix it
	const size_t nframeinstripe = 256;
	// DBN LL evaluation  --result is just kept in 'pred', OutP() just reads them from 'pred'
	msra::dbn::model::evaluator eval(*dnnModel/*model*/, nframeinstripe);

	size_t nframesleft = feat.cols();

	for (size_t s = 0; nframesleft > 0;)
	{
		const size_t curnframes = min(nframesleft, nframeinstripe);
		//printf("\n\n\n\n\nfrom %d with %d of %d this time.\n\n\n\n\n", s, curnframes, feat.cols());
		msra::dbn::matrixstripe nframestripes(feat, s, curnframes);
		msra::dbn::matrixstripe nframelogllsstripes(loglls, s, curnframes);
		eval.logPuv(nframestripes, nframelogllsstripes, _divbyprior/*divbyprior*/, false/*nosoftmax*/);
		s += curnframes;
		nframesleft -= curnframes;
	}
	//delete &eval;
}

void GetAugmentedObservation(size_t dnninputdim,size_t featdim, size_t nframes, size_t t, msra::dbn::matrix &feat, msra::dbn::matrix &outobs, size_t augedt)
{
	// determine augmentNeighborFrames from ratio of model input layer and DNN (a function inside the HDNNModel)
	int neighbors = (dnninputdim / featdim - 1) / 2;    // neighbor frames used in DNN
	if (featdim * (1 + 2 * neighbors) != dnninputdim)   // must be odd multiple
		throw runtime_error("DNNSet: DNN feature dimension must be odd multiple of input feature dimension");

	int dt; // delta t
	for (dt = -neighbors; dt <= neighbors; dt++)    // loop over neighbor frames
	{
		int subvec = dt + neighbors;                // subvector within augmented vector
		int tin = t + dt;                           // frame to copy
		if (tin < 0) tin = 0;
		else if (tin > nframes - 1) tin = nframes - 1;  // clip to within utterance
		// copy the frame  --we should use ReadAsTable(), but GetBufferFeatPtr() avoids the Observation structure
		
		for (size_t frIndex = 0; frIndex < featdim; frIndex++)
			outobs(subvec*featdim + frIndex, augedt) = feat(frIndex, tin);
		
	}
}
void forwardPropatation(const msra::dbn::model *dnnModel, msra::dbn::matrix &feat, msra::dbn::matrix &pred, const bool &_divbyprior = true)
{
	pred.resize(dnnModel->udim(), feat.cols());
	// augmented features (column vectors)
	size_t nframes = feat.cols();
	size_t featDim = feat.rows();

#ifdef SKIP_FRAME_DECODING // skip even frames for forward propagation, but copy the outputs from its previous frames
	msra::dbn::matrix augmentedFeat(dnnModel->vdim(), (feat.cols()+1)/2);

	foreach_column(t, augmentedFeat)
		GetAugmentedObservation(dnnModel->vdim(),featDim, nframes, 2*t, feat, augmentedFeat,t);
	msra::dbn::matrix evenFramePred(dnnModel->udim(), (feat.cols() + 1) / 2);
	ComputeLikelihoods(dnnModel, augmentedFeat, evenFramePred, _divbyprior);
	foreach_coord(i,j,pred)
		pred(i,j)=evenFramePred(i,j/2);

#else

	msra::dbn::matrix augmentedFeat(dnnModel->vdim(), feat.cols());
	foreach_column(t, augmentedFeat)
		GetAugmentedObservation(dnnModel->vdim(),featDim, nframes, t, feat, augmentedFeat,t);
	ComputeLikelihoods(dnnModel, augmentedFeat, pred, _divbyprior);
#endif
}

template<class MATRIX> void msraMatrix2doublearray(MATRIX & feat, double** score)
{
	size_t featdim = feat.rows();
	size_t numframes = feat.cols();
	for (size_t i = 0; i < numframes; i++)
	{
		for (size_t k = 0; k<featdim; k++)
		{
			score[i][k] = -feat(k, i);
		}
	}

};

double convertLogLikelihoodToLogPosterior(double likelihood, int senoneIndex, const msra::dbn::model *dnnModel)
{
	assert(senoneIndex<dnnModel->udim());
	double result = likelihood;
	result += logf(dnnModel->getprior()[senoneIndex]);
	return result;
}

void obtainRelaxedStateScores(WordDuration &seg, map<wstring, wstring> &_state2Ph, map<wstring, vector<wstring>> &_ph2States, const msra::dbn::model *dnnModel, double ** scores, bool relaxBoundary = false, bool inputLikelihood = true)
{

	
	int reIndex = 0;
	for (int stateIndex = 0; stateIndex < seg.nState; stateIndex++)
	{
		for (int f = seg.startTime[stateIndex]; f < seg.endTime[stateIndex]; f++)
		{
			if (relaxBoundary)
			{
				seg.logProbability[reIndex] = LZERO;
				wstring mergedLabel = _state2Ph[seg.stateName[stateIndex]];
				for (vector<wstring>::iterator siter = _ph2States[mergedLabel].begin(); siter != _ph2States[mergedLabel].end(); ++siter)
				{
					double tempScore = inputLikelihood ? convertLogLikelihoodToLogPosterior(scores[f][state2Index[*siter]], state2Index[*siter], dnnModel) : scores[f][state2Index[*siter]];
					seg.logProbability[reIndex] = LAdd(seg.logProbability[reIndex], tempScore);
				}
			}
			else
			{
				double tempScore = inputLikelihood ? convertLogLikelihoodToLogPosterior(scores[f][state2Index[seg.stateName[stateIndex]]], state2Index[seg.stateName[stateIndex]], dnnModel) : scores[f][state2Index[seg.stateName[stateIndex]]];
				seg.logProbability[reIndex] = tempScore;
			}
			reIndex++;
		}
		
	}
}

struct phoneSeg
{
	wstring phLabel;
	int startTime;
	int endTime;
	vector<float> frameScores;
	phoneSeg(wstring _phlabel, int _start, int _endTime) :phLabel(_phlabel),startTime(_start), endTime(_endTime){ frameScores = vector<float>(); };
	phoneSeg(wstring _phlabel, int _start, int _end, vector<float> _score) :phLabel(_phlabel), startTime(_start), endTime(_end), frameScores(_score){};
	float getSegmentScore(bool product){
		float score = 0;
		if (frameScores.empty()) return 0;
		for (int i = 0; i < frameScores.size(); i++) score += product? frameScores[i]: expf(frameScores[i]);
		score /= frameScores.size();
		return score;
	};
	wstring getAlignmentAndScore(bool product=false)
	{
		wstring line = to_wstring(startTime * 100000) + L" " + to_wstring(endTime * 100000) + L" " + phLabel + L" " + to_wstring(getSegmentScore(product));
		return line;
	}
};
wstring stateLabel2PhoneLabel(const wstring &stateLabel, wchar_t sep)
{
	if (stateLabel.find_first_of(sep) != wstring::npos)
		return stateLabel.substr(0, stateLabel.find_first_of(sep));
	else return L"sil";
}
vector<wstring> obtainPhoneLevelScores(const WordDuration &wordRes, bool productScore=false)
{
	wstring wordName = wordRes.wordName;
	vector<phoneSeg> iPhoneSeq;
	int frameIndex = 0;
	for (int k = 0; k < wordRes.nState; k++)
	{
		wstring phLabel = stateLabel2PhoneLabel(wordRes.stateName[k], L'_');
		if (iPhoneSeq.empty() || iPhoneSeq[iPhoneSeq.size() - 1].phLabel.compare(phLabel) != 0)
			iPhoneSeq.push_back(phoneSeg(phLabel, wordRes.startTime[k], wordRes.endTime[k]));
		else
			iPhoneSeq[iPhoneSeq.size() - 1].endTime = wordRes.endTime[k];
		for (int i = wordRes.startTime[k]; i < wordRes.endTime[k]; i++)
		{
			iPhoneSeq[iPhoneSeq.size() - 1].frameScores.push_back(wordRes.logProbability[frameIndex++]);
		}
	}
	
	vector<wstring> phoneRes;
	if ((wordName.compare(L"<s>") == 0 || wordName.compare(L"</s>") == 0) && iPhoneSeq.size() != 1)
	{
		cout << "error in process this file" << endl;
		return vector<wstring>();
	}
	phoneRes.push_back(iPhoneSeq[0].getAlignmentAndScore(productScore) + L" " + wordName);
	for (int i = 1; i < iPhoneSeq.size(); i++)
	{
		if (iPhoneSeq[i].phLabel.compare(L"sil") != 0)
			phoneRes.push_back(iPhoneSeq[i].getAlignmentAndScore(productScore));
	}
	
	return phoneRes;
}


wstring obtainPhoneLevelScoresWithCompetings(wstring& canonicalWord, const vector<WordDuration> &competRes, int competingNumber = 5, bool logScaleValue = true)
{
	// obtain score for each competing phone
	if (competingNumber <= 0) competingNumber = INT_MAX; // all the competings are printed out
	wstring alignedInfo = L"";
	float canonicalWordScore = 0;
	vector<pair<wstring, float>> ph2score;
	float maxScore = INT_MIN;
	wstring mostCompetingAlignInfo = L"";
	for (vector<WordDuration>::const_iterator iter = competRes.begin(); iter != competRes.end(); ++iter)
	{
			
		vector<phoneSeg> iPhoneSeq;
		int frameIndex = 0;
		for (int k = 0; k < iter->nState; k++)
		{
			wstring phLabel = stateLabel2PhoneLabel(iter->stateName[k], L'_');
			if (phLabel.compare(L"sil") == 0 &&(iter->wordName.compare(L"<s>")!=0 && iter->wordName.compare(L"</s>")!=0)) continue;

			if (iPhoneSeq.empty() || iPhoneSeq[iPhoneSeq.size() - 1].phLabel.compare(phLabel) != 0)
				iPhoneSeq.push_back(phoneSeg(phLabel, iter->startTime[k], iter->endTime[k]));
			else
				iPhoneSeq[iPhoneSeq.size() - 1].endTime = iter->endTime[k];
			for (int i = iter->startTime[k]; i < iter->endTime[k]; i++)
			{
				iPhoneSeq[iPhoneSeq.size() - 1].frameScores.push_back(iter->logProbability[frameIndex++]);
			}
		}
		if (iPhoneSeq.empty()) continue;
		if (iPhoneSeq.size() != 1)
		{
			cout << "error: more than two phone segments are detected, not support" << endl;
			return L"";
		}
		float tempScore = iPhoneSeq[0].getSegmentScore(logScaleValue);
		ph2score.push_back(pair<wstring, float>(iPhoneSeq[0].phLabel, tempScore));
		if (tempScore > maxScore) { maxScore = tempScore; mostCompetingAlignInfo = iPhoneSeq[0].getAlignmentAndScore(logScaleValue); }
		if (iter->wordName.compare(canonicalWord) == 0) { alignedInfo = iPhoneSeq[0].getAlignmentAndScore(); canonicalWordScore = iPhoneSeq[0].getSegmentScore(logScaleValue); }
		
	}

	
	wstring resLine = alignedInfo + L" " + canonicalWord + L" " + to_wstring(canonicalWordScore)+L" "+mostCompetingAlignInfo;
	
	std::sort(ph2score.begin(), ph2score.end(), keyValuePair_compare<wstring>);
	vector<wstring> uniquePhones;
	for (vector<pair<wstring, float>>::iterator piter = ph2score.begin(); piter != ph2score.end(); piter++)
	{
		if (uniquePhones.size() > competingNumber) break;
		if (find(uniquePhones.begin(),uniquePhones.end(),piter->first)!=uniquePhones.end()) continue;
		resLine = resLine + L" " + piter->first + L" " + to_wstring(piter->second);
		uniquePhones.push_back(piter->first);
	}
	return resLine;
}


vector<wstring> alignAndScoring(msra::dbn::matrix feat, vector<wstring>& wordlist, bool relaxBoundary = false)
{
	Decode::InitAlignment();
	msra::dbn::matrix pred;
	forwardPropatation(s_pDNNModel, feat, pred); // false for log posterior, true for log likelihood
	size_t featdim = pred.rows();
	size_t numframes = pred.cols();

	double **score = new double*[numframes];
	for (int i = 0; i < numframes; ++i)
	{
		score[i] = new double[featdim];
	}
	msraMatrix2doublearray(pred, score);

	vector<WordDuration> res = Decode::Alignment(wordlist, score, numframes); // silence state in the non-silence phone has been deleted

	if (res.empty()) { cout << "force alignment error" << endl;	return vector<wstring>(); }

	for (int i = 0; i < numframes; i++)
	{
		for (int j = 0; j < featdim; j++)
		{
			score[i][j] = -score[i][j]; // convert to the orinal value 
		}
	}
	for (vector<WordDuration>::iterator iter = res.begin(); iter != res.end();++iter)
		obtainRelaxedStateScores(*iter, state2Ph, ph2States, s_pDNNModel, score, relaxBoundary, divbyprior);

	vector<wstring> PhoneResults;
	for (vector<WordDuration>::iterator iter = res.begin(); iter != res.end(); ++iter)
	{
		vector<wstring> tempWordResult = obtainPhoneLevelScores(*iter); // obtain score and alignment for each phone
		for (int i = 0; i < tempWordResult.size(); i++)  PhoneResults.push_back(tempWordResult[i]);
	}

	for (int i = 0; i < numframes; ++i)
	{
		delete[featdim]score[i];
		score[i] = NULL;
	}
	delete[numframes]score;
	res.clear();
	Decode::DisposeTransGraphObject();
	return PhoneResults;

}

vector<wstring> alignWithCompetingPhonesAndScoring(msra::dbn::matrix feat, vector<wstring>& wordlist, int competingPhNumber=5, bool fixPhoneBoundary = true)
{
	Decode::InitAlignment();
	msra::dbn::matrix pred;
	forwardPropatation(s_pDNNModel, feat, pred); // false for log posterior, true for log likelihood
	size_t featdim = pred.rows();
	size_t numframes = pred.cols();

	double **score = new double*[numframes];
	for (int i = 0; i < numframes; ++i)
	{
		score[i] = new double[featdim];
	}
	msraMatrix2doublearray(pred, score);

	vector<vector<WordDuration>> competAlignRes;
	if (fixPhoneBoundary)
		competAlignRes = Decode::CompeteAlignment_fixBoundary(wordlist, score, numframes);
	else
		competAlignRes = Decode::CompeteAlignment_relaxBoundary(wordlist, score, numframes);

	if (competAlignRes.empty()) { cout << "force alignment error" << endl;	return vector<wstring>(); }

	for (int i = 0; i < numframes; i++)
	{
		for (int j = 0; j < featdim; j++)
		{
			score[i][j] = -score[i][j]; // convert to the orinal value 
		}
	}

	vector<wstring> PhoneResults(competAlignRes.size(), L"");

	concurrency::parallel_for(size_t(0), competAlignRes.size(), [&](size_t i)
	{
		for (vector<WordDuration>::iterator iter = competAlignRes[i].begin(); iter != competAlignRes[i].end(); ++iter)
			obtainRelaxedStateScores(*iter, state2Ph, ph2States, s_pDNNModel, score, true, divbyprior);

		wstring tempLine = obtainPhoneLevelScoresWithCompetings(wordlist[i], competAlignRes[i], competingPhNumber, true); // obtain score and alignment for each phone
		PhoneResults[i]=tempLine;
	});

	for (int i = 0; i < numframes; ++i)
	{
		delete[featdim]score[i];
		score[i] = NULL;
	}
	delete[numframes]score;
	competAlignRes.clear();
	Decode::DisposeTransGraphObject();
	return PhoneResults;

}



// interfaces for online service
int EvaluateStrictBoundary(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::dbn::matrix pred;
	msra::asr::htkfeatreader reader;
	reader.readFromArray(mfcdata, pos, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring
	vector<wstring> PhoneResults = alignAndScoring(feat, wordlist, false);
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}

int EvaluateRelaxBoundary(byte* mfcdata,  wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::dbn::matrix pred;
	msra::asr::htkfeatreader reader;
	reader.readFromArray(mfcdata, pos, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring
	vector<wstring> PhoneResults = alignAndScoring(feat, wordlist, true);
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}
 
void LoadModel(wchar_t *szDataFolder, bool _likelihoodforalign,int cores)
{
	wstring strDataFolder(szDataFolder);
	wstring szDNNModelPath = strDataFolder + L"\\dnnmodel";
	wstring szHMMModelPath = strDataFolder + L"\\hmmmodel";
	wstring szTiedList = strDataFolder + L"\\tiedlist";
	wstring szEvaDictPath = strDataFolder + L"\\evadict";
	wstring szErrDictPath = strDataFolder + L"\\errdict";
	wstring szStateList = strDataFolder + L"\\statelist";

	unLoadModel();

	divbyprior = _likelihoodforalign;
	if (cores<=0) cores = msra::parallel::determine_num_cores();
	msra::parallel::set_cores(cores);
	std::cout << "number of cores:" << cores << endl;
	s_pDNNModel = new msra::dbn::model(szDNNModelPath);

	s_pDNNModel->entercomputation(0);

	state2Index = State2IndexLoad(szStateList);
	ph2States = Ph2StatesLoad(szStateList);
	state2Ph = State2PhoneLoad(szStateList);

	Decode::InitTriPhone2(szHMMModelPath.c_str(), szTiedList.c_str(), szEvaDictPath.c_str(), szErrDictPath.c_str());
	WordSet::setInstanceID(0);
}
void unLoadModel()
 {
	 delete s_pDNNModel;
	 s_pDNNModel = NULL;
	 state2Index.clear();
	 ph2States.clear();
	 state2Ph.clear();

	 PhoneSet::freeInstance();
	 WordSet::freeInstances();
	 StateSet::Clear();
 }

// extra interface for research experiments

struct wordToken{
	wstring word;
	int startTime;
	int endTime;
};

int LikelihoodWithGivenPhoneBoundary(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	WordSet::setInstanceID(0);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featfile);
	reader.read(path, featkind, sampperiod, feat);
	// load word transcription for force alignment

	vector<wordToken> wordlist; // from text to word list
	assert(wordsLength % 3 == 0);
	for (int i = 0; i < wordsLength/3; i++) {
		wordToken iword;
		iword.word = wstring(words[3 * i + 2]);
		iword.startTime = stoi((wstring(words[3 * i])).c_str())/100000;
		iword.endTime = stoi((wstring(words[3 * i+1])).c_str())/100000;
		wordlist.push_back(iword);
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring

	Decode::InitAlignment();
	msra::dbn::matrix pred;
	forwardPropatation(s_pDNNModel, feat, pred); // false for log posterior, true for log likelihood
	size_t featdim = pred.rows();
	size_t numframes = pred.cols();

	double **score = new double*[numframes];
	for (int i = 0; i < numframes; ++i)
	{
		score[i] = new double[featdim];
	}
	msraMatrix2doublearray(pred, score);

	vector<WordDuration> res;
	for (int i = 1; i < wordlist.size() - 1; i++)
	{
		WordDuration ires = Decode::AlignmentSingleWord(WordSet::getInstance()->Get(wordlist[i - 1].word).front()->pronunciation.back(), 
																WordSet::getInstance()->Get(wordlist[i].word).front(),
																WordSet::getInstance()->Get(wordlist[i + 1].word).front()->pronunciation.front(),
																score,
																wordlist[i].startTime, wordlist[i].endTime);
		res.push_back(ires);
	}

	if (res.empty()) { cout << "force alignment error" << endl;	return 0; }

	for (int i = 0; i < numframes; i++)
	{
		for (int j = 0; j < featdim; j++)
		{
			score[i][j] = -score[i][j]; // convert to the orinal value 
		}
	}

	 for (vector<WordDuration>::iterator iter = res.begin(); iter != res.end(); ++iter)
		obtainRelaxedStateScores(*iter, state2Ph, ph2States, s_pDNNModel, score, true, divbyprior);

	vector<wstring> PhoneResults;
	for (vector<WordDuration>::iterator iter = res.begin(); iter != res.end(); ++iter)
	{
		vector<wstring> tempWordResult = obtainPhoneLevelScores(*iter,true); // obtain score and alignment for each phone
		for (int i = 0; i < tempWordResult.size(); i++)  PhoneResults.push_back(tempWordResult[i]);
	}

	for (int i = 0; i < numframes; ++i)
	{
		delete[featdim]score[i];
		score[i] = NULL;
	}
	delete[numframes]score;
	res.clear();
	Decode::DisposeTransGraphObject();
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}

int ForceAlignWithCanonicalWords(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	WordSet::setInstanceID(0);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featfile);
	reader.read(path, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring

	Decode::InitAlignment();
	msra::dbn::matrix pred;
	forwardPropatation(s_pDNNModel, feat, pred); // false for log posterior, true for log likelihood
	size_t featdim = pred.rows();
	size_t numframes = pred.cols();

	double **score = new double*[numframes];
	for (int i = 0; i < numframes; ++i)
	{
		score[i] = new double[featdim];
	}
	msraMatrix2doublearray(pred, score);

	vector<WordDuration> res = Decode::Alignment(wordlist, score, numframes); // silence state in the non-silence phone has been deleted
	vector<wstring> stateResults;
	// obtain alignment result
	for (vector<WordDuration>::iterator iter = res.begin(); iter < res.end(); ++iter)
	{
		for (int s = 0; s < iter->nState; s++)
		{
			int startTime = iter->startTime[s] * 100000;
			int endTime = iter->endTime[s] * 100000;
			wstring line = to_wstring(startTime) + L" " + to_wstring(endTime) + L" " + iter->stateName[s];
			if (s == 0) line = line + L" " + iter->wordName;
			stateResults.push_back(line);
		}
	}
	for (vector<wstring>::iterator iter = stateResults.begin(); iter != stateResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	stateResults.clear();
	return nCopyLength;
}

int ForceAlignWithCanonicalWords_Memo(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	WordSet::setInstanceID(0);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	reader.readFromArray(mfcdata, pos, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring

	Decode::InitAlignment();
	msra::dbn::matrix pred;
	forwardPropatation(s_pDNNModel, feat, pred); // false for log posterior, true for log likelihood
	size_t featdim = pred.rows();
	size_t numframes = pred.cols();

	double **score = new double*[numframes];
	for (int i = 0; i < numframes; ++i)
	{
		score[i] = new double[featdim];
	}
	msraMatrix2doublearray(pred, score);

	vector<WordDuration> res = Decode::Alignment(wordlist, score, numframes); // silence state in the non-silence phone has been deleted
	vector<wstring> stateResults;
	// obtain alignment result
	for (vector<WordDuration>::iterator iter = res.begin(); iter < res.end(); ++iter)
	{
		for (int s = 0; s < iter->nState; s++)
		{
			int startTime = iter->startTime[s] * 100000;
			int endTime = iter->endTime[s] * 100000;
			wstring line = to_wstring(startTime) + L" " + to_wstring(endTime) + L" " + iter->stateName[s];
			if (s == 0) line = line + L" " + iter->wordName;
			stateResults.push_back(line);
		}
	}
	for (vector<wstring>::iterator iter = stateResults.begin(); iter != stateResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	stateResults.clear();
	return nCopyLength;
}

int EvaluateWithCanonicalWords(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength)
{
	WordSet::setInstanceID(0);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::dbn::matrix pred;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featfile);
	reader.read(path, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring
	vector<wstring> PhoneResults = alignAndScoring(feat, wordlist, true);
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}

int EvaluateWithCompetingPhones(wchar_t *featfile, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength, int competingPhoneNumber,bool fixPhoneBoundary)
{
	WordSet::setInstanceID(1);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::dbn::matrix pred;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featfile);
	reader.read(path, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring
	vector<wstring> PhoneResults = alignWithCompetingPhonesAndScoring(feat, wordlist, competingPhoneNumber, fixPhoneBoundary);
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}

int EvaluateWithCompetingPhonesMemoStream(byte* mfcdata, wchar_t* szOutputData, int nMaxCount, wchar_t** words, int wordsLength, int competingPhoneNumber, bool fixPhoneBoundary)
{
	WordSet::setInstanceID(1);
	std::wstring strOutput;
	// got mfc data from file
	int posstart = 0;
	int *pos = &posstart;
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::dbn::matrix pred;
	msra::asr::htkfeatreader reader;
	reader.readFromArray(mfcdata, pos, featkind, sampperiod, feat);
	// load word transcription for force alignment
	vector<wstring> wordlist; // from text to word list
	for (int i = 0; i < wordsLength; i++) {
		wordlist.push_back(wstring(words[i]));
	}
	if (wordlist.empty()){ cout << "No word transcription provided" << endl; return -1; }
	// do alignment and scoring
	vector<wstring> PhoneResults = alignWithCompetingPhonesAndScoring(feat, wordlist, competingPhoneNumber, fixPhoneBoundary);
	for (vector<wstring>::iterator iter = PhoneResults.begin(); iter != PhoneResults.end(); ++iter)
		strOutput += *iter + L"\r\n";
	int nCopyLength = min(nMaxCount - 1, strOutput.size());
	strOutput.copy(szOutputData, nCopyLength, 0);
	PhoneResults.clear();
	return nCopyLength;
}
