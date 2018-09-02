#include "decode.h"
//#define TONAL_PHONEME_ANALYSIS

__declspec(thread) LanguageModel *Decode::lModel = NULL;
bool Decode::triPhone = false;
__declspec(thread) bool Decode::recogniton = false;
__declspec(thread) TransGraph *Decode::decoder = NULL;

void Decode::InitMonoPhone(const wchar_t *modelFile, const wchar_t *dictFile)
{
	triPhone = false;
	PhoneSet::getInstance()->ReadModel(modelFile);
	WordSet::getInstance()->Read(dictFile);
}

void Decode::InitTriPhone(const wchar_t *modelFile, const wchar_t *tiedFile, const wchar_t *dictFile)
{
	triPhone = true;
	PhoneSet::getInstance()->ReadModel(modelFile, true);
	PhoneSet::getInstance()->ReadTied(tiedFile);
	WordSet::getInstance()->Read(dictFile);
}

void Decode::InitTriPhone2(const wchar_t *modelFile, const wchar_t *tiedFile, const wchar_t *evadictFile, const wchar_t *errdictFile)
{
	triPhone = true;
	PhoneSet::getInstance()->ReadModel(modelFile, true);
	PhoneSet::getInstance()->ReadTied(tiedFile);
	WordSet::setInstanceID(0);
	WordSet::getInstance()->Read(evadictFile);
	WordSet::setInstanceID(1);
	WordSet::getInstance()->Read(errdictFile);
}

void Decode::InitLanguageModelFile(const wchar_t *lmFile, double exponent, double wordScore) // Optional
{
	lModel = new LanguageModel(exponent, wordScore);
	lModel->ReadLM(lmFile);
}

void Decode::InitRecognition(int pruning)
{
	cerr << "Initializing recognition ..." << endl;
	recogniton = true;
	if (pruning == 0)
	{
		if (!triPhone)
			decoder = new MonophoneRecognition(lModel);
		else decoder = new TriphoneRecognition(lModel);
	}
	else
	{
		if (!triPhone)
			decoder = new MonophoneRecognition(lModel, pruning);
		else decoder = new TriphoneRecognition(lModel, pruning);
	}
}

void Decode::InitAlignment(void)
{
	cerr << "Initializing alignment ..." << endl;
	recogniton = false;
	if (!triPhone)
		decoder = new MonophoneAlignment();
	else decoder = new TriphoneAlignment();
}
	
void Decode::Recognition(double **score, int length, FILE *out)
{
	decoder->Process(score, length, out);
}

void Decode::Alignment(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel)
{
	decoder->Process(wordList, score, length, out, stateLevel);
}

vector<WordDuration> Decode::Alignment(vector<wstring> wordList, double **score, int length)
{
	return decoder->GetResult(wordList, score, length);
}

vector<CompeteWordDuration> Decode::CompeteAlignment(vector<wstring> wordList, double **score, int length)
{
	int nWord = WordSet::getInstance()->Size(), m = wordList.size();
	vector<CompeteWordDuration> res;
	res.resize(m);
	Concurrency::parallel_for(1, m - 1, [&] (int i)
	{
		TransGraph *decoder;
		if (!triPhone)
			decoder = new MonophoneAlignment();
		else decoder = new TriphoneAlignment();
		vector<wstring> tempList = wordList;
		vector<WordDuration> tempRes;
		for(int j = 0; j < nWord; ++j)
		{
		   if(isdigit(tempList[i][tempList[i].length()-1])!=isdigit(WordSet::getInstance()->Get(j)->name[WordSet::getInstance()->Get(j)->name.length()-1])) 
			   continue;
			tempList[i] = WordSet::getInstance()->Get(j)->name;
			tempRes = decoder->GetResult(tempList, score, length);

			res[i].wordName.push_back(tempList[i]);
			res[i].logProbability.push_back(tempRes[i].logProbability);
			res[i].nStates.push_back(tempRes[i].nState);
			res[i].stateNames.push_back(tempRes[i].stateName);
			res[i].startTimes.push_back(tempRes[i].startTime);
			res[i].endTimes.push_back(tempRes[i].endTime);
		}
	});
	return res;
}

vector<vector<WordDuration>> Decode::CompeteAlignment_relaxBoundary(vector<wstring> wordList, double **score, int length)
{
	int nWord = WordSet::getInstance()->Size(), m = wordList.size();
	vector<vector<WordDuration>> res;
	res.resize(m);
	Concurrency::parallel_for(1, m - 1, [&](int i)
	{
		TransGraph *decoder;
		if (!triPhone)
			decoder = new MonophoneAlignment();
		else decoder = new TriphoneAlignment();
		vector<wstring> tempList = wordList;
		vector<WordDuration> tempRes;
		for (int j = 0; j < nWord; ++j)
		{
			wstring competName = WordSet::getInstance()->Get(j)->name;
			if (competName.length() == 0) continue;
			if (competName.compare(L"<s>") == 0 || competName.compare(L"</s>") == 0) continue;
#ifdef	ONLY_SAME_ENDING
			if (isdigit(tempList[i][tempList[i].length() - 1]) != isdigit(competName[competName.length() - 1]))
				continue;
#endif
			tempList[i] = WordSet::getInstance()->Get(j)->name;
			tempRes = decoder->GetResult(tempList, score, length);
			res[i].push_back(tempRes[i]);
		}
	});
	return res;
}

WordDuration Decode::AlignmentSingleWord(Phone *pre, Word *word, Phone *next, double **score, int startFrame, int endFrame)
{
	return decoder->DecodeSingleWord(pre, word, next, score, startFrame, endFrame);
}


vector<vector<WordDuration>> Decode::CompeteAlignment_fixBoundary(vector<wstring> wordList, double **score, int length)
{
#ifdef ERROR_PATTERN_ALIGN
	WordSet::setInstanceID(0);
	Decode::InitAlignment();
#endif
	vector<WordDuration> temp = Alignment(wordList, score, length);
#ifdef ERROR_PATTERN_ALIGN
	WordSet::setInstanceID(1);
	Decode::InitAlignment();
#endif
	vector<vector<WordDuration>> res;
	res.resize(wordList.size());
	for (size_t i = 0; i < temp.size();i++)
	{// parallel for each word
		vector<WordDuration> iWordRes;
		if (i == 0 || i == temp.size() - 1)
		{
			iWordRes.push_back(temp[i]);
		}
		else
		{
			for (int j = 0; j < WordSet::getInstance()->Size(); ++j)
			{
				wstring competName = WordSet::getInstance()->Get(j)->name;
				if (competName.length() == 0) continue;
				if (competName.compare(L"<s>") == 0 || competName.compare(L"</s>") == 0) continue;
#ifdef ONLY_SAME_ENDING
				if (isdigit(temp[i].wordName[temp[i].wordName.length() - 1]) != isdigit(competName[competName.length() - 1])) continue;
#endif
					WordDuration wd = AlignmentSingleWord(WordSet::getInstance()->Get(temp[i - 1].wordName).front()->pronunciation.back(),
						WordSet::getInstance()->Get(j),
						WordSet::getInstance()->Get(temp[i + 1].wordName).front()->pronunciation.front(),
						score, temp[i].startTime.front(), temp[i].endTime.back());
					iWordRes.push_back(wd);
				
			}
		}
		res[i] = iWordRes;
	}
	return res;
}

#ifndef TONAL_PHONEME_ANALYSIS 

vector<CompeteWordDuration> Decode::CompeteAlignment2(vector<wstring> wordList, double **score, int length)
{
	vector<WordDuration> temp = Alignment(wordList, score, length);
	vector<CompeteWordDuration> res;
	res.resize(wordList.size());
	for(int i = 1; i + 1 < temp.size(); ++i)
	{
		
		for(int j = 0; j < WordSet::getInstance()->Size(); ++j)
		{
			WordDuration wd = AlignmentSingleWord(WordSet::getInstance()->Get(temp[i - 1].wordName).front()->pronunciation.back(),
				WordSet::getInstance()->Get(j),
				WordSet::getInstance()->Get(temp[i + 1].wordName).front()->pronunciation.front(),
				score, temp[i].startTime.front(), temp[i].endTime.back());
			res[i].wordName.push_back(WordSet::getInstance()->Get(j)->name);
			res[i].logProbability.push_back(wd.logProbability);
			res[i].nStates.push_back(wd.nState);
			res[i].stateNames.push_back(wd.stateName);
			res[i].startTimes.push_back(wd.startTime);
			res[i].endTimes.push_back(wd.endTime);
		}
	}
	return res;
}

#else
vector<CompeteWordDuration> Decode::CompeteAlignment2(vector<wstring> wordList, double **score, int length)
{
	bool vowel=true;
	bool tagisvowel=true;
	vector<WordDuration> temp = Alignment(wordList, score, length);
	vector<CompeteWordDuration> res;
	res.clear();
	res.resize(wordList.size());
	for(int i = 1; i + 1 < temp.size(); ++i)
	{
		res[i].wordName.push_back(temp[i].wordName);
		res[i].logProbability.push_back(temp[i].logProbability);
		res[i].nStates.push_back(temp[i].nState);
		res[i].stateNames.push_back(temp[i].stateName);
		res[i].startTimes.push_back(temp[i].startTime);
		res[i].endTimes.push_back(temp[i].endTime);
		if(!isdigit(temp[i].wordName[temp[i].wordName.length()-1]))
		{
			vowel=false;
		}
		else vowel=true;
	//	const wstring basep=temp[i].wordName.substr(0,temp[i].wordName.length()-1);
		for(int j = 0; j < WordSet::getInstance()->Size(); ++j)
		{
			if(wcscmp(WordSet::getInstance()->Get(j)->name.c_str(),temp[i].wordName.c_str())==0)
			{
			  continue;
			}
			if(!isdigit(WordSet::getInstance()->Get(j)->name[WordSet::getInstance()->Get(j)->name.length()-1]))
			{	
				tagisvowel=false;
			}
			else  tagisvowel=true;

			if(vowel!=tagisvowel) continue;

			WordDuration wd = AlignmentSingleWord(WordSet::getInstance()->Get(temp[i - 1].wordName).front()->pronunciation.back(),
				WordSet::getInstance()->Get(j),
				WordSet::getInstance()->Get(temp[i + 1].wordName).front()->pronunciation.front(),
				score, temp[i].startTime.front(), temp[i].endTime.back());
			res[i].wordName.push_back(WordSet::getInstance()->Get(j)->name);
			res[i].logProbability.push_back(wd.logProbability);
			res[i].nStates.push_back(wd.nState);
			res[i].stateNames.push_back(wd.stateName);
			res[i].startTimes.push_back(wd.startTime);
			res[i].endTimes.push_back(wd.endTime);
		}
	}
	return res;
}
#endif

void Decode::DisposeTransGraphObject()
{
	if (decoder != NULL)
	{
		decoder->ReleaseAllofSource();
		delete decoder;
		decoder = NULL;
	}

	if (lModel != NULL)
	{
		delete lModel;
		lModel = NULL;
	}
}