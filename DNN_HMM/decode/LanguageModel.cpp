#include "LanguageModel.h"
#include "..\unicode.h"

const double LanguageModel::INF = 1e20;

LanguageModel::LanguageModel(double exponent, double wordScore)
{
	this->exponent = exponent;
	this->wordScore = wordScore;
	nWord = WordSet::getInstance()->Size();
	unigram_p = new double[nWord];
	unigram_b = new double[nWord];
	bigram = new double[nWord * nWord];
	for(size_t i = 0; i != nWord * nWord; ++i)
		bigram[i] = INF;
}

void LanguageModel::AddBigram(const wstring& s, const wstring& t, double dist)
{
	vector<Word*> wordList1 = WordSet::getInstance()->Get(s, false);
	vector<Word*> wordList2 = WordSet::getInstance()->Get(t, false);
	for(size_t i = 0; i != wordList1.size(); ++i)
		for(size_t j = 0; j != wordList2.size(); ++j)
			bigram[wordList1[i]->id * nWord + wordList2[j]->id] = dist * exponent - wordScore;
}

void LanguageModel::AddUnigram_p(const wstring& s, double p)
{
	vector<Word*> wordList = WordSet::getInstance()->Get(s, false);
	for(size_t i = 0; i != wordList.size(); ++i)
		unigram_p[wordList[i]->id] = p * exponent;
}

void LanguageModel::AddUnigram_b(const wstring& s, double b)
{
	vector<Word*> wordList = WordSet::getInstance()->Get(s, false);
	for(size_t i = 0; i != wordList.size(); ++i)
		unigram_b[wordList[i]->id] = b * exponent - wordScore;
}

void LanguageModel::ReadLM(const wchar_t *lmFile)
{
	cerr << "Reading language model ..." << endl;
	FILE *fp = _wfopen(lmFile, L"r");
	if (fp == NULL)
	{
		cerr << "Language model file not found: \"" << lmFile << "\"\n";
		system("PAUSE");
	}
	char s[100],t[100];
	double dist;
	while(true)
	{
		fgets(s, 100, fp);
		if (strncmp(s, "\\data\\", 6) == 0)
			break;
	}
	while(true)
	{
		fscanf(fp, "%s", s);
		if (strcmp(s, "\\1-grams:") == 0)
			break;
	}
	while(true)
	{
		fscanf(fp, "%s", s);
		if (strcmp(s, "\\2-grams:") == 0)
			break;
		dist = atof(s);
		fscanf(fp, "%s", s);
		AddUnigram_p(char2wstring(s), -dist);
		fgets(t, 100, fp);
		if (sscanf(t, "%lf", &dist) != 1)
			dist = -9999.99;
		AddUnigram_b(char2wstring(s), -dist);
	}
	while(true)
	{
		fscanf(fp, "%s", s);
		if (strcmp(s, "\\end\\") == 0)
			break;
		double dist = atof(s);
		fscanf(fp, "%s", s);
		fscanf(fp, "%s", t);
		AddBigram(char2wstring(s), char2wstring(t), -dist);
	}
	fclose(fp);
	// pre-processing all the questions
	double temp = 0.1 * INF;
	for(size_t i = 0; i != nWord; ++i)
		for(size_t j = 0; j != nWord; ++j)
			if (bigram[i * nWord + j] > temp)
				bigram[i * nWord + j] = unigram_b[i] + unigram_p[j];
}

double LanguageModel::Dist(Word *s, Word *t)
{
	if (s == NULL || t == NULL)
		return 0.0;
	return bigram[s->id * nWord + t->id];
}

double LanguageModel::GetP(Word *s)
{
	if (s == NULL)
		return 0.0;
	return unigram_p[s->id];
}

LanguageModel::~LanguageModel(void)
{
	delete [] bigram;
	delete [] unigram_p;
	delete [] unigram_b;
}
