#pragma once

#include "phone.h"

struct Word // each pronunciation of a word will be an object here
{
	wstring name;
	int id;
	vector<Phone*> pronunciation;
	bool output; // if the word will be output when recognition
};

class WordSet
{
public:
	static WordSet* getInstance();
	static void freeInstances();
	static bool setInstanceID(int id);

	WordSet()
	{
		m_nWord = 0;
	}

	~WordSet()
	{

		m_mapWordPointer.clear();

		while (!m_vecWordList.empty())
		{
			delete m_vecWordList.back();
			m_vecWordList.pop_back();
		}
	}
private:
	int m_nWord; // number of words
	multimap<wstring, Word*> m_mapWordPointer;
	vector<Word*> m_vecWordList;
public:
	int Size(void);
	Word* Get(int id);
	vector<Word*> Get(wstring name, bool hint_error = true); // get all the objects by the word name
	void Read(const wchar_t *dictFile);
};

struct WordDuration
{
	wstring wordName;
	int nState; // number of state belong to the word
	vector<int> startTime, endTime; // start and end time of each state belong to the word
	vector<double> logProbability; // log probability of each frame belong to the word
	vector<wstring> stateName; // state list belong to the word
};

struct CompeteWordDuration
{
	vector<wstring> wordName;
	vector<vector<double> > logProbability; // log probability of each frame belong to the word
	vector<vector<wstring>> stateNames;
	vector<vector<int>> startTimes,endTimes;
	vector<int> nStates;
	vector<WordDuration> convertToWordDurationContainer()
	{
		vector<WordDuration> res;
		for (int i = 0; i < wordName.size(); i++)
		{
			WordDuration temp;
			temp.wordName = wordName[i];
			temp.logProbability = logProbability[i];
			temp.stateName = stateNames[i];
			temp.startTime = startTimes[i];
			temp.endTime = endTimes[i];
			temp.nState = nStates[i];
			res.push_back(temp);
		}
		return res;
	};
};