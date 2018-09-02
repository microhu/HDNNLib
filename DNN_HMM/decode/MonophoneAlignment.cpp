#include "MonophoneAlignment.h"

void MonophoneAlignment::PushWord(wstring name)
	// state list, stay probabilty list, list length, the word
{
	Node *p = endNode;
	Node *q = CreateNode();
	vector<Word*> wordList = WordSet::getInstance()->Get(name);
	for(vector<Word*>::const_iterator
		iter = wordList.begin(); iter != wordList.end(); ++iter)
		AddWord(*iter, p, q);
	endNode = q;
}

void MonophoneAlignment::Init(void)
{
	while(!nodeList.empty())
	{
		delete nodeList.back();
		nodeList.pop_back();
	}
	while(!transList.empty())
	{
		delete transList.back();
		transList.pop_back();
	}
	while(!skipList.empty())
	{
		delete skipList.back();
		skipList.pop_back();
	}
	root = endNode = CreateNode();
}

void MonophoneAlignment::Process(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel)
{
	Init();
	int i, j, k;
	for(i = 0; i < wordList.size(); ++i)
		PushWord(wordList[i]);
	endNode->end = true;
	Recognition(score, length, root);
	endNode->end = false;
	if (!stateLevel)
	{
		int t = 0;
		for(i = 0; i < WordAns.size(); ++i)
		{
			fprintf(out, "%d %d %s\n", t * 100000, WordAns[i].second * 100000,
				WordAns[i].first->name.c_str());
			t = WordAns[i].second;
		}
	}
	else
	{
		Word *word = WordAns[k = 0].first;
		for(i = 0; i < StateAns.size();)
		{
			j = i;
			while(j < StateAns.size() && StateAns[i] == StateAns[j])
				++j;
			fprintf(out, "%d %d %s", i * 100000, j * 100000, StateSet::GetName(StateAns[i]).c_str());
			if (word != NULL)
			{
				fprintf(out, " %s", word->name.c_str());
				word = NULL;
			}
			fprintf(out, "\n");
			if (k + 1 < WordAns.size() && WordAns[k].second == j)
				word = WordAns[++k].first;
			i = j;
		}
	}
	fprintf(out, ".\n");
}