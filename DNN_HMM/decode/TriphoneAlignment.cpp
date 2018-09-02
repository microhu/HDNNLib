#include "TriphoneAlignment.h"

void TriphoneAlignment::Init(void)
{
	if (releaseCount == 1)
	{
		while (!nodeList.empty())
		{
			delete nodeList.back();
			nodeList.pop_back();
		}
		while (!transList.empty())
		{
			delete transList.back();
			transList.pop_back();
		}
		while (!skipList.empty())
		{
			delete skipList.back();
			skipList.pop_back();
		}

		if (!WordAns.empty())
		{
			WordAns.clear();
		}

		if (!StateAns.empty())
		{
			StateAns.clear();
		}
	}

	releaseCount--;
}

void TriphoneAlignment::Process(vector<wstring> sentence, double **score, int length, FILE *out, bool stateLevel)
{
	int i, j, k, t, n = length, m = sentence.size();
	Init();
	int nPhone = PhoneSet::getInstance()->Size();
	int sil = PhoneSet::getInstance()->Get(L"sil")->id;
	Node ***skipNode = new Node** [sentence.size() + 1];
	vector<Word*> *wordList = new vector<Word*>[sentence.size()];
	for(i = 0; i <= m; ++i)
	{
		skipNode[i] = new Node* [nPhone * nPhone];
		for(j = 0; j < nPhone * nPhone; ++j)
			skipNode[i][j] = NULL;
		if (i < m)
			wordList[i] = WordSet::getInstance()->Get(sentence[i]);
	}
	for(i = 0; i <= m; ++i)
	{
		if (i == 0 || i == m)
		{
			skipNode[i][sil * nPhone + sil] = CreateNode();
			continue;
		}
		for(vector<Word*>::const_iterator
			a = wordList[i - 1].begin(); a != wordList[i - 1].end(); ++a)
			for(vector<Word*>::const_iterator
				b = wordList[i].begin(); b != wordList[i].end(); ++b)
			{
				int t = (*a)->pronunciation.back()->id * nPhone 
						+ (*b)->pronunciation.front()->id;
					if (skipNode[i][t] == NULL)
						skipNode[i][t] = CreateNode();
			}
	}
	for(i = 0; i < m; ++i)
		for(vector<Word*>::const_iterator
			iter = wordList[i].begin(); iter != wordList[i].end(); ++iter)
		{
			Word *word = *iter;
			for(j = 0; j < nPhone; ++j)
				for(k = 0; k < nPhone; ++k)
				{
					Node *p = skipNode[i][j * nPhone + word->pronunciation.front()->id];
					Node *q = skipNode[i + 1][word->pronunciation.back()->id * nPhone + k];
					if (p != NULL && q != NULL)
						AddWord(PhoneSet::getInstance()->Get(j), word, PhoneSet::getInstance()->Get(k), p, q);
				}
		}
	t = sil * nPhone + sil;
	Node *start = skipNode[0][t], *end = skipNode[m][t];
	for(i = 0; i <= m; ++i)
		delete [] skipNode[i];
	delete [] skipNode;
	delete [] wordList;
	end->end = true;
	Recognition(score, n, start);
	end->end = false;
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

vector<WordDuration> TriphoneAlignment::GetResult(vector<wstring> sentence, double **score, int length)
{
	int i, j, k, t, n = length, m = sentence.size();
	//Init();
	releaseCount++;
	int nPhone = PhoneSet::getInstance()->Size(), sil = PhoneSet::getInstance()->Get(L"sil")->id;
	Node ***skipNode = new Node** [sentence.size() + 1];
	vector<Word*> *wordList = new vector<Word*>[sentence.size()];
	for(i = 0; i <= m; ++i)
	{
		skipNode[i] = new Node* [nPhone * nPhone];
		for(j = 0; j < nPhone * nPhone; ++j)
			skipNode[i][j] = NULL;
		if (i < m)
			wordList[i] = WordSet::getInstance()->Get(sentence[i]);
	}
	for(i = 0; i <= m; ++i)
	{
		if (i == 0 || i == m)
		{
			skipNode[i][sil * nPhone + sil] = CreateNode();
			continue;
		}
		for(vector<Word*>::const_iterator
			a = wordList[i - 1].begin(); a != wordList[i - 1].end(); ++a)
			for(vector<Word*>::const_iterator
				b = wordList[i].begin(); b != wordList[i].end(); ++b)
			{
				int t = (*a)->pronunciation.back()->id * nPhone 
						+ (*b)->pronunciation.front()->id;
					if (skipNode[i][t] == NULL)
						skipNode[i][t] = CreateNode();
			}
	}
	for(i = 0; i < m; ++i)
		for(vector<Word*>::const_iterator
			iter = wordList[i].begin(); iter != wordList[i].end(); ++iter)
		{
			Word *word = *iter;
			for(j = 0; j < nPhone; ++j)
				for(k = 0; k < nPhone; ++k)
				{
					Node *p = skipNode[i][j * nPhone + word->pronunciation.front()->id];
					Node *q = skipNode[i + 1][word->pronunciation.back()->id * nPhone + k];
					if (p != NULL && q != NULL)
						AddWord(PhoneSet::getInstance()->Get(j), word, PhoneSet::getInstance()->Get(k), p, q, true);
				}
		}
	t = sil * nPhone + sil;
	Node *start = skipNode[0][t], *end = skipNode[m][t];
	for(i = 0; i <= m; ++i)
		delete [] skipNode[i];
	delete [] skipNode;
	delete [] wordList;
	end->end = true;

	// Will fill content into WordAns in Recognition()
	// WordAns and StateAns should be used to caculate score
	// But, WordAns and StateAns will not have values always.
	// If script and wave not match, they will be empty.
	Recognition(score, n, start);
	end->end = false;

	vector<WordDuration> res;
	if (!WordAns.empty())
	{
		Word *word = WordAns[k = 0].first;
		for (i = 0; i < StateAns.size();)
		{
			j = i;
			while (j < StateAns.size() && StateAns[i] == StateAns[j])
				++j;
			if (word != NULL)
			{
				WordDuration wd;
				wd.nState = 0;
				wd.wordName = word->name.c_str();
				res.push_back(wd);
				word = NULL;
			}
			++res.back().nState;
			res.back().startTime.push_back(i);
			res.back().endTime.push_back(j);
			res.back().stateName.push_back(StateSet::GetName(StateAns[i]));
			if (k + 1 < WordAns.size() && WordAns[k].second == j)
				word = WordAns[++k].first;
			i = j;
		}
		for (i = 0; i != res.size(); ++i)
		for (j = res[i].startTime.front(); j < res[i].endTime.back(); ++j)
			res[i].logProbability.push_back(-score[j][StateAns[j]]);
	}
		//add some code to get the phone level results
  /*
	    vector<WordDuration> PResults;
	string curstatelabel=res[0].stateName[0].substr(0,res[0].stateName[0].find("_"));
	
		WordDuration ph;
		ph.logProbability.clear();
		ph.startTime.clear();
		ph.endTime.clear();
		ph.stateName.clear();
		ph.nState=0;
	 for(int i=0;i<res.size();i++)
	 {
		 curstatelabel=res[i].stateName[0].substr(0,res[i].stateName[0].find("_"));
		 for(int j=0;j<res[i].nState;j++)
		 {
			string templabel=res[i].stateName[j].substr(0,res[i].stateName[j].find("_"));
			 if(strcmp(curstatelabel.c_str(),templabel.c_str())!=0) // modified 
			 {
				 if(ph.nState>0)
				 {
				   PResults.push_back(ph);
				 }
				 ph.logProbability.clear();
				  ph.startTime.clear();
				  ph.endTime.clear();
				  ph.stateName.clear();
				  ph.nState=0;
				  curstatelabel=templabel;
			 }
			 else
			 {
			  
			 }
			 if(strcmp(templabel.c_str(),"sil")==0 && res[i].wordName[0]!='<')
			 {
			    continue;
			 }
			  ph.nState++;
			  ph.startTime.push_back(res[i].startTime[j]);
			  ph.endTime.push_back(res[i].endTime[j]);
			  ph.stateName.push_back(res[i].stateName[j]);
			  ph.wordName=templabel;
			  for(int k=res[i].startTime[j]-res[i].startTime[0];k<res[i].endTime[j]-res[i].startTime[0];k++)
			  {
				  ph.logProbability.push_back(res[i].logProbability[k]);
			  }
		  
		 }
      if(ph.nState>0)
	  {
		  PResults.push_back(ph);
	   }
		 ph.logProbability.clear();
		 ph.startTime.clear();
		 ph.endTime.clear();
		 ph.stateName.clear();
		 ph.nState=0;
	 }
	 */
	return res;
}

WordDuration TriphoneAlignment::DecodeSingleWord(Phone *pre, Word *word, Phone *next, double **score, int startFrame, int endFrame)
{
	Node *begin = CreateNode();
	Node *end = CreateNode();
	end->end = true;
	int i, j, k;
	AddWord(pre, word, next, begin, end, true);
	Recognition(score + startFrame, endFrame - startFrame, begin);
	WordDuration res;
	// add by v-wenhu to get the detail info of each word for single word alignment to support CompeteAlignment 
	res.nState=0;
	for(int i=0;i<StateAns.size();)
	{
		j=i;
		while(j<StateAns.size() && StateAns[i]==StateAns[j])
			++j;
		res.nState++;
		res.startTime.push_back(i+startFrame);
		res.endTime.push_back(j+startFrame);
		res.stateName.push_back(StateSet::GetName(StateAns[i]));
		i=j;
	}
	//
	res.wordName = word->name;
	for(j = startFrame; j < endFrame; ++j)
		res.logProbability.push_back(-score[j][StateAns[j - startFrame]]);
	return res;
}

void TriphoneAlignment::ReleaseAllofSource()
{
	Init();
}