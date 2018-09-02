#include "MonophoneRecognition.h"

MonophoneRecognition::MonophoneRecognition(LanguageModel *lModel)
{
	this->lModel = lModel;
	this->pruning = 0;
	root = CreateNode();
	int i, j, nWord = WordSet::getInstance()->Size();
	Node **beginWord = new Node*[nWord];
	Node **endWord = new Node*[nWord];
	Phone *sil = PhoneSet::getInstance()->Get(L"sil");
	for(i = 0; i < nWord; ++i)
	{
		endWord[i] = CreateNode();
		endWord[i]->end = (WordSet::getInstance()->Get(i)->pronunciation.back() == sil);
		AddWord(WordSet::getInstance()->Get(i), root, endWord[i]);
		beginWord[i] = root->firstTrans->trans;
		if (WordSet::getInstance()->Get(i)->pronunciation.front() != sil)
			root->firstTrans = root->firstTrans->next;
	}
	for(i = 0; i < nWord; ++i)
		for(j = 0; j < nWord; ++j)
		{
			double temp = 0.0;
			if (lModel != NULL)
				temp = lModel->Dist(WordSet::getInstance()->Get(i), WordSet::getInstance()->Get(j));
			AddTrans(endWord[i], beginWord[j], temp);
		}
	delete [] beginWord;
	delete [] endWord;
}

MonophoneRecognition::MonophoneRecognition(LanguageModel *lModel, int pruning)
{
	this->lModel = lModel;
	this->pruning = pruning;
	root = CreateNode();
	root->end = true;
	int i, nWord = WordSet::getInstance()->Size();
	for(i = 0; i < nWord; ++i)
		AddWord(WordSet::getInstance()->Get(i), root, root, true);
}

void MonophoneRecognition::Process(double **score, int length, FILE *out)
{
	if (pruning == 0)
		Recognition(score, length, root);
	else Recognition(score, length, root, pruning);
	for(int i = 0; i < WordAns.size(); ++i)
		if (WordAns[i].first->output)
			fprintf(out, "%s\n", WordAns[i].first->name.c_str());
	fprintf(out, ".\n");
}