#include "TriphoneRecognition.h"

TriphoneRecognition::TriphoneRecognition(LanguageModel *lModel)
{
	this->lModel = lModel;
	this->pruning = 0;
	int nPhone = PhoneSet::getInstance()->Size();
	int nWord = WordSet::getInstance()->Size();
	Node **skipNode = new Node* [nWord * nPhone];
	int i, j, k, sil = PhoneSet::getInstance()->Get(L"sil")->id;
	bool *existPhone = new bool[nPhone];
	for(j = 0; j < nPhone; ++j)
		existPhone[j] = false;
	for(i = 0; i < nWord; ++i)
		for(j = 0; j < nPhone; ++j)
		{
			skipNode[i * nPhone + j] = CreateNode();
			skipNode[i * nPhone + j]->end = (WordSet::getInstance()->Get(i)->pronunciation.back()->id == sil && j == sil);
		}
	for(i = 0; i < nWord; ++i)
		for(j = 0; j < WordSet::getInstance()->Get(i)->pronunciation.size(); ++j)
			existPhone[WordSet::getInstance()->Get(i)->pronunciation[j]->id] = true;
	root = CreateNode();
	Node *tNode = CreateNode(), *preNode;
	for(i = 0; i < nWord; ++i)
	{
		Word *word = WordSet::getInstance()->Get(i);
		for(j = 0; j < nWord; ++j)
		{
			double temp = 0.0;
			if (lModel != NULL)
				temp = lModel->Dist(WordSet::getInstance()->Get(j), word);
			int t = j * nPhone + word->pronunciation.front()->id;
			preNode = NULL;
			Phone* front = WordSet::getInstance()->Get(j)->pronunciation.back();
			for(k = 0; k < nPhone; ++k)
			{
				if (!existPhone[k])
					continue;
				AddWord(front, word, PhoneSet::getInstance()->Get(k),
					tNode, skipNode[word->id * nPhone + k], true);
				if (preNode != tNode->firstTrans->trans)
				{
					preNode = tNode->firstTrans->trans;
					AddTrans(skipNode[t], preNode, temp);
					if (word->pronunciation.front()->id == sil)
						AddTrans(root, preNode, 0.0);
				}
			}
			tNode->firstTrans = NULL;
		}
	}
	delete [] skipNode;
	delete [] existPhone;
}

TriphoneRecognition::TriphoneRecognition(LanguageModel *lModel, int pruning)
{
	this->lModel = lModel;
	this->pruning = pruning;
	int nPhone = PhoneSet::getInstance()->Size();
	int nWord = WordSet::getInstance()->Size();
	Node **skipNode = new Node* [nPhone * nPhone];
	int i, j, k, sil = PhoneSet::getInstance()->Get(L"sil")->id;
	bool *existPhone = new bool[nPhone];
	for(j = 0; j < nPhone; ++j)
		existPhone[j] = false;
	for(i = 0; i < nPhone; ++i)
		for(j = 0; j < nPhone; ++j)
			skipNode[i * nPhone + j] = CreateNode();
	for(i = 0; i < nWord; ++i)
		for(j = 0; j < WordSet::getInstance()->Get(i)->pronunciation.size(); ++j)
			existPhone[WordSet::getInstance()->Get(i)->pronunciation[j]->id] = true;
	root = skipNode[sil * nPhone + sil];
	root->end = true;
	for(i = 0; i < WordSet::getInstance()->Size(); ++i)
	{
		Word *word = WordSet::getInstance()->Get(i);
		for(j = 0; j < nPhone; ++j)
			for(k = 0; k < nPhone; ++k)
			{
				if (!existPhone[j] || !existPhone[k])
					continue;
				AddWord(PhoneSet::getInstance()->Get(j), word, PhoneSet::getInstance()->Get(k),
					skipNode[j * nPhone + word->pronunciation.front()->id],
					skipNode[word->pronunciation.back()->id * nPhone + k], true);
			}
	}
	delete [] skipNode;
	delete [] existPhone;
}


void TriphoneRecognition::Process(double **score, int length, FILE *out)
{
	if (pruning == 0)
		Recognition(score, length, root);
	else Recognition(score, length, root, pruning);
	for(int i = 0; i < WordAns.size(); ++i)
		if (WordAns[i].first->output)
			fprintf(out, "%s\n", WordAns[i].first->name.c_str());
	fprintf(out, ".\n");
}
