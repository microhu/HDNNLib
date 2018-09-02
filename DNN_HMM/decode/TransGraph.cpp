#include "TransGraph.h"

const double TransGraph::EPS = 1.0E-8;
const double TransGraph::MAX_SCORE = 1.0E+20;

TransGraph::Node* TransGraph::CreateNode()
{
	Node *n = new Node;
	n->id = nodeList.size();
	n->firstTrans = NULL;
	n->firstSkip = NULL;
	n->outDist = 0.0;
	n->end = false;
	nodeList.push_back(n);
	return n;
}

void TransGraph::AddTrans(Node* p, Node *q, double dist)
{
	Trans *t = new Trans;
	transList.push_back(t);
	t->next = p->firstTrans;
	t->dist = dist;
	t->trans = q;
	p->firstTrans = t;
}

void TransGraph::AddSkip(Node* p, Node *q, Word *word, double dist)
{
	Skip *s = new Skip;
	skipList.push_back(s);
	s->next = p->firstSkip;
	s->dist = dist;
	s->skip = q;
	s->word = word;
	p->firstSkip = s;
}

bool TransGraph::AddState(Node* p, int state, Node *&q, bool tie)
{
	if (tie)
	{
		Trans *t;
		for(t = p->firstTrans; t != NULL; t = t->next)
			if (t->trans->state == state)
				break;
		if (t != NULL)
		{
			q = t->trans;
			return false;
		}
	}
	q = CreateNode();
	q->state = state;
	return true;
}

TransGraph::Node* TransGraph::AddPhone(Node* p, Phone *ph, Hmm* hmm, bool tie)
{
	int i, j;
	Node *tn[5] = {p};
	bool b[5];
	TransMatrix *transMatrix = hmm->transMatrix;
	for(i = 1; i + 1 < transMatrix->nState; ++i)
	{
		b[i] = AddState(tn[i - 1], hmm->state[i], p, tie);
		tn[i] = p;
	}
	for(i = 1; i + 1 < transMatrix->nState; ++i)
		for(j = 1; j + 1 < transMatrix->nState; ++j)
			if (transMatrix->trans[i][j] > EPS && (b[i] || b[j]))
				AddTrans(tn[i], tn[j], -log(transMatrix->trans[i][j]));
	if (b[1])
		AddTrans(tn[0], tn[1], tn[0]->outDist - log(transMatrix->trans[0][1]));
	p->outDist = -log(transMatrix->trans[transMatrix->nState - 2][transMatrix->nState - 1]);
	return p;
}

void TransGraph::AddWord(Word *word, Node *p, Node *q, bool tie)
{
	for(int i = 0; i < word->pronunciation.size(); ++i)
		p = AddPhone(p, word->pronunciation[i], word->pronunciation[i]->hmm, tie);
	AddSkip(p, q, word, p->outDist);
}

void TransGraph::AddWord(Phone *first, Word *word, Phone* last, Node *p, Node *q, bool tie)
{
	int n = word->pronunciation.size();
	Phone *pre, *cur, *next;
	for(int i = 0; i < n; ++i)
	{
		pre = (i > 0) ? word->pronunciation[i - 1] : first;
		next = (i + 1 < n) ? word->pronunciation[i + 1] : last;
		cur = word->pronunciation[i];
		Hmm* hmm = PhoneSet::getInstance()->GetHmm(pre, cur, next);
		p = AddPhone(p, cur, hmm, tie);
	}
	AddSkip(p, q, word, p->outDist);
}

bool TransGraph::Recognition(double **score, int n, Node *start)
{
	int i, j, k, m = nodeList.size();
	Node *p, *q;
	Skip *s;
	Trans *t;
	double **opt;
	int **pre;
	Word ***word;
	opt = new double* [n + 1];
	pre = new int* [n + 1];
	word = new Word** [n + 1];
	for(i = 0; i <= n; ++i)
	{
		opt[i] = new double[m];
		pre[i] = new int[m];
		word[i] = new Word* [m];
	}
	for(i = 0; i <= n; ++i)
		for(j = 0; j < m; ++j)
			opt[i][j] = MAX_SCORE;
	opt[0][start->id] = 0.0;
	word[0][start->id] = NULL;
	for(i = 0; i <= n; ++i)
	{
		for(j = 0; j < m; ++j)
		{
			if (opt[i][j] > MAX_SCORE * 0.1)
				continue;
			p = nodeList[j];
			for(s = p->firstSkip; s != NULL; s = s->next)
			{
				q = s->skip;
				double temp = opt[i][j] + s->dist;
				k = q->id;
				if (opt[i][k] > temp)
				{
					opt[i][k] = temp;
					pre[i][k] = m + j;
					word[i][k] = s->word;
				}
			}
		}
		if (i == n)
			break;
		for(j = 0; j < m; ++j)
		{
			if (opt[i][j] > MAX_SCORE * 0.1)
				continue;
			p = nodeList[j];
			for(t = p->firstTrans; t != NULL; t = t->next)
			{
				q = t->trans;
				k = q->id;
				double temp = opt[i][j] + t->dist + score[i][q->state];
				if (opt[i + 1][k] > temp)
				{
					opt[i + 1][k] = temp;
					pre[i + 1][k] = j;
					word[i + 1][k] = word[i][j];
				}
			}
		}
	}
	StateAns.resize(n);
	WordAns.clear();
	double res = MAX_SCORE;
	for(k = 0; k < m; ++k)
		if (nodeList[k]->end && opt[n][k] < res)
		{
			j = k;
			res = opt[n][k];
		}
		if (res > MAX_SCORE * 0.1)
		{
			//cerr << "Solution not found!" << endl;
			for(i = 0; i <= n; ++i)
			{
				delete [] opt[i];
				delete [] pre[i];
				delete [] word[i];
			}
			delete [] opt;
			delete [] pre;
			delete [] word;
			return false;
		}
		for(i = n; i > 0; --i)
		{
			if (pre[i][j] >= m)
			{
				WordAns.push_back(make_pair(word[i][j], i));
				j = pre[i][j] - m;
			}
			StateAns[i - 1] = nodeList[j]->state;
			j = pre[i][j];
		}
		for(i = 0 ; i + i + 1 < WordAns.size(); ++i)
		{
			pair<Word*, int> temp = WordAns[i];
			WordAns[i] = WordAns[WordAns.size() - 1 - i];
			WordAns[WordAns.size() - 1 - i] = temp;
		}
		for(i = 0; i <= n; ++i)
		{
			delete [] opt[i];
			delete [] pre[i];
			delete [] word[i];
		}
		delete [] opt;
		delete [] pre;
		delete [] word;
		return true;
}

void TransGraph::mysort(TransState *tsList, int k, int l, int r)
{
	if (l + 1 >= r) return;
	if (l >= k) return;
	if (r <= k) return;
	double p = tsList[(l + r - 1) >> 1].score;
	TransState t;
	int i = l - 1, j = r;
	while(true)
	{
		for(++i; tsList[i].score + EPS < p; ++i);
		for(--j; tsList[j].score > p + EPS; --j);
		if (i < j)
		{
			t = tsList[i];
			tsList[i] = tsList[j];
			tsList[j] = t;
		}
		else break;
	}
	mysort(tsList, k, l, j + 1);
	mysort(tsList, k, j + 1, r);
}

bool TransGraph::Recognition(double **score, int n, Node *start, int pruning, int MAXBEAM)
{
	const double GENBEAM = (double) pruning;
	const double WORDBEAM = 0.5 * GENBEAM;
	int NTOKENS = 8;
	if (lModel == NULL)
		NTOKENS = 1;
	int limit = (n + WordSet::getInstance()->Size() + 5) * MAXBEAM * 2 + WordSet::getInstance()->Size() * StateSet::Size();
	TransState *tsList = new TransState[limit];
	vector<int> *existNode = new vector<int>[nodeList.size()];
	tsList[0].node = start;
	tsList[0].pre = 0;
	tsList[0].score = 0.0;
	tsList[0].word = NULL;
	tsList[0].wordEnd = true;
	Node *p, *pq, *q;
	Skip *s;
	Trans *t;
	int i, j, k, l = 0, r = 1;
	for(i = 0; i < n; ++i)
	{
		for(int temp_r = r; l < temp_r; ++l)
		{
			p = tsList[l].node;
			for(t = p->firstTrans; t != NULL; t = t->next)
			{
				q = t->trans;
				double t1 = tsList[l].score + t->dist + score[i][q->state];
				int tr;
				for(j = 0; j != existNode[q->id].size(); ++j)
				{
					tr = existNode[q->id][j];
					if (lModel == NULL || tsList[tr].word == tsList[l].word)
					{
						if (tsList[tr].score > t1)
						{
							tsList[tr].word = tsList[l].word;
							tsList[tr].score = t1;
							tsList[tr].pre = l;
							tsList[tr].wordEnd = false;
						}
						break;
					}
				}
				if (j != existNode[q->id].size())
					continue;
				tsList[r].node = q;
				tsList[r].pre = l;
				tsList[r].score = t1;
				tsList[r].word = tsList[l].word;
				tsList[r].wordEnd = false;
				if (existNode[q->id].size() < NTOKENS)
					existNode[q->id].push_back(r);
				else
				{
					int rt = r;
					for(j = 0; j != existNode[q->id].size(); ++j)
					{
						tr = existNode[q->id][j];
						if (tsList[tr].score > t1)
						{
							existNode[q->id][j] = rt;
							rt = tr;
							t1 = tsList[rt].score;
						}
					}
				}
				++r;
				if (r == limit)
				{
					cerr << "Array is too small!" << endl;
					system("PAUSE");
				}
			}
		}
		for(k = l; k < r; ++k)
			existNode[tsList[k].node->id].clear();
		double maxscore = MAX_SCORE;
		if (i + 1 < n)
		{
			mysort(tsList, l + MAXBEAM, l, r);
			if (l + MAXBEAM < r)
				r = l + MAXBEAM;		
			for(k = l; k < r; ++k)
				if (tsList[k].score < maxscore)
					maxscore = tsList[k].score;
			for(k = j = l; k < r; ++k)
				if (tsList[k].score < maxscore + GENBEAM)
					tsList[j++] = tsList[k];
			r = j;
		}
		int temp_l = l;
		for(int temp_r = r; l < temp_r; ++l)
		{
			p = tsList[l].node;
			for(s = p->firstSkip; s != NULL; s = s->next)
			{
				q = s->skip;
				if (i + 1 == n && !q->end)
					continue;
				double t1 = tsList[l].score + s->dist;
				if (lModel != NULL)
					t1 += lModel->Dist(tsList[l].word, s->word);
				if (t1 > maxscore + GENBEAM + WORDBEAM)
					continue;
				int tr;
				for(j = 0; j != existNode[q->id].size(); ++j)
				{
					tr = existNode[q->id][j];
					if (lModel == NULL || tsList[tr].word == s->word)
					{
						if (tsList[tr].score > t1)
						{
							tsList[tr].word = s->word;
							tsList[tr].score = t1;
							tsList[tr].pre = l;
							tsList[tr].wordEnd = true;
						}
						break;
					}
				}
				if (j != existNode[q->id].size())
					continue;
				tsList[r].node = q;
				tsList[r].pre = l;
				tsList[r].score = t1;
				tsList[r].word = s->word;
				tsList[r].wordEnd = true;
				if (existNode[q->id].size() < NTOKENS)
					existNode[q->id].push_back(r);
				else
				{
					int rt = r;
					for(j = 0; j != existNode[q->id].size(); ++j)
					{
						tr = existNode[q->id][j];
						if (tsList[tr].score > t1)
						{
							existNode[q->id][j] = rt;
							rt = tr;
							t1 = tsList[rt].score;
						}
					}
				}
				++r;
				if (r == limit)
				{
					cerr << "Array is too small!" << endl;
					system("PAUSE");
				}
			}
		}
		for(k = l; k < r; ++k)
			existNode[tsList[k].node->id].clear();
		l = temp_l;
	}
	for(;l < r; ++l)
		if (tsList[l].node->end)
			break;
	if (l == r)
	{
	//	cerr << "Solution not found!" << endl;
		delete [] tsList;
		delete [] existNode;
		return false;
	}
	int best = l;
	for(++l; l < r; ++l)
		if (tsList[l].node->end && tsList[best].score > tsList[l].score)
			best = l;
	StateAns.resize(n);
	WordAns.clear();
	j = best;
	for(i = n; i > 0; --i)
	{
		if (tsList[j].wordEnd)
		{
			WordAns.push_back(make_pair(tsList[j].word, i));
			j = tsList[j].pre;
		}
		StateAns[i - 1] = tsList[j].node->state;
		j = tsList[j].pre;
	}
	for(i = 0 ; i + i + 1 < WordAns.size(); ++i)
	{
		pair<Word*, int> temp = WordAns[i];
		WordAns[i] = WordAns[WordAns.size() - 1 - i];
		WordAns[WordAns.size() - 1 - i] = temp;
	}
	delete [] tsList;
	delete [] existNode;
	return true;
}
