#pragma once

#include "phone.h"

// this class can only get the state of a triphone but can not get the trans matrix, currently not used

class DecisionTreeModel
{
private:
	class Question
	{
	private:
		vector<pair<Phone*, Phone*> > tri; // group of possible combinations in the question
		// the pointer point to null if the phone is "*"
	public:
		void Add(Phone *p, Phone *q) // add a phone combination to the question
		{
			tri.push_back(make_pair(p, q));
		}
		bool Ask(Phone *p, Phone *q) // ask a phone combination and return the answer
		{
			for(vector<pair<Phone*, Phone*> >::const_iterator
				iter = tri.begin(); iter != tri.end(); ++iter)
				if ((iter->first == NULL || p == iter->first)
					&& (iter->second == NULL || q == iter->second))
					return true;
			return false;
		}
	};
	map<string, Question*> question;
	struct Node // node type in the decision tree
	{
		int state; // corresponding state of the node, it is available when leaf node
		Node *wrong, *right; // wrong son and right son, they are not available when leaf node
		Question *question; // point to the question, it point to null when leaf node
		Node(int state)
		{
			this->state = state;
			this->question = NULL;
		}
		Node(void)
		{
		}
	};
	map< pair<Phone*, int>, Node*> phoneTree; // root node of tree, key is the phone and the corresponding state
public:
	void Read(const wchar_t *inFile)
	{
		cerr << "Reading decision tree ..." << endl;
		FILE *fp = _wfopen(inFile, L"r");
		if (fp == NULL)
		{
			cerr << "Decision tree model file not found: \"" << inFile << "\"\n";
			system("PAUSE");
		}
		char st[100], tt[1000], a[100], b[100];
		Phone *p, *q;
		int i, j, k, o;
		while(fscanf(fp, "%s", st) != EOF)
		{
			if (strcmp(st, "QS") == 0)
			{
				fscanf(fp, "%s", st);
				string name = string(st);
				Question *qs = new Question;
				fscanf(fp, "%s", st);
				fscanf(fp, "%s", tt);
				for(i = 0; i < strlen(tt); ++i)
				{
					for(j = ++i; tt[j] != '-'; ++j);
					strncpy(a, tt + i, j - i);
					a[j - i] = '\0';
					if (strcmp(a, "*") != 0)
						p = PhoneSet::Get(a);
					else p = NULL;
					for(i = j; tt[i] != '+'; ++i);
					for(j = ++i; tt[j] != '\"'; ++j);
					strncpy(b, tt + i, j - i);
					b[j - i] = '\0';
					if (strcmp(b, "*") != 0)
						q = PhoneSet::Get(b);
					else q = NULL;
					i = j + 1;
					if (strcmp(a, "*") != 0 && p == NULL)
						continue;
					if (strcmp(b, "*") != 0 && q == NULL)
						continue;
					qs->Add(p, q);
				}
				fscanf(fp, "%s", st);
				question[name] = qs;
			}
			else
			{
				Node *n;
				for(i = 0; st[i] != '['; ++i);
				st[i] = '\0';
				p = PhoneSet::Get(string(st));
				k = atoi(st + i + 1) - 1;
				fscanf(fp, "%s", st);
				if (strcmp(st, "{") != 0)
				{
					st[strlen(st) - 1] = '\0';
					phoneTree[make_pair(p, k)] = new Node(StateSet::Get(string(st + 1)));
				}
				else
				{
					vector<Node*> nodeList;
					i = 0;
					while(fscanf(fp, "%s", st), strcmp(st, "}") != 0)
					{
						while(i >= nodeList.size())
						{
							n = new Node;
							nodeList.push_back(n);
						}
						fscanf(fp, "%s", st);
						nodeList[i]->question = question[st];
						//wrong
						fscanf(fp, "%s", st);
						if (st[0] == '-')
						{
							j = atoi(st + 1);
							while(j >= nodeList.size())
							{
								n = new Node;
								nodeList.push_back(n);
							}
							nodeList[i]->wrong = nodeList[j];
						}
						else
						{
							st[strlen(st) - 1] = '\0';
							nodeList[i]->wrong = new Node(StateSet::Get(string(st + 1)));
						}
						//right
						fscanf(fp, "%s", st);
						if (st[0] == '-')
						{
							j = atoi(st + 1);
							while(j >= nodeList.size())
							{
								n = new Node;
								nodeList.push_back(n);
							}
							nodeList[i]->right = nodeList[j];
						}
						else
						{
							st[strlen(st) - 1] = '\0';
							nodeList[i]->right = new Node(StateSet::Get(string(st + 1)));
						}
						++i;
					}
					phoneTree[make_pair(p, k)] = nodeList[0];
				}
			}
		}
		fclose(fp);
		// pre-processing all the triphones
	}
	int Ask(Phone *a, Phone *b, Phone *c, int k) // get the state of a triphone state by checking the model, a slow function
	{
		Node *n;
		n = phoneTree[make_pair(b, k)];
		if (n == NULL)
		{
			cerr << "Can not access the " << k + 1 << "th state of \"" << a->name
				<< "-" << b->name << "+" << c->name << "\"\n";
			system("PAUSE");
			return 0;
		}
		while(n->question != NULL)
			if (n->question->Ask(a, c))
				n = n->right;
			else n = n->wrong;
		return n->state;
	}

};