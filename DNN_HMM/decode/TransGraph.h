#pragma once

#include "word.h"
#include "LanguageModel.h"
#include <set>

class TransGraph // transition graph, all the distance in the graph is -loglikelihood
{
protected:
	const static double EPS;
	const static double MAX_SCORE;
	struct Trans; // transition in the graph
	struct Skip; // skip in the graph. Skip to a transit node, and generate a word
	struct Node // node in the graph
	{
		int id;
		int state; // state of the node, not available when transit node
		double outDist; // distance of transit out
		Trans *firstTrans; // point to the first transit, null when no transits
		Skip *firstSkip; // point to the first skip, null when no skips 
		bool end; // if this node can be the end node
	};
	vector<Node*> nodeList;
	Node* CreateNode(); // create a new node
	struct Trans
	{
		Node *trans; // transit to this node
		double dist; // transit distance
		Trans *next; // next transit
	};
	vector<Trans*> transList;
	void AddTrans(Node* p, Node *q, double dist);
	struct Skip
	{
		Node *skip; // skip to this node
		double dist; // skip distance
		Word *word;
		Skip *next; // next skip
	};
	vector<Skip*> skipList;
	void AddSkip(Node* p, Node *q, Word *word, double dist);
	bool AddState(Node* p, int state, Node *&q, bool tie); // add a transit node q with certain state after node p
	Node* AddPhone(Node* p, Phone *ph, Hmm *hmm, bool tie); // add a certain phone with certain hmm after node p
	// return the end node of the phone
	void AddWord(Word *word, Node *p, Node *q, bool tie = false); // just for monophone! return the first node of that word
	void AddWord(Phone *first, Word *word, Phone* last, Node *p, Node *q, bool tie = false); // just for triphone! return the first node of that word
	
	////////////result of recognition///////////////////
	vector< pair<Word*, int> > WordAns; //  (w1 , f1) , (w2 , f2), ...
	// word w is detected after f frame
	// word w1 is frame 0 to frame f1 - 1, word w2 is frame f1 to frame f2 - 1, ...
	vector<int> StateAns; // state of each frame
	////////////////////////////////////////////////////
	
	bool Recognition(double **score, int n, Node *start);
	// get the best solution by dynamic programming, may out of memory when the transition graph is too big
	// score is a two-dimension array, score[i][j] means the -loglikelihood of frame i assign to state j
	// n is the number of frames
	// the start node and the end node should all be transit node
	struct TransState // transition state when searching
	{
		Node *node;
		int pre; // previous state in the list
		double score; // score when transit to this node
		Word *word; // last word in the transition process
		bool wordEnd;
	};
	void mysort(TransState *tsList, int k, int l, int r); // Select the best k - l element in index l to r
	bool Recognition(double **score, int n, Node *start, int pruning, int MAXBEAM = 5000);
	// get the best solution by searching, pruning is the number of states reserved
	// may out of memory when pruning is too big
	// score is a two-dimension array, score[i][j] means the -loglikelihood of frame i assign to state j
	// n is the number of frames
	// the start node and the end node should all be transit node
	int pruning;
	LanguageModel *lModel;
public:
	virtual void Process(double **score, int length, FILE *out) {}
	virtual void Process(vector<wstring> wordList, double **score, int length, FILE *out, bool stateLevel) {}
	virtual vector<WordDuration> GetResult(vector<wstring> wordList, double **score, int length)
	{
		vector<WordDuration> wd;
		return wd;
	}
	virtual WordDuration DecodeSingleWord(Phone *pre, Word *word, Phone *next, double **score, int startFrame, int endFrame)
	{
		WordDuration wd;
		return wd;
	}
	virtual void ReleaseAllofSource(){}
};
