#include "PPScore.h"

PPScore::PPScore(const char *inFile)
{
	score = NULL;
	FILE *fp = fopen(inFile, "r");
	if (fp == NULL)
		cerr << "File not found: \"" << inFile <<"\"\n";
	else
	{
		n = 0; m = StateSet::Size();
		int i, j, flag;
		double t;
		while(true)
		{
			for(j = 0; j < m; ++j)
				if ((flag = fscanf(fp, "%lf", &t)) != 1)
					break;
			if (flag == 0)
			{
				cerr << "PP file content does not match: \"" << inFile << "\"\n";
				system("PAUSE");
			}
			if (j < m)
			{
				if (j > 0)
				{
					cerr << "PP file length does not match: \"" << inFile << "\"\n";
					system("PAUSE");
				}
				break;
			}
			++n;
		}
		score = new double* [n];
		for(i = 0; i < n; ++i)
			score[i] = new double[m];
		rewind(fp);
		for(i = 0; i < n; ++i)
		{
			for(j = 0; j < m; ++j)
			{
				fscanf(fp, "%lf", &t);
				score[i][j] = -t;
			}
		}
		fclose(fp);
	}
}

PPScore::~PPScore(void)
{
	if (score == NULL)
		return;
	for(int i = 0; i < n; ++i)
		delete [] score[i];
	delete [] score;
}

int PPScore::Length(void)
{
	return n;
}

double** PPScore::Score(void)
{
	return score;
}
