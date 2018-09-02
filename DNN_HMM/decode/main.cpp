#include "decode.h"

int main(int argc, char **argv)
{
	if (argc == 1)
		return 0;
	char *mode = argv[1];
	if (strcmp(mode, "-r") == 0)
	{
		int pruning = atoi(argv[2]);
		double wordScore = atof(argv[3]);
		char *lmFile = argv[4];
		double exponent = atof(argv[5]);
		char *testFile = argv[6];
		char *outFile = argv[7];
		char *modelFile = argv[8];
		char *dictFile = argv[9];
		Decode::InitMonoPhone(modelFile, dictFile);
		Decode::InitLanguageModelFile(lmFile, exponent, wordScore);
		Decode::InitRecognition(pruning);
		FILE *fp = fopen(testFile, "r");
		char inFile[100];
		FILE *out = fopen(outFile, "w");
		fprintf(out, "#!MLF!#\n");
		while(fscanf(fp, "%s", inFile) != EOF)
		{
			cout << "Reading File: \"" << inFile <<"\"\n";
			PPScore *pp = new PPScore(inFile);
			double **score = pp->Score();
			if (score == NULL)
				break;
			int length = pp->Length();
			cout << "Processing File: \"" << inFile <<"\"\n";
			fprintf(out, "\"%s\"\n", inFile);
			Decode::Recognition(score, length, out);
			fflush(out);
			delete pp;
		}
		fclose(out);
		fclose(fp);
	}

	if (strcmp(mode, "-a") == 0 || strcmp(mode, "-as") == 0)
	{
		char *testFile = argv[2];
		char *contentFile = argv[3];
		char *outFile = argv[4];
		char *modelFile = argv[5];
		char *dictFile = argv[6];
		Decode::InitMonoPhone(modelFile, dictFile);
		Decode::InitAlignment();
		FILE *out = fopen(outFile, "w");
		FILE *fp = fopen(contentFile, "r");
		char st[100], fileName[100], inFile[100];
		fscanf(fp, "%s", st);
		fprintf(out, "%s\n", st);
		vector<string> wordList;
		while(fscanf(fp, "%s", fileName) != EOF)
		{
			wordList.clear();
			while(fscanf(fp, "%s", st), st[0] != '.')
				if (st[0] < '0' || st[0] > '9')
					wordList.push_back(string(st));
			strcpy(st, fileName + 3);
			int i;
			for(i = 0; st[i] != '.'; ++i);
			strcpy(st + i, ".pp");
			FILE *tp = fopen(testFile, "r");
			while(true)
			{
				if (fscanf(tp, "%s", inFile) == EOF)
					break;
				if (strcmp(st, inFile + (strlen(inFile) - strlen(st))) == 0)
				{
					cout << "Reading File: \"" << inFile <<"\"\n";
					PPScore *pp = new PPScore(inFile);
					double **score = pp->Score();
					if (score == NULL)
						break;
					int length = pp->Length();
					cout << "Processing File: \"" << inFile <<"\"\n";
					fprintf(out, "\"%s\"\n", inFile);
					Decode::Alignment(wordList, score, length, out, (strcmp(mode, "-as") == 0));
					fflush(out);
					delete pp;
					break;
				}
			}
			fclose(tp);
		}
		fclose(fp);
		fclose(out);
	}
	
	if (strcmp(mode, "-tr") == 0)
	{
		int pruning = atoi(argv[2]);
		double wordScore = atof(argv[3]);
		char *lmFile = argv[4];
		double exponent = atof(argv[5]);
		char *testFile = argv[6];
		char *outFile = argv[7];
		char *modelFile = argv[8];
		char *dictFile = argv[9];
		char *tiedFile = argv[10];
		Decode::InitTriPhone(modelFile, tiedFile, dictFile);
		Decode::InitLanguageModelFile(lmFile, exponent, wordScore);
		Decode::InitRecognition(pruning);
		FILE *fp = fopen(testFile, "r");
		char inFile[100];
		FILE *out = fopen(outFile, "w");
		fprintf(out, "#!MLF!#\n");
		while(fscanf(fp, "%s", inFile) != EOF)
		{
			cout << "Reading File: \"" << inFile <<"\"\n";
			PPScore *pp = new PPScore(inFile);
			double **score = pp->Score();
			if (score == NULL)
				break;
			int length = pp->Length();
			cout << "Processing File: \"" << inFile <<"\"\n";
			fprintf(out, "\"%s\"\n", inFile);
			Decode::Recognition(score, length, out);
			fflush(out);
			delete pp;
		}
		fclose(out);
		fclose(fp);
	}
	
	if (strcmp(mode, "-ta") == 0 || strcmp(mode, "-tas") == 0)
	{
		char *testFile = argv[2];
		char *contentFile = argv[3];
		char *outFile = argv[4];
		char *modelFile = argv[5];
		char *dictFile = argv[6];
		char *tiedFile = argv[7];
		Decode::InitTriPhone(modelFile, tiedFile, dictFile);
		Decode::InitAlignment();
		FILE *out = fopen(outFile, "w");
		FILE *fp = fopen(contentFile, "r");
		char st[100], fileName[100], inFile[100];
		fscanf(fp, "%s", st);
		fprintf(out, "%s\n", st);
		vector<string> wordList;
		while(fscanf(fp, "%s", fileName) != EOF)
		{
			wordList.clear();
			while(fscanf(fp, "%s", st), st[0] != '.')
				if (st[0] < '0' || st[0] > '9')
					wordList.push_back(string(st));
			strcpy(st, fileName + 3);
			int i;
			for(i = 0; st[i] != '.'; ++i);
			strcpy(st + i, ".pp");
			FILE *tp = fopen(testFile, "r");
			while(true)
			{
				if (fscanf(tp, "%s", inFile) == EOF)
					break;
				if (strcmp(st, inFile + (strlen(inFile) - strlen(st))) == 0)
				{
					cout << "Reading File: \"" << inFile <<"\"\n";
					PPScore *pp = new PPScore(inFile);
					double **score = pp->Score();
					if (score == NULL)
						break;
					int length = pp->Length();
					cout << "Processing File: \"" << inFile <<"\"\n";
					fprintf(out, "\"%s\"\n", inFile);
					Decode::Alignment(wordList, score, length, out, (strcmp(mode, "-tas") == 0));
					fflush(out);
					delete pp;
					break;
				}
			}
			fclose(tp);
		}
		fclose(fp);
		fclose(out);
	}

	if (strcmp(mode, "-ca") == 0 || strcmp(mode, "-ca2") == 0)
	{
		char *testFile = argv[2];
		char *contentFile = argv[3];
		char *outFile = argv[4];
		char *modelFile = argv[5];
		char *dictFile = argv[6];
		char *tiedFile = argv[7];
		Decode::InitTriPhone(modelFile, tiedFile, dictFile);
		Decode::InitAlignment();
		FILE *out = fopen(outFile, "w");
		FILE *fp = fopen(contentFile, "r");
		char st[100], fileName[100], inFile[100];
		fscanf(fp, "%s", st);
		fprintf(out, "%s\n", st);
		vector<string> wordList;
		while(fscanf(fp, "%s", fileName) != EOF)
		{
			wordList.clear();
			while(fscanf(fp, "%s", st), st[0] != '.')
				if (st[0] < '0' || st[0] > '9')
					wordList.push_back(string(st));
			strcpy(st, fileName + 3);
			int i;
			for(i = 0; st[i] != '.'; ++i);
			strcpy(st + i, ".pp");
			FILE *tp = fopen(testFile, "r");
			while(true)
			{
				if (fscanf(tp, "%s", inFile) == EOF)
					break;
				if (strcmp(st, inFile + (strlen(inFile) - strlen(st))) == 0)
				{
					cout << "Reading File: \"" << inFile <<"\"\n";
					PPScore *pp = new PPScore(inFile);
					double **score = pp->Score();
					if (score == NULL)
						break;
					int length = pp->Length();
					cout << "Processing File: \"" << inFile <<"\"\n";
					fprintf(out, "\"%s\"\n", inFile);
					vector<CompeteWordDuration> res;
					if (strcmp(mode, "-ca") == 0)
						res = Decode::CompeteAlignment(wordList, score, length);
					else res = Decode::CompeteAlignment2(wordList, score, length);
					for(i = 1; i + 1 < res.size(); ++i)
					{
						vector<pair <double, string> > candi;
						for(int j = 0; j < res[i].wordName.size(); ++j)
						{
							double lp = 0.0;
							for(int k = 0; k < res[i].logProbability[j].size(); ++k)
								lp += res[i].logProbability[j][k];
							candi.push_back(make_pair(lp / res[i].logProbability[j].size(), res[i].wordName[j]));
						}
						sort(candi.begin(), candi.end());
						for(int j = candi.size() - 1; j > 0; --j)
							fprintf(out, "%s %.6lf\t", candi[j].second.c_str(), candi[j].first);
						fprintf(out, "\n");
					}
					fprintf(out, ".\n");
					fflush(out);
					delete pp;
					break;
				}
			}
			fclose(tp);
		}
		fclose(fp);
		fclose(out);
	}
	
	//system("PAUSE");
	return 0;
}
