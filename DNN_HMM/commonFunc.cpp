#include "commonFunc.h"

std::double_t LAdd(double A, double B)
{
	double temp = 0;
	double diff = 0;
	double C = 0;
	if (A < B)
	{
		temp = A;
		A = B;
		B = temp;
	}
	diff = B - A;
	if (diff < minLogExp)
	{
		return (A < LSMALL) ? LZERO : A;
	}
	else
	{
		C = exp(diff);
		return A + log(1.0 + C);
	}
}
vector<string> readFilePerLine(const string lexicon)
{
	vector<string> words;
	string line;
	string word;
	ifstream reader(lexicon);
	if (reader.is_open())
	{
		while (getline(reader, line))
		{
			words.push_back(line);
		}
		reader.close();
	}
	else
	{
		cerr << "can't open file: " << lexicon.c_str() << endl;
	}
	return words;
}

vector<wstring> split(wstring str, wchar_t sep)
{
	vector<wstring> res;
	if (str.size() == 0) return res;
	size_t startPos = 0;
	size_t endPos = str.find(sep);
	while (endPos != wstring::npos)
	{
		if (endPos>startPos)
			res.push_back(str.substr(startPos, endPos - startPos));
		++endPos;
		startPos = endPos;
		endPos = str.find(sep, endPos);
	}
	if (startPos<str.size())
		res.push_back(str.substr(startPos));
	return res;
}

map<string, string> readKeyValuePair(const string speakerUttPiarFile, const bool uniqueOnly)
{
	map<string, string> speakerMap;
	ifstream reader(speakerUttPiarFile);
	string line;
	string key, value;
	if (reader.is_open())
	{
		while (getline(reader, line))
		{
			istringstream iss(line);
			iss >> key >> value;

			// the first utterance
			if (speakerMap.find(key) == speakerMap.end())
			{
				speakerMap[key] = value;
			}
			else
			{
				// check replicate cases
				cout << "This key has already included! " << key << endl;
			}
		}
		reader.close();
	}
	else
	{
		std::cerr << "can't open file: " << speakerUttPiarFile.c_str() << std::endl;
	}


	return speakerMap;
}

map<string, vector<string>> parseHtkMlf(const string mlf)
{
	map<string, vector<string>> uttWordsDict;

	string line;
	string word;
	string prefix;
	string uttid;

	ifstream reader(mlf);
	if (reader.is_open())
	{
		while (getline(reader, line))
		{
			prefix = line.length() >= 2 ? line.substr(0, 2) : line;
			//istringstream iss(line);
			//iss >> word;

			if (prefix.compare("#!") == 0)
			{
				// skip this line
			}
			else if (prefix.compare("\"*") == 0)
			{
				uttid = line.substr(line.find_last_of('/') + 1, line.find_last_of('.') - line.find_last_of('/') - 1);
				uttWordsDict[uttid] = vector<string>();

			}
			else if (prefix.compare(".") == 0)
			{

			}
			else
			{

				uttWordsDict[uttid].push_back(line);
			}
		}
	}
	else
	{
		cout << "can't open file: " << mlf.c_str() << endl;
	}
	return uttWordsDict;
}

void WriteHtkMlfToFile(const map<string, vector<string>> &uttwords, const string &outFile, const string lab , const bool toUpperCase )
{
	ofstream writer(outFile);
	writer << "#!MLF!#" << endl;
	if (writer.is_open())
	{
		for (map<string, vector<string>>::const_iterator iter = uttwords.begin(); iter != uttwords.end(); ++iter)
		{
			writer << "\"*/" << iter->first << "." << lab << "\"" << endl;
			for (vector<string>::const_iterator siter = iter->second.begin(); siter != iter->second.end(); ++siter)
			{
				if (toUpperCase)
				{
					string upperWord = *siter;
					transform((*siter).begin(), (*siter).end(), upperWord.begin(), toupper);
					writer << upperWord << endl;
				}
				else
				{
					writer << *siter << endl;
				}
			}
			writer << "." << endl;
		}
		writer.close();
	}
	else
	{
		cerr << "can't open file: " << outFile.c_str() << endl;
	}
}

void WriteVectorContentToFile(const vector<string> lines, const string outFile)
{
	ofstream writer(outFile);
	if (writer.is_open())
	{
		for (vector<string>::const_iterator iter = lines.begin(); iter != lines.end(); ++iter)
			writer << *iter << endl;
		writer.close();
	}
	else
	{
		cerr << "can't open file: " << outFile.c_str() << endl;
	}

}

