#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include<string>
#include <sstream>
#include <fstream>
#include <unordered_set>
using namespace std;

const double LZERO = -pow(10, 10);
const double ZERO = pow(10, -10);
const double minLogExp = -log(-LZERO);
const double LSMALL = -0.5*pow(10, 10);
const double M_PI = 3.14159265358979323846;

std::double_t LAdd(double A, double B);
template<typename Ttype>
bool keyValuePair_compare(const pair<Ttype, float> &a, const pair<Ttype, float>& b)
{
	return a.second > b.second;
};
vector<string> readFilePerLine(const string lexicon);
map<string, string> readKeyValuePair(const string speakerUttPiarFile, const bool uniqueOnly = true);

map<string, vector<string>> parseHtkMlf(const string mlf);

void WriteHtkMlfToFile(const map<string, vector<string>> &uttwords, const string &outFile, const string lab = "lab", const bool toUpperCase = false);

void WriteVectorContentToFile(const vector<string> lines, const string outFile);

vector<wstring> split( wstring line, wchar_t sep);
