#pragma once

#include <vector>
#include <map>
#include <string>
#include <iostream>

using namespace std;

class StateSet
{
private:
	static int nState; // number of states
	static vector<wstring> stateName;
	static map<wstring, int> stateID;
public:
    static int Clear();
	static int Size(void);
	static void SetSize(int nState);
	static void AddState(int id, wstring name);
	static wstring GetName(int id);
	static int Get(wstring name);
	static void Read(const wchar_t *stateFile);
};