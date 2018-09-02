#include "state.h"
#include "..\unicode.h"

int StateSet::nState = 0;
vector<wstring> StateSet::stateName;
map<wstring, int> StateSet::stateID;

int StateSet::Size(void)
{
	return nState;
}

int StateSet::Clear()
{
    nState = 0;
    stateName.clear();
    stateID.clear();
    return 0;
}

void StateSet::SetSize(int nState)
{
	if (nState <= 0)
	{
		cerr << "Wrong state number: " << nState << endl;
		system("PAUSE");
		return;
	}
	StateSet::nState = nState;
	stateName.resize(nState);
}

void StateSet::AddState(int id, wstring name)
{
	if (id < 0 || id >= nState)
	{
		cerr << "Wrong state ID: " << id << endl;
		system("PAUSE");
		return;
	}
	stateName[id] = name;
	stateID[name] = id;
}

wstring StateSet::GetName(int id)
{
	if (id < 0 || id >= nState)
	{
		cerr << "Wrong state ID: " << id << endl;
		system("PAUSE");
		return L"";
	}
	return stateName[id];
}

int StateSet::Get(wstring name)
{
	map<wstring, int>::const_iterator iter = stateID.find(name);
	if (iter == stateID.end())
	{
		wcerr << L"Wrong state name: \"" << name << L"\"\n";
		system("PAUSE");
		return -1;
	}
	return iter->second;
}

void StateSet::Read(const wchar_t *stateFile)
{
	cerr << "Reading state ..." << endl;
	FILE *fp = _wfopen(stateFile, L"r");
	if (fp == NULL)
	{
		cerr << "State file not found: \"" << stateFile << "\"\n";
		system("PAUSE");
		return;
	}
	nState = 0;
	stateName.clear();
	stateID.clear();
	char st[100] = {0};
	while(fscanf(fp, "%s", st) != EOF)
	{
		wstring s = char2wstring(st);
		stateID[s] = nState++;
		stateName.push_back(s);
	}
	fclose(fp);
}
