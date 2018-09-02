#include "word.h"
#include "..\unicode.h"
#include "..\fileutil.h"

//instanceID = 0: evaluate
//instanceID = 1: error pattern
static int instanceID;
static WordSet* s_pWordSet[2] = { NULL, NULL };

WordSet* WordSet::getInstance() {
	if(s_pWordSet[instanceID] == NULL) {
		s_pWordSet[instanceID] = new WordSet();
	}

	return s_pWordSet[instanceID];
}

void WordSet::freeInstances()
{
    instanceID = 0;
	delete s_pWordSet[0];
	delete s_pWordSet[1];
	s_pWordSet[0] = NULL;
	s_pWordSet[1] = NULL;
}

int WordSet::Size(void)
{
	return m_nWord;
}

bool WordSet::setInstanceID(int id)
{
	if( id < 0 || id > 1 )	return false;
	instanceID = id;
	return true;
}

Word* WordSet::Get(int id)
{
	if (id < 0 || id >= m_nWord)
	{
		cerr << "Wrong word ID: " << id << endl;
		system("PAUSE");
	}
	return m_vecWordList[id];
}

vector<Word*> WordSet::Get(wstring name, bool hint_error)
{
	multimap<wstring, Word*>::const_iterator iter = m_mapWordPointer.find(name);
	vector<Word*> ans;
	if (iter == m_mapWordPointer.end())
	{
		if (hint_error)
		{
			wcerr << L"Wrong word name: \"" << name << L"\"\n";
			system("PAUSE");
		}
		return ans;
	}
	while(iter != m_mapWordPointer.end() && iter->first == name)
	{
		ans.push_back(iter->second);
		++iter;
	}
	return ans;
}

void WordSet::Read(const wchar_t *dictFile)
{
	cerr << "Reading dictionary ..." << endl;
	m_nWord = 0;
	m_mapWordPointer.clear();
	while(!m_vecWordList.empty())
	{
		delete m_vecWordList.back();
		m_vecWordList.pop_back();
	}

	auto_file_ptr fp = fopenOrDie(dictFile, L"r");
	if (fp == NULL)
	{
		wcerr << L"Dict file not found: \"" << dictFile << L"\"\n";
		system("PAUSE");
	}

	while(!feof(fp))
	{
		WSTRING st = fgetlinew(fp);
		int pos = st.find(' ');
		WSTRING name = st.substr(0, pos);

		Word* word = new Word();
		word->name = name;
		word->id = m_nWord++;
		word->output = true;
		m_mapWordPointer.insert(make_pair(name, word));
		m_vecWordList.push_back(word);

		st = st.substr(pos + 1);
		const wchar_t* p = st.c_str();
		while(*p != '\0')
		{
			wchar_t tt[128] = {0};
			if ((swscanf_s(p, L"%s", tt, _countof(tt))) == 0)
			{
				break;
			}
			if (tt[0] > '9' || tt[0] < '0')
			{
				if (wcscmp(tt, L"[]") == 0)
				{
					word->output = false;
				}
				else
				{
					word->pronunciation.push_back(PhoneSet::getInstance()->Get(tt));
				}
			}
			while(*p == ' ' || *p == '\t') ++p;
			while(*p != ' ' && *p != '\t' && *p != '\n' && *p != '\0') ++p;
		}
	}
}
