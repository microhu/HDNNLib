#pragma once

#include "state.h"
#include "hmm.h"

using namespace std;

struct Phone
{
	wstring name;
	int id;
	Hmm* hmm; // available when monophone
};

class PhoneSet
{
public:
	static PhoneSet* getInstance();
	static void freeInstance();

	PhoneSet()
	{
		m_nPhone = 0;
	}

	~PhoneSet()
	{
		TransMatrixSet::Clear();

		m_mapPhonePointer.clear();

		while (!m_vecPhoneList.empty())
		{
			delete m_vecPhoneList.back();
			m_vecPhoneList.pop_back();
		}

		while (!m_vecTriHmm.empty())
		{
			delete m_vecTriHmm.back();
			m_vecTriHmm.pop_back();
		}

		delete m_ppPreTriHmm;
	}
private:
	int m_nPhone = 0; // number of phones
	vector<Phone*> m_vecPhoneList;
	map<wstring, Phone*> m_mapPhonePointer;
	vector<Phone*> m_vecA, m_vecB, m_vecC; // triphone mention in the model
	vector<Hmm*> m_vecTriHmm; // the hmm of these triphones
	Hmm **m_ppPreTriHmm = NULL; // pre processing hmm of each triphone
public:
	int Size(void);
	Phone* Get(int id);
	Phone* Get(wstring name, bool hint_error = true);
	void ReadModel(const wchar_t *modelFile, bool triPhone = false);
	void ReadTied(const wchar_t *tiedFile);
	void GetTriPhone(wstring name, Phone *&a, Phone *&b, Phone *&c, bool hint_error = true);
	Hmm* GetHmm(Phone *a, Phone *b, Phone *c);
};



