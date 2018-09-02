#include "phone.h"
#include "..\fileutil.h"
#include "..\unicode.h"

static PhoneSet* s_pPhoneSet = NULL;

PhoneSet* PhoneSet::getInstance() 
{
	if(s_pPhoneSet == NULL) {
		s_pPhoneSet = new PhoneSet();
	}
	return s_pPhoneSet;
}

void PhoneSet::freeInstance()
{
	delete s_pPhoneSet;
	s_pPhoneSet = NULL;
}


int PhoneSet::Size(void)
{
	return m_nPhone;
}

Phone* PhoneSet::Get(int id)
{
	if (id < 0 || id >= m_nPhone)
	{
		cerr << "Wrong phone ID: " << id << endl;
		system("PAUSE");
		return NULL;
	}
	return m_vecPhoneList[id];
}

Phone* PhoneSet::Get(wstring name, bool hint_error)
{
	map<wstring, Phone*>::const_iterator iter = m_mapPhonePointer.find(name);
	if (iter == m_mapPhonePointer.end())
	{
		if (hint_error)
		{
			wcerr << L"Wrong phone name: " << name << endl;
			system("PAUSE");
		}
		return NULL;
	}
	return iter->second;
}

void PhoneSet::GetTriPhone(wstring name, Phone *&a, Phone *&b, Phone *&c, bool hint_error)
{
	wstring st[3];
	int i = 0, j = 0;
	bool minus = false, plus = false;
	for(i = 0; i != name.size(); ++i)
	{
		if (name[i] == '-')
		{
			minus = true;
			++j;
		}
		else
		if (name[i] == '+')
		{
			plus = true;
			++j;
		}
		else st[j].push_back(name[i]);
	}
	for(i = 0; i <= j; ++i)
	{
		if (PhoneSet::Get(st[i], hint_error) == NULL)
		{
			Phone *ph = new Phone;
			m_mapPhonePointer[st[i]] = ph;
			m_vecPhoneList.push_back(ph);
			ph->name = st[i];
			ph->id = m_nPhone++;
		}
	}
	if (minus && plus)
	{
		a = PhoneSet::Get(st[0]);
		b = PhoneSet::Get(st[1]);
		c = PhoneSet::Get(st[2]);
	}
	if (minus && !plus)
	{
		a = PhoneSet::Get(st[0]);
		b = PhoneSet::Get(st[1]);
		c = NULL;
	}
	if (!minus && plus)
	{
		a = NULL;
		b = PhoneSet::Get(st[0]);
		c = PhoneSet::Get(st[1]);
	}
	if (!minus && !plus)
	{
		a = NULL;
		b = PhoneSet::Get(st[0]);
		c = NULL;
	}
}

void PhoneSet::ReadModel(const wchar_t *modelFile, bool triPhone)
{
	cerr << "Reading model ..." << endl;

	auto_file_ptr fp = fopenOrDie(modelFile, L"r");
	if (fp == NULL)
	{
		wcerr << L"Model file not found: \"" << modelFile << L"\"\n";
		system("PAUSE");
	}

	m_nPhone = 0;
	m_mapPhonePointer.clear();
	while(!m_vecPhoneList.empty())
	{
		delete m_vecPhoneList.back();
		m_vecPhoneList.pop_back();
	}
	int i, j;
	char st[100];
	while(true)
	{
		if (fscanf(fp, "%s", st) == EOF)
			break;
		wstring name;
		if (strcmp(st, "~t") == 0)
		{
			fscanf(fp, "%s", st);
			for(char *p = st + 1; *p != '\"'; ++p)
				name.push_back(*p);
			TransMatrix *transMatrix = new TransMatrix;
			transMatrix->name = name;
			while(true)
			{
				fscanf(fp, "%s", st);
				if (strcmp(st, "<TRANSP>") == 0)
					break;
			}
			fscanf(fp, "%d", &transMatrix->nState);
			TransMatrixSet::Add(transMatrix);
			for(i = 0; i < transMatrix->nState; ++i)
				for(j = 0; j < transMatrix->nState; ++j)
					fscanf(fp, "%lf", &transMatrix->trans[i][j]);
		}
		else if (strcmp(st, "~h") == 0)
		{
			fscanf(fp, "%s", st);
			for(char *p = st + 1; *p != '\"'; ++p)
				name.push_back(*p);
			Hmm *hmm = new Hmm;
			fscanf(fp, "%s", st);
			while(true)
			{
				if (strcmp(st, "<TRANSP>") == 0 || strcmp(st, "~t") == 0)
					break;
				if (strcmp(st, "<STATE>") == 0)
				{
					int s;
					fscanf(fp, "%d", &s);
					while(true)
					{
						fscanf(fp, "%s", st);
						if (strcmp(st, "<TRANSP>") == 0 || strcmp(st, "~t") == 0 || strcmp(st, "<STATE>") == 0)
						{
							hmm->state[s - 1] = StateSet::Get(name + L"[" + wchar_t(48 + s) + L"]");
							break;
						}
						if (strcmp(st, "~s") == 0)
						{
							fscanf(fp, "%s", st);
							wstring name;
							for(char *p = st + 1; *p != '\"'; ++p)
								name.push_back(*p);
							hmm->state[s - 1] = StateSet::Get(name);
							fscanf(fp, "%s", st);
							break;
						}
					}
				}
				else fscanf(fp, "%s", st);
			}
			TransMatrix *transMatrix;
			if (strcmp(st, "<TRANSP>") == 0)
			{
				transMatrix = new TransMatrix;
				fscanf(fp, "%d", &transMatrix->nState);
				TransMatrixSet::Add(transMatrix);
				for(i = 0; i < transMatrix->nState; ++i)
					for(j = 0; j < transMatrix->nState; ++j)
						fscanf(fp, "%lf", &transMatrix->trans[i][j]);
			}
			else
			{
				fscanf(fp, "%s", st);
				wstring name;
				for(char *p = st + 1; *p != '\"'; ++p)
				{
						name.push_back(*p);
				}
				transMatrix = TransMatrixSet::Get(name);
			}
			hmm->transMatrix = transMatrix;
			if (!triPhone)
			{
				Phone *ph = new Phone;
				m_mapPhonePointer[name] = ph;
				m_vecPhoneList.push_back(ph);
				ph->name = name;
				ph->id = m_nPhone++;
				ph->hmm = hmm;
			}
			else
			{
				Phone *pa, *pb, *pc;
				GetTriPhone(name, pa, pb, pc, false);
				m_vecA.push_back(pa);
				m_vecB.push_back(pb);
				m_vecC.push_back(pc);
				m_vecTriHmm.push_back(hmm);
			}
		}
		else if (strcmp(st, "~s") == 0)
		{
			fscanf(fp, "%s", st);
			for(char *p = st + 1; *p != '\"'; ++p)
				name.push_back(*p);
			int id;
			fscanf(fp, "%*s%*s%d", &id);
			if (id >= StateSet::Size())
				StateSet::SetSize(id + 1);
			StateSet::AddState(id, name);
		}
	}
	fclose(fp);
};

void PhoneSet::ReadTied(const wchar_t *tiedFile)
{
	cerr << "Reading tied file ..." << endl;

	m_ppPreTriHmm = new Hmm*[m_nPhone * m_nPhone * m_nPhone];
	for(int k = 0; k < m_nPhone * m_nPhone * m_nPhone; ++k)
	{
		m_ppPreTriHmm[k] = NULL;
	}
	for(int i = 0; i != m_vecTriHmm.size(); ++i)
	{
		if (m_vecA[i] == NULL && m_vecC[i] == NULL)
		{
			for(int u = 0; u < m_nPhone; ++u)
			{
				for(int v = 0; v < m_nPhone; ++v)
				{
					m_ppPreTriHmm[u * m_nPhone * m_nPhone + m_vecB[i]->id * m_nPhone + v] = m_vecTriHmm[i];
				}
			}
		}
	}
	for(int i = 0; i != m_vecTriHmm.size(); ++i)
	{
		if (m_vecA[i] == NULL && m_vecC[i] != NULL)
		{
			for(int u = 0; u < m_nPhone; ++u)
			{
				m_ppPreTriHmm[u * m_nPhone * m_nPhone + m_vecB[i]->id * m_nPhone + m_vecC[i]->id] = m_vecTriHmm[i];
			}
		}
	}
	for(int i = 0; i != m_vecTriHmm.size(); ++i)
	{
		if (m_vecA[i] != NULL && m_vecC[i] == NULL)
		{
			for(int v = 0; v < m_nPhone; ++v)
			{
				m_ppPreTriHmm[m_vecA[i]->id * m_nPhone * m_nPhone + m_vecB[i]->id * m_nPhone + v] = m_vecTriHmm[i];
			}
		}
	}
	for(int i = 0; i != m_vecTriHmm.size(); ++i)
	{
		if (m_vecA[i] != NULL && m_vecC[i] != NULL)
		{
			m_ppPreTriHmm[m_vecA[i]->id * m_nPhone * m_nPhone + m_vecB[i]->id * m_nPhone + m_vecC[i]->id] = m_vecTriHmm[i];
		}
	}

	auto_file_ptr fp = fopenOrDie(tiedFile, L"r");
	if (fp == NULL)
	{
		wcerr << L"Tied file not found: \"" << tiedFile << L"\"\n";
		system("PAUSE");
	}
	while(!feof(fp))
	{
		WSTRING st = fgetlinew(fp);
		int pos = st.find(' ');
		if(pos == WSTRING::npos) {
			continue;
		}

		WSTRING s1 = st.substr(0, pos);
		WSTRING s2 = st.substr(pos + 1);

		Phone *a1, *b1, *c1, *a2, *b2, *c2;
		GetTriPhone(s1, a1, b1, c1);
		GetTriPhone(s2, a2, b2, c2);
		m_ppPreTriHmm[a1->id * m_nPhone * m_nPhone + b1->id * m_nPhone + c1->id]
			= m_ppPreTriHmm[a2->id * m_nPhone * m_nPhone + b2->id * m_nPhone + c2->id];
	}
	fclose(fp);
}

Hmm* PhoneSet::GetHmm(Phone *a, Phone *b, Phone *c)
{
	Hmm* res = m_ppPreTriHmm[a->id * m_nPhone * m_nPhone + b->id * m_nPhone + c->id];
	if (res == NULL)
	{
		wcerr << L"Can not handle triphone \"" << a->name << L"-" << b->name << L"+" << c->name << L"\"\n";
		system("PAUSE");
		//system("BREAK"); 
	}
	return res;
}