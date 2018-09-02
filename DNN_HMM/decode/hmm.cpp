#include "hmm.h"

map<wstring, TransMatrix*> TransMatrixSet::transMatrixPointer;

TransMatrix* TransMatrixSet::Get(wstring name)
{
	map<wstring, TransMatrix*>::const_iterator iter = transMatrixPointer.find(name);
	if (iter == transMatrixPointer.end())
	{
		wcerr << L"Wrong trans matrix name: \"" << name << L"\"\n";
		system("PAUSE");
		return NULL;
	}
	return iter->second;
}

void TransMatrixSet::Add(TransMatrix *transMatrix)
{
	//If have a same name item, need delete a duplication value
	//here delete old

	if (transMatrixPointer.count(transMatrix->name) == 1)
	{
		delete transMatrixPointer.find(transMatrix->name)->second;
	}

	transMatrixPointer[transMatrix->name] = transMatrix;
}

void TransMatrixSet::Clear()
{
	for (map<wstring, TransMatrix*>::iterator i = transMatrixPointer.begin(); i != transMatrixPointer.end(); i++)
	{
		delete i->second;
	}

	transMatrixPointer.clear();
}
