#pragma once

#include "io.h"
#include <vector>
#include <iostream>
#include <fstream>
//#include "HCopy.h"
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "f0.h"
std::vector<Pitch> get_f0(Wave w, F0_params *par, double start_time, double end_time);
void SetF0Params(F0_params *par);
int check_f0_params(register F0_params *par, register int sample_freq);
