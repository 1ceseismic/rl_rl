#pragma once
#include <GigaLearnCPP/Learner.h>

namespace rl_rl {

void StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report);

}
