// Leaked-source Linux-build shim. GGL's RLGymCPP/src/RLGymCPP/EnvSet/EnvSet.h
// includes "../OBSBuilders/OBSBuilder.h" (all-caps) but the real directory is
// "ObsBuilders/" and the real file is "ObsBuilder.h" — case-sensitive Linux
// fails the lookup. We add ../_shim/_anchor/ to the include path so the
// preprocessor's include-path fallback resolves `../OBSBuilders/OBSBuilder.h`
// to this forwarder file. Kept tiny and not touching the out-of-repo tree.
#pragma once
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
