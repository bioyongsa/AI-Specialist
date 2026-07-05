// ============================================================
// VccbAiEngine.h
// GRU feedforward inference engine
// ============================================================
#ifndef VCCB_VccbAiEngine_H_
#define VCCB_VccbAiEngine_H_

#include "VccbAiTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- Engine ----
// NOTE: C++-only header. FnVCCBAiEngine_Infer uses C++ reference parameters, so
// despite the extern "C" linkage block this header cannot be included from a C
// translation unit (per §2-1-1 only VCCB C++ inference code includes it; the
// shared C-compatible boundary is VccbAiTypes.h).
void FnVCCBAiEngine_Init(void);
bool FnVCCBAiEngine_Infer(const VccbAiInput &input, VccbAiOutput &output);

// ---- UE context initialization ----
// Must be called immediately after 0xFF memset in VLIB_InitCallInfo.
// Prevents h[]/resumeSuccessRate from becoming NaN.
void VccbAi_InitContext(VccbAiCallContext_type *ctx);

#ifdef __cplusplus
}
#endif

#endif // VCCB_VccbAiEngine_H_
