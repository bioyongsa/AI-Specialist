// ============================================================
// VccbAiClient.h
// VCCB <-> AI Engine business logic layer
// ============================================================
#ifndef VCCB_VccbAiClient_H_
#define VCCB_VccbAiClient_H_

#include <stdint.h>
#include <stdbool.h>

struct callInfo_s;
typedef struct callInfo_s callInfo_type;

#ifdef __cplusplus
extern "C" {
#endif

// ---- Timer decision policy constants ----
#define VCCB_AI_KEEP_THRESHOLD          0.6f
#define VCCB_AI_RELEASE_THRESHOLD       0.3f
#define VCCB_AI_MARGIN_FACTOR           1.2f
#define VCCB_AI_TIMER_MIN_SEC           1
#define VCCB_AI_TIMER_MAX_SEC           3600
#define VCCB_AI_MAX_EXTEND_COUNT        3
#define VCCB_AI_FALLBACK_COUNT          10
#define VCCB_AI_INFER_TIMEOUT_US        1000    // Inference timeout: 1ms
#define VCCB_AI_24H_WINDOW_SEC          86400   // 24 hours (seconds)

// ---- Global AI inference statistics (thread-safe: read-only for output) ----
typedef struct VccbAiGlobalStats_s {
    int64_t totalRequests;          // Total inference requests
    int64_t successCount;           // Success count
    int64_t failureCount;           // Inference failure (error/NaN) count
    int64_t timeoutCount;           // Timeout (>1ms) count
    int64_t fallbackCount;          // Fallback response count
    int64_t triggerPt1Count;        // Point 1 trigger count
    int64_t triggerPt2Count;        // Point 2 trigger count
    int64_t triggerPt3Count;        // Point 3 trigger count
    int64_t triggerPt4Count;        // Point 4 trigger count
    int64_t triggerPt5Count;        // Point 5 trigger count
    int64_t totalInferUs;           // Total inference time (us, success only)
    int64_t extendCapHitCount;      // Point 2 extend cap (MAX_EXTEND) forced-INACTIVE count (guardrail)
    int32_t consecutiveFailCount;   // Current consecutive failure count
    bool    isFallbackMode;         // Current Fallback mode status
} VccbAiGlobalStats_type;

// ---- Point 1 (MsgE1apMake.cpp) ----
// Determine Inactivity Timer value when entering INACTIVE.
// Returns defaultTimerSec as-is on failure/fallback/REDCAP/NRDC.
int32_t FnVCCBAiClient_RecommendInactivityTimer(callInfo_type *ptCi,
                                              int32_t defaultTimerSec);

// ---- Point 2 (FeatRrcInactive.cpp) ----
// INACTIVE transition availability. Returns false when extendCount is exceeded.
bool    FnVCCBAiClient_IsAbleToProceedInactive(callInfo_type *ptCi);

// ---- INACTIVE entry stamp (StateSaRrcInact.cpp: FnVCCB_FinalizeRRCSuspend) ----
// Record the moment the UE actually (re-)enters RRC_INACTIVE. This is the start
// time for the resume-duration label consumed at Point 3, so it MUST be stamped
// at the real transition (suspend finalize), NOT at the earlier decision points
// (Point 1 timer recommend / Point 2 extend), which would over-count the idle
// duration. Observation-only: runs regardless of the AI master-enable flag.
void    FnVCCBAiClient_OnInactiveEntry(callInfo_type *ptCi);

// ---- Point 3 (StateSaRrcInact.cpp) ----
// Push Resume history to history[] circular buffer and update statistics.
// resumeCause is the raw over-the-air NRRRC_ResumeCause value; it is converted
// internally to the ML 0..5 encoding (FnVCCBAiClient_MapResumeCause) before
// being stored/emitted. Skips history/statistics accumulation entirely when
// aiCtx.inactiveEntryTs == 0 (no measured INACTIVE entry -> the duration label
// would be a bogus 0s and poison the model input); runtime counters are still
// reset in that case.
void    FnVCCBAiClient_OnResumeRequest(callInfo_type *ptCi, int32_t resumeCause);

// Map a raw NRRRC_ResumeCause enum value to the ML model encoding
// (VCCB_AI_CAUSE_* in VccbAiTypes.h; unknown/unmapped -> VCCB_AI_CAUSE_OTHERS).
// Exposed for the data-collect sweep test; production callers go through
// FnVCCBAiClient_OnResumeRequest which applies it internally.
int32_t FnVCCBAiClient_MapResumeCause(int32_t nrrrcResumeCause);

// ---- Data activity restarted while still RRC_CONNECTED ----
// Called from the E1 InactivityNotification(ACTIVE/DATA_RESTART) path. This is
// the success outcome of a Point-2 deferral (the UE became active again without
// ever suspending), so the extension budget (extendCount) and the pending
// extension timer must be cleared for the NEXT independent idle period.
// Observation-only: runs regardless of the AI master-enable flag.
void    FnVCCBAiClient_OnDataActivityResumed(callInfo_type *ptCi);

// ---- Point 4 (FeatRrcInactive.cpp) ----
// Query AI RNA size recommendation on RNA Update and output log.
void    FnVCCBAiClient_OnRnaUpdate(callInfo_type *ptCi);

// ---- Point 5 (StateSaHo.cpp) ----
// After receiving XnAP HandoverRequest, record UE history as aiCtx seed data.
void    FnVCCBAiClient_SeedFromXnapHistory(callInfo_type *ptCi);

// Record a successful SA handover of this UE into the 24h mobility window
// (stat[4] handover_count_24h). Co-located with the existing HO_Succ stat peg
// (StateSaHo.cpp / StateCommon.cpp). Observation-only: runs regardless of the
// AI master-enable flag, mirroring FnVCCBAiClient_OnResumeRequest bookkeeping.
void    FnVCCBAiClient_OnHandover(callInfo_type *ptCi);

// ---- Fallback management (global) ----
bool    FnVCCBAiClient_IsFallbackMode(void);
void    FnVCCBAiClient_Reset(void);

// ---- Master enable for AI active decisions (default ON) ----
// When disabled, Point 1/2 return legacy behavior (no timer override / no defer);
// observation (history + VCCB_AI_DATA_COLLECT logger) still runs.
void    FnVCCBAiClient_SetEnabled(bool enabled);
bool    FnVCCBAiClient_IsEnabled(void);

// ---- Test-only: force Point 2 (IsAbleToProceedInactive) decision ----
// Production default -1 (OFF: model decides). gtest uses this to drive the
// deterministic extension loop. -1=OFF, 0=force EXTEND(defer), 1=force PROCEED.
// Cleared automatically by FnVCCBAiClient_Reset().
void    FnVCCBAiClient_TestForceInactiveDecision(int32_t mode);

// ---- Statistics output (for itscp_vccb debug command) ----
void    FnVCCBAiClient_PrintStats(void);
void    FnVCCBAiClient_GetStats(VccbAiGlobalStats_type *out);

// ---- itscp cross-process exposure ("get stat ai") ----
// Bind the shared-memory snapshot slot (&VCCB_STAT->aiStat). Call once from
// MainInit after the VCCB_STAT SHM segment is allocated. The argument is a
// VccbAiStatShm_type* (void* keeps this C header free of VccbStruct.h).
// After binding, VCCB mirrors its process-local counters there so the separate
// itscp_vccb process can read them. No-op until bound.
void    FnVCCBAiClient_BindStatShm(void *statShm);

// ---- Governance / evaluation report (itscp: "ai report") ----
// Prints the layer-3 operational-health dashboard: client stats + derived
// guardrail rates (fallback/timeout/success/extend-cap) + persistent-store
// stats (restore hit/miss, saves, evictions). Read-only; safe to call anytime.
void    FnVCCBAiClient_PrintReport(void);

#ifdef __cplusplus
}
#endif

#endif  // VCCB_VccbAiClient_H_
