/**
 * @file        VccbAiTypes.h
 * @brief       Shared POD types for RRC State Predictor (AI) — VCCB + itscp
 *
 * Lives under Base/VccbItscpDefine so VccbStruct.h (callInfo_type.aiCtx) does
 * not pull in VCCB_MAIN/hdr/VccbAiEngine.h. Engine APIs stay in VccbAiEngine.h.
 */
#ifndef VCCB_VccbAiTypes_H_
#define VCCB_VccbAiTypes_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VCCB_AI_HISTORY_LEN 10
#define VCCB_AI_HIDDEN_SIZE 16
#define VCCB_AI_STATIC_INPUT 14
#define VCCB_AI_SEQ_INPUT 2

// ---- resumeCause model encoding (train/serve contract) ----
// The ML pipeline (preprocess.py / synth_data_gen.py / gen_realish_csv.py and
// the gtest eval-set generator) encodes resumeCause as the compact 0..5 codes
// below. The over-the-air NRRRC_ResumeCause enum follows 3GPP 38.331 ASN.1
// ordering (emergency=0, highPriorityAccess=1, mt_Access=2, mo_Signalling=3,
// mo_Data=4, ..., rna_Update=8) which does NOT match, so every value stored in
// aiCtx.history[].resumeCause / emitted to the data-collect CSV MUST first go
// through FnVCCBAiClient_MapResumeCause() (VccbAiClient.h). LOCKSTEP: keep in
// sync with CAUSE_* in ML/preprocess.py and ML/synth_data_gen.py.
#define VCCB_AI_CAUSE_EMERGENCY   0
#define VCCB_AI_CAUSE_MO_DATA     1
#define VCCB_AI_CAUSE_MO_SIGNAL   2
#define VCCB_AI_CAUSE_MT_ACCESS   3
#define VCCB_AI_CAUSE_RNA_UPDATE  4
#define VCCB_AI_CAUSE_OTHERS      5

// ---- Drift-detection histogram bin counts (shared by VCCB writer, SHM layout,
//      itscp reader, and the ML baseline export so all four agree on binning) ----
//   5QI  : index = clamp(round(stat[0]), 0, N-1)         -> categorical 0..9
//   RSRP : index = clamp(stat[6] / (128/N), 0, N-1)      -> width 16 over 0..127
//   prob : index = clamp(resume_prob * N, 0, N-1)        -> width 0.1 over 0..1
#define VCCB_AI_DRIFT_FIVEQI_BINS 10
#define VCCB_AI_DRIFT_RSRP_BINS   8
#define VCCB_AI_DRIFT_PROB_BINS   10

typedef struct VccbAiInput_s {
    float seq[VCCB_AI_HISTORY_LEN][VCCB_AI_SEQ_INPUT];
    float stat[VCCB_AI_STATIC_INPUT];
} VccbAiInput;

typedef struct VccbAiOutput_s {
    float resume_prob;
    float resume_time;
} VccbAiOutput;

typedef struct VccbAiHistory_s {
    float durationSec;
    int32_t resumeCause;
} VccbAiHistory_type;

typedef struct VccbAiCallContext_s {
    VccbAiHistory_type history[VCCB_AI_HISTORY_LEN];
    int32_t writeIdx;
    int32_t count;

    float avgDurationSec;
    float resumeSuccessRate;
    int32_t resumeCount24h;

    int64_t resumeTs[VCCB_AI_HISTORY_LEN];
    int32_t resumeTsWriteIdx;

    // Handover 24h sliding window (mobility proxy, stat[4]). Mirrors the resume
    // 24h ring: each successful SA handover of this UE pushes a timestamp.
    int32_t handoverCount24h;
    int64_t handoverTs[VCCB_AI_HISTORY_LEN];
    int32_t handoverTsWriteIdx;

    int64_t inactiveEntryTs;
    int32_t extendCount;
    float lastInferTimerSec;

    // Timer decided by Point 2 when it defers the INACTIVE transition. While a
    // deferral window is open (>0), Point 1 returns this value instead of
    // re-inferring, so the BearerContextModification triggered by the deferral
    // carries exactly the timer the defer decision was based on. Cleared on
    // Point-2 proceed, INACTIVE entry, resume (Point 3), and data-activity
    // restart. Runtime-only: not persisted.
    float pendingExtendTimerSec;

    // Reserved scratch for the GRU hidden state. NOTE: the current engine
    // (FnVCCBAiEngine_Infer) is stateless — it re-runs the GRU over the whole
    // seq[] window from h=0 on every inference and never reads/writes this
    // buffer. Kept only to avoid a layout change; do not rely on its contents.
    float h[VCCB_AI_HIDDEN_SIZE];

    bool     persistRestored;
    uint64_t persistKey;
} VccbAiCallContext_type;

#ifdef __cplusplus
}
#endif

#endif // VCCB_VccbAiTypes_H_
