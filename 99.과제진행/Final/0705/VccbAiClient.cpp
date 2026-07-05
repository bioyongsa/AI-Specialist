// ============================================================
// VccbAiClient.cpp
// ============================================================
#include "ComInclude.h"
#include "ComLibUtility.h"

#include "VccbStruct.h"

#include "FeatCallInfo.h"

#include "VccbAiClient.h"
#include "VccbAiEngine.h"
#include "VccbAiPersistStore.h"

#include <string.h>     // memset
#include <math.h>       // ceilf
#include <time.h>       // time, localtime
#include <atomic>       // std::atomic (C++11)
#include <stdint.h>
#include <stdio.h>      // snprintf (drift histogram formatting)

#ifdef VCCB_AI_DATA_COLLECT
#include <stdio.h>      // fopen, fprintf  (training-data collection only)
#include <stdlib.h>     // getenv
#endif

// ---- Global state (thread-safe: std::atomic) ---- 
static std::atomic<int32_t> g_consecutiveFailCount(0);
static std::atomic<bool>    g_fallbackMode(false);

// Master enable for AI active decisions (Point 1 timer override, Point 2 defer).
// Default ON for production/verification builds. Set false via FnVCCBAiClient_SetEnabled()
// to preserve legacy call flow (e.g. gtest data collection without AI altering labels).
// Observation paths (history update + VCCB_AI_DATA_COLLECT logger) run regardless.
static std::atomic<bool>    g_aiEnabled(true);

// ---- Test-only deterministic override for Point 2 decision ----
// Production default -1 (OFF): the real AI inference decides extend/proceed.
// Set by FnVCCBAiClient_TestForceInactiveDecision() in gtest so the
// extension loop (defer -> BearerContextModification -> ... -> suspend)
// can be exercised without depending on the probabilistic model output.
//   -1 = OFF (use model)   0 = force EXTEND (defer)   1 = force PROCEED (INACTIVE)
static std::atomic<int32_t> g_testForcedInactiveDecision(-1);

// ---- Global inference statistics ----
static std::atomic<int64_t> g_totalRequests(0);
static std::atomic<int64_t> g_successCount(0);
static std::atomic<int64_t> g_failureCount(0);
static std::atomic<int64_t> g_timeoutCount(0);
static std::atomic<int64_t> g_fallbackCount(0);
static std::atomic<int64_t> g_triggerPt1(0);
static std::atomic<int64_t> g_triggerPt2(0);
static std::atomic<int64_t> g_triggerPt3(0);
static std::atomic<int64_t> g_triggerPt4(0);
static std::atomic<int64_t> g_triggerPt5(0);
static std::atomic<int64_t> g_totalInferUs(0);
static std::atomic<int64_t> g_extendCapHit(0);    // Point 2 forced-INACTIVE at MAX_EXTEND (guardrail)

// ---- Input/output distribution histograms (drift detection) ----
// Cumulative bin counts of what the model actually saw (5QI, RSRP) and emitted
// (resume_prob), updated on every successful inference. Mirrored to SHM by
// publishStatsToShm() so the drift monitor can PSI-compare against a baseline.
static std::atomic<uint64_t> g_distFiveQi[VCCB_AI_DRIFT_FIVEQI_BINS];
static std::atomic<uint64_t> g_distRsrp[VCCB_AI_DRIFT_RSRP_BINS];
static std::atomic<uint64_t> g_distProb[VCCB_AI_DRIFT_PROB_BINS];
static std::atomic<uint64_t> g_distSamples(0);

// ---- itscp cross-process exposure (get stat ai) ----
// VCCB and itscp_vccb are separate processes, so the std::atomic counters above
// are NOT visible to itscp. MainInit binds &VCCB_STAT->aiStat (shared memory)
// here once at boot; publishStatsToShm() then mirrors a snapshot on every AI
// entry point. NULL until bound (e.g. gtest, or boot-time FnVCCBAiClient_Reset()
// which runs before SHM alloc) -> publish is a safe no-op.
static VccbAiStatShm_type *g_aiStatShm = NULL;

static void publishStatsToShm(void)
{
    VccbAiStatShm_type *p = g_aiStatShm;
    if (p == NULL) return;

    p->enabled              = g_aiEnabled.load() ? 1 : 0;
    p->fallbackMode         = g_fallbackMode.load() ? 1 : 0;
    p->consecutiveFailCount = g_consecutiveFailCount.load();
    p->lastUpdateTs         = (int64_t)time(NULL);

    p->totalRequests = g_totalRequests.load();
    p->successCount  = g_successCount.load();
    p->failureCount  = g_failureCount.load();
    p->timeoutCount  = g_timeoutCount.load();
    p->fallbackCount = g_fallbackCount.load();
    p->totalInferUs  = g_totalInferUs.load();

    p->extendCapHitCount = g_extendCapHit.load();
    p->triggerPt1Count   = g_triggerPt1.load();
    p->triggerPt2Count   = g_triggerPt2.load();
    p->triggerPt3Count   = g_triggerPt3.load();
    p->triggerPt4Count   = g_triggerPt4.load();
    p->triggerPt5Count   = g_triggerPt5.load();

    VccbAiPersistStats_type ps;
    FnVCCBAiPersist_GetStats(&ps);
    p->persistEntries     = ps.entries;
    p->persistRestoreHit  = ps.restoreHit;
    p->persistRestoreMiss = ps.restoreMiss;
    p->persistSaves       = ps.saves;
    p->persistEvictions   = ps.evictions;
    p->persistNoKey       = ps.noKey;

    // ---- distribution histograms (drift detection) ----
    p->distSamples = g_distSamples.load(std::memory_order_relaxed);
    for (int i = 0; i < VCCB_AI_DRIFT_FIVEQI_BINS; i++)
        p->distFiveQi[i] = g_distFiveQi[i].load(std::memory_order_relaxed);
    for (int i = 0; i < VCCB_AI_DRIFT_RSRP_BINS; i++)
        p->distRsrp[i] = g_distRsrp[i].load(std::memory_order_relaxed);
    for (int i = 0; i < VCCB_AI_DRIFT_PROB_BINS; i++)
        p->distProb[i] = g_distProb[i].load(std::memory_order_relaxed);

    p->valid = 1;
}

// Bin one successful inference into the drift histograms. Binning MUST match
// VccbAiTypes.h comments and ML/export_baseline.py so PSI is comparable.
static void recordDriftSample(const VccbAiInput &in, const VccbAiOutput &out)
{
    int q = (int)(in.stat[0] + 0.5f);                 // 5QI -> nearest bin
    if (q < 0) q = 0;
    if (q >= VCCB_AI_DRIFT_FIVEQI_BINS) q = VCCB_AI_DRIFT_FIVEQI_BINS - 1;
    g_distFiveQi[q].fetch_add(1, std::memory_order_relaxed);

    int r = (int)in.stat[6] / (128 / VCCB_AI_DRIFT_RSRP_BINS);   // RSRP -> width-16 bin
    if (r < 0) r = 0;
    if (r >= VCCB_AI_DRIFT_RSRP_BINS) r = VCCB_AI_DRIFT_RSRP_BINS - 1;
    g_distRsrp[r].fetch_add(1, std::memory_order_relaxed);

    int b = (int)(out.resume_prob * (float)VCCB_AI_DRIFT_PROB_BINS);  // prob -> width-0.1 bin
    if (b < 0) b = 0;
    if (b >= VCCB_AI_DRIFT_PROB_BINS) b = VCCB_AI_DRIFT_PROB_BINS - 1;
    g_distProb[b].fetch_add(1, std::memory_order_relaxed);

    g_distSamples.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================
// Helpers
// ============================================================

// Convert callInfo_type to VccbAiInput
static bool buildAiInput(const callInfo_type *ptCi, VccbAiInput *input)
{
    if (ptCi == NULL || input == NULL) return false;

    memset(input, 0, sizeof(*input));

    // ---- sequence feature (UE history circular buffer -> chronological order) ----
    const VccbAiCallContext_type *ctx = &ptCi->aiCtx;
    int32_t count = (ctx->count < VCCB_AI_HISTORY_LEN) ? ctx->count : VCCB_AI_HISTORY_LEN;

    // before wrap-around: start from 0; after wrap-around: writeIdx points to the oldest entry
    int32_t startIdx = (ctx->count < VCCB_AI_HISTORY_LEN) ? 0 : ctx->writeIdx;

    for (int32_t i = 0; i < count; i++)
    {
        int32_t idx = (startIdx + i) % VCCB_AI_HISTORY_LEN;

        input->seq[i][0] = ctx->history[idx].durationSec;
        input->seq[i][1] = (float)ctx->history[idx].resumeCause;
    }
    // remaining slots are zero-padded (already 0 from memset)

    // ---- static feature (14 items; was 15 before dropping duplicate stat[12]) ----
    // stat[0]: primary_5qi - Non-Dynamic 5QI of the first active DRB
    {
        uint8_t primary5qi = 0;
        for (int i = 0; i < MAX_DRB_PER_SA_CALL; i++)
        {
            const callDRBInfo_type &d = ptCi->drb[i];
            if (d.usage != CI_use) continue;

            uint16_t qIdx = d.drbQosIdxInCall;
            if (qIdx < MAX_QOS_FLOW_PER_CALL && ptCi->qosFlow[qIdx].usage == CI_use)
            {
                const qosFlowInfo_type &q = ptCi->qosFlow[qIdx];
                if (q.qosParam.qosCharacteristics == ci_qos_char_non_dynamic_5qi)
                {
                    primary5qi = q.qosParam.nonDyna.fiveQi;
                }
            }
            break;
        }
        input->stat[0] = (float)primary5qi;
    }

    // stat[1]: drb_count - Number of active DRBs
    {
        int32_t drbCount = 0;
        for (int i = 0; i < MAX_DRB_PER_SA_CALL; i++)
        {
            if (ptCi->drb[i].usage == CI_use) drbCount++;
        }
        input->stat[1] = (float)drbCount;
    }

    // stat[2]: ue_ambr_dl_mbps - UE DL AMBR (bps to Mbps)
    input->stat[2] = (float)((double)ptCi->ueInfo.ambr.aggrMbrDl / 1000000.0);

    // stat[3]: has_gbr_flow - GBR Flow existence (0/1)
    {
        float hasGbr = 0.0f;
        for (int i = 0; i < MAX_QOS_FLOW_PER_CALL; i++)
        {
            if (ptCi->qosFlow[i].usage == CI_use &&
                ptCi->qosFlow[i].qosParam.gbrQosInformationPresent != 0)
            {
                hasGbr = 1.0f;
                break;
            }
        }
        input->stat[3] = hasGbr;
    }

    // stat[4]: handover_count_24h - Number of successful SA handovers of this UE
    // in the 24h sliding window (mobility proxy). Updated by FnVCCBAiClient_OnHandover
    // at the existing HO_Succ stat-peg sites and persisted across calls (per UE).
    input->stat[4] = (float)ctx->handoverCount24h;

    // stat[5]: cn_stationary - UE stationary status provided by AMF (0/1).
    // Reserved: NGAP ExpectedUEBehaviour is not wired yet, so kept at 0.0.
    input->stat[5] = 0.0f;

    // stat[6]: last_rsrp - Serving Cell SSB RSRP (0..127, 255=invalid).
    // Invalid (no measurement yet) is filled with the training distribution
    // mean (75) rather than 0: 0 is a real "extremely weak signal" value, so
    // mean-fill keeps a missing measurement neutral to the model and out of
    // the weakest drift-histogram bin.
    {
        float rsrp = (float)ptCi->servCellSsbRsrp;
        if (ptCi->servCellSsbRsrp == 255) rsrp = 75.0f;
        input->stat[6] = rsrp;
    }

    // stat[7]: secondary_5qi - Non-Dynamic 5QI of the first active DRB whose 5QI
    // differs from primary_5qi (stat[0]). Captures mixed service-class sessions
    // (e.g. VoNR 5QI=1 + eMBB data 5QI=9 -> primary=1, secondary=9), which is the
    // signal behind service-group differentiated inactivity timers. Value is 0
    // when the session is single service-class (all active DRBs share primary_5qi)
    // or there is no qualifying second flow.
    {
        uint8_t primary5qi   = (uint8_t)(input->stat[0] + 0.5f);
        uint8_t secondary5qi = 0;
        for (int i = 0; i < MAX_DRB_PER_SA_CALL; i++)
        {
            const callDRBInfo_type &d = ptCi->drb[i];
            if (d.usage != CI_use) continue;

            uint16_t qIdx = d.drbQosIdxInCall;
            if (qIdx < MAX_QOS_FLOW_PER_CALL && ptCi->qosFlow[qIdx].usage == CI_use)
            {
                const qosFlowInfo_type &q = ptCi->qosFlow[qIdx];
                if (q.qosParam.qosCharacteristics == ci_qos_char_non_dynamic_5qi)
                {
                    uint8_t fq = q.qosParam.nonDyna.fiveQi;
                    if (fq != 0 && fq != primary5qi)
                    {
                        secondary5qi = fq;
                        break;
                    }
                }
            }
        }
        input->stat[7] = (float)secondary5qi;
    }

    // stat[8]: rna_cell_count - Number of TAIs in RNA (approximated by numInactTAI)
    input->stat[8] = (float)ptCi->ueInfo.RrcInactiveInfo.numInactTAI;

    // stat[9]: avg_inactive_duration_sec - Average INACTIVE duration (EMA, avgDurationSec).
    // (The former stat[12] avg_inactive_duration was an exact duplicate of this
    // value and was removed; STATIC_INPUT is now 14, hour/dow shifted to 12/13.)
    input->stat[9] = ctx->avgDurationSec;

    // stat[10]: resume_count_24h - Number of Resumes in 24h sliding window
    input->stat[10] = (float)ctx->resumeCount24h;

    // stat[11]: resume_success_rate
    input->stat[11] = ctx->resumeSuccessRate;

    // stat[12]: hour_of_day (0-23), stat[13]: day_of_week (0=Sun ~ 6=Sat)
    // localtime_r (not localtime): Call Tasks run on MAX_CALL_THREAD_NUM_VCCB
    // threads and plain localtime() returns a shared static buffer (data race).
    time_t t = time(NULL);
    struct tm tmv = {};
    if (localtime_r(&t, &tmv) != NULL)
    {
        input->stat[12] = (float)tmv.tm_hour;
        input->stat[13] = (float)tmv.tm_wday;
    }

    return true;
}

static bool shouldSkipAi(const callInfo_type *ptCi)
{
    if (ptCi == NULL)          return true;
    if (!g_aiEnabled)          return true;   // AI active decisions disabled (default)
    if (g_fallbackMode)      return true;
    if (IS_REDCAP_UE(ptCi))    return true;   // RedCap UE: AI not applied in initial implementation
    if (IS_NRDC_MN(ptCi))      return true;   // NRDC MN: Not subject to AI intervention
    if (IS_NRDC_SN(ptCi))      return true;   // NRDC SN: Not subject to AI intervention
    return false;
}

// Count a skip as "fallback" only when the AI is actually supposed to be active.
// An intentional master-disable (FnVCCBAiClient_SetEnabled(false)) is not a
// failure mode; counting it would show FallbackRate 100% on the dashboard and
// mask real fallback events.
static void noteSkipFallback(void)
{
    if (g_aiEnabled.load())
    {
        g_fallbackCount.fetch_add(1);
    }
}

// Lazily pull this UE's persisted inference state (history/stats) into aiCtx.
// Runs at most once per call, on the first AI inference, because the stable UE
// identity (5G-S-TMSI) is not populated yet when VLIB_InitCallInfo runs.
// Only restores when the in-call context is still empty (count==0) so we never
// clobber a fresh XnAP HO seed (Point 5) or history accumulated within the call.
static void maybeRestorePersisted(callInfo_type *ptCi)
{
    if (ptCi == NULL) return;
    if (ptCi->aiCtx.persistRestored) return;

    if (ptCi->aiCtx.count == 0)
    {
        FnVCCBAiPersist_Restore(ptCi);              // no-op if no key / no entry (caches persistKey)
    }
    else
    {
        // still cache the key so OnResumeRequest can Save() later
        ptCi->aiCtx.persistKey = FnVCCBAiPersist_DeriveKey(ptCi);
    }

    // Mark the restore as done only once a stable UE identity was derivable.
    // If the 5G-S-TMSI is not populated yet at the first AI touch (key==0),
    // leave the flag false so a later inference retries the restore instead of
    // permanently cold-starting this call.
    ptCi->aiCtx.persistRestored = (ptCi->aiCtx.persistKey != 0);
}

// One inference attempt failed. Exactly ONE of failureCount/timeoutCount is
// pegged (plus one fallbackCount) so the itscp rates stay additive; the caller
// must not peg anything else for the same attempt.
static void noteInferFailure(bool isTimeout)
{
    int32_t cnt = g_consecutiveFailCount.fetch_add(1) + 1;
    if (isTimeout) g_timeoutCount.fetch_add(1);
    else           g_failureCount.fetch_add(1);
    g_fallbackCount.fetch_add(1);

    if (cnt >= VCCB_AI_FALLBACK_COUNT)
    {
        g_fallbackMode.store(true);
    }
}

static void noteInferSuccess(void)
{
    g_consecutiveFailCount.store(0);
    g_successCount.fetch_add(1);
}

// Result of one guarded inference attempt (counters are pegged by the caller
// via noteInferFailure/noteInferSuccess exactly once per attempt).
enum AiInferResult
{
    AI_INFER_OK = 0,
    AI_INFER_FAIL,          // engine rejected input (NaN/Inf) or errored
    AI_INFER_TIMEOUT        // exceeded VCCB_AI_INFER_TIMEOUT_US
};

// Run inference + 1ms timeout check. Pegs no global counters itself.
// elapsed_us_out: Actual inference elapsed time measured (us)
static AiInferResult runInferWithTimeout(const VccbAiInput &input, VccbAiOutput &output, int64_t *elapsed_us_out)
{
    struct timespec t_start = {}, t_end = {};
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    bool ok = FnVCCBAiEngine_Infer(input, output);

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    int64_t elapsed_us = (t_end.tv_sec  - t_start.tv_sec)  * 1000000LL
                       + (t_end.tv_nsec - t_start.tv_nsec) / 1000LL;

    if (elapsed_us_out != NULL)
    {
        *elapsed_us_out = elapsed_us;
    }

    if (!ok)
    {
        return AI_INFER_FAIL;
    }

    if (elapsed_us > VCCB_AI_INFER_TIMEOUT_US)
    {
        return AI_INFER_TIMEOUT;
    }

    // Accumulate statistics on success
    g_totalInferUs.fetch_add(elapsed_us);
    recordDriftSample(input, output);   // bin inputs/output for drift monitoring
    return AI_INFER_OK;
}

// ---- resumeCause: over-the-air NRRRC enum -> ML model 0..5 encoding ----
// The NRRRC_ResumeCause enum follows 3GPP 38.331 ASN.1 ordering (emergency=0,
// highPriorityAccess=1, mt_Access=2, mo_Signalling=3, mo_Data=4, ...,
// rna_Update=8) which does NOT match the compact encoding the model was
// trained on (preprocess.py CAUSE_*: 0=emergency, 1=mo_Data, 2=mo_Signalling,
// 3=mt_Access, 4=rna_Update, 5=others). Feeding raw enum values into
// seq[][1] silently mislabels e.g. mo_Data(4) as "rna_Update" — a train/serve
// skew. Every value stored into aiCtx.history[] or emitted to the
// data-collect CSV must pass through this map (LOCKSTEP: VccbAiTypes.h
// VCCB_AI_CAUSE_* / ML pipeline CAUSE_*).
int32_t FnVCCBAiClient_MapResumeCause(int32_t nrrrcResumeCause)
{
    switch (nrrrcResumeCause)
    {
        case (int32_t)NRRRC_ResumeCause_emergency:      return VCCB_AI_CAUSE_EMERGENCY;
        case (int32_t)NRRRC_ResumeCause_mo_Data:        return VCCB_AI_CAUSE_MO_DATA;
        case (int32_t)NRRRC_ResumeCause_mo_Signalling:  return VCCB_AI_CAUSE_MO_SIGNAL;
        case (int32_t)NRRRC_ResumeCause_mt_Access:      return VCCB_AI_CAUSE_MT_ACCESS;
        case (int32_t)NRRRC_rna_Update:                 return VCCB_AI_CAUSE_RNA_UPDATE;
        default:                                        return VCCB_AI_CAUSE_OTHERS;
    }
}

static int32_t clampTimer(int32_t t)
{
    if (t < VCCB_AI_TIMER_MIN_SEC) return VCCB_AI_TIMER_MIN_SEC;
    if (t > VCCB_AI_TIMER_MAX_SEC) return VCCB_AI_TIMER_MAX_SEC;
    return t;
}

// Count timestamps in the 24h sliding window ring (shared by resume/handover).
static int32_t calcCount24hFromTs(const int64_t ts[VCCB_AI_HISTORY_LEN])
{
    int64_t now = (int64_t)time(NULL);
    int64_t windowStart = now - VCCB_AI_24H_WINDOW_SEC;
    int32_t count = 0;

    for (int i = 0; i < VCCB_AI_HISTORY_LEN; i++)
    {
        if (ts[i] > windowStart)
        {
            count++;
        }
    }
    return count;
}

// Calculate the current valid Resume count in the 24h sliding window
static int32_t calcResumeCount24h(const VccbAiCallContext_type *ctx)
{
    return calcCount24hFromTs(ctx->resumeTs);
}

// ============================================================
// Public API
// ============================================================

int32_t FnVCCBAiClient_RecommendInactivityTimer(callInfo_type *ptCi,
                                              int32_t defaultTimerSec)
{
    g_totalRequests.fetch_add(1);
    g_triggerPt1.fetch_add(1);
    publishStatsToShm();            // expose latest counters via itscp (get stat ai)

    if (shouldSkipAi(ptCi))
    {
        noteSkipFallback();
        // Legacy value passes through UNCHANGED (no clamp): when the AI is
        // disabled/fallback/not applicable, the operator-configured static
        // timer must reach the wire exactly as before the AI was introduced.
        return defaultTimerSec;
    }

    // A Point-2 deferral window is open: the BearerContextModification being
    // built right now must carry the exact timer the defer decision computed,
    // not the result of a second, independent inference (which could disagree
    // — e.g. drift across an hour boundary yielding a mid-range prob and a
    // static default despite the just-made "extend" decision). The pending
    // value stays valid for every fill of the same deferral window (UE-level +
    // per-DRB fills of one message) and is cleared when the window closes
    // (proceed / INACTIVE entry / resume / data-activity restart).
    if (ptCi->aiCtx.pendingExtendTimerSec > 0.0f)
    {
        int32_t pending = clampTimer((int32_t)ptCi->aiCtx.pendingExtendTimerSec);
        ptCi->aiCtx.lastInferTimerSec = (float)pending;
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 1] using Point-2 extension timer=%ds (no re-inference)\n", pending);
        return pending;
    }

    maybeRestorePersisted(ptCi);    // pull cross-call UE history before inference

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    if (!buildAiInput(ptCi, &input))
    {
        noteInferFailure(false);
        return defaultTimerSec;     // fallback: static value unchanged
    }
    AiInferResult inferResult = runInferWithTimeout(input, output, &elapsed_us);
    if (inferResult != AI_INFER_OK)
    {
        noteInferFailure(inferResult == AI_INFER_TIMEOUT);
        return defaultTimerSec;     // fallback: static value unchanged
    }
    noteInferSuccess();

    // NOTE: inactiveEntryTs is NOT stamped here. Point 1 only recommends the E1
    // timer value during bearer setup; the UE has not entered RRC_INACTIVE yet.
    // The entry timestamp is stamped at the real transition in
    // FnVCCB_FinalizeRRCSuspend() via FnVCCBAiClient_OnInactiveEntry(), so the
    // Point-3 duration label is measured from actual INACTIVE entry.

    int32_t result;
    if (output.resume_prob >= VCCB_AI_KEEP_THRESHOLD)
    {
        // KEEP: expected resume time * margin factor
        float t = output.resume_time * VCCB_AI_MARGIN_FACTOR;
        result = (int32_t)ceilf(t);

        // extendCount is intentionally NOT incremented here.
        // Point 1 only recommends the E1 timer value and is invoked once PER DRB
        // and PER PDU session within a single BearerContextSetup/Modify, so
        // incrementing here would inflate extendCount to MAX in one setup and
        // force a premature INACTIVE transition at Point 2. The deferral counter
        // is owned by Point 2 (FnVCCBAiClient_IsAbleToProceedInactive), which is
        // the actual extension event.
    }
    else if (output.resume_prob <= VCCB_AI_RELEASE_THRESHOLD)
    {
        // EARLY_RELEASE
        result = VCCB_AI_TIMER_MIN_SEC;
    }
    else
    {
        // DEFAULT: mid-range probability -> keep static timer. Deliberately not
        // clamped: this branch returns the operator-configured value, and the
        // AI must not silently alter configuration it chose not to override.
        ptCi->aiCtx.lastInferTimerSec = (float)defaultTimerSec;

        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 1] resume_prob=%.3f resume_time=%.1fs -> timer=%ds (static default) (latency=%lldus)\n",
            (double)output.resume_prob, (double)output.resume_time, defaultTimerSec,
            (long long)elapsed_us);

        return defaultTimerSec;
    }

    // KEEP / EARLY_RELEASE: AI-derived values are clamped to the 1..3600s range.
    result = clampTimer(result);
    ptCi->aiCtx.lastInferTimerSec = (float)result;

    VCCB_PRINT_CALL_INFORMATIVE(ptCi,
        "[AI 1] resume_prob=%.3f resume_time=%.1fs -> timer=%ds (latency=%lldus)\n",
        (double)output.resume_prob, (double)output.resume_time, result, (long long)elapsed_us);

    return result;
}

bool FnVCCBAiClient_IsAbleToProceedInactive(callInfo_type *ptCi)
{
    g_totalRequests.fetch_add(1);
    g_triggerPt2.fetch_add(1);
    publishStatsToShm();            // expose latest counters via itscp (get stat ai)

    if (ptCi == NULL) return true;

    // extendCount exceeds upper limit -> Force immediate INACTIVE transition without AI judgment, reset counter
    if (ptCi->aiCtx.extendCount >= VCCB_AI_MAX_EXTEND_COUNT)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 2] extendCount(%d) >= MAX(%d). Forced INACTIVE transition.\n",
            ptCi->aiCtx.extendCount, VCCB_AI_MAX_EXTEND_COUNT);
        ptCi->aiCtx.extendCount = 0;
        ptCi->aiCtx.pendingExtendTimerSec = 0.0f;   // deferral window closed
        g_extendCapHit.fetch_add(1);
        return true;
    }

    // Test-only deterministic override (production default -1 -> no effect).
    // Placed AFTER the extendCount cap so the loop still terminates naturally
    // at VCCB_AI_MAX_EXTEND_COUNT even while EXTEND is being forced.
    {
        int32_t forced = g_testForcedInactiveDecision.load();
        if (forced == 0)            // force EXTEND (defer INACTIVE)
        {
            ptCi->aiCtx.extendCount++;
            VCCB_PRINT_CALL_INFORMATIVE(ptCi,
                "[AI 2][TEST] forced EXTEND -> Defer INACTIVE. extendCount=%d\n",
                ptCi->aiCtx.extendCount);
            return false;
        }
        if (forced == 1)            // force PROCEED (INACTIVE transition)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[AI 2][TEST] forced PROCEED -> INACTIVE.\n");
            return true;
        }
    }

    if (shouldSkipAi(ptCi))
    {
        noteSkipFallback();
        return true;
    }

    maybeRestorePersisted(ptCi);    // pull cross-call UE history before inference

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    if (!buildAiInput(ptCi, &input))
    {
        noteInferFailure(false);
        return true;    // Fallback: Proceed with existing INACTIVE transition
    }
    AiInferResult inferResult = runInferWithTimeout(input, output, &elapsed_us);
    if (inferResult != AI_INFER_OK)
    {
        noteInferFailure(inferResult == AI_INFER_TIMEOUT);
        return true;    // Fallback: Proceed with existing INACTIVE transition
    }
    noteInferSuccess();

    if (output.resume_prob >= VCCB_AI_KEEP_THRESHOLD)
    {
        // Timer extension: Store inference result in aiCtx and return false so caller performs BearerContextModify
        float t = output.resume_time * VCCB_AI_MARGIN_FACTOR;
        int32_t timerSec = (int32_t)ceilf(t);
        if (timerSec < VCCB_AI_TIMER_MIN_SEC) timerSec = VCCB_AI_TIMER_MIN_SEC;
        if (timerSec > VCCB_AI_TIMER_MAX_SEC) timerSec = VCCB_AI_TIMER_MAX_SEC;

        ptCi->aiCtx.lastInferTimerSec = (float)timerSec;
        // Open the deferral window: Point 1 (invoked while building the
        // BearerContextModification this deferral triggers) returns this value
        // instead of re-inferring, so decision and wire value cannot diverge.
        ptCi->aiCtx.pendingExtendTimerSec = (float)timerSec;
        ptCi->aiCtx.extendCount++;
        // inactiveEntryTs is stamped at the real transition (FnVCCB_FinalizeRRCSuspend),
        // not on this extend/defer decision, so the Point-3 duration is not inflated.

        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 2] resume_prob=%.3f >= %.1f -> Defer INACTIVE. extendCount=%d timer=%ds (latency=%lldus)\n",
            (double)output.resume_prob, (double)VCCB_AI_KEEP_THRESHOLD,
            ptCi->aiCtx.extendCount, timerSec, (long long)elapsed_us);

        return false;   // INACTIVE transition deferred: Caller sends BearerContextModification
    }

    ptCi->aiCtx.pendingExtendTimerSec = 0.0f;   // proceeding: deferral window closed

    VCCB_PRINT_CALL_INFORMATIVE(ptCi,
        "[AI 2] resume_prob=%.3f < %.1f -> Proceed INACTIVE. (latency=%lldus)\n",
        (double)output.resume_prob, (double)VCCB_AI_KEEP_THRESHOLD, (long long)elapsed_us);

    return true;    // Low resume_prob -> Proceed with immediate INACTIVE transition
}

void FnVCCBAiClient_OnInactiveEntry(callInfo_type *ptCi)
{
    // Stamp the moment the UE actually (re-)enters RRC_INACTIVE. Called from
    // FnVCCB_FinalizeRRCSuspend() for both the initial CONNECTED->INACTIVE
    // transition and INACTIVE->INACTIVE re-suspend. Point 3 measures the resume
    // duration as (resume time - inactiveEntryTs), so stamping here (rather than
    // at the earlier Point 1/2 decisions) removes the over-count between the
    // transition decision and the real suspend completion.
    //
    // Observation-only: intentionally NOT gated by shouldSkipAi()/the master
    // enable flag, since the resume-history label is collected even when AI
    // active decisions are disabled (matches the always-on observation design).
    if (ptCi == NULL) return;
    ptCi->aiCtx.inactiveEntryTs = (int64_t)time(NULL);
    ptCi->aiCtx.pendingExtendTimerSec = 0.0f;   // deferral window closed by real transition
}

void FnVCCBAiClient_OnDataActivityResumed(callInfo_type *ptCi)
{
    // Data activity restarted while the UE is still RRC_CONNECTED (E1
    // InactivityNotification ACTIVE -> DATA_RESTART). If a Point-2 deferral
    // window was open this is its SUCCESS outcome: the extension kept the UE
    // connected until traffic returned, without a suspend/resume cycle. No
    // RRCResumeRequest will ever arrive for this cycle, so the extension
    // budget must be recycled here — otherwise leftover extendCount silently
    // depletes the deferral budget of the NEXT independent idle period.
    //
    // Observation-only: NOT gated by shouldSkipAi()/master enable, mirroring
    // the other bookkeeping hooks.
    if (ptCi == NULL) return;

    if (ptCi->aiCtx.extendCount > 0)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 2] Data restarted while CONNECTED after %d extension(s) -> extension succeeded; reset budget.\n",
            ptCi->aiCtx.extendCount);
    }
    ptCi->aiCtx.extendCount = 0;
    ptCi->aiCtx.pendingExtendTimerSec = 0.0f;
}

#ifdef VCCB_AI_DATA_COLLECT
// ============================================================
// Training-data collection (enabled only with -DVCCB_AI_DATA_COLLECT)
// ------------------------------------------------------------
// Emits ONE CSV row per Resume event:
//   - feature vector exactly as built by buildAiInput() (raw, model contract)
//   - realized label: durationSec (how long it stayed INACTIVE) + resumeCause
// The feature snapshot is taken BEFORE this resume is pushed into history,
// i.e. it is the exact input the model would have seen at inference time.
//
// Output file: env VCCB_AI_DATA_FILE, default "/tmp/vccb_ai_dataset.csv".
// Run the gtest JSON suite repeatedly with different parameter values
// (qos5qi / AMBR / GBR flow / RSRP / RNA TAI ...) appending to the same file
// to build a diverse, realistic-range dataset.
// ============================================================
static void aiDataCollectEmit(const VccbAiInput &input,
                              float    labelDurationSec,
                              int32_t  labelResumeCause,
                              int32_t  callId)
{
    const char *path = getenv("VCCB_AI_DATA_FILE");
    if (path == NULL || path[0] == '\0')
    {
        path = "/tmp/vccb_ai_dataset.csv";
    }

    // Write the header only when the file does not yet exist.
    bool needHeader = false;
    {
        FILE *probe = fopen(path, "r");
        if (probe == NULL) needHeader = true;
        else               fclose(probe);
    }

    FILE *fp = fopen(path, "a");
    if (fp == NULL) return;

    if (needHeader)
    {
        fprintf(fp, "callId");
        for (int t = 0; t < VCCB_AI_HISTORY_LEN; t++)
        {
            fprintf(fp, ",seq%d_dur,seq%d_cause", t, t);
        }
        for (int i = 0; i < VCCB_AI_STATIC_INPUT; i++)
        {
            fprintf(fp, ",stat%d", i);
        }
        fprintf(fp, ",label_durationSec,label_resumeCause\n");
    }

    fprintf(fp, "%d", callId);
    for (int t = 0; t < VCCB_AI_HISTORY_LEN; t++)
    {
        fprintf(fp, ",%.6f,%.6f", input.seq[t][0], input.seq[t][1]);
    }
    for (int i = 0; i < VCCB_AI_STATIC_INPUT; i++)
    {
        fprintf(fp, ",%.6f", input.stat[i]);
    }
    fprintf(fp, ",%.6f,%d\n", labelDurationSec, labelResumeCause);

    fclose(fp);
}
#endif // VCCB_AI_DATA_COLLECT

void FnVCCBAiClient_OnResumeRequest(callInfo_type *ptCi, int32_t resumeCause)
{
    g_triggerPt3.fetch_add(1);

    if (ptCi == NULL) return;
    VccbAiCallContext_type *ctx = &ptCi->aiCtx;

    // Over-the-air NRRRC enum -> ML 0..5 encoding (train/serve contract).
    // Everything stored/emitted below uses the mapped code, never the raw enum.
    int32_t aiCause = FnVCCBAiClient_MapResumeCause(resumeCause);

    // ---- no measured INACTIVE entry -> skip accumulation entirely ----
    // inactiveEntryTs == 0 means this call context never went through the
    // suspend finalize stamp (e.g. inter-gNB resume on the target side). The
    // duration label would be a bogus 0s: pushing it would teach the model
    // "this UE resumes instantly" and drag avgDurationSec toward 0 — and the
    // poisoned history would survive across calls via the persist store.
    // Per design §1-3, skip history/EMA/CSV; still recycle runtime counters.
    if (ctx->inactiveEntryTs == 0)
    {
        ctx->extendCount = 0;
        ctx->pendingExtendTimerSec = 0.0f;
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 3] Resume(cause=%d) without INACTIVE entry stamp -> history update skipped.\n",
            aiCause);
        return;
    }

    // ---- calculate INACTIVE duration ----
    float durationSec;
    {
        int64_t now  = (int64_t)time(NULL);
        int64_t diff = now - ctx->inactiveEntryTs;
        if (diff < 0) diff = 0;
        durationSec = (float)diff;
    }

#ifdef VCCB_AI_DATA_COLLECT
    // Snapshot the model input BEFORE this resume mutates the history buffer,
    // then pair it with the realized (durationSec, mapped resumeCause) label.
    {
        VccbAiInput snapshot;
        if (buildAiInput(ptCi, &snapshot))
        {
            aiDataCollectEmit(snapshot, durationSec, aiCause, ptCi->callId);
        }
    }
#endif

    // ---- push to history circular buffer ----
    ctx->history[ctx->writeIdx].durationSec = durationSec;
    ctx->history[ctx->writeIdx].resumeCause = aiCause;
    ctx->writeIdx = (ctx->writeIdx + 1) % VCCB_AI_HISTORY_LEN;
    if (ctx->count < VCCB_AI_HISTORY_LEN)
    {
        ctx->count++;
    }

    // ---- Record 24h sliding window timestamp ----
    int64_t now = (int64_t)time(NULL);
    ctx->resumeTs[ctx->resumeTsWriteIdx] = now;
    ctx->resumeTsWriteIdx = (ctx->resumeTsWriteIdx + 1) % VCCB_AI_HISTORY_LEN;
    ctx->resumeCount24h = calcResumeCount24h(ctx);

    // ---- update statistics ----
    // avgDurationSec: exponential moving average (EMA, alpha=0.1)
    if (ctx->count == 1)
    {
        ctx->avgDurationSec = durationSec;          // first entry: use as initial value
    }
    else
    {
        ctx->avgDurationSec = 0.9f * ctx->avgDurationSec + 0.1f * durationSec;
    }

    // resumeSuccessRate: EMA (alpha=0.1)
    // Judgment criteria: emergency (Resume due to emergency call) is considered "prediction failure".
    // Others (mo_Data, mt_Access, rna_Update, etc.) are treated as "predictable Resume" and handled as success.
    {
        const float alpha = 0.1f;
        float isSuccess = (aiCause == VCCB_AI_CAUSE_EMERGENCY) ? 0.0f : 1.0f;
        ctx->resumeSuccessRate = alpha * isSuccess + (1.0f - alpha) * ctx->resumeSuccessRate;
    }

    // prepare for next INACTIVE cycle
    ctx->inactiveEntryTs        = 0;
    ctx->extendCount            = 0;
    ctx->pendingExtendTimerSec  = 0.0f;

    // Persist the updated longitudinal state so the next call from this UE
    // resumes its history instead of cold-starting (no-op if no UE key).
    FnVCCBAiPersist_Save(ptCi);

    publishStatsToShm();            // refresh persist-store stats for itscp (get stat ai)
}

void FnVCCBAiClient_OnHandover(callInfo_type *ptCi)
{
    if (ptCi == NULL) return;
    VccbAiCallContext_type *ctx = &ptCi->aiCtx;

    // Record a 24h sliding-window timestamp for this successful handover (mobility
    // proxy, stat[4]). Observation-only: runs regardless of the AI master-enable
    // flag, mirroring the resume 24h bookkeeping in FnVCCBAiClient_OnResumeRequest.
    int64_t now = (int64_t)time(NULL);
    ctx->handoverTs[ctx->handoverTsWriteIdx] = now;
    ctx->handoverTsWriteIdx = (ctx->handoverTsWriteIdx + 1) % VCCB_AI_HISTORY_LEN;
    ctx->handoverCount24h = calcCount24hFromTs(ctx->handoverTs);

    // Persist so the UE's mobility rate survives across calls (no-op w/o UE key).
    FnVCCBAiPersist_Save(ptCi);

    publishStatsToShm();
}

void FnVCCBAiClient_OnRnaUpdate(callInfo_type *ptCi)
{
    // totalRequests counts EVERY entry (including skips), matching Point 1/2 so
    // the itscp rate denominators mean the same thing at every trigger point.
    g_totalRequests.fetch_add(1);
    g_triggerPt4.fetch_add(1);
    publishStatsToShm();            // expose latest counters via itscp (get stat ai)

    if (ptCi == NULL) return;
    if (shouldSkipAi(ptCi))
    {
        noteSkipFallback();
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[AI 4] RNA Update: AI skipped (Fallback/REDCAP/NRDC).\n");
        return;
    }

    maybeRestorePersisted(ptCi);    // pull cross-call UE history before inference

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    if (!buildAiInput(ptCi, &input))
    {
        noteInferFailure(false);
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[AI 4] RNA Update: inference failed. Fallback.\n");
        return;
    }
    AiInferResult inferResult = runInferWithTimeout(input, output, &elapsed_us);
    if (inferResult != AI_INFER_OK)
    {
        noteInferFailure(inferResult == AI_INFER_TIMEOUT);
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[AI 4] RNA Update: inference failed. Fallback.\n");
        return;
    }
    noteInferSuccess();

    // RNA size recommendation: Based on resume_prob. Higher means recommend keeping/expanding RNA, lower means recommend shrinking RNA.
    // Since there is currently no mechanism in VCCB code to directly change RNA size parameters, only log is output.
    const char *rnaSizeRecommendation;
    if (output.resume_prob >= VCCB_AI_KEEP_THRESHOLD)
    {
        rnaSizeRecommendation = "KEEP_OR_EXPAND";
    }
    else if (output.resume_prob <= VCCB_AI_RELEASE_THRESHOLD)
    {
        rnaSizeRecommendation = "SHRINK";
    }
    else
    {
        rnaSizeRecommendation = "NO_CHANGE";
    }

    VCCB_PRINT_CALL_INFORMATIVE(ptCi,
        "[AI 4] RNA Update: resume_prob=%.3f resume_time=%.1fs -> RNA_size_recommendation=%s (latency=%lldus)\n",
        (double)output.resume_prob, (double)output.resume_time, rnaSizeRecommendation, (long long)elapsed_us);
}

void FnVCCBAiClient_SeedFromXnapHistory(callInfo_type *ptCi)
{
    g_triggerPt5.fetch_add(1);

    if (ptCi == NULL) return;
    VccbAiCallContext_type *ctx = &ptCi->aiCtx;

    // VXIL_Store_XN_UE_HistoryInformation() is called inside FnVCCB_StoreXNMessage(XN_HandoverRequest_id),
    // so this function should be called after visitedCell[] is populated.
    //
    // ueHistory.lastIdx: Index of the latest cell, incremented by VHCM_updateUEHistory().
    //   Initial value is -1, so valid range is 0 ~ MAX_VISITED_CELL-1.
    // visitedCell[idx].uRat.nrCell.stayTime: Stay time of the cell (seconds, max 4095s per 3GPP TS 38.413)
    //   If stayTimeEnhancedPresent is OPT_PRESENT, use stayTimeEnhanced (in 100ms units).

    if (ptCi->ueHistory.lastIdx >= MAX_VISITED_CELL)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 5] XnAP HO seed: no valid visitedCell history (lastIdx=%d).\n",
            (int)ptCi->ueHistory.lastIdx);
        return;
    }

    uint8_t seedIdx = ptCi->ueHistory.lastIdx;
    const visitedCellInfo_type *pVisited = &ptCi->ueHistory.visitedCell[seedIdx];

    if (pVisited->usage != CI_use || pVisited->ratType != ci_nr)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 5] XnAP HO seed: visitedCell[%d] usage=%d ratType=%d - skip.\n",
            (int)seedIdx, (int)pVisited->usage, (int)pVisited->ratType);
        return;
    }

    // If stayTimeEnhanced is present, use more precise value (unit: 100ms), otherwise use stayTime (seconds)
    float stayTimeSec;
    if (pVisited->uRat.nrCell.stayTimeEnhancedPresent == OPT_PRESENT)
    {
        stayTimeSec = (float)pVisited->uRat.nrCell.stayTimeEnhanced * 0.1f;  // 100ms to seconds
    }
    else
    {
        stayTimeSec = (float)pVisited->uRat.nrCell.stayTime;
    }

    // Record seed in AI history buffer. The cause slot uses VCCB_AI_CAUSE_OTHERS:
    // the model was trained on the 0..5 encoding only, so the former "-1 marker"
    // was an out-of-distribution input the GRU has never seen. OTHERS is the
    // trained catch-all and keeps the seed step in-distribution. (Note the dwell
    // itself is a *connected* stay time used as a rough prior for the inactive
    // dwell — an accepted approximation per design §1-5.)
    ctx->history[ctx->writeIdx].durationSec = stayTimeSec;
    ctx->history[ctx->writeIdx].resumeCause = VCCB_AI_CAUSE_OTHERS;
    ctx->writeIdx = (ctx->writeIdx + 1) % VCCB_AI_HISTORY_LEN;
    if (ctx->count < VCCB_AI_HISTORY_LEN)
    {
        ctx->count++;
    }

    // Update avgDurationSec (EMA)
    if (ctx->count == 1)
    {
        ctx->avgDurationSec = stayTimeSec;
    }
    else
    {
        ctx->avgDurationSec = 0.9f * ctx->avgDurationSec + 0.1f * stayTimeSec;
    }

    VCCB_PRINT_CALL_INFORMATIVE(ptCi,
        "[AI 5] XnAP HO seed: visitedCell[%d] stayTime=%.1fs -> aiCtx.avgDurationSec=%.1fs (count=%d)\n",
        (int)seedIdx, (double)stayTimeSec, (double)ctx->avgDurationSec, ctx->count);
}

bool FnVCCBAiClient_IsFallbackMode(void)
{
    return g_fallbackMode;
}

void FnVCCBAiClient_SetEnabled(bool enabled)
{
    g_aiEnabled.store(enabled);
}

bool FnVCCBAiClient_IsEnabled(void)
{
    return g_aiEnabled.load();
}

void FnVCCBAiClient_TestForceInactiveDecision(int32_t mode)
{
    g_testForcedInactiveDecision.store(mode);
}

void FnVCCBAiClient_Reset(void)
{
    g_consecutiveFailCount.store(0);
    g_fallbackMode.store(false);
    g_testForcedInactiveDecision.store(-1);     // clear test override
    FnVCCBAiPersist_Reset();                    // drop persisted UE state (test isolation)

    g_totalRequests.store(0);
    g_successCount.store(0);
    g_failureCount.store(0);
    g_timeoutCount.store(0);
    g_fallbackCount.store(0);
    g_triggerPt1.store(0);
    g_triggerPt2.store(0);
    g_triggerPt3.store(0);
    g_triggerPt4.store(0);
    g_triggerPt5.store(0);
    g_totalInferUs.store(0);
    g_extendCapHit.store(0);

    // clear drift histograms
    g_distSamples.store(0);
    for (int i = 0; i < VCCB_AI_DRIFT_FIVEQI_BINS; i++) g_distFiveQi[i].store(0);
    for (int i = 0; i < VCCB_AI_DRIFT_RSRP_BINS; i++)   g_distRsrp[i].store(0);
    for (int i = 0; i < VCCB_AI_DRIFT_PROB_BINS; i++)   g_distProb[i].store(0);

    publishStatsToShm();            // reflect the cleared counters in itscp (get stat ai)
}

// Bind the shared-memory AI stat snapshot slot (&VCCB_STAT->aiStat).
// Called once from MainInit AFTER the VCCB_STAT segment is allocated, so that
// subsequent publishStatsToShm() calls become visible to itscp_vccb.
void FnVCCBAiClient_BindStatShm(void *statShm)
{
    g_aiStatShm = (VccbAiStatShm_type *)statShm;
    publishStatsToShm();            // seed the slot with current (post-boot-reset) values
}

void FnVCCBAiClient_GetStats(VccbAiGlobalStats_type *out)
{
    if (out == NULL) return;

    out->totalRequests       = g_totalRequests.load();
    out->successCount        = g_successCount.load();
    out->failureCount        = g_failureCount.load();
    out->timeoutCount        = g_timeoutCount.load();
    out->fallbackCount       = g_fallbackCount.load();
    out->triggerPt1Count     = g_triggerPt1.load();
    out->triggerPt2Count     = g_triggerPt2.load();
    out->triggerPt3Count     = g_triggerPt3.load();
    out->triggerPt4Count     = g_triggerPt4.load();
    out->triggerPt5Count     = g_triggerPt5.load();
    out->totalInferUs        = g_totalInferUs.load();
    out->extendCapHitCount   = g_extendCapHit.load();
    out->consecutiveFailCount = g_consecutiveFailCount.load();
    out->isFallbackMode      = g_fallbackMode.load();
}

void FnVCCBAiClient_PrintStats(void)
{
    VccbAiGlobalStats_type s;
    FnVCCBAiClient_GetStats(&s);

    int64_t avgInferUs = (s.successCount > 0) ? (s.totalInferUs / s.successCount) : 0;

    VCCB_PRINTF("========== VCCB AI Client Statistics ==========\n");
    VCCB_PRINTF("  Mode            : %s\n",
                !FnVCCBAiClient_IsEnabled() ? "DISABLED" : (s.isFallbackMode ? "FALLBACK" : "ACTIVE"));
    VCCB_PRINTF("  Engine          : Scenario B (C++ GRU direct)\n");
    VCCB_PRINTF("  ConsecFailCount : %d\n",   (int)s.consecutiveFailCount);
    VCCB_PRINTF("  TotalRequests   : %lld\n", (long long)s.totalRequests);
    VCCB_PRINTF("  Success         : %lld\n", (long long)s.successCount);
    VCCB_PRINTF("  Failure(infer)  : %lld\n", (long long)s.failureCount);
    VCCB_PRINTF("  Timeout(>1ms)   : %lld\n", (long long)s.timeoutCount);
    VCCB_PRINTF("  Fallback(total) : %lld\n", (long long)s.fallbackCount);
    VCCB_PRINTF("  Trigger Pt 1    : %lld\n", (long long)s.triggerPt1Count);
    VCCB_PRINTF("  Trigger Pt 2    : %lld\n", (long long)s.triggerPt2Count);
    VCCB_PRINTF("  Trigger Pt 3    : %lld\n", (long long)s.triggerPt3Count);
    VCCB_PRINTF("  Trigger Pt 4    : %lld\n", (long long)s.triggerPt4Count);
    VCCB_PRINTF("  Trigger Pt 5    : %lld\n", (long long)s.triggerPt5Count);
    VCCB_PRINTF("  AvgInferLatency : %lldus\n", (long long)avgInferUs);
    VCCB_PRINTF("================================================\n");
}

void FnVCCBAiClient_PrintReport(void)
{
    VccbAiGlobalStats_type s;
    FnVCCBAiClient_GetStats(&s);

    VccbAiPersistStats_type p;
    FnVCCBAiPersist_GetStats(&p);

    // ---- derived guardrail rates (layer-3 operational health) ----
    // Denominator uses totalRequests (every Point 1/2/4 inference attempt).
    double denom        = (s.totalRequests > 0) ? (double)s.totalRequests : 1.0;
    double successRate  = 100.0 * (double)s.successCount  / denom;
    double failRate     = 100.0 * (double)s.failureCount  / denom;
    double timeoutRate  = 100.0 * (double)s.timeoutCount  / denom;
    double fallbackRate = 100.0 * (double)s.fallbackCount / denom;
    int64_t avgInferUs  = (s.successCount > 0) ? (s.totalInferUs / s.successCount) : 0;

    double restoreDenom = (double)(p.restoreHit + p.restoreMiss);
    if (restoreDenom < 1.0) restoreDenom = 1.0;
    double restoreHitRate = 100.0 * (double)p.restoreHit / restoreDenom;

    VCCB_PRINTF("=============== VCCB AI Governance / Evaluation Report ===============\n");
    VCCB_PRINTF("  Mode              : %s\n",
                !FnVCCBAiClient_IsEnabled() ? "DISABLED" : (s.isFallbackMode ? "FALLBACK" : "ACTIVE"));
    VCCB_PRINTF("  Engine            : Scenario B (C++ GRU direct, 1266 params)\n");
    VCCB_PRINTF("  --- Inference health (guardrails) ---\n");
    VCCB_PRINTF("  TotalRequests     : %lld\n", (long long)s.totalRequests);
    VCCB_PRINTF("  SuccessRate       : %.2f%% (%lld)\n", successRate,  (long long)s.successCount);
    VCCB_PRINTF("  FailureRate       : %.2f%% (%lld)\n", failRate,     (long long)s.failureCount);
    VCCB_PRINTF("  TimeoutRate(>1ms) : %.2f%% (%lld)\n", timeoutRate,  (long long)s.timeoutCount);
    VCCB_PRINTF("  FallbackRate      : %.2f%% (%lld)\n", fallbackRate, (long long)s.fallbackCount);
    VCCB_PRINTF("  ConsecFailCount   : %d\n",  (int)s.consecutiveFailCount);
    VCCB_PRINTF("  AvgInferLatency   : %lldus (budget %dus)\n",
                (long long)avgInferUs, (int)VCCB_AI_INFER_TIMEOUT_US);
    VCCB_PRINTF("  --- Policy guardrails ---\n");
    VCCB_PRINTF("  ExtendCapHit      : %lld   (Point2 forced INACTIVE at MAX_EXTEND=%d)\n",
                (long long)s.extendCapHitCount, (int)VCCB_AI_MAX_EXTEND_COUNT);
    VCCB_PRINTF("  Trigger Pt1/2/3   : %lld / %lld / %lld\n",
                (long long)s.triggerPt1Count, (long long)s.triggerPt2Count, (long long)s.triggerPt3Count);
    VCCB_PRINTF("  --- Persistent store (cross-call UE history) ---\n");
    VCCB_PRINTF("  ResidentEntries   : %llu\n", (unsigned long long)p.entries);
    VCCB_PRINTF("  RestoreHitRate    : %.2f%% (hit %llu / miss %llu)\n",
                restoreHitRate, (unsigned long long)p.restoreHit, (unsigned long long)p.restoreMiss);
    VCCB_PRINTF("  Saves / Evictions : %llu / %llu\n",
                (unsigned long long)p.saves, (unsigned long long)p.evictions);
    VCCB_PRINTF("  NoKey(no TMSI)    : %llu\n", (unsigned long long)p.noKey);

    // ---- input/output distribution (drift detection) ----
    char dbuf[256];
    size_t  dn;  // [cline] Changed from int to size_t to fix sign-conversion warning
    VCCB_PRINTF("  --- Input/Output distribution (drift) ---\n");
    VCCB_PRINTF("  DistSamples       : %llu\n",
                (unsigned long long)g_distSamples.load(std::memory_order_relaxed));
    dn = 0;
    for (int i = 0; i < VCCB_AI_DRIFT_FIVEQI_BINS; i++)
    {
        if (dn >= sizeof(dbuf)) break;  // [cline] Simplified check for size_t
        dn += (size_t)snprintf(dbuf + dn, sizeof(dbuf) - dn, "%s%llu", i ? "," : "",
                       (unsigned long long)g_distFiveQi[i].load(std::memory_order_relaxed));
    }
    VCCB_PRINTF("  Dist5QI(0..9)     : %s\n", dbuf);
    dn = 0;
    for (int i = 0; i < VCCB_AI_DRIFT_RSRP_BINS; i++)
    {
        if (dn >= sizeof(dbuf)) break;  // [cline] Simplified check for size_t
        dn += (size_t)snprintf(dbuf + dn, sizeof(dbuf) - dn, "%s%llu", i ? "," : "",
                       (unsigned long long)g_distRsrp[i].load(std::memory_order_relaxed));
    }
    VCCB_PRINTF("  DistRSRP(8x16)    : %s\n", dbuf);
    dn = 0;
    for (int i = 0; i < VCCB_AI_DRIFT_PROB_BINS; i++)
    {
        if (dn >= sizeof(dbuf)) break;  // [cline] Simplified check for size_t
        dn += (size_t)snprintf(dbuf + dn, sizeof(dbuf) - dn, "%s%llu", i ? "," : "",
                       (unsigned long long)g_distProb[i].load(std::memory_order_relaxed));
    }
    VCCB_PRINTF("  DistProb(10x0.1)  : %s\n", dbuf);
    VCCB_PRINTF("=====================================================================\n");
}
