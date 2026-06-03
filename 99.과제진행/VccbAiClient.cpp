// ============================================================
// VccbAiClient.cpp
// ============================================================
#include "ComInclude.h"
#include "ComLibUtility.h"

#include "VccbStruct.h"

#include "FeatCallInfo.h"

#include "VccbAiClient.h"
#include "VccbAiEngine.h"

#include <string.h>     // memset
#include <math.h>       // ceilf
#include <time.h>       // time, localtime
#include <atomic>       // std::atomic (C++11)
#include <stdint.h>

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

    // ---- static feature (15 items) ----
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

    // stat[4]: cn_expected_idle_period - Expected idle time provided by AMF
    // Currently kept at 0.0 as this field is not present in 5G NR NGAP CoreNetworkAssistanceInformation.
    // If added to NGAP spec in the future, check the path in ueInfo.RrcInactiveInfo series in VccbStruct.h.
    input->stat[4] = 0.0f;

    // stat[5]: cn_stationary - UE stationary status provided by AMF (0/1)
    // Currently kept at 0.0 for the same reason as stat[4].
    input->stat[5] = 0.0f;

    // stat[6]: last_rsrp - Serving Cell SSB RSRP (0..127, 255=invalid)
    {
        float rsrp = (float)ptCi->servCellSsbRsrp;
        if (ptCi->servCellSsbRsrp == 255) rsrp = 0.0f;
        input->stat[6] = rsrp;
    }

    // stat[7]: neighbor_cell_count - Number of neighbor cells
    // Field path in MeasurementReport storage structure (VccbStruct.h measResultNeighCell series) not yet confirmed.
    // Need to check and replace with the neighbor cell count field stored directly in ptCi from VccbStruct.h.
    input->stat[7] = 0.0f;

    // stat[8]: rna_cell_count - Number of TAIs in RNA (approximated by numInactTAI)
    input->stat[8] = (float)ptCi->ueInfo.RrcInactiveInfo.numInactTAI;

    // stat[9]: avg_cell_stay_sec - Average cell stay duration (reusing avgDurationSec)
    input->stat[9] = ctx->avgDurationSec;

    // stat[10]: resume_count_24h - Number of Resumes in 24h sliding window
    input->stat[10] = (float)ctx->resumeCount24h;

    // stat[11]: resume_success_rate
    input->stat[11] = ctx->resumeSuccessRate;

    // stat[12]: avg_inactive_duration
    input->stat[12] = ctx->avgDurationSec;

    // stat[13]: hour_of_day (0-23), stat[14]: day_of_week (0=Sun ~ 6=Sat)
    time_t t = time(NULL);
    struct tm *tmv = localtime(&t);
    if (tmv != NULL)
    {
        input->stat[13] = (float)tmv->tm_hour;
        input->stat[14] = (float)tmv->tm_wday;
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

static void noteInferFailure(void)
{
    int32_t cnt = g_consecutiveFailCount.fetch_add(1) + 1;
    g_failureCount.fetch_add(1);
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

// Run inference + 1ms timeout check. Returns false if exceeded (Fallback handling).
// elapsed_us_out: Actual inference elapsed time measured (us)
static bool runInferWithTimeout(const VccbAiInput &input, VccbAiOutput &output, int64_t *elapsed_us_out)
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
        return false;
    }

    if (elapsed_us > VCCB_AI_INFER_TIMEOUT_US)
    {
        g_timeoutCount.fetch_add(1);
        g_fallbackCount.fetch_add(1);
        return false;
    }

    // Accumulate statistics on success
    g_totalInferUs.fetch_add(elapsed_us);
    return true;
}

static int32_t clampTimer(int32_t t)
{
    if (t < VCCB_AI_TIMER_MIN_SEC) return VCCB_AI_TIMER_MIN_SEC;
    if (t > VCCB_AI_TIMER_MAX_SEC) return VCCB_AI_TIMER_MAX_SEC;
    return t;
}

// Calculate the current valid Resume count in the 24h sliding window
static int32_t calcResumeCount24h(const VccbAiCallContext_type *ctx)
{
    int64_t now = (int64_t)time(NULL);
    int64_t windowStart = now - VCCB_AI_24H_WINDOW_SEC;
    int32_t count = 0;

    for (int i = 0; i < VCCB_AI_HISTORY_LEN; i++)
    {
        if (ctx->resumeTs[i] > windowStart)
        {
            count++;
        }
    }
    return count;
}

// ============================================================
// Public API
// ============================================================

int32_t FnVCCBAiClient_RecommendInactivityTimer(callInfo_type *ptCi,
                                              int32_t defaultTimerSec)
{
    g_totalRequests.fetch_add(1);
    g_triggerPt1.fetch_add(1);

    if (shouldSkipAi(ptCi))
    {
        g_fallbackCount.fetch_add(1);
        return clampTimer(defaultTimerSec);
    }

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    if (!buildAiInput(ptCi, &input))
    {
        noteInferFailure();
        return clampTimer(defaultTimerSec);
    }
    if (!runInferWithTimeout(input, output, &elapsed_us))
    {
        noteInferFailure();
        return clampTimer(defaultTimerSec);
    }
    noteInferSuccess();

    // record INACTIVE entry timestamp (used at point 3 to calculate duration)
    ptCi->aiCtx.inactiveEntryTs = (int64_t)time(NULL);

    int32_t result;
    if (output.resume_prob >= VCCB_AI_KEEP_THRESHOLD)
    {
        // KEEP: expected resume time * margin factor
        float t = output.resume_time * VCCB_AI_MARGIN_FACTOR;
        result = (int32_t)ceilf(t);

        // [claude] extendCount is intentionally NOT incremented here.
        // [claude] Point 1 only recommends the E1 timer value and is invoked once PER DRB
        // [claude] and PER PDU session within a single BearerContextSetup/Modify, so
        // [claude] incrementing here would inflate extendCount to MAX in one setup and
        // [claude] force a premature INACTIVE transition at Point 2. The deferral counter
        // [claude] is owned by Point 2 (FnVCCBAiClient_IsAbleToProceedInactive), which is
        // [claude] the actual extension event.
    }
    else if (output.resume_prob <= VCCB_AI_RELEASE_THRESHOLD)
    {
        // EARLY_RELEASE
        result = VCCB_AI_TIMER_MIN_SEC;
    }
    else
    {
        // DEFAULT: mid-range probability -> keep static timer
        result = defaultTimerSec;
    }

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

    if (ptCi == NULL) return true;

    // extendCount exceeds upper limit -> Force immediate INACTIVE transition without AI judgment, reset counter
    if (ptCi->aiCtx.extendCount >= VCCB_AI_MAX_EXTEND_COUNT)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 2] extendCount(%d) >= MAX(%d). Forced INACTIVE transition.\n",
            ptCi->aiCtx.extendCount, VCCB_AI_MAX_EXTEND_COUNT);
        ptCi->aiCtx.extendCount = 0;
        return true;
    }

    if (shouldSkipAi(ptCi))
    {
        g_fallbackCount.fetch_add(1);
        return true;
    }

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    if (!buildAiInput(ptCi, &input) || !runInferWithTimeout(input, output, &elapsed_us))
    {
        noteInferFailure();
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
        ptCi->aiCtx.inactiveEntryTs   = (int64_t)time(NULL);
        ptCi->aiCtx.extendCount++;

        VCCB_PRINT_CALL_INFORMATIVE(ptCi,
            "[AI 2] resume_prob=%.3f >= %.1f -> Defer INACTIVE. extendCount=%d timer=%ds (latency=%lldus)\n",
            (double)output.resume_prob, (double)VCCB_AI_KEEP_THRESHOLD,
            ptCi->aiCtx.extendCount, timerSec, (long long)elapsed_us);

        return false;   // INACTIVE transition deferred: Caller sends BearerContextModification
    }

    VCCB_PRINT_CALL_INFORMATIVE(ptCi,
        "[AI 2] resume_prob=%.3f < %.1f -> Proceed INACTIVE. (latency=%lldus)\n",
        (double)output.resume_prob, (double)VCCB_AI_KEEP_THRESHOLD, (long long)elapsed_us);

    return true;    // Low resume_prob -> Proceed with immediate INACTIVE transition
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

    // ---- calculate INACTIVE duration ----
    float durationSec = 0.0f;
    if (ctx->inactiveEntryTs != 0)
    {
        int64_t now  = (int64_t)time(NULL);
        int64_t diff = now - ctx->inactiveEntryTs;
        if (diff < 0) diff = 0;
        durationSec = (float)diff;
    }

#ifdef VCCB_AI_DATA_COLLECT
    // Snapshot the model input BEFORE this resume mutates the history buffer,
    // then pair it with the realized (durationSec, resumeCause) label.
    {
        VccbAiInput snapshot;
        if (buildAiInput(ptCi, &snapshot))
        {
            aiDataCollectEmit(snapshot, durationSec, resumeCause, ptCi->callId);
        }
    }
#endif

    // ---- push to history circular buffer ----
    ctx->history[ctx->writeIdx].durationSec = durationSec;
    ctx->history[ctx->writeIdx].resumeCause = resumeCause;
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
    // Judgment criteria: NRRRC_emergency (Resume due to emergency call) is considered "prediction failure".
    // Others (mo_Data, mt_Access, rna_Update, etc.) are treated as "predictable Resume" and handled as success.
    {
        const float alpha = 0.1f;
        float isSuccess = ((int32_t)NRRRC_ResumeCause_emergency == resumeCause) ? 0.0f : 1.0f;
        ctx->resumeSuccessRate = alpha * isSuccess + (1.0f - alpha) * ctx->resumeSuccessRate;
    }

    // prepare for next INACTIVE cycle
    ctx->inactiveEntryTs = 0;
    ctx->extendCount     = 0;
}

void FnVCCBAiClient_OnRnaUpdate(callInfo_type *ptCi)
{
    g_triggerPt4.fetch_add(1);

    if (ptCi == NULL) return;
    if (shouldSkipAi(ptCi))
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[AI 4] RNA Update: AI skipped (Fallback/REDCAP/NRDC).\n");
        return;
    }

    VccbAiInput  input;
    VccbAiOutput output;
    int64_t elapsed_us = 0;

    g_totalRequests.fetch_add(1);

    if (!buildAiInput(ptCi, &input) || !runInferWithTimeout(input, output, &elapsed_us))
    {
        noteInferFailure();
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

    // Record seed in AI history buffer (resumeCause = -1: HO source cell history marker)
    ctx->history[ctx->writeIdx].durationSec = stayTimeSec;
    ctx->history[ctx->writeIdx].resumeCause = -1;
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

void FnVCCBAiClient_Reset(void)
{
    g_consecutiveFailCount.store(0);
    g_fallbackMode.store(false);

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
