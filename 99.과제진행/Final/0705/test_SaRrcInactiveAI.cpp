#include "VccbBtCommonTestJson.h"
#include "VccbBtCommonTestLegacy.h"
#include "MsgNgapStore.h"
#include "VccbAiClient.h"
#include "VccbAiEngine.h"
#include "VccbAiWeights.h"

#include <cmath>
#include <ctime>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#ifdef VCCB_AI_DATA_COLLECT
#include <cstdint>
#endif

static JsonInjector *s = new JsonInjector(__FILE__, "VCCB");

class test_SaRrcInactive_Json : public VccbBtCommonTestJson, public ::testing::WithParamInterface<string>
{
public:
    void SetUp()
    {
        VccbBtCommonTestJson::SetUp(s);
        // AI ON: exercise Point 1 so the inactivityTimer carried in E1_BearerContextSetupRequest
        // is the AI-inferred value (per stat[] combination), not the static configured value.
        FnVCCBAiClient_Reset();
        FnVCCBAiClient_SetEnabled(true);

        for(uint32_t cellNum = 0; cellNum < MAX_CELL_PER_VNF; cellNum++)
        {
            if(CELLINFO[cellNum].status != EQUIP) continue;

            for(uint32_t measIdx = 0; measIdx < MAX_NR_OF_OBJECTEUTRA_ID_DB; measIdx++)
                CELLINFO[cellNum].measurementsData.measurementObjectsEUTRADb[measIdx].base.usage = CI_no_use;
        }

        // for CHO
        m_setCellInfo(10 /*cellNum*/, GNB_1, 100 /*CellId*/, F1_ENDPOINT_IDX_1, 151 /*PCI*/, ARFCN_1250, TAC_4661);

        json dummy;
        for (uint16_t cellNum =0; cellNum < MAX_CELL_PER_VNF; cellNum++)
        {
            if (CELLINFO[cellNum].status != EQUIP) continue;

            // CHO All Active
            CELLINFO[cellNum].mobilityConfig.conditionalHandoverConfig.allowance = TRUE;
        }

        for (uint16_t gnbNum = 0; gnbNum < MAX_GNB_PER_SYS; gnbNum++)
        {
            if (GNBINFO[gnbNum].status != EQUIP) continue;

            for (uint16_t plmnIdx = 0; plmnIdx < MAX_PLMN_INDEX; plmnIdx++)
            {
                for (uint16_t svcGrpId = 0; svcGrpId < MAX_SVC_GRP_DB_NUM; svcGrpId++)
                {
                    GNBINFO[gnbNum].cfg.svcGrpBasedCfg[svcGrpId].usage = CI_use;
                    GNBINFO[gnbNum].cfg.svcGrpBasedCfg[svcGrpId].numOfChoCandidateCell = MAX_NUM_COND_CELLS;
                }
            }
        }

        t_setAutoConnection(dummy);

        EXPECT_CALL(*loamMock, vpmCollectFloat(_, _, _, _, _, _, _, _, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, mdsGetSrcIp(_, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, mdsGetDesIp(_, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, mdsMakePort(_, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, mdsMakeMdsAddr(_, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, ensAddTemporaryTimer(MDS_ATTR_IP_TYPE_IIIP, _, _, _, _, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, ensDelTimer(_, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, hasGetState(_, _)).WillRepeatedly(Return(0));
        EXPECT_CALL(*lmwMock, hasDirectSync(_, _, _)).WillRepeatedly(Return(0));

        GNBINFO[0].RrcInactiveInfo.featureEnabled = EQUIP;
        GNBINFO[0].RrcInactiveInfo.ranPagingCycle = 1;
        GNBINFO[0].RrcInactiveInfo.rrcInactiveTimer = 20;  // rrc-inactive-timer default "10"
        GNBINFO[0].RrcInactiveInfo.rrcIdleTimer = 19;      // rrc-idle-timer default "10"

        DBG->flag.fdis_stat = 1;
        DBG->log.logStoreStatistics = 1;
        DBG->value.periodicFamilyToLog[0] = 6023; // STAT_RRC_CONN_INFO_REDCAP_PLMN
    }

    void TearDown() { VccbBtCommonTestJson::TearDown(s); }
};

INSTANTIATE_TEST_SUITE_P(test_SaRrcInactive_Json, test_SaRrcInactive_Json, ::testing::ValuesIn(s->getName()), VccbBtClass::PrintToStringParamName());

TEST_P(test_SaRrcInactive_Json, All)
{
    // TC_NAME=[test_SaRrcInactive_Json:All/RRCSuspend_procedure]
    // TC_ID=[FGR-CC0107:FR660:RCC-T59511, FGR-CC0103:FR630:RCC-T81164]

    EXPECT_CALL(*loamMock, PmVNFMPeg_FLT(_, _))
        .WillRepeatedly(Return(0));

    procJson(s, data);
}

class test_SaRrcInactive : public VccbBtCommonTestLegacy
{
public:

    callInfo_type callInfo;
    callInfo_type *ptCi = &callInfo;

    void SetUp()
    {
        VccbBtCommonTestLegacy::SetUp();

        VLIB_InitCallInfo(ptCi, 0, 0, E_CALLTYPE_SA);
        ptCi->servCellNum = 0;
    }

    void TearDown()
    {
        FnVCCBAiClient_SetEnabled(true);
        VccbBtCommonTestLegacy::TearDown();
    }
};

TEST_F(test_SaRrcInactive, storeCoreNetworkAssistanceInformationInInitialUeContextSetup)
{
    NGAP_CoreNetworkAssistanceInformation_type p;
    p.m.NGAPuESpecificDRXPresent = 1;
    p.uEIdentityIndexValue.length = 10;
    p.uEIdentityIndexValue.value[0] = 20;
    p.uEIdentityIndexValue.value[1] = 30;
    p.uESpecificDRX = NGAP_v128;
    p.periodicRegistrationUpdateTimer.value = 3;

    GNBINFO[0].RrcInactiveInfo.t380 = 5;

    int16_t status = VNIL_StoreNGAP_CoreNetworkAssistanceInformation(ptCi, &p);

    ASSERT_EQ(status, GOOD);
    ASSERT_EQ(ptCi->ueInfo.RrcInactiveInfo.ueIdentityIndexValue.length, (unsigned int)10);
    ASSERT_EQ(ptCi->ueInfo.RrcInactiveInfo.ueIdentityIndexValue.value[0], (unsigned char)20);
    ASSERT_EQ(ptCi->ueInfo.RrcInactiveInfo.ueIdentityIndexValue.value[1], (unsigned char)30);
    ASSERT_EQ(ptCi->ueInfo.RrcInactiveInfo.pagingDrx, (uint16_t)NGAP_v128);
    ASSERT_EQ(ptCi->ueInfo.RrcInactiveInfo.t380, (uint16_t)3);
}


// ============================================================
// AI verification helpers
// ============================================================
namespace {

// Build a fully-finite, realistic engine input (drives the GRU directly).
VccbAiInput makeValidEngineInput()
{
    VccbAiInput in;
    memset(&in, 0, sizeof(in));
    for (int t = 0; t < VCCB_AI_HISTORY_LEN; t++)
    {
        in.seq[t][0] = 30.0f + (float)t;   // durationSec
        in.seq[t][1] = 1.0f;               // resumeCause (mo_Data)
    }
    in.stat[0]  = 9.0f;     // primary_5qi
    in.stat[1]  = 1.0f;     // drb_count
    in.stat[2]  = 100.0f;   // ue_ambr_dl_mbps
    in.stat[4]  = 1.0f;     // handover_count_24h (mobility proxy)
    in.stat[6]  = 75.0f;    // last_rsrp
    in.stat[7]  = 1.0f;     // secondary_5qi (mixed session: primary 5QI=9 + VoNR 5QI=1)
    in.stat[8]  = 1.0f;     // rna_cell_count
    in.stat[9]  = 45.0f;    // avg_inactive_duration_sec (EMA; was also stat[12])
    in.stat[11] = 1.0f;     // resume_success_rate
    in.stat[12] = 12.0f;    // hour_of_day   (was stat[13])
    in.stat[13] = 3.0f;     // day_of_week   (was stat[14])
    return in;
}

// Seed a UE context with finite, realistic history so the ACTIVE path runs end-to-end.
void seedRealisticHistory(callInfo_type *ptCi)
{
    for (int t = 0; t < VCCB_AI_HISTORY_LEN; t++)
    {
        ptCi->aiCtx.history[t].durationSec = 30.0f + (float)t;
        ptCi->aiCtx.history[t].resumeCause = VCCB_AI_CAUSE_MO_DATA;   // ML encoding (what production stores post-map)
    }
    ptCi->aiCtx.count    = VCCB_AI_HISTORY_LEN;
    ptCi->aiCtx.writeIdx = 0;
}

template <size_t N>
bool allFinite(const float (&a)[N])
{
    for (size_t i = 0; i < N; i++)
        if (!std::isfinite(a[i])) return false;
    return true;
}

} // namespace


// ============================================================
// Layer 1 — GRU inference engine (deterministic, weight-driven)
//   These run the REAL forward pass and prove the model actually
//   computes a bounded, finite, input-dependent output.
// ============================================================

// Valid input -> inference succeeds and output is bounded/finite.
TEST(VccbAiEngine, Infer_valid_input_returns_bounded_output)
{
    FnVCCBAiEngine_Init();

    VccbAiInput  in  = makeValidEngineInput();
    VccbAiOutput out = {};

    ASSERT_TRUE(FnVCCBAiEngine_Infer(in, out));

    EXPECT_TRUE(std::isfinite(out.resume_prob));
    EXPECT_TRUE(std::isfinite(out.resume_time));
    // resume_prob is a sigmoid output -> must lie in [0, 1].
    EXPECT_GE(out.resume_prob, 0.0f);
    EXPECT_LE(out.resume_prob, 1.0f);
}

// Same input must always yield the same output (no hidden state leakage).
TEST(VccbAiEngine, Infer_is_deterministic)
{
    VccbAiInput  in = makeValidEngineInput();
    VccbAiOutput a = {}, b = {};

    ASSERT_TRUE(FnVCCBAiEngine_Infer(in, a));
    ASSERT_TRUE(FnVCCBAiEngine_Infer(in, b));

    EXPECT_FLOAT_EQ(a.resume_prob, b.resume_prob);
    EXPECT_FLOAT_EQ(a.resume_time, b.resume_time);
}

// The trained model must NOT be degenerate: very different inputs must move the output.
TEST(VccbAiEngine, Infer_responds_to_input_changes)
{
    VccbAiInput lo = makeValidEngineInput();
    VccbAiInput hi = makeValidEngineInput();
    for (int t = 0; t < VCCB_AI_HISTORY_LEN; t++)
    {
        lo.seq[t][0] = 5.0f;
        hi.seq[t][0] = 300.0f;
    }

    VccbAiOutput a = {}, b = {};
    ASSERT_TRUE(FnVCCBAiEngine_Infer(lo, a));
    ASSERT_TRUE(FnVCCBAiEngine_Infer(hi, b));

    EXPECT_TRUE(a.resume_prob != b.resume_prob || a.resume_time != b.resume_time);
}

// NaN/Inf guards in the engine must reject corrupted input.
TEST(VccbAiEngine, Infer_rejects_nan_in_sequence)
{
    VccbAiInput  in  = makeValidEngineInput();
    in.seq[3][0] = NAN;
    VccbAiOutput out = {};
    EXPECT_FALSE(FnVCCBAiEngine_Infer(in, out));
}

TEST(VccbAiEngine, Infer_rejects_inf_in_static)
{
    VccbAiInput  in  = makeValidEngineInput();
    in.stat[5] = INFINITY;
    VccbAiOutput out = {};
    EXPECT_FALSE(FnVCCBAiEngine_Infer(in, out));
}

// Forward pass must complete well within the 1ms Call-Task budget.
TEST(VccbAiEngine, Infer_latency_under_timeout)
{
    VccbAiInput  in  = makeValidEngineInput();
    VccbAiOutput out = {};

    struct timespec t0 = {}, t1 = {};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ASSERT_TRUE(FnVCCBAiEngine_Infer(in, out));
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int64_t us = (t1.tv_sec - t0.tv_sec) * 1000000LL
               + (t1.tv_nsec - t0.tv_nsec) / 1000LL;

    EXPECT_LT(us, (int64_t)VCCB_AI_INFER_TIMEOUT_US);
}


// ============================================================
// Layer 2 — Weights / export pipeline contract
//   Validates that VccbAiWeights.h (generated by export_weights.py)
//   matches the model architecture in the design document.
// ============================================================

TEST(VccbAiWeights, Dimensions_match_design)
{
    EXPECT_EQ(VccbAiWeights::HIDDEN_SIZE,       16);
    EXPECT_EQ(VccbAiWeights::SEQ_INPUT_SIZE,     2);
    EXPECT_EQ(VccbAiWeights::STATIC_INPUT_SIZE, 14);
    EXPECT_EQ(VccbAiWeights::SEQ_LEN,           10);

    // Exported dimensions must agree with the engine's compile-time constants.
    EXPECT_EQ(VccbAiWeights::HIDDEN_SIZE,       VCCB_AI_HIDDEN_SIZE);
    EXPECT_EQ(VccbAiWeights::STATIC_INPUT_SIZE, VCCB_AI_STATIC_INPUT);
    EXPECT_EQ(VccbAiWeights::SEQ_INPUT_SIZE,    VCCB_AI_SEQ_INPUT);
    EXPECT_EQ(VccbAiWeights::SEQ_LEN,           VCCB_AI_HISTORY_LEN);
}

TEST(VccbAiWeights, Array_sizes_match_architecture)
{
    using namespace VccbAiWeights;
    EXPECT_EQ(sizeof(GRU_WEIGHT_IH) / sizeof(float),   (size_t)(3 * 16 * 2));   //  96
    EXPECT_EQ(sizeof(GRU_WEIGHT_HH) / sizeof(float),   (size_t)(3 * 16 * 16));  // 768
    EXPECT_EQ(sizeof(GRU_BIAS_IH)   / sizeof(float),   (size_t)(3 * 16));       //  48
    EXPECT_EQ(sizeof(GRU_BIAS_HH)   / sizeof(float),   (size_t)(3 * 16));       //  48
    EXPECT_EQ(sizeof(MLP_WEIGHT)    / sizeof(float),   (size_t)(16 * 14));      // 224
    EXPECT_EQ(sizeof(MLP_BIAS)      / sizeof(float),   (size_t)16);
    EXPECT_EQ(sizeof(HEAD_PROB_WEIGHT) / sizeof(float), (size_t)32);
    EXPECT_EQ(sizeof(HEAD_TIME_WEIGHT) / sizeof(float), (size_t)32);
    EXPECT_EQ(sizeof(HEAD_PROB_BIAS) / sizeof(float),  (size_t)1);
    EXPECT_EQ(sizeof(HEAD_TIME_BIAS) / sizeof(float),  (size_t)1);
}

// Total parameter count must equal the documented 1,266
// (STATIC_INPUT 15->14 dropped the duplicate avg_inactive_duration: MLP -16).
TEST(VccbAiWeights, Parameter_count_is_1266)
{
    using namespace VccbAiWeights;
    const size_t gru  = sizeof(GRU_WEIGHT_IH) + sizeof(GRU_WEIGHT_HH)
                      + sizeof(GRU_BIAS_IH)   + sizeof(GRU_BIAS_HH);
    const size_t mlp  = sizeof(MLP_WEIGHT) + sizeof(MLP_BIAS);
    const size_t head = sizeof(HEAD_PROB_WEIGHT) + sizeof(HEAD_PROB_BIAS)
                      + sizeof(HEAD_TIME_WEIGHT) + sizeof(HEAD_TIME_BIAS);
    const size_t total = (gru + mlp + head) / sizeof(float);
    EXPECT_EQ(total, (size_t)1266);
}

// No weight may be NaN/Inf (catches a broken export).
TEST(VccbAiWeights, All_weights_finite)
{
    using namespace VccbAiWeights;
    EXPECT_TRUE(allFinite(GRU_WEIGHT_IH));
    EXPECT_TRUE(allFinite(GRU_WEIGHT_HH));
    EXPECT_TRUE(allFinite(GRU_BIAS_IH));
    EXPECT_TRUE(allFinite(GRU_BIAS_HH));
    EXPECT_TRUE(allFinite(MLP_WEIGHT));
    EXPECT_TRUE(allFinite(MLP_BIAS));
    EXPECT_TRUE(allFinite(HEAD_PROB_WEIGHT));
    EXPECT_TRUE(allFinite(HEAD_TIME_WEIGHT));
    EXPECT_TRUE(allFinite(HEAD_PROB_BIAS));
    EXPECT_TRUE(allFinite(HEAD_TIME_BIAS));
}

// Weights must actually be populated (not an all-zero placeholder header).
TEST(VccbAiWeights, Weights_are_not_all_zero)
{
    using namespace VccbAiWeights;
    float absSum = 0.0f;
    for (size_t i = 0; i < sizeof(GRU_WEIGHT_HH) / sizeof(float); i++)
        absSum += fabsf(GRU_WEIGHT_HH[i]);
    EXPECT_GT(absSum, 0.0f);
}


// ============================================================
// Layer 3 — UE AI context initialization
//   Regression guard for §14-1 (aiCtx must be cleanly initialized,
//   never left as 0xFF garbage -> writeIdx=-1 / resumeSuccessRate=NaN).
// ============================================================

TEST(VccbAiContext, InitContext_clears_0xFF_and_seeds_success_rate)
{
    VccbAiCallContext_type ctx;
    memset(&ctx, 0xFF, sizeof(ctx));   // simulate the 0xFF memset in VLIB_InitCallInfo

    VccbAi_InitContext(&ctx);

    EXPECT_EQ(ctx.writeIdx, 0);
    EXPECT_EQ(ctx.count, 0);
    EXPECT_EQ(ctx.resumeTsWriteIdx, 0);
    EXPECT_EQ(ctx.extendCount, 0);
    EXPECT_FLOAT_EQ(ctx.resumeSuccessRate, 1.0f);   // cold-start seed
    EXPECT_FLOAT_EQ(ctx.avgDurationSec, 0.0f);
    for (int i = 0; i < VCCB_AI_HIDDEN_SIZE; i++)
        EXPECT_FLOAT_EQ(ctx.h[i], 0.0f);
}


// ============================================================
// Layer 4 — AI Client policy (Point 1 / 2 / 3)
//   Covers the ACTIVE success path (the previously-missing case),
//   the safety cap, the fallback path, and history bookkeeping.
// ============================================================
class test_SaRrcInactive_AI : public VccbBtCommonTestLegacy
{
public:
    callInfo_type  callInfo;
    callInfo_type *ptCi = &callInfo;

    void SetUp()
    {
        VccbBtCommonTestLegacy::SetUp();
        VLIB_InitCallInfo(ptCi, 0, 0, E_CALLTYPE_SA);
        ptCi->servCellNum = 0;

        // Clean global AI state so each case is isolated.
        FnVCCBAiClient_Reset();
        FnVCCBAiClient_SetEnabled(true);   // ensure inference path active for this suite
    }

    void TearDown()
    {
        FnVCCBAiClient_Reset();
        FnVCCBAiClient_SetEnabled(true);   // restore production default (ON)
        VccbBtCommonTestLegacy::TearDown();
    }
};

// Regression for §14-1: after VLIB_InitCallInfo the aiCtx is clean (no 0xFF garbage).
TEST_F(test_SaRrcInactive_AI, VLIB_InitCallInfo_initializes_aiCtx)
{
    VLIB_InitCallInfo(ptCi, 0, 0, E_CALLTYPE_SA);

    EXPECT_EQ(ptCi->aiCtx.writeIdx, 0);
    EXPECT_EQ(ptCi->aiCtx.count, 0);
    EXPECT_EQ(ptCi->aiCtx.extendCount, 0);
    EXPECT_EQ(ptCi->aiCtx.resumeTsWriteIdx, 0);
    EXPECT_TRUE(std::isfinite(ptCi->aiCtx.resumeSuccessRate));
    EXPECT_FLOAT_EQ(ptCi->aiCtx.resumeSuccessRate, 1.0f);
    for (int i = 0; i < VCCB_AI_HIDDEN_SIZE; i++)
        EXPECT_TRUE(std::isfinite(ptCi->aiCtx.h[i]));
}

// ── Point 1 (RecommendInactivityTimer): ACTIVE success path ──
// THE key case that was missing: valid input -> inference succeeds (NOT fallback)
// -> a timer within the 3GPP TS 38.413 range (1..3600s) is returned.
TEST_F(test_SaRrcInactive_AI, Point1_active_inference_returns_valid_timer)
{
    seedRealisticHistory(ptCi);

    const int32_t defaultTimer = 30;
    int32_t result = FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer);

    // Must have actually inferred, not silently fallen back.
    ASSERT_FALSE(FnVCCBAiClient_IsFallbackMode());

    // Output clamped to the valid range.
    EXPECT_GE(result, VCCB_AI_TIMER_MIN_SEC);
    EXPECT_LE(result, VCCB_AI_TIMER_MAX_SEC);

    // Cached value for reuse must also be valid.
    EXPECT_GE(ptCi->aiCtx.lastInferTimerSec, (float)VCCB_AI_TIMER_MIN_SEC);
    EXPECT_LE(ptCi->aiCtx.lastInferTimerSec, (float)VCCB_AI_TIMER_MAX_SEC);

    // Statistics prove a real inference happened.
    VccbAiGlobalStats_type s;
    FnVCCBAiClient_GetStats(&s);
    EXPECT_GT(s.successCount, 0);
    EXPECT_EQ(s.failureCount, 0);
    EXPECT_GT(s.triggerPt1Count, 0);
}

// ── Point 1: master switch OFF -> legacy behavior (default returned unchanged) ──
TEST_F(test_SaRrcInactive_AI, Point1_disabled_returns_default_unchanged)
{
    FnVCCBAiClient_SetEnabled(false);
    seedRealisticHistory(ptCi);

    const int32_t defaultTimer = 42;
    EXPECT_EQ(FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer), defaultTimer);

    FnVCCBAiClient_SetEnabled(true);
}

// ── Point 1: 10 consecutive inference failures -> Fallback -> default returned ──
TEST_F(test_SaRrcInactive_AI, Point1_fallback_after_failures_returns_default)
{
    const int32_t defaultTimer = 30;

    ASSERT_FALSE(FnVCCBAiClient_IsFallbackMode());

    // NaN in history -> buildAiInput yields seq[0][0]=NaN -> engine isfinite guard fails.
    ptCi->aiCtx.history[0].durationSec = NAN;
    ptCi->aiCtx.count    = 1;
    ptCi->aiCtx.writeIdx = 1;

    for (int i = 0; i < VCCB_AI_FALLBACK_COUNT; i++)
    {
        FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer);
    }

    ASSERT_TRUE(FnVCCBAiClient_IsFallbackMode());
    EXPECT_EQ(FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer), defaultTimer);
}

// ── Point 2 (IsAbleToProceedInactive): ACTIVE decision is consistent & bounded ──
TEST_F(test_SaRrcInactive_AI, Point2_active_decision_is_consistent_and_bounded)
{
    seedRealisticHistory(ptCi);
    ptCi->aiCtx.extendCount = 0;

    bool proceed = FnVCCBAiClient_IsAbleToProceedInactive(ptCi);

    ASSERT_FALSE(FnVCCBAiClient_IsFallbackMode());

    VccbAiGlobalStats_type s;
    FnVCCBAiClient_GetStats(&s);
    EXPECT_GT(s.successCount, 0);
    EXPECT_EQ(s.failureCount, 0);
    EXPECT_GT(s.triggerPt2Count, 0);

    if (!proceed)
    {
        // Deferred: caller will send BearerContextModification with the new timer.
        EXPECT_EQ(ptCi->aiCtx.extendCount, 1);
        EXPECT_GE(ptCi->aiCtx.lastInferTimerSec, (float)VCCB_AI_TIMER_MIN_SEC);
        EXPECT_LE(ptCi->aiCtx.lastInferTimerSec, (float)VCCB_AI_TIMER_MAX_SEC);
    }
    else
    {
        // Immediate INACTIVE transition: no extension recorded.
        EXPECT_EQ(ptCi->aiCtx.extendCount, 0);
    }
}

// ── Point 2: extendCount cap forces immediate INACTIVE (deterministic, pre-inference) ──
TEST_F(test_SaRrcInactive_AI, Point2_extendCount_cap_forces_inactive)
{
    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    ASSERT_EQ(ptCi->aiCtx.extendCount, 0);

    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT + 1;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    ASSERT_EQ(ptCi->aiCtx.extendCount, 0);

    // Master switch OFF -> always proceed (legacy).
    FnVCCBAiClient_SetEnabled(false);
    ptCi->aiCtx.extendCount = 0;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    FnVCCBAiClient_SetEnabled(true);
}

// ── Point 3 (OnResumeRequest): non-emergency resume updates history, keeps success rate ──
TEST_F(test_SaRrcInactive_AI, Point3_resume_updates_history_and_resets_extend)
{
    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT;
    ASSERT_EQ(ptCi->aiCtx.count, 0);

    // A resume always follows a real INACTIVE entry in production; without the
    // stamp OnResumeRequest now (correctly) skips history accumulation.
    FnVCCBAiClient_OnInactiveEntry(ptCi);
    FnVCCBAiClient_OnResumeRequest(ptCi, (int32_t)NRRRC_ResumeCause_mo_Data);

    EXPECT_EQ(ptCi->aiCtx.extendCount, 0);   // reset for next INACTIVE cycle
    EXPECT_EQ(ptCi->aiCtx.count, 1);         // one history entry pushed
    EXPECT_EQ(ptCi->aiCtx.writeIdx, 1);

    // Raw NRRRC mo_Data must be stored as the ML encoding (train/serve contract).
    EXPECT_EQ(ptCi->aiCtx.history[0].resumeCause, VCCB_AI_CAUSE_MO_DATA);

    // EMA(alpha=0.1) on success: 0.1*1.0 + 0.9*1.0 = 1.0
    EXPECT_FLOAT_EQ(ptCi->aiCtx.resumeSuccessRate, 1.0f);
}

// ── Point 3: resume WITHOUT an INACTIVE entry stamp must not pollute history ──
// (e.g. inter-gNB resume target side: a 0s duration label would teach the model
// "this UE resumes instantly" and drag avgDurationSec toward 0.)
TEST_F(test_SaRrcInactive_AI, Point3_resume_without_entry_stamp_skips_history)
{
    ASSERT_EQ(ptCi->aiCtx.inactiveEntryTs, 0);
    ptCi->aiCtx.extendCount = 2;

    FnVCCBAiClient_OnResumeRequest(ptCi, (int32_t)NRRRC_ResumeCause_mo_Data);

    EXPECT_EQ(ptCi->aiCtx.count, 0);         // no bogus 0s history entry
    EXPECT_EQ(ptCi->aiCtx.writeIdx, 0);
    EXPECT_FLOAT_EQ(ptCi->aiCtx.avgDurationSec, 0.0f);
    EXPECT_EQ(ptCi->aiCtx.extendCount, 0);   // runtime counters still recycled
}

// ── Point 3: emergency resume is treated as a prediction failure (success rate drops) ──
TEST_F(test_SaRrcInactive_AI, Point3_emergency_resume_lowers_success_rate)
{
    ASSERT_FLOAT_EQ(ptCi->aiCtx.resumeSuccessRate, 1.0f);

    FnVCCBAiClient_OnInactiveEntry(ptCi);
    FnVCCBAiClient_OnResumeRequest(ptCi, (int32_t)NRRRC_ResumeCause_emergency);

    EXPECT_EQ(ptCi->aiCtx.count, 1);
    EXPECT_EQ(ptCi->aiCtx.history[0].resumeCause, VCCB_AI_CAUSE_EMERGENCY);
    // EMA(alpha=0.1) on failure: 0.1*0.0 + 0.9*1.0 = 0.9
    EXPECT_FLOAT_EQ(ptCi->aiCtx.resumeSuccessRate, 0.9f);
}

// ============================================================
// Point 2 (IsAbleToProceedInactive): full AI EXTENSION LOOP
// ------------------------------------------------------------
// Verifies the end-to-end RRC-INACTIVE decision loop that the AI drives once
// a call is in RRC_CONNECTED and E1 bearers are established. The probabilistic
// model output is pinned with the test-only hook
// FnVCCBAiClient_TestForceInactiveDecision(0 = force EXTEND) so the loop is
// deterministic; the extendCount cap (VCCB_AI_MAX_EXTEND_COUNT) still provides
// the natural termination, exactly as in production.
//
// Each "EXTEND" round maps to the operator-requested sequence:
//   1. E1_BearerContextModificationRequest is sent carrying the AI inactivity
//      timer (here = FnVCCBAiClient_RecommendInactivityTimer, Point 1).
//   2. E1_BearerContextModificationResponse is received.
//   3. Inactivity timer (run by CU-UP) expires -> E1_BearerContextInactivity
//      Notification is received -> re-enters the decision.
//   4. AI re-evaluates: EXTEND -> back to (1); PROCEED -> (5).
//   5. extendCount cap reached -> PROCEED -> RRC Suspend procedure starts.
// ============================================================
TEST_F(test_SaRrcInactive_AI, Point2_extension_loop_then_suspend)
{
    seedRealisticHistory(ptCi);
    ptCi->aiCtx.extendCount = 0;

    const int32_t defaultTimer = 30;   // statically configured value (pre-AI)

    // Pin Point 2 to EXTEND so the loop is deterministic regardless of model output.
    FnVCCBAiClient_TestForceInactiveDecision(0 /* force EXTEND */);

    // ---- EXTEND rounds: defer INACTIVE up to the cap ----
    for (int round = 1; round <= VCCB_AI_MAX_EXTEND_COUNT; ++round)
    {
        // (3) InactivityNotification arrived -> caller asks AI whether to proceed.
        bool proceed = FnVCCBAiClient_IsAbleToProceedInactive(ptCi);

        // (4) AI deferred -> caller must send a BearerContextModification.
        ASSERT_FALSE(proceed) << "round " << round << " expected EXTEND (defer)";
        EXPECT_EQ(ptCi->aiCtx.extendCount, round)
            << "each EXTEND must increment extendCount";

        // (1) The inactivity timer that the BearerContextModificationRequest carries
        //     is the AI-recommended (Point 1) value, not the static default.
        int32_t modTimer = FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer);
        EXPECT_GE(modTimer, VCCB_AI_TIMER_MIN_SEC);
        EXPECT_LE(modTimer, VCCB_AI_TIMER_MAX_SEC);
        // (2) BearerContextModificationResponse reception is implied by looping again.
    }

    // ---- Cap reached: AI must now PROCEED even though EXTEND is still forced ----
    // (5) extendCount(==MAX) >= MAX -> forced INACTIVE transition (RRC Suspend starts).
    bool proceedFinal = FnVCCBAiClient_IsAbleToProceedInactive(ptCi);
    ASSERT_TRUE(proceedFinal) << "cap must force PROCEED to bound the loop";
    EXPECT_EQ(ptCi->aiCtx.extendCount, 0) << "extendCount resets when proceeding";

    // ---- Close the cycle: a later Resume (Point 3) records history & resets state ----
    // seedRealisticHistory() filled the circular buffer (count == VCCB_AI_HISTORY_LEN),
    // so a push advances writeIdx but count is clamped at the buffer length.
    int32_t countBefore   = ptCi->aiCtx.count;
    int32_t expectedCount  = (countBefore < VCCB_AI_HISTORY_LEN) ? (countBefore + 1)
                                                                 : VCCB_AI_HISTORY_LEN;
    auto     widxBefore    = ptCi->aiCtx.writeIdx;
    FnVCCBAiClient_OnInactiveEntry(ptCi);   // real transition precedes the resume
    FnVCCBAiClient_OnResumeRequest(ptCi, (int32_t)NRRRC_ResumeCause_mo_Data);
    EXPECT_EQ(ptCi->aiCtx.count, expectedCount);                  // bounded by buffer length
    EXPECT_NE(ptCi->aiCtx.writeIdx, widxBefore);                  // resume actually recorded
    EXPECT_EQ(ptCi->aiCtx.extendCount, 0);                        // ready for next cycle

    // Inference actually ran during the loop (not a silent fallback).
    ASSERT_FALSE(FnVCCBAiClient_IsFallbackMode());

    FnVCCBAiClient_TestForceInactiveDecision(-1 /* restore: model decides */);
}

// ── Point 2: forced PROCEED -> immediate INACTIVE, no extension recorded ──
TEST_F(test_SaRrcInactive_AI, Point2_forced_proceed_transitions_immediately)
{
    seedRealisticHistory(ptCi);
    ptCi->aiCtx.extendCount = 0;

    FnVCCBAiClient_TestForceInactiveDecision(1 /* force PROCEED */);

    EXPECT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    EXPECT_EQ(ptCi->aiCtx.extendCount, 0) << "PROCEED must not extend";

    FnVCCBAiClient_TestForceInactiveDecision(-1);
}

// ============================================================
// Layer 5 — Phase-1 QUALITY REGRESSION GUARD (always compiled)
// ------------------------------------------------------------
// Runs the REAL deployed C++ engine (VccbAiEngine.cpp + folded
// VccbAiWeights.h) over an in-distribution eval set and asserts the
// Phase-1 metrics still clear conservative floors. This is the gtest
// mirror of ML/metrics.py + ML/evaluate.py: it catches weight/export/
// engine regressions that the unit-level Layer-1 tests cannot
// (e.g. a bad export_weights.py run silently degrading accuracy).
//
// The eval set is generated here with the SAME domain rules as
// ML/preprocess.py (5QI log-normal profiles + RSRP/history label
// modulation) so the trained model is scored in-distribution. We assert
// FLOORS (not exact values) so sampling noise never flakes the test;
// measured headroom on the current model is large (AUROC ~0.95, model
// MAE ~8.7s vs naive ~12s / persistence ~22s).
//
// LOCKSTEP: if ML/preprocess.py label rules change, mirror them below.
// ============================================================
namespace {

// resumeCause encoding — must match ML/preprocess.py.
enum { GEN_CAUSE_EMERGENCY = 0, GEN_CAUSE_MO_DATA = 1, GEN_CAUSE_MO_SIGNAL = 2,
       GEN_CAUSE_MT_ACCESS = 3, GEN_CAUSE_RNA_UPDATE = 4 };
const double GEN_PROB_HORIZON_SEC = 60.0;
const double GEN_MAX_DUR_SEC      = 86400.0;

struct GenProfile {
    int    fiveQi;
    double durMean, durStd;
    int    gbr, cause, drbLo, drbHi;
    double ambrLo, ambrHi;
};

// 5QI traffic-class profiles — mirror FIVEQI_PROFILES in ML/preprocess.py.
const GenProfile GEN_PROFILES[] = {
    {1,  20.0,  10.0,  1, GEN_CAUSE_MO_SIGNAL, 1, 2,  50.0,  300.0},
    {2,  30.0,  15.0,  1, GEN_CAUSE_MO_DATA,   1, 3, 100.0,  600.0},
    {5,  45.0,  25.0,  0, GEN_CAUSE_MO_SIGNAL, 1, 2,  10.0,  100.0},
    {7,  90.0,  45.0,  0, GEN_CAUSE_MT_ACCESS, 1, 3,  50.0,  500.0},
    {8, 150.0,  80.0,  0, GEN_CAUSE_MO_DATA,   1, 4, 100.0, 1000.0},
    {9, 200.0, 110.0,  0, GEN_CAUSE_MO_DATA,   1, 6, 100.0, 2000.0},
};
const int GEN_NUM_PROFILES = (int)(sizeof(GEN_PROFILES) / sizeof(GEN_PROFILES[0]));

// Log-normal inactive duration from a 5QI profile (seconds) — preprocess.py.
double genSampleDuration(const GenProfile &p, std::mt19937 &rng)
{
    double sigma2 = std::log(1.0 + (p.durStd / p.durMean) * (p.durStd / p.durMean));
    double mu     = std::log(p.durMean) - sigma2 / 2.0;
    std::lognormal_distribution<double> d(mu, std::sqrt(sigma2));
    double v = d(rng);
    if (v < 1.0) v = 1.0;
    if (v > GEN_MAX_DUR_SEC) v = GEN_MAX_DUR_SEC;
    return v;
}

struct EvalSample { VccbAiInput in; double yTime; int yProb; };

// Build an in-distribution eval set identical in spirit to preprocess.py.
std::vector<EvalSample> genEvalSet(int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pickProfile(0, GEN_NUM_PROFILES - 1);
    std::normal_distribution<double>   rsrpN(75.0, 25.0);
    std::normal_distribution<double>   succN(0.85, 0.15);
    std::uniform_int_distribution<int> rnaU(1, 16), cntU(0, 10), hourU(0, 23),
                                       dowU(0, 6), histU(0, VCCB_AI_HISTORY_LEN);

    std::vector<EvalSample> out;
    out.reserve(n);
    for (int k = 0; k < n; k++)
    {
        const GenProfile &p = GEN_PROFILES[pickProfile(rng)];
        std::uniform_int_distribution<int>  drbU(p.drbLo, p.drbHi);
        std::uniform_real_distribution<double> ambrU(p.ambrLo, p.ambrHi);

        double lastRsrp = std::min(127.0, std::max(0.0, rsrpN(rng)));
        double succRate = std::min(1.0,  std::max(0.0, succN(rng)));

        EvalSample s;
        memset(&s.in, 0, sizeof(s.in));

        int histLen = histU(rng);
        double sumHist = 0.0;
        for (int t = 0; t < histLen; t++)
        {
            double d = genSampleDuration(p, rng);
            s.in.seq[t][0] = (float)d;
            s.in.seq[t][1] = (float)p.cause;
            sumHist += d;
        }
        double avgDur = (histLen > 0) ? (sumHist / histLen) : p.durMean;

        // stat[7] secondary_5qi: ~40% mixed service-class sessions (mirror
        // preprocess.py). Derived from k (NOT the rng) so the eval RNG stream and
        // label distribution / Phase-1 floors stay unchanged -- only stat[7] and
        // drb_count consistency differ.
        int secondary5qi = 0;
        if ((k % 5) < 2)   // ~40% mixed
            secondary5qi = (p.fiveQi == 8 || p.fiveQi == 9) ? ((k % 2) ? 1 : 2) : 9;
        int drbCount = drbU(rng);
        if (secondary5qi != 0 && drbCount < 2) drbCount = 2;  // 2nd DRB carries the 2nd class

        s.in.stat[0]  = (float)p.fiveQi;
        s.in.stat[1]  = (float)drbCount;
        s.in.stat[2]  = (float)ambrU(rng);
        s.in.stat[3]  = (float)p.gbr;
        // stat[4] handover_count_24h: mobility proxy, derived from k (NOT the rng)
        // so the eval RNG stream / label distribution / Phase-1 floors are unchanged.
        s.in.stat[4]  = (float)(k % 4);   // 0..3 handovers/24h
        s.in.stat[5]  = 0.0f;
        s.in.stat[6]  = (float)lastRsrp;
        s.in.stat[7]  = (float)secondary5qi;
        s.in.stat[8]  = (float)rnaU(rng);
        s.in.stat[9]  = (float)avgDur;       // avg_inactive_duration_sec (was also stat[12])
        s.in.stat[10] = (float)cntU(rng);
        s.in.stat[11] = (float)succRate;
        s.in.stat[12] = (float)hourU(rng);   // hour_of_day (was stat[13])
        s.in.stat[13] = (float)dowU(rng);    // day_of_week (was stat[14])

        // Label: next duration modulated by RSRP + history success + service-group
        // + mobility (LOCKSTEP with preprocess.py gen_synthetic label rules).
        double base       = genSampleDuration(p, rng);
        double rsrpFactor = 1.0 + (75.0 - lastRsrp) / 150.0;
        double histFactor = 1.0 - 0.3 * (succRate - 0.5);
        double svcFactor  = (secondary5qi == 1 || secondary5qi == 2) ? 0.7 : 1.0;
        double mobFactor  = 1.0 - 0.05 * std::min(k % 4, 6);   // stat[4] handover_count_24h
        double nextDur    = base * rsrpFactor * histFactor * svcFactor * mobFactor;
        if (nextDur < 1.0) nextDur = 1.0;
        if (nextDur > GEN_MAX_DUR_SEC) nextDur = GEN_MAX_DUR_SEC;

        s.yTime = nextDur;
        s.yProb = (nextDur < GEN_PROB_HORIZON_SEC) ? 1 : 0;
        out.push_back(s);
    }
    return out;
}

// Rank-based AUROC (Mann-Whitney U) with tie-averaged ranks.
double computeAuroc(const std::vector<int> &y, const std::vector<double> &score)
{
    size_t n = y.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return score[a] < score[b]; });

    std::vector<double> rank(n, 0.0);
    size_t i = 0;
    while (i < n)
    {
        size_t j = i;
        while (j + 1 < n && score[idx[j + 1]] == score[idx[i]]) j++;
        double r = (double)(i + j) / 2.0 + 1.0;
        for (size_t k = i; k <= j; k++) rank[idx[k]] = r;
        i = j + 1;
    }
    double sumPos = 0.0; long nPos = 0;
    for (size_t k = 0; k < n; k++)
        if (y[k] == 1) { sumPos += rank[k]; nPos++; }
    long nNeg = (long)n - nPos;
    if (nPos == 0 || nNeg == 0) return std::nan("");
    return (sumPos - (double)nPos * (nPos + 1) / 2.0) / ((double)nPos * nNeg);
}

} // namespace

// The deployed model must clear Phase-1 quality floors on an in-distribution
// eval set: separate resume vs no-resume (AUROC) and beat the trivial
// regression baselines (global mean, persistence) on positive-subset MAE.
TEST(VccbAiEngine, Phase1_quality_floors_on_eval_set)
{
    FnVCCBAiEngine_Init();

    const int N = 4000;
    std::vector<EvalSample> ds = genEvalSet(N, /*seed=*/12345u);

    std::vector<int>    yProb;     yProb.reserve(N);
    std::vector<double> probHat;   probHat.reserve(N);
    std::vector<double> posErrModel, posTrue;   // positive-subset regression
    std::vector<double> posPersist;             // persistence baseline (stat[9] avg_inactive_duration_sec)

    const double THRESHOLD = 0.6;   // VCCB_AI_KEEP_THRESHOLD
    long tp = 0, fp = 0, fn = 0;
    double sumPosTrue = 0.0;

    for (const EvalSample &s : ds)
    {
        VccbAiOutput out = {};
        ASSERT_TRUE(FnVCCBAiEngine_Infer(s.in, out));
        ASSERT_TRUE(std::isfinite(out.resume_prob));
        ASSERT_TRUE(std::isfinite(out.resume_time));

        yProb.push_back(s.yProb);
        probHat.push_back((double)out.resume_prob);

        int decide = (out.resume_prob >= THRESHOLD) ? 1 : 0;
        if (decide == 1 && s.yProb == 1) tp++;
        else if (decide == 1 && s.yProb == 0) fp++;
        else if (decide == 0 && s.yProb == 1) fn++;

        if (s.yProb == 1)   // regression head is trained on the positive regime
        {
            posTrue.push_back(s.yTime);
            posErrModel.push_back(std::fabs((double)out.resume_time - s.yTime));
            posPersist.push_back((double)s.in.stat[9]);
            sumPosTrue += s.yTime;
        }
    }

    ASSERT_GT(posTrue.size(), (size_t)200) << "need enough positives to be stable";

    double auroc = computeAuroc(yProb, probHat);
    double precision = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    double recall    = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

    double naiveMean = sumPosTrue / (double)posTrue.size();
    double maeModel = 0.0, maeNaive = 0.0, maePersist = 0.0;
    for (size_t i = 0; i < posTrue.size(); i++)
    {
        maeModel   += posErrModel[i];
        maeNaive   += std::fabs(naiveMean      - posTrue[i]);
        maePersist += std::fabs(posPersist[i]  - posTrue[i]);
    }
    maeModel   /= posTrue.size();
    maeNaive   /= posTrue.size();
    maePersist /= posTrue.size();

    // Surface the numbers for CI trend tracking (gtest XML properties).
    RecordProperty("auroc",       (int)std::lround(auroc * 1000));      // x1000
    RecordProperty("f1_at_0_6",   (int)std::lround(f1 * 1000));         // x1000
    RecordProperty("mae_model_ms",   (int)std::lround(maeModel   * 1000));
    RecordProperty("mae_naive_ms",   (int)std::lround(maeNaive   * 1000));
    RecordProperty("mae_persist_ms", (int)std::lround(maePersist * 1000));
    RecordProperty("eval_positives", (int)posTrue.size());

    // ---- Phase-1 floors (conservative; current model has large headroom) ----
    EXPECT_TRUE(std::isfinite(auroc));
    EXPECT_GE(auroc, 0.85) << "classifier must separate resume vs no-resume";
    EXPECT_GE(f1,    0.78) << "operating-point F1 regressed";
    EXPECT_LT(maeModel, maeNaive)
        << "model MAE must beat the global-mean baseline";
    EXPECT_LT(maeModel, maePersist)
        << "model MAE must beat the persistence (avg_inactive_duration) baseline";
    EXPECT_LE(maeModel, 15.0)
        << "absolute regression MAE floor (measured ~8.7s)";
}


#ifdef VCCB_AI_DATA_COLLECT
// ============================================================
// Training-data sweep (compiled ONLY with -DVCCB_AI_DATA_COLLECT)
// ------------------------------------------------------------
// Drives the REAL buildAiInput() extraction path across a controlled
// cross-product of AI-relevant call-processing parameters, then calls
// FnVCCBAiClient_OnResumeRequest() so the logger appends one CSV row per
// combination to VCCB_AI_DATA_FILE (default /tmp/vccb_ai_dataset.csv).
//
// This is what a pure-JSON sweep cannot do: 5QI / DRB / AMBR / GBR / RSRP /
// RNA are baked into recorded Attach scenarios, but here we set them directly
// on callInfo_type so every feature is swept while still exercising the exact
// C++ field paths used in production (contract validation + realistic ranges).
// ============================================================
TEST_F(test_SaRrcInactive_AI, AiClient_DataCollect_Sweep)
{
    const uint8_t  fiveQiSet[] = {1, 2, 5, 7, 8, 9};      // stat[0] primary_5qi
    const int32_t  drbSet[]    = {1, 2, 4};               // stat[1] drb_count
    const double   ambrMbps[]  = {100.0, 600.0, 2000.0};  // stat[2] ue_ambr_dl_mbps
    const int      gbrSet[]    = {0, 1};                  // stat[3] has_gbr_flow
    const uint8_t  rsrpSet[]   = {40, 75, 110};           // stat[6] last_rsrp
    const int32_t  rnaSet[]    = {1, 8};                  // stat[8] rna_cell_count
    // resumeCause: OnResumeRequest takes the raw over-the-air NRRRC enum and maps
    // it to the ML 0..5 encoding internally, so the sweep feeds NRRRC values —
    // exactly like production. rna_Update is intentionally absent: the Point-3
    // call site skips RNA-update resumes, so production never emits such rows.
    const int32_t  causeSet[]  = {
        (int32_t)NRRRC_ResumeCause_emergency,
        (int32_t)NRRRC_ResumeCause_mo_Data,
        (int32_t)NRRRC_ResumeCause_mo_Signalling,
        (int32_t)NRRRC_ResumeCause_mt_Access,
        (int32_t)NRRRC_ResumeCause_mo_VoiceCall,   // maps to OTHERS(5)
    };

    // Per-5QI representative inactive duration (seconds) for the label.
    auto baseDurationFor = [](uint8_t q) -> int64_t {
        switch (q) {
            case 1:  return 20;    // VoNR
            case 2:  return 35;    // video
            case 5:  return 45;    // IMS signalling
            case 7:  return 90;    // interactive
            case 8:  return 150;   // eMBB
            default: return 220;   // 5QI 9 default bearer
        }
    };

    int64_t rows = 0;

    for (uint8_t fiveQi : fiveQiSet)
    for (int32_t drbCount : drbSet)
    for (double ambr : ambrMbps)
    for (int gbr : gbrSet)
    for (uint8_t rsrp : rsrpSet)
    for (int32_t rna : rnaSet)
    for (int32_t cause : causeSet)
    {
        // Fresh UE context each iteration (zeros aiCtx, sets resumeSuccessRate=1)
        VLIB_InitCallInfo(ptCi, 0, 0, E_CALLTYPE_SA);
        ptCi->servCellNum = 0;

        // ---- DRBs + QoS flows (drive stat[0] 5QI, stat[1] drb_count, stat[3] GBR,
        //      stat[7] secondary_5qi) ----
        // item 3a: DRB0 carries the swept primary 5QI; any extra DRB carries a
        // *different* 5QI so buildAiInput's secondary_5qi (stat[7]) is exercised
        // for mixed service-class sessions. Pairing mirrors preprocess.py:
        // data primary (8/9) + voice companion (1); otherwise + default data (9).
        uint8_t secondaryQi = (fiveQi == 8 || fiveQi == 9) ? 1 : 9;
        for (int i = 0; i < drbCount && i < MAX_DRB_PER_SA_CALL && i < MAX_QOS_FLOW_PER_CALL; i++)
        {
            ptCi->drb[i].usage          = CI_use;
            ptCi->drb[i].drbQosIdxInCall = (uint16_t)i;

            ptCi->qosFlow[i].usage = CI_use;
            ptCi->qosFlow[i].qosParam.qosCharacteristics = ci_qos_char_non_dynamic_5qi;
            ptCi->qosFlow[i].qosParam.nonDyna.fiveQi     = (i == 0) ? fiveQi : secondaryQi;
            ptCi->qosFlow[i].qosParam.gbrQosInformationPresent = (i == 0) ? (uint8_t)gbr : 0;
        }

        // ---- UE AMBR (stat[2], bps) ----
        ptCi->ueInfo.ambr.aggrMbrDl = (uint64_t)(ambr * 1000000.0);

        // ---- Serving cell RSRP (stat[6]) ----
        ptCi->servCellSsbRsrp = rsrp;

        // ---- RNA TAI count (stat[8]) ----
        ptCi->ueInfo.RrcInactiveInfo.numInactTAI = (uint8_t)rna;

        // ---- aggregate stats (stat[9..11]; hour/dow are stat[12]/[13]) ----
        int64_t durationSec = baseDurationFor(fiveQi) + (rows % 17);
        ptCi->aiCtx.avgDurationSec    = (float)baseDurationFor(fiveQi);
        ptCi->aiCtx.resumeCount24h    = (int32_t)(rows % 10);
        ptCi->aiCtx.handoverCount24h  = (int32_t)(rows % 4);   // stat[4] mobility proxy
        ptCi->aiCtx.resumeSuccessRate = 0.5f + 0.5f * (float)((rows % 5) / 4.0);

        // ---- seed sequence history (seq[t][0..1]); cold start when len==0 ----
        // History stores the ML 0..5 encoding (what production stores after the
        // internal map), NOT the raw NRRRC enum value we pass to OnResumeRequest.
        int32_t histLen = (int32_t)(rows % (VCCB_AI_HISTORY_LEN + 1));
        for (int32_t t = 0; t < histLen; t++)
        {
            ptCi->aiCtx.history[t].durationSec = (float)(baseDurationFor(fiveQi) + t * 3);
            ptCi->aiCtx.history[t].resumeCause = FnVCCBAiClient_MapResumeCause(cause);
        }
        ptCi->aiCtx.count    = histLen;
        ptCi->aiCtx.writeIdx = histLen % VCCB_AI_HISTORY_LEN;

        // ---- encode the realized INACTIVE duration as the label ----
        ptCi->aiCtx.inactiveEntryTs = (int64_t)time(NULL) - durationSec;

        // Fires the logger (under VCCB_AI_DATA_COLLECT): emits one CSV row.
        FnVCCBAiClient_OnResumeRequest(ptCi, cause);
        rows++;
    }

    RecordProperty("ai_sweep_rows", (int)rows);
    SUCCEED();
}
#endif // VCCB_AI_DATA_COLLECT
