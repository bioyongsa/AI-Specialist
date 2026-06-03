#include "VccbBtCommonTestJson.h"
#include "VccbBtCommonTestLegacy.h"
#include "MsgNgapStore.h"
#include "VccbAiClient.h"
#include "VccbAiEngine.h"

#ifdef VCCB_AI_DATA_COLLECT
#include <ctime>
#include <cstdint>
#endif

static JsonInjector *s = new JsonInjector(__FILE__, "VCCB");

class test_SaRrcInactive_Json : public VccbBtCommonTestJson, public ::testing::WithParamInterface<string>
{
public:
    void SetUp()
    {
        VccbBtCommonTestJson::SetUp(s);
        FnVCCBAiClient_SetEnabled(false);

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
// AI Client verification test cases
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

        // Reset Fallback state to prevent interference between test cases
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

// ── TC1: extendCount >= VCCB_AI_MAX_EXTEND_COUNT forces immediate INACTIVE ──
// The cap check runs before inference, so the result is deterministic regardless
// of the trained weights: it must return true and reset extendCount to 0.
TEST_F(test_SaRrcInactive_AI, AiClient_extendCount_force_INACTIVE)
{
    // At limit: force immediate INACTIVE transition; counter reset to 0 internally.
    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    ASSERT_EQ(ptCi->aiCtx.extendCount, 0);

    // Above limit: still forced INACTIVE, counter reset.
    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT + 1;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    ASSERT_EQ(ptCi->aiCtx.extendCount, 0);

    // When AI active decisions are disabled, the call always proceeds (legacy).
    FnVCCBAiClient_SetEnabled(false);
    ptCi->aiCtx.extendCount = 0;
    ASSERT_TRUE(FnVCCBAiClient_IsAbleToProceedInactive(ptCi));
    FnVCCBAiClient_SetEnabled(true);
}

// ── TC2: 10 consecutive inference failures -> enter Fallback -> return defaultTimerSec ──
TEST_F(test_SaRrcInactive_AI, AiClient_fallback_returns_default_timer)
{
    const int32_t defaultTimer = 30;

    // Verify initial state (not in Fallback)
    ASSERT_FALSE(FnVCCBAiClient_IsFallbackMode());

    // Inject NaN into history[0].durationSec
    // -> buildAiInput constructs seq[0][0] = NaN
    // -> isfinite check in FnVCCBAiEngine_Infer fails -> returns false
    // -> noteInferFailure() is invoked
    ptCi->aiCtx.history[0].durationSec = NAN;
    ptCi->aiCtx.count    = 1;
    ptCi->aiCtx.writeIdx = 1;

    // Trigger VCCB_AI_FALLBACK_COUNT(10) consecutive failures
    for (int i = 0; i < VCCB_AI_FALLBACK_COUNT; i++)
    {
        FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer);
    }

    // Verify Fallback mode is active
    ASSERT_TRUE(FnVCCBAiClient_IsFallbackMode());

    // Verify defaultTimerSec is returned as-is in Fallback mode
    int32_t result = FnVCCBAiClient_RecommendInactivityTimer(ptCi, defaultTimer);
    ASSERT_EQ(result, defaultTimer);
}

// ── TC3: On Resume, extendCount resets and one entry is added to history ──
TEST_F(test_SaRrcInactive_AI, AiClient_OnResumeRequest_resets_extendCount)
{
    // Force extendCount to maximum value
    ptCi->aiCtx.extendCount = VCCB_AI_MAX_EXTEND_COUNT;
    ASSERT_EQ(ptCi->aiCtx.count, 0);

    // [claude] Trigger Resume event with a non-emergency cause. Only emergency
    // [claude] (NRRRC_ResumeCause_emergency, encoded 0 in AiClient_DataCollect_Sweep's
    // [claude] 0=emergency..4=rna convention) is treated as "prediction failure" by the
    // [claude] resumeSuccessRate EMA; any other value counts as success. We pass 1 here.
    FnVCCBAiClient_OnResumeRequest(ptCi, 1 /*non-emergency*/);

    // Verify extendCount is reset to 0
    ASSERT_EQ(ptCi->aiCtx.extendCount, 0);

    // Verify one history entry was added
    ASSERT_EQ(ptCi->aiCtx.count, 1);

    // Verify resumeSuccessRate EMA update (success)
    // alpha=0.1: 0.1*1.0 + 0.9*1.0 = 1.0
    ASSERT_FLOAT_EQ(ptCi->aiCtx.resumeSuccessRate, 1.0f);
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
    const int32_t  causeSet[]  = {0, 1, 2, 3, 4};         // resumeCause (0=emergency..4=rna)

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

        // ---- DRBs + QoS flows (drive stat[0] 5QI, stat[1] drb_count, stat[3] GBR) ----
        for (int i = 0; i < drbCount && i < MAX_DRB_PER_SA_CALL && i < MAX_QOS_FLOW_PER_CALL; i++)
        {
            ptCi->drb[i].usage          = CI_use;
            ptCi->drb[i].drbQosIdxInCall = (uint16_t)i;

            ptCi->qosFlow[i].usage = CI_use;
            ptCi->qosFlow[i].qosParam.qosCharacteristics = ci_qos_char_non_dynamic_5qi;
            ptCi->qosFlow[i].qosParam.nonDyna.fiveQi     = fiveQi;
            ptCi->qosFlow[i].qosParam.gbrQosInformationPresent = (i == 0) ? (uint8_t)gbr : 0;
        }

        // ---- UE AMBR (stat[2], bps) ----
        ptCi->ueInfo.ambr.aggrMbrDl = (uint64_t)(ambr * 1000000.0);

        // ---- Serving cell RSRP (stat[6]) ----
        ptCi->servCellSsbRsrp = rsrp;

        // ---- RNA TAI count (stat[8]) ----
        ptCi->ueInfo.RrcInactiveInfo.numInactTAI = (uint8_t)rna;

        // ---- aggregate stats (stat[9..12]) ----
        int64_t durationSec = baseDurationFor(fiveQi) + (rows % 17);
        ptCi->aiCtx.avgDurationSec    = (float)baseDurationFor(fiveQi);
        ptCi->aiCtx.resumeCount24h    = (int32_t)(rows % 10);
        ptCi->aiCtx.resumeSuccessRate = 0.5f + 0.5f * (float)((rows % 5) / 4.0);

        // ---- seed sequence history (seq[t][0..1]); cold start when len==0 ----
        int32_t histLen = (int32_t)(rows % (VCCB_AI_HISTORY_LEN + 1));
        for (int32_t t = 0; t < histLen; t++)
        {
            ptCi->aiCtx.history[t].durationSec = (float)(baseDurationFor(fiveQi) + t * 3);
            ptCi->aiCtx.history[t].resumeCause = cause;
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
