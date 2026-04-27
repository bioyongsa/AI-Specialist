#include "VccbBtCommonTestJson.h"
#include "VccbBtCommonTestLegacy.h"

#include "MsgNgapStore.h"

static JsonInjector *s = new JsonInjector(__FILE__, "VCCB");

class test_SaRrcInactive_Json : public VccbBtCommonTestJson, public ::testing::WithParamInterface<string>
{
public:
    void SetUp()
    {
        VccbBtCommonTestJson::SetUp(s);

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

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCSuspend_procedure_RedCapFooON]
    // TC_ID=[FGR-CC0103:FR630:RCC-T81163]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCSuspend_without_MRL]
    // TC_ID=[FGR-CC0107:FR660:RCC-T62994]
    
    // TC_NAME=[test_SaRrcInactive_Json:All/RRCRelease_instead_of_suspend_by_rATRestrictions_in_MRL]
    // TC_ID=[FGR-CC0107:FR660:RCC-T62993]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCRelease_instead_of_suspend_by_forbiddenAreaInformation_in_MRL]
    // TC_ID=[FGR-CC0107:FR660:RCC-T59508]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCRelease_instead_of_suspend_by_serviceAreaInformation_in_MRL]
    // TC_ID=[FGR-CC0107:FR660:RCC-T59509]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCResume_fallback_to_RRCSetup_procedure_due_to_NoUeContext]
    // TC_ID=[FGR-CC0103:FR490:RCC-T58711, FGR-CC0103:FR490:RCC-T59423]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCResume_fallback_to_RRCSetup_procedure_due_to_NoUeContext_when_resume_cause_is_RNAUpdate]
    // TC_ID=[FGR-CC0103:FR490:RCC-T58714]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCInactiveToIdle_procedure]
    // TC_ID=[FGR-CC0103:FR630:RCC-T81167]

    // TC_NAME=[test_SaRrcInactive_Json:All/RRCInactiveToIdle_procedure_RedCapFooOn]
    // TC_ID=[FGR-CC0103:FR630:RCC-T81166]

    // TC_NAME=[test_SaRrcInactive_Json:All/Explicit_release_triggered_by_E1BearerContextReleaseRequest_in_suspend_state]
    // TC_ID=[FGR-MC1016:FR330:RCC-T76351]

    // TC_NAME=[test_SaRrcInactive_Json:All/T76352_Explicit_release_triggered_by_E1BearerContextReleaseRequest_in_suspend]
    // TC_ID=[FGR-MC1016:FR800:RCC-T76352]

    // TC_NAME=[test_SaRrcInactive_Json:All/T76353_Explicit_release_triggered_by_NGUEContextReleaseCommand_in_suspend]
    // TC_ID=[FGR-MC1016:FR800:RCC-T76353]

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
