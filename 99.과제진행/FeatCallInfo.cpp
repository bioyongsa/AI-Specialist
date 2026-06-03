/**
 * @file        VlibCallInfo.c
 * @brief       functions for CP call context
 */

#include "FeatCallInfo.h"

#include "ComInclude.h"
#include "ComLibUtility.h"
#include "ComCrrMapper.hpp"

#include "VccbAiEngine.h"   // [claude] VccbAi_InitContext() : init per-UE AI context after 0xFF memset

#include "FeatAupf.h"
#include "FeatCommon.h"
#include "FeatHoCommon.h"
#include "FeatMco.h"
#include "FeatSon.h"
#include "FeatMrdc.h"
#include "FeatCallInfo.h"
#include "FeatCallProc.h"
#include "FeatSecurity.h"
#include "FeatRrcSegment.h"
#include "FeatFullConfig.h"
#include "FeatMeasurements.hpp"
#include "FeatLb.hpp"
#include "FeatUlPathControlNRDC.h"
#include "FeatCa.hpp"
#include "FeatAnr.hpp"
#include "UplinkStatusTriggeredMobility.hpp"
#include "StateCommon.h"
#include "MsgQueueManager.h"

#include "NRRRC.h"

#include <cstddef>
#include <iomanip>

/**
 * @brief     Find the highest order value among 5QI and set it to select5QI.
 * @param     ptCi : call context
 * @param     pDrxProfile : gNB Drx Profile
 * @param     select5QI : Sets to 5QI with the highest order within the function
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int8_t FnVCCB_SelectOrder5QI(callInfo_type *ptCi, drxConfigProfile_type *pDrxProfile, uint8_t *select5QI)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);
    VALID_E_POINT_CHECK_ERROR(select5QI);

    uint8_t selectOrder    = UNKNOWN_SELECTION_ORDER;  // Valid Range: 0~15
    uint8_t selectionOrder = UNKNOWN_SELECTION_ORDER;  // Valid Range: 0~15
    uint8_t isFindFirstQosFlow = FALSE;

    uint16_t pduIdx;
    for (pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
    {
        if (!IS_VALID_NODE(PDU_NODE, ptCi, &(ptCi->pduSess[pduIdx]))) continue;

        uint16_t pduQosIdx;
        for (pduQosIdx = 0; pduQosIdx < MAX_QOS_FLOW_PER_PDU_SESS; pduQosIdx++)
        {
            uint16_t qosIdx = ptCi->pduSess[pduIdx].qosFlowIndex[pduQosIdx];
            if (qosIdx >= MAX_QOS_FLOW_PER_CALL) continue;

            qosFlowInfo_type *ptQi = &ptCi->qosFlow[qosIdx];
            if (ptQi->usage != CI_use) continue;

            if (VLIB_IsCommonValidStateOfPduQosDrb(ptQi->qosFlowStatus) == ERROR) continue;

            uint8_t fiveqi = 0;
            VLIB_getFiveQi(ptQi, &fiveqi);  // Range check is not required because of data type

            if (IS_REDCAP_UE(ptCi) && pDrxProfile->drxConfigFlagPer5qiRedCap == TRUE)
            {
                selectionOrder = pDrxProfile->drxConfigCu5qiRedCap[fiveqi].selectionOrder;
            }
            else
            {
                selectionOrder = pDrxProfile->drxConfigCu5qi[fiveqi].selectionOrder;
            }

            if (isFindFirstQosFlow == FALSE)
            {
                selectOrder = selectionOrder;
                *select5QI = fiveqi;
                isFindFirstQosFlow = TRUE;
            }
            else if (selectOrder > selectionOrder)
            {
                selectOrder = selectionOrder;
                *select5QI = fiveqi;
            }
            else if (selectOrder == selectionOrder)
            {
                if (*select5QI > fiveqi)
                {
                    *select5QI = fiveqi;
                }
            }
        }
    }

    if (isFindFirstQosFlow == FALSE)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "[C-DRX] There is no QosFlow!\n");
        return ERROR;
    }

    return GOOD;
}

/**
 * @brief     Find the highest order value among QCI and set it to selectQCI.
 * @param     ptCi : call context
 * @param     pDrxProfile : gNB Drx Profile
 * @param     selectQCI : Sets to QCI with the highest order within the function
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int8_t FnVCCB_SelectOrderQCI(callInfo_type *ptCi, drxConfigProfile_type *pDrxProfile, uint16_t *selectQCI)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);
    VALID_E_POINT_CHECK_ERROR(selectQCI);

    uint8_t isFindFirstDrb = FALSE;
    uint8_t selectOrder = UNKNOWN_SELECTION_ORDER;  // Valid Range: 0~15

    // QCI having lowest selection order is chosen as the reference QCI
    // if more than one QCI have same selection order, then lowest QCI is selected
    uint8_t rab;
    for (rab = 0; rab < MAX_ERAB_PER_CALL; rab++)
    {
        if (!IS_VALID_NODE(ERAB_NODE, ptCi, &(ptCi->erabInfo[rab]))) continue;
        if ((ptCi->erabInfo[rab].erabStatus != ERAB_Setup) && (ptCi->erabInfo[rab].erabStatus != ERAB_Modify) &&
            (ptCi->erabInfo[rab].erabStatus != ERAB_Idle))
            continue;
        uint16_t qci = ptCi->erabInfo[rab].erabQos.qciValue.QCI;
        VCCB_CHECK_E_RANGE_AND_RETURN(qci, MAX_QCI_VALUE);

        if (isFindFirstDrb == FALSE)
        {
            selectOrder = pDrxProfile->drxConfigCuQci[qci].selectionOrder;
            *selectQCI = qci;
            isFindFirstDrb = TRUE;
        }
        else if (selectOrder > pDrxProfile->drxConfigCuQci[qci].selectionOrder)
        {
            selectOrder = pDrxProfile->drxConfigCuQci[qci].selectionOrder;
            *selectQCI = qci;
        }
        else if (selectOrder == pDrxProfile->drxConfigCuQci[qci].selectionOrder)
        {
            if (*selectQCI > qci)
            {
                *selectQCI = qci;
            }
        }
    }
    if (isFindFirstDrb == FALSE)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "[%s] There is no Active QCI\n", __func__);
        return ERROR;
    }
    return GOOD;
}

int8_t FnVCCB_SetDrxShortCycle(
    callInfo_type *ptCi, drxState_t shortDrxConfigSetup, uint8_t shortCycle, uint8_t shortCycleTimer, uint8_t longCycle)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    ptCi->drxCuInfo.shortCycleStatus = DRX_OFF;

    if (dbgFlag == VCCB_FLAG_ON)
    {
        VCCB_PRINT_CALL_INFORMATIVE(
            ptCi, "[C-DRX] shortDrxConfigSetup(%s)\n", shortDrxConfigSetup ? "DRX_ON" : "DRX_OFF");
    }

    if (shortDrxConfigSetup == DRX_ON)
    {
        int longCycleTime = FnVCCB_GetLongDrxCycleTime(static_cast<enumVccbLongDrxCycle_type>(longCycle));
        if (longCycleTime < 0)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_GetLongDrxCycleTime is Failed! longCycle(%d)\n", longCycle);
            return ERROR;
        }
        int shortCycleTime = FnVCCB_GetShortDrxCycleTime(static_cast<enumVccbShortDrxCycle_type>(shortCycle));
        if (shortCycleTime <= 0)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_GetShortDrxCycleTime is Failed! shortCycle(%d)\n", shortCycle);
            return ERROR;
        }
        if (longCycleTime % shortCycleTime == 0)
        {
            ptCi->drxCuInfo.shortCycleStatus = DRX_ON;
            ptCi->drxCuInfo.shortCycle = shortCycle;
            ptCi->drxCuInfo.shortCycleTimer = shortCycleTimer;
            if (dbgFlag == VCCB_FLAG_ON)
            {
                VCCB_PRINT_CALL_INFORMATIVE(
                    ptCi, "[C-DRX] shortCycle(%dms), shortCycleTimer(%u) are set\n", shortCycleTime, shortCycleTimer);
            }
        }
        else if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi,
                "[C-DRX] shortDrxCycle is OFF (longCycle(%dms) mod shortCycle(%dms) is not zero)\n", longCycleTime,
                shortCycleTime);
        }
    }

    return GOOD;
}

int8_t FnVCCB_SetDrxCycle(callInfo_type *ptCi, drxConfigCu_type *pDrxCfg)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxCfg);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    ptCi->drxCuInfo.status = DRX_OFF;
    if (dbgFlag == VCCB_FLAG_ON)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxConfigSetup(%s)\n", pDrxCfg->drxConfigSetup ? "DRX_ON" : "DRX_OFF");
    }
    if (pDrxCfg->drxConfigSetup == DRX_ON)
    {
        // Set Long Drx Cycle
        ptCi->drxCuInfo.status = DRX_ON;
        ptCi->drxCuInfo.longCycle = pDrxCfg->longCycle;

        if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] longCycle(%dms) is set\n",
                FnVCCB_GetLongDrxCycleTime(static_cast<enumVccbLongDrxCycle_type>(pDrxCfg->longCycle)));
        }

        if (ERROR == FnVCCB_SetDrxShortCycle(ptCi, pDrxCfg->shortDrxConfigSetup, pDrxCfg->shortCycle,
                         pDrxCfg->shortCycleTimer, pDrxCfg->longCycle))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxShortCycle is Failed!\n");
            return ERROR;
        }
    }
    return GOOD;
}

/**
 * @brief     Set LongCycle using 5QI for SA User
 * @param     ptCi : call context
 * @return    GOOD : success
 * @return    ERROR : failure
 */
void VLIB_SetCuDrxCycleUsing5QI(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    uint8_t select5QI;

    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    if (ERROR == FnVCCB_SelectOrder5QI(ptCi, pDrxProfile, &select5QI))
    {
        if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] This gNB(gnbNum = %u) has no Active 5QI\n", ptCi->gnbNum);
        }
        return;
    }

    if (dbgFlag == VCCB_FLAG_ON)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] 5QI(%u) is Selected\n", select5QI);
    }

    //SET REDCAP OR NORM_VAL
    drxConfigCu_type *pDrxCfg;

    if (IS_REDCAP_UE(ptCi) && pDrxProfile->drxConfigFlagPer5qiRedCap == TRUE)
    {
        pDrxCfg = &pDrxProfile->drxConfigCu5qiRedCap[select5QI];
    }
    else
    {
        pDrxCfg = &pDrxProfile->drxConfigCu5qi[select5QI];
    }

    if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrxCfg))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
        return;
    }

    return;
}

/**
 * @brief     Set LongCycle using QCI for NSA User
 * @param     ptCi : call context
 * @return    GOOD : success
 * @return    ERROR : failure
 */
void VLIB_SetCuDrxCycleUsingQCI(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    uint16_t selectQCI = 0;

    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    if (ERROR == FnVCCB_SelectOrderQCI(ptCi, pDrxProfile, &selectQCI))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] This gNB(gnbNum = %u) has no Active QCI\n", ptCi->gnbNum);
        return;
    }

    if (dbgFlag == VCCB_FLAG_ON)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] QCI(%u) is Selected\n", selectQCI);
    }

    drxConfigCu_type *pDrxQciCfg = &pDrxProfile->drxConfigCuQci[selectQCI];
    if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrxQciCfg))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
        return;
    }

    return;
}

/**
 * @brief     Set LongCycle using 5QI for SA User
 * @param     ptCi : call context
 * @param     pDrxProfile : gNB DrX Profile
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int8_t FnVCCB_SetDrxCycleToNgranRfsp(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    uint16_t rfspId = ptCi->ueInfo.rfsp.subscriberProfile;
    if (rfspId > MAX_ELEM_DRX_PER_RFSP || rfspId == 0)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] subscriberProfile(drx-rfsp) is out of range(%u)\n", rfspId);
        return ERROR;
    }

    drxConfigCu_type *pDrxCfg;

    if (IS_REDCAP_UE(ptCi))
    {
        pDrxCfg = &pDrxProfile->drxNgranRfspRedCap[rfspId];
    }
    else
    {
        pDrxCfg = &pDrxProfile->drxNgranRfsp[rfspId];
    }

    if (pDrxCfg->status == EQUIP)
    {
        if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrxCfg))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
            return ERROR;
        }
        return GOOD;
    }
    else
    {
        if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxNgranRfsp(%u) status is not equip\n", rfspId);
        }
        return ERROR;
    }
}

/**
 * @brief     Set LongCycle using QCI for NSA User
 * @param     ptCi : call context
 * @param     pDrxProfile : gNB DrX Profile
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int8_t FnVCCB_SetDrxCycleToEutranSpid(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    uint16_t spidId = ptCi->ueInfo.rfsp.subscriberProfile;
    if (spidId > MAX_ELEM_DRX_PER_SPID || spidId == 0)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] subscriberProfile(drx-spid) is out of range(%d)\n", spidId);
        return ERROR;
    }
    drxConfigCu_type *pDrxCfg = &pDrxProfile->drxEutranSpid[spidId];
    if (pDrxCfg->status == EQUIP)
    {
        if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrxCfg))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
            return ERROR;
        }
        return GOOD;
    }
    else
    {
        if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxEutranSpid(%u) status is not equip\n", spidId);
        }
        return ERROR;
    }
}

int8_t FnVCCB_SetDrxQci128ForArp2UE(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    static const uint8_t ARP2 = 2;
    static const uint8_t QCI128 = 128;
    static const uint8_t FiveQI128 = 128;
    if (pDrxProfile->drxArp2TypeActivation == TRUE)
    {
        if (ptCi->callType == E_CALLTYPE_SA)
        {
            for (uint16_t pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
            {
                if (!IS_VALID_NODE(PDU_NODE, ptCi, &(ptCi->pduSess[pduIdx]))) continue;

                for (uint8_t pduQosIdx = 0; pduQosIdx < MAX_QOS_FLOW_PER_PDU_SESS; pduQosIdx++)
                {
                    uint16_t qosIdx = ptCi->pduSess[pduIdx].qosFlowIndex[pduQosIdx];
                    if (qosIdx >= MAX_QOS_FLOW_PER_CALL) continue;

                    qosFlowInfo_type *ptQi = &ptCi->qosFlow[qosIdx];
                    if (ptQi->usage != CI_use) continue;

                    if (VLIB_IsCommonValidStateOfPduQosDrb(ptQi->qosFlowStatus) == ERROR) continue;

                    if (ptQi->qosParam.allocAndRetent.priorityLevel == ARP2)
                    {
                        if (dbgFlag == VCCB_FLAG_ON)
                        {
                            VCCB_PRINT_CALL_INFORMATIVE(
                                ptCi, "[C-DRX][%s] PduSess[%u], qosFlow[%u] ARP is 2 \n", __func__, pduIdx, qosIdx);
                        }

                        drxConfigCu_type *pDrx5qiCfg;

                        if (IS_REDCAP_UE(ptCi) && pDrxProfile->drxConfigFlagPer5qiRedCap == TRUE)
                        {
                            pDrx5qiCfg = &pDrxProfile->drxConfigCu5qiRedCap[FiveQI128];
                        }
                        else
                        {
                            pDrx5qiCfg = &pDrxProfile->drxConfigCu5qi[FiveQI128];
                        }

                        if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrx5qiCfg))
                        {
                            VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
                        }
                        return GOOD;
                    }
                }
            }
            if (dbgFlag == VCCB_FLAG_ON)
            {
                VCCB_PRINT_CALL_INFORMATIVE(
                    ptCi, "[C-DRX][%s] drxArp2TypeActivation is True, but ue is not ARP 2 \n", __func__);
            }
        }
        else
        {
            for (uint8_t erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
            {
                callERABInfo_type *ptRi = &ptCi->erabInfo[erabIdx];
                if (CI_use != ptRi->usage) continue;
                if (VLIB_IsCommonValidStateOfPduQosDrb(ptRi->erabStatus) == ERROR) continue;
                if (ptRi->erabQos.aRP.priority == ARP2)
                {
                    if (dbgFlag == VCCB_FLAG_ON)
                    {
                        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX][%s] erabInfo[%u] ARP is 2 \n", __func__, erabIdx);
                    }
                    drxConfigCu_type *pDrxQciCfg = &pDrxProfile->drxConfigCuQci[QCI128];
                    if (ERROR == FnVCCB_SetDrxCycle(ptCi, pDrxQciCfg))
                    {
                        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
                    }
                    return GOOD;
                }
            }
        }
    }
    return ERROR;
}

int8_t FnVCCB_SetDrxServiceGroup(callInfo_type *ptCi, struct drxConfigProfile *pDrxProfile)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pDrxProfile);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    if (pDrxProfile->drxConfigFlagPerServiceGrp != TRUE){
       return ERROR;
    }

    uint8_t svcGrpId = 0;
    uint8_t isReleaseProcedure = FALSE;  // FALSE : Do not count 'S_Release' PDU as a service group candidate.
    if (ERROR == FnVCCB_GetServiceGroupId(ptCi, &svcGrpId, isReleaseProcedure))
    {
        if (dbgFlag == VCCB_FLAG_ON) {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX][%s] FnVCCB_GetServiceGroupId is ERROR \n", __func__);
        }
        return ERROR;
    }

    if (dbgFlag == VCCB_FLAG_ON) {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxConfigFlagPerServiceGrp: %s, svcGrpId: %d status: %d\n",
            pDrxProfile->drxConfigFlagPerServiceGrp ? "TRUE" : "FALSE", svcGrpId, pDrxProfile->drxConfigCuServiceGrp[svcGrpId].status);
    }

    if (pDrxProfile->drxConfigCuServiceGrp[svcGrpId].status != EQUIP){
        if (dbgFlag == VCCB_FLAG_ON) {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX][%s] pDrxProfile->drxConfigCuServiceGrp[svcGrpId:%d].status != EQUIP \n", __func__, svcGrpId );
        }
        return ERROR;
    }

    if (ERROR == FnVCCB_SetDrxCycle(ptCi, &(pDrxProfile->drxConfigCuServiceGrp[svcGrpId])))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "[C-DRX] FnVCCB_SetDrxCycle is Failed!\n");
    }

    return GOOD;
}

/**
 * @brief     Make connected DRX configuration
 * @param     ptCi : call context
 * @return    GOOD : success
 * @return    ERROR : failure
 */
void VLIB_ConfigDrxCuInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    int8_t dbgFlag = VCCB_FLAG_OFF;
    if (DBG->flag.drxConfig == VCCB_FLAG_ON)
    {
        dbgFlag = VCCB_FLAG_ON;
    }

    if (IS_DRX_DISABLED_FOR_SRS(ptCi))
    {
        ptCi->drxCuInfo.status = DRX_OFF;
        if (dbgFlag == VCCB_FLAG_ON)
        {
            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] Drx set as OFF because SrsPosCfg is on going.\n");
        }
        return;
    }

    // initialize DRX as off
    ptCi->drxCuInfo.status = DRX_OFF;
    uint16_t cellNum = ptCi->servCellNum;
    // For the HandOver procedure, DRX must be set to the changed cell.
    if ((IntraRatHO_cmd == ptCi->stateCmd) && (ptCi->hoInfo.hoType.nb == HO_TYPE_INTRA_NB))
    {
        cellNum = ptCi->hoInfo.targetCellNum;
    }
    VCCB_CHECK_E_RANGE_AND_VOID(cellNum, MAX_CELL_NUM_VCCB);
    VccbCellInfo_type *const pCell = &CELLINFO[cellNum];

    uint8_t drxProfileIdx = pCell->profileDBInfo.drxProfileIdx;

    VCCB_CHECK_E_RANGE_AND_VOID(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    VCCB_CHECK_E_RANGE_AND_VOID(drxProfileIdx, MAX_DRX_PROFILE);
    struct drxConfigProfile *pDrxProfile = &(pGnb->drxConfigProfile[drxProfileIdx]);

    // decide DRX on or off
    // check UE capa
    if (VCCM_getUeLongDrxCycleCapa(ptCi, cellNum) != TRUE)
    {
        VCCB_PRINT_CALL_DATA(ptCi, "[VLIB_ConfigDrxCuInfo] UE capa does not support long DRX\n");
        return;
    }

    // [SVR26B] Service Group DRX
    if (GOOD == FnVCCB_SetDrxServiceGroup(ptCi, pDrxProfile))
    {
        return;
    }

    // [SVR22C2][SKT]
    // drxArp2TypeActivation is true and UE is ARP2 -> set QCI 128
    if (GOOD == FnVCCB_SetDrxQci128ForArp2UE(ptCi, pDrxProfile))
    {
        return;
    }

    if (dbgFlag == VCCB_FLAG_ON)
    {
        VCCB_PRINT_CALL_INFORMATIVE(
            ptCi, "[C-DRX] Drx Config (cellNum:%u, drxProfileIdx:%u redCap:%d)\n", cellNum, drxProfileIdx, IS_REDCAP_UE(ptCi));
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] rfspPresent: %s, subscriberProfile: %d\n",
            ptCi->ueInfo.rfsp.rfspPresent ? "OPT_PRESENT" : "OPT_NOT_PRESENT", ptCi->ueInfo.rfsp.subscriberProfile);
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxConfigFlagPerRfsp: %s, drxConfigFlagPerSpid: %s\n",
            pDrxProfile->drxConfigFlagPerRfsp ? "TRUE" : "FALSE", pDrxProfile->drxConfigFlagPerSpid ? "TRUE" : "FALSE");
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "[C-DRX] drxConfigFlagPerRfspRedCap: %s, drxConfigFlagPer5qiRedCap: %s\n",
            pDrxProfile->drxConfigFlagPerRfspRedCap ? "TRUE" : "FALSE", pDrxProfile->drxConfigFlagPer5qiRedCap ? "TRUE" : "FALSE");
    }

    if (ptCi->callType == E_CALLTYPE_SA)
    {
        if (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT &&
           (pDrxProfile->drxConfigFlagPerRfsp == TRUE || (IS_REDCAP_UE(ptCi) && pDrxProfile->drxConfigFlagPerRfspRedCap == TRUE)))
        {
            if (GOOD == FnVCCB_SetDrxCycleToNgranRfsp(ptCi, pDrxProfile))
            {
                return;
            }
        }
        VLIB_SetCuDrxCycleUsing5QI(ptCi, pDrxProfile);
    }
    else
    {
        if (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT && pDrxProfile->drxConfigFlagPerSpid == TRUE)
        {
            if (GOOD == FnVCCB_SetDrxCycleToEutranSpid(ptCi, pDrxProfile))
            {
                return;
            }
        }
        VLIB_SetCuDrxCycleUsingQCI(ptCi, pDrxProfile);
    }
    return;
}

void FnVCCB_StoreSdapInfoPrev(callDRBInfo_type *ptDi)
{
    VALID_E_POINT_CHECK_VOID(ptDi);

    if (ptDi->rbInfo.sdapInfoPrevFlag == VCCB_FLAG_ON) return;

    ptDi->rbInfo.sdapInfoPrevFlag = VCCB_FLAG_ON;
    ptDi->rbInfo.sdapInfoPrev.defaultDRB = ptDi->rbInfo.sdapInfo.defaultDRB;
    ptDi->rbInfo.sdapInfoPrev.headerDl = ptDi->rbInfo.sdapInfo.headerDl;
    ptDi->rbInfo.sdapInfoPrev.headerUl = ptDi->rbInfo.sdapInfo.headerUl;
}

/**
 * @brief     Initialize PDU information in call context
 * @param     ptCi : call context
 * @param     ptPi : PDU information
 */
void VLIB_InitCallPDUInfo(callInfo_type *ptCi, callPDUSessInfo_type *ptPi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(ptPi);

    VCCB_CHECK_E_RANGE_AND_VOID(ptCi->gnbNum, MAX_GNB_PER_SYS);

    memset(reinterpret_cast<uint8_t *>(ptPi), 0x00, sizeof(callPDUSessInfo_type));
    memset(reinterpret_cast<uint8_t *>(ptPi->drbIndex), 0xFF, sizeof(ptPi->drbIndex));
    memset(reinterpret_cast<uint8_t *>(ptPi->qosFlowIndex), 0xFF, sizeof(ptPi->qosFlowIndex));
    memset(reinterpret_cast<uint8_t *>(ptPi->mappingRuleDrbIdx), 0xFF, sizeof(ptPi->mappingRuleDrbIdx));
    ptPi->defaultGroupPresent = OPT_NOT_PRESENT;
    ptPi->mappingDrbIdxDefaultGroup = 0xFF;
    ptPi->e1Idx = UNKNOWN_UPIDX;

    ptPi->dataForwarding.notPossible = OPT_NOT_PRESENT;

    FnVCCB_InitPduDataFwdTunnel(&ptPi->nGuFwdTunnel);
    ptPi->directFwdPathAvailable = false;

    ptPi->firstReqStatusInThisProcedure = S_Idle;
    ptPi->prevPduSessStatus = S_Idle;
    ptPi->pduSessStatus = S_Idle;

    ptPi->securityInd.integProtectInd = ci_integ_protect_not_needed;
    ptPi->securityInd.confidenProtectInd = ci_confiden_protect_required;
    ptPi->securityInd.maxIPrate = ci_max_ip_rate_bitrate64kbs;
    ptPi->securityInd.maxIPrateDL = ci_max_ip_rate_bitrate64kbs;
    ptPi->securityInd.securityInfoNotAvailable = FALSE;
    ptPi->securityInd.maxIPrateDLPresent = OPT_NOT_PRESENT;

    ptPi->securityResult.integrityProtectionResult = ci_integ_protect_not_performed;
    ptPi->securityResult.confidentialityProtectionResult = ci_confiden_protect_performed;

    ptCi->numPDUSess++;
    ptPi->usage = CI_use;

    ptPi->cslAllocOrder = ptCi->cslInfo.cslErabPduDrbAllocIdx;
}

/**
 * @brief     Initialize refusedTargetFR1CellsForFR2FR1HO list with restricted cells
 * @param     ptCi : call context
 */
void VLIB_InitRefusedTargetFR1CellsForFR2FR1HO(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    uint8_t loop = 0;

    for (loop = 0; loop < MAX_REFUSED_TARGET_FR1_CELLS_PER_CALL; loop++)
    {
        ptCi->refusedTargetFR1CellsForFR2FR1HO[loop].status = N_EQUIP;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[loop].globalGnbId.gnbId.id = 0;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[loop].globalGnbId.gnbId.idLength = 0;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[loop].cellId = 0;
    }

#ifdef NEW_HOST_SIM
    if (VCCB_FLAG_ON == DBG->flag.refusedIntraFr1TargetCell)
    {
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].status = EQUIP;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].globalGnbId.gnbId.idLength = 22;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].cellId = 1;

        ptCi->refusedTargetFR1CellsForFR2FR1HO[1].status = EQUIP;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[1].globalGnbId.gnbId.idLength = 22;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[1].cellId = 100;
    }
    else if (VCCB_FLAG_ON == DBG->flag.refusedInterFr1TargetCell)
    {
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].status = EQUIP;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].globalGnbId.gnbId.id = 1;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].globalGnbId.gnbId.idLength = 22;
        ptCi->refusedTargetFR1CellsForFR2FR1HO[0].cellId = 100;
    }
#endif

    return;
}

/**
 * @brief     Allocate PDU in call context
 * @param     ptCi : call context
 * @param     pduIdx : PDU Session index to allocate / uint16_t
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int16_t VLIB_AllocCallPDUInfo(callInfo_type *ptCi, uint8_t *ptAllocatedPduIdx)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAllocatedPduIdx);

    uint8_t loop;

    callPDUSessInfo_type *ptPi = NULL;

    for (loop = 0; loop < MAX_PDU_SESS_PER_CALL; loop++)
    {
        if (ptCi->pduSess[loop].usage != CI_use) break;
    }

    if (loop == MAX_PDU_SESS_PER_CALL)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get Free pduIdx...\n");
        return ERROR;
    }

    ptPi = &ptCi->pduSess[loop];
    VLIB_InitCallPDUInfo(ptCi, ptPi);

    *ptAllocatedPduIdx = loop;

    return GOOD;
}

void FnVCCB_InitDrbInfo(callDRBInfo_type *ptDi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptDi);

    memset(reinterpret_cast<uint8_t *>(ptDi), 0x00, sizeof(callDRBInfo_type));
    memset(reinterpret_cast<uint8_t *>(ptDi->qosFlowIndex), 0xFF, sizeof(ptDi->qosFlowIndex));

    ptDi->usage = CI_use;

    ptDi->pduIdxInCall = 0xFF;
    ptDi->preemptionReqCellNum = 0xFFFF;

    ptDi->failCause = CAUSE_NO_FAULT;
    ptDi->firstReqStatusInThisProcedure = S_Idle;
    ptDi->prevDrbStatus = S_Idle;
    ptDi->drbStatus = S_Idle;
    ptDi->retainStat.inSessionFlagByE1 = VCCB_FLAG_OFF;
    SET_DRB_RETAIN_STAT_CAPABLE_FLAG(ptDi, VCCB_FLAG_OFF);
    CLEAR_DRB_RETAIN_STAT_EVENT(ptDi);

    ptDi->prevSnDrbStatus = S_Idle;
    ptDi->snDrbStatus = S_Idle;

    FnVCCB_InitDrbDataFwdTunnel(&ptDi->dataFwdInfo.gtpTunnel);
    FnVCCB_InitDrbPdcpSnStatus(&ptDi->dataFwdInfo.snStatus);

    ptDi->drbQosIdxInCall = 0xFFFF;

    ptDi->ulPdcpPathPrev = UlPdcpPath_ScgShared;
    ptDi->ulPdcpPath = UlPdcpPath_ScgShared;

    ptDi->ulPdcpPathToMod = UlConfigBearer_ModOff;
    ptDi->ulPdcpPathEnq = VCCB_FLAG_OFF;
    ptDi->ulPdcpPathEnqToSet = UlPdcpPath_ScgShared;

    ptDi->activityStatus = ActivityStatus_ACTIVE;
    ptDi->qfiIdsToRelease = 0;  // NG_HOREQ_PDU_QFI_DIFF_21B

    ptDi->selectedBearerTypeFrozenFlag = VCCB_FLAG_OFF;
    ptDi->bearerType.val = MN_Term_MCG_Bearer; // for SA-Basic
    ptDi->reqBearerType.val = INIT_BEARER_TYPE;
    ptDi->selectedBearerType.val = INIT_BEARER_TYPE;

    ptDi->prevSelectedBearerTypeFrozenFlag = VCCB_FLAG_OFF;
    ptDi->prevReqBearerType.val = INIT_BEARER_TYPE;
    ptDi->prevSelectedBearerType.val = INIT_BEARER_TYPE;

    ptDi->xnModReqUlPdcpConfigChangeFlag = FALSE;
    ptDi->drbReflectiveQosFlag = VCCB_FLAG_OFF;

    ptDi->qosMonitorActivated = VCCB_FLAG_OFF;
    ptDi->drbQosMonitoring = QOS_MONITORING_REQUEST_NOT_PRESENT;

    ptDi->rbInfo.pdcpInfo.isPdcpConfigured = VCCB_FLAG_OFF;

    ptDi->rbInfo.pdcpInfo.headerCompression.rohc_m.ulRohcEnabled = OPT_NOT_PRESENT;
    ptDi->rbInfo.pdcpInfo.headerCompression.rohc_m.symmetricRohcEnabled = OPT_NOT_PRESENT;
    ptDi->rbInfo.pdcpInfo.m.headerCompression = OPT_NOT_PRESENT;

    //EHC
    ptDi->rbInfo.pdcpInfo.m.ethernet_headerCompression = OPT_NOT_PRESENT;
    ptDi->pdcpDuplicationActive = FALSE;
}

/**
 * @brief     Allocate QoS Flow in PDU info
 * @param     ptCi : call context
 * @param     ptAllocatedIdx : allocated index / uint16_t
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int16_t VLIB_AllocCallDrbInfo(callInfo_type *ptCi, callPDUSessInfo_type *ptPi, uint8_t *ptAllocatedIdx)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptPi);
    VALID_E_POINT_CHECK_ERROR(ptAllocatedIdx);

    uint8_t drbIdxInPdu;
    uint8_t drbIdxInCall;

    callDRBInfo_type *ptDi = NULL;

    uint8_t maxDrbCount = MAX_DRB_PER_SA_CALL;
    VLIB_GetUeTypeMaxDrbCount (ptCi, &maxDrbCount);

    const uint8_t MaxDrbPerSaCall = maxDrbCount;

    for (drbIdxInPdu = 0; drbIdxInPdu < MAX_DRB_PER_PDU_SESS; drbIdxInPdu++)
    {
        if (ptPi->drbIndex[drbIdxInPdu] >= MaxDrbPerSaCall)  // Not yet used
            break;
    }

    if (drbIdxInPdu == MAX_DRB_PER_PDU_SESS)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get Free drbIdx in PDU Session...\n");
        return ERROR;
    }

    for (drbIdxInCall = 0; drbIdxInCall < MaxDrbPerSaCall; drbIdxInCall++)
    {
        ptDi = &ptCi->drb[drbIdxInCall];

        if (ptDi->usage != CI_use) break;
    }

    if (drbIdxInCall == MaxDrbPerSaCall)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get Free drbIdx in Call...\n");
        return ERROR;
    }

    ptPi->drbIndex[drbIdxInPdu] = drbIdxInCall;
    ptPi->numDRBPerPDUSess++;

    FnVCCB_InitDrbInfo(ptDi);
    ptDi->cslAllocOrder = ptCi->cslInfo.cslErabPduDrbAllocIdx;

    *ptAllocatedIdx = drbIdxInPdu;

    return GOOD;
}

void VLIB_InitCallQosFlowInfo(qosFlowInfo_type *ptQi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptQi);

    memset(reinterpret_cast<uint8_t *>(ptQi), 0x00, sizeof(qosFlowInfo_type));

    ptQi->usage = CI_use;

    ptQi->drbIdxInCall = 0xFF;

    ptQi->failCause = CAUSE_NO_FAULT;
    ptQi->firstReqStatusInThisProcedure = S_Idle;

    ptQi->prevQosFlowStatus = S_Idle;
    ptQi->qosFlowStatus = S_Idle;
    ptQi->qosFlowSetupRequestItemeRABIDPresent = OPT_NOT_PRESENT;
    ptQi->erabId = 0xFF;
    VCCB_SetBearerReleaseInfo(&ptQi->releaseInfo, e_ReleaseTriggerNotRelease, CAUSE_NO_FAULT); /* Init Value */

    FnVCCB_InitQosFlowDataFwdInfo(&ptQi->dataFwdInfo);
}

/**
 * @brief     Allocate QoS Flow in PDU info
 * @param     ptCi : call context
 * @param     ptAllocatedIdx : allocated index / uint16_t
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int16_t VLIB_AllocCallQosFlowInfo(callInfo_type *ptCi, callPDUSessInfo_type *ptPi, uint8_t *ptAllocatedIdx)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptPi);
    VALID_E_POINT_CHECK_ERROR(ptAllocatedIdx);

    uint8_t qosIdxInPdu;
    uint8_t qosIdxInCall;

    qosFlowInfo_type *ptQi = NULL;

    for (qosIdxInPdu = 0; qosIdxInPdu < MAX_QOS_FLOW_PER_PDU_SESS; qosIdxInPdu++)
    {
        if (ptPi->qosFlowIndex[qosIdxInPdu] >= MAX_QOS_FLOW_PER_CALL)  // Not yet used
            break;
    }

    if (qosIdxInPdu == MAX_QOS_FLOW_PER_PDU_SESS)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get Free qosIdx in PDU Session...\n");
        return ERROR;
    }

    for (qosIdxInCall = 0; qosIdxInCall < MAX_QOS_FLOW_PER_CALL; qosIdxInCall++)
    {
        ptQi = &ptCi->qosFlow[qosIdxInCall];

        if (ptQi->usage == CI_no_use) break;
    }

    if (qosIdxInCall == MAX_QOS_FLOW_PER_CALL)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get Free qosIdx in Call...\n");
        return ERROR;
    }

    ptPi->qosFlowIndex[qosIdxInPdu] = qosIdxInCall;
    ptPi->numQoSFlowPerPDUSess++;

    VLIB_InitCallQosFlowInfo(ptQi);

    *ptAllocatedIdx = qosIdxInPdu;

    return GOOD;
}

/**
 * @brief           Allocate gNB information in call context
 * @details         Allocate new gNB information e.g. for HO purpose. If given gNB index is MAX_GNB_PER_CALL,
 *                  function will try to allocate it on first free element of gNB info array and return selected index
 * in ptGnbIdx. If function won't find free space in gNB info array, it will return MAX_GNB_PER_CALL value in ptGnbIdx.
 *
 * @param[in/out]   ptCi : call context / callInfo_type
 * @param[in/out]   ptGnbIdx : gNB index to allocate / uint16_t
 * @return          GOOD, ERROR / int16_t
 */
int16_t VLIB_AllocCallGnbInfo(callInfo_type *ptCi, uint16_t *ptGnbIdx)
{
    AUTO_TRACE_FUNC;

    uint16_t hoGnbIdx = 0;
    uint16_t loop = 0;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptGnbIdx);

    hoGnbIdx = *ptGnbIdx;

    if (hoGnbIdx < MAX_GNB_PER_CALL)
    {
        if (IS_VALID_NODE(GNB_NODE, ptCi, &(ptCi->gnbInfo[hoGnbIdx])))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "gNB info with index(%d) already allocated!\n", hoGnbIdx);
            return ERROR;
        }
    }
    else
    {
        for (loop = 0; loop < MAX_GNB_PER_CALL; loop++)
        {
            if (!IS_VALID_NODE(GNB_NODE, ptCi, &(ptCi->gnbInfo[loop])))
            {
                break;
            }
        }

        if (MAX_GNB_PER_CALL == loop)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Fail to get free gNB info\n");
            *ptGnbIdx = MAX_GNB_PER_CALL - 1;
            return ERROR;
        }

        hoGnbIdx = loop;
    }

    VLIB_InitCallGnbInfo(&ptCi->gnbInfo[hoGnbIdx], hoGnbIdx);
    ptCi->numGNB++;
    *ptGnbIdx = hoGnbIdx;

    return GOOD;
}

/**
 * @brief     Allocate path information in call context
 * @param     ptCi : call context
 * @param     hoGnbIdx : vNB index to allocate / uint16_t
 * @param     erabIdx : ERAB index / uint16_t
 * @param     pathType : forwarding direction DL or UL / uint16_t
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int16_t VLIB_AllocPathInfo(callInfo_type *ptCi, uint16_t hoGnbIdx, uint16_t erabIdx, uint16_t pathType)
{
    AUTO_TRACE_FUNC;

    callGNBInfo_type *ptEi;
    pathInfo_type *ptPi;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    if (hoGnbIdx >= MAX_GNB_PER_CALL)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Invalid hoGnbIdx(%d) !! \n", hoGnbIdx);
        return ERROR;
    }

    if (erabIdx >= MAX_ERAB_PER_CALL)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Invalid erabIdx(%d) !! \n", erabIdx);
        return ERROR;
    }

    if (!IS_VALID_NODE(GNB_NODE, ptCi, &(ptCi->gnbInfo[hoGnbIdx])))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Invalid hoGnbIdx (%d)\n", hoGnbIdx);
        return ERROR;
    }

    ptEi = &(ptCi->gnbInfo[hoGnbIdx]);

    if (!IS_VALID_NODE(ERAB_NODE, ptCi, &(ptCi->erabInfo[erabIdx])))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Invalid erabIdx (%d)\n", erabIdx);
        return ERROR;
    }

    if (pathType == DL_FORWARDING)
    {
        if (IS_VALID_NODE(PATH_NODE, ptEi, &(ptEi->dlPath[erabIdx])))
        {
            VCCB_PRINT_CALL_ERROR(
                ptCi, "Invalid Path Index (hoGnbIdx/erabIdx/pathType = %d/%d/%d)\n", hoGnbIdx, erabIdx, pathType);
            return ERROR;
        }

        ptPi = &(ptEi->dlPath[erabIdx]);
    }
    else
    {
        if (IS_VALID_NODE(PATH_NODE, ptEi, &(ptEi->ulPath[erabIdx])))
        {
            VCCB_PRINT_CALL_ERROR(
                ptCi, "Invalid Path Index (hoGnbIdx/erabIdx/pathType = %d/%d/%d)\n", hoGnbIdx, erabIdx, pathType);
            return ERROR;
        }

        ptPi = &(ptEi->ulPath[erabIdx]);
    }

    VLIB_InitPathInfo(ptCi, ptPi, hoGnbIdx, erabIdx, pathType);

    ptEi->hoGnbIdx = hoGnbIdx;

    return GOOD;
}

/**
 * @brief     Allocate DRB in PDU info
 * @param     ptCi : call context
 * @param     ptAllocatedIdx : DRB index to allocate
 * @return    GOOD : success
 * @return    ERROR : failure
 **/
int16_t VLIB_AllocNewCallDrbInfo(callInfo_type *ptCi, uint8_t pduIdx, uint16_t qosFlowIdxInPdu,
    uint16_t *ptAllocatedIdx)  // drbIndexInCall
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(pduIdx, MAX_PDU_SESS_PER_CALL);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowIdxInPdu, MAX_QOS_FLOW_PER_PDU_SESS);

    callPDUSessInfo_type *ptPi = &ptCi->pduSess[pduIdx];
    callDRBInfo_type *ptDi = NULL;
    qosFlowInfo_type *ptQi = NULL;

    VCCB_CHECK_E_RANGE_AND_RETURN(ptPi->qosFlowIndex[qosFlowIdxInPdu], MAX_QOS_FLOW_PER_CALL);

    ptQi = &ptCi->qosFlow[ptPi->qosFlowIndex[qosFlowIdxInPdu]];

    uint8_t drbIdxInPdu;
    uint16_t drbIdxInCall;

    if ((ERROR == VLIB_AllocCallDrbInfo(ptCi, ptPi, &drbIdxInPdu)) || (drbIdxInPdu >= MAX_DRB_PER_SA_CALL))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocCallDrbInfo ERROR!\n");
        return ERROR;
    }

    drbIdxInCall = ptPi->drbIndex[drbIdxInPdu];
    VCCB_CHECK_E_RANGE_AND_RETURN(drbIdxInCall, MAX_DRB_PER_SA_CALL);

    SET_DRB_STATUS(ptCi, drbIdxInCall, S_Setup);

    ptDi = &ptCi->drb[drbIdxInCall];

    bool isDrbIdAllocated = true;
    ALLOCATE_DRB_ID_AND_RETURN(ptCi, ptDi, isDrbIdAllocated);

    // FGR-BC0323, FR480 : If DRB wrap around happens, empty all the DRB ID bits from DrbIdPool
    // and allocate DRB ID based on free DRB IDs and enqueue Intra Cell Handover.
    if(!isDrbIdAllocated)
    {
        RESET_DRB_ID_IN_POOL(ptCi);
        ALLOCATE_DRB_ID_AND_RETURN(ptCi, ptDi, isDrbIdAllocated);
        if (GOOD != VCCM_enqueueActionWithForce(ptCi, ENQUEUE_ACTION_INTRA_CELL_HO, true))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_EnqueueIntraCellHO ERROR !!\n");
            return ERROR;
        }
    }

    // After allocate DRB ID, ON_BIT_DRB_ID_IN_POOL should be executed for managing DRB ID.
    ON_BIT_DRB_ID_IN_POOL(ptCi, ptDi->dRBId);

    ptDi->pduIdxInCall = pduIdx;

    int drbQosIdx;
    // ptDi->numQoSFlowPerDRB is initialized as 0 in VLIB_AllocCallDrbInfo.
    for (drbQosIdx = 0; drbQosIdx < MAX_QOS_FLOW_PER_DRB; drbQosIdx++)
    {
        // Find qosFlowIndex in DRB DB using callQosFlowIdx
        if (ptDi->qosFlowIndex[drbQosIdx] == 0xFFFF)
        {
            break;
        }
    }
    if (drbQosIdx >= MAX_QOS_FLOW_PER_DRB)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "No qosflow for drb allocation\n");
        return ERROR;
    }
    ptDi->qosFlowIndex[drbQosIdx] = ptPi->qosFlowIndex[qosFlowIdxInPdu];

    ptDi->numQoSFlowPerDRB++;

    ptQi->drbIdxInCall = drbIdxInCall;

    ptDi->snDrbStatus = S_Idle;

    ptDi->rbInfo.rlcModePresent = OPT_NOT_PRESENT;

    *ptAllocatedIdx = drbIdxInCall;

    return GOOD;
}

int16_t VLIB_FindqosFlowGroupIdxForSlice(
    callInfo_type *ptCi, callPDUSessInfo_type *ptPi, uint16_t qosFlowGroupIdx, uint8_t isDedicated)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptPi);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowGroupIdx, MAX_DEDICATED_QOS_FLOW_GROUP_INDEX);

    uint32_t sstSd = 0xFFFFFFFF;

    uint8_t sliceInfoIndex;

    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    if (isDedicated == TRUE)
    {
        dedicatedQosFlowGroup_type *ptrDedicatedQosFlow = NULL;
        ptrDedicatedQosFlow = &pGnb->cfg.qosDrbMappingConf.dedicatedQosFlowGroupInfo[qosFlowGroupIdx];

        for (sliceInfoIndex = 0; sliceInfoIndex < MAX_SLICE_INFO_PER_DRB; sliceInfoIndex++)
        {
            sstSd = 0xFFFFFFFF;
            if (ptrDedicatedQosFlow->sliceInfo[sliceInfoIndex].status == N_EQUIP) continue;

#if __BYTE_ORDER == __LITTLE_ENDIAN
            sstSd = (static_cast<uint32_t>((ptrDedicatedQosFlow->sliceInfo[sliceInfoIndex].sst & 0xFF) << 24) |
                     (ptrDedicatedQosFlow->sliceInfo[sliceInfoIndex].sd & 0xFFFFFF));
#else
            sstSd = (((ptrDedicatedQosFlow->sliceInfo[sliceInfoIndex].sd & 0xFFFFFF) << 8) |
                     (ptrDedicatedQosFlow->sliceInfo[sliceInfoIndex].sst & 0xFF));
#endif
            if (sstSd == ptPi->sNssai.sstSd) break;
        }
    }

    else
    {
        mappingRule_type *ptrMappingRule = NULL;

        ptrMappingRule = &pGnb->cfg.qosDrbMappingConf.mappingRuleInfo[qosFlowGroupIdx];

        for (sliceInfoIndex = 0; sliceInfoIndex < MAX_SLICE_INFO_PER_DRB; sliceInfoIndex++)
        {
            sstSd = 0xFFFFFFFF;
            if (ptrMappingRule->sliceInfo[sliceInfoIndex].status == N_EQUIP) continue;

#if __BYTE_ORDER == __LITTLE_ENDIAN
            sstSd = (static_cast<uint32_t>((ptrMappingRule->sliceInfo[sliceInfoIndex].sst & 0xFF) << 24) |
                     (ptrMappingRule->sliceInfo[sliceInfoIndex].sd & 0xFFFFFF));
#else
            sstSd = (((ptrMappingRule->sliceInfo[sliceInfoIndex].sd & 0xFFFFFF) << 8) |
                     (ptrMappingRule->sliceInfo[sliceInfoIndex].sst & 0xFF));
#endif
            if (sstSd == ptPi->sNssai.sstSd) break;
        }
    }

    if (sliceInfoIndex < MAX_SLICE_INFO_PER_DRB)
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "VLIB_FindqosFlowGroupIdxForSlice Supported Slice Information found for the Qos Drb Mapping!\n");
        return GOOD;
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "VLIB_FindqosFlowGroupIdxForSlice No supported Slice Information found for the Qos Drb Mapping!\n");
        return ERROR;
    }
}
int16_t VLIB_FindqosFlowGroupIdxForPlmn(callInfo_type *ptCi, uint16_t qosFlowGroupIdx, uint8_t isDedicated)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowGroupIdx, MAX_DEDICATED_QOS_FLOW_GROUP_INDEX);

    uint8_t plmnInfoIndex;
    plmnId_type plmn;
    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    if (isDedicated == TRUE)
    {
        dedicatedQosFlowGroup_type *ptrDedicatedQosFlow = NULL;

        ptrDedicatedQosFlow = &pGnb->cfg.qosDrbMappingConf.dedicatedQosFlowGroupInfo[qosFlowGroupIdx];

        for (plmnInfoIndex = 0; plmnInfoIndex < MAX_PLMN_INFO_PER_DRB; plmnInfoIndex++)
        {
            if (ptrDedicatedQosFlow->plmnInfo[plmnInfoIndex].status == N_EQUIP) continue;

            plmn = ptrDedicatedQosFlow->plmnInfo[plmnInfoIndex].plmn;

            if (plmn.s2.mcc == ptCi->ueInfo.servedPlmn.s2.mcc && plmn.s2.mnc == ptCi->ueInfo.servedPlmn.s2.mnc) break;
        }
    }

    else
    {
        mappingRule_type *ptrMappingRule = NULL;
        ptrMappingRule = &pGnb->cfg.qosDrbMappingConf.mappingRuleInfo[qosFlowGroupIdx];

        for (plmnInfoIndex = 0; plmnInfoIndex < MAX_PLMN_INFO_PER_DRB; plmnInfoIndex++)
        {
            if (ptrMappingRule->plmnInfo[plmnInfoIndex].status == N_EQUIP) continue;

            plmn = ptrMappingRule->plmnInfo[plmnInfoIndex].plmn;

            if (plmn.s2.mcc == ptCi->ueInfo.servedPlmn.s2.mcc && plmn.s2.mnc == ptCi->ueInfo.servedPlmn.s2.mnc) break;
        }
    }

    if (plmnInfoIndex < MAX_PLMN_INFO_PER_DRB)
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "VLIB_FindqosFlowGroupIdxForPlmn Supported Plmn Information found for the Qos Drb Mapping!\n");
        return GOOD;
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "VLIB_FindqosFlowGroupIdxForPlmn No supported Plmn Information found for the Qos Drb Mapping!\n");
        return ERROR;
    }
}
int16_t VLIB_AllocDrbByDedicatedQosFlowGroup_NoConfiguration_Available(
    callInfo_type *ptCi, uint8_t pduIdx, uint16_t qosFlowIdxInPdu)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(pduIdx, MAX_PDU_SESS_PER_CALL);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowIdxInPdu, MAX_QOS_FLOW_PER_PDU_SESS);

    callPDUSessInfo_type *ptPi = NULL;

    uint16_t drbIdxInCall;

    ptPi = &ptCi->pduSess[pduIdx];

    VCCB_CHECK_E_RANGE_AND_RETURN(ptPi->qosFlowIndex[qosFlowIdxInPdu], MAX_QOS_FLOW_PER_CALL);

    if (ERROR == VLIB_AllocNewCallDrbInfo(ptCi, pduIdx, qosFlowIdxInPdu, &drbIdxInCall))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocNewCallDrbInfo ERROR!\n");
        return ERROR;
    }
    return GOOD;
}

int16_t VLIB_AllocDrbByDedicatedQosFlowGroup(
    callInfo_type *ptCi, uint8_t pduIdx, uint16_t qosFlowIdxInPdu, uint8_t *isDedicatedAndMappingRuleListEmpty)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(isDedicatedAndMappingRuleListEmpty);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(pduIdx, MAX_PDU_SESS_PER_CALL);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowIdxInPdu, MAX_QOS_FLOW_PER_PDU_SESS);

    callPDUSessInfo_type *ptPi = NULL;
    qosFlowInfo_type *ptQi = NULL;

    uint8_t qosFlowFiveQi;
    uint8_t qosFlowPriorityLevel;
    uint8_t qosFlowGroupIdx;
    uint16_t drbIdxInCall;

    dedicatedQosFlowGroup_type *ptDedicatedQosFlowGroup = NULL;

    ptPi = &ptCi->pduSess[pduIdx];

    VCCB_CHECK_E_RANGE_AND_RETURN(ptPi->qosFlowIndex[qosFlowIdxInPdu], MAX_QOS_FLOW_PER_CALL);

    ptQi = &ptCi->qosFlow[ptPi->qosFlowIndex[qosFlowIdxInPdu]];

    VLIB_getFiveQi(ptQi, &qosFlowFiveQi);
    qosFlowPriorityLevel = ptQi->qosParam.allocAndRetent.priorityLevel;

    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    for (qosFlowGroupIdx = 0; qosFlowGroupIdx < MAX_DEDICATED_QOS_FLOW_GROUP_INDEX; qosFlowGroupIdx++)
    {
        ptDedicatedQosFlowGroup = &pGnb->cfg.qosDrbMappingConf.dedicatedQosFlowGroupInfo[qosFlowGroupIdx];

        if (ptDedicatedQosFlowGroup->status != EQUIP) continue;
        *isDedicatedAndMappingRuleListEmpty = FALSE;
        // Slice-based QoS flow to DRB mapping
        if (IsSupport_SliceMappingUsage(ptCi->gnbNum))
        {
            if (ERROR == VLIB_FindqosFlowGroupIdxForSlice(ptCi, ptPi, qosFlowGroupIdx, TRUE))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VLIB_FindqosFlowGroupIdxForSlice Slice info not matched!\n");
                continue;
            }
        }
        // Plmn-based QoS flow to DRB mapping
        if (IsSupport_PlmnMappingUsage(ptCi->gnbNum))
        {
            if (ERROR == VLIB_FindqosFlowGroupIdxForPlmn(ptCi, qosFlowGroupIdx, TRUE))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VLIB_FindqosFlowGroupIdxForPlmn Plmn info not matched!\n");
                continue;
            }
        }

        bool isArpMatched = (0 == ptDedicatedQosFlowGroup->qos5qi &&
                qosFlowPriorityLevel == ptDedicatedQosFlowGroup->arpPriorityLevel)? true:false;
        bool isQos5qiMatched = (0 == ptDedicatedQosFlowGroup->arpPriorityLevel && qosFlowFiveQi == ptDedicatedQosFlowGroup->qos5qi)?true:false;
        bool isQosArpNotConfigured = (0 == ptDedicatedQosFlowGroup->qos5qi && 0 == ptDedicatedQosFlowGroup->arpPriorityLevel)?true:false;
        bool isQosArpMatched = (qosFlowPriorityLevel == ptDedicatedQosFlowGroup->arpPriorityLevel &&
                qosFlowFiveQi == ptDedicatedQosFlowGroup->qos5qi)?true:false;

        const char* method = isArpMatched? "isArpMatched":(isQos5qiMatched? "isQos5qiMatched":(isQosArpNotConfigured?"isQosArpNotConfigured":(isQosArpMatched?"isQosArpMatched":"none")));

        if ((isArpMatched) || (isQos5qiMatched) || (isQosArpNotConfigured) || (isQosArpMatched))
        {
            if (ERROR == VLIB_AllocNewCallDrbInfo(ptCi, pduIdx, qosFlowIdxInPdu, &drbIdxInCall))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocNewCallDrbInfo ERROR!\n");
                return ERROR;
            }

            if(DBG->flag.logDrbPartial)
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "SDAP[%s]: pdu[%d]qosFlow[%d] is allocated to drb[%d]\n", method, ptCi->pduSess[pduIdx].pduSessId, ptPi->qosFlowIndex[qosFlowIdxInPdu], ptCi->drb[drbIdxInCall].dRBId);
            }
            return GOOD;
        }
    }

    return BREAK;
}

int16_t VLIB_AllocDrbByMappingRule(
    callInfo_type *ptCi, uint8_t pduIdx, uint16_t qosFlowIdxInPdu, uint8_t *isDedicatedAndMappingRuleListEmpty)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(isDedicatedAndMappingRuleListEmpty);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(pduIdx, MAX_PDU_SESS_PER_CALL);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowIdxInPdu, MAX_QOS_FLOW_PER_PDU_SESS);

    callPDUSessInfo_type *ptPi = NULL;
    qosFlowInfo_type *ptQi = NULL;
    callDRBInfo_type *ptDi = NULL;

    uint8_t qosFlowFiveQi;
    uint8_t qosFlowPriorityLevel;
    uint8_t mappingRuleIdx;
    uint16_t drbIdxInCall;
    uint8_t minQosFlowGroupId;

    mappingRule_type *ptMappingRule = NULL;

    ptPi = &ptCi->pduSess[pduIdx];

    VCCB_CHECK_E_RANGE_AND_RETURN(ptPi->qosFlowIndex[qosFlowIdxInPdu], MAX_QOS_FLOW_PER_CALL);

    ptQi = &ptCi->qosFlow[ptPi->qosFlowIndex[qosFlowIdxInPdu]];

    VLIB_getFiveQi(ptQi, &qosFlowFiveQi);
    qosFlowPriorityLevel = ptQi->qosParam.allocAndRetent.priorityLevel;
    minQosFlowGroupId = 0xFF;
    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    for (mappingRuleIdx = 0; mappingRuleIdx < MAX_MAPPING_RULE_INDEX; mappingRuleIdx++)
    {
        ptMappingRule = &pGnb->cfg.qosDrbMappingConf.mappingRuleInfo[mappingRuleIdx];

        if (ptMappingRule->status != EQUIP) continue;
        *isDedicatedAndMappingRuleListEmpty = FALSE;
        // Slice-based QoS flow to DRB mapping
        if (IsSupport_SliceMappingUsage(ptCi->gnbNum))
        {
            if (ERROR == VLIB_FindqosFlowGroupIdxForSlice(ptCi, ptPi, mappingRuleIdx, FALSE))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VLIB_FindqosFlowGroupIdxForSlice Slice info not matched!\n");
                continue;
            }
        }
        // Plmn-based QoS flow to DRB mapping
        if (IsSupport_PlmnMappingUsage(ptCi->gnbNum))
        {
            if (ERROR == VLIB_FindqosFlowGroupIdxForPlmn(ptCi, mappingRuleIdx, FALSE))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VLIB_FindqosFlowGroupIdxForPlmn Plmn info not matched!\n");
                continue;
            }
        }

        if ((0 == ptMappingRule->qos5qi && qosFlowPriorityLevel == ptMappingRule->arpPriorityLevel) ||
            (0 == ptMappingRule->arpPriorityLevel && qosFlowFiveQi == ptMappingRule->qos5qi) ||
            (0 == ptMappingRule->qos5qi && 0 == ptMappingRule->arpPriorityLevel) ||
            (qosFlowPriorityLevel == ptMappingRule->arpPriorityLevel && qosFlowFiveQi == ptMappingRule->qos5qi))
        {
            // find minimum qosFlowGroupId
            if (minQosFlowGroupId > ptMappingRule->qosFlowGroupId)
            {
                minQosFlowGroupId = ptMappingRule->qosFlowGroupId;
            }
        }
    }

#ifdef VCCB_IPC_TEST
    if (DBG->testPartialDrb)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Test PartialDBR (minQosFlowGroupId (%d->%d)\n", minQosFlowGroupId, qosFlowFiveQi);
        minQosFlowGroupId = qosFlowFiveQi;
    }
#endif

    if (minQosFlowGroupId != 0xFF)
    {
        // if mapping DRB is not exist, allocate new DRB
        if (ptPi->mappingRuleDrbIdx[minQosFlowGroupId] >= MAX_DRB_PER_SA_CALL)
        {
            if (ERROR == VLIB_AllocNewCallDrbInfo(ptCi, pduIdx, qosFlowIdxInPdu, &drbIdxInCall))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocNewCallDrbInfo ERROR!\n");
                return ERROR;
            }
            ptPi->mappingRuleDrbIdx[minQosFlowGroupId] = drbIdxInCall;

            return GOOD;
        }
        // mapping DRB is already exist
        else
        {
            bool abnormalRecovery = false, nrdcSpecific = false;

            drbIdxInCall = ptPi->mappingRuleDrbIdx[minQosFlowGroupId];

            // recovery against abnormal condition.
            if (drbIdxInCall >= MAX_DRB_PER_SA_CALL || ptCi->drb[drbIdxInCall].usage != CI_use)
            {
                VCCB_PRINT_CALL_COMMENT(ptCi,
                    "Alloc new DRB due to Invalid [drbIdxInCall(%u) or drb usage is not CI_use] in %s!\n", drbIdxInCall,
                    __func__);
                abnormalRecovery = true;
            }
            else if (IS_NRDC_MN(ptCi) && ptCi->drb[drbIdxInCall].selectedBearerTypeFrozenFlag)
            {
                VCCB_PRINT_CALL_COMMENT(ptCi,
                    "Alloc new DRB due to drb's selectedBearerTypeFrozenFlag(%u) for NRDC in %s!\n",
                    ptCi->drb[drbIdxInCall].selectedBearerTypeFrozenFlag, __func__);
                nrdcSpecific = true;
            }

            if (abnormalRecovery || nrdcSpecific)
            {
                if (ERROR == VLIB_AllocNewCallDrbInfo(ptCi, pduIdx, qosFlowIdxInPdu, &drbIdxInCall))
                {
                    VCCB_PRINT_CALL_ERROR(
                        ptCi, "VLIB_AllocNewCallDrbInfo during recovery against abnormal condition, ERROR!\n");
                    return ERROR;
                }

                if (abnormalRecovery)
                {
                    ptPi->mappingRuleDrbIdx[minQosFlowGroupId] = drbIdxInCall;
                }
                else
                {
                    VCCB_CHECK_E_RANGE_AND_RETURN(drbIdxInCall, MAX_DRB_PER_SA_CALL);
                    ptCi->drb[drbIdxInCall].mappingType = MAPPING_DedicatedGroup;
                    VCCB_PRINT_CALL_MSG(ptCi,
                        "QoS flow[%d] is allocated to new drbIndex[%d] as dedicatedGroup due to NRDC in %s\n",
                        ptPi->qosFlowIndex[qosFlowIdxInPdu], drbIdxInCall, __FUNCTION__);
                }
                return GOOD;
            }

            ptDi = &ptCi->drb[drbIdxInCall];
            // release when number of QosFlow in ptDi reaches the max value
            if (ptDi->numQoSFlowPerDRB >= MAX_QOS_FLOW_PER_DRB)
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "%s Number of QoS flow[%d] in DRB[%d] reaches the max value ERROR!\n",
                    __func__, ptDi->numQoSFlowPerDRB, drbIdxInCall);
                return ERROR;
            }
            // if number of QosFlow in ptDi is valid, add QosFlow in that DRB
            else
            {
                uint8_t drbQosIdx = 0;

                for (drbQosIdx = 0; drbQosIdx < MAX_QOS_FLOW_PER_DRB; drbQosIdx++)
                {
                    if (ptDi->qosFlowIndex[drbQosIdx] < MAX_QOS_FLOW_PER_CALL) continue; // already allocated

                    ptDi->qosFlowIndex[drbQosIdx] = ptPi->qosFlowIndex[qosFlowIdxInPdu];
                    ptDi->numQoSFlowPerDRB++;
                    ptQi->drbIdxInCall = drbIdxInCall;

                    if (ptDi->drbStatus == S_Idle)
                    {
                        SET_DRB_STATUS(ptCi, drbIdxInCall, S_Modify);
                    }  // if drbStatus is S_Setup, remain status

                    break;
                }

                if (drbQosIdx >= MAX_QOS_FLOW_PER_DRB)
                {
                    VCCB_PRINT_CALL_ERROR(ptCi, "%s All of qosFlowIndex in DRB[%d] is already allocated !\n",
                        __func__, drbIdxInCall);
                    return ERROR;
                }

                return GOOD;
            }
        }
    }

    return BREAK;
}

int16_t VLIB_AllocDrbByDefault(callInfo_type *ptCi, uint8_t pduIdx, uint16_t qosFlowIdxInPdu)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(pduIdx, MAX_PDU_SESS_PER_CALL);
    VCCB_CHECK_E_RANGE_AND_RETURN(qosFlowIdxInPdu, MAX_QOS_FLOW_PER_PDU_SESS);

    callPDUSessInfo_type *ptPi = NULL;
    qosFlowInfo_type *ptQi = NULL;
    callDRBInfo_type *ptDi = NULL;

    uint16_t drbIdxInCall;

    ptPi = &ptCi->pduSess[pduIdx];

    VCCB_CHECK_E_RANGE_AND_RETURN(ptPi->qosFlowIndex[qosFlowIdxInPdu], MAX_QOS_FLOW_PER_CALL);

    ptQi = &ptCi->qosFlow[ptPi->qosFlowIndex[qosFlowIdxInPdu]];

    // if default DRB is not exist, allocate new DRB
    if (ptPi->defaultGroupPresent != OPT_PRESENT)
    {
        if (ERROR == VLIB_AllocNewCallDrbInfo(ptCi, pduIdx, qosFlowIdxInPdu, &drbIdxInCall))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocNewCallDrbInfo ERROR!\n");
            return ERROR;
        }
        ptPi->defaultGroupPresent = OPT_PRESENT;
        ptPi->mappingDrbIdxDefaultGroup = drbIdxInCall;
    }

    // if default DRB is already exist
    else
    {
        drbIdxInCall = ptPi->mappingDrbIdxDefaultGroup;
        ptDi = &ptCi->drb[drbIdxInCall];

        // release when number of QosFlow in ptDi reaches the max value
        if (ptDi->numQoSFlowPerDRB >= MAX_QOS_FLOW_PER_DRB)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "%s Number of QoS flow[%d] in DRB[%d] reaches the max value ERROR!\n", __func__,
                ptDi->numQoSFlowPerDRB, drbIdxInCall);
            return ERROR;
        }

        // if number of QosFlow in ptDi is valid, just add QosFlow in that DRB
        else
        {
            uint8_t drbQosIdx = 0;

            for (drbQosIdx = 0; drbQosIdx < MAX_QOS_FLOW_PER_DRB; drbQosIdx++)
            {
                if (ptDi->qosFlowIndex[drbQosIdx] < MAX_QOS_FLOW_PER_CALL) continue; // already allocated

                ptDi->qosFlowIndex[drbQosIdx] = ptPi->qosFlowIndex[qosFlowIdxInPdu];
                ptDi->numQoSFlowPerDRB++;
                ptQi->drbIdxInCall = drbIdxInCall;

                if (ptDi->drbStatus == S_Idle)
                {
                    SET_DRB_STATUS(ptCi, drbIdxInCall, S_Modify);
                }  // if drbStatus is S_Setup, remain status

                break;
            }

            if (drbQosIdx >= MAX_QOS_FLOW_PER_DRB)
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "%s All of qosFlowIndex in DRB[%d] is already allocated !\n", __func__,
                    drbIdxInCall);
                return ERROR;
            }
        }
    }
    return GOOD;
}

/**
 * @brief     Allocate QoS Flow in PDU info
 * @param     ptCi : call context
 * @param     ptAllocatedQosIdx : QoS Flow index to allocate / uint16_t
 * @return    GOOD : success
 * @return    ERROR : failure
 */
int16_t VLIB_SelectRepresentativeQoSFlowInDRB(callInfo_type *ptCi, callDRBInfo_type *ptDi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptDi);

    uint16_t selectedQosFlowInCall = 0xFFFF;
    uint16_t selectedArpPriority = 0xFFFF;
    uint16_t selectedQoSPriority = 0xFFFF;
    uint8_t selected5qi = 0xFF;
    uint8_t index;

    uint16_t tempSelectedQoSPriority;
    uint8_t tempSelected5qi;

    qosFlowInfo_type *ptQi = NULL;

    if (ptDi->numQoSFlowPerDRB == 1)  // single QoS flow exist in DRB
    {
        for (index = 0; index < MAX_QOS_FLOW_PER_DRB; index++)
        {
            if (ptDi->qosFlowIndex[index] >= MAX_QOS_FLOW_PER_CALL) continue;
            if (!IS_VALID_NODE(QOSFLOW_NODE, ptCi, &ptCi->qosFlow[ptDi->qosFlowIndex[index]])) continue;

            ptQi = &ptCi->qosFlow[ptDi->qosFlowIndex[index]];

            if (VLIB_IsCommonValidStateOfPduQosDrb(ptQi->qosFlowStatus) != GOOD) continue;
            if ((ptQi->qosParam.reflectiveQosAttributePresent == OPT_PRESENT) &&
                (ptQi->qosParam.reflectiveQosAttribute == subject_to_reflective_QoS))
            {
                ptDi->drbReflectiveQosFlag = VCCB_FLAG_ON;
            }
            selectedQosFlowInCall = ptDi->qosFlowIndex[index];
            break;
        }
    }
    else  // multiple QoS flow exist in DRB
    {
        for (index = 0; index < MAX_QOS_FLOW_PER_DRB; index++)
        {
            if (ptDi->qosFlowIndex[index] >= MAX_QOS_FLOW_PER_CALL) continue;
            if (!IS_VALID_NODE(QOSFLOW_NODE, ptCi, &ptCi->qosFlow[ptDi->qosFlowIndex[index]])) continue;

            ptQi = &ptCi->qosFlow[ptDi->qosFlowIndex[index]];

            if (VLIB_IsCommonValidStateOfPduQosDrb(ptQi->qosFlowStatus) != GOOD) continue;

            if ((ptQi->qosParam.reflectiveQosAttributePresent == OPT_PRESENT) &&
                (ptQi->qosParam.reflectiveQosAttribute == subject_to_reflective_QoS))
            {
                ptDi->drbReflectiveQosFlag = VCCB_FLAG_ON;
            }

            if (selectedQosFlowInCall == 0xFFFF)
            {
                // Save selected Values (CallQoSflowIndex, Arp/QoSPriority, 5qi)
                selectedQosFlowInCall = ptDi->qosFlowIndex[index];
                selectedArpPriority = ptQi->qosParam.allocAndRetent.priorityLevel;
                VLIB_getPriorityLevelQos(ptQi, &selectedQoSPriority);
                VLIB_getFiveQi(ptQi, &selected5qi);
            }
            else
            {
                if (selectedArpPriority > ptQi->qosParam.allocAndRetent.priorityLevel)
                {
                    // Save selected Values (CallQoSflowIndex, Arp/QoSPriority, 5qi)
                    selectedQosFlowInCall = ptDi->qosFlowIndex[index];
                    selectedArpPriority = ptQi->qosParam.allocAndRetent.priorityLevel;
                    VLIB_getPriorityLevelQos(ptQi, &selectedQoSPriority);
                    VLIB_getFiveQi(ptQi, &selected5qi);
                }
                else if (selectedArpPriority == ptQi->qosParam.allocAndRetent.priorityLevel)
                {
                    VLIB_getPriorityLevelQos(ptQi, &tempSelectedQoSPriority);
                    if (selectedQoSPriority > tempSelectedQoSPriority)
                    {
                        // Save selected Values (CallQoSflowIndex, Arp/QoSPriority, 5qi)
                        selectedQosFlowInCall = ptDi->qosFlowIndex[index];
                        selectedArpPriority = ptQi->qosParam.allocAndRetent.priorityLevel;
                        VLIB_getPriorityLevelQos(ptQi, &selectedQoSPriority);
                        VLIB_getFiveQi(ptQi, &selected5qi);
                    }
                    else if (selectedQoSPriority == tempSelectedQoSPriority)
                    {
                        VLIB_getFiveQi(ptQi, &tempSelected5qi);
                        if (selected5qi > tempSelected5qi)
                        {
                            // Save selected Values (CallQoSflowIndex, Arp/QoSPriority, 5qi)
                            selectedQosFlowInCall = ptDi->qosFlowIndex[index];
                            selectedArpPriority = ptQi->qosParam.allocAndRetent.priorityLevel;
                            VLIB_getPriorityLevelQos(ptQi, &selectedQoSPriority);
                            VLIB_getFiveQi(ptQi, &selected5qi);
                        }
                    }
                }
            }
        }
    }

    ptDi->drbQosIdxInCall = selectedQosFlowInCall;
    return GOOD;
}

int FnVCCB_CollectUpdatedQoSFlowDRBIndex_ForSN(callInfo_type const &rCall, std::set<uint16_t> &updatedDrbIdxSet)
{
    const auto isQosFlowUpdated = [](auto const &rQosFlow) -> bool
    {
        return (rQosFlow.qosFlowStatus == S_Setup || rQosFlow.qosFlowStatus == S_Modify ||
                rQosFlow.qosFlowStatus == S_Release);
    };

    for (auto const &rQosFlow : rCall.qosFlow)
    {
        if (rQosFlow.usage != CI_use) continue;

        if (!isQosFlowUpdated(rQosFlow)) continue;

        uint16_t drbIdxInCall = rQosFlow.drbIdxInCall;
        if (drbIdxInCall >= MAX_DRB_PER_SA_CALL) continue;

        if (rCall.drb[drbIdxInCall].drbStatus == S_Release) continue;

        updatedDrbIdxSet.insert(drbIdxInCall);
    }

    if (updatedDrbIdxSet.size() == 0)
    {
        return BREAK;
    }
    return GOOD;
}

int FnVCCB_SelectRepresentativeQosFlowOfDrb_ForSN(callInfo_type &rCall, std::set<uint16_t> const &updatedDrbIdxSet)
{
    for (auto const &drbIdx : updatedDrbIdxSet)
    {
        VCCB_CHECK_E_RANGE_AND_CONTINUE(drbIdx, MAX_DRB_PER_SA_CALL);
        auto &drb = rCall.drb[drbIdx];

        VCCB_RETURN_ON_CALL_ERROR(&rCall, VLIB_SelectRepresentativeQoSFlowInDRB(&rCall, &drb),
            "DRB[%d] VLIB_SelectRepresentativeQoSFlowInDRB ERROR!!\n", drbIdx);

        if (drb.drbQosIdxInCall >= MAX_QOS_FLOW_PER_CALL)
        {
            VCCB_PRINT_CALL_NRDC_BASIC(&rCall, "Failed to select representative qosFlow in DRB[%d]!! (drbQosIdxInCall:%d)\n",
                drbIdx, drb.drbQosIdxInCall);
        }
    }
    return GOOD;
}

int FnVCCB_UpdateRepresentativeQosFlowForAddModDrb_ForSN(callInfo_type &rCall)
{
    if (!IS_NRDC_SN(&rCall)) return GOOD;

    std::set<uint16_t> updatedDrbIdxSet;

    if (FnVCCB_CollectUpdatedQoSFlowDRBIndex_ForSN(rCall, updatedDrbIdxSet) == BREAK)
    {
        // there is no updated qosflow
        return GOOD;
    }

    if (FnVCCB_SelectRepresentativeQosFlowOfDrb_ForSN(rCall, updatedDrbIdxSet) == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(&rCall, "FnVCCB_SelectRepresentativeQosFlowForDrbSet ERROR!!\n");
        return ERROR;
    }

    return GOOD;
}

void FnVCCB_Decide_QosMonitoringInfoForAllDrbs(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    for(auto &rDrb : ptCi->drb)
    {
        if( !rDrb.usage ) continue;
        if (VLIB_IsCommonValidStateOfPduQosDrb(rDrb.drbStatus) == ERROR) continue;

        if (FnVCCB_Decide_QosMonitoringInfo_ForDrbQos(ptCi, &rDrb, &rDrb.drbQosMonitoring) == ERROR)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_Decide_QosMonitoringInfo_ForDrbQos ERROR\n");
        }
    }
}

/**
 * @brief     Initialize AUPF selection Mode from confD
 * @param     ptCi : call context
 */
enumAUPFSelectionMode_type VLIB_GetAupfSelectionModeFromGnb(const callInfo_type *ptCi,
    const VccbGnbInfo_type &gnbInfo)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_RET_VAL(ptCi, AUPFSelectionMode_ENDCCapacityBased);
    switch (ptCi->callType)
    {
        case E_CALLTYPE_SA:
            if (IS_NRDC_SN(ptCi) && IS_SELECTED_MRDC(MRDC_NRDC))
                return gnbInfo.aupf.aupfSelectionModeNRDCSN;
            else
                return gnbInfo.aupf.aupfSelectionModeSA;
        case E_CALLTYPE_NSA:
            return gnbInfo.aupf.aupfSelectionModeENDC;
        default:
            return gnbInfo.aupf.aupfSelectionModeENDC;
    }
}

void VLIB_InitExtHoInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    memset(&ptCi->extDb.m, 0, sizeof(ptCi->extDb.m));
}

void VLIB_ResetScgFailAttSuccStats(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->scgFailAttSuccStats.onOff  = VCCB_FLAG_OFF;
    ptCi->scgFailAttSuccStats.cause  = ScgFailCauseInfo_NoScgFailure;
    ptCi->scgFailAttSuccStats.cellId = MAX_CELL_IDENTITY;
}

/**
 * @brief      This function is used to initialize needForGapNrUeData
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitNeedForGapNrUeData(needForGapNrUeData_type *pNeedForGapNrUeData)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(pNeedForGapNrUeData);

    pNeedForGapNrUeData->configInstalledFlag = FLAG_OFF;
    pNeedForGapNrUeData->configNewFlag = FLAG_OFF;
    pNeedForGapNrUeData->forceReconfig = FLAG_OFF;
    pNeedForGapNrUeData->f1CtxModFlag = FLAG_OFF;
    pNeedForGapNrUeData->intraFreqServCellIdBitmask = 0;
    pNeedForGapNrUeData->intraFreqGapIndBitmask = 0;
    memset(pNeedForGapNrUeData->bandFilterNew, 0, sizeof(pNeedForGapNrUeData->bandFilterNew));
    memset(pNeedForGapNrUeData->bandFilterInstalled, 0, sizeof(pNeedForGapNrUeData->bandFilterInstalled));
}

void FnVCCB_InitSACallInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t pduIdx, drbIdx, qosFlowIdx, loop;

    ptCi->ueInfo.ambr.UECtxModAMBRPresent = FALSE;

    for (pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
    {
        ptCi->pduSess[pduIdx].usage = CI_no_use;
        for (loop = 0; loop < MAX_VCCB_PDU_STAT_FAMILY_TYPE; loop++)
        {
            ptCi->pduSess[pduIdx].pduStatistic.attemptFlag[loop] = NO_ATTEMPT;
        }
        ptCi->pduSess[pduIdx].pduSessAmbrPresent = OPT_NOT_PRESENT;
        ptCi->pduSess[pduIdx].pduSessAmbrDL = MAX_EXTENDED_BITRATE;
        ptCi->pduSess[pduIdx].pduSessAmbrUL = MAX_EXTENDED_BITRATE;

        ptCi->pduSess[pduIdx].nonGBRResourcesOffered = OPT_NOT_PRESENT;
    }

    ptCi->drbIdPool = 0;
    for (drbIdx = 0; drbIdx < MAX_DRB_PER_SA_CALL; drbIdx++)
    {
        ptCi->drb[drbIdx].usage = CI_no_use;
        ptCi->drb[drbIdx].drbConnStat.UlRoHCdrbCountFlagRrcConn = VCCB_FLAG_OFF;
        ptCi->drb[drbIdx].drbConnStat.SymmRoHCdrbCountFlagRrcConn = VCCB_FLAG_OFF;
        ptCi->drb[drbIdx].drbConnStat.EHCdrbCountFlagRrcConn = VCCB_FLAG_OFF;
        ptCi->drb[drbIdx].drbConnStat.drbCountFlagOfRedCapRrcConn = VCCB_FLAG_OFF;
        ptCi->drb[drbIdx].drbConnStat.drbCountFlagOfRedCapRrcInact = VCCB_FLAG_OFF;
        ptCi->drb[drbIdx].vomaStatus.upStatus = VOMA_UNKNOWN;
        ptCi->drb[drbIdx].vomaStatus.duStatus = VOMA_UNKNOWN;
        for (loop = 0; loop < MAX_VCCB_DRB_STAT_FAMILY_TYPE; loop++)
        {
            ptCi->drb[drbIdx].drbStatistic.attemptFlag[loop] = NO_ATTEMPT;
        }
        ptCi->drb[drbIdx].activityStatus = ActivityStatus_ACTIVE;
    }

    for (qosFlowIdx = 0; qosFlowIdx < MAX_QOS_FLOW_PER_CALL; qosFlowIdx++)
    {
        ptCi->qosFlow[qosFlowIdx].usage = CI_no_use;
        ptCi->qosFlow[qosFlowIdx].isSDAPatSN = VCCB_FLAG_OFF;
        for (loop = 0; loop < MAX_VCCB_QOS_FLOW_STAT_FAMILY_TYPE; loop++)
        {
            ptCi->qosFlow[qosFlowIdx].qosFlowStatistic.attemptFlag[loop] = NO_ATTEMPT;
        }
    }
}
void FnVCCB_InitCslInfo_MobilityStat(callInfo_type &rCall)
{
    memset(&rCall.cslInfo.cslMobilityInfo, 0, sizeof(rCall.cslInfo.cslMobilityInfo));
}

void FnVCCB_InitCslInfo_HoStat(callInfo_type &rCall)
{
    memset(&rCall.cslInfo.hoInfo, 0, sizeof(rCall.cslInfo.hoInfo)); // SVR25B FGR-MC1009 FR610
}

void FnVCCB_InitCslInfo_MobilityNsaStat(callInfo_type &rCall)
{
    memset(&rCall.cslInfo.cslMobilityInfo_nsa, 0, sizeof(rCall.cslInfo.cslMobilityInfo_nsa));
}

/**
 * @brief           This function is used to initialize QciAttemptSuccessInfo
 * @param[in/out]   ptCi : call context
 * @return          void
 */
void FnVCCB_InitQciAttemptSuccessInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    for (uint8_t loop1 = 0; loop1 < MAX_QCI_HISTORY_STORING_CAPACITY; ++loop1)
    {
        ptCi->qciCount.QciAttemptSuccessData[loop1].qci = MAX_QCI_VALUE;
        ptCi->qciCount.QciAttemptSuccessData[loop1].attempt = 0;
        ptCi->qciCount.QciAttemptSuccessData[loop1].success = 0;
    }
}

/**
 * @brief           This function is used to initialize LatestReleasedQci
 * @param[in/out]   ptCi : call context
 * @return          void
 */
void FnVCCB_InitLatestReleasedQci(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    for (uint8_t loop = 0; loop < MAX_QCI_HISTORY_STORING_CAPACITY; ++loop)
    {
        ptCi->qciCount.LatestReleasedQci[loop].qci = MAX_QCI_VALUE;
        ptCi->qciCount.LatestReleasedQci[loop].gNBInitiatedErabRelease = 0;
        ptCi->qciCount.LatestReleasedQci[loop].gNBInitiatedAbnormalErabRelease = 0;
        ptCi->qciCount.LatestReleasedQci[loop].gNBInitiatedAbnormalErabReleaseWithActiveBearer = 0;
        ptCi->qciCount.LatestReleasedQci[loop].MeNBInitiatedErabRelease = 0;
        ptCi->qciCount.LatestReleasedQci[loop].MeNBInitiatedAbnormalErabRelease = 0;
        ptCi->qciCount.LatestReleasedQci[loop].MeNBInitiatedAbnormalErabReleaseWithActiveBearer = 0;
    }
}

/**
 * @brief This function initializes SCellCSL Flag
 * 
 * @param ptCi : call context
 */
void FnVCCB_InitCslInfo_CaScellFlag(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->sCellCslRelCauseByCuTrigger = UNDEFINED_CA_CSL_CAUSE;
    ptCi->bitFlag.skipCslCheckBeforeRrcReconfig = VCCB_FLAG_OFF;
}

/**
 * @brief This function initializes SCellCSL List
 * 
 * @param ptCi : call context
 */
void FnVCCB_InitCslInfo_CaScellList(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    
    memset(ptCi->cslInfo.sCellConfigInfo, 0, sizeof(ptCi->cslInfo.sCellConfigInfo));
    memset(ptCi->sCellCsl, 0, sizeof(ptCi->sCellCsl));
}

/**
 * @brief This function initializes SCellCSL information
 * 
 * @param ptCi : call context
 */
void FnVCCB_InitCslInfo_CaScellInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    FnVCCB_InitCslInfo_CaScellFlag(ptCi);
    FnVCCB_InitCslInfo_CaScellList(ptCi);
}

/**
 * @brief Initialize SA/NSA measurement data.
 * 
 * @param rCall call info
 */
void VCCB_InitMeasData(callInfo_type &rCall)
{
    AUTO_TRACE_FUNC;
        
    VCCM_InitNsaMeasData(&rCall);
    VCCM_InitPrevNsaMeasData(&rCall);
    VCCM_InitSaMeasData(&rCall);
    VCCM_InitPrevSaMeasData(&rCall);
}

/**
 * @brief Initialize refusedTargetGnbId structure in call info.
 * 
 * @param rCi call info
 */
void VCCB_InitRefusedTargetGnbId(callInfo_type &rCi)
{
    AUTO_TRACE_FUNC;

    for (int i = 0; i < MAX_REFUSED_TARGET_GNB_PER_CALL; ++i)
    {
        rCi.refusedTargetGnbId[i].gnbId.id       = 0;
        rCi.refusedTargetGnbId[i].gnbId.idLength = 0;
    }
}

/**
 * @brief     Initialize call context
 * @param     ptCi : call context
 */
void VLIB_InitCallInfo(callInfo_type *ptCi, uint8_t threadId, uint16_t gnbNum, enumCallType_type callType)
{
    AUTO_TRACE_FUNC;
    int32_t loop1, loop2;

    VALID_E_POINT_CHECK_VOID(ptCi);

    uint32_t callId = ptCi->callId;
    memset(reinterpret_cast<uint8_t *>(ptCi), 0xFF, sizeof(callInfo_type));

    // [claude] Zero-initialize the per-UE AI context immediately after the 0xFF memset.
    // [claude] Without this, aiCtx.h[]/resumeSuccessRate/avgDurationSec become NaN
    // [claude] (0xFFFFFFFF reinterpreted as float) and count/writeIdx become -1. That
    // [claude] makes every inference fail the isfinite() check (-> permanent Fallback,
    // [claude] AI never applies a timer) and corrupts the history circular buffer
    // [claude] (out-of-bounds write at history[-1]). VccbAi_InitContext() memset(0)s
    // [claude] aiCtx and seeds resumeSuccessRate=1.0 (cold start).
    VccbAi_InitContext(&ptCi->aiCtx);

    ptCi->gnbCuUeId = UNKNOWN_CP_CALLID;
    ptCi->callId = callId;

    VCCB_CHECK_E_RANGE_AND_VOID(gnbNum, MAX_GNB_PER_SYS);
    ptCi->gnbNum = gnbNum;

    VccbGnbInfo_type *const pGnb = &GNBINFO[gnbNum];
    ptCi->gnbId = pGnb->globalGnbId.gnbId.id;

    ptCi->servCellNum = 0xFFFF;
    ptCi->servCellBand = BAND_NOT_FOUND;
    ptCi->rrcConnStat.callCountFlag = FLAG_OFF;
    ptCi->rrcConnStat.rrcConnectionCountFlag = FLAG_OFF;
    ptCi->rrcConnStat.rrcConnectionPowerClassCountFlag = FLAG_OFF;
    ptCi->rrcConnStat.rrcInactiveCountFlag = FLAG_OFF;
    ptCi->rrcConnStat.vonrEmerCallFlag            = FLAG_OFF;
    ptCi->rrcConnStat.vonrEmerCountFlagOfRrcConn  = FLAG_OFF;
    ptCi->rrcConnStat.vonrEmerCountFlagOfRrcInact = FLAG_OFF;
    ptCi->callType = callType;
    ptCi->NrPpaContext.isNrTARequested = FALSE;
    ptCi->NrPpaContext.isTimAdvReceived = FALSE;
    ptCi->nrDlTdoaContext.locationMeasIndStatus = lmi_off;
    ptCi->NrPpaContext.valueTimAdvNR = 0xFFFF;

    VLIB_InitExtHoInfo(ptCi);

    // Call Backup
    ptCi->backupSeqNum_First = 0;
    ptCi->backupSeqNum_last = 0;
    ptCi->needBackupAfterStateAll = VCCB_FLAG_ON;
    ptCi->intraBandENDC_support = VCCB_FLAG_OFF;  // FGR-CC5501
    ptCi->cgConfigInfo.configRestrictInfo.maxROHCsessionsSN_NR_Flag = FLAG_OFF;
    ptCi->cgConfigInfo.configRestrictInfo.pdcchBlindDetectionSCG_Flag = FLAG_OFF;

    ptCi->isULOnlyRoHCCall = FLAG_OFF;

    // BasicOperation for NR SA FR1120 : Change initial value from NULL(0xFF) to ci_relocatedCall(10) to handle
    // exceptional cases case1. Normal Case          - Refer to 'establishmentCause' included in the RRCSetupRequest.
    // case2. Inter-gNB H/O or RRE - Set to 'ci_relocatedCall'(initial value) since it does not have establishmentCause
    // in RRCSetupRequest,
    ptCi->ueInfo.establishmentCause = ci_relocatedCall;
    ptCi->aupfSelectionMode = VLIB_GetAupfSelectionModeFromGnb(ptCi, *pGnb);
    ptCi->endPointInfo.srvAupfNum = 0;

    for (auto &aupf : ptCi->endPointInfo.aupf)
    {
        FeatAupf::initCallAupf(aupf);
    }

    VCCB_CLEAR_TIMER(&ptCi->timer, INIT_TIMER);
    FnVCCB_DetachCallMultiTimer(ptCi);
    VCCB_CLEAR_TIMER(&ptCi->mgapReTriggerTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->forcedReleaseTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->scgSuspendResumeTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->secuModeCmpTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->dlNasWaitingAckTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->endcFr2MeasTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->lbMrTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->saPriorHOTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->finalizeInterMnHo.timer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->saHoMrQueueInfo.t4, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->saHoMrQueueInfo.t5, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->forcedHO.timer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->longCallDetectionTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->rlfHandlingTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->iratLbMrWaitTimer, INIT_TIMER); // SVR23B FGR-LC0711
    VCCB_CLEAR_TIMER(&ptCi->voNrLbMrWaitTimer, INIT_TIMER); // SVR24B FGR-LC0712

    ptCi->periodicLbGuardTimeStamp.tv_sec = 0;
    ptCi->periodicLbGuardTimeStamp.tv_nsec = 0;
#ifdef __CNF__
    VCCB_CLEAR_TIMER(&ptCi->redisExpireUpdateTimer, INIT_TIMER);
#endif

    ptCi->finalizeInterMnHo.isFinalizeActive = VCCB_FLAG_OFF;

    ptCi->transactionId = (((MAX_SRB_PER_CALL + 1) << SRB_SHIFT) & SRB_MASK) +
                          (((MAX_ERAB_PER_CALL + 1) << ERAB_SHIFT) & ERAB_MASK) +
                          (((MAX_GNB_PER_CALL + 1) << GNB_SHIFT) & GNB_MASK);

    // Do not use 'SET_STATE_CMD' and 'MACRO_CHG_CALL_STATE'. 0xFFFF value is used as prevValue.
    ptCi->stateCmd = Idle_cmd;
    ptCi->prevStateCmd = Idle_cmd;
    ptCi->state = STATE_II;
    ptCi->prevFnState = STATE_II;

    ptCi->srcCuCpUeIdInterRATHo = INVALID_SRC_CALLID_FOR_IRAT_HO;
    ptCi->stateInitTime = time(NULL);
    SET_RRC_STATE(ptCi, RRC_IDLE);
    SET_PROC_FOR_STAT(ptCi, MAX_PROC_FOR_STAT);
    ptCi->traceVoNR = FALSE;

    VCCB_CLEAR_TIMER(&ptCi->CallUserInactivity.sigInactivityTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->CallUserInactivity.releaseTimer, INIT_TIMER);
    ptCi->activityStatus.ue = ActivityStatus_INACTIVE;
    ptCi->activityStatus.ueReported = ActivityStatus_ACTIVE;

    ptCi->CallUserInactivity.ueInactivityState = UserInactivityState_ACTIVE;
    if (callType == E_CALLTYPE_NSA)
    {
        ptCi->CallUserInactivity.activityNotificationLevel = pGnb->UserInactivity.activityNotificationLevel;

        ptCi->CallUserInactivity.featureEnabled = pGnb->UserInactivity.featureEnabled;  // 22B_UserInactivity_FR214
    }
    else  // SA_CALL
    {
        ptCi->CallUserInactivity.activityNotificationLevel = pGnb->UserInactivitySA.activityNotificationLevel;
        ptCi->CallUserInactivity.featureEnabled = pGnb->UserInactivitySA.featureEnabled;
        ptCi->CallUserInactivity.dataInactivityPerCellOnOff = pGnb->UserInactivitySA.dataInactivityPerCellOnOff;
    }
    ptCi->CallUserInactivity.desiredNotiLevelChanged = FALSE;
    ptCi->CallUserInactivity.isReleaseTimerTimeout = FALSE;
    ptCi->CallUserInactivity.nrdcSN_UELevel_InactivityStatus = ActivityStatus_ACTIVE;
    ptCi->CallUserInactivity.ueLevelActivityReportSentToMN = ActivityStatus_ACTIVE;

    ptCi->CallUserInactivity.rfspBasedControlFlag = VCCB_FLAG_OFF;
    ptCi->CallUserInactivity.rfspVal = 0;

    ptCi->dUinactivityMonitoring = 1;

    ptCi->isEndcVipUe = FALSE;
    ptCi->synchInd = 1;  // 090629 YS.CHO :: Default InSync
    ptCi->maxRetransInd = 0;

    ptCi->debugTrace = VCCB_FLAG_OFF;

    memset(&ptCi->flag, 0, sizeof(ptCi->flag));
    memset(&ptCi->bitFlag, 0, sizeof(ptCi->bitFlag));

    if (pGnb->cfg.allowCallProcessingDuringRlfDetected == VCCB_FLAG_OFF) // KT
    {
        ptCi->bitFlag.usingRlfRreWaitState = VCCB_FLAG_ON; // Use RRE Wait State
    }
    else
    {
        ptCi->bitFlag.usingRlfRreWaitState = VCCB_FLAG_OFF; // Not use RRE Wait State
    }

    ptCi->bitFlag.mnBcReselectionFeatureOn = FALSE;
    if (GNBINFO[ptCi->gnbNum].cfg.nrdc.mnBcReselection == VCCB_FLAG_ON)
    {
        ptCi->bitFlag.mnBcReselectionFeatureOn = TRUE;
    }

    VCCB_INIT_UE_DRB_PARTIAL_INFO(ptCi);

    memset(&ptCi->dlNasAckInfo, 0, sizeof(ptCi->dlNasAckInfo));
    ptCi->dlNasAckInfo.usingDlNasAck = pGnb->cfg.f1RrcDeliveryReportOnOff & e_BitMask_1bit;
    ptCi->dlNasAckInfo.savedSendingDlNasPdcpSn = 0xFFFFFFFF;

    MACRO_RESET_CALL_FAULT_CAUSES;
    SET_CALL_RETAIN_STAT_CAPABLE_FLAG(ptCi, VCCB_FLAG_OFF);
    CLEAR_RETAIN_STAT_EVENT(ptCi->retainStat);

    SET_VONR_EMER_RETAIN_STAT_CAPABLE_FLAG(ptCi, VCCB_FLAG_OFF);
    CLEAR_RETAIN_STAT_EVENT(ptCi->retainStatOfVonrEmer);

    // nasInfo_type
    INIT_NAS_INFORMATION(ptCi->nasInfo);
    INIT_NAS_INFORMATION(ptCi->nasTransportInfo);

    // ueInfo_type
    ptCi->ueInfo.receivedUeCapaFlag = FALSE;

    ptCi->ueInfo.ueId.usage = CI_no_use;
    ptCi->ueInfo.ueId.ueIdType = ciUNKNOWN;
    ptCi->ueInfo.ueId.sTmsi.amfSetPointer = 0xFFFF;
    ptCi->ueInfo.ueId.sTmsi.fiveGTmsi.n = 0xFFFFFFFF;
    for (loop1 = 0; loop1 < MAX_5G_TMSI_LEN; loop1++) ptCi->ueInfo.ueId.sTmsi.fiveGTmsi.data[loop1] = 0xFF;

    /* IMSI Trace */
    memset(&ptCi->ueInfo.ueId.imsi, 0x00, sizeof(IMSI_type));

    ptCi->ueInfo.rfsp.rfspPresent = OPT_NOT_PRESENT;
    ptCi->ueInfo.rfsp.rfspChanged = FALSE;
    ptCi->ueInfo.rfsp.subscriberProfileNeedMeasReconfig = FALSE;
    ptCi->ueInfo.rfsp.subscriberProfile = 0;
    ptCi->ueInfo.rfsp.prevSubscriberProfile = 0;

    ptCi->ueInfo.ueSecurityCapa.encryptionCapa = 0;
    ptCi->ueInfo.ueSecurityCapa.integrityCapa = 0;
    ptCi->ueInfo.srcGnbSecurityAlgo.srcIntegrityAlgo = 0;
    ptCi->ueInfo.srcGnbSecurityAlgo.srcCipherAlgo = 0;

    ptCi->ueInfo.selectedPlmnIdentity = 0xFF;

    ptCi->ueInfo.mpsPriorityIndication = VCCB_FLAG_OFF;

    VLIB_InitUeRoHCInfo(&ptCi->ueInfo.roHCCapability);
    VLIB_InitUeEHCInfo(&ptCi->ueInfo.ehcCapability);

    memset(&ptCi->ueInfo.imeisvBuf.maskedIMEISV, 0, sizeof(ptCi->ueInfo.imeisvBuf.maskedIMEISV));
    ptCi->ueInfo.imeisvBuf.maskedIMEISVPresent = OPT_NOT_PRESENT;

    memset(reinterpret_cast<uint8_t *>(&ptCi->ueInfo.ngapBuf.m), 0, sizeof(ptCi->ueInfo.ngapBuf.m));
    /* UE Capability Related Configurations */
    memset(reinterpret_cast<uint8_t *>(&ptCi->ueInfo.ueCapa), 0, sizeof(ptCi->ueInfo.ueCapa));
    ptCi->ueInfo.ueCapa.savedInfo.uceRatSupportType = pGnb->capabilityRequestFilter.uceOptionInfo.uceRatRequestType;

    ptCi->ueInfo.scgConfigQuery = VCCB_FLAG_OFF;

    FnVCCB_InitAllowedBCList(ptCi, &ptCi->cgConfigInfo.configRestrictInfo.allowedBC_ListMRDC);

    VLIB_InitUeReleasePreference(ptCi);
    FnVCCB_InitUeNrRrcOtherConfig(ptCi);

    INIT_UE_CAPA_ENQUIRY_NEED_FLAG(ptCi);
    ptCi->ueCapaEnquiry.sendToAmfFlag = VCCB_FLAG_OFF;
    ptCi->ueCapaEnquiry.rxUeCapaInfoBitFlag = UE_CAPA_REQ_RAT_TYPE_NONE;
    ptCi->ueCapaEnquiry.includeFr2InNrBand = VCCB_FLAG_OFF;     // SVR26A FGR-CC3103
    ptCi->ueInfo.ueSupportUlRrcSegment = VCCB_FLAG_OFF;
    VCCB_CLEAR_TIMER(&ptCi->ueCapaEnquiry.ueCapaInfoTimer, INIT_TIMER);

    ptCi->ueContainerNR.rxFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerMRDC.rxFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerEUTRA.rxFlag = VCCB_FLAG_OFF;

    ptCi->ueContainerNR.isBigCapaFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerMRDC.isBigCapaFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerEUTRA.isBigCapaFlag = VCCB_FLAG_OFF;

    ptCi->ueContainerNR.numocts = 0;
    ptCi->ueContainerMRDC.numocts = 0;
    ptCi->ueContainerEUTRA.numocts = 0;

    ptCi->ueInfo.registeredAmf.amfIdentifierFlag = VCCB_FLAG_OFF;
    ptCi->ueInfo.guamiTypePresent = VCCB_FLAG_OFF;
    ptCi->ueInfo.networkSlice.numSnssai = 0;

    for (loop1 = 0; loop1 < MAX_NUM_S_NSSAI_PER_CALL; loop1++)
    {
        ptCi->ueInfo.networkSlice.snssai[loop1].sdPresent = OPT_NOT_PRESENT;
        ptCi->ueInfo.networkSlice.snssai[loop1].sstSd = 0xFFFFFFFF;
    }

    for (loop1 = 0; loop1 < MAX_BLACK_CELL; loop1++)  // NEDC_SCG_FAILURE_20C
        VCCM_InitBlacklistCell(ptCi, loop1);

    ptCi->ueInfo.ueId.iRNTI = UNKNOWN_IRNTI;
    FnVCCB_InitRRCInactiveInfo(ptCi);
    FnVCCB_InitRRCResumeInfo(ptCi);

    ptCi->ranPagingInfo.dlDataFlag = VCCB_FLAG_OFF;
    ptCi->ranPagingInfo.pagingTriggeredByDLNasTransportFlag = FALSE;
    ptCi->ranPagingInfo.exceptionalFlag = FALSE;
    ptCi->ranPagingInfo.ngUeContextModFromQueue = FALSE;

    ptCi->servCellIndex = 0;
    ptCi->servCellSsbRsrp = UNKNOWN_RSRP;
    for (loop1 = 0; loop1 < MAX_SCELL_PER_CALL; loop1++)
    {
        ptCi->sCell[loop1].state = SCELL_STATE_RELEASED;
        ptCi->sCell[loop1].action = SCELL_ACTION_KEEP;
        ptCi->sCell[loop1].isActionQueued = FALSE;
        ptCi->sCell[loop1].sCellIndex = 0;  // 0 : not used Idx, 1..31
        memset(&ptCi->sCell[loop1].cgi, 0, sizeof(ptCi->sCell[loop1].cgi));
        ptCi->sCell[loop1].sccFlag_cellMeasured = FALSE;
        ptCi->sCell[loop1].sCellUlDlMode = sCell_ul_dl_Unknown;

        ptCi->sCellDuConfig[loop1].state = SCELL_STATE_RELEASED;
        ptCi->sCellDuConfig[loop1].action = SCELL_ACTION_KEEP;
        ptCi->sCellDuConfig[loop1].isActionQueued = FALSE;
        ptCi->sCellDuConfig[loop1].sCellIndex = 0;  // 0 : not used Idx, 1..31
        memset(&ptCi->sCellDuConfig[loop1].cgi, 0, sizeof(ptCi->sCellDuConfig[loop1].cgi));
        ptCi->sCellDuConfig[loop1].sccFlag_cellMeasured = FALSE;
        ptCi->sCellDuConfig[loop1].sCellUlDlMode = sCell_ul_dl_Unknown;
    }
    // caInfo_type
    ptCi->caInfo.totalMaxBw = UNKNOWN_BANDWIDTH;
    ptCi->caInfo.prevTotalMaxBw = UNKNOWN_BANDWIDTH;
    ptCi->caInfo.prevMode = CaOperation_Off;
    ptCi->caInfo.mode = CaOperation_Off;
    ptCi->caInfo.shouldCaBeRestored = FLAG_OFF;
    ptCi->caInfo.lateCaMeasurements = FLAG_OFF;
    ptCi->caInfo.componentCarriersLimit = CcLimit_Off;
    ptCi->caInfo.caControlIdx = MAX_NR_CA_CONTROL_ENTRIES;
    ptCi->caInfo.mode2AnyScellReleased = FALSE;

    ptCi->caInfo.failures.numFailedCarrier = 0;
    ptCi->caInfo.failures.currentFreeFailedCarrierPosition = 0;
    ptCi->caInfo.failures.numFailedCell = 0;
    ptCi->caInfo.failures.currentFreeFailedCellPosition = 0;
    for (loop1 = 0; loop1 < MAX_COLOCCELL; ++loop1)
    {
        ptCi->caInfo.failures.failedCarrier[loop1] = 0;
    }
    memset(ptCi->caInfo.failures.failedCell, 0, sizeof(ptCi->caInfo.failures.failedCell));
    FnVCCB_InitQciAttemptSuccessInfo(ptCi);
    FnVCCB_InitLatestReleasedQci(ptCi);

    ptCi->caInfo.suspendA4Meas       = FALSE;
    ptCi->caInfo.prevSuspendA4Meas   = FALSE;
    ptCi->caInfo.isPeriodicalOngoing = FALSE;
    ptCi->caInfo.bandUsedForPCSelection = 0;
    ptCi->caInfo.releaseCaA4ForVoNROn= FALSE;
    ptCi->caInfo.reserved            = 0;
    VCCB_CLEAR_TIMER(&ptCi->caModTriggerTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->caInfo.caPeriodicMeasTimer, INIT_TIMER);

    ptCi->servingCellMeasInitialInfo.rsrpFlag = FALSE;
    ptCi->servingCellMeasInitialInfo.rsrqFlag = FALSE;

    FnVCCB_InitCslInfo_CaScellInfo(ptCi);

    // securityInfo_type
    VLIB_InitSecurityInfo(ptCi);

    // VNMB allocate vnbUeNG1APId.
    // vnbUeNG1APId is updated when receive UeIdRsp message.

    // mmeInfo_type
    //    ptCi->NGC.associateIdx          = 0xFFFF;

    // handoverInfo_type
    VLIB_InitHoInfo(ptCi);

    // conditionalHandoverInfo_type
    FnVCCB_InitCondHoInfo(ptCi);
    FnVCCB_InitCHOBlockList(ptCi);

    // LTM
    FnVCCB_InitLtmUeInfo(ptCi);
    FnVCCB_InitLtmBlockList(*ptCi);    

    VLIB_InitMroInfo(ptCi);

    VLIB_InitRlfReportInfo(ptCi);

    // hoRestrictInfo_type
    ptCi->hoRestrict.existFlag = FALSE;
    ptCi->hoRestrict.forbiddenRat = 0x00;
    ptCi->hoRestrict.nrRestrictionType = 0x00;
    // store this in VCCM_ALL_CallSetupRequest after getting gnbNum
    // ptCi->hoRestrict.servedPlmn          = GNBINFO.globalGnbId.plmn; // 090730 DM.LEE
    ptCi->hoRestrict.numEplmn = 0x00;
    // ptCi->hoInfo.hoRestrict.numForbidLaPlmn     = 0x00;
    ptCi->hoRestrict.numForbidTaPlmn = 0x00;
    ptCi->hoRestrict.lastEutraPlmnPresent = OPT_NOT_PRESENT;
    ptCi->hoRestrict.lastEutraPlmn.longId = 0;
    ptCi->hoRestrict.encoded.length = 0;  // HANDLE_5GC_MRL_CONTAINER_21B

    // cgConfigInfo
    ptCi->cgConfigInfo.configRestrictInfoFlag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.drxInfoMCG_Flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.measConfigMN_Flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.measConfigMN.measuredFrequenciesMN_Flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.measConfigMN.measGapConfigFR1.flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.measConfigMN.measGapConfigFR2.flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.measConfigMN.gapPurposeFlag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.sourceConfigSCG_Flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.configRestrictInfo.servCellIndexRangeSCG_Flag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.mrdcAssistanceInfoFlag = VCCB_FLAG_OFF;
    ptCi->cgConfigInfo.ueAssistanceInformationSCG_v1620_Flag = VCCB_FLAG_OFF;
    ptCi->phInfoListMCG_DUFlag = VCCB_FLAG_OFF;
    ptCi->servCellInfoListMCG_DUFlag = VCCB_FLAG_OFF;
    ptCi->phInfoListSCG_DUFlag = VCCB_FLAG_OFF;
    ptCi->servCellInfoListSCG_DUFlag = VCCB_FLAG_OFF;

    // SRB Info
    for (loop1 = 0; loop1 < MAX_SRB_PER_CALL; loop1++)
    {
        ptCi->srbInfo[loop1].usage = CI_no_use;
        ptCi->srbInfo[loop1].srbStatus = ERAB_Idle;
        VCCB_CLEAR_TIMER(&ptCi->srbInfo[loop1].timer, INIT_TIMER);
        ptCi->srbInfo[loop1].flag.setupFlag = 0;
        ptCi->srbInfo[loop1].flag.releaseFlag = 0;
        ptCi->srbInfo[loop1].flag.respFlag = 0;
        //        ptCi->srbInfo[loop1].rbInfo.maxRetransInd = 0;
    }
    //    ptCi->ptSi = PTR_NULL;

    ptCi->drbIdsToRelease = 0;  // SA + NSA Init
    ptCi->drbIdsReuse = 0;

    // DRB Info
    if (callType == E_CALLTYPE_NSA)
    {
        for (loop1 = 0; loop1 < MAX_ERAB_PER_CALL; loop1++)
        {
            callERABInfo_type *ptRi = &ptCi->erabInfo[loop1];
            ptRi->usage = CI_no_use;
            ptRi->erabStatus = ERAB_Idle;
            ptRi->activityStatus = ActivityStatus_ACTIVE;
            ptRi->activityStatusPrev = ActivityStatus_ACTIVE;
            ptRi->prevErabStatus = ERAB_Idle;
            ptRi->addSuccessFlagForStat = FLAG_OFF;
            ptRi->sgnbInitRelFlag = FLAG_OFF;
            ptRi->callCountFlag = FLAG_OFF;
            // VCCB_CLEAR_TIMER(&ptRi->timer, INIT_TIMER);
            ptRi->flag.setupFlag = 0;
            ptRi->flag.releaseFlag = 0;
            ptRi->flag.respFlag = 0;
            //        ptRi->rbInfo.maxRetransInd      = 0;
            ptRi->holdTime.isRunning = FALSE;
            ptRi->dataUsage[0].useFlag = FALSE;
            ptRi->dataUsage[1].useFlag = FALSE;
        }
    }
    else  // E_CALLTYPE_SA
    {
        FnVCCB_InitSACallInfo(ptCi);
    }

    ptCi->f1Info.f1UeContextToRemove_flag = VCCB_FLAG_OFF;

    //    ptCi->ptRi = PTR_NULL;

    //    memset( &(ptCi->ueInfo.ambr), 0, sizeof(ambrInfo_type) );

    ptCi->ueInfo.aggrMbrDl = MAX_EXTENDED_BITRATE;
    ptCi->ueInfo.aggrMbrUl = MAX_EXTENDED_BITRATE;
    ptCi->ueInfo.ambr.subAmbrDl = MAX_EXTENDED_BITRATE;
    ptCi->ueInfo.ambr.subAmbrUl = MAX_EXTENDED_BITRATE;
    ptCi->ueInfo.ambr.aggrMbrDl = MAX_EXTENDED_BITRATE;
    ptCi->ueInfo.ambr.aggrMbrUl = MAX_EXTENDED_BITRATE;

    ptCi->ueInfo.sliceMbrInfo.isSliceMBRListProvided = VCCB_FLAG_OFF;
    ptCi->ueInfo.sliceMbrInfo.fNeedModifyDl = VCCB_FLAG_OFF;
    ptCi->ueInfo.sliceMbrInfo.fNeedModifyUl = VCCB_FLAG_OFF;
    ptCi->ueInfo.sliceMbrInfo.fNeedModifySN = VCCB_FLAG_OFF;
    ptCi->ueInfo.sliceMbrInfo.ueCtxModSliceMBRPresent = VCCB_FLAG_OFF;
    ptCi->ueInfo.sliceMbrInfo.nrdcUeSliceMbrCfg = {0, 100, 100, 100, 100};
    FnVCCB_InitUeSliceMBRList(ptCi->ueInfo.sliceMbrInfo.detSliceMbr);
    FnVCCB_InitUeSliceMBRList(ptCi->ueInfo.sliceMbrInfo.subSliceMbr);
    FnVCCB_InitUeSliceMBRList(ptCi->ueInfo.sliceMbrInfo.sliceGfbr);
    FnVCCB_InitUeSliceMBRList(ptCi->ueInfo.sliceMbrInfo.sliceGfbrSN);
    FnVCCB_ClearUeSliceMBRforNRDC(*ptCi);

    // ENB Info
    for (loop1 = 0; loop1 < MAX_GNB_PER_CALL; loop1++)
    {
        ptCi->gnbInfo[loop1].usage = CI_no_use;
        VCCB_CLEAR_TIMER(&ptCi->gnbInfo[loop1].timer, INIT_TIMER);
        ptCi->gnbInfo[loop1].flag.setupFlag = 0;
        ptCi->gnbInfo[loop1].flag.releaseFlag = 0;
        ptCi->gnbInfo[loop1].flag.respFlag = 0;

        for (loop2 = 0; loop2 < MAX_ERAB_PER_CALL; loop2++)
        {
            ptCi->gnbInfo[loop1].dlPath[loop2].usage = CI_no_use;
            VCCB_CLEAR_TIMER(&ptCi->gnbInfo[loop1].dlPath[loop2].timer, INIT_TIMER);
            ptCi->gnbInfo[loop1].dlPath[loop2].flag.setupFlag = 0;
            ptCi->gnbInfo[loop1].dlPath[loop2].flag.releaseFlag = 0;
            ptCi->gnbInfo[loop1].dlPath[loop2].flag.respFlag = 0;

            ptCi->gnbInfo[loop1].ulPath[loop2].usage = CI_no_use;
            VCCB_CLEAR_TIMER(&ptCi->gnbInfo[loop1].ulPath[loop2].timer, INIT_TIMER);
            ptCi->gnbInfo[loop1].ulPath[loop2].flag.setupFlag = 0;
            ptCi->gnbInfo[loop1].ulPath[loop2].flag.releaseFlag = 0;
            ptCi->gnbInfo[loop1].ulPath[loop2].flag.respFlag = 0;
        }
    }

    VCCB_InitRefusedTargetGnbId(*ptCi);

    // init callInfo array with refused target FR1 Cells for FR2->FR1 HO
    VLIB_InitRefusedTargetFR1CellsForFR2FR1HO(ptCi);

    // etc...
    ptCi->numSRB = 0;
    ptCi->numERAB = 0;
    ptCi->numGNB = 0;

    ptCi->srb3Idx = 0xFF;
    ptCi->srb3Prohibited = 0;

    // ptCi->myThreadId = 0;
    ptCi->myThreadId = threadId;

    // Init RRE values
    VLIB_InitRreInfo(ptCi);

    ptCi->simCallFlag = 0;

    ptCi->qciGrpId = 0;
    ptCi->qciGrpIdPrev = 0;

    ptCi->RelFlag = FALSE;

    //    ptCi->retransMsg4 = 0;

    for (loop1 = 0; loop1 < ACTION_QUEUE_SIZE; loop1++)
    {
        ptCi->actionQueue[loop1] = ENQUEUE_NO_ACTION;
    }
    ptCi->queuedMsgCount = 0;

    //    VLIB_InitArpInfo(ptCi);        //SLR410_A078

    ptCi->cellGroupConfig.numocts = 0;

    ptCi->cellGroupConfigFullFlag = VCCB_FLAG_OFF;

    ptCi->usage = CI_use;

    // memset for statistics : set all values to not attempted.
    memset(ptCi->stat.flag, 0x00, MAX_VCCB_FAMILY_TYPE);

    ptCi->amf.needToSetRrcTimeToWaitAmfReloc = FALSE;
    ptCi->amf.reroute.amfSetId = NO_REROUTE_AMF_SET_ID;
    ptCi->gNBSelectedGumaiFlag = VCCB_FLAG_OFF;
    ptCi->servedGuami.amf.setId = INVALID_AMF_SET_ID;

    VCCB_InitMeasData(*ptCi);

    VCCB_CLEAR_TIMER(&ptCi->forcedHO.timer, INIT_TIMER);

    FnVCCB_ClearUeHistoryInformation(ptCi);

    ptCi->cslInfo.scgFailureHistory.lastIdx = 0x00;
    ptCi->cslInfo.scgFailureHistory.failureCount = 0x00;
    for (loop1 = 0; loop1 < MAX_SCG_FAILURE_COUNT; loop1++)
    {
        ptCi->cslInfo.scgFailureHistory.scgFailure[loop1].usage = CI_no_use;
    }

    ptCi->scgFailureFlag = VCCB_FLAG_OFF;
    ptCi->scgSuspendStateForMeas = VCCB_FLAG_OFF;
    ptCi->noCgConfigFlag = VCCB_FLAG_OFF;
    ptCi->scgFailureRelease = SCG_INITIAL_FAILURE;

    VLIB_ResetScgFailAttSuccStats(ptCi);

    ptCi->cslInfo.useNextInitialMeas = VCCB_FLAG_ON;
    VLIB_ClearCslMeasReport(&ptCi->cslInfo.cslMeasInitialInfo);
    VLIB_ClearCslMeasReport(&ptCi->cslInfo.cslMeasLastInfo);
    FnVCCB_InitCslInfo_MobilityStat(*ptCi);
    FnVCCB_InitCslInfo_HoStat(*ptCi);
    FnVCCB_InitCslInfo_MobilityNsaStat(*ptCi);

    FnVCCB_InitCslInfo_ConnectInfo(ptCi);       // New in SVR26A
    FnVCCB_InitCslInfo_CaInfo(ptCi);            // New in SVR26A
    FnVCCB_InitCslInfo_RreInfo(ptCi);           // New in SVR26A

    memset(&ptCi->measurementsConfiguration, 0, sizeof(ptCi->measurementsConfiguration));
    memset(&ptCi->measurementsProcedure, 0, sizeof(ptCi->measurementsProcedure));
    VCCB_CLEAR_TIMER(&ptCi->measurementsConfigShared.pa4m.timer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->measurementsConfigShared.lpm.timer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->measurementsConfigShared.altgap.timer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->pccData.pccTmr, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->pccData.slbTmr, INIT_TIMER);
    VLIB_InitMeasurementsStructs(ptCi);
    VLIB_InitNeedForGapNrUeData(&ptCi->needForGapNrUeData);
    ptCi->ulGapFr2ConfigUe.usage = VCCB_FLAG_OFF;
    ptCi->ulGapFr2ConfigDu.usage = VCCB_FLAG_OFF;

    memset(&ptCi->trafficMonitoringCfg, 0, sizeof(ptCi->trafficMonitoringCfg));

    ptCi->trafficMonitoringState.pdcpFr2PscellChangeFlag = VCCB_FLAG_OFF;
    ptCi->trafficMonitoringState.TrfBasedNrToEutraHoFlag = VCCB_FLAG_OFF;

    ptCi->cslInfo.cslEvent = csl_event_undefined;
    ptCi->cslInfo.cslHoInterruptedByFail = FALSE;
    ptCi->cslInfo.cslErabPduDrbAllocIdx = 0;
    ptCi->cslInfo.cslRreInfo.rreCount = 0;
    ptCi->cslInfo.cslRreInfo.totalHoldingDuration = 0;
    ptCi->cslInfo.cslRreInfo.rreCase = csl_RRE_none;
    ptCi->cslInfo.cslRreInfo.rlfCause = csl_Rlf_none;
    ptCi->cslInfo.cslRreInfo.ueCtxtFetchResult = VCCB_FLAG_OFF;
    ptCi->cslInfo.cslRreInfo.fiveQiExistFlag = VCCB_FLAG_OFF;
    ptCi->cslInfo.cslRreInfo.interFreqFlag = csl_InterGnbRre_Freq_none;
    ptCi->cslInfo.connectInfo.dcNodeType = dcNodeType_none;

    // Clear Info for Common CSL operation
    FnVCCB_InitCslInfo_RetainStat(ptCi);

    // Clear Basic Operation related CSL
    FnVCCB_InitCslInfo_BasicCallStat(ptCi);

    // Clear Emergency/VoNR Requested.
    ptCi->emergencyRequestedForCsl = 0xFF;
    ptCi->vonrRequestedForCsl      = 0xFF;

    ptCi->scgFailCauseInfo = ScgFailCauseInfo_NoScgFailure;
    ptCi->scgSuspendState = VCCB_FLAG_OFF;
    ptCi->x2ModReqS1UTunnelStatus = X2ModReqS1UTunnelStatus_NoNeedToUpdate;
    memset(&ptCi->duSelectionRelatedData, 0x00, sizeof(ptCi->duSelectionRelatedData));
    ptCi->backupDuSelectionRelatedData.selectedBC = UINT_MAX;

    memset(&ptCi->managementTraceInfo.session, 0x00, MAX_TRACE_SESSION_ENTRY*sizeof(struct callInfo_s::Trace_s::TraceRecordingSession_s));
    memset(&ptCi->managementTraceInfo.mdtInfo, 0x00, sizeof(struct callInfo_s::Trace_s::MDT_s));
    memset(&ptCi->collectorTraceInfo, 0x00, sizeof(struct callInfo_s::collectorTraceInfo_s));
    ptCi->managementTraceInfo.traceRfspSpidAllowed = traceRfspSpidUnknown;


    ptCi->traceType = Trace_None;
    ptCi->dbTriggeredTraceDeactType = stopTrigger_none;
    ptCi->traceReportingState = reporting_unknown;
#ifdef __CNF__
    ptCi->selectedCtp = VCCB_MAX_POD_ID_NUM;
#endif
    ptCi->managementTraceInfo.tracePLMNavailable = TRUE;
    ptCi->specificM1InstallOngoing = VCCB_FLAG_OFF;

    ptCi->signalingTraceInfo.trace.traceStatus = ueTraceDisabled;
    ptCi->signalingTraceInfo.mdt.mdtStatus = mdtDisabled;
    ptCi->signalingTraceInfo.mdt.mdt_Activation = mdtUnknown;
    ptCi->signalingTraceInfo.trace.traceDepth = minimum;
    ptCi->signalingTraceInfo.trace.msgSeqNum = 0;
    ptCi->signalingTraceInfo.trace.traceInterfaces = 0;
    ptCi->signalingTraceInfo.failureIndication.cause = no_error;
    ptCi->signalingTraceInfo.trace.ueThroughputAndRfTraceFlag = VCCB_FLAG_OFF;
    ptCi->signalingTraceInfo.nrdcSnActivationSentFlag = VCCB_FLAG_OFF;

    ptCi->queuedActionDuringSNInitSgNBMod = DURING_SN_INIT_SGNB_MOD_ACTION_NONE;

    MsgQueueManager::getInstance().initMsgQueueMapOfCall(*ptCi);

    ptCi->numPDUSess = 0;
    ptCi->redirectionInfo.redirType = NO_REDIRECTION;
    ptCi->redirectionInfo.isEPSFallbackRequired = FALSE;
    ptCi->redirectionInfo.emergencyFallbackInd = FALSE;
    ptCi->redirectionInfo.actualRedirectionMethod = BLIND_REDIR_ONLY;
    ptCi->redirectionInfo.voiceFallbackIndicator = FALSE;
    ptCi->redirectionInfo.onDemandRedirect = FALSE;  // RETAIN_STAT_HO_21C
    ptCi->redirectionInfo.requestedByA2Blind = FALSE;
    ptCi->redirectionInfo.rfspBasedBlind = FALSE;
    ptCi->redirectionInfo.nonNeighbor = FALSE; // SVR24B FGR-MC1028 FR10
    ptCi->redirectionInfo.isSpecificSliceRejected = FALSE;
    ptCi->redirectionInfo.specificSliceRedirectInBasicSA = FALSE;
    ptCi->redirectionInfo.specificSliceFailCause = CAUSE_NO_FAULT;  // SVR25B FGR-CC3101
    VCCB_CLEAR_TIMER(&ptCi->redirectionInfo.mrWaitTimer, INIT_TIMER);

    for (loop1 = 0; loop1 < MAX_NR_OF_OBJECT_ID_DB; loop1++)
    {
        ptCi->anrInfo.nrMeasPciStatInfo[loop1].usage = CI_no_use;
        ptCi->anrInfo.nrMeasMrCountInfo[loop1].usage = CI_no_use;
    }

    for (loop1 = 0; loop1 < MAX_NR_OF_OBJECTEUTRA_ID_DB; loop1++)
    {
        ptCi->anrInfo.eutraMeasPciStatInfo[loop1].usage = CI_no_use;
        ptCi->anrInfo.eutraMeasMrCountInfo[loop1].usage = CI_no_use;
    }

    ptCi->ueInfo.servedPlmn.longId = pGnb->globalGnbId.plmn.longId;
    ptCi->ueInfo.selectedPlmnfromMN.longId = 0;
    ptCi->rrcConnStat.markedCellNumForRrcConnStat = 0xFFFF;
    ptCi->rrcConnStat.markedPlmnIdxForRrcConnStat = 0xFF;
    ptCi->rrcConnStat.markedCellNumForRrcInactiveStat = 0xFFFF;
    ptCi->rrcConnStat.markedPlmnIdxForRrcInactiveStat = 0xFF;
    ptCi->rrcConnStat.markedRfspForRrcConnStat     = 0;
    ptCi->rrcConnStat.markedRfspForRrcInactiveStat = 0;
    ptCi->rrcConnStat.markedRfspForNonDssUeStat    = 0;
    ptCi->rrcConnStat.markedCellNumForNonDssUeStat = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForEnbNonDssUeStat = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForLoggedMdtUeStat = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForVonrEmerRrcConn = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForVonrEmerRrcInact = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForULOnlyRoHCRrcConnStat = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForSymmRoHCRrcConnStat = 0xFFFF;
    ptCi->rrcConnStat.markedCellNumForPowerClassRrcConnStat = 0xFFFF;

    ptCi->ueInfo.rfsp.indexToRFSP = NO_RFSP_VALUE;
    ptCi->hoRestrict.servedPlmn = pGnb->globalGnbId.plmn;
    ptCi->integrityFailCount = 0;
    ptCi->mgapReTriggerCount = 0;

    // STAT
    ptCi->rrcConnStat.addProcTime.isRunning = FALSE;
    ptCi->rrcConnStat.rrcEstabProcTime.isRunning = FALSE;
    ptCi->rrcConnStat.ngEstabProcTime.isRunning = FALSE;
    ptCi->rrcConnStat.rrcConnHoldingTime.isRunning = FALSE;
    ptCi->rrcConnStat.rrcInactHoldingTime.isRunning = FALSE;
    ptCi->rrcConnStat.rrcConnInactHoldingTime.isRunning = FALSE;
    ptCi->rrcConnStat.rrcEndcConnHoldingTime.isRunning = FALSE;

    // FOR UPM
    VCOM_DisableUPM(ptCi);

    ptCi->noRabInX2Resp = VCCB_FLAG_OFF;
    ptCi->locationReporting_flag = VCCB_FLAG_OFF;

    memset(&ptCi->locRpt, 0, sizeof(ptCi->locRpt));
    ptCi->locRpt.reportOn = CI_no_use;

    // FOR MRDC
    memset(&ptCi->mrdcInfo, 0x00, sizeof(ptCi->mrdcInfo));
    ptCi->selectedMrdcType = MRDC_NONE;
    ptCi->snAddFailCntWithSelectedMrdcType = 0;
    ptCi->blindModeSNAddReTrialCnt = 0;

    memset(&ptCi->loadBasedNrdcScgBandSelectInfo, 0, sizeof(ptCi->loadBasedNrdcScgBandSelectInfo));
    ptCi->fastAccBoundaryTime = FAST_ACCESS_BOUNDARY_TIME_NOT_SET; // SVR26A:LC0705:FR20
    FnVCCB_InitNrdcAddCacRejectedFreq(ptCi);

    FnVCCB_InitLtmBufferMapOfCall(*ptCi);

    // full configuration
    VLIB_InitFullConfigurationInfo(ptCi);

    ptCi->invalidLocInfoSN = TRUE;      // NEDC_SCG_FAILURE_20C
    RESET_NEDC_SCG_FAILINFO();          // NEDC_SCG_FAILURE_20C
    ptCi->psCellInfo.status = N_EQUIP;  // NEDC_SCG_FAILURE_20C

    INIT_MOD_PROC_INFO(ptCi);

    // For UL Path Switch
    ptCi->isUlPathSwitchEnabled = true;
    ptCi->ulPathControl.curr.nrUlCoverageState = IN_NR_UL_COVERAGE_STATE;
    ptCi->ulPathControl.prev.nrUlCoverageState = IN_NR_UL_COVERAGE_STATE;
    ptCi->ulPathControl.curr.ulPdcpPath = UlPdcpPath_ScgShared;
    ptCi->ulPathControl.prev.ulPdcpPath = UlPdcpPath_ScgShared;
    ptCi->ulPathControl.curr.ulPrimaryPathMode = initial_SCG;
    ptCi->ulPathControl.prev.ulPrimaryPathMode = initial_SCG;
    ptCi->ulPathControl.curr.qci128SwitchFlag = VCCB_FLAG_OFF;
    ptCi->ulPathControl.prev.qci128SwitchFlag = VCCB_FLAG_OFF;
    ptCi->ulPathControl.ueCongestionState = false;
    ptCi->ulPathControl.revertUlPathChange = false;
    ptCi->ulPathControl.prbUsageInformation = 0xFFFF;
    ptCi->isDataStallPreventionTriggerReceived = UlPath_DataStall_FunctionOff;
    ptCi->HighLatencyIndication = Low_Latency;
    ptCi->changeTriggeredByE1PrivateMessage = false;

    // UE Capability Information
    ptCi->ueContainerNR.rxFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerMRDC.rxFlag = VCCB_FLAG_OFF;
    ptCi->ueContainerEUTRA.rxFlag = VCCB_FLAG_OFF;

    // Rat Type : NR CAPA
    ptCi->ueContainerNR.numocts = 0;
    FnVCCB_InitExtUeCapaInfo(ptCi, &ptCi->ueContainerNR_Ext);
    // Rat Type : MRDC CAPA
    ptCi->ueContainerMRDC.numocts = 0;
    FnVCCB_InitExtUeCapaInfo(ptCi, &ptCi->ueContainerMRDC_Ext);
    // Rat Type : EUTRA CAPA
    ptCi->ueContainerEUTRA.numocts = 0;
    FnVCCB_InitExtUeCapaInfo(ptCi, &ptCi->ueContainerEUTRA_Ext);
    // extUeCapaInfo_type : UL Dedicated Segment Messages, NR/MRDC/EUTRA  capa
    FnVCCB_InitExtUeCapaInfo(ptCi, &ptCi->ulRrcSegment);

    // SVR26A FGR-CC3103 IODT RACS
    ptCi->ueCapaID.length = 0;

    // NRDC AS Key Management
    for (loop1 = 0; loop1 < MAX_REFUSED_TARGET_GNB_PER_CALL; loop1++)
    {
        ptCi->SNgNodeGnbIdNotSupportSecurityAlgo[loop1] = INVALID_GNB_ID;  // For storing gnbID
    }

    // DSS Release flag
    ptCi->dssCauseRelease = FALSE;

    ptCi->snAddFailCntWithSelectedMrdcType = 0;

    // Nrppa
    ptCi->NrPpaContext.isNrppaPosContextValid = FALSE;
    ptCi->isNrPpaContextValid = FALSE;
    ptCi->NrPpaContext.isMeasReportRequired = FALSE;
    ptCi->NrPpaContext.tRPDirectNormal.geoCoordinatesPresent = OPT_NOT_PRESENT;
    ptCi->NrPpaContext.numOfIratMeasuredQuantities = 0;
    ptCi->NrPpaContext.numOfMeasuredQuantities = 0;
    ptCi->NrPpaContext.isMeasConfigRequired = FALSE;
    ptCi->NrPpaContext.isMeasGapConfiguredInOnDemand = FALSE;
    ptCi->NrPpaContext.measResultCell.numCellReport = 0;
    ptCi->NrPpaContext.measResultCellEUTRA.numCellReport = 0;
    ptCi->NrPpaContext.isTriggerMeasReconfig = FALSE;
    VLIB_Init_NrppaMeasId(ptCi);
    VCCB_CLEAR_TIMER(&ptCi->nrppaTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->nrppaTimerAoA, INIT_TIMER);
    ptCi->NrPpaContext.isPcellFilled               = FALSE;
    ptCi->NrPpaContext.isNrAoARequested            = FALSE;
    ptCi->NrPpaContext.isAoAReceived               = FALSE;
    ptCi->NrPpaContext.isMeasPeriodicityAoAPresent = FALSE;
    ptCi->NrPpaContext.isSRSConfigReceivedECID     = FALSE;
    ptCi->NrPpaContext.isSRSConfigReceivedMultiRTT = FALSE;
    ptCi->NrPpaContext.aoANotReceivedInTime        = FALSE;
    ptCi->NrPpaContext.otherMeasNotReceivedInTime  = FALSE;
    ptCi->NrPpaContext.isF1EcidMeasFailureReceivedDuringSrsOn = FALSE;
    ptCi->NrPpaContext.measResultCell.measResultNR[0].physCellId = 0xFFFF;

    // SVR23B FGR-CC3101 FR120
    ptCi->rlfFailF1Cause = CauseF1apRadioRLFailure;

    // Multi-Carrier Operation
    ptCi->pccData.state = PCC_NOT_SET;
    ptCi->pccData.pccMode = PccMode_NotSet;
    ptCi->pccData.interRatState = IRatPCCState_Idle;
    ptCi->pccData.interRatConfigIndex = MCO_CONFIG_INDEX_NOT_SET;
    std::fill_n(ptCi->pccData.scellAltBandTab, MAX_MCO_MFBI_CELLS, BAND_NOT_FOUND);
    std::fill_n(ptCi->pccData.scellAltBandNrArfcnTab, MAX_MCO_MFBI_CELLS, UNKNOWN_ARFCN);
    ptCi->pccData.pscellNbrIdx = MAX_NBR_PER_CELL;
    for (loop1 = 0; loop1 < MAX_PCC_CANDIDATES; loop1++)
    {
        ptCi->pccData.candidate[loop1].mrRcvPcellFlag = false;
        ptCi->pccData.candidate[loop1].cellNum = MAX_CELL_NUM_VCCB;
    }
    VCCB_CLEAR_TIMER(&ptCi->pccData.interRatWaitingTmr, INIT_TIMER);
    FnVCCB_InitSccData(ptCi, SCC_ALL_INIT_AND_TIMER_INIT);

    RESET_NRDC_SCG_FAILINFO(ptCi);  // NRDC_SCG_FAILURE_21D
    // SVR22C2 FGR-CC5203 FR111
    ptCi->nrdcScgFailureInfo.ueSignatureFlag = VCCB_FLAG_OFF;
    ptCi->nrdcScgFailureInfo.allowSNAdditionUECustomSignature =
        (ptCi->gnbNum < MAX_GNB_PER_SYS)
            ? pGnb->cfg.scgFailureHandlingInfoNrDc.allowSNAdditionUECustomSignature & e_BitMask_1bit
            : VCCB_FLAG_ON & e_BitMask_1bit;

    // SVR23A FGR-CC5203
    memset(&ptCi->blockListInfo, 0, sizeof(NrdcScgBlockListInfo_type));
    ptCi->blockListInfo.usage = pGnb->cfg.scgFailureHandlingInfoNrDc.blockedGnbSupport;
    ptCi->blockListInfo.currentIndex = 0xFF;

    ptCi->ueMeasForSon.selectedNrArfcn = UNKNOWN_ARFCN;
    ptCi->ueMeasForSon.selectedCbrsArfcn = UNKNOWN_ARFCN;
    ptCi->ueMeasForSon.selectedEutraArfcn = UNKNOWN_ARFCN;
    VCCB_CLEAR_TIMER(&ptCi->ueMeasForSon.measTimer, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->ueMeasForSon.measDelayTimer, INIT_TIMER);
    ptCi->ueMeasForSon.ueMeasForSonState = UeMeasForSonState_idle;

    ptCi->drxCuInfo.status = DRX_OFF;
    ptCi->drxCuInfo.shortCycleStatus = DRX_OFF;
    ptCi->drxCuInfo.srsConfigTriggeredType = SRS_Not_Configured;
    ptCi->saPrevInfo.srsConfigTriggeredType = SRS_Not_Configured;
    memset(&ptCi->drxCuInfo.drxConfigInfo, 0, sizeof(VccbDrxConfigInfo_type));
    memset(&ptCi->drxCuInfo.backupDrxConfigInfo, 0, sizeof(VccbDrxConfigInfo_type));

    ptCi->redCapUeInfo.redCapUeFlag = 0;
    ptCi->redCapUeInfo.supportOfRedCap_r17 = 0;
    ptCi->redCapUeInfo.supportOf16DRB = 0;
    ptCi->redCapUeInfo.supportOf18bitPdcpSN = 0;
    ptCi->redCapUeInfo.isAttachFailedByRedCap = 0;

    // REDCAP Conn count flag
    ptCi->rrcConnStat.redCapCountFlag               = VCCB_FLAG_OFF;
    ptCi->rrcConnStat.redCapInactiveCountFlag       = VCCB_FLAG_OFF;

    ptCi->cacTriggeredRedirInfo.usage = CI_no_use;
    ptCi->xnStatusTransferEnhancementOnOff = pGnb->cfg.xnStatusTransferEnhancementOnOff;
    ptCi->supportRreDuringUeCapaEnquiry = pGnb->cfg.supportRreDuringUeCapaEnquiry;
    ptCi->caProfileIdx = -1;

    // Init MRS
    ptCi->mrsInfo.rejectStatus = VCCB_FLAG_OFF;
    FnVCCB_InitMrsInfo(ptCi, MRS_ALL_INIT);

    ptCi->pMaxFR1 = UNKNOWN_PMAX;

    ptCi->selectableUpsOverloadFlag = FALSE;

    ptCi->invalidCellSgNBAddReject.spidFlag = FALSE;
    ptCi->endcAddCpFailDetailCause = StatEndcAddPrepCpFail_Unspecified;
    ptCi->spCellSelFailStats.num = 0;

    FnVCCB_InitUeAssistanceInformation(*ptCi);

    ptCi->saLbHoExecuted = FALSE;
    ptCi->caOverloadedStatus = VCCB_FLAG_OFF;

    SET_DRX_RELEASE_FLAG_OFF(ptCi);
    //SamsungexpectedThroughputList
    ptCi->samsungExpectedTPListFlag = VCCB_FLAG_OFF;
    ptCi->samsungExpectedTPList.numOfItems = 0;
    memset(&ptCi->samsungExpectedTPList, 0, sizeof(ptCi->samsungExpectedTPList));

    // PH-Info and ServCellList from DU
    memset(&ptCi->servCellInfoDuMcgList, 0, sizeof(ptCi->servCellInfoDuMcgList));
    memset(&ptCi->phInfoDuMcgList, 0, sizeof(ptCi->phInfoDuMcgList));
    memset(&ptCi->servCellInfoDuScgList, 0, sizeof(ptCi->servCellInfoDuScgList));
    memset(&ptCi->phInfoDuScgList, 0, sizeof(ptCi->phInfoDuScgList));

    VCCB_CLEAR_TIMER(&ptCi->sliceAwareLbInfo.waitTimer, INIT_TIMER); //SVR23B FGR-LC0710
    ptCi->sliceAwareLbDecisionCount = 0; //SVR23B FGR-LC0710
    FnVccb_InitNrdcUlTrafficMonitorCtx(ptCi);
    FnVCCB_InitRestrictedGnbsforNrdcSn(ptCi);
    VLIB_InitSliceLbInfo(ptCi); //SVR23B FGR-LC0710
    ptCi->isNrdcDisabledByCpcOvld = false;
    FnVCCB_InitNrdc2NrcaLBCtxt(*ptCi);
    INIT_NRDC_ADD_A4MR_RECEIVED(ptCi);
    ptCi->isNrdcProhibitedByLtmHo = false;

#ifdef IPC_TEST
    memset(&ptCi->checker.drb, 0, sizeof(ptCi->checker.drb));
    ptCi->checker.isSendtoVNMB = false;
    ptCi->checker.rxSecurityFailure = VCCB_FLAG_OFF;
#endif

    if (callType == E_CALLTYPE_SA)
    {
        // SVR23B FGR-CC5200 Basic Operation For NR-DC :: FR410 :: Nrdc Control by cpc overload
        FnVCCB_SetNrdcOverloadControl(ptCi);
    }

    FnVCCB_InitOnDemandHoInfo(ptCi);
    FnVCCB_USTM_Init(*ptCi);
    FnVCCB_InitNrConnectedMobilityControlInfo(*ptCi);
    FnVCCB_InitSpecificSliceForcedMobilityInfo(*ptCi);

    OFF_VCCB_UE_INFO_REQ_FLAG(ptCi);

    VCCB_NSA_CleanHoMrQueueInfo(*ptCi);

    ptCi->initialContextSetupRequestPresentFlag = VCCB_FLAG_OFF;

    ptCi->qInfo.qMsgId = 0;             // SVR25B PMB FGR-MC1021:FR130
    ptCi->qInfo.qRedirectionMsgId = 0;  // SVR25B PMB FGR-MC1021:FR130

    for (auto &mrCellInfo: ptCi->mrCellInfo)
    {
        mrCellInfo.state = mrCellState_Disable;
        mrCellInfo.rsrp = UNKNOWN_RSRP;
        mrCellInfo.cellIdentity = MAX_CELL_IDENTITY;
        mrCellInfo.seValue = static_cast<uint32_t>(INVALID_SE_VALUE) * 1000;
        mrCellInfo.receivedTime = timespec {};
    }

    memset(&ptCi->reqSpecificSliceInfoForHo, 0x00, sizeof(ptCi->reqSpecificSliceInfoForHo)); // SVR26A FGR-CC3101 FR500
    // SVR26A FGR-TA0601
    memset(&ptCi->vomaInfo, 0, sizeof(ptCi->vomaInfo));
    ptCi->vomaInfo.vomaEvent = VOMA_unknown;

    VccbAi_InitContext

    return;
}
/**
 * @brief           Initialize gNB information in call context
 *
 * @param[in/out]   ptCi : call context / callInfo_type
 * @param[out]      ptGi : gNB information / callGNBInfo_type
 * @param[in]       hoGnbIdx : gNB index to allocate / uint16_t
 * @return          None
 */
void VLIB_InitCallGnbInfo(callGNBInfo_type *ptGi, uint16_t hoGnbIdx)
{
    AUTO_TRACE_FUNC;

    uint16_t pathIdx = 0;

    VALID_E_POINT_CHECK_VOID(ptGi);

    memset(ptGi, 0xFF, sizeof(callGNBInfo_type));

    ptGi->usage = CI_use;
    ptGi->hoGnbIdx = hoGnbIdx;

    VCCB_CLEAR_TIMER(&ptGi->timer, INIT_TIMER);

    ptGi->transactionId = static_cast<uint16_t>((((MAX_SRB_PER_CALL + 1) << SRB_SHIFT) & SRB_MASK) +
                                                (((MAX_ERAB_PER_CALL + 1) << ERAB_SHIFT) & ERAB_MASK) +
                                                (static_cast<uint16_t>((hoGnbIdx) << GNB_SHIFT) & GNB_MASK));

    ptGi->flag.setupFlag = 0;
    ptGi->flag.releaseFlag = 0;
    ptGi->flag.respFlag = 0;

    ptGi->globalGnbId.gnbId.id = 0;
    ptGi->globalGnbId.gnbId.idLength = 0;
    ptGi->globalGnbId.plmn.longId = 0;
    ptGi->targetCellId.ci = 0;
    ptGi->targetCellId.plmn.longId = 0;
    ptGi->targetPci = 0;
    ptGi->targetSsbFrequency = 0;
    ptGi->targetExtendedTac = 0;

    ptGi->myVnbUeX2APId = INVALID_CALL_ID;
    ptGi->peerVnbUeX2APId = INVALID_CALL_ID;

    // ptGi->associateIdx      = 0xFFFF; // TODO

    for (pathIdx = 0; pathIdx < MAX_ERAB_PER_CALL; pathIdx++)
    {
        ptGi->dlPath[pathIdx].usage = CI_no_use;
        ptGi->ulPath[pathIdx].usage = CI_no_use;
    }
}

/**
 * @brief     Initialize path of NR X2 interface in call context
 * @param     ptCi : call context
 * @param     ptPi : NR X2 interface path information
 * @param     hoGnbIdx : vNB index / uint16_t
 * @param     erabIdx : ERAB index to allocate / uint16_t
 * @param     pathType : forwarding direction DL or UL / uint16_t
 */
void VLIB_InitPathInfo(callInfo_type *ptCi, pathInfo_type *ptPi, uint16_t hoGnbIdx, uint16_t erabIdx, uint16_t pathType)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(ptPi);

    memset(reinterpret_cast<uint8_t *>(ptPi), 0xFF, sizeof(pathInfo_type));

    ptPi->usage = CI_use;

    VCCB_CLEAR_TIMER(&ptPi->timer, INIT_TIMER);

    ptPi->transactionId =
        static_cast<uint16_t>((static_cast<uint16_t>((pathType) << DIRECTION_SHIFT) & DIRECTION_MASK) +
                              (((MAX_SRB_PER_CALL + 1) << SRB_SHIFT) & SRB_MASK) +
                              (static_cast<uint16_t>((erabIdx) << ERAB_SHIFT) & ERAB_MASK) +
                              (static_cast<uint16_t>((hoGnbIdx) << GNB_SHIFT) & GNB_MASK));

    ptPi->flag.setupFlag = 0;
    ptPi->flag.releaseFlag = 0;
    ptPi->flag.respFlag = 0;

    // SLR450_A073
    /*
    if(pathType == DL_FORWARDING)
    {
        ptPi->sTEID                     = VNB_GTP_NGX2TEID(1);
    }
    else
    {
        ptPi->sTEID                     = VNB_GTP_NGX2TEID(2);
    }
    */
}

/**
 * @brief      This function is used to initialize handover stat flag before/after handover event.
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitHoStat(callInfo_type *ptCi)
{  // {{{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t idx;

    // This Family have to exist in pmStatMappingTable of ComLibStatistic.cpp
    int arrCheckFamily[] = {STAT_HO_OUT, STAT_HO_IN, STAT_HO_EUTRAN_OUT, STAT_HO_EUTRAN_OUT_FALLBACK,
        STAT_HO_EUTRAN_IN, STAT_HO_INTER_FREQ_OUT, STAT_HO_INTRA_FREQ_OUT, STAT_CHO_IN, STAT_CHO_OUT,
        STAT_HO_EUTRAN_FREQ_OUT, STAT_HO_INTER_FREQ_OUT_REDCAP, STAT_HO_INTRA_FREQ_OUT_REDCAP,
        STAT_HO_EUTRAN_FREQ_OUT_REDCAP, STAT_HO_IN_REDCAP, STAT_HO_EUTRAN_IN_REDCAP, STAT_LB_HO,
        STAT_FORCED_MOBILITY_NR, STAT_FORCED_MOBILITY_EUTRAN, STAT_INTRA_NR_LB_HO, STAT_HO_IN_SERVICE_TYPE,
        STAT_HO_EUTRAN_IN_SERVICE_TYPE, STAT_HO_OUT_SERVICE_TYPE, STAT_HO_EUTRAN_FREQ_OUT_SERVICE_TYPE, 
        STAT_HO_EUTRAN_OUT_SERVICE_TYPE, STAT_HO_EUTRAN_OUT_FALLBACK_SERVICE_TYPE, STAT_HO_INTER_FREQ_OUT_SERVICE_TYPE,
        STAT_HO_INTRA_FREQ_OUT_SERVICE_TYPE}; 

    for (idx = 0; idx < GET_ARR_SIZE(arrCheckFamily); idx++)
    {
        int key = GET_FAMIDX(arrCheckFamily[idx]);

        VCCB_CHECK_RANGE_MIN_MAX_AND_CONTINUE(key, 0, MAX_VCCB_FAMILY_TYPE - 1);

        // Check - Attempt and prepSuccess
        SET_CALL_STAT_FLAG(ptCi, key, NO_ATTEMPT);
    }

    return;
}  // }}}

/**
 * @brief This function is used to initialize qInfo structure.
 * 
 * @param rCall call info
 */
void VCCB_InitQInfo(callInfo_type &rCall)
{
    AUTO_TRACE_FUNC;

    rCall.qInfo.qMrAction = MR_ACTION_NONE;
    rCall.qInfo.qRelDuringHoAction = DURING_HO_ACTION_NONE;
    rCall.qInfo.measResultList2NRSize = 0;
    rCall.qInfo.measResult2NRType = CandidateCellInfoList_Not_Set;
    rCall.qInfo.qRelType = 0xffff;
    rCall.qInfo.qTargetNbrIdx = 0xffff;
    rCall.qInfo.qTargetCellNum = 0xffff;
    rCall.qInfo.qTargetCellBand = 0xffff;
}

/**
 * @brief      This function is used to initialize hoInfo after switchover.
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitHoInfo(callInfo_type *ptCi, bool initQInfo)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    // Get the length from the start of hoInfo to measResultList2NRPacked
    const int memset_length =
        (reinterpret_cast<std::byte *>(&ptCi->hoInfo.measResultList2NRPacked[0]) - reinterpret_cast<std::byte *>(&ptCi->hoInfo));

    // If there is an unexpired timer, cancel it. ( Below this, it is initialized to All 0xFF. )
    VCCB_DetachHoTimer(&ptCi->hoInfo);

    memset(&ptCi->hoInfo, 0xFF, memset_length);

    ptCi->hoInfo.cause = HoPurpose_DLQuality;  // = 0x00
    ptCi->hoInfo.hoType.rat = 0x00;
    ptCi->hoInfo.hoType.nb = 0x00;
    ptCi->hoInfo.hoType.du = 0x00;
    ptCi->hoInfo.hoType.interface = 0x00;
    ptCi->hoInfo.hoType.system = HO_TYPE_INTRA_SYSTEM;
    ptCi->hoInfo.tArfcn = UINT32_MAX;  // IRAT_HO
    ptCi->hoInfo.srcServingCellMO = 0;
    ptCi->hoInfo.mrWithMGap = FALSE;  // NR_CONN_MC_21D
    ptCi->hoInfo.rrcSpCellChgTime.isRunning = FALSE;

    if (ptCi->callType == E_CALLTYPE_SA)
    {
        ptCi->hoInfo.hoType.initiated = HO_TYPE_SA_INIT;
    }
    else
    {
        ptCi->hoInfo.hoType.initiated = HO_TYPE_SGNB_INITIATED;
    }

    ptCi->hoInfo.hoGnbIdx = MAX_GNB_PER_CALL;
    ptCi->hoInfo.isTarget = FALSE;
    ptCi->hoInfo.isLbHo = FALSE;
    ptCi->hoInfo.isVoNrLbHo = FALSE; // FGR-LC0712
    ptCi->hoInfo.endcHoRollbackActive = FALSE;
    ptCi->hoInfo.scgMeasResultList2NRSize = 0;
    ptCi->hoInfo.nsaLbMeasState = NSA_LB_MEAS_STATE_UNKNOWN;
    ptCi->hoInfo.newUlTnlInfoRequired = FALSE;
    ptCi->hoInfo.hoUsingSRB3 = FALSE;

    ptCi->hoInfo.tgtPlmn.longId = 0;
    ptCi->hoInfo.hoPreparationInfo.numocts = 0;
    ptCi->hoInfo.HOTimeFlag = FALSE;
    ptCi->hoInfo.OutHOTimeFlag = FALSE; // IRAT_HO_23B

    // Change all FLAGs associated with this flag to OFF.
    ptCi->hoInfo.clearFlag = 0;
    ptCi->hoInfo.cancelControl.clearFlag = 0;

    ptCi->hoInfo.nASSecurityParametersFromNGRAN.length = 0;
    ptCi->hoInfo.targetcRNTIUsage = CI_no_use;
    ptCi->hoInfo.sgnbAddTrigInd = SgnbAddTrigInd_Unknown;  // Common Init MRDC
    ptCi->hoInfo.choTriggerInd = e_ChoTriggerCause_Unknown;

    ptCi->hoInfo.directDataFwdFromTarget = FALSE;
    ptCi->hoInfo.internalFwdAvailable = FALSE;
    ptCi->hoInfo.allPduSessDirectFwdPathAvailable = FALSE;

    ptCi->hoInfo.sourceNGRANNodeID.gnbId.id = 0;
    ptCi->hoInfo.sourceNGRANNodeID.gnbId.idLength = 0;

    VCOM_InitSelectedTAI(&ptCi->hoInfo.selectedTAI);
    VLIB_InitMGapDuData(ptCi->hoInfo.mGapDuTarget);
    ptCi->hoInfo.ulGapFr2ConfigDu.usage = VCCB_FLAG_OFF;

    VCCB_CLEAR_TIMER(&ptCi->hoInfo.timer.e1, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->hoInfo.timer.f1, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->hoInfo.timer.xn, INIT_TIMER);
    VCCB_CLEAR_TIMER(&ptCi->hoInfo.timer.spec, INIT_TIMER);

    if (initQInfo)
    {
        VCCB_InitQInfo(*ptCi);
    }

    return;
}

void VLIB_InitMroInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->mroInfo.waitUeInfoResp, INIT_TIMER);

    memset(&ptCi->mroInfo, 0x00, sizeof(mro_type));

    ptCi->mroInfo.waitRrcUeInfoResp = FALSE;
    ptCi->mroInfo.isHoCmdTransmitted = FALSE;

    ptCi->mroInfo.ue.hoInitiated = HO_TYPE_SA_INIT;

    ptCi->mroInfo.ue.timestamp.tv_sec = -1;
    ptCi->mroInfo.ue.timestamp.tv_nsec = -1;
    ptCi->mroInfo.ue.needToCheckLoad = TRUE;
    ptCi->mroInfo.ue.isOverloaded = FALSE;

    FnVCCB_InitUpmAlgoInfo(&ptCi->mroInfo.upmAlgoInfoInstalled);  // Clear UPM Info (Installed CIO)

    return;
}

void VLIB_InitRlfReportInfo(callInfo_type *ptCi)
{
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->rlfReportInfo.received = OPT_NOT_PRESENT;
    ptCi->rlfReportInfo.pDecoded = NULL;
    ptCi->rlfReportInfo.isLastServingCellValid = FALSE;
}

/**
 * @brief      This function is used to initialize reestablishInfo after switchover.
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitRreInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->reestablishFlag = FALSE;
    ptCi->ueInfo.securityModFlag = VCCB_FLAG_OFF;
    ptCi->ueInfo.snKeyChangeWOMnKeyChange = VCCB_FLAG_OFF;

    ptCi->ueInfo.reestablishInfo.reestablishType = RRE_TYPE_UNKNOWN;
    ptCi->ueInfo.reestablishInfo.isTargetGnb = FALSE;
    ptCi->ueInfo.reestablishInfo.isNrdcCall = FALSE;
    ptCi->ueInfo.reestablishInfo.isIntraGnbPlmnChg = FALSE;

    ptCi->ueInfo.reestablishInfo.swappedPlmn.longId = 0;

    ptCi->ueInfo.reestablishInfo.rreGnbIdx = 0xFFFF;
    ptCi->ueInfo.reestablishInfo.rreMatchType = RRE_INVALID_MATCH;

    ptCi->ueInfo.reestablishInfo.firstOldCellFlag = VCCB_FLAG_ON;
    ptCi->ueInfo.reestablishInfo.firstOldCellNum = UNKNOWN_CELL;

    ptCi->ueInfo.reestablishInfo.oldF1Idx = UNKNOWN_DUIDX;
    ptCi->ueInfo.reestablishInfo.oldCellNum = UNKNOWN_CELL;
    ptCi->ueInfo.reestablishInfo.oldCellBand = BAND_NOT_FOUND;
    ptCi->ueInfo.reestablishInfo.oldDuUeId = UNKNOWN_UE_APID;

    ptCi->ueInfo.reestablishInfo.srcIntegrityAlgo = 0xFFFF;
    ptCi->ueInfo.reestablishInfo.srcCipherAlgo = 0xFFFF;

    ptCi->ueInfo.reestablishInfo.duReleaseCount = 0;
    ptCi->ueInfo.reestablishInfo.xnReleaseCount = 0;
    ptCi->ueInfo.reestablishInfo.reestCount_byReconfigFail = 0;

    ptCi->ueInfo.reestablishInfo.numReestabNCell = 0;

    FnVCCB_InitUpmStatInfo(&ptCi->ueInfo.reestablishInfo.upmStatInfo);

    return;
}

void FnVCCB_InitRRCInactiveInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VCCB_CHECK_E_RANGE_AND_VOID(ptCi->gnbNum, MAX_GNB_PER_SYS);

    ptCi->ueInfo.RrcInactiveInfo.ability = 0;
    ptCi->ueInfo.RrcInactiveInfo.pagingDrx = 0xFFFF;
    ptCi->ueInfo.RrcInactiveInfo.ppi = 0xFF;
    ptCi->ueInfo.RrcInactiveInfo.resumeCause = MAX_VALID_NR_RRC_RESUME_CAUSE;
    VCCB_CLEAR_TIMER(&ptCi->ueInfo.RrcInactiveInfo.guardTimer, INIT_TIMER);
    ptCi->ueInfo.RrcInactiveInfo.activityNotificationLevel =
        GNBINFO[ptCi->gnbNum].RrcInactiveInfo.activityNotificationLevel;
    ptCi->ueInfo.RrcInactiveInfo.redCapActivityNotificationLevel =
        GNBINFO[ptCi->gnbNum].RrcInactiveInfo.redCapActivityNotificationLevel;

    // SVR22B : FGR-CC0103
    ptCi->ueInfo.RrcInactiveInfo.oldInactReportType = NGAP_cancel_report;  // 2
    ptCi->ueInfo.RrcInactiveInfo.newInactReportType = NGAP_cancel_report;  // 2
    ptCi->ueInfo.RrcInactiveInfo.transitionReportedBy = 0;
    ptCi->ueInfo.RrcInactiveInfo.singleReportByPathSwitch = RRC_INACTIVE_REPORT_SENT_NONE;  // 0

    ptCi->ueInfo.RrcInactiveInfo.lastUsedIRNTIUeId = MAX_IRNTI_UE_ID;

    // SVR24B: FGR-CC0103
    ptCi->ueInfo.RrcInactiveInfo.numInactTAI = 0;

    // SVR24B: FGR-MC1030
    ptCi->ueInfo.RrcInactiveInfo.extUEIdentityIndexValue = UNKNOWN_UEID_IDX;
    ptCi->ueInfo.RrcInactiveInfo.subGroupId = UNKNOWN_SUBGROUP_ID;

    // SVR25B: FGR-CC6201
    ptCi->ueInfo.RrcInactiveInfo.edrxAbility = RRC_INACTIVE_NOT_SUPPORT;
    ptCi->ueInfo.RrcInactiveInfo.nrPagingEDRXCycle = 0xFF;
    ptCi->ueInfo.RrcInactiveInfo.nrPagingTimeWindow = 0xFF;
    ptCi->ueInfo.RrcInactiveInfo.hashedUEIdentityIndexFlag = VCCB_FLAG_OFF;
    memset(&ptCi->ueInfo.RrcInactiveInfo.hashedUEIdentityIndex, 0, sizeof(NGAP_HashedUEIdentityIndexValue_type));

    return;
}

void FnVCCB_InitRRCResumeInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    RrcResumeInfo_type *pRRCResumeInfo = &ptCi->ueInfo.rrcResumeInfo;

    pRRCResumeInfo->resumeType = RESUME_TYPE_UNKNOWN;
    pRRCResumeInfo->goingToRRCIdleFlag = VCCB_FLAG_OFF;
    pRRCResumeInfo->ranPaged = VCCB_FLAG_OFF;

    pRRCResumeInfo->srcIntegrityAlgo = EIA0;
    pRRCResumeInfo->srcCipherAlgo = EEA0;
    pRRCResumeInfo->isDRBIntegrityProtectionEnabled = VCCB_FLAG_OFF;

    pRRCResumeInfo->isRRCResumeRequestReceived = VCCB_FLAG_OFF;
    pRRCResumeInfo->targetCellSsbArfcn = UNKNOWN_ARFCN;
    pRRCResumeInfo->isRrcReleaseRelayed = VCCB_FLAG_OFF;
    pRRCResumeInfo->rrcFallBackFlag = VCCB_FLAG_OFF;
    pRRCResumeInfo->fullConfigRequiredFlag = VCCB_FLAG_OFF;
    pRRCResumeInfo->actionInSNRelease = SN_REL_IN_RRC_INACTIVE_INITIAL_STATE;
    pRRCResumeInfo->preventSNAddFlag = VCCB_FLAG_OFF;

    return;
}

/**
 ** @brief     Clear DRB PreEmption after switchover
 ** @param     ptCi : call context
 **/
void FnVCCB_ClearDRBPreemption(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);
    uint8_t callDrbIdx = 0;
    callDRBInfo_type *ptDi = NULL;

    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_SA_CALL; callDrbIdx++)
    {
        ptDi = &ptCi->drb[callDrbIdx];
        if (ptDi->usage != CI_use) continue;

        if (ptDi->markedToBeReleasedDrbByF1.toBeReleased == VCCB_FLAG_ON)
        {
            // Clear markedToBeReleasedDrb. It's serving cell is changed.
            memset(reinterpret_cast<uint8_t *>(&ptDi->markedToBeReleasedDrbByF1), 0x00, sizeof(preemptionInfo_type));
            ptDi->preemptionReqCellNum = 0xFFFF;
        }
        
        if (ptDi->markedToBeReleasedDrbByE1.toBeReleased == VCCB_FLAG_ON)
        {
            memset(reinterpret_cast<uint8_t *>(&ptDi->markedToBeReleasedDrbByE1), 0x00, sizeof(preemptionInfo_type));
        }        
    }

    ptCi->bitFlag.existDrbPreemptionReqForOnlyDup = VCCB_FLAG_OFF;

    return;
}

/**
 * @brief      This function is used to initialize measurementsConfigShared.mGapConfiguration
 * @param      rmGapConfig : m-gap configuration
 * @return     void
 */
void VLIB_InitMGapConfiguration(mGapCallConfiguration_type &rmGapConfig)
{
    AUTO_TRACE_FUNC;

    rmGapConfig.mGapMnCtrlState = MGAPIdle;
    rmGapConfig.mGapLocalCtrlState = MGAPIdle;
    rmGapConfig.mGapCfgSn.fr2.flag = VCCB_FLAG_OFF;
    rmGapConfig.mGapCfgSn.fr1.flag = VCCB_FLAG_OFF;
    rmGapConfig.mGapCfgSn.ue.flag = VCCB_FLAG_OFF;
    rmGapConfig.freqSnInstalled.numItems = 0;
    rmGapConfig.freqSnNew.numItems = 0;
    rmGapConfig.moDuInstalled.numItems = 0;
    rmGapConfig.mGapCfgSnSentFlag = VCCB_FLAG_OFF;
    rmGapConfig.procRxRefuseFlag = VCCB_FLAG_OFF;
    rmGapConfig.endcFr2OnlyFlag = VCCB_FLAG_OFF;
    rmGapConfig.mgrpPreference = MGRP_PREFERENCE_DISABLED;
    rmGapConfig.mrPurposeInfo = MrPurposeInfo_null;

    return;
}

/**
 * @brief      This function is used to initialize mGapDuData_type
 * @param      rmGapDu : m-gap DU data
 * @return     void
 */
void VLIB_InitMGapDuData(mGapDuData_type &rmGapDu)
{
    AUTO_TRACE_FUNC;

    rmGapDu.moDuPreparedFlag = VCCB_FLAG_OFF;
    rmGapDu.moDuSentFlag = VCCB_FLAG_OFF;
    rmGapDu.mgrpPreference = MGRP_PREFERENCE_DISABLED;
    rmGapDu.mrPurposeInfo = MrPurposeInfo_null;
    rmGapDu.moDuNew.numItems = 0;
    rmGapDu.mGapCfgNew.fr2.flag = VCCB_FLAG_OFF;
    rmGapDu.mGapCfgNew.fr1.flag = VCCB_FLAG_OFF;
    rmGapDu.mGapCfgNew.ue.flag = VCCB_FLAG_OFF;

    return;
}

/**
 * @brief      This function is used to initialize measurementsConfigShared.pa4m
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitMeasConfigPeriodicMeasurement(measurementsConfigShared_type::s_PeriodicMeasurement_type *ptPm)
{
    VALID_E_POINT_CHECK_VOID(ptPm);
    ptPm->state = PeriodicMeasState_Idle;
    ptPm->needRestartTimerFlag = VCCB_FLAG_OFF;
}

/**
 * @brief      This function is used to initialize measurementsConfigShared.altgap
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitMeasConfigAltgap(measurementsConfigShared_type::s_AlternateMeasGap_type *pAltGap)
{
    VALID_E_POINT_CHECK_VOID(pAltGap);
    pAltGap->stateWithinPa4m = AltGapStateWithinPa4m_Idle;
    pAltGap->stateWithoutPa4m = AltGapStateWithoutPa4m_Idle;
    pAltGap->prev_stateWithoutPa4m = AltGapStateWithoutPa4m_Idle;
}

/**
 ** @brief      This function is used to initialize
 **             mrCount and measConfigCount only
 **             after sending response to VCSB
 ** @param      void
 ** @return     void
 **/
void VLIB_InitFacMeasConfig(void)
{
    uint32_t cellNum;
    uint16_t index;
    VccbCellInfo_type *pCell = NULL;

    for (cellNum = 0; cellNum < MAX_CELL_NUM_VCCB; cellNum++)
    {
        pCell = &CELLINFO[cellNum];
        if (pCell->status == N_EQUIP) continue;

        for (index = 0; MAX_MEAS_OBJECT_ENTRIES > index; ++index)
        {
            if (pCell->facMeasResult[index].ssbFreq == UNKNOWN_ARFCN) continue;
            pCell->facMeasResult[index].measConfigCount = 0;
            pCell->facMeasResult[index].mrCount = 0;
        }
    }
}

void VLIB_InitMeasurementsConfiguration(measurementsConfiguration_type &rMeasConf)
{
    rMeasConf.tgtCellMoId = 0;
    rMeasConf.tgtCellMoSsb = UNKNOWN_ARFCN;
    rMeasConf.svcGrpChangeProcedure = FALSE;
    rMeasConf.svcGrpId = 0;
    rMeasConf.svcGrpNewId = 0;
    rMeasConf.mGapCfgInstalled.fr2.flag = VCCB_FLAG_OFF;
    rMeasConf.mGapCfgInstalled.fr1.flag = VCCB_FLAG_OFF;
    rMeasConf.mGapCfgInstalled.ue.flag = VCCB_FLAG_OFF;
    rMeasConf.lpmPurposeMask = 0;
    rMeasConf.sMeasureConfig.installedItem.usage = CI_no_use;

    rMeasConf.quantityConfigurations.installedItem.eutra.usage = CI_no_use;

    uint8_t i;
    for (i = 0; i < MAX_NR_OF_QUANTITY_CONFIG; ++i)
    {
        rMeasConf.quantityConfigurations.installedItem.nr[i].usage = CI_no_use;
    }

    for (i = 0; i < VCCB_ARRAY_COUNT(rMeasConf.measurementObjects.installedItems); ++i)
    {
        rMeasConf.measurementObjects.installedItems[i].usage = CI_no_use;
    }

    for (i = 0; i < MAX_NR_OF_REPORT_CONFIG_ID; ++i)
    {
        rMeasConf.reportConfigurations.installedItems[i].usage = CI_no_use;
    }

    for (i = 0; i < MAX_NR_OF_MEAS_ID; ++i)
    {
        rMeasConf.measurementIdentities.installedItems[i].usage = CI_no_use;
    }

    return;
}

void VLIB_InitMeasurementsProcedure(measurementsProcedure_type &rMeasProc)
{
    rMeasProc.tgtCellNum = MAX_CELL_NUM_VCCB;
    rMeasProc.tgtCellNewMoIdx = MAX_NR_OF_OBJECT_ID;
    rMeasProc.measConfigReadyFlag = FLAG_OFF;
    rMeasProc.pa4mAggregatedPerMiCnt = 0;
    rMeasProc.lpmPurposeMask = 0;
    rMeasProc.lateFreqPrioMeas = FLAG_OFF;
    rMeasProc.lateLbMeas = FLAG_OFF;
    rMeasProc.caAwareMeasNeeded = FLAG_OFF;

    return;
}

void VLIB_InitMeasurementsConfigShared(measurementsConfigShared_type &rMeasConfShared)
{
    VCCB_CLEAR_TIMER(&rMeasConfShared.pa4m.timer, CANCEL_TIMER);
    VCCB_CLEAR_TIMER(&rMeasConfShared.lpm.timer, CANCEL_TIMER);
    VCCB_CLEAR_TIMER(&rMeasConfShared.altgap.timer, CANCEL_TIMER);

    VLIB_InitMGapConfiguration(rMeasConfShared.mGapConfiguration);
    VLIB_InitMGapDuData(rMeasConfShared.mGapDu);
    VLIB_InitMeasConfigPeriodicMeasurement(&rMeasConfShared.pa4m);
    VLIB_InitMeasConfigPeriodicMeasurement(&rMeasConfShared.lpm);
    VLIB_InitMeasConfigAltgap(&rMeasConfShared.altgap);

    rMeasConfShared.ltmMeasNeeded = FALSE;
    rMeasConfShared.queuedMeasReconfigAction = MeasReconfigPriorAction_None;
    rMeasConfShared.lpmRepetitionCount = 0;
    return;
}

/**
 * @brief      This function is used to initialize measurements structs.
 * @param      ptCi : call context
 * @return     void
 */
void VLIB_InitMeasurementsStructs(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->endcFr2MeasTimer, CANCEL_TIMER);

    VLIB_InitMeasurementsConfiguration(ptCi->measurementsConfiguration);
    VLIB_InitMeasurementsProcedure(ptCi->measurementsProcedure);
    VLIB_InitMeasurementsConfigShared(ptCi->measurementsConfigShared);

    return;
}

void VLIB_SgnbAdd_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    uint16_t f1CauseEnd = CauseF1apMax;
    uint16_t e1CauseEnd = CauseE1apMax;
    uint16_t x2CauseEnd = CauseX2apMax;

    int famIdx = GET_FAMIDX(STAT_SGNB_ADD_PER_GNB);
    int famIdxSetupCU = GET_FAMIDX(STAT_F1AP_UE_SETUP_CU);
    int famIdxModCU = GET_FAMIDX(STAT_F1AP_CU_INIT_UE_MOD_CU);

    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdxSetupCU, 0, MAX_VCCB_FAMILY_TYPE - 1);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdxModCU, 0, MAX_VCCB_FAMILY_TYPE - 1);

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddFail_DuFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddFail_UpFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < x2CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddFail_MenbFail, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddFail_CpFail, ptCi, 0, 0, 0, 0, 0);
        }
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddPrepFail_DuFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < x2CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddPrepFail_MenbFail, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_SGNB_ADD_PER_GNB, EndcAddPrepFail_CpFail, ptCi, 0, 0, 0, 0, 0);
        }
    }

    if (GET_CALL_STAT_FLAG(ptCi, famIdxSetupCU) == ATTEMPT_MADE)
    {
        MACRO_STAT_PMCC(STAT_F1AP_UE_SETUP_CU, CU_F1UeSetupFail_Others, ptCi, 0, 0, 0, 0, 0);
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdxModCU) == ATTEMPT_MADE)
    {
        if (GET_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag) && GET_F1_CTXT_MODRSP_TGT_FLAG(ptCi->flag.respFlag))
        {
            // Statistics should be counted twice on source and target DU.
            MACRO_STAT_PMCC(STAT_F1AP_CU_INIT_UE_MOD_CU, CU_F1CuInitUeModFail_Others, ptCi, 0, 0, 0, 0, 0);
            OFF_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

            // To count STAT_F1AP_CU_INIT_UE_MOD_CU success or fail when other DU sends F1 ModRsp.
            SET_CALL_STAT_FLAG(ptCi, GET_FAMIDX(STAT_F1AP_CU_INIT_UE_MOD_CU), ATTEMPT_MADE);
        }
        MACRO_STAT_PMCC(STAT_F1AP_CU_INIT_UE_MOD_CU, CU_F1CuInitUeModFail_Others, ptCi, 0, 0, 0, 0, 0);
    }
}

void VLIB_SgnbAddSgnbChg_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_SGNB_ADD_SGNB_CHG);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgFail_CpFail);
        }
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgPrepFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgPrepFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgPrepFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddSgnbChg(*ptCi, EndcAddSgnbChgPrepFail_CpFail);
        }
    }
}
void VLIB_SgnbAddMnHoInterEnb_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_SGNB_ADD_MNHO_INTER_ENB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbFail_CpFail);
        }
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbPrepFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbPrepFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbPrepFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddMnHoInterEnb(*ptCi, EndcAddMnHoInterEnbPrepFail_CpFail);
        }
    }
}
void VLIB_SgnbAddMnHoIntraEnb_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_SGNB_ADD_MNHO_INTRA_ENB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbFail_CpFail);
        }
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < CauseF1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbPrepFail_DuFail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < CauseE1apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbPrepFail_UpFail);
        }
        else if ((cause == CauseCpInternalNsaX2UEContextRelease) ||
                 (cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < CauseX2apMax))
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbPrepFail_MenbFail);
        }
        else
        {
            VCCB_NSA_PegSgnbAddMnHoIntraEnb(*ptCi, EndcAddMnHoIntraEnbPrepFail_CpFail);
        }
    }
}
void VLIB_MenbInitSgnbMod_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_MENB_INIT_SGNB_MOD_PER_GNB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    uint16_t f1CauseEnd = CauseF1apMax;
    uint16_t e1CauseEnd = CauseE1apMax;
    uint16_t x2CauseEnd = CauseX2apMax;

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_DuFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < x2CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_MenbFail, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_CpFail, ptCi, 0, 0, 0, 0, 0);
        }
    }
    else if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepFail_DuFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < x2CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepFail_MenbFail, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepFail_CpFail, ptCi, 0, 0, 0, 0, 0);
        }
    }
}

void VLIB_SgnbInitSgnbMod_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_SGNB_INIT_SGNB_MOD_PER_GNB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    uint16_t f1CauseEnd = CauseF1apMax;
    uint16_t e1CauseEnd = CauseE1apMax;
    uint16_t x2CauseEnd = CauseX2apMax;

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModBySgnbFail_DuFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModBySgnbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
        }
        else if ((cause == CauseCpInternalNsaX2SctpOutOfService) ||
                 (cause >= CAUSE_BASE_X2AP_RADIONETWORK && cause < x2CauseEnd))
        {
            MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModBySgnbFail_MenbFail, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModBySgnbFail_CpFail, ptCi, 0, 0, 0, 0, 0);
        }
    }
}

/**
 * @brief           Increment ErabRel by Gnb Stats based on Gnb Mode and CRR
 * @param[in/out]   ptCi : call context / callInfo_type
 * @param[in/out]   ptErabInfo : ERAB context / callERABInfo_type
 * @return          void
 */
void VLIB_IncrementReleaseErabStatsByGNB(callInfo_type *ptCi, callERABInfo_type *ptErabInfo)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(ptErabInfo);

    uint8_t procForStat = GET_PROC_FOR_STAT(ptCi);
    uint16_t faultCauseType = ptCi->cause.callFault.callFaultCause;

    if (RESET_BY_GNB_PROC == procForStat)
    {
        MACRO_STAT_PMCC_QCI(
            STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_X2Reset, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        return;
    }
    else if (INTRA_SN_MOB_BY_GNB_PROC == procForStat)
    {
        MACRO_STAT_PMCC_QCI(
            STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnbQci_IntraSNMobility, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        return;
    }

    if ((CAUSE_NO_FAULT == faultCauseType) && (CauseF1apRadioReleaseDueToPreEmption == ptCi->cause.f1Cause))
    {
        faultCauseType = CauseF1apRadioReleaseDueToPreEmption;
    }
    else if ((CAUSE_NO_FAULT == faultCauseType) &&
             (CauseE1apRadioNetworkLayerReleaseDueToPreEmption == ptErabInfo->cause.e1Cause))
    {
        faultCauseType = CauseE1apRadioNetworkLayerReleaseDueToPreEmption;
    }

    switch (faultCauseType)
    {
        case CauseCpInternalScgMobility:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnbQci_InterSNMobility, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalUserInactivity:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnbQci_UserInactivity, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalMeasurementReportBasedRelease:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnb_MR, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseF1apRadioReleaseDueToPreEmption:
        case CauseE1apRadioNetworkLayerReleaseDueToPreEmption:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnb_Preemption, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaX2SctpOutOfService:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_X2LinkFail, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaT310Expiry:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnb_DL_t310Expiry, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaSynchReconfigFailureScg:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_DL_syncReconfFailure, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRandomAccessProblem:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_DL_randomAccessProblem, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRandomAccessProblemAfterInitialAccess:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN,
                ErabRelAbnormalbyGnbQci_DL_randomAccessProblem_AfterInitialAccess, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalDuRlcMaxNumRetx:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_DL_rlcMaxNumRetx, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaSrb3IntegrityFailure:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_DL_srb3IntegrityFailure, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaScgReconfigFailure:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_DL_scgReconfFailure, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalDuULRadioLinkFailure:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_UL_outOfSync, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRlcMaxNumRetx:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_UL_rlcMaxNumRetx, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalCpSrbIntegrityFailure:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQci_UL_srb3IntegrityFailure, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalOMIntervention:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnb_SwitchOffOngoing, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        default:
            break;
    }

    int16_t CrrClassification = VIIL_CrrClassification(faultCauseType);

    if (ABNORMAL_CRR == CrrClassification || SEMINORMAL_CRR == CrrClassification)
    {
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
    }
    else
    {
        // NORMAL_CRR and UNKNOWN_CRR handling.
        // If CRR is unknown or CAUSE_NO_FAULT(0), collect normal counters too.
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
    }

    MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyGnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
}

/**
 * @brief           Increment EndcCtxRel by Gnb Stats based on Gnb Mode and CRR
 * @param[in/out]   ptCi : call context / callInfo_type
 * @return          void
 */
void VLIB_IncrementReleaseCtxtStatsByGNB(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint8_t procForStat = GET_PROC_FOR_STAT(ptCi);
    uint16_t faultCauseType = ptCi->cause.callFault.callFaultCause;

    if (RESET_BY_GNB_PROC == procForStat)
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_X2Reset, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnbTot, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelbyGnbTot, ptCi, 0, 0, 0, 0, 0);
        return;
    }
    else if (INTRA_SN_MOB_BY_GNB_PROC == procForStat)
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_IntraSNMobility, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnbTot, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelbyGnbTot, ptCi, 0, 0, 0, 0, 0);
        return;
    }

    if ((CAUSE_NO_FAULT == faultCauseType) && (CauseF1apRadioReleaseDueToPreEmption == ptCi->cause.f1Cause))
    {
        faultCauseType = CauseF1apRadioReleaseDueToPreEmption;
    }

    // EndcCtx not released when EN-DC addition failed on SpCellSelection procedure.
    if (TRUE == ptCi->bitFlag.selectSpCellError)
    {
        return;
    }

    switch (faultCauseType)
    {
        case CauseCpInternalScgMobility:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_InterSNMobility, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalUserInactivity:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_UserInactivity, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalMeasurementReportBasedRelease:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_MR, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseF1apRadioReleaseDueToPreEmption:
        case CauseE1apRadioNetworkLayerReleaseDueToPreEmption:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_Preemption, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaX2SctpOutOfService:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_X2LinkFail, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaT310Expiry:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_t310Expiry, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaSynchReconfigFailureScg:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_syncReconfFailure, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRandomAccessProblem:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_randomAccessProblem, ptCi, 0,
                0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRandomAccessProblemAfterInitialAccess:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN,
                EndcCtxRelAbnormalbyGnb_DL_randomAccessProblem_AfterInitialAccess, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalDuRlcMaxNumRetx:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_rlcMaxNumRetx, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaSrb3IntegrityFailure:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_srb3IntegrityFailure, ptCi,
                0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaScgReconfigFailure:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_DL_scgReconfFailure, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalDuULRadioLinkFailure:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_UL_outOfSync, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalNsaRlcMaxNumRetx:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_UL_rlcMaxNumRetx, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalCpSrbIntegrityFailure:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnb_UL_srb3IntegrityFailure, ptCi,
                0, 0, 0, 0, 0);
            break;
        }
        case CauseCpInternalOMIntervention:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnb_SwitchOffOngoing, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        default:
            break;
    }

    int16_t CrrClassification = VIIL_CrrClassification(faultCauseType);

    if ((ABNORMAL_CRR == CrrClassification || SEMINORMAL_CRR == CrrClassification))
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyGnbTot, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        // NORMAL_CRR and UNKNOWN_CRR handling.
        // If CRR is unknown or CAUSE_NO_FAULT(0), collect normal counters too.
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyGnbTot, ptCi, 0, 0, 0, 0, 0);
    }

    MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelbyGnbTot, ptCi, 0, 0, 0, 0, 0);
    ptCi->bitFlag.selectSpCellError = FALSE;
}

/**
 * @brief           Increment ErabRel by Enb Stats based on Enb Mode and CRR x2 cause
 * @param[in/out]   ptCi : call context / callInfo_type
 * @param[in/out]   ptErabInfo : ERAB context / callERABInfo_type
 * @return          void
 */
void VLIB_IncrementReleaseErabStatsByENB(callInfo_type *ptCi, callERABInfo_type *ptErabInfo)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(ptErabInfo);

    uint8_t procForStat = GET_PROC_FOR_STAT(ptCi);
    uint16_t faultCauseType = ptCi->cause.callFault.callFaultCause;
    int16_t CrrClassification = VIIL_CrrClassification(faultCauseType);

    if (RESET_BY_ENB_PROC == procForStat)
    {
        MACRO_STAT_PMCC_QCI(
            STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQci_X2Reset, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        return;
    }

    if (INTRA_SN_MOB_BY_ENB_PROC == procForStat)
    {
        // For interEnb HO with E-RAB removing do not collect ErabRelNormalbyEnbQci_IntraSNMobility counter
        if (ERAB_Release != ptErabInfo->erabStatus)
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnbQci_IntraSNMobility, ptCi, ptErabInfo, 0, 0, 0, 0);
        }
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
        return;
    }

    switch (faultCauseType)
    {
        case CauseX2apRadioNetworkLayerMcgMobility:
        {
            if (FALSE == ptCi->RelCtxtKeptInd)
            {
                MACRO_STAT_PMCC_QCI(
                    STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnbQci_McgMobility, ptCi, ptErabInfo, 0, 0, 0, 0);
            }
            break;
        }
        case CauseX2apRadioNetworkLayerUserInactivity:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnbQci_UserInactivity, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerFailureInTheRadioInterfaceProcedure:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnb_FailureintheRadioInterfaceProcedure,
                ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerTimeCriticalAction:
        {
            MACRO_STAT_PMCC_QCI(
                STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnb_TimeCriticalAction, ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerRadioConnectionWithUeLost:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQci_RadioConnectionWithUeLost, ptCi,
                ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerTdcprepExpiry:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQci_RadioNetworkLayerTdcprepExpiry,
                ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerUnspecified:
        {
            MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQci_RadioNetworkLayerUnspecified,
                ptCi, ptErabInfo, 0, 0, 0, 0);
            break;
        }
        default:
            break;
    }

    if (ABNORMAL_CRR == CrrClassification || SEMINORMAL_CRR == CrrClassification)
    {
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelAbnormalbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
    }
    else
    {
        // NORMAL_CRR and UNKNOWN_CRR handling.
        // If CRR is unknown or CAUSE_NO_FAULT(0), collect normal counters too.
        MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelNormalbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
    }

    MACRO_STAT_PMCC_QCI(STAT_ERAB_REL_ALL_PER_GNB_PLMN, ErabRelbyEnbQciTot, ptCi, ptErabInfo, 0, 0, 0, 0);
}

/**
 * @brief           Increment EndcCtxRel by Enb Stats based on CRR x2 cause
 * @param[in/out]   ptCi : call context / callInfo_type
 * @return          void
 */
void VLIB_IncrementReleaseCtxtStatsByENB(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t faultCauseType = ptCi->cause.callFault.callFaultCause;
    uint8_t procForStat = GET_PROC_FOR_STAT(ptCi);
    int16_t CrrClassification = VIIL_CrrClassification(faultCauseType);

    if (RESET_BY_ENB_PROC == procForStat)
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnb_X2Reset, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnbTot, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelRelbyEnbTot, ptCi, 0, 0, 0, 0, 0);
        return;
    }

    if (INTRA_SN_MOB_BY_ENB_PROC == procForStat)
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnb_IntraSNMobility, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnbTot, ptCi, 0, 0, 0, 0, 0);
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelRelbyEnbTot, ptCi, 0, 0, 0, 0, 0);
        return;
    }

    switch (faultCauseType)
    {
        case CauseX2apRadioNetworkLayerMcgMobility:
        {
            if (FALSE == ptCi->RelCtxtKeptInd)
            {
                MACRO_STAT_PMCC(
                    STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnb_McgMobility, ptCi, 0, 0, 0, 0, 0);
            }
            break;
        }
        case CauseX2apRadioNetworkLayerUserInactivity:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnb_UserInactivity, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerFailureInTheRadioInterfaceProcedure:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN,
                EndcCtxRelNormalbyEnb_FailureintheRadioInterfaceProcedure, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerTimeCriticalAction:
        {
            MACRO_STAT_PMCC(
                STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnb_TimeCriticalAction, ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerRadioConnectionWithUeLost:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnb_RadioConnectionWithUeLost, ptCi,
                0, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerTdcprepExpiry:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnb_RadioNetworkLayerTdcprepExpiry,
                ptCi, 0, 0, 0, 0, 0);
            break;
        }
        case CauseX2apRadioNetworkLayerUnspecified:
        {
            MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnb_RadioNetworkLayerUnspecified,
                ptCi, 0, 0, 0, 0, 0);
            break;
        }
        default:
            break;
    }

    if (ABNORMAL_CRR == CrrClassification || SEMINORMAL_CRR == CrrClassification)
    {
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelAbnormalbyEnbTot, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        // NORMAL_CRR and UNKNOWN_CRR handling.
        // If CRR is unknown or CAUSE_NO_FAULT(0), collect normal counters too.
        MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelNormalbyEnbTot, ptCi, 0, 0, 0, 0, 0);
    }

    MACRO_STAT_PMCC(STAT_ENDC_CTX_REL_ALL_PER_GNB_PLMN, EndcCtxRelRelbyEnbTot, ptCi, 0, 0, 0, 0, 0);
}

/**
 * @brief           Classification of release EN-DC context and ERAB release procedure for collect Stats by Enb / by gNB
 * @param[in/out]   ptCi : call context / callInfo_type
 * @param[in/out]   erabIncrement : flag / uint8_t
 * @param[in/out]   ptErabInfo : ERAB context / callERABInfo_type
 * @return          void
 */
void VLIB_IncrementReleaseAllStats(callInfo_type *ptCi, uint8_t erabIncrement, callERABInfo_type *ptErabInfo)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    if (TRUE == erabIncrement) VALID_E_POINT_CHECK_VOID(ptErabInfo);
    uint8_t procForStat = GET_PROC_FOR_STAT(ptCi);

    if ((MOD_BY_ENB_PROC == procForStat) || (RESET_BY_ENB_PROC == procForStat) ||
        (RELEASE_BY_ENB_PROC == procForStat) || (INTRA_SN_MOB_BY_ENB_PROC == procForStat))
    {
        if (TRUE == erabIncrement)
        {
            VLIB_IncrementReleaseErabStatsByENB(ptCi, ptErabInfo);
        }
        else
        {
            VLIB_IncrementReleaseCtxtStatsByENB(ptCi);
        }
    }
    else
    {
        if (TRUE == erabIncrement)
        {
            VLIB_IncrementReleaseErabStatsByGNB(ptCi, ptErabInfo);
        }
        else
        {
            VLIB_IncrementReleaseCtxtStatsByGNB(ptCi);
        }
    }
}

void VLIB_DrbSetup_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    uint16_t cause = ptCi->cause.callFault.callFaultCause;

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        callDRBInfo_type *ptDi = &ptCi->drb[callDrbIdx];
        if (ptDi->usage != CI_use) continue;
        MACRO_STAT_COLLECT_ALL_DRB_SETUP_WITH_CAUSE(ptCi, ptDi, callDrbIdx, cause);
    }
}

void VLIB_DrbSetup_CollectStat_FailedDrbItem(callInfo_type *ptCi, uint16_t cause, uint8_t callDrbIdx, int32_t mFamily)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VCCB_CHECK_E_RANGE_AND_VOID(callDrbIdx, MAX_DRB_PER_SA_CALL);

    uint16_t f1CauseEnd, e1CauseEnd, ngCauseEnd;

    f1CauseEnd = CauseF1apMax;
    e1CauseEnd = CauseE1apMax;
    ngCauseEnd = CauseNgapMax;

    // Check slice Fail and DuNotSupportVonr5qi
    int additionalOption     = 0;
    if(ptCi->drb[callDrbIdx].rxDuNotSupportVoNr5qi)
    {
        additionalOption |= DRB_STAT_ADDITIONAL_DuVoice5QINotSupport;
    }
    if(ptCi->drb[callDrbIdx].rxNotSupportSlice)
    {
        additionalOption |= DRB_STAT_ADDITIONAL_RxNotSupportSlice;
    }
    if(ptCi->drb[callDrbIdx].isLteFallbackTriggered)
    {
        additionalOption |= DRB_STAT_ADDITIONAL_LteFallbackTriggered;
    }
    int mType = DrbSetupFail_CpFail;

    if ((cause == CauseCpInternalF1SctpOutOfService) || (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
    {
        mType = DrbSetupFail_F1Fail;
    }
    else if ((cause == CauseCpInternalE1SctpOutOfService) ||
             (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
    {
        mType = DrbSetupFail_E1Fail;
    }
    else if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
             (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
    {
        mType = DrbSetupFail_NgFail;
    }
    else if ((cause == CauseCpInternalSaTmOutRrcUeCapabilityEnquiry) ||
             (cause == CauseCpInternalSaTmOutRrcSecurityModeCommand) ||
             (cause == CauseCpInternalTmOutRrcConnectionReconfig))
    {
        mType = DrbSetupFail_RrcTimeout;
    }
    else if ((cause == CauseCpInternalTmOutE1BearerSetup) || (cause == CauseCpInternalTmOutE1BearerContextModification))
    {
        mType = DrbSetupFail_E1Timeout;
    }
    else if ((cause == CauseCpInternalTmOutF1UeContextSetup) || (cause == CauseCpInternalTmOutF1UeContextModification))
    {
        mType = DrbSetupFail_F1Timeout;
    }
    else
    {
        mType = DrbSetupFail_CpFail;
    }

    MACRO_STAT_PMCC_DRB(mFamily, mType, ptCi, &ptCi->drb[callDrbIdx], additionalOption, 0, 0, 0);
}

void VLIB_DrbRelErase_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        if (ptCi->drb[callDrbIdx].drbStatistic.attemptFlag[DRB_RELEASE_AND_ERASE] != ATTEMPT_MADE) continue;

        VLIB_DrbRelErase_CollectStat_FailedDrbItem(ptCi, cause, callDrbIdx);
    }
}

void VLIB_DrbRelErase_CollectStat_FailedDrbItem(callInfo_type *ptCi, uint16_t cause, uint8_t callDrbIdx)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VCCB_CHECK_E_RANGE_AND_VOID(callDrbIdx, MAX_DRB_PER_SA_CALL);

    uint16_t f1CauseEnd, e1CauseEnd, ngCauseEnd;

    f1CauseEnd = CauseF1apMax;
    e1CauseEnd = CauseE1apMax;
    ngCauseEnd = CauseNgapMax;

    if ((cause == CauseCpInternalF1SctpOutOfService) || (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_F1Fail, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else if ((cause == CauseCpInternalE1SctpOutOfService) ||
             (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_E1Fail, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
             (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_NgFail, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else if ((cause == CauseCpInternalSaTmOutRrcUeCapabilityEnquiry) ||
             (cause == CauseCpInternalSaTmOutRrcSecurityModeCommand) ||
             (cause == CauseCpInternalTmOutRrcConnectionReconfig))
    {
        MACRO_STAT_PMCC_DRB(
            STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_RrcTimeout, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else if ((cause == CauseCpInternalTmOutE1BearerSetup) || (cause == CauseCpInternalTmOutE1BearerContextModification))
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_E1Timeout, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else if ((cause == CauseCpInternalTmOutF1UeContextSetup) || (cause == CauseCpInternalTmOutF1UeContextModification))
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_F1Timeout, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
    else
    {
        MACRO_STAT_PMCC_DRB(STAT_DRB_RELEASE_AND_ERASE, DrbRelFail_CpFail, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
    }
}

void VLIB_NeDcDrb_CollectStat_FailedDrbItem(callInfo_type *ptCi, uint16_t mFamily, uint16_t mType, uint8_t callDrbIdx)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    uint16_t mFamilyIdx = DRB_SETUP;

    switch (mFamily)
    {
        case STAT_NEDC_DRB_SETUP:
            mFamilyIdx = NEDC_DRB_SETUP;
            break;
        // current assumption is SN originated modification is not supported
        case STAT_NEDC_DRB_MOD:
            mFamilyIdx = NEDC_DRB_MOD;
            break;

        default:
            break;
    }

    if (callDrbIdx >= MAX_DRB_PER_SA_CALL)
    {
        return;
    }

    if (ptCi->drb[callDrbIdx].drbStatistic.attemptFlag[mFamilyIdx] != ATTEMPT_MADE)
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "VLIB_NeDcDrb_CollectStat_FailedDrbItem mFamily %d no attempt was made !\n", mFamily);
        return;
    }

    MACRO_STAT_PMCC_DRB(mFamily, mType, ptCi, &ptCi->drb[callDrbIdx], 0, 0, 0, 0);
}

void VLIB_NeDcDrb_CollectStat_UpFail(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        if (S_Setup == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_SETUP, NedcDrbSetupFail_UpFail, callDrbIdx);
        }
        else if (S_Modify == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_MOD, NedcDrbModByMgnbFail_UpFail, callDrbIdx);
        }
    }
}

void VLIB_NeDcDrb_CollectStat_UpTimeout(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        if (S_Setup == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_SETUP, NedcDrbSetupFail_UpTimeout, callDrbIdx);
        }
        else if (S_Modify == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_MOD, NedcDrbModByMgnbFail_UpTimeout, callDrbIdx);
        }
    }
}

void VLIB_NeDcDrb_CollectStat_SenbFail(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        if (S_Setup == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_SETUP, NedcDrbSetupFail_SenbFail, callDrbIdx);
        }
        else if (S_Modify == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_MOD, NedcDrbModByMgnbFail_SenbFail, callDrbIdx);
        }
    }
}

void VLIB_NeDcDrb_CollectStat_CpFail(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    // attempt has been counted but not success/fail
    uint8_t callDrbIdx = 0;
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_CALL; callDrbIdx++)
    {
        if (S_Setup == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_SETUP, NedcDrbSetupFail_CpFail, callDrbIdx);
        }
        else if (S_Modify == ptCi->drb[callDrbIdx].snDrbStatus)
        {
            VLIB_NeDcDrb_CollectStat_FailedDrbItem(ptCi, STAT_NEDC_DRB_MOD, NedcDrbModByMgnbFail_CpFail, callDrbIdx);
        }
    }
}

void VLIB_PduSessionResourceSetup_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;

    // attempt has been counted but not success/fail
    uint8_t callPduIdx = 0;
    for (callPduIdx = 0; callPduIdx < MAX_PDU_SESS_PER_CALL; callPduIdx++)
    {
        if (ptCi->pduSess[callPduIdx].usage != CI_use) continue;

        if (ptCi->pduSess[callPduIdx].pduStatistic.attemptFlag[PDU_SESSION_RESOURCE_SETUP_PLMN_SNSSAI] == ATTEMPT_MADE)
        {
            VLIB_PduSessionResourceSetup_CollectStat_FailedPduItem(
                ptCi, cause, callPduIdx, STAT_PDU_SESSION_RESOURCE_SETUP_PLMN_SNSSAI);
        }
        if (ptCi->pduSess[callPduIdx].pduStatistic.attemptFlag[PDU_SESSION_RESOURCE_MODIFY_PLMN_SNSSAI_RFSP] == ATTEMPT_MADE)
        {
            VLIB_PduSessionResourceSetup_CollectStat_FailedPduItem(
                ptCi, cause, callPduIdx, STAT_PDU_SESSION_RESOURCE_MODIFY_PLMN_SNSSAI_RFSP);
        }
    }
}

void VLIB_PduSessionResourceSetup_CollectStat_FailedPduItem(
    callInfo_type *ptCi, uint16_t cause, uint8_t callPduIdx, int32_t mFamily)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VCCB_CHECK_E_RANGE_AND_VOID(callPduIdx, MAX_PDU_SESS_PER_CALL);

    uint16_t f1CauseEnd, e1CauseEnd, ngCauseEnd;

    int sliceFail = ptCi->pduSess[callPduIdx].rxNotSupportSlice;
    int mType     = PduSetupPlmnSnssaiFail_CpFail;

    f1CauseEnd = CauseF1apMax;
    e1CauseEnd = CauseE1apMax;
    ngCauseEnd = CauseNgapMax;

    if ((cause == CauseCpInternalF1SctpOutOfService) || (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
    {
        mType = PduSetupPlmnSnssaiFail_F1Fail;
    }
    else if ((cause == CauseCpInternalE1SctpOutOfService) ||
             (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
    {
        mType = PduSetupPlmnSnssaiFail_E1Fail;
    }
    else if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
             (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
    {
        mType = PduSetupPlmnSnssaiFail_NgFail;
    }
    else if ((cause == CauseCpInternalSaTmOutRrcUeCapabilityEnquiry) ||
             (cause == CauseCpInternalSaTmOutRrcSecurityModeCommand) ||
             (cause == CauseCpInternalTmOutRrcConnectionReconfig))
    {
        mType = PduSetupPlmnSnssaiFail_RrcTimeout;
    }
    else if ((cause == CauseCpInternalTmOutE1BearerSetup) || (cause == CauseCpInternalTmOutE1BearerContextModification))
    {
        mType = PduSetupPlmnSnssaiFail_E1Timeout;
    }
    else if ((cause == CauseCpInternalTmOutF1UeContextSetup) || (cause == CauseCpInternalTmOutF1UeContextModification))
    {
        mType = PduSetupPlmnSnssaiFail_F1Timeout;
    }
    else
    {
        mType = PduSetupPlmnSnssaiFail_CpFail;
    }

    MACRO_STAT_PMCC_PDU(mFamily, mType, ptCi, &ptCi->pduSess[callPduIdx], sliceFail, 0, 0, 0);
}

void VLIB_RrcConnEstab_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    int famIdx = GET_FAMIDX(STAT_RRC_ESTAB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        MACRO_STAT_PMCC(STAT_RRC_ESTAB, ConnEstabFail_CpFail, ptCi, 0, 0, 0, 0, 0);
    }
}

void VLIB_NgConnEstab_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;

    int famIdx = GET_FAMIDX(STAT_NG_ESTAB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    uint16_t f1CauseEnd = CauseF1apMax;
    uint16_t e1CauseEnd = CauseE1apMax;
    uint16_t ngCauseEnd = CauseNgapMax;

    // attempt has been counted but not success/fail
    if (GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE)
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_STAT_NG_ESTAB_GROUP(ptCi, NgConnEstabFail_F1Fail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_STAT_NG_ESTAB_GROUP(ptCi, NgConnEstabFail_E1Fail);
        }
        else if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
                 (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
        {
            MACRO_STAT_NG_ESTAB_GROUP(ptCi, NgConnEstabFail_NgFail);
        }
        else
        {
            MACRO_STAT_NG_ESTAB_GROUP(ptCi, NgConnEstabFail_CpFail);
        }
    }
}

void VLIB_RRE_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    int famIdx = GET_FAMIDX(STAT_RRC_REESTAB);
    VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_VOID(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);

    uint16_t f1CauseEnd = CauseF1apMax;
    uint16_t e1CauseEnd = CauseE1apMax;
    uint16_t ngCauseEnd = CauseNgapMax;

    // attempt has been counted but not success/fail
    if ((GET_CALL_STAT_FLAG(ptCi, famIdx) == ATTEMPT_MADE) || (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE))
    {
        if ((cause == CauseCpInternalF1SctpOutOfService) ||
            (cause >= CAUSE_BASE_F1AP_RADIONETWORK && cause < f1CauseEnd))
        {
            MACRO_RRE_STAT(ptCi, ReEstabFail_F1Fail);
        }
        else if ((cause == CauseCpInternalE1SctpOutOfService) ||
                 (cause >= CAUSE_BASE_E1AP_RADIONETWORK && cause < e1CauseEnd))
        {
            MACRO_RRE_STAT(ptCi, ReEstabFail_E1Fail);
        }
        else if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
                 (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
        {
            MACRO_RRE_STAT(ptCi, ReEstabFail_NgFail);
        }
        else
        {
            MACRO_RRE_STAT(ptCi, ReEstabFail_CpFail);
        }
    }
}

void VLIB_CollectRetainIntraGnbRREStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    uint16_t callDrbIdx = 0;
    callDRBInfo_type *ptDi = NULL;

    uint8_t isOldCell = FALSE;

    if ((ptCi->ueInfo.reestablishInfo.reestablishType == RRE_TYPE_INTRA_DU) ||
        (ptCi->ueInfo.reestablishInfo.reestablishType == RRE_TYPE_INTER_DU))
    {
        isOldCell = TRUE;  // oldCellNum
    }
    else  // ptCi->ueInfo.reestablishInfo.reestablishType == RRE_TYPE_INTER_GNB
    {
        isOldCell = FALSE;  // servedCellNum
    }
    uint32_t isMobility = 1;  // In Mobility Case(ex. HO, RRE), Don't collect total stat(UeContextRelTotal, DrbRelTotal).

    /* UE CONTEXT */
    MACRO_STAT_PMCC_RETAIN(STAT_SETUP_UECONTEXT_RELEASE, ptCi, NULL, ptCi->retainStat.savedEventCategory,
        ptCi->retainStat.savedEventItem, isOldCell, isMobility);

    /* DRB */
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_SA_CALL; callDrbIdx++)
    {
        ptDi = &ptCi->drb[callDrbIdx];

        if (ptDi->usage != CI_use) continue;
        if (ptCi->drb[callDrbIdx].retainStat.capableFlag != VCCB_FLAG_ON) continue;

        MACRO_STAT_PMCC_RETAIN(STAT_SETUP_DRB_RELEASE, ptCi, ptDi, ptDi->retainStat.savedEventCategory,
            ptDi->retainStat.savedEventItem, isOldCell, isMobility);
    }

    /* VoNR & Emergency Call */
    if(ptCi->rrcConnStat.vonrEmerCallFlag == VCCB_FLAG_ON)
    {
        MACRO_STAT_PMCC_RETAIN(STAT_VONR_EMERGENCY_RELEASE, ptCi, NULL, ptCi->retainStatOfVonrEmer.savedEventCategory,
                ptCi->retainStatOfVonrEmer.savedEventItem, isOldCell, isMobility);
    }

    if ((ptCi->ueInfo.reestablishInfo.reestablishType == RRE_TYPE_INTRA_DU) ||
        (ptCi->ueInfo.reestablishInfo.reestablishType == RRE_TYPE_INTER_DU))
    {
        uint32_t collectFailToSetupStat = VCCB_FLAG_ON;
        SET_ON_ALL_RETAIN_STAT_CAPABLE_FLAG(ptCi, collectFailToSetupStat, VCCB_FLAG_OFF);
    }
}

void VLIB_CollectStat_RRC_Release(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;
    uint16_t ngCauseEnd;

    ngCauseEnd = CauseNgapMax;

    if (ptCi->stateCmd == RRCSuspend_cmd)
    {
        MACRO_STAT_PMCC(STAT_RRC_RELEASE, ConnRelease_withSuspendConfig_ByRrcSuspend, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        if ((cause == CauseCpInternalSaNgSctpOutOfService) ||
            (cause >= CAUSE_BASE_NGAP_RADIONETWORK && cause < ngCauseEnd))
        {
            MACRO_STAT_PMCC(STAT_RRC_RELEASE, ConnRelease_ByAmf, ptCi, 0, 0, 0, 0, 0);
        }
        else
        {
            MACRO_STAT_PMCC(STAT_RRC_RELEASE, ConnRelease_ByGnb, ptCi, 0, 0, 0, 0, 0);
        }
    }
}

/**
 * @brief     Function to collect Ue Capability Size Distribution counters
 * @param     ptCi : callInfo
 */
void VLIB_CollectStat_NSA_UeCapaSizeDist(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint8_t base = __UINT8_MAX__;
    uint32_t length = 0;

    if(E_CALLTYPE_NSA == ptCi->callType)
    {
        base = NSANRUESize0_DUID;
        length = FnVCCB_isBigCapabilityRatContainer(ptCi, NRRRC_RAT_Type_nr) ? ptCi->ueContainerNR_Ext.capaLength : ptCi->ueContainerNR.numocts;
        VLIB_CollectStatPerCapaType_NSA_UeCapaSizeDist(ptCi, base, length);

        base = NSAEUTRANNRSize0_DUID;
        length = FnVCCB_isBigCapabilityRatContainer(ptCi, NRRRC_RAT_Type_eutra_nr) ? ptCi->ueContainerMRDC_Ext.capaLength : ptCi->ueContainerMRDC.numocts;
        VLIB_CollectStatPerCapaType_NSA_UeCapaSizeDist(ptCi, base, length);
    }
    else
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VLIB_CollectStat_NSA_UeCapaSizeDist Not supported callType(%i)!\n", ptCi->callType);
    }
}

/**
 * @brief     Function where actual collection of Ue Capability Size Distribution counters takes place
 * @param     ptCi : callInfo
 * @param     base : indicator, which group of counters should be collected
 * @param     length : Ue Capa Length
 */
void VLIB_CollectStatPerCapaType_NSA_UeCapaSizeDist(callInfo_type *ptCi, uint8_t base, uint32_t length)
{
    AUTO_TRACE_FUNC;

    uint8_t counter_type = __UINT8_MAX__;

    //choose which exact counter should be collected
    if(length > UE_CAPA_SIZE_DIST_LAST_CNT_LIMIT)
    {
        counter_type = static_cast<uint8_t>(base + AMOUNT_OF_ENUMS_UE_CAPA_SIZE);
    }
    else if (0 == length)
    {
        counter_type = base;
    }
    else
    {
        counter_type = static_cast<uint8_t>(base + static_cast<uint8_t>((length - 1) / INCREMENT_VALUE_UE_CAPA_SIZE));
    }

    VCCB_CHECK_E_RANGE_AND_VOID(counter_type, __UINT8_MAX__);
    MACRO_STAT_PMCC(STAT_UE_CAPA_SIZE_DIST_DU_ID, counter_type, ptCi, 0, 0, 0, 0, 0);
}

void VLIB_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    uint8_t incrementStatsForErab = FALSE;

    // Add Default Stats for uncategorized calls
    if (ptCi->callType == E_CALLTYPE_NSA)
    {
        VLIB_SgnbAdd_CollectStat_PutCi(ptCi);
        VLIB_SgnbAddSgnbChg_CollectStat_PutCi(ptCi);
        VLIB_SgnbAddMnHoInterEnb_CollectStat_PutCi(ptCi);
        VLIB_SgnbAddMnHoIntraEnb_CollectStat_PutCi(ptCi);
        VLIB_MenbInitSgnbMod_CollectStat_PutCi(ptCi);
        VLIB_SgnbInitSgnbMod_CollectStat_PutCi(ptCi);
        VLIB_IncrementReleaseAllStats(ptCi, incrementStatsForErab, NULL);
        VLIB_UpdateCounters_SuccessMrBasedResyncSCGFailure(ptCi, FALSE);
    }
    else if (ptCi->callType == E_CALLTYPE_SA)
    {
        VLIB_NgConnEstab_CollectStat_PutCi(ptCi);
        VLIB_DrbRelErase_CollectStat_PutCi(ptCi);
        VLIB_PduSessionResourceSetup_CollectStat_PutCi(ptCi);
        VLIB_QosFlowSetup_CollectStat_PutCi(ptCi);
        VLIB_RRE_CollectStat_PutCi(ptCi);
        FnVCCB_CheckStat_NrdcSnAddition(ptCi);
        // FGR-CC5200:FR760 FR770
        FnVCCB_NrdcSnConnInfoStatsCmd(ptCi, ptCi->servCellNum, ConnInfoStatsCmd_HoldingTimeStop);
        FnVCCB_NrdcSnConnInfoStatsCmd(ptCi, ptCi->servCellNum, ConnInfoStatsCmd_Decrease);

        // FGR-CC5200:FR640
        FnVCCB_NrdcCtxRelStat_Collect(ptCi,__func__, SRC_SN);
        //FGR-CC5200 FR690
        FnVCCB_UpdateNrdcModStats(ptCi, NrdcModMn_MnInit_PrepFail, NULL, false);
        FnVCCB_UpdateNrdcModStats(ptCi, NrdcModMn_SnInit_Fail, NULL, false);
        FnVCCB_UpdateNrdcModStats(ptCi, NrdcModSn_MnInit_PrepFail, NULL, false);
        FnVCCB_UpdateNrdcModStats(ptCi, NrdcModSn_SnInit_Fail, NULL, false);

        FnVCCB_ForcedMobilityNr_CollectStat_PutCi(ptCi); // ODF_HO_24A
        FnVCCB_USTM_CollectStat_PutCi(ptCi);

        // Collection of abnormal statistics related to report CGI (NCGI)
        FnVCCB_CgiReportAbnormalStat(ptCi);
    }
}

void VLIB_FillBeamMeasCsl(
    rsIndexResults_type *ptRsIndexResults, cslBeamMeas_type *ptCslBeamMeas, uint16_t numOfBeamToFill, enumRSType rsType)
{
    AUTO_TRACE_FUNC;

    measQuantityResults_type *ptMeasResult = NULL;
    uint8_t beamIdx = 0;

    VALID_E_POINT_CHECK_VOID(ptCslBeamMeas);
    VALID_E_POINT_CHECK_VOID(ptRsIndexResults);

    for (beamIdx = 0; beamIdx < numOfBeamToFill; beamIdx++)
    {
        if (RSType_SSB == rsType &&
            TRUE == ptRsIndexResults->resultsSSB_Indexes.resultsPerSSB_Index[beamIdx].ssbResultsFlag)
        {
            ptMeasResult = &ptRsIndexResults->resultsSSB_Indexes.resultsPerSSB_Index[beamIdx].ssbResults;
            ptCslBeamMeas->beamId = ptRsIndexResults->resultsSSB_Indexes.resultsPerSSB_Index[beamIdx].ssbIndex;
        }
        else if (RSType_CSIRS == rsType &&
                 TRUE == ptRsIndexResults->resultsCSI_RS_Indexes.resultsPerCSI_RS_Index[beamIdx].csi_RS_ResultsFlag)
        {
            ptMeasResult = &ptRsIndexResults->resultsCSI_RS_Indexes.resultsPerCSI_RS_Index[beamIdx].csi_RS_Results;
            ptCslBeamMeas->beamId =
                ptRsIndexResults->resultsCSI_RS_Indexes.resultsPerCSI_RS_Index[beamIdx].csi_RS_Index;
        }
        if (ptMeasResult != NULL)
        {
            if (TRUE == ptMeasResult->rsrpFlag)
            {
                ptCslBeamMeas->rsrp = ptMeasResult->rsrp;
            }
            if (TRUE == ptMeasResult->rsrqFlag)
            {
                ptCslBeamMeas->rsrq = ptMeasResult->rsrq;
            }
            //            if(TRUE == ptMeasResult->sinrFlag)
            //            {
            //                ptCslBeamMeas->sinr = ptMeasResult->sinr;
            //            }
            ptMeasResult = NULL;
            ptCslBeamMeas++;
        }
    }
}
void VLIB_FillCellMeasCsl(cellResults_type *ptCellResults, uint8_t rsType, uint16_t *rsrp, uint16_t *rsrq)
{
    AUTO_TRACE_FUNC;

    measQuantityResults_type *ptMeasResult = NULL;

    VALID_E_POINT_CHECK_VOID(ptCellResults);
    VALID_E_POINT_CHECK_VOID(rsrp);
    VALID_E_POINT_CHECK_VOID(rsrq);

    if (RSType_SSB == rsType)
    {
        ptMeasResult = &ptCellResults->resultsSSB_Cell;
    }
    else if (RSType_CSIRS == rsType)
    {
        ptMeasResult = &ptCellResults->resultsCSI_RS_Cell;
    }

    if (ptMeasResult != NULL)
    {
        if (TRUE == ptMeasResult->rsrpFlag)
        {
            *rsrp = ptMeasResult->rsrp;
        }
        if (TRUE == ptMeasResult->rsrqFlag)
        {
            *rsrq = ptMeasResult->rsrq;
        }
        //        if(TRUE == ptMeasResult->sinrFlag)
        //        {
        //            *sinr = ptMeasResult->sinr;
        //        }
    }
}

int16_t VLIB_getRsType(measResult2NR_type *ptMeasResult2NR, enumRSType *rsType)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptMeasResult2NR);
    VALID_E_POINT_CHECK_ERROR(rsType);

    if (TRUE == ptMeasResult2NR->ssbFrequency_Flag)
    {
        *rsType = RSType_SSB;
    }
    else if (TRUE == ptMeasResult2NR->refFreqCSI_RS_Flag)
    {
        *rsType = RSType_CSIRS;
    }
    else
    {
        VCCB_PRINT_COMMENT("%s: no ssbFrequency nor refFreqCSI_RS!\n", __FUNCTION__);
        return BREAK;
    }

    return GOOD;
}
void VLIB_getNrOfBeams(rsIndexResults_type *ptRsIndexResults, enumRSType rsType, uint8_t *nrofBeams)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptRsIndexResults);
    VALID_E_POINT_CHECK_VOID(nrofBeams);

    if (RSType_SSB == rsType)
    {
        *nrofBeams = ptRsIndexResults->resultsSSB_Indexes.nrofSSB;
    }
    else if (RSType_CSIRS == rsType)
    {
        *nrofBeams = ptRsIndexResults->resultsCSI_RS_Indexes.nrofCSI_RS;
    }
    else
    {
        *nrofBeams = 0;
    }
}
int16_t VLIB_FillCslMeasNbrInfo(
    callInfo_type *ptCi, uint16_t cslNbrIdx, measResult2NR_type *ptMeasResult2NR, enumRSType rsType, uint8_t nbrIdx)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResult2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(cslNbrIdx, MAX_NBR_CELL_CSL);
    VCCB_CHECK_E_RANGE_AND_RETURN(nbrIdx, MAX_NUM_CELL_REPORT);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    cslCuCpAdjacentInfo_type *ptCslAdjacentInfo = NULL;

    measResultNR_type *ptMeasResultNR = NULL;

    uint16_t targetCellNum = 0;
    uint16_t idx = 0, nbrFound_flag = FALSE;
    uint32_t measResultArfcn = 0;
    uint32_t usedCellArfcn = 0;
    int status = GOOD;

    ptMeasResultNR = &ptMeasResult2NR->measResultNeighCellListNR.measResultNR[nbrIdx];
    if (FALSE == ptMeasResultNR->physCellIdFlag ||
        ERROR == VLIB_getArfcnFromMeasResult2NR(ptMeasResult2NR, rsType, &measResultArfcn))
    {
        VCCB_PRINT_ERROR("VLIB_FillCslMeasNbrInfo invalid ptMeasResult2NR \n");
        return ERROR;
    }

    for (idx = 0; idx < MAX_NBR_PER_CELL; idx++)  // go through all the neighbors
    {
        if (N_EQUIP == CELLINFO[ptCi->servCellNum].Nbr[idx].status)
        {
            continue;
        }

        ptCslAdjacentInfo = &ptCi->cslInfo.cslMeasLastInfo.adjacentInfo[cslNbrIdx];
        if (VCCB_IS_INTRA_GNB_NEIGHBOR(ptCi->servCellNum, idx))  // relation type INTRA NB
        {
            VCCB_GET_INTRA_GNB_NBR_CELLNUM(ptCi->servCellNum, idx, targetCellNum);
            if (targetCellNum >= MAX_CELL_NUM_VCCB)
            {
                continue;
            }

            usedCellArfcn =
                (RSType_SSB == rsType) ? CELLINFO[targetCellNum].ssbArfcn : CELLINFO[targetCellNum].nrArfcnDl;
            if (CELLINFO[targetCellNum].phyCellId != ptMeasResultNR->physCellId || usedCellArfcn != measResultArfcn)
            {
                continue;
            }

            ptCslAdjacentInfo->gNBid = GNBINFO[ptCi->gnbNum].globalGnbId.gnbId;
            ptCslAdjacentInfo->arfcnDL = measResultArfcn;
            cgi_type cgi = FnVCCB_GetNrcgiFromCellNum(targetCellNum);
            copyCgiToCslCgi(&cgi, &ptCslAdjacentInfo->cgi);
            ptCslAdjacentInfo->pci = CELLINFO[targetCellNum].phyCellId;

            nbrFound_flag = TRUE;
            break;
        }
        else
        {
            nbrGnbCellInfo_type *ptNbrGnbCell;

            status = getNbrGnbCellByCellNbrIdx(ptCi->servCellNum, idx, &ptNbrGnbCell);
            if (status != GOOD)
            {
                continue;
            }
            if (N_EQUIP == ptNbrGnbCell->status)
            {
                continue;
            }

            usedCellArfcn = (RSType_SSB == rsType) ? ptNbrGnbCell->ssbArfcn : ptNbrGnbCell->nrArfcnDl;

            if (ptNbrGnbCell->nrPci != ptMeasResultNR->physCellId || usedCellArfcn != measResultArfcn)
            {
                continue;
            }

            ptCslAdjacentInfo->gNBid = ptNbrGnbCell->globalGnbId.gnbId;
            ptCslAdjacentInfo->arfcnDL = measResultArfcn;
            ptCslAdjacentInfo->pci = ptNbrGnbCell->nrPci;
            ptCslAdjacentInfo->cgi.plmn = ptNbrGnbCell->globalGnbId.plmn;
            ptCslAdjacentInfo->cgi.ci =
                VLIB_GnbIdCellId2NrCellId(&ptNbrGnbCell->globalGnbId.gnbId, ptNbrGnbCell->cellId);

            nbrFound_flag = TRUE;
            break;
        }
    }

    if (FALSE == nbrFound_flag)
    {
        return ERROR;
    }

    return GOOD;
}

int16_t VLIB_FillCslMeasNbrInfoAdditionRequest(
    callInfo_type *ptCi, uint16_t cslNbrIdx, measResult2NR_type *ptMeasResult2NR, enumRSType rsType, uint8_t nbrIdx)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResult2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(cslNbrIdx, MAX_NBR_CELL_CSL);
    VCCB_CHECK_E_RANGE_AND_RETURN(nbrIdx, MAX_NUM_CELL_REPORT);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);

    uint16_t idx = 0;
    uint32_t measResultArfcn = 0;
    // uint32_t usedCellArfcn = 0;
    cslCuCpAdjacentInfo_type *ptCslAdjacentInfo = &ptCi->cslInfo.cslMeasInitialInfo.adjacentInfo[cslNbrIdx];
    measResultNR_type *ptMeasResultNR = &ptMeasResult2NR->measResultNeighCellListNR.measResultNR[nbrIdx];

    if (VCCB_FLAG_OFF == ptMeasResultNR->physCellIdFlag ||
        ERROR == VLIB_getArfcnFromMeasResult2NR(ptMeasResult2NR, rsType, &measResultArfcn))
    {
        VCCB_PRINT_ERROR("VLIB_FillCslMeasNbrInfo invalid ptMeasResult2NR \n");
        return ERROR;
    }

    if (ptMeasResultNR->physCellId == CELLINFO[ptCi->servCellNum].phyCellId &&
        measResultArfcn == CELLINFO[ptCi->servCellNum].ssbArfcn)
    {
        // not store results for serving cell as neighbour cell
        return ERROR;
    }

    for (idx = 0; idx < MAX_CELL_NUM_VCCB; idx++)
    {
        if (CELLINFO[idx].gnbNum != ptCi->gnbNum) continue;

        // usedCellArfcn = (RSType_SSB == rsType) ? CELLINFO[idx].ssbArfcn : CELLINFO[idx].nrArfcnDl;
        if (ptMeasResultNR->physCellId == CELLINFO[idx].phyCellId && measResultArfcn == CELLINFO[idx].ssbArfcn)
        {
            ptCslAdjacentInfo->gNBid = GNBINFO[ptCi->gnbNum].globalGnbId.gnbId;
            ptCslAdjacentInfo->arfcnDL = measResultArfcn;
            cgi_type cgi = FnVCCB_GetNrcgiFromCellNum(idx);
            copyCgiToCslCgi(&cgi, &ptCslAdjacentInfo->cgi);
            ptCslAdjacentInfo->pci = CELLINFO[idx].phyCellId;

            return GOOD;
        }
    }

    return ERROR;
}

int16_t VLIB_FindSCellWithPciArfcn(
    callInfo_type *ptCi, uint16_t pci, uint32_t arfcn, enumRSType rsType, uint8_t *foundSCellIdx, uint8_t isCslPurpose)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(foundSCellIdx);

    uint16_t idx = 0, maxScellIdx = MAX_SCELL_INDEX;
    uint32_t usedCellArfcn = 0;

    if (TRUE == isCslPurpose) maxScellIdx = MAX_NUM_SCELL_CSL;

    scellList_type &scells = FnVCCB_GetScellInfoList(*ptCi);
    for (idx = 0; idx < MAX_SCELL_PER_CALL; idx++)
    {
        if ((SCELL_STATE_ADDED == scells[idx].state) && (scells[idx].sCellIndex <= maxScellIdx) &&
            (scells[idx].sCellIndex >= 1))
        {
            usedCellArfcn = (RSType_SSB == rsType) ? scells[idx].ssbArfcn : scells[idx].nrArfcn;
            if ((scells[idx].pci == pci) && (usedCellArfcn == arfcn))
            {
                *foundSCellIdx = scells[idx].sCellIndex;
                return GOOD;
            }
        }
    }

    return ERROR;
}

int16_t VLIB_FillCslAdjacentCellInfo(
    callInfo_type *ptCi, cslMeasInfo_type *ptCslMeasInfo, measResultList2NR_type *ptMeasResultList2NR)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VALID_E_POINT_CHECK_ERROR(ptCslMeasInfo);

    measResult2NR_type *ptMeasResult2NR = NULL;
    cslCuCpAdjacentInfo_type *ptCslAdjacentInfo = NULL;
    measResultNR_type *ptMeasResultNbr = NULL;

    enumRSType rsType;
    uint8_t i = 0, nbrIdx = 0;
    uint8_t numOfBeam = 0;
    int16_t result = GOOD;

    ptCslMeasInfo->numNeighborCell = 0;

    for (i = 0; i < MIN(ptMeasResultList2NR->nrofFreq, MAX_NUM_FREQ); i++)
    {
        ptMeasResult2NR = &ptMeasResultList2NR->measResult2NR[i];
        if (GOOD != VLIB_getRsType(ptMeasResult2NR, &rsType))
        {
            continue;
        }

        for (nbrIdx = 0; nbrIdx < MIN(ptMeasResult2NR->measResultNeighCellListNR.nrofCellReport, MAX_NUM_CELL_REPORT);
             nbrIdx++)
        {
            if (csl_SgNB_Addition == ptCi->cslInfo.cslEvent ||
                (csl_SgNB_Change == ptCi->cslInfo.cslEvent && TRUE == ptCi->hoInfo.isTarget))
            {
                result = VLIB_FillCslMeasNbrInfoAdditionRequest(
                    ptCi, ptCslMeasInfo->numNeighborCell, ptMeasResult2NR, rsType, nbrIdx);
            }
            else
            {
                result = VLIB_FillCslMeasNbrInfo(ptCi, ptCslMeasInfo->numNeighborCell, ptMeasResult2NR, rsType, nbrIdx);
            }
            if (ERROR == result)
            {
                continue;
            }

            ptCslAdjacentInfo = &ptCslMeasInfo->adjacentInfo[ptCslMeasInfo->numNeighborCell];
            ptMeasResultNbr = &ptMeasResultList2NR->measResult2NR[i].measResultNeighCellListNR.measResultNR[nbrIdx];

            numOfBeam = 0;
            VLIB_getNrOfBeams(&ptMeasResultNbr->rsIndexResults, rsType, &numOfBeam);
            ptCslAdjacentInfo->rsType = rsType;
            ptCslAdjacentInfo->numOfBeam = MIN(numOfBeam, sizeof(ptCslAdjacentInfo->beamMeas) / sizeof(beamMeas_type));

            VLIB_FillBeamMeasCsl(&ptMeasResultNbr->rsIndexResults, &ptCslAdjacentInfo->beamMeas[0],
                ptCslAdjacentInfo->numOfBeam, rsType);

            VLIB_FillCellMeasCsl(
                &ptMeasResultNbr->cellResults, rsType, &ptCslAdjacentInfo->rsrp, &ptCslAdjacentInfo->rsrq);

            ptCslMeasInfo->numNeighborCell++;
            if (ptCslMeasInfo->numNeighborCell >= MAX_NBR_CELL_CSL)
            {
                return GOOD;
            }
        }
    }

    return GOOD;
}

int8_t VLIB_CslMeasReportLast_findPcellIdx(callInfo_type *ptCi, measResultList2NR_type *ptMeasResultList2NR,
    enumRSType rsType, uint32_t measResultArfcn, uint8_t measIdx, uint8_t *pCellIdx)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VALID_E_POINT_CHECK_ERROR(pCellIdx);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);

    uint32_t usedCellArfcn = 0;

    if (RSType_SSB == rsType)
        usedCellArfcn = CELLINFO[ptCi->servCellNum].ssbArfcn;
    else
        usedCellArfcn = CELLINFO[ptCi->servCellNum].nrArfcnDl;

    // Check if CELLINFO[].phyCellId == ptMeasResultList2NR..physCellId, and return pCellIdx
    if ((CELLINFO[ptCi->servCellNum].phyCellId ==
            ptMeasResultList2NR->measResult2NR[measIdx].measResultServingCell.physCellId) &&
        usedCellArfcn == measResultArfcn)
    {
        *pCellIdx = measIdx;
        return GOOD;
    }

    return ERROR;
}

int16_t VLIB_FillCslScgFailure(callInfo_type *ptCi, measResultList2NR_type *ptMeasResultList2NR, uint8_t failureType)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);

    cslScgFailure_type *ptScgFailure = NULL;
    rsIndexResults_type *ptRsIndexResults = NULL;
    enumRSType rsType;
    uint8_t i = 0;
    uint8_t pCellIdx = 0, sCellIdx = 0, sCellNum = 0, foundSCellIdx = 0;
    uint8_t numOfBeam = 0;
    uint8_t pCellFound_flag = FALSE;
    uint32_t measResultArfcn = 0;
    nbrGnbCellInfo_type *ptSCell = NULL;

    ptCi->cslInfo.scgFailureHistory.lastIdx++;
    if (ptCi->cslInfo.scgFailureHistory.lastIdx >= MAX_SCG_FAILURE_COUNT)
    {
        ptCi->cslInfo.scgFailureHistory.lastIdx = 0;
    }

    ptScgFailure = &ptCi->cslInfo.scgFailureHistory.scgFailure[ptCi->cslInfo.scgFailureHistory.lastIdx];
    memset(ptScgFailure, 0xFF, sizeof(cslScgFailure_type));

    VLIB_SetCurrentTime(&ptScgFailure->failureTime);

    ptScgFailure->suspendTimeState = VCCB_SUSPEND_STATE_OFF;
    ptScgFailure->recoveryTimeFlag = VCCB_FLAG_OFF;

    ptScgFailure->usage = CI_use;
    ptScgFailure->failureCause = failureType;

    for (i = 0; i < MIN(ptMeasResultList2NR->nrofFreq, MAX_NUM_FREQ); i++)
    {
        if (GOOD != VLIB_getRsType(&ptMeasResultList2NR->measResult2NR[i], &rsType))
        {
            continue;
        }

        if (FALSE == ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellIdFlag ||
            ERROR == VLIB_getArfcnFromMeasResult2NR(&ptMeasResultList2NR->measResult2NR[i], rsType, &measResultArfcn))
        {
            continue;
        }

        if (GOOD ==
            VLIB_CslMeasReportLast_findPcellIdx(ptCi, ptMeasResultList2NR, rsType, measResultArfcn, i, &pCellIdx))
        {
            pCellFound_flag = FLAG_ON;
            break;
        }
    }

    ptScgFailure->pCellBeamMeas.cellId = CELLINFO[ptCi->servCellNum].cellIdentity;

    if (FLAG_ON == pCellFound_flag)
    {
        ptRsIndexResults = &ptMeasResultList2NR->measResult2NR[pCellIdx].measResultServingCell.rsIndexResults;
        VLIB_getNrOfBeams(ptRsIndexResults, rsType, &numOfBeam);

        VLIB_FillBeamMeasCsl(ptRsIndexResults, &ptScgFailure->pCellBeamMeas.beamMeas,
            MIN(numOfBeam, sizeof(ptScgFailure->pCellBeamMeas.beamMeas) / sizeof(beamMeas_type)), rsType);
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VLIB_FillCslScgFailure not found pCell meas result!\n");
    }

    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];
    scellList_type &scells = FnVCCB_GetScellInfoList(*ptCi);
    for (i = 0; i < MAX_NUM_SCELL_CSL; i++)  // fill cellID for scell
    {
        const uint32_t gnbId = VLIB_NrCellId2GnbId(pGnb->globalGnbId.gnbId.idLength, scells[i].cgi.ci);

        if ((SCELL_STATE_ADDED == scells[i].state) && (scells[i].sCellIndex <= MAX_NUM_SCELL_CSL) &&
            (scells[i].sCellIndex >= 1) &&
            VCCB_IS_SAME_GNB(gnbId, scells[i].cgi.plmn.longId, ptCi->gnbNum))  // Skip inter-gnb cell.
        {
            ptSCell = VLIB_GetNbrGnbCellInfoBasedOnCgi(ptCi, scells[i].cgi, scells[i].gnbIdLength);
            VALID_E_POINT_CHECK_CONTINUE(ptSCell);
            ptScgFailure->sCellBeamMeas[scells[i].sCellIndex - 1].cellId = ptSCell->cellId;
        }
    }

    for (i = 0; i < MIN(ptMeasResultList2NR->nrofFreq, MAX_NUM_FREQ); i++)  // fill meas result for sCell
    {
        if (GOOD != VLIB_getRsType(&ptMeasResultList2NR->measResult2NR[i], &rsType))
        {
            continue;
        }

        if (FALSE == ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellIdFlag ||
            ERROR == VLIB_getArfcnFromMeasResult2NR(&ptMeasResultList2NR->measResult2NR[i], rsType, &measResultArfcn))
        {
            continue;
        }

        if (ERROR == VLIB_FindSCellWithPciArfcn(ptCi,
                         ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellId, measResultArfcn,
                         rsType, &foundSCellIdx, TRUE))
        {
            continue;
        }
        sCellIdx = static_cast<uint8_t>(foundSCellIdx - 1);
        if (sCellIdx >= MAX_NUM_SCELL_CSL)
        {
            continue;
        }
        ptRsIndexResults = &ptMeasResultList2NR->measResult2NR[i].measResultServingCell.rsIndexResults;

        numOfBeam = 0;
        VLIB_getNrOfBeams(ptRsIndexResults, rsType, &numOfBeam);
        VLIB_FillBeamMeasCsl(ptRsIndexResults, &ptScgFailure->sCellBeamMeas[sCellIdx].beamMeas,
            MIN(numOfBeam, sizeof(ptScgFailure->sCellBeamMeas[sCellIdx].beamMeas) / sizeof(beamMeas_type)), rsType);

        sCellNum++;

        if (sCellNum >= MAX_NUM_SCELL_CSL)
        {
            break;
        }
    }

    return GOOD;
}
void VLIB_FillRecoveryTimeCsl(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VCCB_CHECK_E_RANGE_AND_VOID(ptCi->cslInfo.scgFailureHistory.lastIdx, MAX_SCG_FAILURE_COUNT);

    if (CI_use == ptCi->cslInfo.scgFailureHistory.scgFailure[ptCi->cslInfo.scgFailureHistory.lastIdx].usage &&
        VCCB_FLAG_OFF ==
            ptCi->cslInfo.scgFailureHistory.scgFailure[ptCi->cslInfo.scgFailureHistory.lastIdx].recoveryTimeFlag)
    {
        VLIB_SetCurrentTime(
            &ptCi->cslInfo.scgFailureHistory.scgFailure[ptCi->cslInfo.scgFailureHistory.lastIdx].recoveryTime);
        ptCi->cslInfo.scgFailureHistory.scgFailure[ptCi->cslInfo.scgFailureHistory.lastIdx].recoveryTimeFlag =
            VCCB_FLAG_ON;
    }

    return;
}

int16_t VLIB_FillCslMeasReport(
    callInfo_type *ptCi, measResultList2NR_type *ptMeasResultList2NR, uint16_t reportType, uint8_t reportPurpose)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);
    int16_t ret = VLIB_FillCslMeasReportLast(ptCi, ptMeasResultList2NR, reportType, reportPurpose);
    if (GOOD == ret && VCCB_FLAG_ON == ptCi->cslInfo.useNextInitialMeas)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "useNextInitialMeas enabled, initial measurement will be set\n");
        ptCi->cslInfo.cslMeasInitialInfo = ptCi->cslInfo.cslMeasLastInfo;
        ptCi->cslInfo.useNextInitialMeas = VCCB_FLAG_OFF;
    }

    return ret;
}

void VLIB_ClearCslMeasReport(cslMeasInfo_type *ptCslMeasInfo)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCslMeasInfo);
    memset(ptCslMeasInfo, 0xFF, sizeof(cslMeasInfo_type));
    ptCslMeasInfo->receivedFlag = FLAG_OFF;
    ptCslMeasInfo->pCellPresentFlag = FLAG_OFF;
    ptCslMeasInfo->numNeighborCell = 0;
}
int16_t VLIB_FillCslMeasReportLast(
    callInfo_type *ptCi, measResultList2NR_type *ptMeasResultList2NR, uint16_t reportType, uint8_t reportPurpose)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);

    cslMeasInfo_type *ptCslMeasInfo = &ptCi->cslInfo.cslMeasLastInfo;
    measResult2NR_type *ptMeasResult2NR = NULL;

    enumRSType rsType;
    uint8_t i = 0, pCellIdx = 0;
    uint8_t numOfBeam = 0;
    uint32_t measResultArfcn = 0;

    ptCslMeasInfo->receivedFlag = FLAG_ON;
    VLIB_SetCurrentTime(&ptCslMeasInfo->receivedTime);
    ptCslMeasInfo->reportType = reportType;
    ptCslMeasInfo->reportPurpose = reportPurpose;

    ptCslMeasInfo->pCellPresentFlag = FLAG_OFF;
    for (i = 0; i < UMIN(ptMeasResultList2NR->nrofFreq, MAX_NUM_FREQ); i++)
    {
        if (GOOD != VLIB_getRsType(&ptMeasResultList2NR->measResult2NR[i], &rsType))
        {
            continue;
        }

        if (FALSE == ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellIdFlag ||
            ERROR == VLIB_getArfcnFromMeasResult2NR(&ptMeasResultList2NR->measResult2NR[i], rsType, &measResultArfcn))
        {
            continue;
        }

        if (GOOD ==
            VLIB_CslMeasReportLast_findPcellIdx(ptCi, ptMeasResultList2NR, rsType, measResultArfcn, i, &pCellIdx))
        {
            ptCslMeasInfo->pCellPresentFlag = FLAG_ON;
            break;
        }
    }

    if (FLAG_ON == ptCslMeasInfo->pCellPresentFlag)
    {
        ptMeasResult2NR = &ptMeasResultList2NR->measResult2NR[pCellIdx];

        VLIB_getNrOfBeams(&ptMeasResult2NR->measResultServingCell.rsIndexResults, rsType, &numOfBeam);
        ptCslMeasInfo->numOfServBeam = MIN(numOfBeam, sizeof(ptCslMeasInfo->servBeamMeas) / sizeof(beamMeas_type));

        VLIB_FillBeamMeasCsl(&ptMeasResult2NR->measResultServingCell.rsIndexResults, &ptCslMeasInfo->servBeamMeas[0],
            ptCslMeasInfo->numOfServBeam, rsType);

        VLIB_FillCellMeasCsl(&ptMeasResult2NR->measResultServingCell.cellResults, rsType,
            &ptCslMeasInfo->rsrpServingCell, &ptCslMeasInfo->rsrqServingCell);

        ptCslMeasInfo->pci = ptMeasResultList2NR->measResult2NR[pCellIdx].measResultServingCell.physCellId;
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VLIB_FillCslMeasReportLast not found pCell meas result\n");
    }

    if (ERROR == VLIB_FillCslAdjacentCellInfo(ptCi, ptCslMeasInfo, ptMeasResultList2NR))
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VLIB_FillCslMeasReportLast nbr meas result not filled\n");
    }

    return GOOD;
}
int16_t VLIB_FillCslMeasReportAdditionRequest(
    callInfo_type *ptCi, measResultList2NR_type *ptMeasResultList2NR, uint16_t reportType, uint8_t reportPurpose)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptMeasResultList2NR);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->servCellNum, MAX_CELL_NUM_VCCB);

    cslMeasInfo_type *ptCslMeasInfo = &ptCi->cslInfo.cslMeasInitialInfo;
    measResultNR_type *ptMeasResultNR = NULL;

    enumRSType rsType;
    uint8_t i = 0, nbrIdx = 0;
    uint8_t numOfBeam = 0;
    uint32_t usedCellArfcn = 0;
    uint32_t measResultArfcn = 0;

    ptCslMeasInfo->receivedFlag = FLAG_ON;
    VLIB_SetCurrentTime(&ptCslMeasInfo->receivedTime);
    ptCslMeasInfo->reportType = reportType;
    ptCslMeasInfo->reportPurpose = reportPurpose;

    ptCslMeasInfo->pCellPresentFlag = FLAG_OFF;
    for (i = 0; i < MIN(ptMeasResultList2NR->nrofFreq, MAX_NUM_FREQ); i++)
    {
        if (GOOD != VLIB_getRsType(&ptMeasResultList2NR->measResult2NR[i], &rsType) ||
            ERROR == VLIB_getArfcnFromMeasResult2NR(&ptMeasResultList2NR->measResult2NR[i], rsType, &measResultArfcn))
        {
            continue;
        }

        usedCellArfcn =
            (RSType_SSB == rsType) ? CELLINFO[ptCi->servCellNum].ssbArfcn : CELLINFO[ptCi->servCellNum].nrArfcnDl;

        if (VCCB_FLAG_ON == ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellIdFlag &&
            CELLINFO[ptCi->servCellNum].phyCellId ==
                ptMeasResultList2NR->measResult2NR[i].measResultServingCell.physCellId &&
            usedCellArfcn == measResultArfcn)
        {
            ptCslMeasInfo->pCellPresentFlag = FLAG_ON;
            ptMeasResultNR = &ptMeasResultList2NR->measResult2NR[i].measResultServingCell;
            break;
        }

        for (nbrIdx = 0; nbrIdx < MIN(ptMeasResultList2NR->measResult2NR[i].measResultNeighCellListNR.nrofCellReport,
                                      MAX_NUM_CELL_REPORT);
             nbrIdx++)
        {
            if (VCCB_FLAG_ON == ptMeasResultList2NR->measResult2NR[i]
                                    .measResultNeighCellListNR.measResultNR[nbrIdx]
                                    .physCellIdFlag &&
                CELLINFO[ptCi->servCellNum].phyCellId ==
                    ptMeasResultList2NR->measResult2NR[i].measResultNeighCellListNR.measResultNR[nbrIdx].physCellId &&
                usedCellArfcn == measResultArfcn)
            {
                ptMeasResultNR = &ptMeasResultList2NR->measResult2NR[i].measResultNeighCellListNR.measResultNR[nbrIdx];
                ptCslMeasInfo->pCellPresentFlag = FLAG_ON;
                break;
            }
        }
    }

    if (FLAG_ON == ptCslMeasInfo->pCellPresentFlag && ptMeasResultNR != NULL)
    {
        VLIB_getNrOfBeams(&ptMeasResultNR->rsIndexResults, rsType, &numOfBeam);
        ptCslMeasInfo->numOfServBeam = MIN(numOfBeam, sizeof(ptCslMeasInfo->servBeamMeas) / sizeof(beamMeas_type));

        VLIB_FillBeamMeasCsl(
            &ptMeasResultNR->rsIndexResults, &ptCslMeasInfo->servBeamMeas[0], ptCslMeasInfo->numOfServBeam, rsType);

        VLIB_FillCellMeasCsl(
            &ptMeasResultNR->cellResults, rsType, &ptCslMeasInfo->rsrpServingCell, &ptCslMeasInfo->rsrqServingCell);

        ptCslMeasInfo->pci = ptMeasResultNR->physCellId;
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VLIB_FillCslMeasReportAdditionRequest not found pCell meas result\n");
    }

    if (ERROR == VLIB_FillCslAdjacentCellInfo(ptCi, ptCslMeasInfo, ptMeasResultList2NR))
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VLIB_FillCslMeasReportAdditionRequest nbr meas result not filled\n");
    }

    ptCi->cslInfo.useNextInitialMeas = VCCB_FLAG_OFF;

    return GOOD;
}
/**
 * @brief     Initialize activeUeBitmap for VNMB-VCCB call resource audit
 * @param     none
 * @return    none
 */
void FnVCCB_InitActiveUeBitmap(void)
{
    AUTO_TRACE_FUNC;

    MUTEX_LOCK(&gActiveUeBitmapMtx);
#ifdef __CNF__  // CallTakeover
    for (int distCode = 0; distCode < MAX_DIST_CODE_PER_SYS; ++distCode)
    {
        memset(ENV->distCodeInfo[distCode].activeUeBitmap, 0, sizeof(ENV->distCodeInfo[distCode].activeUeBitmap));
    }
#else
    memset(ENV->keepAliveInfo.activeUeBitmap, 0, sizeof(ENV->keepAliveInfo.activeUeBitmap));
#endif
    MUTEX_UNLOCK(&gActiveUeBitmapMtx);
}

#ifdef __CNF__
static inline
int16_t FnVCCB_SetActiveUeBitmap_BitSetTRUE_ForCNF(DistCodeInfo_type *pDistCodeInfo, uint32_t index, uint32_t remainder, uint16_t distCode, uint32_t distCallId)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(pDistCodeInfo);

    if (index >= MAX_NUM_ACTIVE_UE_ARRAY)
    {
        VCCB_PRINT_ERROR(
            "(distCode:%u, distCallId:%u) invalid index:%u in %s\n", distCode, distCallId, index, __func__);
        return ERROR;
    }

    MUTEX_LOCK(&gActiveUeBitmapMtx);

    if (pDistCodeInfo->activeUeBitmap[index] & ((uint64_t)1 << ((BIT_COUNT_OF_UINT64 - 1) - remainder)))
    {
        VCCB_PRINT_ERROR("(distCode:%u distCallId:%u) is already set in %s\n", distCode, distCallId, __func__);
    }
    else
    {
        pDistCodeInfo->activeUeBitmap[index] |= ((uint64_t)1 << ((BIT_COUNT_OF_UINT64 - 1) - remainder));
    }

    MUTEX_UNLOCK(&gActiveUeBitmapMtx);

    return GOOD;
}

static inline
int16_t FnVCCB_SetActiveUeBitmap_BitSetFALSE_ForCNF(DistCodeInfo_type *pDistCodeInfo, uint32_t index, uint32_t remainder, uint16_t distCode, uint32_t distCallId)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(pDistCodeInfo);

    if (index >= MAX_NUM_ACTIVE_UE_ARRAY)
    {
        VCCB_PRINT_ERROR(
            "(distCode:%u, distCallId:%u) invalid index:%u in %s\n", distCode, distCallId, index, __func__);
        return ERROR;
    }

    MUTEX_LOCK(&gActiveUeBitmapMtx);

    if (!(pDistCodeInfo->activeUeBitmap[index] & ((uint64_t)1 << ((BIT_COUNT_OF_UINT64 - 1) - remainder))))
    {
        VCCB_PRINT_ERROR("(distCode:%u distCallId:%u) is already unset in %s\n", distCode, distCallId, __func__);
    }
    else
    {
        pDistCodeInfo->activeUeBitmap[index] &= ~((uint64_t)1 << ((BIT_COUNT_OF_UINT64 - 1) - remainder));
    }

    MUTEX_UNLOCK(&gActiveUeBitmapMtx);

    return GOOD;
}
#endif

int16_t FnVCCB_SetActiveUeBitmap(const uint32_t cuCpUeId, uint8_t bitSet)
{
    AUTO_TRACE_FUNC;

#ifdef __CNF__  // CallTakeover
    uint16_t distCode = VCCB_GET_DIST_CODE(cuCpUeId);
    VCCB_CHECK_E_RANGE_AND_RETURN(distCode, MAX_DIST_CODE_PER_SYS);
    DistCodeInfo_type *pDistCodeInfo = &ENV->distCodeInfo[distCode];

    uint32_t distCallId = VCCB_GET_DIST_CALL_ID(cuCpUeId);
    VCCB_CHECK_E_RANGE_AND_RETURN(distCallId, MAX_CALL_COUNT_PER_DISTCODE);

    uint32_t index = distCallId / BIT_COUNT_OF_UINT64;
    uint32_t remainder = distCallId % BIT_COUNT_OF_UINT64;

    if (TRUE == bitSet)
    {
        return FnVCCB_SetActiveUeBitmap_BitSetTRUE_ForCNF(pDistCodeInfo, index, remainder, distCode, distCallId);
    }
    else  // FALSE == bitSet
    {
        return FnVCCB_SetActiveUeBitmap_BitSetFALSE_ForCNF(pDistCodeInfo, index, remainder, distCode, distCallId);
    }
#else
    uint32_t callId = VCCB_GET_VCCB_CALL_ID(cuCpUeId);
    VCCB_CHECK_E_RANGE_AND_RETURN(callId, VCCB_MAX_CALL_COUNT_PER_CPC);

    uint32_t index = callId / BIT_COUNT_OF_UINT64;
    uint32_t remainder = callId % BIT_COUNT_OF_UINT64;

    if (index >= MAX_NUM_ACTIVE_UE_ARRAY)
    {
        VCCB_PRINT_ERROR("FnVCCB_SetActiveUeBitmap invalid index:%d, callId:%d\n", index, callId);
        return ERROR;
    }

    if (TRUE == bitSet)
    {
        MUTEX_LOCK(&gActiveUeBitmapMtx);

        if (ENV->keepAliveInfo.activeUeBitmap[index] &
            (1ULL << ((BIT_COUNT_OF_UINT64 - 1) - remainder)))
        {
            VCCB_PRINT_ERROR("%s callId:%u is already set\n", __func__, callId);
        }
        else
        {
            ENV->keepAliveInfo.activeUeBitmap[index] |=
                (1ULL << ((BIT_COUNT_OF_UINT64 - 1) - remainder));
        }

        MUTEX_UNLOCK(&gActiveUeBitmapMtx);
    }
    else  // FALSE == bitSet
    {
        MUTEX_LOCK(&gActiveUeBitmapMtx);

        if (!(ENV->keepAliveInfo.activeUeBitmap[index] &
                (static_cast<uint64_t>(1) << ((BIT_COUNT_OF_UINT64 - 1) - remainder))))
        {
            VCCB_PRINT_ERROR("%s callId:%u is already unset\n", __func__, callId);
        }
        else
        {
            ENV->keepAliveInfo.activeUeBitmap[index] &=
                ~(static_cast<uint64_t>(1) << ((BIT_COUNT_OF_UINT64 - 1) - remainder));
        }

        MUTEX_UNLOCK(&gActiveUeBitmapMtx);
    }

    return GOOD;
#endif
}
/**
 * @brief     Commit/Acknowledge RadioBearerConfig by UE
 * @param     ptCi : call context
 */
void VLIB_CommitRadioBearerConfigUE(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    if (ptCi->callType == E_CALLTYPE_SA)
    {
        VCCB_NrdcUlPathControl_CollectPathChangeStats(ptCi);

        uint8_t callDrbIdx = 0;
        callDRBInfo_type *ptDi = NULL;

        for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_SA_CALL; callDrbIdx++)
        {
            ptDi = &ptCi->drb[callDrbIdx];

            if (!IS_VALID_NODE(DRB_NODE, ptCi, ptDi)) continue;

            ptDi->ulPdcpPathPrev = ptDi->ulPdcpPath;
            ptDi->qfiIdsToRelease = 0;
        }
    }
    else  // ptCi->callType == E_CALLTYPE_NSA
    {
        FnVCCB_EndcUlPathControl_CollectPathChangeStats(ptCi);

        int erabIdx = 0;
        callERABInfo_type *ptRi = NULL;

        for (erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
        {
            ptRi = &ptCi->erabInfo[erabIdx];
            if (CI_use != ptRi->usage)
            {
                continue;
            }

            ptRi->ulPdcpPathPrev = ptRi->ulPdcpPath;
            OFF_DRB_SRC_CFG_FLAG(ptRi->flag.setupFlag);
        }
    }

    ptCi->drbIdsToRelease = 0;  // SA + NSA Init
}

/**
 * @brief     Get NR Band Index
 * @param     rCi   : call context
 * @param     bandNR : bandNR to be checked
 * @return    MAX_NUM_NR_BAND (1024) : cannot find / else bandNRindex (0..1023)
 */
uint16_t VLIB_GetNrBandIndex(const callInfo_type &rCi, uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    uint16_t bandNRidx;
    for (bandNRidx = 0; bandNRidx < MIN(rCi.ueInfo.ueCapa.nr.rf.nrOfBands, MAX_NUM_UECAPA_NR_BANDS); bandNRidx++)
    {
        if (rCi.ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx].bandNR == bandNR)
        {
            return bandNRidx;
        }
    }

    return MAX_NUM_UECAPA_NR_BANDS;
}

/**
 * @brief     Check if cell is multi Band cell
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @return    true/false
 */
bool VLIB_isMfbiCell(const nbrGnbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_RET_VAL(ptCell, false);

    int count = 0, cellBIIdx = 0;
    for (cellBIIdx = 0; cellBIIdx < MAX_FREQBAND_PER_CELL; cellBIIdx++)
    {
        if (EQUIP != ptCell->freqBandInfoList[cellBIIdx].status)
        {
            continue;
        }

        ++count;
        if (count >= 2)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief     Check if cell is multi Band cell and DU RRC version is at least 16.5
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @param     spCellNum : UE's current or target cell num
 * @return    true/false
 */
bool VLIB_shouldBwBeChecked(const callInfo_type *ptCi, const nbrGnbCellInfo_type *ptCell, uint16_t spCellNum)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_RET_VAL(ptCi, false);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, false);

    if (ptCi->bitFlag.isMcoLteAccessCase)  // don't check BW for MCO Lte Access Case (as rrc version is not known)
    {
        return false;
    }

    // check DU's rrc version
    VCCB_CHECK_E_RANGE_AND_VAL(spCellNum, MAX_CELL_NUM_VCCB, false);
    const VccbCellInfo_type *pSpCell = &CELLINFO[spCellNum];
    VCCB_CHECK_E_RANGE_AND_VAL(pSpCell->gnbNum, MAX_GNB_PER_SYS, false);
    VCCB_CHECK_E_RANGE_AND_VAL(pSpCell->endPointF1Index, MAX_GNB_DU_PER_GNB_CU_CP, false);
    const gnbDuInfo_type *pGnbDu = &GNBINFO[pSpCell->gnbNum].gnbDu[pSpCell->endPointF1Index];
    const uint32_t minimumRrcVersion = static_cast<uint32_t> TS_3GPP_REL_VERSION(16, 5, 0);
    const uint32_t duRrcVersion =
        static_cast<uint32_t> TS_3GPP_REL_VERSION(pGnbDu->rrcVersion[0], pGnbDu->rrcVersion[1], pGnbDu->rrcVersion[2]);
    if (duRrcVersion < minimumRrcVersion)
    {
        return false;
    }

    return VLIB_isMfbiCell(ptCell);
}

/**
 * @brief     Check if any NR freq band is configured for Cell
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @return    ERROR : preconditions failed / GOOD : any NR freq band is configured / BREAK : otherwise
 */
int16_t VLIB_IsAnyNrFreqBandConfiguredInCell(const nbrGnbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCell);

    int cellBIIdx = 0;

    for (cellBIIdx = 0; cellBIIdx < MAX_FREQBAND_PER_CELL; cellBIIdx++)
    {
        if (EQUIP == ptCell->freqBandInfoList[cellBIIdx].status)
        {
            return GOOD;
        }
    }
    return BREAK;
}

/**
 * @brief     Checks if cell is N77/N78 mixed cell (frequency startPoint < 3650MHz < frequency endPoint)
 *            startPoint = nrArfcnDl. endPoint = startPoint + BW
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @return    true/false
 */
bool VLIB_isN77N78MixedCell(const nbrGnbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCell, false);

    constexpr uint32_t FREQUENCY_3650MHZ = 3650000; //kHz

    const enumNrSubCarrierSpacing_type cellScs = static_cast<enumNrSubCarrierSpacing_type>(ptCell->nrScsDl);
    const enumNrSystemBandwidth_type cellNrb = static_cast<enumNrSystemBandwidth_type>(ptCell->nrbDl);

    const uint32_t startPoint = VLIB_Arfcn2Freq(ptCell->nrArfcnDl); //kHz
    const uint32_t endPoint = startPoint + VLIB_NrbScs2Bw(cellNrb, cellScs) * 1000; //kHz

    return (startPoint < FREQUENCY_3650MHZ) && (FREQUENCY_3650MHZ < endPoint);
}

/**
 * @brief     Checks if band N77 should be prioritized
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @param     spCellNum : UE's current or target cell num
 * @param     isForSpCell : informs if it is band selection for spCell or sCell
 * @return    true/false
 */
bool VLIB_shouldN77BePrioritized(callInfo_type *ptCi, const nbrGnbCellInfo_type *ptCell,
        uint16_t spCellNum, bool isForSpCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCi, false);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, false);

    // for both FGR-CC3201 FR51 and 110 UE needs support extended N77
    if (TRUE != ptCi->ueInfo.ueCapa.nr.rf.extendedBand_n77_2_r17)
    {
        return false;
    }

    // FR51: gNB-CU shall select band indicator at n77/n78 mixed MFBI cell considering extended n77 UE capa
    if ((TRUE == FOO.VF_MultipleFrequencyBandIndicator_MixedMFBICellCanada) &&
        (VLIB_isN77N78MixedCell(ptCell)))
    {
        return true;
    }

    // FR110
    uint16_t cellNum = VLIB_GetCellNumBasedOnNbrGnbCellInfo(*ptCi, *ptCell);
    if (MAX_CELL_NUM_VCCB == cellNum)
    {
        // in case of ERROR or inter-gNB use spCell cellNum
        cellNum = spCellNum;
    }

    VCCB_CHECK_VALID_CELLNUM_RET_VAL(cellNum, false);
    VCCB_CHECK_VALID_GNBNUM_RET_VAL(ptCi->gnbNum, false);
    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    // special-band-change-flag is turned on
    if (!CELLINFO[cellNum].mfbiConfig.specialBandChangeFlag)
    {
        return false;
    }

    if (isForSpCell)
    {
        // for spCell PCC is turned off
        if (TRUE == pGnb->multiCarrierOperation.capabilityAwareCarrierChange)
        {
            return false;
        }

        // if it initial attach, RRE, Resume with PDU sessions enqueue intra-cell HO to change bandNR
        if ((RRCResume_cmd == ptCi->stateCmd) ||
            (RrcReestablish_cmd == ptCi->stateCmd) ||
            (RRCConnection_cmd == ptCi->stateCmd))
        {
            if (0 == ptCi->numPDUSess)
            {
                ptCi->bitFlag.changeBandAfterPduSetup = TRUE;
            }
            else if (TRUE != ptCi->bitFlag.changeBandDuringIntraCellHo)
            {
                int16_t status = VCCM_enqueueActionWithForce(ptCi, ENQUEUE_ACTION_INTRA_CELL_HO, true);
                if (ERROR == status)
                {
                    VCCB_PRINT_CALL_ERROR(ptCi, "[MFBI] VCCM_enqueueAction(ENQUEUE_ACTION_INTRA_CELL_HO) ERROR\n");
                }
                else if (GOOD == status)
                {
                    ptCi->bitFlag.changeBandDuringIntraCellHo = TRUE;
                }
            }
            return false;
        }
    }
    else
    {
        // for scell SCC is turned off
        if (TRUE == pGnb->multiCarrierOperation.sccChange)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief     Get list of prioritized eligible common FreqBandInd for Cell and UE based on custom cell (spCellNum)
 *            DU version and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 *            If it is for spCell selection UL+DL BW and UL+DL band support in band combination is checked.
 *            Otherwise (for sCell) only DL BW is checked.
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @param     spCellNum : UE's current or target cell num
 * @param     isForSpCell : informs if it is band selection for spCell or sCell
 * @param     bandList[out] : list of eligible bands
 * @return    Number of bands on bandList
 */
uint8_t VLIB_GetFreqBandIndListForNeighborCellWithCustomSpCell(callInfo_type *ptCi,
    const nbrGnbCellInfo_type *ptCell, uint16_t spCellNum, bool isForSpCell,
    uint16_t bandList[MAX_FREQBAND_PER_CELL])
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ZERO(ptCi);
    VALID_E_POINT_CHECK_ZERO(ptCell);
    VALID_E_POINT_CHECK_ZERO(bandList);

    int16_t n77Idx = -1, n78Idx = -1;
    uint8_t bandCnt = 0;

    const bool shouldBwBeChecked = VLIB_shouldBwBeChecked(ptCi, ptCell, spCellNum);
    const bool shouldBandSupportedAsPcellBandBeChecked = isForSpCell && VLIB_isMfbiCell(ptCell);

    for (uint32_t cellBIIdx = 0; cellBIIdx < MAX_FREQBAND_PER_CELL; cellBIIdx++)
    {
        if (EQUIP != ptCell->freqBandInfoList[cellBIIdx].status)
        {
            continue;
        }
        uint16_t cellBandNr = ptCell->freqBandInfoList[cellBIIdx].nRFrequencyBand;

        for (uint32_t ueCapaIdx = 0; ueCapaIdx < UMIN(ptCi->ueInfo.ueCapa.nr.rf.nrOfBands, MAX_NUM_UECAPA_NR_BANDS); ueCapaIdx++)
        {
            const bandNrUeCapa_type &rBandNrUeCapa = ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[ueCapaIdx];
            if (cellBandNr != rBandNrUeCapa.bandNR)
            {
                continue;
            }
            // bandNR in supportedBandListNR is unique. After this point break should be used.

            if ((shouldBandSupportedAsPcellBandBeChecked) &&
                (FLAG_OFF == rBandNrUeCapa.isBandSupportedAsPcellBand))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "BandNr %u not supported as spCell band\n", cellBandNr);
                break;
            }

            if (shouldBwBeChecked)
            {
                if (GOOD != VLIB_checkIntraNRChannelBWDl(ptCi, ptCell, cellBandNr, spCellNum, FALSE))
                {
                    break;
                }

                if ((isForSpCell) &&
                    (GOOD != VLIB_checkIntraNRChannelBWUl(ptCi, ptCell, cellBandNr, spCellNum, FALSE)))
                {
                    break;
                }
            }

            if (N77_BAND == cellBandNr)
            {
                n77Idx = bandCnt;
            }
            else if (N78_BAND == cellBandNr)
            {
                n78Idx = bandCnt;
            }

            bandList[bandCnt] = cellBandNr;
            ++bandCnt;
            break;
        }
    }

    if ((n78Idx >= 0) && // n78 supported
        (n77Idx > n78Idx) && // n77 supported and worse position than n78
        (VLIB_shouldN77BePrioritized(ptCi, ptCell, spCellNum, isForSpCell)))
    {
        //swap
        bandList[n77Idx] = N78_BAND;
        bandList[n78Idx] = N77_BAND;
    }

    return bandCnt;
}

/**
 * @brief     Get common FreqBandInd for Cell (ptCell) and UE based on custom cell (spCellNum) DU version
 *            and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @param     spCellNum : UE's current or target cell num
 * @param     isForSpCell : informs if it is band selection for spCell or sCell
 * @return    0 : cannot find / else FreqBandInd (1..1024)
 */
uint16_t VLIB_GetFreqBandIndForNeighborCellWithCustomSpCell(callInfo_type *ptCi,
    const nbrGnbCellInfo_type *ptCell, uint16_t spCellNum, bool isForSpCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCi, BAND_NOT_FOUND);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, BAND_NOT_FOUND);

    uint16_t bandList[MAX_FREQBAND_PER_CELL];
    uint8_t bandCnt = VLIB_GetFreqBandIndListForNeighborCellWithCustomSpCell(
            ptCi, ptCell, spCellNum, isForSpCell, bandList);

    if (0 < bandCnt)
    {
        return bandList[0];
    }

    if (isForSpCell)
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "Cannot find supported band for spCell\n");
    }

    return BAND_NOT_FOUND;
}

/**
 * @brief     Get common FreqBandInd for spCell (ptCell) and UE based on current spCell DU version (with UL BW check)
 *            and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @return    0 : cannot find / else FreqBandInd (1..1024)
 */
uint16_t VLIB_GetFreqBandIndForNeighborCell(callInfo_type *ptCi, const nbrGnbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCi, BAND_NOT_FOUND);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, BAND_NOT_FOUND);

    uint16_t spCellNum = FnVCCB_GetCellSelectionForConfig(ptCi).cellNum;
    return VLIB_GetFreqBandIndForNeighborCellWithCustomSpCell(ptCi, ptCell, spCellNum, true);
}

/**
 * @brief     Get common FreqBandInd for sCell (ptCell) and UE based on current spCell DU version (without UL BW check)
 *            and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 * @param     ptCi : call context
 * @param     ptCell : pointer to nbrGnbCellInfo_type
 * @return    0 : cannot find / else FreqBandInd (1..1024)
 */
uint16_t VLIB_GetFreqBandIndForNeighborServingCell(callInfo_type *ptCi, const nbrGnbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCi, BAND_NOT_FOUND);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, BAND_NOT_FOUND);

    uint16_t spCellNum = FnVCCB_GetCellSelectionForConfig(ptCi).cellNum;
    return VLIB_GetFreqBandIndForNeighborCellWithCustomSpCell(ptCi, ptCell, spCellNum, false);
}

/**
 * @brief     Get common FreqBandInd for spCell (ptCell) and UE based on current spCell DU version (with UL BW check)
 *            and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 * @param     ptCi : call context
 * @return    0 : cannot find / else FreqBandInd (1..1024)
 */
uint16_t VLIB_GetFreqBandIndForCell(callInfo_type *ptCi, VccbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_RET_VAL(ptCi, BAND_NOT_FOUND);
    VALID_E_POINT_CHECK_RET_VAL(ptCell, BAND_NOT_FOUND);

    uint16_t bandNR = VLIB_GetFreqBandIndForNeighborCell(ptCi, &ptCell->intraGnbNbrInfo);

    return bandNR;
}

/**
 * @brief     Get common FreqBandInd for spCell with provided cellNum and UE based on current spCell DU version (with UL BW check)
 *            and restrictPscellChangeBwMismatch/targetCellBandwidthMode setting.
 * @param     ptCi : call context
 * @return    0 : cannot find / else FreqBandInd (1..1024)
 */
uint16_t VLIB_GetFreqBandIndForCellNum(callInfo_type *ptCi, uint16_t cellNum)
{
    AUTO_TRACE_FUNC;

    VCCB_CHECK_VALID_CELLNUM_RET_VAL(cellNum, BAND_NOT_FOUND);

    uint16_t bandNR = VLIB_GetFreqBandIndForCell(ptCi, &CELLINFO[cellNum]);

    return bandNR;
}

// FGR-CC5200:FR1010
int16_t FnVCCB_GetFreqAndBandListFromInstalledMeasIdForSCGBandSel(
    callInfo_type *ptCi, enumMeasurementReportType reportType, uint8_t reportPurpose, std::vector<frequencyAndBandIndiNR_type> &vFreqBand)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    if (reportType != measReport_A4 || reportPurpose != a4Purpose_NrdcAddition)
    {
        return ERROR;;
    }
    // check if AnrMns measurementsConfiguration Activated
    measurementsConfiguration_type &rMeasConf = FnVCCB_GetMeasurementsConfiguration(*ptCi);
    int16_t idx = 0;

    // get ratType
    enumMeasRatType ratType;
    for (idx = 0; idx < MAX_NR_OF_MEAS_ID; idx++)
    {
        uint8_t reportConfigId;
        uint8_t objectId;
        if (CI_use != rMeasConf.measurementIdentities.installedItems[idx].usage) continue;

        measurementIdentity_type *pSelectMeasId =
            &rMeasConf.measurementIdentities.installedItems[idx];
        if (ERROR == VCCB_FindReportConfigById(pSelectMeasId->reportConfigId,
                         rMeasConf.reportConfigurations.installedItems,
                         MAX_NR_OF_REPORT_CONFIG_ID, &reportConfigId))
            continue;
        if (ERROR == VCCB_FindMeasurementObjectById(pSelectMeasId->measObjectId,
                         rMeasConf.measurementObjects.installedItems,
                         VCCB_ARRAY_COUNT(rMeasConf.measurementObjects.installedItems),
                         &objectId))
            continue;

        if (MAX_NR_OF_REPORT_CONFIG_ID <= reportConfigId) continue;
        if (VCCB_ARRAY_COUNT(rMeasConf.measurementObjects.installedItems) <= objectId) continue;

        ratType = VCCB_GetRatType(static_cast<enumMeasurementReportType>(rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportType));
        if (ratType != measRatNr) continue;

        const measurementObjectUe_type & mo = rMeasConf.measurementObjects.installedItems[objectId];
        uint32_t moFreq = VCCB_GetMeasObjectSsbFreq(mo);

        if (IS_FR1(moFreq)) continue;

        if (mo.conf.rat.nr.minimal.freqBandIndicatorNR != FR2_39GHz_n260 &&
                mo.conf.rat.nr.minimal.freqBandIndicatorNR != FR2_28GHz_n261)
        {
            continue;
        }

        if (reportType == rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportType &&
            reportPurpose == rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportPurpose)
        {
            frequencyAndBandIndiNR_type freqAndBandIndi;
            freqAndBandIndi.ssbFreq = moFreq;
            freqAndBandIndi.bandIndiNR = mo.conf.rat.nr.minimal.freqBandIndicatorNR;
            vFreqBand.push_back(freqAndBandIndi);
        }
    }
    return GOOD;
}

int16_t VCCB_FindInstalledMeasId_ByFreqReportPurpose(
    callInfo_type *ptCi, uint32_t freq, enumMeasurementReportType reportType, uint8_t reportPurpose)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    // check if AnrMns measurementsConfiguration Activated
    measurementsConfiguration_type &rMeasConf = FnVCCB_GetMeasurementsConfiguration(*ptCi);
    int16_t idx = 0;

    // get ratType
    enumMeasRatType ratType;

    for (idx = 0; idx < MAX_NR_OF_MEAS_ID; idx++)
    {
        uint8_t reportConfigId;
        uint8_t objectId;
        if (CI_use != rMeasConf.measurementIdentities.installedItems[idx].usage) continue;

        measurementIdentity_type *pSelectMeasId =
            &rMeasConf.measurementIdentities.installedItems[idx];
        if (ERROR == VCCB_FindReportConfigById(pSelectMeasId->reportConfigId,
                         rMeasConf.reportConfigurations.installedItems,
                         MAX_NR_OF_REPORT_CONFIG_ID, &reportConfigId))
            continue;
        if (ERROR == VCCB_FindMeasurementObjectById(pSelectMeasId->measObjectId,
                         rMeasConf.measurementObjects.installedItems,
                         VCCB_ARRAY_COUNT(rMeasConf.measurementObjects.installedItems),
                         &objectId))
            continue;

        if (MAX_NR_OF_REPORT_CONFIG_ID <= reportConfigId) continue;
        if (VCCB_ARRAY_COUNT(rMeasConf.measurementObjects.installedItems) <= objectId) continue;

        ratType = VCCB_GetRatType(static_cast<enumMeasurementReportType>(rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportType));
        if (ratType == measRatUnknown) continue;

        uint32_t moFreq;
        if (ratType == measRatNr)
            moFreq = VCCB_GetMeasObjectSsbFreq(rMeasConf.measurementObjects.installedItems[objectId]);
        else
            moFreq = VCCB_GetMeasObjectCarrierFreq(rMeasConf.measurementObjects.installedItems[objectId]);

        if (freq == 0 && reportType == measReport_A4 && reportPurpose == a4Purpose_NrdcAddition)
        {
            // to pass checking freq
            moFreq = freq;
        }

        if ((freq == moFreq) &&
            (reportType == rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportType) &&
            (reportPurpose == rMeasConf.reportConfigurations.installedItems[reportConfigId].conf.reportPurpose))
        {
            return idx;
        }
    }
    return ERROR;
}

/**
 * @brief This Function checks Measurement Config is installed in UE by callback function ID.
 * @param [in] ptCi : pointer of callInfo_type
 * @param [in] callbackId : callback function ID
 * @return TRUE(Installed) / FALSE (Not installed)
 */
uint8_t FnVCCB_CheckMeasInstalledByCallbackId(callInfo_type *ptCi, enumMeasReportCallbackId_type callbackId)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_FALSE(ptCi);

    measurementsConfiguration_type &rMeasConf = FnVCCB_GetMeasurementsConfiguration(*ptCi);
    for (uint16_t loop = 0; loop < MAX_NR_OF_MEAS_ID; loop++)
    {
        if (CI_use == rMeasConf.measurementIdentities.installedItems[loop].usage)
        {
            // Find installed measurement config by callback function ID
            if (rMeasConf.measurementIdentities.installedItems[loop].callbackFuncId == callbackId)
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "%s:Measurement Config is installed.\n", __func__);
                return TRUE;
            }
        }
    }
    return FALSE;
}

/**
 * @brief This function detects exceptional cases: BWs supported only for downlink for FR1 NR band
 *        in version 15.7.0 of 5.3.5-1 Table from 38.101-1 3gpp specification.
 * @param [in] bw : channelBW DL or UP
 * @param [in] scs : subcarrier spacing
 * @param [in] bandNR : FR1 NR band
 * @param [in] uplink : BW for uplink (TRUE/FALSE)
 * @return TRUE(exceptional case) / FALSE (not exceptional case)
 */
uint8_t VLIB_IsBandwidthSupportedFR1v1570_exceptionalCase(uint16_t bw, uint8_t scs, uint16_t bandNR, uint8_t uplink)
{
    AUTO_TRACE_FUNC;

    const bool isBand50Exception {
        (ci_nr_bandwidth_80MHz == bw) &&
        (ci_Nr_scs30 == scs || ci_Nr_scs60 == scs) &&
        (50 == bandNR)
    };

    const bool isBand70Exception {
        (ci_nr_bandwidth_20MHz == bw || ci_nr_bandwidth_25MHz == bw) &&
        (ci_Nr_scs15 == scs || ci_Nr_scs30 == scs || ci_Nr_scs60 == scs) &&
        (70 == bandNR)
    };

    if ((TRUE== uplink) && (isBand50Exception || isBand70Exception))
    {
        return TRUE;
    }

    return FALSE;
}

/**
 * @brief This function checks channel bandwidths (5MHz, 10MHz, 15MHz, 20MHz, 25MHz, 30MHz)
 *        supporting for each FR1 NR band in version 15.7.0 of 5.3.5-1 Table from 38.101-1 3gpp specification.
 * @param [in] bw : channelBW DL or UP
 * @param [in] scs : subcarrier spacing
 * @param [in] bandNR : FR1 NR band
 * @return TRUE(supported) / FALSE (BW not supported or not found)
 */
uint8_t VLIB_IsBandwidthSupportedFR1v1570_below30MHz(uint16_t bw, uint8_t scs, uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    if ((ci_nr_bandwidth_5MHz == bw) &&
        (ci_Nr_scs15 == scs))
    {
        switch (bandNR)
        {
            case 1:
            case 2:
            case 3:
            case 5:
            case 7:
            case 8:
            case 12:
            case 20:
            case 25:
            case 28:
            case 34:
            case 38:
            case 39:
            case 40:
            case 50:
            case 51:
            case 66:
            case 70:
            case 71:
            case 74:
            case 75:
            case 76:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (((ci_nr_bandwidth_10MHz == bw) ||
              (ci_nr_bandwidth_15MHz == bw)) &&
             ((ci_Nr_scs15 == scs) ||
              (ci_Nr_scs30 == scs)))
    {
        switch (bandNR)
        {
            case 1:
            case 2:
            case 3:
            case 5:
            case 7:
            case 8:
            case 12:
            case 20:
            case 25:
            case 28:
            case 34:
            case 38:
            case 39:
            case 40:
            case 41:
            case 50:
            case 66:
            case 70:
            case 71:
            case 74:
            case 75:
            case 77:
            case 78:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if ((ci_nr_bandwidth_10MHz == bw) ||
             (ci_nr_bandwidth_15MHz == bw))
    {
        switch (bandNR)
        {
            case 1:
            case 2:
            case 3:
            case 7:
            case 25:
            case 34:
            case 38:
            case 39:
            case 40:
            case 41:
            case 50:
            case 66:
            case 70:
            case 74:
            case 75:
            case 77:
            case 78:
            case 80:
            case 84:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if ((ci_nr_bandwidth_20MHz == bw) &&
            ((ci_Nr_scs15 == scs) ||
             (ci_Nr_scs30 == scs)))
    {
        switch (bandNR)
        {
            case 1:
            case 2:
            case 3:
            case 5:
            case 7:
            case 8:
            case 20:
            case 25:
            case 28:
            case 38:
            case 39:
            case 40:
            case 41:
            case 50:
            case 66:
            case 70:
            case 71:
            case 74:
            case 75:
            case 77:
            case 78:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (ci_nr_bandwidth_20MHz == bw)
    {
        switch (bandNR)
        {
            case 1:
            case 2:
            case 3:
            case 7:
            case 25:
            case 38:
            case 39:
            case 40:
            case 41:
            case 50:
            case 66:
            case 70:
            case 74:
            case 75:
            case 77:
            case 78:
            case 80:
            case 84:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (ci_nr_bandwidth_25MHz == bw)
    {
        switch (bandNR)
        {
            case 3:
            case 39:
            case 40:
            case 70:
            case 80:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (ci_nr_bandwidth_30MHz == bw)
    {
        switch (bandNR)
        {
            case 3:
            case 39:
            case 40:
            case 80:
                return TRUE;
            default:
                return FALSE;
        }
    }

    return FALSE;
}

/**
 * @brief This function checks channel bandwidths (40MHz, 50MHz, 60MHz, 80MHz, 90MHz, 100MHz)
 *        supporting for each FR1 NR band in version 15.7.0 of 5.3.5-1 Table from 38.101-1 3gpp specification.
 * @param [in] bw : channelBW DL or UP
 * @param [in] scs : subcarrier spacing
 * @param [in] bandNR : FR1 NR band
 * @return TRUE(supported) / FALSE (BW not supported or not found)
 */
uint8_t VLIB_IsBandwidthSupportedFR1v1570_above30MHz(uint16_t bw, uint8_t scs, uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    if (ci_nr_bandwidth_40MHz == bw)
    {
        switch (bandNR)
        {
            case 39:
            case 40:
            case 41:
            case 50:
            case 66:
            case 77:
            case 78:
            case 79:
            case 86:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (ci_nr_bandwidth_50MHz == bw)
    {
        switch (bandNR)
        {
            case 40:
            case 41:
            case 50:
            case 77:
            case 78:
            case 79:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if (((ci_nr_bandwidth_60MHz == bw) ||
              (ci_nr_bandwidth_80MHz == bw)) &&
              (ci_Nr_scs15 != scs))
    {
        switch (bandNR)
        {
            case 40:
            case 41:
            case 50:
            case 77:
            case 78:
            case 79:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if ((ci_nr_bandwidth_90MHz == bw) &&
             (ci_Nr_scs15 != scs))
    {
        switch (bandNR)
        {
            case 41:
            case 77:
            case 78:
                return TRUE;
            default:
                return FALSE;
        }
    }
    else if ((ci_nr_bandwidth_100MHz == bw) &&
             (ci_Nr_scs15 != scs))
    {
        switch (bandNR)
        {
            case 41:
            case 77:
            case 78:
            case 79:
                return TRUE;
            default:
                return FALSE;
        }
    }

    return FALSE;
}


/**
 * @brief This function checks channel bandwidths supporting for each FR1 NR band
 *        in version 15.7.0 of 5.3.5-1 Table from 38.101-1 3gpp specification.
 * @param [in] bw : channelBW DL or UP
 * @param [in] scs : subcarrier spacing
 * @param [in] bandNR : FR1 NR band
 * @param [in] uplink : BW for uplink (TRUE/FALSE)
 * @return TRUE(supported) / FALSE (BW not supported or not found)
 */
uint8_t VLIB_IsBandwidthSupportedFR1v1570(uint16_t bw, uint8_t scs, uint16_t bandNR, uint8_t uplink)
{
    AUTO_TRACE_FUNC;

    uint8_t result = FALSE;

    if (ci_Nr_scs60 < scs)
    {
        return FALSE;
    }

    if (ci_nr_bandwidth_100MHz < bw)
    {
        return FALSE;
    }

    // Exceptional cases supported only for downlink
    if (VLIB_IsBandwidthSupportedFR1v1570_exceptionalCase(bw, scs, bandNR, uplink))
    {
        return FALSE;
    }

    if (ci_nr_bandwidth_30MHz >= bw)
    {
        result = VLIB_IsBandwidthSupportedFR1v1570_below30MHz(bw, scs, bandNR);
    }
    else
    {
        result = VLIB_IsBandwidthSupportedFR1v1570_above30MHz(bw, scs, bandNR);
    }

    return result;
}

/**
 * @brief This function checks channel bandwidths supporting for each FR2 NR band
 *        in version 15.7.0 of 5.3.5-2 Table from 38.101-2 3gpp specification.
 * @param [in] bw : channelBW DL or UP
 * @param [in] scs : subcarrier spacing
 * @param [in] bandNR : FR2 NR band
 * @return TRUE(supported) / FALSE (BW not supported or not found)
 */
uint8_t VLIB_IsBandwidthSupportedFR2v1570(uint16_t bw, uint8_t scs, uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    if ((ci_nr_bandwidth_50MHz == bw) ||
        (ci_nr_bandwidth_100MHz == bw) ||
        (ci_nr_bandwidth_200MHz == bw))
    {
        if ((ci_Nr_scs60 == scs) ||
            (ci_Nr_scs120 == scs))
        {
            switch (bandNR)
            {
                case 257:
                case 258:
                case 260:
                case 261:
                    return TRUE;
                default:
                    return FALSE;
            }
        }
    }
    else if (ci_nr_bandwidth_400MHz == bw)
    {
        if (ci_Nr_scs120 == scs)
        {
            switch (bandNR)
            {
                case 257:
                case 258:
                case 260:
                case 261:
                    return TRUE;
                default:
                    return FALSE;
            }
        }
    }

    return FALSE;
}

/**
 * @brief This function checks NR FR1 / FR2 band is within the range
 *        of 5.3.5.1-2 Table from 38.101.1-2 v15.7.0 3gpp specification.
 * @param [in] bandNR : FR1/FR2 NR band
 * @return TRUE (NR band within the range of 38.101.1-2 v15.7.0)
 * @return FALSE (NR band out of range 38.101.1-2 v15.7.0)
 */
uint8_t  VLIB_IsBandIncludedInV1570(uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    switch (bandNR)
    {
        case 1:
        case 2:
        case 3:
        case 5:
        case 7:
        case 8:
        case 12:
        case 20:
        case 25:
        case 28:
        case 34:
        case 38:
        case 39:
        case 40:
        case 41:
        case 50:
        case 51:
        case 66:
        case 70:
        case 71:
        case 74 ... 84:
        case 86:
        case 257:
        case 258:
        case 260:
        case 261:
            return TRUE;
        default:
            return FALSE;
    }
}

/**
 *  @brief Function checks if 90MHz is supported for Dl
 *  @param callInfo_type* ptCi
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t isUe90MHzSupportDl(const callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    return (ptCi->ueInfo.ueCapa.nr.rf.bw90MHzSupportedDl && IS_ENDC(ptCi)) ? GOOD : BREAK;
}

/**
 *  @brief Function checks if 90MHz is supported for Ul
 *  @param callInfo_type* ptCi
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t isUe90MHzSupportUl(const callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    return (ptCi->ueInfo.ueCapa.nr.rf.bw90MHzSupportedUl && IS_ENDC(ptCi)) ? GOOD : BREAK;
}

/**
 *  @brief Function checks bandwidth mismatch allowed yang parameter:
 *         - NSA : restrict-pscell-change-bandwidth-mismatch
 *         - SA  : target-cell-bandwidth-mode
 *  @param [in] ptCi : pointer to callInfo_type
 *  @param [in] cellNum : cell num for checking yang parameter
 *  @param [in] ignoreMisMatchBWAllowed - if TRUE BW mismatch is always not allowed
 *  @param [in] rMisMatchBW_notAllowed - value of yang parameter to set by reference
 *  @return GOOD/ERROR
 */
int16_t VLIB_getBwMismatchAllowedParam(const callInfo_type *ptCi, const uint16_t cellNum,
    uint16_t ignoreMisMatchBWAllowed, uint16_t &rMisMatchBW_notAllowed)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(cellNum, MAX_CELL_NUM_VCCB);

    int16_t status = GOOD;
    uint8_t plmnIndex = 0;
    const VccbCellInfo_type *const pCell = &CELLINFO[cellNum];

    /* NSA condition */
    if (TRUE == ignoreMisMatchBWAllowed)
    {
        rMisMatchBW_notAllowed = TRUE;
    }
    else if (E_CALLTYPE_NSA == ptCi->callType)
    {
        status = VLIB_getMultipleGnbPlmnIndexInfo(ptCi, &plmnIndex, ptCi->ueInfo.servedPlmn.longId);
        if ((GOOD == status) &&
            (plmnIndex < MAX_PLMN_INDEX) &&
            (CI_use == pCell->mobilityConfig.plmn[plmnIndex].usage))
        {
            rMisMatchBW_notAllowed = pCell->mobilityConfig.plmn[plmnIndex].restrictPscellChangeBwMismatch;
        }
        else
        {
            rMisMatchBW_notAllowed = pCell->mobilityConfig.restrictPscellChangeBwMismatch;
        }
    }
    else if (E_CALLTYPE_SA == ptCi->callType)
    {
        if (ptCi->gnbNum < MAX_GNB_PER_SYS)
        {
            if (GNBINFO[ptCi->gnbNum].cfg.targetCellBandwidthMode == SaHandoverTargetCellBandwidthMode_1)
            {
                rMisMatchBW_notAllowed = TRUE;
            }
        }
    }

    // SVR23B FGR-MC1027 Connected mobility control for RedCap UE
    if (IS_REDCAP_UE(ptCi))
    {
        rMisMatchBW_notAllowed = TRUE;
    }

    VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi, "Bandwidth mismatch for [cellNum(%u), CI(%u)] allowed(%s).\n",
        cellNum, pCell->cellIdentity, rMisMatchBW_notAllowed ? "true" : "false");

    return GOOD;
}

/**
 * @brief Function checks specific scs-XXkHz entry presence in UE-NR-Capability
 * @param [in] scsFlagPresenceInUeCapa : pointer to ScsBitFlag_type
 * @param [in] cellScs : cell subcarrier spacing
 * @return FLAG_ON(specific scs present in UE Capa) / FLAG_OFF (specific scs absent in UE Capa)
 */
uint8_t VLIB_CheckScsPresenceEntryInUeCapa(const scsBitFlag_type *scsFlagPresenceInUeCapa, uint8_t cellScs)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_FALSE(scsFlagPresenceInUeCapa);

    switch(cellScs)
    {
        case ci_Nr_scs15:
            return scsFlagPresenceInUeCapa->scs_15;
        case ci_Nr_scs30:
            return scsFlagPresenceInUeCapa->scs_30;
        case ci_Nr_scs60:
            return scsFlagPresenceInUeCapa->scs_60;
        case ci_Nr_scs120:
            return scsFlagPresenceInUeCapa->scs_120;
        default:
            return FLAG_OFF;
    }
}

/**
 * @brief [FGR-CC5000][SVR25A] FR2206
 *        VCCB will check cell bandwidth UE supported by 38.101.1-2 Tab 5.3.5-1 in version 15.7.0 when all conditions were met:

 *        (1) SP Cell selection for EN-DC procedure.
 *        (2) channelBWs_DL/channelBWs_UL or specific scs-XXkHz entry not received in UE-NR-Capability
 *        (3) restrict-pscell-change-bandwidth-mismatch parameter set to enable (misMatchBW_notAllowed == TRUE)
 *        (4) bandwidth not in version 1590
 *        (5) 100MHz BW for band n41, n48, n77, n78, n79, n90 not applicable as defined in TS 38.101-1 [2]
 *        (6) band is included in 38.101-1 Tab 5.3.5-1 v1570
 * @param [in] ptCi : pointer to callInfo_type
 * @param [in] channelBWsDlUl_present_flag : channelBW DL or UL including flag in UE-NR-Capability
 * @param [in] specificScs_presence_flag : scs-XXkHz entry including flag in UE-NR-Capability
 * @param [in] misMatchBW_notAllowed : yang parameter value
 * @param [in] bandNR : FR1/FR2 NR band
 * @param [in] bw : channelBW DL or UP
 * @return TRUE(check by Tab 5.3.5.1-2 in version 15.7.0) / FALSE (check by legacy bahaviour)
 */
uint16_t VLIB_CheckBwByTabV1570(const callInfo_type *ptCi, uint8_t channelBWsDlUl_present_flag,
    uint8_t specificScs_presence_flag, uint16_t misMatchBW_notAllowed, uint16_t bandNR, uint16_t bw)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_FALSE(ptCi);

    // (1) Return if not EN-DC Addition
    if(FALSE == ptCi->bitFlag.isEndcSpCellSelectionProcedure) return FALSE;

    // (2) Return if channel BW and SCS parameters are present in UE capability
    if((FLAG_ON == channelBWsDlUl_present_flag) && (FLAG_ON == specificScs_presence_flag)) return FALSE;

    // (3) Return if mismatched BW allowed.
    if(FALSE == misMatchBW_notAllowed) return FALSE; 

    // (4) Return if v1590 version check by table shall be applied.
    if(MAX_FR1_NR_BAND >= bandNR) //FR1
    {
        if ((ci_nr_bandwidth_5MHz != bw) &&
            (ci_nr_bandwidth_10MHz != bw) &&
            (ci_nr_bandwidth_15MHz != bw) &&
            (ci_nr_bandwidth_20MHz != bw) &&
            (ci_nr_bandwidth_25MHz != bw) &&
            (ci_nr_bandwidth_30MHz != bw) &&
            (ci_nr_bandwidth_40MHz != bw) &&
            (ci_nr_bandwidth_50MHz != bw) &&
            (ci_nr_bandwidth_60MHz != bw) &&
            (ci_nr_bandwidth_80MHz != bw) &&
            (ci_nr_bandwidth_90MHz != bw) &&
            (ci_nr_bandwidth_100MHz != bw))
        {
            return FALSE;
        }
    }

    // (5) For 100MHz and band n41, n48, n77, n78, n79, n90 Tab 5.3.5-1 v1570 not applicable as defined in TS 38.101-1 [2].
    // Below bands for 100MHz will be always treated as supported.
    if ((ci_nr_bandwidth_100MHz == bw) &&
        ((41 == bandNR) ||
         (48 == bandNR) ||
         (77 == bandNR) ||
         (78 == bandNR) ||
         (79 == bandNR) ||
         (90 == bandNR)))
    {
        return FALSE;
    }

    // (6) Return if band is not included in 38.101-1/2 Tab 5.3.5-1 v1570.
    if(false == VLIB_IsBandIncludedInV1570(bandNR)) return FALSE;

    // Check by default table v1570.
    return TRUE;
}

/**
 *  @brief Function checks if cellBandwidthBits (cell bandwidth mask) is supported by ueBandwidthBits (UE supported
 *         bandwidths bits)
 *  @param callInfo_type* ptCi
 *  @param uint16_t pCellNum - primary cell num (needed when ignoreMisMatchBWAllowed = FALSE)
 *  @param uint16_t cellBandwidthBits - cell bandwidth mask
 *  @param uint16_t ueBandwidthBits - UE supported bandwidths bits
 *  @param int16_t misMatchBW_notAllowed - if TRUE BW mismatch is not allowed
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t VLIB_IsBandwidthSupported(const callInfo_type *ptCi, uint16_t cellBandwidthBits,
    uint16_t ueBandwidthBits, uint16_t misMatchBW_notAllowed)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    int16_t status = GOOD;

    if (0 == cellBandwidthBits)
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "Invalid BW masks: cellBwBits(0x%x), ueBwBits(0x%x)\n", cellBandwidthBits, ueBandwidthBits);
        return BREAK;
    }

    if (TRUE == misMatchBW_notAllowed)
    {
        if ((cellBandwidthBits & ueBandwidthBits) == 0)
        {
            VCCB_PRINT_CALL_COMMENT(ptCi,
                "No common bandwidth masks between cell and ue:cellBwBits(0x%x), ueBwBits(0x%x)\n", cellBandwidthBits,
                ueBandwidthBits);
            status = BREAK;
        }
    }
    else
    {
        if (ueBandwidthBits < cellBandwidthBits)
        {
            VCCB_PRINT_CALL_COMMENT(ptCi,
                "cellBandwidth is smaller than minimum UE supported bandwidth:cellBwBits(0x%x), ueBwBits(0x%x)\n",
                cellBandwidthBits, ueBandwidthBits);
            status = BREAK;
        }
    }

    return status;
}

/**
 *  @brief Function returns bitmask filter for checking UE capa for channelBW and channelBW_v1590 in BW ascending order.
 *  @param frequencyType - FR1/FR2
 *  @param bandwidth - channelBW for which filter should be prepared.
 *  @return uint16_t - bitmask filter
 */
uint16_t VLIB_getBandWidthFilter(enumFrequencyRange_type frequencyType, uint16_t bandwidth)
{
    AUTO_TRACE_FUNC;

    uint16_t filter = 0;
    if (FrequencyRange_FR1 == frequencyType)
    {
        switch (bandwidth)
        {
            case ci_nr_bandwidth_5MHz:
                filter = 0B100000000000;
                break;
            case ci_nr_bandwidth_10MHz:
                filter = 0B010000000000;
                break;
            case ci_nr_bandwidth_15MHz:
                filter = 0B001000000000;
                break;
            case ci_nr_bandwidth_20MHz:
                filter = 0B000100000000;
                break;
            case ci_nr_bandwidth_25MHz:
                filter = 0B000010000000;
                break;
            case ci_nr_bandwidth_30MHz:
                filter = 0B000001000000;
                break;
            case ci_nr_bandwidth_40MHz:
                filter = 0B000000100000;
                break;
            case ci_nr_bandwidth_50MHz:
                filter = 0B000000010000;
                break;
            case ci_nr_bandwidth_60MHz:
                filter = 0B000000001000;
                break;
            case ci_nr_bandwidth_70MHz:
                filter = 0B000000000100;
                break;
            case ci_nr_bandwidth_80MHz:
                filter = 0B000000000010;
                break;
            case ci_nr_bandwidth_90MHz:
                filter = 0B000000000001;
                break;
            default:
                break;
        }
    }
    else if (FrequencyRange_FR2 == frequencyType)
    {
        switch (bandwidth)
        {
            case ci_nr_bandwidth_50MHz:
                filter = 0B100;
                break;
            case ci_nr_bandwidth_100MHz:
                filter = 0B010;
                break;
            case ci_nr_bandwidth_200MHz:
                filter = 0B001;
                break;
            default:
                break;
        }
    }
    return filter;
}

/**
 *  @brief Function that combines channelBWs and channelBWs_v1590 into one sequence of bits in BW ascending order.
 *  @param uint16_t channelBWsBits
 *  @param uint16_t channelBWsBits_v1590
 *  @param bool support for 90MHz
 *  @return uint16_t combined sequence of bits
 */
uint16_t VLIB_getCombinedChannelBWsBits(uint16_t channelBWsBits, uint16_t channelBWsBits_v1590, const bool support90MHz)
{
    AUTO_TRACE_FUNC;
    /*
    ...other bits...   4    3    2    1    0
    ================+----+----+----+----+----+
    ...other BWs... | 50 | 60 | 70 | 80 | 90 |
    ================+----+----+----+----+----+
    */
    uint16_t bw70MHz = (channelBWsBits_v1590 >> 13) & 0b100;
    uint16_t bw80MHz = ((0b1 & channelBWsBits) << 1);
    uint16_t bw90MHz = support90MHz ? 1 : 0;

    uint16_t result = (static_cast<uint16_t>(channelBWsBits << 2) & 0xFFF8) | bw70MHz | bw80MHz | bw90MHz;

    return result;
}

/**
 * @brief This function gets Ue supported bandwidths sequence of bits for downlink.
 * @param [in] ptCi : pointer to callInfo
 * @param [in] pBandNrUeCapa : pointer to bandNrUeCapa
 * @param [in] frequencyType - FR1/FR2
 * @param [in] cellScs : cell subcarrier spacing
 * @param [in] pUeBandwidthBits : pointer to UE BW Bits for creating
 * @return GOOD/ERROR
 */
int16_t VLIB_getUeBandwidthBitsDl(const callInfo_type *ptCi, const bandNrUeCapa_type *pBandNrUeCapa,
    enumFrequencyRange_type frequencyType, uint8_t cellScs, uint16_t *pUeBandwidthBits)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pBandNrUeCapa);
    VALID_E_POINT_CHECK_ERROR(pUeBandwidthBits);

    *pUeBandwidthBits = 0;

    if (FrequencyRange_FR1 == frequencyType)
    {
        const bool ueSupport90MHzDl = isUe90MHzSupportDl(ptCi) == GOOD;

        if (ci_Nr_scs15 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_DL.frType.fr1.scs_15kHz,
                pBandNrUeCapa->channelBWs_DL_v1590.frType.fr1.scs_15kHz, ueSupport90MHzDl);
        }
        else if (ci_Nr_scs30 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_DL.frType.fr1.scs_30kHz,
                pBandNrUeCapa->channelBWs_DL_v1590.frType.fr1.scs_30kHz, ueSupport90MHzDl);
        }
        else if (ci_Nr_scs60 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_DL.frType.fr1.scs_60kHz,
                pBandNrUeCapa->channelBWs_DL_v1590.frType.fr1.scs_60kHz, ueSupport90MHzDl);
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid nrScsDl(%u) for cell FR1!\n", cellScs);
            return ERROR;
        }
    }
    else if (FrequencyRange_FR2 == frequencyType)
    {
        if (ci_Nr_scs60 == cellScs)
        {
            *pUeBandwidthBits = pBandNrUeCapa->channelBWs_DL.frType.fr2.scs_60kHz;
        }
        else if (ci_Nr_scs120 == cellScs)
        {
            *pUeBandwidthBits = pBandNrUeCapa->channelBWs_DL.frType.fr2.scs_120kHz;
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid nrScsDl(%u) for cell FR2!\n", cellScs);
            return ERROR;
        }
    }

    return GOOD;
}

/**
 * @brief This function gets Ue supported bandwidths sequence of bits for uplink.
 * @param [in] ptCi : pointer to callInfo
 * @param [in] pBandNrUeCapa : pointer to bandNrUeCapa
 * @param [in] frequencyType - FR1/FR2
 * @param [in] cellScs : cell subcarrier spacing
 * @param [in] pUeBandwidthBits : pointer to UE BW Bits for creating
 * @return GOOD/ERROR
 */
int16_t VLIB_getUeBandwidthBitsUl(const callInfo_type *ptCi, const bandNrUeCapa_type *pBandNrUeCapa,
    enumFrequencyRange_type frequencyType, uint8_t cellScs, uint16_t *pUeBandwidthBits)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pBandNrUeCapa);
    VALID_E_POINT_CHECK_ERROR(pUeBandwidthBits);

    *pUeBandwidthBits = 0;

    if (FrequencyRange_FR1 == frequencyType)
    {
        const bool ueSupport90MHzUl = isUe90MHzSupportUl(ptCi) == GOOD;

        if (ci_Nr_scs15 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_UL.frType.fr1.scs_15kHz,
                pBandNrUeCapa->channelBWs_UL_v1590.frType.fr1.scs_15kHz, ueSupport90MHzUl);
        }
        else if (ci_Nr_scs30 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_UL.frType.fr1.scs_30kHz,
                pBandNrUeCapa->channelBWs_UL_v1590.frType.fr1.scs_30kHz, ueSupport90MHzUl);
        }
        else if (ci_Nr_scs60 == cellScs)
        {
            *pUeBandwidthBits = VLIB_getCombinedChannelBWsBits(pBandNrUeCapa->channelBWs_UL.frType.fr1.scs_60kHz,
                pBandNrUeCapa->channelBWs_UL_v1590.frType.fr1.scs_60kHz, ueSupport90MHzUl);
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid nrScsUl(%u) for cell FR1!\n", cellScs);
            return ERROR;
        }
    }
    else if (FrequencyRange_FR2 == frequencyType)
    {
        if (ci_Nr_scs60 == cellScs)
        {
            *pUeBandwidthBits = pBandNrUeCapa->channelBWs_UL.frType.fr2.scs_60kHz;
        }
        else if (ci_Nr_scs120 == cellScs)
        {
            *pUeBandwidthBits = pBandNrUeCapa->channelBWs_UL.frType.fr2.scs_120kHz;
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid nrScsUl(%u) for cell FR2!\n", cellScs);
            return ERROR;
        }
    }

    return GOOD;
}

/**
 *  @brief Function that verifies whether INTER_NB cell DL bandwidth matches ChannelBWs in UE Capa
 *  @param callInfo_type* ptCi - pointer to callInfo
 *  @param nbrGnbCellInfo_type* pNbrGnbCell - pointer to neighbor cell
 *  @param uint16_t bandNR - bandNR to be checked
 *  @param uint16_t cellNum - primary cell num (needed when ignoreMisMatchBWAllowed = FALSE)
 *  @param uint16_t ignoreMisMatchBWAllowed - if TRUE BW mismatch is always not allowed
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t VLIB_checkIntraNRChannelBWDl(const callInfo_type *ptCi, const nbrGnbCellInfo_type *pNbrGnbCell, uint16_t bandNR,
    uint16_t pCellNum, uint16_t ignoreMisMatchBWAllowed)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pNbrGnbCell);

    int16_t status = GOOD;
    uint16_t misMatchBW_notAllowed = FALSE;
    uint16_t bandNRidx = static_cast<uint16_t>(bandNR - 1);
    VCCB_CHECK_E_RANGE_AND_RETURN(bandNRidx, MAX_NUM_NR_BAND);

    if (MAX_NR_ARFCN == pNbrGnbCell->nrArfcnDl)  // SUL cell
    {
        return GOOD;
    }

    uint16_t bandwidthDl = pNbrGnbCell->bandwidthDl;
    if (0 == bandwidthDl)
    {
        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
            "Cell (cellId = %u) has incorrect DL nrb nrScs values. DL: nrb(%u), scs(%u)!!\n", pNbrGnbCell->cellId,
            pNbrGnbCell->nrbDl, pNbrGnbCell->nrScsDl);
        return GOOD;
    }

    // SVR23B FGR-MC1027 Connected mobility control for RedCap UE
    if (IS_REDCAP_UE(ptCi))
    {
        uint16_t maxRedCapBwDl = (IS_FR1(pNbrGnbCell->nrArfcnDl)) ? ci_nr_bandwidth_20MHz : ci_nr_bandwidth_100MHz;

        if (bandwidthDl > maxRedCapBwDl)
            bandwidthDl = maxRedCapBwDl;
    }

    if (ERROR == VLIB_getBwMismatchAllowedParam(ptCi, pCellNum, ignoreMisMatchBWAllowed, misMatchBW_notAllowed))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_getBwMismatchAllowedParam Error!\n");
        return ERROR;
    }

    for (bandNRidx = 0; bandNRidx < UMIN(ptCi->ueInfo.ueCapa.nr.rf.nrOfBands, MAX_NUM_UECAPA_NR_BANDS); bandNRidx++)
    {
        if (bandNR != ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx].bandNR)
        {
            continue;
        }

        const bandNrUeCapa_type *pBandNrUeCapa = &ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx];
        uint16_t cellBandwidthBits = 0, ueBandwidthBits = 0;

        // FR1
        if (FnVCCB_IsFr1Freq(pNbrGnbCell->nrArfcnDl))
        {
            uint8_t specificScs_presence_flag = 
                VLIB_CheckScsPresenceEntryInUeCapa(&pBandNrUeCapa->channelBWs_DL.frType.fr1.scs_flag, pNbrGnbCell->nrScsDl);
            if(VLIB_CheckBwByTabV1570(ptCi, pBandNrUeCapa->channelBWs_DL_present_flag, specificScs_presence_flag, misMatchBW_notAllowed,
                   bandNR, bandwidthDl))
            {
                VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                    "Checking cell bandwidth supported by 38.101.1 v15.7.0: FR1 CellDl BW(%u), scs(%u), bandNR(%u)\n",
                     bandwidthDl, pNbrGnbCell->nrScsDl, bandNR);
                if (FALSE == VLIB_IsBandwidthSupportedFR1v1570 (bandwidthDl, pNbrGnbCell->nrScsDl, bandNR, FALSE))
                {
                    VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                        "UE doesn't support FR1 CellDl BW(%u), scs(%u), bandNR(%u)\n", bandwidthDl, pNbrGnbCell->nrScsDl, bandNR);

                    // FGR2206: SgNB Addition Request Reject with "No radio resource available in target cell" X2 cause.
                    //          It was covered by FR2200 and ptCi->bitFlag.isSgNBAddReqRejectNoRadioResourcesAvailableInTargetCell
                    //          flag will be setting to TRUE
                    status = BREAK;
                    break;
                }
            }
            else //Checking by channelBWs_DL & channelBWs_DL_v1590
            {
                // The assumption is that UE always supports 100MHz BW. Cell BW greater than 80MHz (90MHz for ENDC) means it equals 100MHz.
                // Then no further check is needed and BW requirement is met.
                if ((ci_nr_bandwidth_80MHz >= bandwidthDl) ||
                    ((ci_nr_bandwidth_90MHz >= bandwidthDl) && (true == IS_ENDC(ptCi))))
                {
                    // check FR1 DL 5-80 MHz (5-90 MHz for ENDC)
                    cellBandwidthBits = VLIB_getBandWidthFilter(FrequencyRange_FR1, bandwidthDl);

                    if (GOOD != VLIB_getUeBandwidthBitsDl(ptCi, pBandNrUeCapa, FrequencyRange_FR1,
                                    pNbrGnbCell->nrScsDl, &ueBandwidthBits))
                    {
                        status = ERROR;
                        break;
                    }

                    if (GOOD != VLIB_IsBandwidthSupported(
                                    ptCi, cellBandwidthBits, ueBandwidthBits, misMatchBW_notAllowed))
                    {
                        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                            "UE doesn't support FR1 CellDl BW(%u), scs(%u),ueBandwidthBits(0x%x),cellBandwidthBits(0x%x), "
                            "bandNR(%u)\n",
                            bandwidthDl, pNbrGnbCell->nrScsDl, ueBandwidthBits, cellBandwidthBits, bandNR);
                        status = BREAK;
                        break;
                    }
                }
            }
        }
        else if (FnVCCB_IsFr2Freq(pNbrGnbCell->nrArfcnDl))
        {
             uint8_t specificScs_presence_flag = 
                VLIB_CheckScsPresenceEntryInUeCapa(&pBandNrUeCapa->channelBWs_DL.frType.fr2.scs_flag, pNbrGnbCell->nrScsDl);
            if (VLIB_CheckBwByTabV1570(ptCi, pBandNrUeCapa->channelBWs_DL_present_flag, specificScs_presence_flag, misMatchBW_notAllowed,
                    bandNR, bandwidthDl))
            {
                VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                    "Checking cell bandwidth supported by 38.101.2 v15.7.0: FR2 CellDl BW(%u), scs(%u), bandNR(%u)\n",
                     bandwidthDl, pNbrGnbCell->nrScsDl, bandNR);
                if (FALSE == VLIB_IsBandwidthSupportedFR2v1570 (bandwidthDl, pNbrGnbCell->nrScsDl, bandNR))
                {
                    VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                        "UE doesn't support FR2 CellDl BW(%u), scs(%u), bandNR(%u)\n", bandwidthDl, pNbrGnbCell->nrScsDl, bandNR);

                    // FGR2206: SgNB Addition Request Reject with "No radio resource available in target cell" X2 cause.
                    //          It was covered by FR2200 and ptCi->bitFlag.isSgNBAddReqRejectNoRadioResourcesAvailableInTargetCell
                    //          flag will be setting to TRUE

                    status = BREAK;
                    break;
                }
            }
            else //Checking by channelBWs_DL & channelBWs_DL_v1590
            {
                if ((FrequencyRange_FR2 == pBandNrUeCapa->channelBWs_DL.fr_flag) &&
                    ((ci_nr_bandwidth_200MHz >= bandwidthDl) && (bandwidthDl > 0)))
                {
                    // check FR2 DL 50-200 MHz
                    cellBandwidthBits = VLIB_getBandWidthFilter(FrequencyRange_FR2, bandwidthDl);

                    if (GOOD != VLIB_getUeBandwidthBitsDl(ptCi, pBandNrUeCapa, FrequencyRange_FR2,
                                    pNbrGnbCell->nrScsDl, &ueBandwidthBits))
                    {
                        status = ERROR;
                        break;
                    }

                    if (GOOD != VLIB_IsBandwidthSupported(
                                    ptCi, cellBandwidthBits, ueBandwidthBits, misMatchBW_notAllowed))
                    {
                        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                            "UE doesn't support FR2 CellDl BW(%u), scs(%u),ueBandwidthBits(0x%x),cellBandwidthBits(0x%x), "
                            "bandNR(%u)\n",
                            bandwidthDl, pNbrGnbCell->nrScsDl, ueBandwidthBits, cellBandwidthBits, bandNR);
                        status = BREAK;
                        break;
                    }
                }
            }
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid Frequency Range Id for nrArfcnDl[%u]!\n", pNbrGnbCell->nrArfcnDl);
            status = ERROR;
            break;
        }

        break;  // break as there shouldn't be bandNR duplications.
    }

    return status;
}

/**
 *  @brief Function that verifies whether INTER_NB cell UL bandwidth matches ChannelBWs in UE Capa
 *  @param callInfo_type* ptCi - pointer to callInfo
 *  @param nbrGnbCellInfo_type* pNbrGnbCell - pointer to neighbor cell
 *  @param uint16_t bandNR - bandNR to be checked
 *  @param uint16_t cellNum - primary cell num (needed when ignoreMisMatchBWAllowed = FALSE)
 *  @param uint16_t ignoreMisMatchBWAllowed - if TRUE BW mismatch is always not allowed
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t VLIB_checkIntraNRChannelBWUl(const callInfo_type *ptCi, const nbrGnbCellInfo_type *pNbrGnbCell, uint16_t bandNR,
    uint16_t pCellNum, uint16_t ignoreMisMatchBWAllowed)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pNbrGnbCell);

    int16_t status = GOOD;
    uint16_t misMatchBW_notAllowed = FALSE;
    uint16_t bandNRidx = static_cast<uint16_t>(bandNR - 1);
    VCCB_CHECK_E_RANGE_AND_RETURN(bandNRidx, MAX_NUM_NR_BAND);

    if (MAX_NR_ARFCN == pNbrGnbCell->nrArfcnUl)  // SDL cell
    {
        return GOOD;
    }

    uint16_t bandwidthUl = pNbrGnbCell->bandwidthUl;
    if (0 == bandwidthUl)
    {
        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
            "Cell (cellId = %u) has incorrect UL nrb nrScs values. UL: nrb(%u), scs(%u)!\n", pNbrGnbCell->cellId,
            pNbrGnbCell->nrbUl, pNbrGnbCell->nrScsUl);
        return GOOD;
    }

    // SVR23B FGR-MC1027 Connected mobility control for RedCap UE
    if (IS_REDCAP_UE(ptCi))
    {
        uint16_t maxRedCapBwUl = (IS_FR1(pNbrGnbCell->nrArfcnUl)) ? ci_nr_bandwidth_20MHz : ci_nr_bandwidth_100MHz;

        if (bandwidthUl > maxRedCapBwUl)
            bandwidthUl = maxRedCapBwUl;
    }

    if (ERROR == VLIB_getBwMismatchAllowedParam(ptCi, pCellNum, ignoreMisMatchBWAllowed, misMatchBW_notAllowed))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_getBwMismatchAllowedParam Error!\n");
        return ERROR;
    }

    for (bandNRidx = 0; bandNRidx < UMIN(ptCi->ueInfo.ueCapa.nr.rf.nrOfBands, MAX_NUM_UECAPA_NR_BANDS); bandNRidx++)
    {
        if (bandNR != ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx].bandNR)
        {
            continue;
        }

        const bandNrUeCapa_type *pBandNrUeCapa = &ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx];
        uint16_t cellBandwidthBits = 0, ueBandwidthBits = 0;

        // FR1
        if (FnVCCB_IsFr1Freq(pNbrGnbCell->nrArfcnUl))
        {
            uint8_t specificScs_presence_flag = 
                VLIB_CheckScsPresenceEntryInUeCapa(&pBandNrUeCapa->channelBWs_UL.frType.fr1.scs_flag, pNbrGnbCell->nrScsUl);
            if(VLIB_CheckBwByTabV1570(ptCi, pBandNrUeCapa->channelBWs_UL_present_flag, specificScs_presence_flag, misMatchBW_notAllowed,
                  bandNR, bandwidthUl))
            {
                VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                    "Checking cell bandwidth supported by 38.101.1 v15.7.0: FR1 CellUl BW(%u), scs(%u), bandNR(%u)\n",
                    bandwidthUl, pNbrGnbCell->nrScsUl, bandNR);
                if (FALSE == VLIB_IsBandwidthSupportedFR1v1570 (bandwidthUl, pNbrGnbCell->nrScsUl, bandNR, TRUE))
                {
                    VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                        "UE doesn't support FR1 CellUl BW(%u), scs(%u), bandNR(%u)\n", bandwidthUl, pNbrGnbCell->nrScsUl, bandNR);

                    // FGR2206: SgNB Addition Request Reject with "No radio resource available in target cell" X2 cause.
                    //          It was covered by FR2200 and ptCi->bitFlag.isSgNBAddReqRejectNoRadioResourcesAvailableInTargetCell
                    //          flag will be setting to TRUE

                    status = BREAK;
                    break;
                }
            }
            else //Checking by channelBWs_UL & channelBWs_UL_v1590
            {
                // The assumption is that UE always supports 100MHz BW. Cell BW greater than 80MHz (90MHz for ENDC) means it equals 100MHz.
                // Then no further check is needed and BW requirement is met.
                if ((ci_nr_bandwidth_80MHz >= bandwidthUl) ||
                   ((ci_nr_bandwidth_90MHz >= bandwidthUl) && (true == IS_ENDC(ptCi))))
                {
                    // check FR1 UL 5-80 MHz (5-90 MHz for ENDC)
                    cellBandwidthBits = VLIB_getBandWidthFilter(FrequencyRange_FR1, bandwidthUl);

                    if (GOOD != VLIB_getUeBandwidthBitsUl(ptCi, pBandNrUeCapa, FrequencyRange_FR1,
                                    pNbrGnbCell->nrScsUl, &ueBandwidthBits))
                    {
                        status = ERROR;
                        break;
                    }

                    if (GOOD != VLIB_IsBandwidthSupported(
                                    ptCi, cellBandwidthBits, ueBandwidthBits, misMatchBW_notAllowed))
                    {
                        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                            "UE doesn't support FR1 CellUl BW(%u), scs(%u),ueBandwidthBits(0x%x),cellBandwidthBits(0x%x), "
                            "bandNR(%u)\n",
                            bandwidthUl, pNbrGnbCell->nrScsUl, ueBandwidthBits, cellBandwidthBits, bandNR);
                        status = BREAK;
                        break;
                    }
                }
            }
        }
        else if (FnVCCB_IsFr2Freq(pNbrGnbCell->nrArfcnUl))
        {
            uint8_t specificScs_presence_flag = 
                VLIB_CheckScsPresenceEntryInUeCapa(&pBandNrUeCapa->channelBWs_UL.frType.fr2.scs_flag, pNbrGnbCell->nrScsUl);
            if(VLIB_CheckBwByTabV1570(ptCi, pBandNrUeCapa->channelBWs_UL_present_flag, specificScs_presence_flag, misMatchBW_notAllowed,
                   bandNR, bandwidthUl))
            {
                VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                    "Checking cell bandwidth supported by 38.101.2 v15.7.0: FR2 CellUl BW(%u), scs(%u), bandNR(%u)\n",
                     bandwidthUl, pNbrGnbCell->nrScsUl, bandNR);
                if (FALSE == VLIB_IsBandwidthSupportedFR2v1570 (bandwidthUl, pNbrGnbCell->nrScsUl, bandNR))
                {
                    VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                        "UE doesn't support FR2 CellUl BW(%u), scs(%u), bandNR(%u)\n", bandwidthUl, pNbrGnbCell->nrScsUl, bandNR);

                    // FGR2206: SgNB Addition Request Reject with "No radio resource available in target cell" X2 cause.
                    //          It was covered by FR2200 and ptCi->bitFlag.isSgNBAddReqRejectNoRadioResourcesAvailableInTargetCell
                    //          flag will be setting to TRUE

                    status = BREAK;
                    break;
                }
            }
            else //Checking by channelBWs_UL & channelBWs_UL_v1590
            {
                if ((FrequencyRange_FR2 == pBandNrUeCapa->channelBWs_UL.fr_flag) &&
                    (ci_nr_bandwidth_200MHz >= bandwidthUl) && (bandwidthUl > 0))
                {
                    // check FR2 UL 50-200 MHz
                    cellBandwidthBits = VLIB_getBandWidthFilter(FrequencyRange_FR2, bandwidthUl);

                    if (GOOD != VLIB_getUeBandwidthBitsUl(ptCi, pBandNrUeCapa, FrequencyRange_FR2,
                                    pNbrGnbCell->nrScsUl, &ueBandwidthBits))
                    {
                        status = ERROR;
                        break;
                    }

                    if (GOOD != VLIB_IsBandwidthSupported(
                                    ptCi, cellBandwidthBits, ueBandwidthBits, misMatchBW_notAllowed))
                    {
                        VCCB_PRINT_CALL_PRINT_FOR_BW(ptCi,
                            "UE doesn't support FR2 CellUl BW(%u), scs(%u),ueBandwidthBits(0x%x),cellBandwidthBits(0x%x), "
                            "bandNR(%u)\n",
                            bandwidthUl, pNbrGnbCell->nrScsUl, ueBandwidthBits, cellBandwidthBits, bandNR);
                        status = BREAK;
                        break;
                    }
                }
            }
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid Frequency Range Id for nrArfcnDl[%u]!\n", pNbrGnbCell->nrArfcnUl);
            status = ERROR;
            break;
        }

        break;  // break as there shouldn't be bandNR duplications.
    }

    return status;
}

/**
 *  @brief Function that verifies whether INTER_NB cell bandwidth matches ChannelBWs in UE Capa
 *  @param callInfo_type* ptCi - pointer to callInfo
 *  @param nbrGnbCellInfo_type* pNbrGnbCell - pointer to neighbor cell
 *  @param uint16_t bandNR - bandNR to be checked
 *  @param uint16_t cellNum - primary cell num (needed when ignoreMisMatchBWAllowed = FALSE)
 *  @param uint16_t ignoreMisMatchBWAllowed - if TRUE BW mismatch is always not allowed
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t VLIB_checkIntraNRChannelBWs(const callInfo_type *ptCi, const nbrGnbCellInfo_type *pNbrGnbCell, uint16_t bandNR,
    uint16_t pCellNum, uint16_t ignoreMisMatchBWAllowed)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(pNbrGnbCell);

    int16_t status = VLIB_checkIntraNRChannelBWDl(ptCi, pNbrGnbCell, bandNR, pCellNum, ignoreMisMatchBWAllowed);
    if (GOOD != status)
    {
        return status;
    }

    status = VLIB_checkIntraNRChannelBWUl(ptCi, pNbrGnbCell, bandNR, pCellNum, ignoreMisMatchBWAllowed);
    if (GOOD != status)
    {
        return status;
    }

    return GOOD;
}

/**
 *  @brief Function that verifies whether target cell bandwidth matches ChannelBWs in UE Capa
 *  @param callInfo_type* ptCi
 *  @param uint16_t cellNum
 *  @param uint16_t bandNR
 *  @return int16_t GOOD/BREAK/ERROR (1/0/-1)
 */
int16_t VLIB_checkChannelBWs(callInfo_type *ptCi, uint16_t cellNum, uint16_t bandNR)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(cellNum, MAX_CELL_NUM_VCCB);

    int16_t status;
    nbrGnbCellInfo_type *pNbrGnbCell;

    pNbrGnbCell = &(CELLINFO[cellNum].intraGnbNbrInfo);

    status = VLIB_checkIntraNRChannelBWs(ptCi, pNbrGnbCell, bandNR, cellNum, FALSE);

    return status;
}

/**
 *  @brief Checks DSS feature support on different level (gNB, gNB band list, cell, band, UE). Validation component can be selected by manipulation function input parameters. Return GOOD if validation succeeded.
 *  @param callInfo_type* ptCi          : call context pointer
 *  @param uint16_t bandNbr             : band number or BAND_NOT_SET for extraction from ptCell
 *  @param bool checkGnbBandList        : true for validation DSS bands in gNB list, false otherwise
 *  @param VccbCellInfo_type* ptCell    : required if bandNbr==BAND_NOT_SET or cell level dss support validation required otherwise NULL
 *  @return int16_t GOOD/ERROR
 */
int16_t VLIB_checkDssAllowed(callInfo_type *ptCi, uint16_t bandNbr, bool checkGnbBandList, VccbCellInfo_type *ptCell)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);

    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];
    uint16_t bandNR = bandNbr;

    if (CI_no_use == pGnb->dssConfiguration.dssUESupportControl)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "[DSS] Feature disabled\n");
        return GOOD;
    }

    if (BAND_NOT_SET == bandNR)
    {
        // extract bandNR from ptCell
        VALID_E_POINT_CHECK_ERROR(ptCell);
        bandNR = VLIB_GetFreqBandIndForCell(ptCi, ptCell);
        if (BAND_NOT_FOUND == bandNR) // no common band found
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "[DSS] UE and Cell do not have common frequency band\n");
            ptCi->dssCauseRelease = TRUE;
            return ERROR;
        }
    }

    if ( (NULL != ptCell) && (0 == ptCell->dssEnabled) )
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "[DSS] Cell does not support DSS functionality\n");
        return GOOD;
    }

    uint16_t bandNRidx = static_cast<uint16_t>(bandNR - 1);
    VCCB_CHECK_E_RANGE_AND_RETURN(bandNRidx, MAX_NUM_NR_BAND);

    if ( (!checkGnbBandList) | ((pGnb->dssConfiguration.dssBands[bandNRidx/BANDS_PER_ARRAY_ENTRY] >>
             (bandNRidx%BANDS_PER_ARRAY_ENTRY))&0x01))
    {
        for (bandNRidx = 0; bandNRidx < UMIN(ptCi->ueInfo.ueCapa.nr.rf.nrOfBands, MAX_NUM_UECAPA_NR_BANDS); bandNRidx++)
        {
            if (ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx].bandNR == bandNR)
            {
                if (ptCi->ueInfo.ueCapa.nr.rf.supportedBandListNR[bandNRidx].rateMatchingLTE_CRS == FLAG_ON)
                {
                    VCCB_PRINT_CALL_COMMENT(ptCi, "[DSS] UE support rateMatchingLTE_CRS on selected band(%d)\n", bandNR);
                    return GOOD;
                }

                VCCB_PRINT_CALL_ERROR(ptCi, "[DSS] UE does not support rateMatchingLTE_CRS on selected band(%d)\n", bandNR);
                return ERROR;
            }
        }
    }
    else
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "[DSS] No band(%d) entry in gNB DSS band list\n", bandNR);
        return GOOD;
    }

    VCCB_PRINT_CALL_COMMENT(ptCi, "[DSS] UE does not support selected band(%d)\n", bandNR);
    return ERROR;
}

/**
 * @brief Check if current operation of modification involves SPLIT bearer configuration (either to split or from split)
 * @param [in]  ptCi                                : callInfo DB
 * @return      GOOD                                : At least one modified bearer involves split config
 * @return      ERROR                               : None of the bearers is modified from or to split config
 */

int16_t VLIB_CheckDrbTypeModInvolvingSplitOngoing(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    int16_t erabIdx = 0;

    for (erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
    {
        if (!IS_VALID_NODE(ERAB_NODE, ptCi, &(ptCi->erabInfo[erabIdx])))
        {
            continue;
        }

        if (ERAB_Modify != ptCi->erabInfo[erabIdx].erabStatus)
        {
            continue;
        }

        if ((SPLIT_Bearer == ptCi->erabInfo[erabIdx].bearerType ||
                SPLIT_Bearer == ptCi->erabInfo[erabIdx].preveBearerType) &&
            ptCi->erabInfo[erabIdx].bearerType != ptCi->erabInfo[erabIdx].preveBearerType)
        {
            return GOOD;
        }
    }

    return ERROR;
}

/**
 * @brief Check if current operation of modification involves only SN MCG to setup
 * @param [in]  ptCi                                : callInfo DB
 * @return      TRUE                                : Only SN MCG bearers to setup, modify or release
 * @return      FALSE                               : Other types of bearers to Add Mod Rel
 * @return      ERROR                               : ptCi pointer check error
 */
int16_t VLIB_CheckDrbTypeSN_MCG_AddModReleaseOnly(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    uint16_t erabIdx = 0;
    uint16_t SN_MCG_erab_ToSetupNum = 0;
    uint16_t SN_MCG_erab_ToModReleaseNum = 0;
    uint16_t other_erab_ToSetupNum = 0;
    uint16_t erab_ToModRelNum = 0;

    for (erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
    {
        if (!IS_VALID_NODE(ERAB_NODE, ptCi, &(ptCi->erabInfo[erabIdx])))
        {
            continue;
        }

        if (ERAB_Setup == ptCi->erabInfo[erabIdx].prevErabStatus && ERAB_Modify == ptCi->erabInfo[erabIdx].erabStatus)
        {
            if (SN_Terminated == ptCi->erabInfo[erabIdx].pDCPatSgNB && MCG_Bearer == ptCi->erabInfo[erabIdx].bearerType)
            {
                SN_MCG_erab_ToSetupNum++;
            }
        }
        else if (ERAB_Setup == ptCi->erabInfo[erabIdx].erabStatus)
        {
            other_erab_ToSetupNum++;
        }
        else if (ERAB_Modify == ptCi->erabInfo[erabIdx].erabStatus ||
                 ERAB_Release == ptCi->erabInfo[erabIdx].erabStatus)
        {
            if (MCG_Bearer == ptCi->erabInfo[erabIdx].bearerType &&
                MCG_Bearer == ptCi->erabInfo[erabIdx].preveBearerType)
            {
                SN_MCG_erab_ToModReleaseNum++;
            }
            else
            {
                erab_ToModRelNum++;
            }
        }
    }

    if ((SN_MCG_erab_ToSetupNum > 0 || SN_MCG_erab_ToModReleaseNum > 0) && other_erab_ToSetupNum == 0 &&
        erab_ToModRelNum == 0)
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

/**
 * @brief Check if current operation of modification need to setup F1 Ue Context.
 * @param [in]  ptCi                                : callInfo DB
 * @return      TRUE                                : Need to setup F1 link, some bearers are adding/modifing to
 * SCG/Split
 * @return      FALSE                               : Only SN MCG bearers on call
 * @return      ERROR                               : ptCi pointer check error
 */
int16_t VXIL_Check_F1UeContextToSetup(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    uint16_t erabIdx = 0;

    // No need to establish F1 Ue Context for given UE
    if (GET_F1_RESOURCE_FLAG(ptCi->flag.setupFlag))
    {
        return FALSE;
    }

    for (erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
    {
        if (!IS_VALID_NODE(ERAB_NODE, ptCi, &(ptCi->erabInfo[erabIdx])))
        {
            continue;
        }
        // If at least one bearer adding/modifing to SCG/Split and F1 Ue context did not established
        if ((ERAB_Modify == ptCi->erabInfo[erabIdx].erabStatus || ERAB_Setup == ptCi->erabInfo[erabIdx].erabStatus) &&
            (SPLIT_Bearer == ptCi->erabInfo[erabIdx].bearerType || SCG_Bearer == ptCi->erabInfo[erabIdx].bearerType))
        {
            return TRUE;
        }
    }

    return FALSE;
}

int8_t FnVCCB_IsDrbEstablishedScenario(callInfo_type *ptCi, callDRBInfo_type *ptDi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_FALSE(ptCi);
    VALID_E_POINT_CHECK_FALSE(ptDi);

    if (ptDi->drbStatus == S_Setup)
    {
        switch (ptCi->stateCmd)
        {
            case RRCConnection_cmd:
            case PDUSetup_cmd:
            case PDUModify_cmd:
                return TRUE;

            default:
                return FALSE;
        }
    }
    else  // ptDi->drbStatus != S_Setup
    {
        return FALSE;
    }
}

int16_t VCCB_InitializePdcpAllSRBs(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    callSRBInfo_type *ptSi = NULL;
    int srbIdx;
    for (srbIdx = 0; srbIdx < MAX_SRB_PER_CALL; srbIdx++)
    {
        ptSi = &(ptCi->srbInfo[srbIdx]);
        if (ptSi->usage != CI_use) continue;
        if (0 != FnPDCP_C_Init(&ptSi->pdcpRbContext, ptSi->rbInfo.rbId))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "RbId(%d) FnPDCP_C_Init ERROR!\n", ptSi->rbInfo.rbId);
            return ERROR;
        }
    }
    return GOOD;
}

enumEstablishmentCause_type FnVCCB_GetEstablishmentCause(const callInfo_type &callInfo)
{
    AUTO_TRACE_FUNC;
    enumEstablishmentCause_type rRCEstablishmentCause;

    // Fill rrcEstablishCause for gnb-counter
    // BasicOperation for NR SA FR1120: to handle exceptional cases
    // case1. Normal Case       - Refer to 'establishmentCause' included in the RRCSetupRequest.
    // case2. Inter-gNB H/O or RRE - Set to 'ci_relocatedCall'(initial value) since it does not have
    // establishmentCause in RRCSetupRequest,

    enumEstablishmentCause_type uEEstablishmentCause = static_cast<enumEstablishmentCause_type>(
        callInfo.ueInfo.establishmentCause);

    if (uEEstablishmentCause > ci_mcs_PriorityAccess)
    {
        rRCEstablishmentCause = ci_relocatedCall;
    }
    else
    {
        rRCEstablishmentCause = uEEstablishmentCause;
    }

    VCCB_PRINT_CALL_DATA(&callInfo, "establishmentCause:%d,result:%d in %s\n", uEEstablishmentCause,
        rRCEstablishmentCause, __func__);

    return rRCEstablishmentCause;
}

enumResumeCause_type FnVCCB_GetResumeCause(const callInfo_type &callInfo)
{
    AUTO_TRACE_FUNC;

    enumResumeCause_type rRCResumeCause = static_cast<enumResumeCause_type>(
        callInfo.ueInfo.RrcInactiveInfo.resumeCause);

    if (rRCResumeCause >= MAX_VALID_NR_RRC_RESUME_CAUSE)
    {
        VCCB_PRINT_CALL_ERROR(&callInfo, "Invalid resumeCause(%d) ERROR in %s\n", rRCResumeCause, __func__);
    }

    VCCB_PRINT_CALL_DATA(&callInfo, "resumeCause:%d in %s\n", rRCResumeCause, __func__);

    return rRCResumeCause;
}

void VLIB_CollectRetainStat_IntraGnbHO(callInfo_type *ptCi)  // RETAIN_STAT_HO_21C
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t callDrbIdx;
    callDRBInfo_type *ptDi = NULL;

    if (!IS_INTRA_GNB_HO())
    {
        return;
    }

    /* UE CONTEXT */
    MACRO_STAT_PMCC_RETAIN(STAT_SETUP_UECONTEXT_RELEASE, ptCi, NULL, ptCi->retainStat.savedEventCategory,
        ptCi->retainStat.savedEventItem, 0, true);

    /* DRB */
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_SA_CALL; callDrbIdx++)
    {
        ptDi = &ptCi->drb[callDrbIdx];

        if (ptDi->usage != CI_use) continue;
        if (ptCi->drb[callDrbIdx].retainStat.capableFlag != VCCB_FLAG_ON) continue;

        MACRO_STAT_PMCC_RETAIN(STAT_SETUP_DRB_RELEASE, ptCi, ptDi, ptDi->retainStat.savedEventCategory,
            ptDi->retainStat.savedEventItem, 0, true);
    }

    /* VoNR & Emergency Call */
    if(ptCi->rrcConnStat.vonrEmerCallFlag == VCCB_FLAG_ON)
    {
        MACRO_STAT_PMCC_RETAIN(STAT_VONR_EMERGENCY_RELEASE, ptCi, NULL, ptCi->retainStatOfVonrEmer.savedEventCategory,
                ptCi->retainStatOfVonrEmer.savedEventItem, 0, true);
    }
}

uint8_t FnVCCB_GetUeCapaSplitDRB_withUL_Both_MCG_SCG_NRDC(const callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_FALSE(ptCi);

    return ((ptCi->ueInfo.ueCapa.nr.nrdc.splitDRB_withUL_Both_MCG_SCG) ||
            (ptCi->ueInfo.ueCapa.nr.nrdc.tddCapaAdd.general.splitDRB_withUL_Both_MCG_SCG));
}

void VLIB_QosFlowSetup_CollectStat_PutCi(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    uint16_t cause = ptCi->cause.callFault.callFaultCause;

    // attempt has been counted but not success/fail
    uint8_t callPduIdx = 0;
    uint8_t pduQosFlowIdx = 0;
    uint16_t callQosFlowIdx = 0;

    for (callPduIdx = 0; callPduIdx < MAX_PDU_SESS_PER_CALL; callPduIdx++)
    {
        if (ptCi->pduSess[callPduIdx].usage != CI_use) continue;

        for (pduQosFlowIdx = 0; pduQosFlowIdx < MAX_QOS_FLOW_PER_PDU_SESS; pduQosFlowIdx++)
        {
            callQosFlowIdx = ptCi->pduSess[callPduIdx].qosFlowIndex[pduQosFlowIdx];

            if (callQosFlowIdx >= MAX_QOS_FLOW_PER_CALL) continue;
            if (ptCi->qosFlow[callQosFlowIdx].usage != CI_use) continue;
            if (ptCi->qosFlow[callQosFlowIdx].qosFlowStatistic.attemptFlag[QOS_FLOW_SETUP] != ATTEMPT_MADE) continue;
            if (ptCi->qosFlow[callQosFlowIdx].qosFlowStatistic.attemptFlag[QOS_FLOW_SETUP] == ATTEMPT_MADE)
            {
                VLIB_QosFlowSetup_CollectStat_FailedQosflowItem(
                    ptCi, cause, &ptCi->pduSess[callPduIdx], &ptCi->qosFlow[callQosFlowIdx], STAT_QOS_FLOW_SETUP);
            }

            if (ptCi->qosFlow[callQosFlowIdx].qosFlowStatistic.attemptFlag[QOS_FLOW_SETUP_PLMN_SNSSAI] == ATTEMPT_MADE)
            {
                VLIB_QosFlowSetup_CollectStat_FailedQosflowItem(ptCi, cause, &ptCi->pduSess[callPduIdx],
                    &ptCi->qosFlow[callQosFlowIdx], STAT_QOS_FLOW_SETUP_PLMN_SNSSAI);
            }
        }
    }
}

void VLIB_InitUeReleasePreference(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    memset(&ptCi->ueNrRrcOtherConfig.currConfig.ueRelPrefConfig, 0,
        sizeof(ptCi->ueNrRrcOtherConfig.currConfig.ueRelPrefConfig));
    memset(&ptCi->ueNrRrcOtherConfig.newConfig.ueRelPrefConfig, 0,
        sizeof(ptCi->ueNrRrcOtherConfig.newConfig.ueRelPrefConfig));

    ptCi->ueNrRrcOtherConfig.currConfig.ueRelPrefConfig.releasePreferenceProhibitTimer_r16 = 0xFF;
    ptCi->ueNrRrcOtherConfig.newConfig.ueRelPrefConfig.releasePreferenceProhibitTimer_r16 = 0xFF;

    return;
}

static inline void INIT_OTHER_CONFIG_INFO(ueNrRrcOtherConfigInfoIE_type *ptConf)
{
    VALID_E_POINT_CHECK_VOID(ptConf);

    ptConf->configured = VCCB_FLAG_OFF;
    ptConf->changeFlag = NR_RRC_TYPE_NO_ACTION;
    ptConf->prohibitTimer = 0xFF;
}

void FnVCCB_InitUeNrRrcOtherConfig(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.currConfig.ueOverheatAssiConfig);
    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.newConfig.ueOverheatAssiConfig);

    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.currConfig.ueMaxBWPrefConfig);
    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.newConfig.ueMaxBWPrefConfig);

    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.currConfig.ueMaxCCPrefConfig);
    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.newConfig.ueMaxCCPrefConfig);

    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.currConfig.ueMaxMIMOLayerPrefConfig);
    INIT_OTHER_CONFIG_INFO(&ptCi->ueNrRrcOtherConfig.newConfig.ueMaxMIMOLayerPrefConfig);

    return;
}

void FnVCCB_InitUeAssistanceInformation(callInfo_type &rCall)
{
    AUTO_TRACE_FUNC;

    rCall.sCellCountForOverheating = 0;
    memset(&rCall.overheatingAssistance, 0, sizeof(OverheatingAssistance_type));
    memset(&rCall.preferenceConfigForPowersaving, 0, sizeof(PowerSavingAssistance_type));
    memset(&rCall.ulGapFR2PreferenceAssistance, 0, sizeof(UlGapFR2PreferenceAssistance_type));
}

void VLIB_InitSecurityInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    memset(&ptCi->security, 0xFF, sizeof(ptCi->security));

    ptCi->security.nCC = 0;
    ptCi->security.newNCC = 0;
    ptCi->security.pdcpCtx.callId = ptCi->callId;
    ptCi->security.pdcpCtx.dbgFlag = 0;
    ptCi->security.pdcpCtx.securityInfo.mode = SECURITY_IDLE;
    ptCi->security.pdcpCtx.securityInfo.integrityAlgo = EIA0;
    ptCi->security.pdcpCtx.securityInfo.cipherAlgo = EEA0;
    ptCi->security.newPdcpCtx.callId = ptCi->callId;
    ptCi->security.newPdcpCtx.dbgFlag = 0;
    ptCi->security.newPdcpCtx.securityInfo.mode = SECURITY_IDLE;
    ptCi->security.newPdcpCtx.securityInfo.integrityAlgo = EIA0;
    ptCi->security.newPdcpCtx.securityInfo.cipherAlgo = EEA0;
    ptCi->security.newSecurityContextChangeInd = FALSE;
    ptCi->security.nasc.numocts = 0;
    ptCi->security.snCounter = -1;
    ptCi->security.snCounterChangeFlag = FALSE;
    ptCi->newSecurityKeyChangeIndSN = FALSE;
    ptCi->ueInfo.securityModFlag = VCCB_FLAG_OFF;
    ptCi->ueInfo.snKeyChangeWOMnKeyChange = VCCB_FLAG_OFF;
    ptCi->ueInfo.srcGnbSecurityAlgo.sourceSecurityAlgoFlag = VCCB_FLAG_OFF;
}

void VLIB_InitUeRoHCInfo(ueNrRohcCapability_type *pUeRoHCCapability)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(pUeRoHCCapability);
    pUeRoHCCapability->ulOnlyRohcCapable = FALSE;
    pUeRoHCCapability->isRetryDueToUPFailure = FALSE;
    pUeRoHCCapability->curAllocatedRohcCid = 0;
    pUeRoHCCapability->curAllocatedSNRohcCid = 0;
    pUeRoHCCapability->tempAllocatedRohcCid = 0;
    pUeRoHCCapability->symmetricRohcCapable = FALSE;
    pUeRoHCCapability->curAllocatedSNRohcCidCalcFlag = FALSE;
}

void VLIB_InitUeEHCInfo(ueNrEhcCapability_type *pUeEhcCapability)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(pUeEhcCapability);

    memset(pUeEhcCapability, 0x00, sizeof(ueNrEhcCapability_type));

}

int16_t FnVCCB_FillSdapConfiguration(callInfo_type *ptCi, callDRBInfo_type *ptDi, uint8_t updateOption)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptDi);
    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->gnbNum, MAX_GNB_PER_SYS);

    VccbGnbInfo_type *const pGnb = &GNBINFO[ptCi->gnbNum];

    if (ERROR == VLIB_SelectRepresentativeQoSFlowInDRB(ptCi, ptDi))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_FillCallRbInfoPer5qi Error! Invalid 5QI\n");
        return ERROR;
    }
    VCCB_CHECK_E_RANGE_AND_RETURN(ptDi->drbQosIdxInCall, MAX_QOS_FLOW_PER_CALL);

    uint8_t fiveQi;
    qosFlowInfo_type *ptQi = &ptCi->qosFlow[ptDi->drbQosIdxInCall];
    if (ptQi->usage != CI_use)
    {
        return GOOD;
    }
    VLIB_getFiveQi(ptQi, &fiveQi);

    // Default DRB
    if (ptCi->pduSess[ptDi->pduIdxInCall].mappingDrbIdxDefaultGroup == ptQi->drbIdxInCall)
    {
        ptDi->rbInfo.sdapInfo.defaultDRB = ci_default_drb_true;
    }
    else
    {
        ptDi->rbInfo.sdapInfo.defaultDRB = ci_default_drb_false;
    }

    // SDAP Header UL/DL
    if (ptDi->drbStatus == S_Setup)  // sdap-HeaderUL, sdap-HeaderDL cannot be changed after a DRB is established.
    {
        if (ptQi->qosParam.gbrQosInformationPresent == OPT_PRESENT)  // For GBR QoS Flow
        {
            ptDi->rbInfo.sdapInfo.headerDl = ci_sdap_header_absent;
            ptDi->rbInfo.sdapInfo.headerUl = ci_sdap_header_absent;
        }
        else if (ptDi->drbReflectiveQosFlag == VCCB_FLAG_ON &&
                ptDi->rbInfo.sdapInfoPrevFlag ==
                    VCCB_FLAG_OFF &&  // During DRB Establishment (Attach/PDU Setup). Other cases target
                pGnb->cfg.nasReflectiveQosUsage == TRUE)  // gNB shall maintain source configuration
        {
            ptDi->rbInfo.sdapInfo.headerDl = ci_sdap_header_present;
            ptDi->rbInfo.sdapInfo.headerUl = ci_sdap_header_present;

            VCCB_PRINT_CALL_INFORMATIVE(ptCi, "sdapHdr was succeeded with current(dl:%s, ul:%s, yang:%d) !\n",
                ptDi->rbInfo.sdapInfo.headerDl == ci_sdap_header_present ? "present" : "absent",
                ptDi->rbInfo.sdapInfo.headerUl == ci_sdap_header_present ? "present" : "absent",
                pGnb->cfg.nasReflectiveQosUsage);
        }
        else  // For Non-GBR QoS Flow
        {
            ptDi->rbInfo.sdapInfo.headerDl = ci_sdap_header_absent;
            ptDi->rbInfo.sdapInfo.headerUl = ci_sdap_header_present;
        }

        // Mobility Case, preserve the source configuration
        if (updateOption == NR_PDCP_UPDATE_FINAL)
        {
            if (ptDi->rbInfo.sdapInfoPrevFlag)
            {
                ptDi->rbInfo.sdapInfo.defaultDRB = ptDi->rbInfo.sdapInfoPrev.defaultDRB;
                ptDi->rbInfo.sdapInfo.headerDl = ptDi->rbInfo.sdapInfoPrev.headerDl;
                ptDi->rbInfo.sdapInfo.headerUl = ptDi->rbInfo.sdapInfoPrev.headerUl;
                VCCB_PRINT_CALL_DATA(ptCi, "sdapInfo was succeeded with prev(default:%s, dl:%s, ul:%s)!\n",
                    ptDi->rbInfo.sdapInfoPrev.defaultDRB == ci_default_drb_true ? "true" : "false",
                    ptDi->rbInfo.sdapInfoPrev.headerDl == ci_sdap_header_present ? "present" : "absent",
                    ptDi->rbInfo.sdapInfoPrev.headerUl == ci_sdap_header_present ? "present" : "absent");
            }
            else
            {
                FnVCCB_StoreSdapInfoPrev(ptDi);
            }
        }
    }

    return GOOD;
}

void VLIB_GetUeTypeMaxDrbCount(callInfo_type *ptCi, uint8_t *maxDrbCount)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);
    VALID_E_POINT_CHECK_VOID(maxDrbCount);

    if (IS_REDCAP_UE(ptCi) && (E_CALLTYPE_SA == ptCi->callType))
    {
        if (TRUE == ptCi->redCapUeInfo.supportOf16DRB)
        {
            *maxDrbCount = MAX_DRB_PER_SA_CALL; // 16
        }
        else
        {
            *maxDrbCount = MAX_DRB_PER_REDCAP_CALL; // 8
        }

        VCCB_PRINT_CALL_DATA(ptCi, "[RedCapUe] MaxDrbPerSaCall is %d\n", *maxDrbCount);
    }
}

bool FnVCCB_IsActiveUeContextStat(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    uint16_t callDrbIdx = 0;
    callDRBInfo_type *ptDi = NULL;

    if (IS_NRDC_SN(ptCi))
        return FALSE;

    if (ptCi->retainStat.savedEventCategory == No_CallDrb_RetainCategory)
        return FALSE;

    // Check UE Based Flag
    if (ptCi->bitFlag.ueBasedInSessionFlagByE1 == VCCB_FLAG_ON)
        return TRUE;

    // Check DRB Based Flag
    for (callDrbIdx = 0; callDrbIdx < MAX_DRB_PER_SA_CALL; callDrbIdx++)
    {
        ptDi = &ptCi->drb[callDrbIdx];

        if (ptDi->usage != CI_use) continue;
        if (ptDi->retainStat.capableFlag != VCCB_FLAG_ON) continue;

        // Check SN Term Bearer
        if( VCCB_IS_SN_TERM_DRB_WITH_ONLY_CHECK_BEARER_TYPE(ptDi) ) continue;

        if (ptDi->retainStat.inSessionFlagByE1 ) return TRUE;
    }

    return FALSE;
}

/**
 * @brief       FGR-MC1007 NR Connected Mobility Control FR391
 *              This function initializes the timerBasedMrBasedRedirectInfo.
 * @param[in]   rCall : reference to call info
 * @param[in]   option : info that determines whether to cancel or init the timer
 */
void FnVCCB_InitTimerBasedMrBasedRedirectionInfo(callInfo_type &rCall, enumTimerType option)
{
    AUTO_TRACE_FUNC;

    rCall.timerBasedMrBasedRedirectInfo.mrCarrierFreq = UINT32_MAX;
    rCall.timerBasedMrBasedRedirectInfo.cellIdentity = 0xFFFF;

    VCCB_CLEAR_TIMER(&rCall.timerBasedMrBasedRedirectInfo.timer, option);
}

void FnVCCB_InitServiceBorderMobilityInfo(callInfo_type &rCall, enumTimerType option)
{
    AUTO_TRACE_FUNC;

    rCall.saToNsaBorderHoInfo.cellIdentity = 0xFFFF;

    VCCB_CLEAR_TIMER(&rCall.saToNsaBorderHoInfo.timer, option);
}

void FnVCCB_InitSpecificSliceForcedMobilityInfo(callInfo_type &rCall, enumTimerType option)
{
    AUTO_TRACE_FUNC;

    rCall.specificSliceForcedHoInfo.forcedMobilityCgiReport = FALSE;
    rCall.specificSliceForcedHoInfo.cellIdentity= 0xFFFF;

    VCCB_CLEAR_TIMER(&rCall.specificSliceForcedHoInfo.timer, option); 
}

/**
 * @brief       FGR-MC1007 NR Connected Mobility Control FR391
 *              This function initializes the nrConnectedMobilityControlInfo.
 * @param[in]   rCall : reference to call info
 */
void FnVCCB_InitNrConnectedMobilityControlInfo(callInfo_type &rCall)
{
    AUTO_TRACE_FUNC;

    FnVCCB_InitTimerBasedMrBasedRedirectionInfo(rCall);
    FnVCCB_InitServiceBorderMobilityInfo(rCall);
}

/**
 * @brief       FGR-CC3401 NR Retainability
 *              This function initializes information to collect CSL statistics.
 * @param[in]   ptCi : pointer to call info
 */

void FnVCCB_InitCslInfo_RetainStat(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    memset(&ptCi->cslInfo.retainStatInfo, 0xFF, sizeof(ptCi->cslInfo.retainStatInfo));

    ptCi->cslInfo.retainStatInfo.collectOrder      = 0;
    ptCi->cslInfo.retainStatInfo.abnRelCause5qiNum = 0;
    ptCi->cslInfo.retainStatInfo.drbRetainStatNum  = 0;

    for (uint8_t idx = 0; idx < MAX_RETAIN_STAT_CSL; ++idx)
    {
        ptCi->cslInfo.retainStatInfo.drbRetain[idx].collectOrder = 0;
    }
}

/**
 * @brief       FGR-CC3101 Basic Operation For NR SA
 *              This function initializes information to collect CSL statistics.
 * @param[in]   ptCi : pointer to call info
 */
void FnVCCB_InitCslInfo_BasicCallStat(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    memset(&ptCi->cslInfo.basicCallStatInfo, 0xFF, sizeof(ptCi->cslInfo.basicCallStatInfo));

    ptCi->cslInfo.basicCallStatInfo.collectOrder              = 0;
    ptCi->cslInfo.basicCallStatInfo.initialDrbSetupStatNum    = 0;
    ptCi->cslInfo.basicCallStatInfo.drbSetupStatNum           = 0;
    ptCi->cslInfo.basicCallStatInfo.setupFailCause5qi1StatNum = 0;

    for (uint8_t idx = 0; idx < MAX_BASIC_CALL_STAT_CSL; ++idx)
    {
        ptCi->cslInfo.basicCallStatInfo.initialDrbSetup[idx].collectOrder = 0;
        ptCi->cslInfo.basicCallStatInfo.drbSetup[idx].collectOrder        = 0;
    }
}

/**
 * @brief       This function initializes Connection information to collect CSL statistics.
 * @param[in]   ptCi : pointer to call info
 */
void FnVCCB_InitCslInfo_ConnectInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->cslInfo.connectInfo.nrdcAddVonr = 0xFFFF;
    ptCi->cslInfo.connectInfo.nrdcRelVonr = 0xFFFF;
    ptCi->cslInfo.connectInfo.nrdcStatus = 0xFF;
}

/**
 * @brief       This function initializes CA Config information to collect CSL statistics.
 * @param[in]   ptCi : pointer to call info
 */
void FnVCCB_InitCslInfo_CaInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    for (uint8_t idx = 0; idx < MAX_SCELL_CA_CSL; ++idx)
    {
        auto &sCellInfo = ptCi->cslInfo.sCellConfigInfo[idx];
        
        // Reset DL CA configuration counters for VoNR
        sCellInfo.dlCaConfigVonrCnt = 0;
        sCellInfo.dlCaDeConfigVonrCnt = 0;
        
        // Reset UL CA configuration counters for VoNR
        sCellInfo.ulCaConfigVonrCnt = 0;
        sCellInfo.ulCaDeConfigVonrCnt = 0;
    }
}

/**
 * @brief       This function initializes RRE information to collect CSL statistics.
 * @param[in]   ptCi : pointer to call info
 */
void FnVCCB_InitCslInfo_RreInfo(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->cslInfo.cslRreInfo.vonrRreFailCause = 0xFF;
}

void FnVCCB_PrintUeSecurityKeyLog(callInfo_type *ptCi)
{
    // Warning : use with caution. must off after the test.
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    int loop1 = 0;
    std::ostringstream oss1, oss2, oss3;
    oss1 << std::hex << std::setfill('0');
    oss2 << std::hex << std::setfill('0');
    oss3 << std::hex << std::setfill('0');
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| AMF-UE-NGAP-ID:%ld, RAN-UE-NGAP-ID:%d\n", ptCi->ngCallId, ptCi->callId);

    uint8_t i = 0, strLen = 0;
    int len = MIN(int(ptCi->ueInfo.ueId.imsi.n), int(sizeof(ptCi->ueInfo.ueId.imsi.digit)));
    char imsiPrintBuf[16];
    imsiPrintBuf[0] = '\0';
    uint32_t buffSize = sizeof(imsiPrintBuf);
#define IMSI_PRINT_BUF imsiPrintBuf + strLen
    for (i = 0; i < len; i++)
    {
        strLen = static_cast<uint8_t>(
            strLen + snprintf(IMSI_PRINT_BUF, buffSize - strLen, "%X", ptCi->ueInfo.ueId.imsi.digit[i]));
    }
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| IMSI : %s\n", imsiPrintBuf);

    // securityInfo_type
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| security        : integrityAlgo [%d] / cipherAlgo [%d] \n",
        ptCi->security.pdcpCtx.securityInfo.integrityAlgo, ptCi->security.pdcpCtx.securityInfo.cipherAlgo);

    for (loop1 = 0; loop1 < 32; loop1++)
    {
        oss1 << std::setw(2) << static_cast<int>(ptCi->security.sKgNB[loop1]);
    }
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| security        : sKgNB [%s]\n", oss1.str().c_str());

    for (loop1 = 0; loop1 < 16; loop1++)
    {
        oss2 << std::setw(2) << static_cast<int>(ptCi->security.pdcpCtx.securityInfo.keyRrcIntCurr[loop1]);
    }
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| security        : keyRrcIntCurr [%s]\n", oss2.str().c_str());

    for (loop1 = 0; loop1 < 16; loop1++)
    {
        oss3 << std::setw(2) << static_cast<int>(ptCi->security.pdcpCtx.securityInfo.keyRrcEncCurr[loop1]);
    }
    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "| security        : keyRrcEncCurr [%s]\n", oss3.str().c_str());

    return;
}

/**
 * @brief Set cause for X2 SgNB Modification Required depending on the HO trigger.
 *
 * @param call info
 */
void VCCB_NSA_SetCauseForX2SgnbModificationRequired(callInfo_type &call)
{
    AUTO_TRACE_FUNC;

    call.cause.x2Cause = CauseX2apRadioNetworkLayerScgMobility;
}

qosFlowInfo_type *GetQosFlowFromPduByQfi(callInfo_type &call, callPDUSessInfo_type &rPdu, XNAP_QoSFlowIdentifier qfi)
{
    AUTO_TRACE_FUNC;

    for (auto qosFlowIndex : rPdu.qosFlowIndex)
    {
        if (qosFlowIndex >= MAX_QOS_FLOW_PER_CALL) continue;
        auto &qosFlow = call.qosFlow[qosFlowIndex];

        if (qosFlow.usage != CI_use) continue;

        if (qosFlow.flowId == qfi)
        {
            return &qosFlow;
        }
    }

    return nullptr;
}

void FnVCCB_InitSavedRequestedFlagForPduQosFlowVonr(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    ptCi->bitFlag.isEmergencyCallByReqValInThisProcedure = VCCB_FLAG_OFF;
    ptCi->bitFlag.isVonrRequestedInThisProcedure         = VCCB_FLAG_OFF;
    ptCi->bitFlag.isQosReleaseInThisProcedure            = VCCB_FLAG_OFF;

    for (auto &qosFlow: ptCi->qosFlow)
    {
        if (qosFlow.usage != CI_use) continue;
        qosFlow.isSavedRequestedQosParam = VCCB_FLAG_OFF;
    }

    for (auto &pduSess: ptCi->pduSess)
    {
        if (pduSess.usage != CI_use) continue;
        pduSess.isSavedRequestedPduParam = VCCB_FLAG_OFF;
    }

    return;
}

/* EOF */
