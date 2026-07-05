/****************************************************************
 * System      : 5G vNB
 * Subsystem   : CALL
 * Module      : VCCB
 * File        : StateE1ap.cpp
 * Description :
 * Author      :
 * Dept.       : NG Access Lab.
 * Created     : 2019.03.01
 * Procedures defined :
 * Procedures called  :
 * Version     :
 * Revision history   :
 *          date      :
 *       revised by   :
 * revised information:
 ****************************************************************/

#include "StateE1ap.h"

#include "StateCommon.h"
#include "StateF1ap.h"
#include "StateTime.h"
#include "StateMeasGap.h"
#include "StateHandleAll.h"
#include "StateHandleSaMrdc.h"
#include "StateMeasurements.h"
#include "StateCa.h"

#include "ComInclude.h"
#include "ComLibUtility.h"

#include "VccbAiClient.h"   // FnVCCBAiClient_OnDataActivityResumed (AI extension-loop bookkeeping)

#include "FeatCommon.h"
#include "FeatCa.hpp"
#include "FeatMco.h"
#include "FeatMrdc.h"
#include "FeatSecurity.h"
#include "FeatCallInfo.h"
#include "FeatCallProc.h"
#include "FeatFullConfig.h"
#include "FeatMeasurements.hpp"
#include "FeatUlPathControlENDC.h"
#include "FeatUlPathControlNRDC.h"
#include "FeatTrafficMeasurements.hpp"
#include "FeatHo.h"

#include "MsgF1apSend.h"
#include "MsgE1apSend.h"
#include "MsgE1apStore.hpp"
#include "MsgXnapSend.h"
#include "MsgX2apSend.h"
#include "MsgNrRrcMake.h"
#include "MsgNrRrcSend.h"
#include "MsgNmbSend.h"
#include "MsgNgapSend.h"

#include "MsgQueueManager.h"
#include "FeatPreemption.h"
#include "StateXnap.h"
#include "StateNrRrc.h"

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Setup Response message in DCE1Setup state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCE1Setup_S_E1apBearerContextSetupResponse(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = VFIL_sendF1_UeContextSetupRequest(ptCi, NULL, 0);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextSetupRequest ERROR!\n");
        ptCi->endcAddCpFailDetailCause = StatEndcAddPrepCpFail_TxF1UEContextSetupRequestFail;
        return status;
    }

    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextSetup_id);

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Setup Failure message in DCE1Setup state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_DCE1Setup_S_E1apBearerContextSetupFailure(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // Statistics
    VLIB_StatEndcAdd(ptCi, StatEndcAddPrepFail_UpFail, __FUNCTION__);

    // ERAB_ADD Fail
    for (int i = 0; i < MAX_ERAB_PER_CALL; i++)
    {
        auto &erabInfo = ptCi->erabInfo[i];
        if (erabInfo.usage != CI_use) continue;
        if ((erabInfo.erabStatus != ERAB_Setup) && (erabInfo.erabStatus != ERAB_FailToSetup)) continue;

        // Statistics Update
        ptCi->ptTempRi = &erabInfo;
        MACRO_STAT_PMCC_QCI(STAT_ERAB_ADD_PER_GNB, ErabAddFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
    }
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Modification Response message in DCAddPrepare state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCAddPrepare_S_E1apBearerModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");
    if (MULTI_UP_ERROR == status)
    {
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
    if (ERROR == status) return status;

    status = VXIL_sendX2_SgNBAdditionRequestAcknowledge(ptCi);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VXIL_sendX2_SgNBAdditionRequestAcknowledge ERROR!\n");
        ptCi->endcAddCpFailDetailCause = StatEndcAddPrepCpFail_TxX2SgNBAdditionRequestAckFail;
        return status;
    }

    ON_MENB_CONNECTION_FLAG(ptCi->flag.setupFlag);
    ON_SGNB_REC_CMPL_FLAG(ptCi->flag.respFlag);

    if (VLIB_IsAmDrbPresent(ptCi))
    {
        ON_X2_SN_STATUS_FLAG(ptCi->flag.respFlag);
    }

    /* Statistics */
    VLIB_StatEndcAdd(ptCi, StatEndcAddPrepSucc, __FUNCTION__);
    ptCi->endcAddCpFailDetailCause = StatEndcAddCpFail_Unspecified; // Init with Unspecified state

    // timX2_DcOverall_id
    VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Modification Failure message in DCAddPrepare state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCAddPrepare_S_E1apBearerModificationFailure(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // Statistics
    VLIB_StatEndcAdd(ptCi, StatEndcAddPrepFail_UpFail, __FUNCTION__);

    for (int i = 0; i < MAX_ERAB_PER_CALL; i++)
    {
        auto &erabInfo = ptCi->erabInfo[i];
        if (erabInfo.usage != CI_use) continue;
        if (erabInfo.erabStatus != ERAB_Modify && erabInfo.erabStatus != ERAB_FailToModify) continue;
        // Changed erabStatus to ERAB_Modify after receiving F1_UEContextSetupResponse

        // Statistics Update
        ptCi->ptTempRi = &erabInfo;
        MACRO_STAT_PMCC_QCI(STAT_ERAB_ADD_PER_GNB, ErabAddFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Failure message in DB2DBRenb state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DB2DBRenb_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // Even if a failure occurs, proceed with next procedure.
    // For sending transmissionStopIndicator
    int16_t status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
        return ERROR;
    }

    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_DB2DBRF1Mod);

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Response message in DB2DBRE1SNStatus state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DB2DBRE1SNStatus_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VCCB_RETURN_ON_CALL_ERROR(
        ptCi, VXIL_sendX2_SNStatusTransferIfNeed(ptCi), "VXIL_sendX2_SNStatusTransferIfNeed ERROR!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Failure message in DB2DBRE1SNStatus state
 *                  after receiving responses from all AUPFs
 * @param[in/out]   ptCi :      Call information / callInfo_type
 ****************************************************************************/
void VCCM_DB2DBRE1SNStatus_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
}

/****************************************************************************
 * @brief           Handle received E1 BearerContextModificationResponse (SetupRespone/ReleaseComplete) message in
 *DCE1Mod state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCE1Mod_S_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    int16_t f1UeContextToSetupFlag = FALSE;
    int16_t SN_MCG_AddModReleaseOnlyFlag = FALSE;
    uint8_t newEndcSvcGrpId = 0;

    // timE1_BearerContextModification_id
    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    for (int i = 0; i < MAX_ERAB_PER_CALL; i++)
    {
        auto &erabInfo = ptCi->erabInfo[i];
        if (erabInfo.usage != CI_use) continue;
        if (((ERAB_Setup == erabInfo.erabStatus) ||
                ((ERAB_Setup == erabInfo.prevErabStatus) && (ERAB_Modify == erabInfo.erabStatus))) &&
            (erabInfo.rbInfo.rlcMode == AM_RLC))
        {
            ON_X2_SN_STATUS_FLAG(ptCi->flag.respFlag);
            break;
        }
    }

    // For ERAB Release Case (When PDCP_SN_Status_Information is received from UP)
    VCCB_RETURN_ON_CALL_ERROR(
        ptCi, VXIL_sendX2_SNStatusTransferIfNeed(ptCi), "VXIL_sendX2_SNStatusTransferIfNeed ERROR!\n");

    if (!GET_F1_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc))  // No need to execute F1 Procedure
    {
        if (GET_ACK_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc))
        {
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE),
                "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR !!\n");

            OFF_ACK_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc);
        }

        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepSucc, ptCi, 0, 0, 0, 0, 0);

        SN_MCG_AddModReleaseOnlyFlag = VLIB_CheckDrbTypeSN_MCG_AddModReleaseOnly(ptCi);
        if (ERROR == SN_MCG_AddModReleaseOnlyFlag)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_CheckDrbTypeSN_MCG_AddModReleaseOnly ERROR !!\n");
            return ERROR;
        }

        if (GET_F1CPLT_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc) || SN_MCG_AddModReleaseOnlyFlag)
        {
            // Wait X2_SgNBReconfigurationComplete
            VLIB_SetCurStateAll(ptCi, STATE_DCUeMod_S);
            VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
        }
        else
        {
            // Modification Procedure END
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbSucc, ptCi, 0, 0, 0, 0, 0);
            VLIB_SetNoAttemptFlag(ptCi, STAT_MENB_INIT_SGNB_MOD_PER_GNB);
            VCCM_SetStableState_CallSetup(ptCi);
        }
    }
    else  // F1 Procedure
    {
        f1UeContextToSetupFlag = VXIL_Check_F1UeContextToSetup(ptCi);
        if (ERROR == f1UeContextToSetupFlag)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VXIL_Check_F1UeContextToSetup ERROR !!\n");
            return ERROR;
        }

        int16_t newMeasConfigPrepared = FnVCCB_CheckIfBearerChangeToMcgMeasUpdateNeeded(*ptCi);
        if (ERROR == newMeasConfigPrepared)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCB_PrepareNewMeasConfig in FnVCCB_CheckIfBearerChangeToMcgMeasUpdateNeeded ERROR!\n");
            return ERROR;
        }

        // in case of SCG/MCG->SPLIT, SPLIT->SCG/MCG modification
        if ((GOOD == VLIB_CheckDrbTypeModInvolvingSplitOngoing(ptCi)) ||
            CHECK_MOD_EVENT_WITH_STATECMD(ptCi, allowedBC) || f1UeContextToSetupFlag)
        {
            if (CHECK_MOD_EVENT_WITH_STATECMD(ptCi, allowedBC) || ptCi->f1Info.f1UeContextToRemove_flag ||
                f1UeContextToSetupFlag)
            {
                VLIB_ConfigSecondaryCellList(ptCi);
            }
            
            if (GOOD != newMeasConfigPrepared)
            {
                VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");
            }
        }
        else
        {
            if (GOOD != VCCB_GetEndcSvcGrpId(ptCi, ptCi->servCellNum, &newEndcSvcGrpId))
            {
                newEndcSvcGrpId = 0;
            }

            if (newEndcSvcGrpId != ptCi->measurementsConfiguration.svcGrpId && GOOD != newMeasConfigPrepared)
            {
                VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");
            }
        }

        // check if SN MCG bearer add only, if true skip F1_UeContextModificationRequest sending
        SN_MCG_AddModReleaseOnlyFlag = VLIB_CheckDrbTypeSN_MCG_AddModReleaseOnly(ptCi);
        if (ERROR == SN_MCG_AddModReleaseOnlyFlag)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_CheckDrbTypeSN_MCG_AddModReleaseOnly ERROR !!\n");
            return ERROR;
        }
        if (SN_MCG_AddModReleaseOnlyFlag)
        {
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE),
                "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR!\n");

            // Collect Stat
            MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepSucc, ptCi, 0, 0, 0, 0, 0);

            VLIB_SetCurStateAll(ptCi, STATE_DCUeMod_S);

            // timX2_DcOverall_id
            VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
        }
        else
        {
            // Setup F1 link if needed
            if (f1UeContextToSetupFlag)
            {
                VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextSetupRequest(ptCi, NULL, 0),
                    "VFIL_sendF1_UeContextSetupRequest ERROR!\n");

                VLIB_SetCurStateAll(ptCi, STATE_DCF1Setup_S);
                VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextSetup_id);

                OFF_F1_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc);
            }
            else
            {
                VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextModificationRequest(ptCi, NULL),
                    "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

                VLIB_SetCurStateAll(ptCi, STATE_DCF1Mod_S);
                VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);

                OFF_F1_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc);
            }
        }
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 BearerContext ModificationFailure/SetupFailure message in DCE1Mod state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, BREAK, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCE1Mod_S_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    int setupSuccessNum = 0;
    int setupFailureNum = 0;
    int modFailureNum = 0;

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // Collect Stat
    if (SgnbInitBearerRelease_cmd == ptCi->stateCmd)
    {
        MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModBySgnbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    // ERAB_MOD Fail
    for (uint16_t loop = 0; loop < MAX_ERAB_PER_CALL; loop++)
    {
        if (ptCi->erabInfo[loop].usage != CI_use) continue;

        ptCi->ptTempRi = &ptCi->erabInfo[loop];

        // Statistics Update
        switch (ptCi->erabInfo[loop].erabStatus)
        {
            case ERAB_Setup:
                setupSuccessNum++;
                break;
            case ERAB_FailToSetup:
                MACRO_STAT_PMCC_QCI(STAT_ERAB_ADD_PER_GNB, ErabAddFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
                SET_CALL_ERAB_INFO_USAGE_OFF(ptCi, loop);
                setupFailureNum++;
                break;
            case ERAB_Modify:
            case ERAB_FailToModify:
                if (ptCi->erabInfo[loop].prevErabStatus == ERAB_Setup)
                {
                    MACRO_STAT_PMCC_QCI(STAT_ERAB_MOD_PER_GNB, ErabAddFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
                }
                else
                {
                    if (ptCi->erabInfo[loop].preveBearerType != ptCi->erabInfo[loop].bearerType)
                    {
                        MACRO_STAT_PMCC_QCI(
                            STAT_ERAB_CHG_PER_GNB, ErabModFail_UpFail_Change, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
                    }
                    MACRO_STAT_PMCC_QCI(STAT_ERAB_MOD_PER_GNB, ErabModFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
                }
                modFailureNum++;
                break;
            default:
                break;
        }
    }

    if (SgNBInitBearerChangeToMcg_cmd == ptCi->stateCmd)
    {
        // Increment internal statistic for SN init bearer type change
        VRRM_endcSNinitErabChgCount(ptCi, StatEndcErabModFail_UpFail);
        return GOOD;
    }

    int16_t status = VXIL_sendX2_SgNBModificationRequestReject(ptCi, FALSE);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequestReject ERROR!\n");

    if (setupFailureNum > 0 && setupSuccessNum == 0 && modFailureNum == 0)
    {
        // In case of ERAB Addition Failure, VCCB should maintain the call. (DRB CAC Case in GNB-CU-UP)
        // In case of ERAB Modification Failure, VCCB will release the call.

        MACRO_RESET_CALL_FAULT_CAUSES;

        if (true == VRRM_isE1PartialSuccess(ptCi))
        {
            // need to revert AMBR value
            status = VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendE1SetupModRelRequest (Modification) ERROR!\n");
        }

        SET_STATE_CMD(Idle_cmd);
        VLIB_SetCurStateAll(ptCi, STATE_DB);
        return BREAK;
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Response message commonly,
 *                  which sends F1 UE Context Modification Request.
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_Common_SendF1ModReq_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

    // VLIB_SetCurStateAll will be called after this function returns GOOD

    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Failure message in DCF1Mod state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_DCF1Mod_S_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Release Complete message in DCSnChg state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCSnChg_Fin_E1apBearerContextReleaseComplete(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1RelCmpl) Waiting for other UPs to respond\n");

    if (status == ERROR)
    {
        VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
        return ERROR;
    }

    if (GET_F1_REL_COMPLT_RCV_FLAG(ptCi->flag.respFlag))
    {
        return GOOD;
    }

    VCCB_NSA_PegInterSnPscellChange(*ptCi, EndcInterChgSrcSucc);
    MACRO_STAT_PMCC(STAT_INTRA_FREQ_PSCELL_CHANGE_SCGMOB_PER_GNB_DU_ID, EndcIntraFreqInterSnChgSucc, ptCi, 0, 0, 0, 0, 0);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VLIB_PutCi(ptCi);

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Release Complete message in COMMrelack state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_COMMrelack_E1apBearerContextReleaseComplete(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1RelCmpl) Waiting for other UPs to respond\n");

    if (status == ERROR)
    {
        VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
        return ERROR;
    }

    return GOOD;
}

int16_t VCCM_ALL_E1apBearerContextReleaseRequestStatForNSA(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    int famIdx;

    if (ptCi->callType != E_CALLTYPE_NSA) return GOOD;

    if (CHECK_STATECMD_RELEASE(ptCi))
    {
        VCCB_PRINT_CALL_ERROR(ptCi,
            "Received E1BearerContextReleaseRequest during release procedure(stateCmd:%d, state:%d))\n", ptCi->stateCmd,
            ptCi->state);
        return BREAK;
    }

    if (IntraRatHO_cmd == ptCi->stateCmd && HO_TYPE_INTRA_NB == ptCi->hoInfo.hoType.nb)
    {
        // HO case
        if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)
        {
            famIdx = GET_FAMIDX(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB);
            VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_ERROR(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);
            if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
            {
                MACRO_STAT_PMCC(
                    STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobFail_UpFail, ptCi, 0, 0, 0, 0, 0);
            }
            else
            {
                MACRO_STAT_PMCC(
                    STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
            }
        }
        else if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated)
        {
            famIdx = GET_FAMIDX(STAT_SGNB_ADD_INTRA_LTE_HO_NO_SGNB_CHG);
            VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_ERROR(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);
            if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
            {
                VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgFail_UpFail);
            }
            else
            {
                VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
            }
        }
        else
        {
            famIdx = GET_FAMIDX(STAT_INTRA_SN_PSCELL_CHANGE_PER_GNB);
            VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_ERROR(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);
            if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
            {
                VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgFail_UpFail);
            }
            else
            {
                VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepFail_UpFail);
            }
        }
    }
    else if (IntraRatHO_cmd == ptCi->stateCmd && HO_TYPE_INTER_NB == ptCi->hoInfo.hoType.nb)
    {
        // SgNB Change case
        famIdx = GET_FAMIDX(STAT_INTER_SN_PSCELL_CHANGE_SRC_PER_GNB);
        VCCB_CHECK_RANGE_MIN_MAX_AND_RETURN_ERROR(famIdx, 0, MAX_VCCB_FAMILY_TYPE - 1);
        if (GET_CALL_STAT_FLAG(ptCi, famIdx) == PREP_SUCCESS_MADE)
        {
            VCCB_NSA_PegInterSnPscellChange(*ptCi, EndcInterChgSrcFail_UpFail);
        }
        else
        {
            VCCB_NSA_PegInterSnPscellChange(*ptCi, EndcInterChgSrcPrepFail_UpFail);
        }
    }

    return GOOD;
}

int16_t VCCM_ALL_E1apBearerContextReleaseRequestStatForSA(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    if (ptCi->callType != E_CALLTYPE_SA) return GOOD;

    if (CHECK_SA_STATE_RELEASE(ptCi))
    {
        VCCB_PRINT_CALL_ERROR(ptCi,
            "Received E1BearerContextReleaseRequest during release procedure(stateCmd:%d, state:%d)\n", ptCi->stateCmd,
            ptCi->state);
        return BREAK;
    }

    if (VHCM_isSaHandoverUE(ptCi))
    {
        FnVCCB_CollectingSaHandoverStatistics_E1Fail(ptCi);

        // CHO_IN
        if (ptCi->hoInfo.handoverExecutionFlag == TRUE)
        {
            if (ptCi->hoInfo.isTarget == TRUE)
            {
                /*
                    This counter is the number of times that the CHO Candidate Release procedure is performed
                    because of a failure in the UP after completing the Inter-gNB Xn CHO Preparation procedure.
                */
                if (FnVCCB_IsConditionalHandoverUE(ptCi))
                {
                    CHO_STAT(ptCi, CHO_CandidateRel_UpInit, ptCi->hoInfo.cgi);
                }
            }
        }
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Release Request message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1apBearerContextReleaseRequest(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (FALSE == VCCM_IsUeIdleState(ptCi->callType, ptCi->state))
    {
        // SVR24B FGR-MC1016 FR800
        if (IS_RRC_SUSPEND_COMPLETED(ptCi))
        {
            // only one bearer is remained : send ran-paging and enqueue
            VCCB_CLEAR_TIMER(&ptCi->ueInfo.RrcInactiveInfo.guardTimer, CANCEL_TIMER);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, FnVCCB_SendCallPagingReq(ptCi, procDedicated), "FnVCCB_SendCallPagingReq ERROR!\n");

            ptCi->ueInfo.rrcResumeInfo.goingToRRCIdleFlag = VCCB_FLAG_ON;
            ptCi->ueInfo.rrcResumeInfo.ranPaged = VCCB_FLAG_ON;

            // bugfix @SVR25B
            // in order to lead to RRC_Connected when RRCResumeRequest1 comes in response of RAN-Paging, then it goes to RRC_Idle after dequeueing this message
            ptCi->ranPagingInfo.dlDataFlag = VCCB_FLAG_ON;

            // timer to wait RRC Resume Request1 over UL-CCCH1
            VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCResumeRequest1_id);
            VLIB_SetCurStateAll(ptCi, STATE_Inact_ResumeReq);
        }

        if (ptCi->endPointInfo.srvAupfNum > 1 /* Multi UP */ ||
            IS_RRC_SUSPEND_COMPLETED(ptCi))
        {
            if (ERROR == VCCM_enqueueAction(ptCi, ENQUEUE_ACTION_E1_MSG_FROM_UP, ptAll, E1_BearerContextReleaseRequest_id))
            {
                VCCB_PRINT_CALL_INFORMATIVE(ptCi, "VCCM_enqueueAction(ENQUEUE_ACTION_E1_MSG_FROM_UP) in %s fail\n", __FUNCTION__);
            }
            else
            {
                VCCB_PRINT_CALL_COMMENT(ptCi, "E1apBearerContextReleaseRequest: Enqueued\n");
            }
            return GOOD;
        }
    }

    status = FnVCCB_StoreE1Message(E1_BearerContextReleaseRequest_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextReleaseRequest_id) ERROR!\n");

    // [Multi UP] Support
    if (BREAK == status)
    {
        VCCB_PRINT_CALL_COMMENT(
            ptCi, "Do not release call, total Serving AUPFs num: %d\n", ptCi->endPointInfo.srvAupfNum);

        if (ptCi->callType == E_CALLTYPE_NSA)
        {
            SET_STATE_CMD(SgnbInitBearerRelease_cmd);

            ptCi->cause.x2Cause = CauseX2apRadioNetworkLayerActionDesirableForRadioReasons;
            status = VXIL_sendX2_SgNBModificationRequired(ptCi);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequired ERROR!\n");

            VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
            VLIB_SetCurStateAll(ptCi, STATE_DCSgNBInitRel_X2ModRqrd);
        }
        else
        {
            status = VHCM_TriggerPduReleaseWithReleaseEvent(ptCi, E1_BearerContextReleaseRequest_id, TRUE);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status,
                "VHCM_TriggerPduReleaseWithReleaseEvent(E1 REL, msgId: %u, flag:TRUE) ERROR!\n",
                E1_BearerContextReleaseRequest_id);
        }
        return GOOD;
    }

    // All serving AUPF released - release call

    if (ptCi->callType == E_CALLTYPE_NSA)
    {
        status = VCCM_ALL_E1apBearerContextReleaseRequestStatForNSA(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ALL_E1apBearerContextReleaseRequestStatForNSA ERROR!\n");
    }
    else
    {
        status = VCCM_ALL_E1apBearerContextReleaseRequestStatForSA(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ALL_E1apBearerContextReleaseRequestStatForSA ERROR!\n");
    }

    if (status == BREAK)
    {
        // Release procedure is ongoing
        return GOOD;
    }

    VCCB_CALL_RELEASE_TRIGGER(ptCi);  // Call Job Release

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 User Data Usage Report message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1apDataUsageReport(callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    ptCi->e1DataUsageReportCnt = 0;
    int16_t status = FnVCCB_StoreE1Message(E1_DataUsageReport_id, ptCi, ptAll, ptUnpack);

    if (GOOD != status)
    {
        return ERROR;
    }

    if (ptCi->e1DataUsageReportCnt > 0)
    {
        // FR60.CACP.SR1.HD1.VCCB.DD1 - Use valid reports to prepare X2 Secondary RAT Data Usage Report message

        status = VXIL_sendX2_SecondaryRATDataUsageReport(ptCi);

        if (GOOD != status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VXIL_sendX2_SecondaryRATDataUsageReport ERROR!\n");
            return ERROR;
        }
    }
    else
    {
        // FR20.CACP.SR1.HD4.VCCB.DD1 - Do not send X2 Secondary RAT Data Usage Report message if there
        //                              is no valid reports to be used
        VCCB_PRINT_COMMENT("No valid data usage reports to be send to MeNB\n");
        return GOOD;
    }

    return GOOD;
}

int16_t VCCM_DCSNRelRqrd_E1UeBearerModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");
    if (ERROR == status) return status;

    return status;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Failure in state E1DrbModPrep
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCDuHo_E1apDrbModPrep_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    switch (ptCi->hoInfo.hoType.initiated)
    {
        case HO_TYPE_MENB_INITIATED:  // gNB-DU HO MCG SRB
            MACRO_STAT_PMCC(
                STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
            break;
        case HO_TYPE_MENB_INTERMNHO:
            VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
            VLIB_StatErabAddInterMnHO(ptCi, ErabAddFail_UpFail, TRUE);
            break;
        default:  // using/not using SRB3
            VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepFail_UpFail);
            break;
    }
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state E1DrbModSecu
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCIntraDuHo_E1apDrbModSecu_E1UeBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    ptCi->noCgConfigFlag = VCCB_FLAG_ON;
    int16_t status = VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR!\n");

    // If security info is changed, initiate PDCP SN after RRC Reconfiguration
    if (VCCB_FLAG_ON == ptCi->ueInfo.securityModFlag)
    {
        if (MAX_SRB_PER_CALL > ptCi->srb3Idx)
        {
            callSRBInfo_type *ptSi = &(ptCi->srbInfo[ptCi->srb3Idx]);

            if (0 != FnPDCP_C_Init(&ptSi->pdcpRbContext, ptSi->rbInfo.rbId))
            {
                VCCB_PRINT_ERROR("FnPDCP_C_Init ERROR!\n");

                return ERROR;
            }
        }
    }

    VLIB_SetCurStateAll(ptCi, STATE_DCIntraDuHo_X2ModCnf);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);

    return status;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state E1DrbModSecu
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCIntraDuHo_E1apDrbModSecu_E1UeBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgFail_UpFail);
}

/****************************************************************************
 * @brief           Handling Bearer Modification Response in state E1DrbModResume
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCIntraDuHo_E1apDrbModResume_E1UeBearerModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VLIB_FillRecoveryTimeCsl(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (GET_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag) || GETBIT(ptCi->flag.respFlag, SGNB_MOD_CONF_FLAG))
    {
        return GOOD;
    }

    int16_t status = VLIB_CollectNsaHoSuccessStatAndFinalizeHandover_IntraDuHo_E1apDrbModResume(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(
        ptCi, status, "VLIB_CollectNsaHoSuccessStatAndFinalizeHandover_IntraDuHo_E1apDrbModResume ERROR!\n");

    return status;
}
/****************************************************************************
 * @brief           Handling E1 Bearer Context Modification Response in state after starting TDcOverall timer during
 *Intra-DU Handover
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCIntraDuHo_TDcOverall_Common_E1UeBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    if (((HO_TYPE_SGNB_INITIATED == ptCi->hoInfo.hoType.initiated) && GET_SGNB_MOD_CONF_FLAG(ptCi->flag.respFlag)) ||
        ((HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated) && GET_SGNB_REC_CMPL_FLAG(ptCi->flag.respFlag)) ||
        ((HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated) &&
            (GET_SGNB_REC_CMPL_FLAG(ptCi->flag.respFlag) || GET_X2_SN_STATUS_FLAG(ptCi->flag.respFlag))))
    {
        VCCB_PRINT_CALL_MSG(ptCi, "Essential message is not received yet(0x%016llX)\n", 1ULL * ptCi->flag.respFlag);
        return GOOD;
    }

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextModificationRequest(ptCi, NULL),
        "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
    VCCB_RETURN_ON_CALL_ERROR(ptCi, VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification),
        "VEIL_sendE1SetupModRelRequest (Modification) ERROR!\n");

    VLIB_SetCurStateAll(ptCi, STATE_DCIntraDuHo_E1DrbModResume);
    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    VLIB_AttachTimer(&ptCi->callMultiTimer.e1, ptCi, timE1_BearerContextModification_id);

    return GOOD;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Context Modification Failure in state after starting TDcOverall timer during
 *Intra-DU Handover
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCIntraDuHo_TDcOverall_Common_E1UeBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)  // gNB-DU HO MCG SRB
    {
        MACRO_STAT_PMCC(
            STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated)
    {
        VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
        VLIB_StatErabAddInterMnHO(ptCi, ErabAddFail_UpFail, TRUE);
    }
    else  // not using SRB3
    {
        VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepFail_UpFail);
    }
}

/****************************************************************************
 * @brief           Handling Bearer Modification Failure in state E1DrbModResume
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_DCCommonDuHo_E1apDrbModResume_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgFail_UpFail);
    }
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state EDAddition
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_EDAddition_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = VFIL_sendF1_UeContextSetupRequest(ptCi, NULL, 0);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextSetupRequest ERROR !!\n");

    VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_F1UeCtxSetReq);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextSetup_id);

    status = VHCM_UpdateUeHistoryInformation_Prepare(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VHCM_UpdateUeHistoryInformation_Prepare ERROR !!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state EDAddition
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_DCInterDuHo_EDAddition_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
    VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
    VLIB_StatErabAddInterMnHO(ptCi, ErabAddFail_UpFail, TRUE);
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state E1DrbModPrep
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_E1apDrbModPrep_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    for (callERABInfo_type &erabInfo : ptCi->erabInfo)
    {
        if (!IS_VALID_NODE(ERAB_NODE, ptCi, &erabInfo)) continue;

        if (erabInfo.erabStatus != ERAB_Release)
        {
            if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated && ERAB_Setup == erabInfo.erabStatus)
            {
                erabInfo.erabStatus = ERAB_Idle;
                erabInfo.prevErabStatus = ERAB_Setup;
            }
            else
            {
                erabInfo.erabStatus = ERAB_Idle;
                erabInfo.prevErabStatus = ERAB_Idle;
            }
        }
    }

    int16_t statusFR1FR2HO = VCCM_CheckFR1FR2HO(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, statusFR1FR2HO, "VCCM_CheckFR1FR2HO ERROR!\n");

    if (TRUE == ptCi->hoInfo.newUlTnlInfoRequired)
    {
        ptCi->hoInfo.sendF1ModReqToTarget = TRUE;
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextModificationRequest(ptCi, NULL),
            "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
    }

    if (IS_FULLCONFIGURATION_IN_PROCESS(ptCi))
    {
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCM_SetCallInfoForFullConfig(ptCi), "VCCM_SetCallInfoForFullConfig ERROR!\n");

        VCCB_RETURN_ON_CALL_ERROR(ptCi, VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification),
            "VEIL_sendE1SetupModRelRequest (Modification) ERROR!\n");
    }

    VLIB_UpdateCounters_SuccessMrBasedResyncSCGFailure(ptCi, FALSE);

    if (ptCi->hoInfo.hoType.initiated == HO_TYPE_MENB_INITIATED)
    {
        // Inter gNB DU Ho using MCG SRB
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE),
            "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR!\n");

        ON_SGNB_REC_CMPL_FLAG(ptCi->flag.respFlag);

        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepSucc, ptCi, 0, 0, 0, 0, 0);

        // If security info is changed, initiate PDCP SN after RRC Reconfiguration
        if (VCCB_FLAG_ON == ptCi->ueInfo.securityModFlag)
        {
            if (MAX_SRB_PER_CALL > ptCi->srb3Idx)
            {
                callSRBInfo_type *ptSi = &(ptCi->srbInfo[ptCi->srb3Idx]);

                if (0 != FnPDCP_C_Init(&ptSi->pdcpRbContext, ptSi->rbInfo.rbId))
                {
                    VCCB_PRINT_ERROR("FnPDCP_C_Init ERROR!\n");
                    return ERROR;
                }
            }
        }

        // Source DU <- TransmissionActionIndicator : stop
        if (FALSE == ptCi->hoInfo.interDuHoWithoutSourceDu)
        {
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextModificationRequest(ptCi, NULL),
                "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
        }

        VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_X2RecCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
    }
    else if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated)
    {
        // Inter-MeNB handover without SgNB change Inter-gNB-DU HO
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBAdditionRequestAcknowledge(ptCi),
            "VXIL_sendX2_SgNBAdditionRequestAcknowledge ERROR!\n");

        VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepSucc);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
        VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_X2SgnbRelReq);
    }
    else if ((ptCi->srb3Idx < MAX_SRB_PER_CALL) && (VCCB_FLAG_OFF == ptCi->scgFailureFlag) &&
             !GET_BAND_COMB_CHG_FLAG(ptCi->flag.setupFlag) && (NonFRPSCellChange == statusFR1FR2HO) &&
             (IS_FULLCONFIGURATION_NOT_IN_PROCESS(ptCi)))
    {
        // Inter-gNB-DU Handover using SRB3, no FR1<->FR2 HO case

        int16_t status = FnVCCB_SendNRRRC_RRCReconfiguration_F1ModiReq(ptCi, FALSE, FALSE, &ptCi->timer);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendNRRRC_RRCReconfiguration_F1ModiReq ERROR !!\n");

        ptCi->hoInfo.hoUsingSRB3 = TRUE;
        VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepSucc);
        MACRO_STAT_PMCC(STAT_INTRA_FREQ_PSCELL_CHANGE_SCGMOB_PER_GNB_DU_ID, EndcIntraFreqIntraSnChgPrepSucc, ptCi, 0, 0, 0, 0, 0);
        VLIB_CalculateSpCellChgTime(ptCi, CAL_START, SpCellChg_ThroughSRB3);
        VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_RRCRecCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);
    }
    else
    {
        // Inter-gNB-DU Handover not using SRB3 or
        // Inter-gNB-DU FR1<->FR2 Handover using SRB3 common handling
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextModificationRequest(ptCi, NULL),
            "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

        VCCB_NSA_SetCauseForX2SgnbModificationRequired(*ptCi);
        VCCB_RETURN_ON_CALL_ERROR(
            ptCi, VXIL_sendX2_SgNBModificationRequired(ptCi), "VXIL_sendX2_SgNBModificationRequired ERROR!\n");

        VLIB_CalculateSpCellChgTime(ptCi, CAL_START, SpCellChg_ThroughMCBSRB);
        VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepSucc);
        MACRO_STAT_PMCC(STAT_INTRA_FREQ_PSCELL_CHANGE_SCGMOB_PER_GNB_DU_ID, EndcIntraFreqIntraSnChgPrepSucc, ptCi, 0, 0, 0, 0, 0);
        VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_X2ModCnf);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state E1DrbModSecu
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_E1apDrbModSecu_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    callSRBInfo_type *ptSi = NULL;

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    ptCi->noCgConfigFlag = VCCB_FLAG_ON;
    int16_t status = VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR !!\n");

    // If security info is changed, initiate PDCP SN after RRC Reconfiguration
    if (VCCB_FLAG_ON == ptCi->ueInfo.securityModFlag)
    {
        if (MAX_SRB_PER_CALL > ptCi->srb3Idx)
        {
            ptSi = &(ptCi->srbInfo[ptCi->srb3Idx]);
            if (0 != FnPDCP_C_Init(&ptSi->pdcpRbContext, ptSi->rbInfo.rbId))
            {
                VCCB_PRINT_ERROR("FnPDCP_C_Init ERROR!\n");
                return ERROR;
            }
        }
    }

    VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_X2ModCnf);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);

    return status;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state E1DrbModSecu
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCInterDuHo_E1apDrbModSecu_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgFail_UpFail);
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state E1DrbModResume
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_E1apDrbModResume_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    const bool hasTgtDuResponse = (0 == GET_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag));
    const bool hasSrcDuResponse = (0 == GET_F1_REL_COMPLT_RCV_FLAG(ptCi->flag.respFlag));
    const bool hasMnResponse    = (0 == GET_SGNB_MOD_CONF_FLAG(ptCi->flag.respFlag));

    if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated)
    {
        // TODO s.pecak: inconsistent handling, does not check GET_F1_REL_COMPLT_RCV_FLAG.
        if (hasTgtDuResponse)
        {
            SET_PROC_FOR_STAT(ptCi, INTRA_SN_MOB_BY_ENB_PROC);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VHCM_FinalizeHandover(ptCi), "VHCM_FinalizeHandover ERROR!\n");
        }

        return GOOD;
    }

    const bool hasAllResponses = (hasTgtDuResponse && hasSrcDuResponse && hasMnResponse);
    if (hasAllResponses)
    {
        VLIB_FillRecoveryTimeCsl(ptCi);

        int16_t status = VLIB_CollectNsaHoSuccessStatAndFinalizeHandover(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VLIB_CollectNsaHoSuccessStatAndFinalizeHandover ERROR!\n");
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Context Response in state after starting TDcOverall timer during
 *Inter-DU Handover
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_TDcOverall_Common_E1UeBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    const bool hasAllResponses = (0 == ptCi->flag.respFlag);
    if (not hasAllResponses)
    {
        VCCB_PRINT_CALL_MSG(ptCi, "Essential message is not received yet(0x%016llX)\n", 1ULL * ptCi->flag.respFlag);
        return GOOD;
    }

    int16_t status = VHCM_DCInterDuHo_StartResume(ptCi);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VHCM_DCInterDuHo_StartResume ERROR!\n");
        return ERROR;
    }

    return status;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Context Modification Failure in state after starting TDcOverall timer during
 *Inter-DU Handover
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCInterDuHo_TDcOverall_Common_E1UeBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)  // gNB-DU HO MCG SRB
    {
        MACRO_STAT_PMCC(
            STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else if (HO_TYPE_MENB_INTERMNHO == ptCi->hoInfo.hoType.initiated)
    {
        VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
        VLIB_StatErabAddInterMnHO(ptCi, ErabAddFail_UpFail, TRUE);
    }
    else  // not using SRB3
    {
        VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgPrepFail_UpFail);
    }
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Context Response in state DCInterDuHo_E1ModRecov
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCInterDuHo_E1ModRecov_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = VFIL_sendF1_UeContextReleaseCommand(ptCi, NULL, 0, ID_SRB0, DU_HO_TARGET);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextReleaseCommand ERROR!\n");

    VLIB_SetCurStateAll(ptCi, STATE_DCInterDuHo_F1TgtDuRel);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextRelease_id);

    return GOOD;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Context Modification Failure in state DCInterDuHo_E1ModRecov
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCInterDuHo_E1ModRecov_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
    MACRO_STAT_PMCC(STAT_INTER_DU_PSCELL_CHANGE_GETTING_BACK, InterDuPscellChgGetBackFail, ptCi, 0, 0, 0, 0, 0);
    VCCB_NSA_PegIntraSnPscellChange(*ptCi, EndcIntraChgFail_UpFail);
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Response message in STATE_DCNonDuHo_E1DrbModPrep
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCNonDuHo_E1DrbModPrep_E1BearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VLIB_UpdateCounters_SuccessMrBasedResyncSCGFailure(ptCi, FALSE);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)
    {
        // Intra-gNB Handover using MCG SRB with no NR SCG resource established
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBModificationRequestAcknowledge(ptCi, FALSE),
            "VXIL_sendX2_SgNBModificationRequestAcknowledge ERROR!\n");

        ON_SGNB_REC_CMPL_FLAG(ptCi->flag.respFlag);

        // If security info is changed, initiate PDCP SN after RRC Reconfiguration
        if (VCCB_FLAG_ON == ptCi->ueInfo.securityModFlag)
        {
            if (MAX_SRB_PER_CALL > ptCi->srb3Idx)
            {
                callSRBInfo_type *ptSi = &(ptCi->srbInfo[ptCi->srb3Idx]);
                if (0 != FnPDCP_C_Init(&ptSi->pdcpRbContext, ptSi->rbInfo.rbId))
                {
                    VCCB_PRINT_ERROR("FnPDCP_C_Init ERROR!\n");
                    return ERROR;
                }
            }
        }

        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepSucc, ptCi, 0, 0, 0, 0, 0);
        VLIB_SetCurStateAll(ptCi, STATE_DCNonDuHo_X2RecCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
    }
    else  // HO_TYPE_MENB_INTERMNHO
    {
        // Inter-MN handover without SgNB change with no NR SCG resource established
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VXIL_sendX2_SgNBAdditionRequestAcknowledge(ptCi),
            "VXIL_sendX2_SgNBAdditionRequestAcknowledge ERROR!\n");

        VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepSucc);
        VLIB_SetCurStateAll(ptCi, STATE_DCNonDuHo_X2SgnbRelReq);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Failure in STATE_DCNonDuHo_E1DrbModPrep
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCNonDuHo_E1DrbModPrep_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    switch (ptCi->hoInfo.hoType.initiated)
    {
        case HO_TYPE_MENB_INITIATED:
            MACRO_STAT_PMCC(
                STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobPrepFail_UpFail, ptCi, 0, 0, 0, 0, 0);
            break;
        case HO_TYPE_MENB_INTERMNHO:
            VCCB_NSA_PegSgnbAddIntraLteHoNoSgnbChg(*ptCi, EndcAddLteHoNoSgnbChgPrepFail_UpFail);
            VLIB_StatErabAddInterMnHO(ptCi, ErabAddFail_UpFail, TRUE);
            break;
        default:
            break;
    }
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Response message in STATE_DCNonDuHo_E1DrbModResume
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCNonDuHo_E1DrbModResume_E1BearerModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    int16_t status = GOOD;

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobSucc, ptCi, 0, 0, 0, 0, 0);
        status = VIIL_sendCallResourceChangeInd(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VIIL_sendCallResourceChangeInd ERROR!\n");
    }

    status = VHCM_FinalizeHandover(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VHCM_FinalizeHandover ERROR!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handling E1 Bearer Modification Failure in STATE_DCNonDuHo_E1DrbModResume
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCNonDuHo_E1DrbModResume_E1BearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (HO_TYPE_MENB_INITIATED == ptCi->hoInfo.hoType.initiated)
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_MCGMOB_PER_GNB, EndcModByMenbMcgMobFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state DCMod_CompltInd
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCMod_CompltInd_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (!GETBIT(ptCi->flag.respFlag, F1_CTXT_MODRSP_FLAG) && !GETBIT(ptCi->flag.respFlag, E1_CTXT_MODRSP_FLAG) &&
        !GETBIT(ptCi->flag.respFlag, X2_SN_STATUS_FLAG))
    {
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VLIB_CollectNsaModSuccessStatAndSetStableState(ptCi),
            "VLIB_CollectNsaModSuccessStatAndSetStableState ERROR!\n");
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state DCMod_CompltInd
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCMod_CompltInd_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    // Note: For SN init SN modification counter counting probably never occur below. For SGNB_INIT_SGNB_MOD_PER_GNB
    // statistic EndcModBySgnbSucc is counting whenever gNB receives SgNBModificationConfirm, EndcModBySgnbFail_UpFail
    // by design is counting between SgNB Modification Required and SgNB Modification Confirm.
    if (SgNBModificationConfirm_cmd == ptCi->stateCmd)
    {
        MACRO_STAT_PMCC(STAT_SGNB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    FnVCCB_DetachCallMultiTimer(ptCi);

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state DCModE1_CompltInd
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCModE1_CompltInd_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    uint8_t needTriggerMeasGap = FALSE;

    if (!GETBIT(ptCi->flag.respFlag, F1_CTXT_MODRSP_FLAG) && !GETBIT(ptCi->flag.respFlag, E1_CTXT_MODRSP_FLAG) &&
        !GETBIT(ptCi->flag.respFlag, X2_SN_STATUS_FLAG))
    {
        if (GET_F1_RESOURCE_FLAG(ptCi->flag.setupFlag) &&
            (CHECK_MOD_EVENT_WITHOUT_STATECMD(ptCi, allowedBC) || (GOOD == VCCM_AreAllPrevBearersMcgSnTerm(ptCi))))
        {
            needTriggerMeasGap = TRUE;
        }

        if (SgNBInitBearerChangeToMcg_cmd == ptCi->stateCmd)
        {
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VFIL_sendF1_UeContextReleaseCommand(ptCi, NULL, 0, ID_SRB0, DU_HO_SOURCE),
                "VFIL_sendF1_UeContextReleaseCommand ERROR!\n");

            ptCi->bitFlag.isSnInitBearerChangeToMcgConfirmed = FALSE;

            // Increment internal statistic for SN init bearer type change
            VRRM_endcSNinitErabChgCount(ptCi, StatEndcErabModSucc);
        }

        // Make sure F1 timer is canceled bacause gNB does not wait for F1 UE Context Release Complete.
        VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.f1, CANCEL_TIMER);

        VLIB_CollectErabAddModStat(ptCi, ErabAddSucc, ErabModSucc, ErabModSucc_Change);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VLIB_CollectNsaModSuccessStatAndSetStableState(ptCi),
            "VLIB_CollectNsaModSuccessStatAndSetStableState ERROR!\n");

        if (TRUE == needTriggerMeasGap)
        {
            VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCM_EnqueueActionMeasReconfig(ptCi, MeasReconfigPriorAction_Mgap),
                "Queuing X2_SgNBModificationRequired(MGap Trigger) fail !!\n");
        }
    }
    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state DCModE1_CompltInd
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCModE1_CompltInd_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    // Note: For SGNB_INIT_SGNB_MOD_PER_GNB statistic EndcModBySgnbSucc is counting whenever gNB receives SgNBModificationConfirm,
    // EndcModBySgnbFail_UpFail by design is counting between SgNB Modification Required and SgNB Modification Confirm.

    if (SgNBInitBearerChangeToMcg_cmd != ptCi->stateCmd)
    {
        MACRO_STAT_PMCC(STAT_MENB_INIT_SGNB_MOD_PER_GNB, EndcModByMenbFail_UpFail, ptCi, 0, 0, 0, 0, 0);
    }
    else
    {
        // Increment internal statistic for SN init bearer type change
        VRRM_endcSNinitErabChgCount(ptCi, StatEndcErabModFail_UpFail);
    }

    VLIB_CollectErabAddModStat(ptCi, ErabAddFail_UpFail, ErabModFail_UpFail, ErabModFail_UpFail_Change);

    FnVCCB_DetachCallMultiTimer(ptCi);

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state DCSnChg_Suspend
 *                  after receivinng resposnees from all AUPFs
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCSnChg_Suspend_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_NSA_PegInterSnPscellChange(*ptCi, EndcInterChgSrcFail_UpFail);
    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // For sending transmissionStopIndicator
    const int16_t status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR !!\n");

    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_DCSnChg_F1TxStop);

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Response in state DCSnChg_E1SNStatus
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCSnChg_E1SNStatus_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = VXIL_sendX2_SNStatusTransferIfNeed(ptCi);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VXIL_sendX2_SNStatusTransferIfNeed ERROR!\n");
        return ERROR;
    }

    if (VCCB_FLAG_ON == ptCi->bitFlag.isReceivedUeContextRelease)
    {
        // X2UEContextRelease received
        status = VHCM_SgNBChange_GoToFinalState(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VHCM_SgNBChange_GoToFinalState ERROR !!\n");
            return ERROR;
        }
    }
    else
    {
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_UeContextRelease_id);
        VLIB_SetCurStateAll(ptCi, STATE_DCSnChg_X2UeCtxRel);
    }

    return status;
}

/****************************************************************************
 * @brief           Handling Bearer Context Modification Failure in state DCSnChg_E1SNStatus
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCSnChg_E1SNStatus_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_NSA_PegInterSnPscellChange(*ptCi, EndcInterChgSrcFail_UpFail);
    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (VCCB_FLAG_ON == ptCi->bitFlag.isReceivedUeContextRelease)
    {
        // X2UEContextRelease received
        VCCB_RETURN_ON_CALL_ERROR(
            ptCi, VHCM_SgNBChange_GoToFinalState(ptCi), "VHCM_SgNBChange_GoToFinalState ERROR !!\n");
    }
    else
    {
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_UeContextRelease_id);
        VLIB_SetCurStateAll(ptCi, STATE_DCSnChg_X2UeCtxRel);
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1BearerContextModificationResponse message in DCScgSusp_E1BCtxtModReq state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCScgSusp_E1BCtxtModRes_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if ((ScgFailCauseInfo_DuUlRadioLinkFailureOthers != ptCi->scgFailCauseInfo) &&
        (ScgFailCauseInfo_DuRlcMaxRetransmission != ptCi->scgFailCauseInfo) &&
        (ScgFailCauseInfo_MrBasedSuspend != ptCi->scgFailCauseInfo))
    {
        VCCB_RETURN_ON_CALL_ERROR(
            ptCi, VCCM_handleScgFailureInSuspend(ptCi, FALSE), "VCCM_handleScgFailureInSuspend ERROR !!\n");

        return GOOD;
    }

    VCCB_RETURN_ON_CALL_ERROR(
        ptCi, VCCM_handleScgFailureInSuspend(ptCi, TRUE), "VCCM_handleScgFailureInSuspend ERROR !!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1BearerContextModificationFailure message in DCScgSusp_E1BCtxtModReq state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCScgSusp_E1BCtxtModRes_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Required message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1apBearerContextModificationRequired(
    callInfo_type *ptCi, message_type *ptAll, msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    callDRBInfo_type *ptDi = NULL;
    uint8_t flagPDCPCountWrapAround_SA = false;  // boolean

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    auto &msg = *reinterpret_cast<const SctpDataTransfer_type *>(ptAll);
    bool flagModifiedBearer = false;
    bool flagPDCPCountWrapAround = false;  // boolean

    int16_t status = GOOD;

    // [MultiUp] Support
    uint16_t srvUpIdx = VRRM_FindServingUpIndex(ptCi, msg.endPointIdx);
    if (MAX_CU_UP_PER_CALL <= srvUpIdx)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "%s: E1idx:%u Is Not serving AUPF for this call \n", __func__, msg.endPointIdx);
        return GOOD;
    }

    ptCi->bitFlag.storeAndSendMsgInE1DrbPreemption = VCCB_FLAG_OFF;

    // Check E1 DRB Preemption.
    if( GOOD == FnVCCB_IsDrbPreemptionRequestByE1(ptCi, ptUnpack))
    {
        // Set storeAndSendMsgInE1DrbPreemption as ON for special operation in E1 Store/Send.
        // Do DRB Preemption Handling.
        ptCi->bitFlag.storeAndSendMsgInE1DrbPreemption = VCCB_FLAG_ON;
        status = FnVCCB_Execute_DRBPreemptionByE1(ptCi, ptAll, ptUnpack, srvUpIdx);
        ptCi->bitFlag.storeAndSendMsgInE1DrbPreemption = VCCB_FLAG_OFF;

        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_Execute_DRBPreemptionByE1 ERROR !!\n");
        return GOOD;
    }

    if (!VCCM_IsUeIdleState(ptCi->callType, ptCi->state))
    {
        if (GOOD == CheckUlPrimaryPathSwitchIsRequired(ptCi, ptUnpack))
        {

            status = FnVCCB_StoreE1Message(E1_BearerContextModificationRequired_id, ptCi, ptAll, ptUnpack);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL::StoreBearerContextModificationRequired ERROR !!\n");

            status = FnVCCB_SendE1Message(E1_BearerContextModificationConfirm_id, ptCi, srvUpIdx);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendBearerContextModificationConfirm ERROR !!\n");

            if (ERROR == VCCM_enqueueAction(ptCi, ENQUEUE_ACTION_E1B_MOD_REQ))
            {
                VCCB_PRINT_CALL_INFORMATIVE(
                    ptCi, "VCCM_enqueueAction(ENQUEUE_ACTION_E1B_MOD_REQ) in %s fail\n", __FUNCTION__);
            }
            else
            {
                VCCB_PRINT_CALL_COMMENT(ptCi, "UP init Mod for UL Primary Path Switch: Enqueued\n");
            }
        }
        else
        {
            if (ERROR ==
                VCCM_enqueueAction(ptCi, ENQUEUE_ACTION_E1_MSG_FROM_UP, ptAll, E1_BearerContextModificationRequired_id))
            {
                VCCB_PRINT_CALL_INFORMATIVE(
                    ptCi, "VCCM_enqueueAction(ENQUEUE_ACTION_E1_MSG_FROM_UP) in %s fail\n", __FUNCTION__);
            }
            else
            {
                VCCB_PRINT_CALL_COMMENT(ptCi, "E1apBearerContextModificationRequired: Enqueued\n");
            }
        }
        return GOOD;
    }

    // SVR22B FGR-CC5203: If in SCG Suspended status this message comes for NOT SCG resume purpose,
    // SN sends E1 Bearer Context Modification Confirm
    // (w/ current Cell Group Information of DRB Modified List, not the one of E1 Bearer Context Modification Required)
    // (FR61)
    // => Making E1 Bearer Context Modification Confirm before saving IEs (=> CHECK!!!)
    if (ptCi->callType == E_CALLTYPE_SA && IS_SCG_SUSPENDED(ptCi))
    {
        status = FnVCCB_SendE1Message(E1_BearerContextModificationConfirm_id, ptCi, srvUpIdx);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendBearerContextModificationConfirm ERROR !!\n");

        return GOOD;
    }

    status = FnVCCB_StoreE1Message(E1_BearerContextModificationRequired_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL::StoreBearerContextModificationRequired ERROR !!\n");

    status = FnVCCB_SendE1Message(E1_BearerContextModificationConfirm_id, ptCi, srvUpIdx);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendBearerContextModificationConfirm ERROR !!\n");

    // When VCCB receives "Bearer Context Modification Required" with cause "PDCP Count Wrap Around"

    if (ptCi->callType == E_CALLTYPE_SA)
    {
        for (int pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
        {
            callPDUSessInfo_type *pPDUSessInfo = &ptCi->pduSess[pduIdx];
            if (CI_use != pPDUSessInfo->usage) continue;
            VCCB_PRINT_CALL_DATA(ptCi, "[E1DBG] ptCi->pduSess[%d].usage is CI_use and pduSessStatus:%d != S_Setup\n",
                pduIdx, pPDUSessInfo[pduIdx].pduSessStatus);

            for (int drbidx = 0; drbidx < MAX_DRB_PER_PDU_SESS; drbidx++)
            {
                if (pPDUSessInfo->drbIndex[drbidx] >= MAX_DRB_PER_SA_CALL)  // Not yet allocated
                    continue;

                ptDi = &ptCi->drb[pPDUSessInfo->drbIndex[drbidx]];

                if (ptDi->cause.e1Cause == CauseE1apRadioNetworkLayerPdcpCountWrapAround)
                {
                    flagPDCPCountWrapAround_SA = true;
                }
                if (VCCB_FLAG_ON == ptDi->ulPdcpPathEnq)
                {
                    flagModifiedBearer = true;
                }
            }
        }
    }

    if (flagPDCPCountWrapAround_SA)
    {
        if (IS_MRDC_SN(ptCi))
        {
            if (ERROR == VLIB_EnqueueSNPdcpCountWrapAround(ptCi))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_EnqueueSNPdcpCountWrapAround ERROR !!\n");
            }
        }
        else
        {
            if (ERROR == VLIB_EnqueueIntraCellHO(ptCi))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_EnqueueIntraCellHO ERROR !!\n");
            }
        }
    }

    if (E_CALLTYPE_NSA == ptCi->callType)
    {
        for (callERABInfo_type &erabInfo : ptCi->erabInfo)
        {
            if (erabInfo.usage != CI_use) continue;

            if (VCCB_FLAG_ON == erabInfo.ulPdcpPathEnq)
            {
                flagModifiedBearer = true;
            }
            if (erabInfo.cause.e1Cause == CauseE1apRadioNetworkLayerPdcpCountWrapAround)
            {
                flagPDCPCountWrapAround = true;
            }
        }
    }

    if (flagPDCPCountWrapAround == true)
    {
        SET_STATE_CMD(SNInitSgNBMod_cmd);
        VCCB_PRINT_CALL_COMMENT(ptCi, "Bearer Modification start\n");
        ptCi->cause.x2Cause = CauseX2apRadioNetworkLayerCountReachesMaxValue;
        status = VXIL_sendX2_SgNBModificationRequired(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequired ERROR!\n");
        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
        VLIB_SetCurStateAll(ptCi, STATE_DCSNInitMod_X2ModRqrd);
        VCCB_PRINT_CALL_COMMENT(ptCi, "Bearer Modification end, state Command %d \n", ptCi->stateCmd);
    }

    if (ptCi->stateCmd == SgnbInitBearerRelease_cmd)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "Bearer Modification start\n");
        ptCi->cause.x2Cause = CauseX2apRadioNetworkLayerActionDesirableForRadioReasons;
        status = VXIL_sendX2_SgNBModificationRequired(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequired ERROR!\n");

        VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
        VLIB_SetCurStateAll(ptCi, STATE_DCSgNBInitRel_X2ModRqrd);
        VCCB_PRINT_CALL_COMMENT(ptCi, "Bearer Modification end, state Command %d \n", ptCi->stateCmd);
    }

    if (flagModifiedBearer == false)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "VCCM_ALL_E1apBearerContextModificationRequired - No bearer to modify\n");
        return GOOD;
    }

    status = VCCM_HandleDequeuedE1BModReqAction(ptCi, ENQUEUE_ACTION_E1B_MOD_REQ);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_HandleDequeuedE1BModReqAction ERROR !!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for E1 Bearer Context Modification Required NE-DC
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_HandleDequeuedE1BModReqActionSA(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    callDRBInfo_type *ptDi = NULL;
    int16_t status = ERROR;

    SET_STATE_CMD(E1UlPathSwitch_cmd);

    for (int pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
    {
        callPDUSessInfo_type *pPDUSessInfo = &ptCi->pduSess[pduIdx];
        if (CI_use != pPDUSessInfo->usage) continue;

        for (int drbidx = 0; drbidx < MAX_DRB_PER_PDU_SESS; drbidx++)
        {
            if (pPDUSessInfo->drbIndex[drbidx] >= MAX_DRB_PER_SA_CALL) continue;

            ptDi = &ptCi->drb[pPDUSessInfo->drbIndex[drbidx]];

            if (CI_use != ptDi->usage) continue;

            if (VCCB_FLAG_ON == ptDi->ulPdcpPathEnq)
            {
                if (IS_SELECTED_MRDC(MRDC_NRDC))
                {
                    SET_SPLIT_DRB_UL_PDCP_PATH_NRDC(ptCi, ptDi->ulPdcpPathEnqToSet, pPDUSessInfo->drbIndex[drbidx]);
                }
                else
                {
                    SET_SPLIT_PER_DRB_UL_PDCP_PATH_NEDC(ptCi, ptDi->ulPdcpPathEnqToSet, pPDUSessInfo->drbIndex[drbidx]);
                }
                ptDi->ulPdcpPathEnq = VCCB_FLAG_OFF;
            }
        }
    }

    if (IS_SELECTED_MRDC(MRDC_NRDC) && IS_MRDC_SN(ptCi))
    {
        status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

        VLIB_SetCurStateAll(ptCi, STATE_SN_SNE1UlPath_F1ModReq);
        VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    }
    else
    {
        // set MNTERM_DRB_TO_BE_MODIFIED_TO_SN_FLAG(ptDi->mrdcFlag.respFlag) for transfer
        for (int pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
        {
            callPDUSessInfo_type *pPDUSessInfo = &ptCi->pduSess[pduIdx];
            if (CI_use != pPDUSessInfo->usage) continue;

            for (int drbidx = 0; drbidx < MAX_DRB_PER_PDU_SESS; drbidx++)
            {
                if (pPDUSessInfo->drbIndex[drbidx] >= MAX_DRB_PER_SA_CALL) continue;

                ptDi = &ptCi->drb[pPDUSessInfo->drbIndex[drbidx]];

                if (ptDi->bearerType.val == MN_Term_SPLIT_Bearer)
                {
                    ON_MNTERM_DRB_TO_BE_MODIFIED_TO_SN_FLAG(ptDi->mrdcFlag.respFlag);
                }
            }
        }

        SET_MRDC_PROCEDURE(ptCi, MRDC_PROC_CODE_SnMod, MRDC_PROC_PURPOSE_ULPathChange, MRDC_PROC_INITIATOR_MN);
        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        INIT_MRDC_PROCEDURE(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage ERROR!\n");

        VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
        VLIB_SetCurStateAll(ptCi, STATE_SNE1UlPath_XnModReq);
    }
    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for E1 Bearer Context Modification Required ENDC
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_HandleDequeuedE1BModReqActionENDC(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    uint8_t needModFlag = FALSE;

    for (callERABInfo_type &erabInfo : ptCi->erabInfo)
    {
        if (erabInfo.usage != CI_use) continue;

        if (VCCB_FLAG_ON == erabInfo.ulPdcpPathEnq)
        {
            ptCi->ulPathControl.prev.ulPdcpPath = ptCi->ulPathControl.curr.ulPdcpPath;
            erabInfo.ulPdcpPathEnq = VCCB_FLAG_OFF;
            ptCi->ulPathControl.curr.ulPdcpPath = (enumUlPdcpPath)erabInfo.ulPdcpPathEnqToSet;
            erabInfo.ulPdcpPathToMod = UlConfigBearer_ModOn;

            needModFlag = TRUE;
        }
    }

    if (FALSE == needModFlag)
    {
        VCCB_PRINT_CALL_COMMENT(ptCi, "not need to trigger UL Primary Path Switch\n");
        return GOOD;
    }

    VCCM_SetSplitDrbUlPdcpPath(ptCi);

    SET_STATE_CMD(E1UlPathSwitch_cmd);

    ptCi->cause.x2Cause = CauseX2apRadioNetworkLayerActionDesirableForRadioReasons;
    int16_t status = VXIL_sendX2_SgNBModificationRequired(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VXIL_sendX2_SgNBModificationRequired ERROR !!\n");

    VLIB_AttachTimer(&ptCi->timer, ptCi, timX2_DcOverall_id);
    VLIB_SetCurStateAll(ptCi, STATE_DCE1UlPath_X2ModCnf);

    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for E1 Bearer Context Modification Required
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_HandleDequeuedE1BModReqAction(callInfo_type *ptCi, enqueuedAction)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    int16_t status = GOOD;

    if (E_CALLTYPE_SA == ptCi->callType)
    {
        status = VCCM_HandleDequeuedE1BModReqActionSA(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCM_HandleDequeuedE1BModReqActionSA ERROR !!\n");
            return ERROR;
        }
    }
    else
    {
        status = VCCM_HandleDequeuedE1BModReqActionENDC(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCM_HandleDequeuedE1BModReqActionENDC ERROR !!\n");
            return ERROR;
        }
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for E1 msg
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_HandleDequeuedE1Msg(callInfo_type *ptCi, enqueuedAction qAction)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    message_type *ptAll;
    msgUnpack_type *ptUnpack;

    VCCB_CHECK_RANGE_AND_RETURN_ERROR(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB, "ptCi->myThreadId");
    ptAll = reinterpret_cast<message_type*>(&THREAD_INFO->call[ptCi->myThreadId].buf.rx);
    ptUnpack = &THREAD_INFO->call[ptCi->myThreadId].buf.unpack.call.buff;

    int16_t status = MsgQueueManager::getInstance().dequeueMsgQueueMap(ptCi, ptAll, qAction);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "dequeueMsgQueueMap ERROR!\n");

    status = VCCB_GetMsgUnpack_checkMsgId(ptAll, ptUnpack, ptCi->myThreadId);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCB_GetMsgUnpack_checkMsgId ERROR!\n");

    if (E1_BearerContextModificationRequired_id == ptUnpack->msgId)
    {
        status = VCCM_ALL_E1apBearerContextModificationRequired(ptCi, ptAll, ptUnpack);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ALL_E1apBearerContextModificationRequired ERROR!\n");
    }
    else if (E1_BearerContextReleaseRequest_id == ptUnpack->msgId)
    {
        status = VCCM_ALL_E1apBearerContextReleaseRequest(ptCi, ptAll, ptUnpack);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ALL_E1apBearerContextReleaseRequest ERROR!\n");
    }
    else
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Dequeue unexpected MsgId (%d) ERROR !!\n", ptUnpack->msgId);
    }

    return status;
}

int16_t FnVCCB_Proc_E1InactivityNotification_ByRRCInactive(callInfo_type *ptCi, uint8_t *ptDrbFlag)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptDrbFlag);

    int16_t status = GOOD;

    if (IS_RRC_INACTIVE_DRB_LEVEL(ptCi))
    {
        *ptDrbFlag = VCCB_FLAG_ON;
    }

    const auto &ueActivityStatus = RscCall::getDataActivityStatus(*ptCi);

    ptCi->ranPagingInfo.dlDataFlag = VCCB_FLAG_OFF;

    if (ueActivityStatus == ActivityStatus_ACTIVE)
    {
        // AI: data restarted while still RRC_CONNECTED. If Point 2 had deferred the
        // INACTIVE transition, this is the deferral's success outcome — recycle the
        // extension budget (extendCount / pending timer) for the next idle period.
        // Observation-only bookkeeping; does not alter the legacy flow below.
        FnVCCBAiClient_OnDataActivityResumed(ptCi);

        status = FnVCCB_UpdateRRCInactStateAndDoActionIfExpiry(ptCi, RRCInactiveEvent_DATA_RESTART);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_UpdateRRCInactStateAndDoActionIfExpiry ERROR\n");
    }
    else
    {
        status = FnVCCB_UpdateRRCInactStateAndDoActionIfExpiry(ptCi, RRCInactiveEvent_DATA_EXPIRY);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_UpdateRRCInactStateAndDoActionIfExpiry ERROR\n");
    }

    return status;
}

int16_t FnVCCB_Proc_E1InactivityNotification_ByUserInactivity(callInfo_type *ptCi, uint8_t *ptDrbFlag)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptDrbFlag);

    int16_t status = GOOD;

    enumActivityNotifLvl_type selectedLevel = FnVCCB_Get_UserInactivity_NotiLevel(ptCi);
    if (selectedLevel == activityNotifLvl_DRB)
    {
        *ptDrbFlag = VCCB_FLAG_ON;
    }

    const auto &ueActivityStatus = RscCall::getDataActivityStatus(*ptCi);
    if (ueActivityStatus == ActivityStatus_ACTIVE)
    {
        FnVCCB_UpdateUserInactStateAndDoActionIfExpiry(ptCi, UserInactivityEvent_DATA_RESTART);
    }
    else
    {
        FnVCCB_UpdateUserInactStateAndDoActionIfExpiry(ptCi, UserInactivityEvent_DATA_EXPIRY);
    }

    const bool isNedcInactivityUe = (IS_NEDC(ptCi) && IsSupport_NEDC_InActivity());
    if (isNedcInactivityUe && (selectedLevel == activityNotifLvl_DRB))
    {
        if (ptCi->CallUserInactivity.ueInactivityState != UserInactivityState_DATA_SIG_EXPIRED)
        {
            if (VCCM_checkForSplitBearerSnNodeRelease(ptCi))
            {
                if (ptCi->state == STATE_Conn)
                {
                    ptCi->mrdcInfo.sNodeRelCause = MrdcRelCause_InactivityRelease;
                    status = VCCM_TriggerMgNBInitSNReleaseProcedure(ptCi);
                    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_TriggerMgNBInitSNReleaseProcedure ERROR !!\n");
                }
                else
                {
                    if (ERROR == VCCM_enqueueAction(ptCi, ENQUEUE_ACTION_E1_INACTIVITY_NOTI))
                    {
                        VCCB_PRINT_CALL_INFORMATIVE(
                            ptCi, "VCCM_enqueueAction(ENQUEUE_ACTION_E1_INACTIVITY_NOTI) in %s fail\n", __FUNCTION__);
                    }
                }
            }
        }
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Inactivity Notification message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1apBearerContextInactivityNotification(
    callInfo_type *ptCi, message_type *ptAll, msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    uint8_t needDrbBackupFlag = VCCB_FLAG_OFF;

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextInactivityNotification_id, ptCi, ptAll, ptUnpack);
    if (ERROR == status) return ERROR;

    // -----------------------------------------------------------------------------------
    // Caution : When need action which is call_release or SN_Release or Suspend,
    //           Do not procedure here if state is not Conn or DB.
    //           Check State and using Event Queueing : ENQUEUE_ACTION_E1_INACTIVITY_NOTI.
    // -----------------------------------------------------------------------------------

    // FGR-CC0103
    if (IS_SUPPORT_RRC_INACTIVE(ptCi))
    {
        if (IS_NRDC_MN(ptCi))
        {
            // SVR25B
            if (ptCi->ueInfo.rrcResumeInfo.actionInSNRelease == SN_REL_IN_RRC_INACTIVE_DECIDED)
            {
                // handles E1_BearerContextInactivityNotification (in-active) in the middle of SN-release.
                ptCi->ueInfo.rrcResumeInfo.actionInSNRelease = SN_REL_IN_RRC_INACTIVE_DECIDED_AND_CONFIRMED;

                VCCB_PRINT_CALL_DATA(ptCi,
                        "[RRCInactive] Received E1_BearerContextInactivityNotification in the middle of SN-Release! (state:%u/actionInSNRelease:%u)\n",
                        ptCi->state, ptCi->ueInfo.rrcResumeInfo.actionInSNRelease);
            }
            else
            {
                // processing as usual...
                status = FnVCCB_Proc_E1InactivityNotification_ByRRCInactive(ptCi, &needDrbBackupFlag);
                VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_Proc_E1InactivityNotification_ByRRCInactive ERROR!!\n");
            }
        }
        else if (ptCi->callType == E_CALLTYPE_SA)
        {
            // processing as usual...
            status = FnVCCB_Proc_E1InactivityNotification_ByRRCInactive(ptCi, &needDrbBackupFlag);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_Proc_E1InactivityNotification_ByRRCInactive ERROR!!\n");
        }
    }
    else
    {
        ptCi->ueInfo.rrcResumeInfo.preventSNAddFlag = FALSE;    // just in case...
        status = FnVCCB_Proc_E1InactivityNotification_ByUserInactivity(ptCi, &needDrbBackupFlag);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_Proc_E1InactivityNotification_ByUserInactivity ERROR!!\n");
    }

    FnVCCB_BackupE1InactivityInfo(ptCi, needDrbBackupFlag);
    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for E1 Bearer Context Inactivity Notification
 *
 * @param[in/out]   ptCi    : Call information / callInfo_type
 * @param[in]       qAction : no use / enqueuedAction
 * @return          status  : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t FnVCCB_HandleDequeuedE1InactivityNoti(callInfo_type *ptCi, enqueuedAction)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "Start to dequeue action for E1_BearerContextInactivityNotification\n");

    int16_t status = FnVCCB_DoInactivityActionInIdleState(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_DoInactivityActionInIdleState ERROR !!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Private message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1apPrivateMessage(callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    E1_PrivateMessage *request = reinterpret_cast<E1_PrivateMessage *>(
        const_cast<unsigned char *>(ptUnpack->unpack + ptUnpack->unpackMsgOffset));

    int16_t status = GOOD;
    status = FnVCCB_StoreE1Message(E1_PrivateMessage_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_StoreE1Message(E1_PrivateMessage_id) ERROR!\n");

    status = VCCM_ProcE1PrivateMessage(ptCi, request);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ProcE1PrivateMessage ERROR!\n");

    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 ErrorIndication message in ALL state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR, BREAK / int16_t
 ****************************************************************************/
int16_t VCCM_ALL_E1ErrorIndication(callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;
    status = FnVCCB_StoreE1Message(E1_ErrorIndication_id, ptCi, ptAll, ptUnpack);

    if (ERROR == status || MULTI_UP_ERROR == status)
    {
        // Check for Class 2 procedures. (Elementary procedure with no response message e.g. Trace Start)
        // Idle State means there is no related state function for response message
        if (TRUE == VCCM_IsUeIdleState(ptCi->callType, ptCi->state))
        {
            if (ptCi->endPointInfo.srvAupfNum > 0 &&
                true == CheckifPartialSuccesAndEnueueBearersRelease(ptCi))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "Multi-UP: Parital release triggered by E1 ErrorIndication\n");
            }
            else
            {
                // In case it was last serving UP release call
                VCCB_CALL_RELEASE_TRIGGER(ptCi);  // Call Job Release
            }
        }
    }

    return GOOD;
}

void VCCM_DCAddCommon_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_PRINT_CALL_ERROR(ptCi, "E1_BearerContextModificationFailure !!\n");
    FnVCCB_DetachCallMultiTimer(ptCi);

    // Statistics
    VLIB_StatEndcAdd(ptCi, StatEndcAddFail_UpFail, __FUNCTION__);

    // ERAB_ADD Fail
    for (int i = 0; i < MAX_ERAB_PER_CALL; i++)
    {
        auto &erabInfo = ptCi->erabInfo[i];
        if (erabInfo.usage != CI_use) continue;
        if (erabInfo.erabStatus != ERAB_Modify) continue;
        // Changed erabStatus to ERAB_Modify after receiving F1_UEContextSetupResponse

        // Statistics Update
        ptCi->ptTempRi = &erabInfo;
        MACRO_STAT_PMCC_QCI(STAT_ERAB_ADD_PER_GNB, ErabAddFail_UpFail, ptCi, ptCi->ptTempRi, 0, 0, 0, 0);
    }
}

int16_t VCCM_DCAddCommon_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, message_type *ptAll, msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");
    if (ERROR == status)
    {
        ptCi->endcAddCpFailDetailCause = StatEndcAddCpFail_RxE1BearerContextModificationResponseFail;
        return status;
    }

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Setup Response message in E2C state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_E2C_E1Setup_E1apBearerContextSetupSuccess(callInfo_type *ptCi, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);
    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    // Allocate SRB2 only if UE has PDU Session (When E1 Setup)
    if (ERROR == VLIB_AllocCallSrbInfo(ptCi, 1))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocCallSrbInfo ERROR!!\n");
        MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalResourceAllocFail);
        return ERROR;
    }
    ptCi->srbInfo[1].srbStatus = S_Setup;

    VLIB_ConfigSecondaryCellList(ptCi);

    FnVCCB_SetMrdcSelectedBearerTypePerDrb(ptCi);  // NRDC

    int16_t status = VCCB_PrepareNewMeasConfig(ptCi);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VCCB_PrepareNewMeasConfig ERROR!\n");
        return ERROR;
    }

    if (GET_SECURITY_MODE_CMD_FLAG(ptCi->flag.setupFlag))  // Security Mode Command procedure is already done
    {
        status = VFIL_sendF1_UeContextSetupRequest(ptCi, NULL, 0);
        if (status == ERROR)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextSetupRequest ERROR !!\n");
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalSendF1MessageError);
            VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
            return ERROR;
        }
        VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextSetup_id);
    }
    else
    {
        // Send Security And F1UeContextSetupRequest (with pDbg->flag option)
        status = FnVCCB_SendSecurityCommandAndF1UeContextSetupRequest(ptCi, ptUnpack->msgId);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendSecurityCommandAndF1UeContextSetupRequest() ERROR !! \n");
    }

    VLIB_SetCurStateAll(ptCi, STATE_E2C_F1Setup);

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Setup Failure message in E2C state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 ****************************************************************************/
void VCCM_E2C_E1Setup_E1apBearerContextSetupUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Setup Response message in PDUChg_E1Setup state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_PDUChg_E1Setup_E1apBearerContextSetupSuccess(callInfo_type *ptCi, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);
    int16_t status = GOOD;

    FnVCCB_SetMrdcSelectedBearerTypePerDrb(ptCi);

    if (IS_NRDC_MN(ptCi) && ptCi->numSRB >= 2)  // not first drb on nrdc
    {
        // Set NR-DC E1 Setup.
        ptCi->drbPartialInfo.isNrDcMnE1Setup = TRUE;

        // this will be happened if mn-term bearer is going to be added in condition where the only sn-term bearer is
        // exist.
        status = FnVCCB_NrdcSNModPrepare(ptCi, MRDC_PROC_PURPOSE_PDUSessionChange);
        if (status == GOOD || status == ERROR)
        {
            if (status == ERROR)
            {
                VCCB_PRINT_CALL_ERROR(
                    ptCi, "FnVCCB_NrdcSNModPrepare(MRDC_PROC_PURPOSE_PDUSessionChange) ERROR in %s!\n", __FUNCTION__);
            }
            return status;
        }

        VLIB_ConfigSecondaryCellList(ptCi);

        if (VCCM_AnySCellIsDuringAnyAction(ptCi))
        {
            status = VCCB_PrepareNewMeasConfig(ptCi);
            VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCB_PrepareNewMeasConfig ERROR!\n");
        }

        status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
            return ERROR;
        }

        VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);
        VLIB_SetCurStateAll(ptCi, STATE_PDUChg_F1Mod);
    }
    else
    {
        // Allocate SRB2
        // E1 Setup can be failed, if failed occurs, there is no release action for SRB2. So SRB2 assign here.
        if (ERROR == VLIB_AllocCallSrbInfo(ptCi, 1))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VLIB_AllocCallSrbInfo ERROR!!\n");
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalResourceAllocFail);
            return ERROR;
        }
        ptCi->srbInfo[1].srbStatus = S_Setup;

        VLIB_ConfigSecondaryCellList(ptCi);

        status = VCCB_PrepareNewMeasConfig(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCB_PrepareNewMeasConfig ERROR!\n");
            return ERROR;
        }

        status = VFIL_sendF1_UeContextSetupRequest(ptCi, NULL, 0);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextSetupRequest ERROR!\n");
            return ERROR;
        }

        VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextSetup_id);
        VLIB_SetCurStateAll(ptCi, STATE_PDUChg_F1Setup);
    }

    return status;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Response message in PDUChg_E1Mod_TnlUpdate state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_PDUChg_E1Mod_TnlUpdate_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CHECK_E_RANGE_AND_RETURN(ptCi->myThreadId, MAX_CALL_THREAD_NUM_VCCB);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    int16_t status = FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer(ptCi, FALSE, FALSE, ID_SRB1);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer ERROR !!\n");

    VLIB_SetCurStateAll(ptCi, STATE_PDUChg_RRCRecCmpl);
    if (status == BREAK)
    {
        // No Need RRC Reconfig. Do RrcReconfComplt
        status = FnVCCB_Common_PDUChg_RrcReconfigComplt_Success(ptCi, true);
    }
    else
    {
        VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);
    }

    return status;
}


/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Failure message in PDUChg_E1Mod_TnlUpdate state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_PDUChg_E1Mod_TnlUpdate_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_PDUChg_E1Mod_Rollback_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    VCCM_HandlePDUSessionChangeFailure(ptCi);
    VCCM_SetStableState_CallSetup(ptCi);

    return GOOD;
}

int16_t FnVCCB_UECMod_E1Mod_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    if (GET_F1_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc))
    {
        F1_MODIFY_MSG_SEND_AND_SET_STATE_FOR_UE_CTXT_MOD_PROC(ptCi, STATE_UECMod_F1Mod);
        OFF_F1_PROC_REQUIRED_FLAG(ptCi->modProcInfo.selectedModProc);
    }
    else
    {
        NG_RSP_MSG_SEND_AND_PROCEDURE_DONE_FOR_UE_CTXT_MOD_PROC(ptCi);
    }
    
    return GOOD;
}

int16_t FnVCCB_UECMod_E1Mod_E1apBearerContextModificationUnsuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
    return GOOD;
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Modification Response message in E2C_RRCRecCmpl state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_E2C_RRCRecCmpl_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
}

/****************************************************************************
 * @brief           Handling E1 Bearer Context Modification Response commonly,
 *                  and go to DB state.
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_Common_GoToDB_E1apBearerContextModificationSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);

    SET_STATE_CMD(Idle_cmd);
    VLIB_SetCurStateAll(ptCi, STATE_DB);
    ptCi->cslInfo.cslEvent = csl_event_undefined;

    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Modification Failure in state E1DrbModPrep
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
void VCCM_DCUPPS_E1DrbMod_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
}

/****************************************************************************
 * @brief           Handling Bearer Modification Response in state DCE1UlPath_E1ModReq
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_DCE1UlPath_E1ModReq_E1apBearerContextModificationResponse(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    int16_t status = GOOD;

    // send F1_UeContextModificationRequest
    status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    if (status == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR!!\n");
        return ERROR;
    }
    VLIB_SetCurStateAll(ptCi, STATE_DCE1UlPath_F1Mod);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timF1_UeContextModification_id);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on DCE1UlPath_E1ModReq state
 * @param[in/out]   ptCi :      Call information / callInfo_type
 ****************************************************************************/
void VCCM_DCE1UlPath_E1ModReq_E1apBearerContextModificationUnSuccess(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_VOID(ptCi);

    VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
}

/****************************************************************************
 * @brief           Handle received E1 Bearer Context Release Complete message in Conn_RelCmpl state
 *
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_Conn_RelCmpl_E1apBearerContextReleaseComplete(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1RelCmpl) Waiting for other UPs to respond\n");

    if (status == ERROR)
    {
        VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
        return ERROR;
    }

    return status;
}

/****************************************************************************
 * @brief  Handle received E1BearerContextModificationResp on SNAddPrepare_E1Mod1,2 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(uint32_t msgId, callInfo_type *ptCi,
    const message_type *ptAll, const msgUnpack_type *ptUnpack, uint8_t timerCancelFlag)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(msgId, ptCi, ptAll, ptUnpack);
    if (status == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message, ERROR!\n");
        if (timerCancelFlag)
        {
            VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
        }
        return status;
    }
    else if (status == BREAK)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message, BREAK!\n");
        return status;
    }

    if (timerCancelFlag)
    {
        VCCB_CLEAR_TIMER(&ptCi->timer, CANCEL_TIMER);
    }

    return status;
}

int16_t FnVCCB_MrdcParallelMsgHandling_Common_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_MrdcParallelMsgHandling_Common_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}
/****************************************************************************
 * @brief  Handle the E1BearerContextModificationResp received on SNAddPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepare_E1Mod1_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    SET_STATE_CMD(MrdcXnSNAdd_cmd);

    uint8_t targetXN = e_Snode_MnToSrcSn;
    status = FnVCCB_SendXNMessage(XN_SNodeAdditionRequest_id, ptCi, &targetXN, nullptr);
    if (ERROR == status) return ERROR;

    VLIB_StatNrdcAdd(ptCi, StatNrdcAddMn_Att);

    // FGR-CC5200:FR760 FR770
    FnVCCB_NrdcSnConnInfoStatsCmd(ptCi, ptCi->servCellNum, ConnInfoStatsCmd_SetupTimeStart);

    /*
      xn_snConn should be set after receiving SNodeAdditionRequestAck.
      xn_snConn will be used to check whether vccb has to trigger the release procedure to release the resource of peer
      on the abnormal or normal scenario. Vccb need to know the SN's XNAP UE ID when sending ueContextReleaseRequest on
      SNAddPrepare Failure condition. So xn_snConn will be set after receiving SNodeAdditionRequestAck. cf) Although
      vccb doesn't send UeContextRelease to SNode on SNAddPrepare Failure condition, SNode will release its internal
      resource after expiration of the DCPrepare Timer.

        ON_MRDC_RSC_SETUP_FLAG(xn_snConn);
    */

    VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNAddPrepare_XnAdd);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on SNAddPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepare_Common_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    // TODO Something...

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the expiration for E1BearerContextModificationResp on SNAddPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t FnVCCB_SNModPrepare_E1Mod1_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************************
 * @brief  Handle the E1BearerContextModificationResp received on SNAddPrepare_E1Mod2 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepare_E1Mod2_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    FnVCCB_InitMrsInfo(ptCi, MRS_ALL_INIT);
    FnVCCB_InitSccData(ptCi, SCC_ALL_INIT);
    
    VLIB_ConfigSecondaryCellList(ptCi);
    if (ERROR == VCCB_PrepareNewMeasConfig(ptCi))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VCCB_PrepareNewMeasConfig ERROR!\n");
        return ERROR;
    }

    SET_STATE_CMD(MrdcF1DCUeSetup_cmd);
    ON_UECAPA_NEED_FWD_FLAG(ptCi->flag.setupFlag);

    if (ERROR == VFIL_sendF1_UeContextModificationRequest(ptCi, NULL))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
        return ERROR;
    }

    ON_MRDC_RSC_SETUP_FLAG(f1_mrdcUeSetup);

    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNAddPrepare_F1Mod);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the expiration for E1BearerContextModificationResp on SNAddPrepare_E1Mod2 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepare_E1Mod2_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************************
 * @brief  Handle the E1BearerContextModificationResp received on SNModPrepare_E1Mod state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNModPrepare_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
        return ERROR;
    }

    VLIB_FinalizeMrdcProcedure(ptCi);

    return GOOD;
}

int16_t FnVCCB_SNModPrepare_E1Mod2_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    SET_STATE_CMD(MrdcF1DCUeMod_cmd);

    if (ERROR == VFIL_sendF1_UeContextModificationRequest(ptCi, NULL))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
        return ERROR;
    }

    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNModPrepare_F1Mod);

    return GOOD;
}

int16_t FnVCCB_SNModPrepare_E1Mod2_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNModPrepareFail_E1Mod2_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNModPrepareFail_E1Mod2_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    return FnVCCB_SNModPrepare_E1Mod2_timE1_BearerContextModification(ptCi, ptAll, ptUnpack);
}
/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on SNModPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepareFail_ResourceRel_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
        }

        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    OFF_MRDC_RSC_SETUP_FLAG(e1_scgSetup);

    return status;
}

int16_t FnVCCB_SNAddModComplt_E1F1NGMod_timCommon(callInfo_type *ptCi, message_type *ptAll, msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    // LOGGING
    logShmIntMsg(ptCi, RX_MSG, BLK_TIME, ptUnpack->msgId, sizeof(messageHdr_type), (uint8_t *)ptAll);

    switch (ptUnpack->msgId)
    {
        case timF1_UeContextModification_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.f1, INIT_TIMER);
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutF1UeContextModification);
            break;

        case timE1_BearerContextModification_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, INIT_TIMER);
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutE1BearerContextModification);
            break;

        case timNG_PDUSessModifyConfirm_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.ng, INIT_TIMER);
            // Will be changed to CauseCpInternalTmOutNgPDUSessModifyConfirm in 22C2
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseNgapMiscUnspecifiedFailure);
            break;

        default:
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid timer msg id(%d) in %s!!!\n", ptUnpack->msgId, __func__);
            return ERROR;
    }

    OFF_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
    OFF_PDUSESS_MODIFY_CONFIRM_FLAG(ptCi->flag.respFlag);

    return GOOD;
}
/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on SNModPrepare_E1Mod state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNModPrepare_E1Mod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    VLIB_NeDcDrb_CollectStat_UpFail(ptCi);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the expiration for E1BearerContextModificationResp on SNModPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNModPrepare_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VLIB_NeDcDrb_CollectStat_UpTimeout(ptCi);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on SNAddPrepareFail_ResourceRel state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepareFail_ResourceRel_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
        }
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the expiration for E1BearerContextModificationResp on SNRelPrepare_E1Mod state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNRelPrepare_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the E1BearerContextModificationFailure received on SNRelPrepare_E1Mod state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNRelPrepare_E1Mod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);
    if (ERROR == status) return ERROR;

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    return GOOD;
}

/****************************************************************************
 * @brief  Handle the E1BearerContextModificationResp received on SNRelPrepare_E1Mod state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNRelPrepare_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
        return ERROR;
    }

    if (IS_NEDC_MN(ptCi))
    {
        SET_STATE_CMD(MrdcF1DCUeRel_cmd);
        ON_CGCFINFO_NEED_FWD_FLAG(ptCi->flag.setupFlag);

        VCCB_clearAllFailedCarrier(&ptCi->caInfo.failures);
        VLIB_ConfigSecondaryCellList(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");
        VCCB_SetMeasGapState(ptCi);

        if (ERROR == VFIL_sendF1_UeContextModificationRequest(ptCi, NULL))
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VFIL_sendF1_UeContextModificationRequest ERROR !!\n");
            return ERROR;
        }

        VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
        VLIB_SetCurStateAll(ptCi, STATE_SNRelPrepare_F1Mod);
    }
    else if (IS_NRDC_MN(ptCi))
    {
        uint8_t targetXN = 0;
        if (ERROR == FnVCCB_SendXNMessage(XN_SNodeReleaseRequest_id, ptCi, &targetXN, nullptr))
        {
            return ERROR;
        }
        VLIB_AttachTimer(&ptCi->timer, ptCi, timXN_SNodeRelease_id);
        VLIB_SetCurStateAll(ptCi, STATE_SNRelPrepare_XnRel);
    }

    return GOOD;
}

/****************************************************************************
 *** @brief  Handle the E1BearerContextModificationResp received on SNUPPS_E1DrbMod state
 *** @ related Feature : NEDC
 *** @param[in/out]   ptCi :      Call information / callInfo_type
 *** @param[in]       ptAll :     Whole received message / message_type
 *** @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 *** @return          status :    GOOD, ERROR / int16_t
 ******************************************************************************/
int16_t VCCM_SNUPPS_E1DrbMod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
        return ERROR;
    }
    if (IS_NRDC_MN(ptCi))
    {
        status = VCCB_PrepareNewMeasConfig(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCB_PrepareNewMeasConfig ERROR!\n");
            return ERROR;
        }

        status = VCCM_SendRRCReconfigSrbX(ptCi);
        if (ERROR == status)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "VCCM_SendRRCReconfigSrbX ERROR !!\n");
            return ERROR;
        }
        if (BREAK == status)
        {
            VCCM_SetStableState_CallSetup(ptCi);
            return GOOD;
        }

        VLIB_SetCurStateAll(ptCi, STATE_SNUPPS_RRCRecCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);
        return GOOD;
    }
    if (IS_NRDC_SN(ptCi))
    {
        if (ptCi->ulPathControl.revertUlPathChange)
        {
            ptCi->ulPathControl.revertUlPathChange = false;
            VCCM_SetStableState_CallSetup(ptCi);
            return GOOD;
        }
        return VCCB_NrdcUlPathControl_SendConfigurationToMN(ptCi);
    }
    else  // for NE-DC legacy implementation
    {
        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        if (ERROR == status) return ERROR;

        VLIB_SetCurStateAll(ptCi, STATE_SNUPPS_XnModReq);
        VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
    }
    return GOOD;
}

/****************************************************************************
 * @brief           Handling Bearer Modification Failure in state SNUPPS_E1DrbMod
 *
 * @param[in/out]   ptCi : Call information / callInfo_type
 * @param[in]       ptAll : Call information / message_type
 * @param[in]       ptUnpack : Call information / msgUnpack_type
 * @return          status : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNUPPS_E1DrbMod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);
    if (ERROR == status) return ERROR;

    return status;
}

int16_t FnVCCB_NeedToSendSNStatusToPeer(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    uint8_t idx;

    for (idx = 0; idx < MAX_DRB_PER_SA_CALL; idx++)
    {
        callDRBInfo_type *pDi = &ptCi->drb[idx];

        if (pDi->usage != CI_use) continue;
        if (pDi->rbInfo.rlcMode != AM_RLC) continue;

        if (pDi->dataFwdInfo.snStatus.needSendFlag == TRUE)
        {
            return TRUE;
        }
    }

    return FALSE;
}

int16_t FnVCCB_SNAddModComplt_E1F1NGMod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (TRUE == FnVCCB_NeedToSendSNStatusToPeer(ptCi))
    {
        uint8_t targetXN = e_Snode_MnToSrcSn;
        status = FnVCCB_SendXNMessage(XN_SNStatusTransfer_id, ptCi, &targetXN, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage(XN_SNStatusTransfer) ERROR !!\n");
    }

    return GOOD;
}

int16_t FnVCCB_SNAddModComplt_E1F1NGMod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    return GOOD;
}

int16_t FnVCCB_NGU_ULTNL_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
        }
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (GOOD == FnVCCB_NeedSnNGuUlTnlUpdate(ptCi))
    {  // Alternative procedure
        SET_STATE_CMD(MrdcXnSNMod_cmd);
        SET_MRDC_PROCEDURE(ptCi, MRDC_PROC_CODE_SnMod, MRDC_PROC_PURPOSE_NGuUlTnlUpdate, MRDC_PROC_INITIATOR_MN);

        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        if (status == ERROR)
        {
            INIT_MRDC_PROCEDURE(ptCi);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendXNMessage ERROR!\n");
        }
        else
        {
            VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
            VLIB_SetCurStateAll(ptCi, STATE_NGU_ULTNL_XnMod);
        }
    }
    else
    {
        FnVCCB_FinalizePduChgforMrdc(ptCi);
        VCCM_SetStableState_CallSetup(ptCi);
    }

    return status;
}

int16_t FnVCCB_SNAddModComplt_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (GOOD == FnVCCB_NeedSnNGuUlTnlUpdate(ptCi))
    {  // Alternative procedure
        SET_STATE_CMD(MrdcXnSNMod_cmd);
        SET_MRDC_PROCEDURE(ptCi, MRDC_PROC_CODE_SnMod, MRDC_PROC_PURPOSE_NGuUlTnlUpdate, MRDC_PROC_INITIATOR_MN);

        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        if (status == ERROR)
        {
            INIT_MRDC_PROCEDURE(ptCi);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendXNMessage ERROR!\n");
        }
        else
        {
            VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
            if (ptCi->state == STATE_SNAddComplt_E1Mod)
            {
                VLIB_SetCurStateAll(ptCi, STATE_SNAddComplt_XnMod);
            }
            else
            {
                VLIB_SetCurStateAll(ptCi, STATE_SNModComplt_XnMod);
            }
        }
    }
    else
    {
        VLIB_FinalizeMrdcProcedure(ptCi);
    }

    return status;
}

int16_t FnVCCB_SNAddModComplt_E1Mod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
        E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF);
    if (status == BREAK)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
    }
    else
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
        VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    }
    return status;
}

int16_t FnVCCB_SNAddModComplt_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNAddModComplt_E1Rel_E1apBearerContextReleaseComplete(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id, ptCi, ptAll, ptUnpack);
    // TBD; VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id) Waiting
    // for other UPs to respond\n");
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (status == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextReleaseComplete_id) ERROR!\n");
        return ERROR;
    }

    if (GOOD == FnVCCB_NeedSnNGuUlTnlUpdate(ptCi))
    {  // Alternative procedure
        SET_STATE_CMD(MrdcXnSNMod_cmd);
        SET_MRDC_PROCEDURE(ptCi, MRDC_PROC_CODE_SnMod, MRDC_PROC_PURPOSE_NGuUlTnlUpdate, MRDC_PROC_INITIATOR_MN);

        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        if (status == ERROR)
        {
            INIT_MRDC_PROCEDURE(ptCi);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendXNMessage ERROR!\n");
        }
        else
        {
            VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
            if (ptCi->state == STATE_SNAddComplt_E1Rel)
            {
                VLIB_SetCurStateAll(ptCi, STATE_SNAddComplt_XnMod);
            }
            else
            {
                VLIB_SetCurStateAll(ptCi, STATE_SNModComplt_XnMod);
            }
        }
    }
    else
    {
        VLIB_FinalizeMrdcProcedure(ptCi);
    }

    return status;
}

int16_t FnVCCB_SNAddModComplt_E1Rel_timE1_BearerContextRelease(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, INIT_TIMER);

    // LOGGING
    logShmIntMsg(ptCi, RX_MSG, BLK_TIME, timE1_BearerContextRelease_id, sizeof(messageHdr_type), (uint8_t *)ptAll);

    // TBD
    // if (GET_E1_REL_COMPLT_RCV_FLAG(ptCi->flag.respFlag))
    //{
    //    MACRO_STAT_PMCC(STAT_XXXX, xxx_UpTimeout, ptCi, 0, 0, 0, 0, 0);
    //}

    return GOOD;
}

int16_t FnVCCB_SNModPrepare_E1Mod1_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    SET_STATE_CMD(MrdcXnSNMod_cmd);

    status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
    if (ERROR == status) return ERROR;

    VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNModPrepare_XnMod);

    return GOOD;
}

int16_t FnVCCB_SNModPrepare_Common_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

/****************************************************************************
 * @brief   Handle the expiration for E1BearerContextModificationResp on SNAddPrepare_E1Mod1 state
 * @ related Feature : NEDC
 * @param[in/out]   ptCi :      Call information / callInfo_type
 * @param[in]       ptAll :     Whole received message / message_type
 * @param[in]       ptUnpack :  Unpacked message / msgUnpack_type
 * @return          status :    GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t VCCM_SNAddPrepare_E1Mod1_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNRelPrepare_E1Setup_E1apBearerContextSetupResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextSetupResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(
        ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextSetupResponse_id) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        return ERROR;
    }

    if (GET_MRDC_PROCEDURE_INITIATOR(ptCi) == MRDC_PROC_INITIATOR_MN && 
            GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) == MRDC_SNREL_OPR_XN_Fallback)
    {
        uint8_t targetXN = e_Snode_MnToSrcSn;
        status = FnVCCB_SendXNMessage(XN_XnUAddressIndication_id, ptCi, &targetXN, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage(XN_XnUAddressIndication) ERROR !!\n");
    }

    return FnVCCB_SNRelPrepare_E1Mod2_or_E1Setup_TriggerF1Procedurs(ptCi);
}

int16_t FnVCCB_SNRelPrepare_E1Setup_E1apBearerContextSetupFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextSetupFailure_id, ptCi, ptAll, ptUnpack);
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        return ERROR;
    }

    return status;
}

int16_t FnVCCB_SNRelPrepare_E1Setup_timE1_BearerContextSetup(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, INIT_TIMER);

    logShmIntMsg(ptCi, RX_MSG, BLK_TIME, timE1_BearerContextSetup_id, sizeof(messageHdr_type), (uint8_t *)ptAll);

    OFF_E1_RESOURCE_FLAG(ptCi->flag.respFlag);

    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutE1BearerSetup);

    return GOOD;
}

int16_t FnVCCB_SNRelPrepare_E1Mod2_or_E1Setup_TriggerF1Procedurs(callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    int16_t status = GOOD;

    // FnVCCB_SNRelPrepare_E1Mod2_E1apBearerContextModificationResponse
    if (ptCi->state == STATE_SNRelPrepare_E1Mod2)
    {
        if (GET_MRDC_PROCEDURE_INITIATOR(ptCi) != MRDC_PROC_INITIATOR_SN && 
            GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) != MRDC_SNREL_OPR_INTERNAL_Fallback)
        {
            OFF_MRDC_RSC_SETUP_FLAG(e1_scgSetup);
        }

        FnVCCB_SetPDUSessAndDrbStatusForF1BearerFallback(ptCi);

        SET_STATE_CMD(MrdcF1DCUeRel_cmd);

        VCCB_clearAllFailedCarrier(&ptCi->caInfo.failures);
        VLIB_ConfigSecondaryCellList(ptCi);

        VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");
    }
    // FnVCCB_SNRelPrepare_E1Setup_E1apBearerContextSetupResponse(
    else if (ptCi->state == STATE_SNRelPrepare_E1Setup)
    {
        SET_STATE_CMD(MrdcF1DCUeRel_cmd);

        VCCB_clearAllFailedCarrier(&ptCi->caInfo.failures);
        VLIB_ConfigSecondaryCellList(ptCi);

        VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");
        FnVCCB_SetPDUSessAndDrbStatusForF1BearerFallback(ptCi);
    }

    status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);

    VLIB_SetCurStateAll(ptCi, STATE_SNRelPrepare_F1Mod);

    return GOOD;
}

int16_t FnVCCB_SNRelPrepare_E1Mod2_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == (status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack)))
    {
        return status;
    }

    if (GET_MRDC_PROCEDURE_INITIATOR(ptCi) == MRDC_PROC_INITIATOR_MN && 
            GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) == MRDC_SNREL_OPR_XN_Fallback)
    {
        uint8_t targetXN = e_Snode_MnToSrcSn;
        status = FnVCCB_SendXNMessage(XN_XnUAddressIndication_id, ptCi, &targetXN, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage(XN_XnUAddressIndication) ERROR !!\n");
    }

    return FnVCCB_SNRelPrepare_E1Mod2_or_E1Setup_TriggerF1Procedurs(ptCi);
}

int16_t FnVCCB_SNRelPrepare_E1Mod2_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (status == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextSetupFailure_id) ERROR !!\n");
        return ERROR;
    }
    return status;
}

int16_t FnVCCB_SNRelPrepare_E1Mod2_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNRelPrepare_E1Mod3_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (GOOD != (status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack)))
    {
        VCCB_RETURN_ON_CALL_ERROR(
            ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextModificationResponse) ERROR !!\n");
        return status;
    }

    if (!IS_NRDC_SNREL_INVOKED(ptCi))
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "Not IS_NRDC_SNREL_INVOKED in %s!\n", __func__);
        return ERROR;
    }

    if (GET_MRDC_PROCEDURE_INITIATOR(ptCi) == MRDC_PROC_INITIATOR_SN || 
            GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) == MRDC_SNREL_OPR_INTERNAL_Fallback)
    {
        OFF_MRDC_RSC_SETUP_FLAG(e1_scgSetup);
    }    

    if (GET_MRDC_PROCEDURE_INITIATOR(ptCi) == MRDC_PROC_INITIATOR_SN)
    {
        status = FnVCCB_SendXNMessage(XN_SNodeReleaseConfirm_id, ptCi, nullptr, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage(XN_SNodeReleaseConfirm_id) ERROR !!\n");

        if (FnVCCB_NeedWaitForSNStatus(ptCi) && (GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) == MRDC_SNREL_OPR_XN_Fallback))
        {
            ON_SN_STATUS_FLAG(ptCi->flag.respFlag);
            VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_SnStatusTransfer_id);
            VCCB_PRINT_CALL_COMMENT(ptCi, "[NRDC]start timXN_SnStatusTransfer!\n");
        }
    }

    SET_STATE_CMD(MrdcUuScgRel_cmd);
    status = FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer(ptCi, FALSE, FALSE, ID_SRB1);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer ERROR !!\n");
    VLIB_AttachTimer(&ptCi->callMultiTimer.rrc, ptCi, timNRRRC_RRCReconfig_id);

    if (FnVCCB_NeedWaitForSNStatus(ptCi) && (GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) == MRDC_SNREL_OPR_XN_Fallback))
    {
        VLIB_SetCurStateAll(ptCi, STATE_SNRelExec_SNStatusTransfer);
    }
    else
    {
        VLIB_SetCurStateAll(ptCi, STATE_SNRelExec);
    }

    return GOOD;
}

int16_t FnVCCB_SNRelPrepare_E1Mod3_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (status == ERROR)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextSetupFailure) ERROR !!\n");
        return ERROR;
    }
    return status;
}

int16_t FnVCCB_SNRelPrepare_E1Mod3_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1F1NGMod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1F1NGMod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == ERROR)
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck %s !!\n",
            status == ERROR ? "ERROR" : "BREAK");
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1F1NGMod_timCommon(callInfo_type *ptCi, message_type *ptAll, msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    // LOGGING
    logShmIntMsg(ptCi, RX_MSG, BLK_TIME, ptUnpack->msgId, sizeof(messageHdr_type), (uint8_t *)ptAll);

    switch (ptUnpack->msgId)
    {
        case timF1_UeContextModification_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.f1, INIT_TIMER);
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutF1UeContextModification);
            break;

        case timE1_BearerContextModification_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, INIT_TIMER);
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutE1BearerContextModification);
            break;

        case timNG_PDUSessModifyConfirm_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.ng, INIT_TIMER);
            // Will be changed to CauseCpInternalTmOutNgPDUSessModifyConfirm in 22C2
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseNgapMiscUnspecifiedFailure);
            break;

        case timXN_SnStatusTransfer_id:
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.xn, INIT_TIMER);
            MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseXnapMiscUnspecified);
            break;

        default:
            VCCB_PRINT_CALL_ERROR(ptCi, "Invalid timeOut msg id(%d) in %s!\n", ptUnpack->msgId, __func__);
            return ERROR;
    }

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
    OFF_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
    OFF_PDUSESS_MODIFY_CONFIRM_FLAG(ptCi->flag.respFlag);
    OFF_SN_STATUS_FLAG(ptCi->flag.respFlag);

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (GET_MRDC_PROCEDURE_SNREL_OPRTYPE(ptCi) != MRDC_SNREL_OPR_INTERNAL_Fallback)
    {
        if (FnVCCB_PrepareXNAP_UeContextRelease_ForSNode(ptCi) == GOOD)
        {
            uint8_t targetXN = e_Snode_MnToSrcSn;
            if (ERROR == FnVCCB_SendXNMessage(XN_UEContextRelease_id, ptCi, &targetXN, nullptr))
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendXNMessage(XN_UEContextRelease_id) ERROR !!\n");
            }
        }
    }

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1Mod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNRelComplt_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_UEAssiInfo_E1F1Mod_E1BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK !!\n");
        }
        else
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR !!\n");
        }
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_UEAssiInfo_E1F1Mod_E1BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK !!\n");
        }
        else
        {
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR !!\n");
        }
        return status;
    }

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_E1Mod1_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "XN_SNodeModificationRequest ERROR!\n");

    VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNModFbkPrepare_XnMod);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_Common_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_Common_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_E1Setup_E1apBearerContextSetupResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextSetupResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(
        ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextSetupResponse_id) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextSetupResponse_id) ERROR!\n");
        return ERROR;
    }

    VCCB_clearAllFailedCarrier(&ptCi->caInfo.failures);
    VLIB_ConfigSecondaryCellList(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, VCCB_PrepareNewMeasConfig(ptCi), "VCCB_PrepareNewMeasConfig ERROR!\n");

    FnVCCB_SetPDUSessAndDrbStatusForF1BearerFallback(ptCi);
    status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");
    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);

    VLIB_SetCurStateAll(ptCi, STATE_SNModFbkPrepare_F1Mod);

    return status;
}

int16_t FnVCCB_SNModFbkPrepare_E1Setup_E1apBearerContextSetupFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextSetupFailure_id, ptCi, ptAll, ptUnpack);
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextSetupFailure) ERROR!\n");

    return status;
}

int16_t FnVCCB_SNModFbkPrepare_E1Setup_timE1_BearerContextSetup(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, INIT_TIMER);

    logShmIntMsg(ptCi, RX_MSG, BLK_TIME, timE1_BearerContextSetup_id, sizeof(messageHdr_type), (uint8_t *)ptAll);

    OFF_E1_RESOURCE_FLAG(ptCi->flag.respFlag);

    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, CauseCpInternalTmOutE1BearerSetup);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_E1Mod2_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status, "FnVCCB_StoreE1Message(E1ModRsp) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id) ERROR!\n");
        return status;
    }

    FnVCCB_SetPDUSessAndDrbStatusForF1BearerFallback(ptCi);

    status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest ERROR!\n");

    VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_SNModFbkPrepare_F1Mod);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_E1Mod2_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id) ERROR!\n");

    return status;
}

int16_t FnVCCB_SNModFbkPrepare_E1Mod3_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    uint8_t xnFlag = 0;
    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack);
    VCCB_RETURN_ON_CALL_NBREAK(ptCi, status,
        "FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id) Waiting for other UPs to respond\n");

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    if (ERROR == status)
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1Message(E1_BearerContextModificationResponse_id) ERROR!\n");
        return ERROR;
    }

    if (FnVCCB_NeedDataFwdTnlTobeSentToSN(ptCi))
    {
        uint8_t targetXN = e_Snode_MnToSrcSn;
        status = FnVCCB_SendXNMessage(XN_XnUAddressIndication_id, ptCi, &targetXN, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendXNMessage(XN_XnUAddressIndication) ERROR !!\n");
    }
    else
    {
        VCCB_PRINT_CALL_INFORMATIVE(ptCi, "No data forward data Info!\n");
    }

    status = FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer(ptCi, FALSE, FALSE, ID_SRB1);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer ERROR !!\n");
    VLIB_AttachTimer(&ptCi->callMultiTimer.rrc, ptCi, timNRRRC_RRCReconfig_id);

    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "start parallel processing <rrc(1), xn(%d)>!\n", xnFlag);

    VLIB_SetCurStateAll(ptCi, STATE_SNModFbkExec);

    return GOOD;
}

int16_t FnVCCB_SNModFbkPrepare_E1Mod3_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;

    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack);
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "FnVCCB_StoreE1Message(E1_BearerContextModificationFailure_id) ERROR!\n");

    return status;
}

int16_t FnVCCM_SNE1UlPath_E1ModReq_E1BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    status = VCCM_SendRRCReconfigSrbX(ptCi);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_SendRRCReconfigSrbX ERROR!\n");

    VLIB_SetCurStateAll(ptCi, STATE_SNE1UlPath_RRCRecCmpl);
    VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);

    return GOOD;
}

int16_t FnVCCM_SNE1UlPath_E1ModReq_E1BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCM_SNE1UlPath_E1ModReq_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

int16_t FnVCCM_SNE1UlPathFail_E1ModReq_E1BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);
    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return GOOD;
}

int16_t FnVCCB_SNModFbkComplt_E1F1NGMod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = FnVCCB_MrdcParallelMsgHandling_Common_E1apBearerContextModificationResponse(ptCi, ptAll, ptUnpack);
    if (status != GOOD) return status;

    if (!FnVCCB_IsAnySpecificBearerAndReqBearer(ptCi, MN_Term_SPLIT_Bearer, MN_Term_SPLIT_Bearer) &&
        !FnVCCB_IsAnySpecificBearerAndReqBearer(
            ptCi, MN_Term_SPLIT_Bearer, INIT_BEARER_TYPE))  // regards as the current mn-split even if abnormal case
    {
        OFF_MRDC_RSC_SETUP_FLAG(e1_scgSetup);
        VCCB_PRINT_CALL_DATA(ptCi, "Not anymore MN-Term-SPLIT Bearers !\n");
    }
    return GOOD;
}

int16_t FnVCCB_SNModFbkComplt_E1F1NGMod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    return FnVCCB_MrdcParallelMsgHandling_Common_E1apBearerContextModificationFailure(ptCi, ptAll, ptUnpack);
}

int16_t FnVCCB_SNModFbkComplt_E1Mod_E1apBearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);
    VALID_E_POINT_CHECK_ERROR(ptAll);
    VALID_E_POINT_CHECK_ERROR(ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationResponse_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    if (IS_ON_MRDC_NEED_MODIFY_FLAG(addiNgUlUpInfo))
    {
        SET_STATE_CMD(MrdcXnSNMod_cmd);
        SET_MRDC_PROCEDURE(ptCi, MRDC_PROC_CODE_SnMod, MRDC_PROC_PURPOSE_NGuUlTnlUpdate, MRDC_PROC_INITIATOR_MN);

        status = FnVCCB_SendXNMessage(XN_SNodeModificationRequest_id, ptCi, nullptr, nullptr);
        if (status == ERROR)
        {
            INIT_MRDC_PROCEDURE(ptCi);
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendXNMessage ERROR!\n");
        }
        else
        {
            VLIB_AttachTimer(&ptCi->callMultiTimer.xn, ptCi, timXN_DcPrepare_id);
            VLIB_SetCurStateAll(ptCi, STATE_SNModFbkComplt_XnMod);
        }

        OFF_MRDC_NEED_MODIFY_FLAG(addiNgUlUpInfo);
    }
    else
    {
        VLIB_FinalizeMrdcProcedure(ptCi);
    }

    return status;
}

int16_t FnVCCB_SNModFbkComplt_E1Mod_E1apBearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    int16_t status = GOOD;

    if (GOOD != (status = FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck(
                     E1_BearerContextModificationFailure_id, ptCi, ptAll, ptUnpack, VCCB_FLAG_OFF)))
    {
        if (status == BREAK)
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck BREAK!\n");
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_StoreE1apModificationRespWithWrongDeliveryCheck ERROR!\n");
            OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
            VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
        }
        return status;
    }
    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);

    return status;
}

int16_t FnVCCB_SNModFbkComplt_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

// SVR22B SCG Failure Handling for NR-DC

// 1. Suspend Procedure
// 1-1. @SN
/****************************************************************
 *  Function     : FnVCCB_SN_SCGSuspend_E1Mod_BearerContextModificationResponse
 *  Description  : Normal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_SN_SCGSuspend_E1Mod_BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    ptCi->cause.xnCause = CauseXnapRadioNetworkLayerRadioConnectionWithUeLost;

    // SVR24B FGR-CC5203
    if (NrDcScgFailure_MnInit_Suspend == ptCi->nrdcScgFailureInfo.event)
    {
        int16_t status = FnVCCB_SendXNMessage(XN_SNodeModificationRequestAcknowledge_id, ptCi, nullptr, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status,
            "FnVCCB_SendXNMessage(XN_SNodeModificationRequestAcknowledge) ERROR!\n");

        // If SN RRC Reconfig is included in SNode Modification Request Acknowledge,
        if (FnVCCB_NeedToRecvXnSNodeReconfigurationComplete(ptCi) == true)
        {
            VLIB_SetCurStateAll(ptCi, STATE_SN_SCGSuspend_XnReconfigCmpl);
            VLIB_AttachTimer(&ptCi->timer, ptCi, timXN_SNodeReconfigurationComplete_id);
        }
        else
        {
            // suspend procedure completed in SN!!
            FnVCCB_FinalizeSCGSuspend_MRDC(ptCi);
        }
    }
    else if (NrDcScgFailure_SnInit_Suspend == ptCi->nrdcScgFailureInfo.event)
    {
        int16_t status = FnVCCB_SendXNMessage(XN_SNodeModificationRequired_id, ptCi, nullptr, nullptr);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status,
            "FnVCCB_SendXNMessage(XN_SNodeModificationRequired) ERROR!\n");

        VLIB_SetCurStateAll(ptCi, STATE_SN_SCGSuspend_XnReconfigCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timXN_SNodeReconfigurationConfirm_id);
    }

    // start suspend timer
    VLIB_AttachTimer(&ptCi->scgSuspendResumeTimer, ptCi, timInternalNrdcScgSuspend_id);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_SN_SCGSuspend_E1Mod_BearerContextModificationFailure
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_SN_SCGSuspend_E1Mod_BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    // FGR-CC5203: FR230
    ptCi->cause.xnCause = CauseXnapRadioNetworkLayerRadioConnectionWithUeLost;
    ptCi->nrdcScgFailureInfo.failAction = NrDcScgFailureSnRelease;
    ptCi->mrdcInfo.sNodeRelCause = MrdcRelCause_SCGFailure;

    // FGR-CC5203: FR230
    CONVERT_AND_SAVE_NRDC_CTX_REL(ptCi, XN_SNodeReleaseRequired_id);

    // CALL FAULT
    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, ptCi->cause.e1Cause);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_SN_SCGSuspend_E1Mod_timE1_BearerContextModification
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_SN_SCGSuspend_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

// 1-2. @MN
/****************************************************************
 *  Function     : FnVCCB_MN_SCGSuspend_E1Mod_BearerContextModificationResponse
 *  Description  : Normal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGSuspend_E1Mod_BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    int16_t status = GOOD;

    if (DBG->flag.mrdcScgFailureSuspendResume == TRUE)
    {
        // SVR22B FGR-CC5203: Deconfigure A4 for SCG resume and Update MeasConfig
        ptCi->nrdcScgFailureInfo.A4Configured = VCCB_FLAG_ON;
        status = VCCB_PrepareNewMeasConfig(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCB_PrepareNewMeasConfig ERROR!\n");
    }

    status = FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer(ptCi, FALSE, FALSE, ID_SRB1);

    if (status == GOOD)
    {
        VLIB_SetCurStateAll(ptCi, STATE_MN_SCGSuspend_RRCReconfigCmpl);
        VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);
    }
    else
    {
        VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer fails !!\n");

        // MN-Init SN-Release
        status = VCCM_ProcSCGFailWithSnRelease(ptCi, MrdcRelCause_SCGFailure);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCM_ProcSCGFailWithSnRelease ERROR!\n");
    }

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGSuspend_E1Mod_BearerContextModificationFailure
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGSuspend_E1Mod_BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    // CALL FAULT
    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, ptCi->cause.e1Cause);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGSuspend_E1Mod_timE1_BearerContextModification
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGSuspend_E1Mod_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

// 2. Resume Procedure
// 2-1. @SN
// Use Mobility State Machine...

// 2-2. @MN
/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModULConfig_BearerContextModificationResponse
 *  Description  : Normal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModULConfig_BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    if (DBG->flag.mrdcScgFailureSuspendResumeNoF1Proc == FALSE || IS_ON_MRDC_XN_RCVINDICATOR_FLAG(selectedBC))
    {
        VCCB_clearAllFailedCarrier(&ptCi->caInfo.failures);
        VLIB_ConfigSecondaryCellList(ptCi);

        int16_t status = VCCB_PrepareNewMeasConfig(ptCi);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VCCB_PrepareNewMeasConfig! ERROR!\n");

        status = VFIL_sendF1_UeContextModificationRequest(ptCi, NULL);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VFIL_sendF1_UeContextModificationRequest! ERROR!\n");

        ON_F1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

        VLIB_SetCurStateAll(ptCi, STATE_MN_SCGResume_F1Mod);
        VLIB_AttachTimer(&ptCi->callMultiTimer.f1, ptCi, timF1_UeContextModification_id);
    }
    else
    {
        int16_t status = FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer(ptCi, FALSE, FALSE, ID_SRB1);

        if (status == GOOD)
        {
            VLIB_SetCurStateAll(ptCi, STATE_MN_SCGResume_RRCReconfigCmpl);
            VLIB_AttachTimer(&ptCi->timer, ptCi, timNRRRC_RRCReconfig_id);
        }
        else
        {
            VCCB_PRINT_CALL_ERROR(ptCi, "FnVCCB_SendNRRRC_RRCReconfiguration_F1DLRRCMsgTransfer fails !!\n");

            // Send Release Request to SN
            ptCi->cause.xnCause = CauseXnapRadioNetworkLayerRadioConnectionWithUeLost;
            ptCi->mrdcInfo.sNodeRelCause =
                MrdcRelCause_OtherReason;  // TODO - need to change VCCM_TriggerMgNBInitSNReleaseProcedure function

            if (VCCM_TriggerMgNBInitSNReleaseProcedure(ptCi) == ERROR)
            {
                VCCB_PRINT_CALL_ERROR(ptCi, "VCCM_TriggerMgNBInitSNReleaseProcedure ERROR !\n");
                return ERROR;
            }
        }
    }

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModULConfig_BearerContextModificationFailure
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModULConfig_BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    // CALL FAULT
    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, ptCi->cause.e1Cause);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModULConfig_timE1_BearerContextModification
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModULConfig_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModResume_BearerContextModificationResponse
 *  Description  : Normal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModResume_BearerContextModificationResponse(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    // resume procedure completed in MN!!
    FnVCCB_FinalizeSCGResume_MRDC(ptCi, ptCi->servCellNum);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModResume_BearerContextModificationFailure
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModResume_BearerContextModificationFailure(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    VCCB_CLEAR_TIMER(&ptCi->callMultiTimer.e1, CANCEL_TIMER);
    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    // CALL FAULT
    MACRO_CALL_FAULT_SAVE(ptUnpack->msgId, ptCi->cause.e1Cause);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModResume_timE1_BearerContextModification
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_MN_SCGResume_E1ModResume_timE1_BearerContextModification(
    callInfo_type *ptCi, const message_type *ptAll, const msgUnpack_type *ptUnpack)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINTER_CHECK_FOR_STATE_FUNC_ERROR(ptCi, ptAll, ptUnpack);

    OFF_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);

    FnVCCB_Common_Multi_timE1_BearerContextModification(ptCi, (message_type *)ptAll, (msgUnpack_type *)ptUnpack);

    return GOOD;
}

/****************************************************************
 *  Function     : FnVCCB_MN_SCGResume_E1ModResume_timE1_BearerContextModification
 *  Description  : Abnormal Case
 *  Procedures defined :
 *  Procedures called  :
 *  Procedure calling  :
 *  Input  : callInfo_type*, message_type*, msgUnpack_type*
 *  Output : int16_t
 *  Statistics Pegged :
 ***************************************************************/
int16_t FnVCCB_ALL_timE1apBearerContextInactivityNotification(
    callInfo_type *ptCi)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    if (IS_SUPPORT_RRC_INACTIVE(ptCi) &&
        ptCi->ueInfo.rrcResumeInfo.actionInSNRelease == SN_REL_IN_RRC_INACTIVE_DECIDED)
    {
        // initialize the variable actionInSNRelease for delivering the original inactivity-timer value
        ptCi->ueInfo.rrcResumeInfo.actionInSNRelease = SN_REL_IN_RRC_INACTIVE_INITIAL_STATE;

        int16_t status = VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification);
        VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendE1SetupModRelRequest (Modification) ERROR!\n");
        VLIB_AttachTimer(&ptCi->callMultiTimer.e1, ptCi, timE1_BearerContextModification_id);
        ON_E1_CTXT_MODRSP_FLAG(ptCi->flag.respFlag);
    }

    return GOOD;
}

/****************************************************************************
 * @brief           Handle Dequeue Action for internalE1InactNotiDuringSNRelease expiry
 *
 * @param[in/out]   ptCi    : Call information / callInfo_type
 * @param[in]       qAction : no use / enqueuedAction
 * @return          status  : GOOD, ERROR / int16_t
 ****************************************************************************/
int16_t FnVCCB_HandleDequeuedInternalE1InactNotiDuringSNRelease(callInfo_type *ptCi, enqueuedAction)
{
    AUTO_TRACE_FUNC;
    VALID_E_POINT_CHECK_ERROR(ptCi);

    VCCB_PRINT_CALL_INFORMATIVE(ptCi, "Start to dequeue action for internalE1InactNotiDuringSNRelease expiry\n");

    int16_t status = VEIL_sendE1SetupModRelRequest(ptCi, E1Procedure_BearerContextModification);
    VCCB_RETURN_ON_CALL_ERROR(ptCi, status, "VEIL_sendE1SetupModRelRequest(E1Procedure_BearerContextModification) ERROR!\n");

    VLIB_AttachTimer(&ptCi->callMultiTimer.e1, ptCi, timE1_BearerContextModification_id);
    VLIB_SetCurStateAll(ptCi, STATE_Inact_SNRelCmpl);   // new state

    ptCi->ueInfo.rrcResumeInfo.preventSNAddFlag = FALSE;

    return GOOD;
}
/* EOF */
