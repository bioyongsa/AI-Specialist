std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgSaByUe(
    callInfo_type *ptCi,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority,
    int8_t order)
{
    std::vector<std::pair<int8_t, int8_t>> fallback_pairs;
    const std::vector<std::pair<int8_t, int8_t>>* pPairs = &input_pairs;

    if (input_pairs.empty())
    {
        if (ptCi == nullptr || ptCi->gnbNum >= MAX_GNB_PER_SYS)
            return {-1, -1};

        uint8_t uePlmnIndex;
        if (GOOD != VLIB_getMultipleGnbPlmnIndexInfo(ptCi, &uePlmnIndex, ptCi->ueInfo.servedPlmn.longId))
            return {-1, -1};

        callSNSSAI_type sNssaiList[MAX_PDU_SESS_PER_CALL];
        uint8_t fiveQiList[MAX_QOS_FLOW_PER_CALL];
        uint8_t arpList[MAX_QOS_FLOW_PER_CALL];
        uint8_t numSNssai = 0, numFiveQi = 0, numArp = 0;

        for (uint8_t pduIdx = 0; pduIdx < MAX_PDU_SESS_PER_CALL; pduIdx++)
        {
            callPDUSessInfo_type *ptPi = &ptCi->pduSess[pduIdx];
            if (ptPi->usage != CI_use) continue;

            bool sNssaiDup = false;
            for (uint8_t s = 0; s < numSNssai; s++)
            {
                if (sNssaiList[s].sstSd == ptPi->sNssai.sstSd) { sNssaiDup = true; break; }
            }
            if (!sNssaiDup && numSNssai < MAX_PDU_SESS_PER_CALL)
                sNssaiList[numSNssai++] = ptPi->sNssai;

            for (uint8_t qIdx = 0; qIdx < MAX_QOS_FLOW_PER_PDU_SESS; qIdx++)
            {
                uint16_t qosFlowIdx = ptPi->qosFlowIndex[qIdx];
                if (qosFlowIdx >= MAX_QOS_FLOW_PER_CALL) continue;

                qosFlowInfo_type *ptQi = &ptCi->qosFlow[qosFlowIdx];
                if (ptQi->usage != CI_use) continue;

                uint8_t fiveQi;
                VLIB_getFiveQi(ptQi, &fiveQi);
                if (fiveQi != 0xFF && numFiveQi < MAX_QOS_FLOW_PER_CALL)
                {
                    bool dup = false;
                    for (uint8_t f = 0; f < numFiveQi; f++) { if (fiveQiList[f] == fiveQi) { dup = true; break; } }
                    if (!dup) fiveQiList[numFiveQi++] = fiveQi;
                }

                uint8_t arpVal = (uint8_t)ptQi->qosParam.allocAndRetent.priorityLevel;
                if (arpVal < MAX_ARP_LEVEL && numArp < MAX_QOS_FLOW_PER_CALL)
                {
                    bool dup = false;
                    for (uint8_t a = 0; a < numArp; a++) { if (arpList[a] == arpVal) { dup = true; break; } }
                    if (!dup) arpList[numArp++] = arpVal;
                }
            }
        }

        uint16_t rfspVal = ptCi->ueInfo.rfsp.indexToRFSP;
        bool rfspPresent = (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT);
        bool isRedcapUe = IS_REDCAP_UE(ptCi);

        VccbGnbInfo_type *pGnb = &GNBINFO[ptCi->gnbNum];
        for (uint8_t idx = 0; idx < MAX_SVC_SA_ENTRIES; idx++)
        {
            svcCfgSa_type *pEntry = &pGnb->cfg.svcCfgSaEntries[idx];
            if (pEntry->service_sa_usage != SVC_CFG_USAGE_USE) continue;
            if (pEntry->plmn_index != uePlmnIndex) continue;

            if (FnVCCB_IsSvcCfgSaProfileMatch(pEntry, sNssaiList, numSNssai,
                fiveQiList, numFiveQi, arpList, numArp, rfspVal, rfspPresent, isRedcapUe))
            {
                fallback_pairs.push_back({(int8_t)pEntry->plmn_index, (int8_t)pEntry->service_sa_index});
            }
        }

        pPairs = &fallback_pairs;
    }

    std::pair<int8_t, int8_t> best = {-1, -1};
    bool found = false;

    for (const auto& p : *pPairs)
    {
        if (order == 0)
        {
            if (p.second < priority)
            {
                if (!found || p.second < best.second ||
                    (p.second == best.second && p.first < best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
        else
        {
            if (p.second > priority)
            {
                if (!found || p.second > best.second ||
                    (p.second == best.second && p.first > best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
    }

    return best;
}

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgSaByDrb(
    callInfo_type *ptCi,
    callDRBInfo_type *pDrb,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority,
    int8_t order)
{
    std::vector<std::pair<int8_t, int8_t>> fallback_pairs;
    const std::vector<std::pair<int8_t, int8_t>>* pPairs = &input_pairs;

    if (input_pairs.empty())
    {
        if (ptCi == nullptr || pDrb == nullptr || ptCi->gnbNum >= MAX_GNB_PER_SYS)
            return {-1, -1};

        uint8_t uePlmnIndex;
        if (GOOD != VLIB_getMultipleGnbPlmnIndexInfo(ptCi, &uePlmnIndex, ptCi->ueInfo.servedPlmn.longId))
            return {-1, -1};

        uint8_t pduIdx = pDrb->pduIdxInCall;
        if (pduIdx >= MAX_PDU_SESS_PER_CALL)
            return {-1, -1};

        callSNSSAI_type sNssaiList[1];
        sNssaiList[0] = ptCi->pduSess[pduIdx].sNssai;
        uint8_t numSNssai = 1;

        uint8_t fiveQiList[MAX_QOS_FLOW_PER_DRB];
        uint8_t arpList[MAX_QOS_FLOW_PER_DRB];
        uint8_t numFiveQi = 0, numArp = 0;

        for (uint8_t drbQosIdx = 0; drbQosIdx < MAX_QOS_FLOW_PER_DRB; drbQosIdx++)
        {
            uint16_t qosFlowIdx = pDrb->qosFlowIndex[drbQosIdx];
            if (qosFlowIdx >= MAX_QOS_FLOW_PER_CALL) continue;

            qosFlowInfo_type *ptQi = &ptCi->qosFlow[qosFlowIdx];
            if (ptQi->usage != CI_use) continue;

            uint8_t fiveQi;
            VLIB_getFiveQi(ptQi, &fiveQi);
            if (fiveQi != 0xFF && numFiveQi < MAX_QOS_FLOW_PER_DRB)
            {
                bool dup = false;
                for (uint8_t f = 0; f < numFiveQi; f++) { if (fiveQiList[f] == fiveQi) { dup = true; break; } }
                if (!dup) fiveQiList[numFiveQi++] = fiveQi;
            }

            uint8_t arpVal = (uint8_t)ptQi->qosParam.allocAndRetent.priorityLevel;
            if (arpVal < MAX_ARP_LEVEL && numArp < MAX_QOS_FLOW_PER_DRB)
            {
                bool dup = false;
                for (uint8_t a = 0; a < numArp; a++) { if (arpList[a] == arpVal) { dup = true; break; } }
                if (!dup) arpList[numArp++] = arpVal;
            }
        }

        uint16_t rfspVal = ptCi->ueInfo.rfsp.indexToRFSP;
        bool rfspPresent = (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT);
        bool isRedcapUe = IS_REDCAP_UE(ptCi);

        VccbGnbInfo_type *pGnb = &GNBINFO[ptCi->gnbNum];
        for (uint8_t idx = 0; idx < MAX_SVC_SA_ENTRIES; idx++)
        {
            svcCfgSa_type *pEntry = &pGnb->cfg.svcCfgSaEntries[idx];
            if (pEntry->service_sa_usage != SVC_CFG_USAGE_USE) continue;
            if (pEntry->plmn_index != uePlmnIndex) continue;

            if (FnVCCB_IsSvcCfgSaProfileMatch(pEntry, sNssaiList, numSNssai,
                fiveQiList, numFiveQi, arpList, numArp, rfspVal, rfspPresent, isRedcapUe))
            {
                fallback_pairs.push_back({(int8_t)pEntry->plmn_index, (int8_t)pEntry->service_sa_index});
            }
        }

        pPairs = &fallback_pairs;
    }

    std::pair<int8_t, int8_t> best = {-1, -1};
    bool found = false;

    for (const auto& p : *pPairs)
    {
        if (order == 0)
        {
            if (p.second < priority)
            {
                if (!found || p.second < best.second ||
                    (p.second == best.second && p.first < best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
        else
        {
            if (p.second > priority)
            {
                if (!found || p.second > best.second ||
                    (p.second == best.second && p.first > best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
    }

    return best;
}

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgEndcByUe(
    callInfo_type *ptCi,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority,
    int8_t order)
{
    std::vector<std::pair<int8_t, int8_t>> fallback_pairs;
    const std::vector<std::pair<int8_t, int8_t>>* pPairs = &input_pairs;

    if (input_pairs.empty())
    {
        if (ptCi == nullptr || ptCi->gnbNum >= MAX_GNB_PER_SYS)
            return {-1, -1};

        uint8_t uePlmnIndex;
        if (GOOD != VLIB_getMultipleGnbPlmnIndexInfo(ptCi, &uePlmnIndex, ptCi->ueInfo.servedPlmn.longId))
            return {-1, -1};

        uint16_t spidVal = ptCi->ueInfo.rfsp.subscriberProfile;
        bool spidPresent = (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT);

        uint8_t qciList[MAX_ERAB_PER_CALL];
        uint8_t arpList[MAX_ERAB_PER_CALL];
        uint8_t numQci = 0, numArp = 0;

        for (uint8_t erabIdx = 0; erabIdx < MAX_ERAB_PER_CALL; erabIdx++)
        {
            callERABInfo_type *ptRi = &ptCi->erabInfo[erabIdx];
            if (ptRi->usage != CI_use) continue;

            uint8_t qci = (uint8_t)ptRi->erabQos.qciValue.QCI;
            if (numQci < MAX_ERAB_PER_CALL)
            {
                bool dup = false;
                for (uint8_t q = 0; q < numQci; q++) { if (qciList[q] == qci) { dup = true; break; } }
                if (!dup) qciList[numQci++] = qci;
            }

            uint8_t arpVal = (uint8_t)ptRi->erabQos.aRP.priority;
            if (arpVal < MAX_ARP_LEVEL && numArp < MAX_ERAB_PER_CALL)
            {
                bool dup = false;
                for (uint8_t a = 0; a < numArp; a++) { if (arpList[a] == arpVal) { dup = true; break; } }
                if (!dup) arpList[numArp++] = arpVal;
            }
        }

        VccbGnbInfo_type *pGnb = &GNBINFO[ptCi->gnbNum];
        for (uint8_t idx = 0; idx < MAX_SVC_ENDC_ENTRIES; idx++)
        {
            svcCfgEndc_type *pEntry = &pGnb->cfg.svcCfgEndcEntries[idx];
            if (pEntry->service_endc_usage != SVC_CFG_USAGE_USE) continue;
            if (pEntry->plmn_index != uePlmnIndex) continue;

            if (FnVCCB_IsSvcCfgEndcProfileMatch(pEntry, spidVal, spidPresent,
                qciList, numQci, arpList, numArp))
            {
                fallback_pairs.push_back({(int8_t)pEntry->plmn_index, (int8_t)pEntry->service_endc_index});
            }
        }

        pPairs = &fallback_pairs;
    }

    std::pair<int8_t, int8_t> best = {-1, -1};
    bool found = false;

    for (const auto& p : *pPairs)
    {
        if (order == 0)
        {
            if (p.second < priority)
            {
                if (!found || p.second < best.second ||
                    (p.second == best.second && p.first < best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
        else
        {
            if (p.second > priority)
            {
                if (!found || p.second > best.second ||
                    (p.second == best.second && p.first > best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
    }

    return best;
}

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgEndcByDrb(
    callInfo_type *ptCi,
    callERABInfo_type *pErab,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority,
    int8_t order)
{
    std::vector<std::pair<int8_t, int8_t>> fallback_pairs;
    const std::vector<std::pair<int8_t, int8_t>>* pPairs = &input_pairs;

    if (input_pairs.empty())
    {
        if (ptCi == nullptr || pErab == nullptr || ptCi->gnbNum >= MAX_GNB_PER_SYS)
            return {-1, -1};

        uint8_t uePlmnIndex;
        if (GOOD != VLIB_getMultipleGnbPlmnIndexInfo(ptCi, &uePlmnIndex, ptCi->ueInfo.servedPlmn.longId))
            return {-1, -1};

        uint16_t spidVal = ptCi->ueInfo.rfsp.subscriberProfile;
        bool spidPresent = (ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT);

        uint8_t qciList[1];
        qciList[0] = (uint8_t)pErab->erabQos.qciValue.QCI;
        uint8_t numQci = 1;

        uint8_t arpList[1];
        arpList[0] = (uint8_t)pErab->erabQos.aRP.priority;
        uint8_t numArp = (arpList[0] < MAX_ARP_LEVEL) ? 1 : 0;

        VccbGnbInfo_type *pGnb = &GNBINFO[ptCi->gnbNum];
        for (uint8_t idx = 0; idx < MAX_SVC_ENDC_ENTRIES; idx++)
        {
            svcCfgEndc_type *pEntry = &pGnb->cfg.svcCfgEndcEntries[idx];
            if (pEntry->service_endc_usage != SVC_CFG_USAGE_USE) continue;
            if (pEntry->plmn_index != uePlmnIndex) continue;

            if (FnVCCB_IsSvcCfgEndcProfileMatch(pEntry, spidVal, spidPresent,
                qciList, numQci, arpList, numArp))
            {
                fallback_pairs.push_back({(int8_t)pEntry->plmn_index, (int8_t)pEntry->service_endc_index});
            }
        }

        pPairs = &fallback_pairs;
    }

    std::pair<int8_t, int8_t> best = {-1, -1};
    bool found = false;

    for (const auto& p : *pPairs)
    {
        if (order == 0)
        {
            if (p.second < priority)
            {
                if (!found || p.second < best.second ||
                    (p.second == best.second && p.first < best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
        else
        {
            if (p.second > priority)
            {
                if (!found || p.second > best.second ||
                    (p.second == best.second && p.first > best.first))
                {
                    best = p;
                    found = true;
                }
            }
        }
    }

    return best;
}
