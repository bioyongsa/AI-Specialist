# FR10.CACP.SR1.HD1.VCCB.DD2 — Service Configuration Matching (SVR26B)

VCCB exposes API functions that determine the **best** service index (with PLMN index) by matching stored profiles against runtime data and applying **priority/order–based** selection. Each service-configuration entry carries a pre-computed `priority` value derived from legacy service-group mapping rules via `FnVCCB_UpdateSvcCfgPriorities`.

| FR aspect | Detailed Design |
|---|---|
| QoS, Slice, RFSP, UE-type; UE-level or DRB-level | **SA:** `FnVCCB_GetServiceCfgSaByUe` (UE-level) / `FnVCCB_GetServiceCfgSaByDrb` (DRB-level) |
| Distinguish SA and ENDC | **ENDC:** `FnVCCB_GetServiceCfgEndcByUe` (UE-level) / `FnVCCB_GetServiceCfgEndcByDrb` (DRB-level) |
| Priority initialization | `FnVCCB_UpdateSvcCfgPriorities` (called after configuration load) |

The service-configuration container defines operator SA and ENDC service profiles.
VCCB loads and updates them per FR10.CACP.SR1.HD1.VCCB.DD1, initializes each entry's `priority` via `FnVCCB_UpdateSvcCfgPriorities`, then resolves the **single best** service index per FR10.CACP.SR1.HD1.VCCB.DD2 when processing UEs or DRB-level context.

---

## Design Principles

| Principle | Description |
|---|---|
| **SA vs ENDC** | Separate entry lists in configuration, separate public APIs, separate profile match rules. |
| **Wildcard** | If a profile dimension (slice / 5QI / ARP / RFSP / UE-type for SA; SPID / QCI / ARP for ENDC) has no configured values (count 0), that dimension always matches. |
| **Conjunction across dimensions** | All non-wildcard dimensions must match (**AND**); within one dimension, any configured value may match any collected runtime value (**OR**). |
| **Usage** | Only entries with `usage == SVC_CFG_USAGE_USE` participate in matching for that stack (SA or ENDC). |
| **Unified dispatcher** | All 4 public APIs delegate to a single internal function `FnVCCB_GetServiceCfgBest` parameterized by `SvcCfgType` (SA / ENDC) and `SvcCfgLevel` (UE / DRB). |
| **Pre-computed priority** | Each entry's `priority` field is populated at configuration-load time by `FnVCCB_UpdateSvcCfgPriorities`, which maps the entry's profile to legacy service-group priority via `FnVCCB_FindSvcGrpIdByMappingRule`. |
| **Best-of selection** | Instead of returning all matches, the API returns the single best `{plmn_index, service_index}` pair filtered and selected by the entry's **`priority`** field, controlled by the caller's `priority` threshold and `order` direction. |
| **Input-pair bypass** | If the caller supplies a non-empty `input_pairs` vector, runtime collection and profile matching are skipped; only the given pairs are validated against configuration and then subjected to priority/order selection. |

### Matching Logic (Dimension Examples)

**Within one dimension — OR logic:**
If multiple values are set in one dimension, **any one** matching is sufficient.
Example: profile 5QI = {9, 8, 5} and UE 5QI = 8 → 5QI dimension is satisfied.

**Between dimensions — AND logic:**
All dimensions must independently be satisfied.
Example: Slice ✓, 5QI ✓, ARP ✗ → overall result is **no match**.

**Wildcard behavior:**
A dimension with count 0 is treated as "always matching"; remaining dimensions still follow AND logic.

---

## Comparison to Legacy Service Group

| Aspect | Legacy (`svgGrpCfg` / `FnVCCB_GetServiceGroupId`) | Service Configuration (SVR26B) |
|---|---|---|
| YANG location | `mobility-config-gnb/service-group-config` | `gnb-cu-cp-function-entries/service-configuration` |
| Key | Single key: `{service-group-id}` | Composite: `{plmn-index, service-sa-index}` or `{plmn-index, service-endc-index}` |
| Indexing | Key as an array index | Read: list order; Subscribe: linear search by composite key |
| Profile | Union / ruleOption | Flat parallel profile fields |
| Priority | Inline in service-group config | Pre-computed from legacy service-group mapping (`FnVCCB_UpdateSvcCfgPriorities`) |
| API result | One ID (priority inside API) | Single best `{plmn_index, service_index}` pair selected by entry priority/order |

---

## 1. Data Structures

### 1.1 Enumerations (SVR26B)

```cpp
enum class SvcCfgType  : uint8_t { SA = 0, ENDC = 1 };
enum class SvcCfgLevel : uint8_t { UE = 0, DRB = 1 };
```

`SvcCfgType` selects the SA or ENDC entry array; `SvcCfgLevel` selects whether runtime data is collected from all bearers in the UE or scoped to a single DRB/ERAB.

### 1.2 Profile Entry Types

`@VccbStruct.h`

| Type | Fields |
|---|---|
| `svcCfgSa_type` | Keys (`plmn_index`, `service_sa_index`), `name`, `service_sa_usage`; `slice[]` with `numSlice`, `qos_5qi[]` with `numQos5qi`, `arp[]` with `numArp`, `rfsp[]` with `numRfsp`, `ue_type[]` with `numUeType`; **`priority`** (pre-computed) |
| `svcCfgEndc_type` | Keys (`plmn_index`, `service_endc_index`), `name`, `service_endc_usage`; `spid[]` with `numSpid`, `qos_qci[]` with `numQosQci`, `arp[]` with `numArp`; **`priority`** (pre-computed) |
| `gnbConfig_type` | `svcCfgSaEntries[MAX_SVC_SA_ENTRIES]`, `svcCfgEndcEntries[MAX_SVC_ENDC_ENTRIES]` |

### 1.3 Internal Candidate Structure

Within `FnVCCB_GetServiceCfgBest`, matched entries are stored as:

```cpp
struct Candidate { int8_t plmnIdx; int8_t svcIdx; uint8_t prio; };
```

Each candidate carries the entry's pre-computed `priority` value, which is used for best-of selection.

### 1.4 Return Type (SVR26B)

All 4 public APIs return:

```cpp
std::pair<int8_t, int8_t>   // { plmn_index, service_index }
```

A return value of `{-1, -1}` indicates no match or an error condition.

---

## 2. Public API Signatures (SVR26B)

### 2.1 Priority Initialization

```cpp
void FnVCCB_UpdateSvcCfgPriorities(uint8_t gnbNum);
```

Called after configuration loading to populate the `priority` field of every SA and ENDC entry.

### 2.2 Matching APIs

```cpp
std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgSaByUe(
    callInfo_type *ptCi,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority, int8_t order);

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgSaByDrb(
    callInfo_type *ptCi, callDRBInfo_type *pDrb,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority, int8_t order);

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgEndcByUe(
    callInfo_type *ptCi,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority, int8_t order);

std::pair<int8_t, int8_t> FnVCCB_GetServiceCfgEndcByDrb(
    callInfo_type *ptCi, callERABInfo_type *pErab,
    const std::vector<std::pair<int8_t, int8_t>>& input_pairs,
    int8_t priority, int8_t order);
```

### Parameter Description

| Parameter | Type | Description |
|---|---|---|
| `ptCi` | `callInfo_type*` | Pointer to the call-info (UE context). |
| `pDrb` | `callDRBInfo_type*` | Pointer to a specific DRB (SA DRB-level only). |
| `pErab` | `callERABInfo_type*` | Pointer to a specific ERAB (ENDC DRB-level only). |
| `input_pairs` | `std::vector<std::pair<int8_t,int8_t>>` | Pre-selected candidate list of `{plmn_index, service_index}`. If **empty**, the API performs full runtime collection + profile matching. If **non-empty**, it bypasses matching and validates these pairs against configuration directly. |
| `priority` | `int8_t` | Threshold value for candidate filtering. Compared against each candidate's **pre-computed `priority` field** (from `FnVCCB_UpdateSvcCfgPriorities`), not against `service_index`. |
| `order` | `int8_t` | Selection direction: **0** = keep candidates whose `prio <= priority`, then select the one with the **smallest** `prio`; **1** (non-zero) = keep candidates whose `prio >= priority`, then select the one with the **largest** `prio`. |

---

## 3. Priority Initialization — `FnVCCB_UpdateSvcCfgPriorities`

### 3.1 Purpose

Called once after gNB configuration is loaded (or updated). It populates the `priority` field of every SA and ENDC service-configuration entry by mapping each entry's profile to legacy service-group priorities.

### 3.2 Signature

```cpp
void FnVCCB_UpdateSvcCfgPriorities(uint8_t gnbNum);
```

### 3.3 SA Entry Priority Computation

For each `svcCfgSaEntries[i]` with `service_sa_usage == SVC_CFG_USAGE_USE`:

1. Initialize `bestPriority = WORST_SVC_GRP_PRIORITY` (value 8).
2. Build a **single** `serviceGroupMappingInfo_type` (zero-initialized) combining all profile dimensions:
   - **Slice (sstSd)**: Iterate over `slice[0..numSlice)`, writing `mapInfo.sstSd = (SST << 24) | (SD & 0x00FFFFFF)` each time. Since each iteration overwrites the previous value, the **last** slice entry's `sstSd` is the one that remains when service-group matching runs. If `numSlice == 0`, `sstSd` stays `0` (zero-initialized).
   - **5QI map**: `mapInfo.qos5qiMap[q]` = `TRUE` for each 5QI value `q` in the entry's `qos_5qi[]`.
   - **ARP map**: `mapInfo.arpMap[a]` = `TRUE` for each ARP value `a` in the entry's `arp[]` (where `a < MAX_ARP_LEVEL`).
   - **ARP×5QI cross-product**: `mapInfo.arpQos5qiMap[q][a]` = `TRUE` for every combination of `qos_5qi[q]` × `arp[a]` (where `arp[a] < MAX_ARP_LEVEL`).
3. Loop over `svcGrpCfg[MIN_SVC_GRP_CONFIG_ID .. MAX_SVC_GRP_DB_NUM)`:
   - Skip entries where `usage != TRUE` or `plmnIndex != pSaCfg->plmn_index`.
   - Skip if the service group's priority is already worse than current `bestPriority`, or if equal priority but higher `svcGrpId`.
   - Call `FnVCCB_FindSvcGrpIdByMappingRule` to attempt matching; update `bestSvcGrpId` and `bestPriority` on success.
4. Store `bestPriority` → `pSaCfg->priority`.

For entries with `usage != SVC_CFG_USAGE_USE`, set `priority = WORST_SVC_GRP_PRIORITY`.

### 3.4 ENDC Entry Priority Computation

For each `svcCfgEndcEntries[i]` with `service_endc_usage == SVC_CFG_USAGE_USE`:

1. Initialize `bestPriority = WORST_SVC_GRP_PRIORITY`.
2. Build a **single** `serviceGroupMappingInfo_type` (zero-initialized):
   - **QCI → 5QI map**: `mapInfo.qos5qiMap[q]` = `TRUE` for each QCI value `q` in the entry's `qos_qci[]` (QCI values are mapped directly into the 5QI map array).
   - **ARP map**: `mapInfo.arpMap[a]` = `TRUE` for each ARP value `a` (where `a < MAX_ARP_LEVEL`).
   - **ARP×QCI cross-product**: `mapInfo.arpQos5qiMap[q][a]` = `TRUE` for every combination of `qos_qci[q]` × `arp[a]` (where `arp[a] < MAX_ARP_LEVEL`).
3. Loop over `svcGrpCfg[]` with same filtering and matching logic as SA.
4. Store `bestPriority` → `pEndcCfg->priority`.

For entries with `usage != SVC_CFG_USAGE_USE`, set `priority = WORST_SVC_GRP_PRIORITY`.

### 3.5 Flow Diagram

```
FnVCCB_UpdateSvcCfgPriorities(gnbNum)
  │
  ├─ [SA entries] for each svcCfgSaEntries[i]:
  │   ├─ usage != USE → priority = WORST_SVC_GRP_PRIORITY, skip
  │   └─ usage == USE:
  │       ├─ Build single serviceGroupMappingInfo_type (zero-init):
  │       │   ├─ Iterate slices → overwrite mapInfo.sstSd each time (last slice wins)
  │       │   ├─ Populate qos5qiMap from qos_5qi[]
  │       │   ├─ Populate arpMap from arp[] (guarded by < MAX_ARP_LEVEL)
  │       │   └─ Populate arpQos5qiMap cross-product (guarded by < MAX_ARP_LEVEL)
  │       ├─ for each svcGrpCfg[] (usage + PLMN filter + priority pruning):
  │       │   └──► FnVCCB_FindSvcGrpIdByMappingRule → update bestPriority
  │       └─ pSaCfg->priority = bestPriority
  │
  └─ [ENDC entries] for each svcCfgEndcEntries[i]:
      ├─ usage != USE → priority = WORST_SVC_GRP_PRIORITY, skip
      └─ usage == USE:
          ├─ Build single serviceGroupMappingInfo_type (zero-init):
          │   ├─ Populate qos5qiMap from qos_qci[] (QCI mapped as 5QI index)
          │   ├─ Populate arpMap from arp[] (guarded by < MAX_ARP_LEVEL)
          │   └─ Populate arpQos5qiMap cross-product (guarded by < MAX_ARP_LEVEL)
          ├─ for each svcGrpCfg[] (usage + PLMN filter + priority pruning):
          │   └──► FnVCCB_FindSvcGrpIdByMappingRule → update bestPriority
          └─ pEndcCfg->priority = bestPriority
```

---

## 4. Execution Flow — `FnVCCB_GetServiceCfgBest`

### 4.1 Function Call Graph

All four public APIs are thin wrappers that delegate to a single unified dispatcher:

```
FnVCCB_GetServiceCfgSaByUe(ptCi, input_pairs, priority, order)
  └──► FnVCCB_GetServiceCfgBest(ptCi, SA, UE, nullptr, input_pairs, priority, order)

FnVCCB_GetServiceCfgSaByDrb(ptCi, pDrb, input_pairs, priority, order)
  └──► FnVCCB_GetServiceCfgBest(ptCi, SA, DRB, pDrb, input_pairs, priority, order)

FnVCCB_GetServiceCfgEndcByUe(ptCi, input_pairs, priority, order)
  └──► FnVCCB_GetServiceCfgBest(ptCi, ENDC, UE, nullptr, input_pairs, priority, order)

FnVCCB_GetServiceCfgEndcByDrb(ptCi, pErab, input_pairs, priority, order)
  └──► FnVCCB_GetServiceCfgBest(ptCi, ENDC, DRB, pErab, input_pairs, priority, order)
```

### 4.2 `FnVCCB_GetServiceCfgBest` — Unified Dispatcher

```
FnVCCB_GetServiceCfgBest(ptCi, cfgType, level, pBearer, input_pairs, priority, order)
  │
  ├─ [Precondition] Validate ptCi, resolve uePlmnIndex via VLIB_getMultipleGnbPlmnIndexInfo
  │
  ├─ [Branch A] input_pairs is EMPTY → full runtime matching
  │   ├─ cfgType == SA
  │   │   ├─ level == UE  → collect S-NSSAI / 5QI / ARP from all active PDU sessions & QoS flows
  │   │   └─ level == DRB → collect S-NSSAI / 5QI / ARP scoped to pDrb's PDU session & QoS flows
  │   │   └─ Collect RFSP, isRedcapUe from UE context
  │   │   └─ Loop svcCfgSaEntries[]: filter by usage + PLMN
  │   │       └──► FnVCCB_IsSvcCfgSaProfileMatch → append {plmnIdx, svcIdx, pEntry->priority}
  │   │
  │   └─ cfgType == ENDC
  │       ├─ level == UE  → collect QCI / ARP from all active ERABs
  │       └─ level == DRB → collect QCI / ARP from single pErab
  │       └─ Collect SPID from UE subscriber profile
  │       └─ Loop svcCfgEndcEntries[]: filter by usage + PLMN
  │           └──► FnVCCB_IsSvcCfgEndcProfileMatch → append {plmnIdx, svcIdx, pEntry->priority}
  │
  ├─ [Branch B] input_pairs is NON-EMPTY → bypass matching
  │   ├─ cfgType == SA  → for each input pair, find matching entry in svcCfgSaEntries[] by
  │   │                    plmn_index + service_sa_index + usage → append {plmnIdx, svcIdx, pEntry->priority}
  │   └─ cfgType == ENDC → for each input pair, find matching entry in svcCfgEndcEntries[] by
  │                         plmn_index + service_endc_index + usage → append {plmnIdx, svcIdx, pEntry->priority}
  │
  └─ [Best-of selection] from candidates, filter by prio vs priority threshold,
                          then select single best by prio → return {plmnIdx, svcIdx}
```

### 4.3 Preconditions (Common to All 4 APIs)

1. Service-configuration for the serving gNB has been read and/or subscribed into VCCB so that `gnbConfig.svcCfgSaEntries[]` and `svcCfgEndcEntries[]` are populated.
2. `FnVCCB_UpdateSvcCfgPriorities` has been called for this gNB so that each entry's `priority` field is initialized.
3. `ptCi` must be non-null and `ptCi->gnbNum < MAX_GNB_PER_SYS`.
4. UE PLMN must be resolvable via `VLIB_getMultipleGnbPlmnIndexInfo`.
5. Return value `{-1, -1}` means no suitable match was found or a precondition failed.

---

## 5. Profile Matchers (Internal)

### 5.1 `FnVCCB_IsSvcCfgSaProfileMatch`

```cpp
static bool FnVCCB_IsSvcCfgSaProfileMatch(
    const svcCfgSa_type *pEntry,
    const callSNSSAI_type *pSNssaiList, uint8_t numSNssai,
    const uint8_t *pFiveQiList, uint8_t numFiveQi,
    const uint8_t *pArpList, uint8_t numArp,
    uint16_t rfspVal, bool rfspPresent,
    bool isRedcapUe);
```

**Dimensions and match logic:**

| Dimension | Wildcard condition | Match rule |
|---|---|---|
| **Slice** | `numSlice == 0` → always match | Compare `(SST<<24 \| SD)` from entry against each S-NSSAI in runtime list (OR within) |
| **5QI** | `numQos5qi == 0` → always match | Entry 5QI value == any runtime 5QI (OR within) |
| **ARP** | `numArp == 0` → always match | Entry ARP value == any runtime ARP (OR within) |
| **RFSP** | `numRfsp == 0` → always match | Checked **only if** `rfspPresent == true`; entry RFSP == UE's RFSP index (OR within) |
| **UE-type** | `numUeType == 0` → always match | `currentUeType` = `SVC_CFG_UE_TYPE_REDCAP` if `isRedcapUe`, else `SVC_CFG_UE_TYPE_NORMAL`; entry value == `currentUeType` (OR within) |

**Final result:** `sliceMatch && fiveQiMatch && arpMatch && rfspMatch && ueTypeMatch` (AND across all dimensions).

### 5.2 `FnVCCB_IsSvcCfgEndcProfileMatch`

```cpp
static bool FnVCCB_IsSvcCfgEndcProfileMatch(
    const svcCfgEndc_type *pEntry,
    uint16_t spidVal, bool spidPresent,
    const uint8_t *pQciList, uint8_t numQci,
    const uint8_t *pArpList, uint8_t numArp);
```

**Dimensions and match logic:**

| Dimension | Wildcard condition | Match rule |
|---|---|---|
| **SPID** | `numSpid == 0` → always match | Checked **only if** `spidPresent == true`; entry SPID == subscriber profile SPID (OR within) |
| **QCI** | `numQosQci == 0` → always match | Entry QCI value == any runtime QCI (OR within) |
| **ARP** | `numArp == 0` → always match | Entry ARP value == any runtime ARP (OR within) |

**Final result:** `spidMatch && qciMatch && arpMatch` (AND across all dimensions).

---

## 6. Runtime Data Collection Detail

### 6.1 SA — UE Level

| Data | Source | Deduplication |
|---|---|---|
| S-NSSAI list | `ptCi->pduSess[*].sNssai` for all active (`CI_use`) PDU sessions | By `sstSd` value; max `MAX_PDU_SESS_PER_CALL` |
| 5QI list | `VLIB_getFiveQi(ptQi)` for all active QoS flows across all active PDU sessions | By value; skip `0xFF`; max `MAX_QOS_FLOW_PER_CALL` |
| ARP list | `ptQi->qosParam.allocAndRetent.priorityLevel` | By value; skip `>= MAX_ARP_LEVEL`; max `MAX_QOS_FLOW_PER_CALL` |
| RFSP | `ptCi->ueInfo.rfsp.indexToRFSP` | Present if `rfspPresent == OPT_PRESENT` |
| UE type | `IS_REDCAP_UE(ptCi)` → `SVC_CFG_UE_TYPE_REDCAP` or `SVC_CFG_UE_TYPE_NORMAL` | — |

### 6.2 SA — DRB Level

| Data | Source | Deduplication |
|---|---|---|
| S-NSSAI | `ptCi->pduSess[pDrb->pduIdxInCall].sNssai` | Single value |
| 5QI list | QoS flows mapped to this DRB (`pDrb->qosFlowIndex[]`) | By value; max `MAX_QOS_FLOW_PER_DRB` |
| ARP list | Same QoS flows | By value; max `MAX_QOS_FLOW_PER_DRB` |
| RFSP / UE type | Same as UE level (from `ptCi`) | — |

### 6.3 ENDC — UE Level

| Data | Source | Deduplication |
|---|---|---|
| QCI list | `ptCi->erabInfo[*].erabQos.qciValue.QCI` for all active ERABs | By value; max `MAX_ERAB_PER_CALL` |
| ARP list | `ptCi->erabInfo[*].erabQos.aRP.priority` | By value; skip `>= MAX_ARP_LEVEL`; max `MAX_ERAB_PER_CALL` |
| SPID | `ptCi->ueInfo.rfsp.subscriberProfile` | Present if `ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT` |

### 6.4 ENDC — DRB Level

| Data | Source | Deduplication |
|---|---|---|
| QCI | `pErab->erabQos.qciValue.QCI` | Single value |
| ARP | `pErab->erabQos.aRP.priority` | Single value; valid only if `< MAX_ARP_LEVEL` |
| SPID | Same as UE level (from `ptCi`) | — |

---

## 7. Best-of Selection Logic

After collecting all matching candidates (each carrying `{plmnIdx, svcIdx, prio}`), the function selects a single best pair based on the entry's **pre-computed `prio`** field (not `service_index`):

```
for each candidate {plmnIdx, svcIdx, prio}:
    if order == 0:                                  // "lower-is-better"
        keep candidate where prio <= priority       // filter: entry priority <= input threshold
        prefer smallest prio; break ties by smallest svcIdx
    else (order != 0):                              // "higher-is-better"
        keep candidate where prio >= priority       // filter: entry priority >= input threshold
        prefer largest prio; break ties by largest svcIdx
```

| `order` value | Filter condition | Preference |
|---|---|---|
| **0** | `candidate.prio <= priority` | Smallest `prio`; then smallest `svcIdx` |
| **non-zero** (1) | `candidate.prio >= priority` | Largest `prio`; then largest `svcIdx` |

If no candidate survives the filter, the return value is `{-1, -1}`.

**Key distinction from prior version:** The selection criteria uses the entry's **`priority`** field (pre-computed from service-group mapping) — not the `service_index`. The `service_index` is only used as a tie-breaker when two candidates share the same `priority`.

---

## 8. Input-Pair Bypass Mode

When `input_pairs` is **non-empty**, the function skips all runtime data collection and profile matching. Instead:

1. For each pair `(plmn_index, service_index)` in `input_pairs`:
   - Locate the corresponding entry in `svcCfgSaEntries[]` (SA) or `svcCfgEndcEntries[]` (ENDC).
   - Verify the entry's `plmn_index`, `service_sa_index` / `service_endc_index`, and `usage == SVC_CFG_USAGE_USE`.
   - If valid, add to candidates with the entry's pre-computed `priority`.
2. Apply the same best-of selection (Section 7) on the validated candidates.

This mode allows callers to re-evaluate a known set of service indices (e.g., from a previous matching result) against the current priority/order criteria without repeating the full profile match.

---

## 9. Entry Filtering

Before profile matching (or input-pair validation), each entry is filtered by:

| Filter | SA entries | ENDC entries |
|---|---|---|
| Usage | `service_sa_usage == SVC_CFG_USAGE_USE` | `service_endc_usage == SVC_CFG_USAGE_USE` |
| PLMN | `plmn_index == uePlmnIndex` | `plmn_index == uePlmnIndex` |

`uePlmnIndex` is resolved from the UE's served PLMN via `VLIB_getMultipleGnbPlmnIndexInfo`.

---

## 10. Error Handling

| Condition | Return |
|---|---|
| `ptCi == nullptr` | `{-1, -1}` |
| `ptCi->gnbNum >= MAX_GNB_PER_SYS` | `{-1, -1}` |
| PLMN index resolution failure | `{-1, -1}` |
| `pDrb == nullptr` (DRB-level SA) | `{-1, -1}` |
| `pErab == nullptr` (DRB-level ENDC) | `{-1, -1}` |
| `pDrb->pduIdxInCall >= MAX_PDU_SESS_PER_CALL` | `{-1, -1}` |
| No candidate survives priority/order filter | `{-1, -1}` |
| `gnbNum >= MAX_GNB_PER_SYS` (in `FnVCCB_UpdateSvcCfgPriorities`) | Early return, no entries updated |
