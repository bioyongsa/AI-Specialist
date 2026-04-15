# FR10.CACP.SR1.HD1.VCCB.DD2 — Service Configuration Matching (SVR26B)

VCCB exposes API functions that determine the **best** service index (with PLMN index) by matching stored profiles against runtime data and applying priority/order–based selection.

| FR aspect | Detailed Design |
|---|---|
| QoS, Slice, RFSP, UE-type; UE-level or DRB-level | **SA:** `FnVCCB_GetServiceCfgSaByUe` (UE-level) / `FnVCCB_GetServiceCfgSaByDrb` (DRB-level) |
| Distinguish SA and ENDC | **ENDC:** `FnVCCB_GetServiceCfgEndcByUe` (UE-level) / `FnVCCB_GetServiceCfgEndcByDrb` (DRB-level) |

The service-configuration container defines operator SA and ENDC service profiles.
VCCB loads and updates them per FR10.CACP.SR1.HD1.VCCB.DD1, then resolves the **single best** service index per FR10.CACP.SR1.HD1.VCCB.DD2 when processing UEs or DRB-level context.

---

## Design Principles

| Principle | Description |
|---|---|
| **SA vs ENDC** | Separate entry lists in configuration, separate public APIs, separate profile match rules. |
| **Wildcard** | If a profile dimension (slice / 5QI / ARP / RFSP / UE-type for SA; SPID / QCI / ARP for ENDC) has no configured values (count 0), that dimension always matches. |
| **Conjunction across dimensions** | All non-wildcard dimensions must match (**AND**); within one dimension, any configured value may match any collected runtime value (**OR**). |
| **Usage** | Only entries with `usage == SVC_CFG_USAGE_USE` participate in matching for that stack (SA or ENDC). |
| **Unified dispatcher** | All 4 public APIs delegate to a single internal function `FnVCCB_GetServiceCfgBest` parameterized by `SvcCfgType` (SA / ENDC) and `SvcCfgLevel` (UE / DRB). |
| **Best-of selection** | Instead of returning all matches, the API returns the single best `{plmn_index, service_index}` pair filtered by `priority` and ordered by `order`. |
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
| API result | One ID (priority inside API) | Single best `{plmn_index, service_index}` pair selected by priority/order |

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
| `svcCfgSa_type` | Keys (`plmn_index`, `service_sa_index`), `name`, `service_sa_usage`; `slice[]` with `numSlice`, `qos_5qi[]` with `numQos5qi`, `arp[]` with `numArp`, `rfsp[]` with `numRfsp`, `ue_type[]` with `numUeType` |
| `svcCfgEndc_type` | Keys (`plmn_index`, `service_endc_index`), `name`, `service_endc_usage`; `spid[]` with `numSpid`, `qos_qci[]` with `numQosQci`, `arp[]` with `numArp` |
| `gnbConfig_type` | `svcCfgSaEntries[MAX_SVC_SA_ENTRIES]`, `svcCfgEndcEntries[MAX_SVC_ENDC_ENTRIES]` |

### 1.3 Return Type (SVR26B)

All 4 public APIs return:

```cpp
std::pair<int8_t, int8_t>   // { plmn_index, service_index }
```

A return value of `{-1, -1}` indicates no match or an error condition.

---

## 2. Public API Signatures (SVR26B)

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
| `priority` | `int8_t` | Threshold value for candidate filtering (compared against `service_index`). |
| `order` | `int8_t` | Selection direction: **0** = select the candidate with the **lowest** `service_index` below `priority`; **1** = select the candidate with the **highest** `service_index` above `priority`. |

---

## 3. Execution Flow

### 3.1 Function Call Graph

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

### 3.2 `FnVCCB_GetServiceCfgBest` — Unified Dispatcher

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
  │   │       └──► FnVCCB_IsSvcCfgSaProfileMatch → append to candidates
  │   │
  │   └─ cfgType == ENDC
  │       ├─ level == UE  → collect QCI / ARP from all active ERABs
  │       └─ level == DRB → collect QCI / ARP from single pErab
  │       └─ Collect SPID from UE subscriber profile
  │       └─ Loop svcCfgEndcEntries[]: filter by usage + PLMN
  │           └──► FnVCCB_IsSvcCfgEndcProfileMatch → append to candidates
  │
  ├─ [Branch B] input_pairs is NON-EMPTY → bypass matching
  │   ├─ cfgType == SA  → for each input pair, find matching entry in svcCfgSaEntries[] by
  │   │                    plmn_index + service_sa_index + usage → append to candidates
  │   └─ cfgType == ENDC → for each input pair, find matching entry in svcCfgEndcEntries[] by
  │                         plmn_index + service_endc_index + usage → append to candidates
  │
  └─ [Best-of selection] from candidates, apply priority/order filter → return single best pair
```

### 3.3 Preconditions (Common to All 4 APIs)

1. Service-configuration for the serving gNB has been read and/or subscribed into VCCB so that `gnbConfig.svcCfgSaEntries[]` and `svcCfgEndcEntries[]` are populated.
2. `ptCi` must be non-null and `ptCi->gnbNum < MAX_GNB_PER_SYS`.
3. UE PLMN must be resolvable via `VLIB_getMultipleGnbPlmnIndexInfo`.
4. Return value `{-1, -1}` means no suitable match was found or a precondition failed.

---

## 4. Profile Matchers (Internal)

### 4.1 `FnVCCB_IsSvcCfgSaProfileMatch`

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

### 4.2 `FnVCCB_IsSvcCfgEndcProfileMatch`

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

## 5. Runtime Data Collection Detail

### 5.1 SA — UE Level

| Data | Source | Deduplication |
|---|---|---|
| S-NSSAI list | `ptCi->pduSess[*].sNssai` for all active (`CI_use`) PDU sessions | By `sstSd` value; max `MAX_PDU_SESS_PER_CALL` |
| 5QI list | `VLIB_getFiveQi(ptQi)` for all active QoS flows across all active PDU sessions | By value; skip `0xFF`; max `MAX_QOS_FLOW_PER_CALL` |
| ARP list | `ptQi->qosParam.allocAndRetent.priorityLevel` | By value; skip `>= MAX_ARP_LEVEL`; max `MAX_QOS_FLOW_PER_CALL` |
| RFSP | `ptCi->ueInfo.rfsp.indexToRFSP` | Present if `rfspPresent == OPT_PRESENT` |
| UE type | `IS_REDCAP_UE(ptCi)` → `SVC_CFG_UE_TYPE_REDCAP` or `SVC_CFG_UE_TYPE_NORMAL` | — |

### 5.2 SA — DRB Level

| Data | Source | Deduplication |
|---|---|---|
| S-NSSAI | `ptCi->pduSess[pDrb->pduIdxInCall].sNssai` | Single value |
| 5QI list | QoS flows mapped to this DRB (`pDrb->qosFlowIndex[]`) | By value; max `MAX_QOS_FLOW_PER_DRB` |
| ARP list | Same QoS flows | By value; max `MAX_QOS_FLOW_PER_DRB` |
| RFSP / UE type | Same as UE level (from `ptCi`) | — |

### 5.3 ENDC — UE Level

| Data | Source | Deduplication |
|---|---|---|
| QCI list | `ptCi->erabInfo[*].erabQos.qciValue.QCI` for all active ERABs | By value; max `MAX_ERAB_PER_CALL` |
| ARP list | `ptCi->erabInfo[*].erabQos.aRP.priority` | By value; skip `>= MAX_ARP_LEVEL`; max `MAX_ERAB_PER_CALL` |
| SPID | `ptCi->ueInfo.rfsp.subscriberProfile` | Present if `ptCi->ueInfo.rfsp.rfspPresent == OPT_PRESENT` |

### 5.4 ENDC — DRB Level

| Data | Source | Deduplication |
|---|---|---|
| QCI | `pErab->erabQos.qciValue.QCI` | Single value |
| ARP | `pErab->erabQos.aRP.priority` | Single value; valid only if `< MAX_ARP_LEVEL` |
| SPID | Same as UE level (from `ptCi`) | — |

---

## 6. Best-of Selection Logic

After collecting all matching `{plmn_index, service_index}` candidates, the function selects a single best pair:

```
for each candidate (plmn_index, service_index):
    if order == 0:                              // "lower-is-better"
        keep candidate where service_index < priority
        prefer smallest service_index; break ties by smallest plmn_index
    else (order == 1):                          // "higher-is-better"
        keep candidate where service_index > priority
        prefer largest service_index; break ties by largest plmn_index
```

| `order` value | Filter | Preference |
|---|---|---|
| **0** | `service_index < priority` | Smallest `service_index`; then smallest `plmn_index` |
| **1** (non-zero) | `service_index > priority` | Largest `service_index`; then largest `plmn_index` |

If no candidate survives the filter, the return value is `{-1, -1}`.

---

## 7. Input-Pair Bypass Mode

When `input_pairs` is **non-empty**, the function skips all runtime data collection and profile matching. Instead:

1. For each pair `(plmn_index, service_index)` in `input_pairs`:
   - Locate the corresponding entry in `svcCfgSaEntries[]` (SA) or `svcCfgEndcEntries[]` (ENDC).
   - Verify the entry's `plmn_index`, `service_sa_index` / `service_endc_index`, and `usage == SVC_CFG_USAGE_USE`.
   - If valid, add to `candidates`.
2. Apply the same best-of selection (Section 6) on the validated candidates.

This mode allows callers to re-evaluate a known set of service indices (e.g., from a previous matching result) against the current priority/order criteria without repeating the full profile match.

---

## 8. Entry Filtering

Before profile matching (or input-pair validation), each entry is filtered by:

| Filter | SA entries | ENDC entries |
|---|---|---|
| Usage | `service_sa_usage == SVC_CFG_USAGE_USE` | `service_endc_usage == SVC_CFG_USAGE_USE` |
| PLMN | `plmn_index == uePlmnIndex` | `plmn_index == uePlmnIndex` |

`uePlmnIndex` is resolved from the UE's served PLMN via `VLIB_getMultipleGnbPlmnIndexInfo`.

---

## 9. Error Handling

| Condition | Return |
|---|---|
| `ptCi == nullptr` | `{-1, -1}` |
| `ptCi->gnbNum >= MAX_GNB_PER_SYS` | `{-1, -1}` |
| PLMN index resolution failure | `{-1, -1}` |
| `pDrb == nullptr` (DRB-level SA) | `{-1, -1}` |
| `pErab == nullptr` (DRB-level ENDC) | `{-1, -1}` |
| `pDrb->pduIdxInCall >= MAX_PDU_SESS_PER_CALL` | `{-1, -1}` |
| No candidate survives priority/order filter | `{-1, -1}` |
