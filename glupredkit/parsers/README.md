# Parsers Documentation

## IOBP2

### Overview

In the iLet trial, continuous data is primarily collected for participants using the Bionic Pancreas. The control arm has limited data (11 subjects with short time horizons and substantial missing data), so we filter to include only the treatment groups in the processed data.

**Study Design:**
- 66% of pediatric participants and 80% of adult participants use the Bionic Pancreas (protocol chapter 3.3)
- Control group continues current diabetes management with Dexcom G6 (chapters 3.4 and A.2)
- **Treatment groups:** BP (216), BPFiasp (113), Control (107) - *Control excluded from processing*

### Data Fields

| Field | Value | Notes |
|-------|-------|-------|
| `insulin_delivery_device` | Beta Bionics Gen 4 iLet | All treatment group subjects |
| `insulin_delivery_algorithm` | iLet Bionic Pancreas | All treatment group subjects |
| `insulin_delivery_modality` | AID | Automated Insulin Delivery for all |
| `cgm_device` | Dexcom G6 | Protocol requirement for all participants |
| `is_pregnant` | False | Pregnancy is exclusion criterion (Protocol Summary, Exclusion 13) |

**Insulin Types:**
- **BPFiasp group:** Both `insulin_type_bolus` and `insulin_type_basal` set to "Fiasp"
- **BP group:** Determined from `IOBP2Insulin.txt` using pump route data
  - Priority system when both available: enrollment status → valid dates → default selection
  - Same insulin type assigned to both bolus and basal

### Processed Data Statistics

**Dataset Overview:**
- **Total subjects:** 332 (BP + BPFiasp groups only)
- **Time points:** 9,676,885 at 5-minute intervals
- **Dataset shape:** (9,676,885, 22)
- **Output:** `data/processed/IOBP2.csv`

**Data Coverage:**
- Glucose readings: 7,804,251
- Bolus events: 1,711,927
- Basal events: 6,233,455
- Meal events: 87,161

**Demographics:**
| Metric | Statistics |
|--------|------------|
| Age | Mean: 31.8 ± 19.2 years (range: 6-82) |
| Gender | Female: 169, Male: 163 |
| Height | Mean: 5.39 ± 0.51 ft (range: 3.54-6.37) |
| Weight | Mean: 161.6 ± 51.9 lbs (range: 49.4-350.0) |
| Age at diagnosis | Mean: 13.8 ± 11.4 years (range: 0-58) |

**Ethnicity Distribution:**
- White: 254 (76.5%)
- Black/African American: 37 (11.1%)
- White, Hispanic/Latino: 21 (6.3%)
- More than one race: 7 (2.1%)
- More than one race, Hispanic/Latino: 4 (1.2%)

**Insulin Distribution:**
- Humalog (Lispro): 126 subjects
- Fiasp: 113 subjects  
- Novolog (Aspart): 90 subjects


### Assumptions and Limitations

**Excluded Data:**
- **`IOBP2ManualInsulinInj.txt`:** Insulin types unclear; excluded following babelbetes approach. Contains only 198 samples (0.002% of total dataset).
- **`IOBP2MealDose.txt`:** Purpose unclear; minimal meal data per subject with no overlap to continuous study data.

**Data Processing Notes:**
- Control group excluded due to limited continuous monitoring data
- Insulin type determination prioritizes enrollment status and date validity


