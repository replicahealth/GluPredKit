# Parsers Documentation


## IOBP2

66% of pediatric participants and 80% of adult participants will use the bionic pancreas, given by chapter 3.3 in the protocol.

Control group will continue on their current diabetes management, but all will use Dexcom G6 (Chapter 3.4 and A.2 in the protocol).

The total participants in the processed data is 436, where around 100 are in the pediatric group. This is what we see in the patient roster: 
Treatment groups: {'BP': 216, 'BPFiasp': 113, 'Control': 107}

### `insulin_delivery_device`
All of the subjects that are not in the control arm, have the insulin delivery device named "Gen 4 iLet Bionic Pancreas". 

The final value distribution count (subjects for each device) is: 
{'Gen 4 iLet Bionic Pancreas': 329, 'Insulin Pen': 39, 't:slim X2': 31, 'OmniPod': 18, 'MiniMed 670G': 14, 'MiniMed 630G': 2, 'MiniMed 507': 1, 'MiniMed 551 (530G)': 1, nan: 1}

### `insulin_delivery_algorithm`
As for the insulin delivery device, we set the subjects that are not in the control arm to the "iLet Bionic Pancreas" insulin delivery algorithm.

Other algorithms are derived from the insulin pump name, or nan if no insulin delivery device is found. 

{'Bionic Pancreas': 329, 'Multiple Daily Injections': 39, 'basal-bolus': 31, 'SmartGuard': 16, 'Control:IQ': 14, 'Basal:IQ': 5, 'Low-Glucose Suspend': 1, nan: 1}

### `cgm_device`
As stated in the protocol, this is set to Dexcom G6 for all participants.

### `is_pregnant`
Pregnancy is exclusion criterion, as stated in "Protocol Summary", "Exclusion", 13: "Pregnant (positive urine hCG), breast feeding, plan to become pregnant in the next 3 months, or sexually active without use of contraception".

Hence, the value is set to False for all participants. 

### `insulin_delivery_modality`
For all iLet users, it is set to AID. SAP, AID, or MDI for the other users is based on device and algorithm. 

Distribution: 
{'AID': 366, 'MDI': 39, 'SAP': 31}

### `insulin_type_bolus` and `insulin_type_basal`

AID and SAP users will have the same bolus and basal insulin. 
Bionic Pancreas group will have either Fiasp or another rapid-acting insulin that they are assigned, based on the control group they belong to and the insulin type noted in the data.

The insulin type data is in `IOBP2Insulin.txt`. There might be several samples per subject. This is because it includes the insulin from before the study, as well as the insulin in the transition period after the study. **The constitution data is only available during the study, and not in the transition period.** Which means we can safely assign the same insulin type information to all the data for a single subject. 

We can filter on type "Started after enrollment", and having the available start and end date, which will give us the relevant insulin type.

RCT Period (13 weeks)
- Control group: stays on their usual insulin regimen (MDI or pump, with CGM).
- BP groups (iLet users): stay on the iLet system for the entire 13 weeks.
- No therapy switching occurs during the RCT; this is the main data collection phase.

Transition Phase (2–4 days, immediately after RCT)
- Applies only to BP group participants.
- Randomized to either:
  - Transition back to their usual therapy (MDI or pump) guided by dosing recommendations from the iLet system.
  - Transition back to their own pre-study regimen (with safety adjustments allowed).

Extension Study (optional, for Control group only)
- Control group participants who complete the RCT may be offered entry into a separate Extension Study where they can use the iLet system.

Relevant Protocol Sections
- Protocol Summary 
- Section 2.2.2 Transition Phase 
- Section 5.4 Transition Phase 
- Chapter 6: Transition Phase

### Assumptions and Limitations

- It is not clear which types of insulin is injected in the `IOBP2ManualInsulinInj.txt`. This data is not included in the processing of the data, adhering to a decision also made in babelbetes. The file has 198 samples, which is 0,002% of the samples in the total dataset. 
- It is not clear what `IOBP2MealDose.txt` is doing. There are only a few meals per subjects, and the data is not overlapping with the continuous study data. 


