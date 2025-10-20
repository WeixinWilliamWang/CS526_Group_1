import pandas as pd
import numpy as np

from AI_clinician.preprocessing.columns import *

def calculate_onset(abx, bacterio, stay_id, chart_events=None, lab_events=None):
    """
    Calculate the onset of sepsis for a given ICU stay using Sepsis-3 definition:
    1. Identify suspected infection window (antibiotics + cultures)
    2. Calculate SOFA scores
    3. Mark onset when SOFA increases by ≥2 points after infection suspicion
    """
    try:
        # Get antibiotic and bacterio events for this stay
        abx_stay = abx[abx[C_ICUSTAYID] == stay_id].copy()
        bacterio_stay = bacterio[bacterio[C_ICUSTAYID] == stay_id].copy()
        
        # Check if we have any events
        if len(abx_stay) == 0 or len(bacterio_stay) == 0:
            return None
        
        # Convert timestamps to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(abx_stay[C_STARTDATE]):
            abx_stay = abx_stay.copy()  # Create a copy to avoid SettingWithCopyWarning
            abx_stay[C_STARTDATE] = pd.to_datetime(abx_stay[C_STARTDATE], unit='s', errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(bacterio_stay[C_CHARTTIME]):
            bacterio_stay = bacterio_stay.copy()  # Create a copy to avoid SettingWithCopyWarning
            bacterio_stay[C_CHARTTIME] = pd.to_datetime(bacterio_stay[C_CHARTTIME], unit='s', errors='coerce')
        
        # Get first antibiotic administration
        first_abx = abx_stay[C_STARTDATE].min()
        if pd.isna(first_abx):
            return None
        
        # Get bacterio events within 24 hours before first antibiotic
        bacterio_before_abx = bacterio_stay[
            (bacterio_stay[C_CHARTTIME] <= first_abx) & 
            (bacterio_stay[C_CHARTTIME] >= first_abx - pd.Timedelta(hours=24))
        ]
        
        if len(bacterio_before_abx) == 0:
            return None
        
        # Get the first bacterio event as potential infection time
        infection_time = bacterio_before_abx[C_CHARTTIME].min()
        
        # If we have chart and lab events, calculate SOFA scores
        if chart_events is not None and lab_events is not None:
            # Get events for this stay
            stay_chart = chart_events[chart_events[C_ICUSTAYID] == stay_id]
            stay_lab = lab_events[lab_events[C_ICUSTAYID] == stay_id]
            
            # Calculate baseline SOFA (24h before infection)
            baseline_window = pd.Timedelta(hours=24)
            baseline_start = infection_time - baseline_window
            baseline_end = infection_time
            
            baseline_chart = stay_chart[
                (stay_chart[C_CHARTTIME] >= baseline_start) & 
                (stay_chart[C_CHARTTIME] <= baseline_end)
            ]
            baseline_lab = stay_lab[
                (stay_lab[C_CHARTTIME] >= baseline_start) & 
                (stay_lab[C_CHARTTIME] <= baseline_end)
            ]
            
            # Calculate baseline SOFA score
            baseline_sofa = 0
            
            # Respiratory (PaO2/FiO2)
            if C_PAO2 in baseline_chart.columns and C_FIO2_1 in baseline_chart.columns:
                pf_ratio = baseline_chart[C_PAO2].mean() / baseline_chart[C_FIO2_1].mean()
                if pd.notna(pf_ratio):
                    if pf_ratio < 100:
                        baseline_sofa += 4
                    elif pf_ratio < 200:
                        baseline_sofa += 3
                    elif pf_ratio < 300:
                        baseline_sofa += 2
                    elif pf_ratio < 400:
                        baseline_sofa += 1
            
            # Coagulation (Platelets)
            if C_PLATELETS_COUNT in baseline_lab.columns:
                platelets = baseline_lab[C_PLATELETS_COUNT].mean()
                if pd.notna(platelets):
                    if platelets < 20:
                        baseline_sofa += 4
                    elif platelets < 50:
                        baseline_sofa += 3
                    elif platelets < 100:
                        baseline_sofa += 2
                    elif platelets < 150:
                        baseline_sofa += 1
            
            # Liver (Bilirubin)
            if C_TOTAL_BILI in baseline_lab.columns:
                bili = baseline_lab[C_TOTAL_BILI].mean()
                if pd.notna(bili):
                    if bili > 12:
                        baseline_sofa += 4
                    elif bili > 6:
                        baseline_sofa += 3
                    elif bili > 2:
                        baseline_sofa += 2
                    elif bili > 1.2:
                        baseline_sofa += 1
            
            # Cardiovascular (MAP or vasopressors)
            if C_MEANBP in baseline_chart.columns:
                map_value = baseline_chart[C_MEANBP].mean()
                if pd.notna(map_value):
                    if map_value < 70:
                        baseline_sofa += 1
            
            # CNS (Glasgow Coma Scale)
            if C_GCS in baseline_chart.columns:
                gcs_value = baseline_chart[C_GCS].mean()
                if pd.notna(gcs_value):
                    if gcs_value < 6:
                        baseline_sofa += 4
                    elif gcs_value < 10:
                        baseline_sofa += 3
                    elif gcs_value < 13:
                        baseline_sofa += 2
                    elif gcs_value < 15:
                        baseline_sofa += 1
            
            # Renal (Creatinine or Urine Output)
            if C_CREATININE in baseline_lab.columns:
                creat = baseline_lab[C_CREATININE].mean()
                if pd.notna(creat):
                    if creat > 5:
                        baseline_sofa += 4
                    elif creat > 3.5:
                        baseline_sofa += 3
                    elif creat > 2:
                        baseline_sofa += 2
                    elif creat > 1.2:
                        baseline_sofa += 1
            
            # Calculate SOFA scores after infection
            post_infection_window = pd.Timedelta(hours=48)  # Look 48h after infection
            post_start = infection_time
            post_end = infection_time + post_infection_window
            
            post_chart = stay_chart[
                (stay_chart[C_CHARTTIME] >= post_start) & 
                (stay_chart[C_CHARTTIME] <= post_end)
            ]
            post_lab = stay_lab[
                (stay_lab[C_CHARTTIME] >= post_start) & 
                (stay_lab[C_CHARTTIME] <= post_end)
            ]
            
            # Calculate SOFA scores at each timepoint
            timestamps = sorted(pd.concat([
                post_chart[C_CHARTTIME],
                post_lab[C_CHARTTIME]
            ]).unique())
            
            for ts in timestamps:
                window_chart = post_chart[post_chart[C_CHARTTIME] <= ts]
                window_lab = post_lab[post_lab[C_CHARTTIME] <= ts]
                
                current_sofa = 0
                
                # Respiratory (PaO2/FiO2)
                if C_PAO2 in window_chart.columns and C_FIO2_1 in window_chart.columns:
                    pf_ratio = window_chart[C_PAO2].mean() / window_chart[C_FIO2_1].mean()
                    if pd.notna(pf_ratio):
                        if pf_ratio < 100:
                            current_sofa += 4
                        elif pf_ratio < 200:
                            current_sofa += 3
                        elif pf_ratio < 300:
                            current_sofa += 2
                        elif pf_ratio < 400:
                            current_sofa += 1
                
                # Coagulation (Platelets)
                if C_PLATELETS_COUNT in window_lab.columns:
                    platelets = window_lab[C_PLATELETS_COUNT].mean()
                    if pd.notna(platelets):
                        if platelets < 20:
                            current_sofa += 4
                        elif platelets < 50:
                            current_sofa += 3
                        elif platelets < 100:
                            current_sofa += 2
                        elif platelets < 150:
                            current_sofa += 1
                
                # Liver (Bilirubin)
                if C_TOTAL_BILI in window_lab.columns:
                    bili = window_lab[C_TOTAL_BILI].mean()
                    if pd.notna(bili):
                        if bili > 12:
                            current_sofa += 4
                        elif bili > 6:
                            current_sofa += 3
                        elif bili > 2:
                            current_sofa += 2
                        elif bili > 1.2:
                            current_sofa += 1
                
                # Cardiovascular (MAP or vasopressors)
                if C_MEANBP in window_chart.columns:
                    map_value = window_chart[C_MEANBP].mean()
                    if pd.notna(map_value):
                        if map_value < 70:
                            current_sofa += 1
                
                # CNS (Glasgow Coma Scale)
                if C_GCS in window_chart.columns:
                    gcs_value = window_chart[C_GCS].mean()
                    if pd.notna(gcs_value):
                        if gcs_value < 6:
                            current_sofa += 4
                        elif gcs_value < 10:
                            current_sofa += 3
                        elif gcs_value < 13:
                            current_sofa += 2
                        elif gcs_value < 15:
                            current_sofa += 1
                
                # Renal (Creatinine or Urine Output)
                if C_CREATININE in window_lab.columns:
                    creat = window_lab[C_CREATININE].mean()
                    if pd.notna(creat):
                        if creat > 5:
                            current_sofa += 4
                        elif creat > 3.5:
                            current_sofa += 3
                        elif creat > 2:
                            current_sofa += 2
                        elif creat > 1.2:
                            current_sofa += 1
                
                # If SOFA increased by ≥2 points from baseline, this is the onset
                if current_sofa - baseline_sofa >= 2:
                    return ts
            
            # If no SOFA increase ≥2 points found, return infection time
            return infection_time
        
        # If no chart/lab events available, return infection time
        return infection_time
        
    except Exception as e:
        print(f"Error calculating onset for stay_id {stay_id}: {str(e)}")
        return None

def calculate_gcs(chart_events):
    """
    Calculate Glasgow Coma Scale from its components in MIMIC-IV.
    Components:
    - Motor (itemid 227012)
    - Verbal (itemid 227014)
    - Eye (itemid 229813)
    """
    try:
        # Get GCS components
        motor = chart_events[chart_events['itemid'] == 227012]['valuenum']
        verbal = chart_events[chart_events['itemid'] == 227014]['valuenum']
        eye = chart_events[chart_events['itemid'] == 229813]['valuenum']
        
        # Calculate total GCS
        gcs = pd.Series(index=chart_events.index, dtype=float)
        
        # Sum components where available
        if not motor.empty:
            gcs = gcs.add(motor, fill_value=0)
        if not verbal.empty:
            gcs = gcs.add(verbal, fill_value=0)
        if not eye.empty:
            gcs = gcs.add(eye, fill_value=0)
            
        return gcs
        
    except Exception as e:
        print(f"Error calculating GCS: {str(e)}")
        return pd.Series(index=chart_events.index, dtype=float)

def compute_sofa(chart_events, lab_events):
    """
    Calculate SOFA score from chart and lab events.
    SOFA components:
    - Respiratory (PaO2/FiO2)
    - Coagulation (Platelets)
    - Liver (Bilirubin)
    - Cardiovascular (MAP or vasopressors)
    - CNS (Glasgow Coma Scale)
    - Renal (Creatinine or Urine Output)
    """
    try:
        sofa = 0
        
        # Calculate GCS from components
        gcs = calculate_gcs(chart_events)
        
        # Respiratory (PaO2/FiO2)
        if C_PAO2_FIO2 in chart_events.columns:
            pf_ratio = chart_events[C_PAO2_FIO2].mean()
            if pd.notna(pf_ratio):
                if pf_ratio < 100:
                    sofa += 4
                elif pf_ratio < 200:
                    sofa += 3
                elif pf_ratio < 300:
                    sofa += 2
                elif pf_ratio < 400:
                    sofa += 1
        
        # Coagulation (Platelets)
        if C_PLATELET in lab_events.columns:
            platelets = lab_events[C_PLATELET].mean()
            if pd.notna(platelets):
                if platelets < 20:
                    sofa += 4
                elif platelets < 50:
                    sofa += 3
                elif platelets < 100:
                    sofa += 2
                elif platelets < 150:
                    sofa += 1
        
        # Liver (Bilirubin)
        if C_TOTAL_BILI in lab_events.columns:
            bili = lab_events[C_TOTAL_BILI].mean()
            if pd.notna(bili):
                if bili > 12:
                    sofa += 4
                elif bili > 6:
                    sofa += 3
                elif bili > 2:
                    sofa += 2
                elif bili > 1.2:
                    sofa += 1
        
        # Cardiovascular (MAP or vasopressors)
        if C_MEANBP in chart_events.columns:
            map_value = chart_events[C_MEANBP].mean()
            if pd.notna(map_value):
                if map_value < 70:
                    sofa += 1
        
        # CNS (Glasgow Coma Scale)
        if not gcs.empty:
            gcs_value = gcs.mean()
            if pd.notna(gcs_value):
                if gcs_value < 6:
                    sofa += 4
                elif gcs_value < 10:
                    sofa += 3
                elif gcs_value < 13:
                    sofa += 2
                elif gcs_value < 15:
                    sofa += 1
        
        # Renal (Creatinine or Urine Output)
        if C_CREATININE in lab_events.columns:
            creat = lab_events[C_CREATININE].mean()
            if pd.notna(creat):
                if creat > 5:
                    sofa += 4
                elif creat > 3.5:
                    sofa += 3
                elif creat > 2:
                    sofa += 2
                elif creat > 1.2:
                    sofa += 1
        
        return sofa
        
    except Exception as e:
        print(f"Error computing SOFA score: {str(e)}")
        return 0

def compute_pao2_fio2(df, impute_computed=True):
    if C_PAO2 not in df.columns or C_FIO2_1 not in df.columns:
        return pd.NA
    
    result = df[C_PAO2] / df[C_FIO2_1]
    result[np.isinf(result)] = pd.NA
    
    if impute_computed:
        d = np.nanmean(result)
        print("Replacing P/F ratio with average value", d)
        result[pd.isna(result)] = d
    else:
        print("Keeping P/F ratio as null when it can't be computed")
        
    return result
        
def compute_shock_index(df, impute_computed=True):
    # recompute SHOCK INDEX without NAN and INF
    result = df[C_SHOCK_INDEX]
    if C_HR in df.columns and C_SYSBP in df.columns:
        result = df[C_HR] / df[C_SYSBP]
        
    result[np.isinf(result)] = pd.NA
    
    if impute_computed:
        d = np.nanmean(result)
        print("Replacing shock index with average value", d)
        result[pd.isna(result)] = d  # replace NaN with average value ~ 0.8
    else:
        print("Keeping shock index as null when it can't be computed")
        
    return result

def compute_sofa(df, timestep_resolution=4.0, impute_computed=True):
    s = df[[C_PAO2_FIO2, C_PLATELETS_COUNT, C_TOTAL_BILI,
            C_MEANBP, C_MAX_DOSE_VASO, C_GCS, C_CREATININE,
            C_OUTPUT_STEP]]

    s1 = pd.DataFrame(
        [s[C_PAO2_FIO2] > 400, 
        (s[C_PAO2_FIO2] >= 300) & (s[C_PAO2_FIO2] < 400),
        (s[C_PAO2_FIO2] >= 200) & (s[C_PAO2_FIO2] < 300), 
        (s[C_PAO2_FIO2] >= 100) & (s[C_PAO2_FIO2] < 200),
        s[C_PAO2_FIO2] < 100], index=range(5))
    s2 = pd.DataFrame(
        [s[C_PLATELETS_COUNT] > 150, 
        (s[C_PLATELETS_COUNT] >= 100) & (s[C_PLATELETS_COUNT] < 150), 
        (s[C_PLATELETS_COUNT] >= 50) & (s[C_PLATELETS_COUNT] < 100), 
        (s[C_PLATELETS_COUNT] >= 20) & (s[C_PLATELETS_COUNT] < 50), 
        s[C_PLATELETS_COUNT] < 20],
        index=range(5))
    s3 = pd.DataFrame(
        [s[C_TOTAL_BILI] < 1.2, 
        (s[C_TOTAL_BILI] >= 1.2) & (s[C_TOTAL_BILI] < 2), 
        (s[C_TOTAL_BILI] >= 2) & (s[C_TOTAL_BILI] < 6), 
        (s[C_TOTAL_BILI] >= 6) & (s[C_TOTAL_BILI] < 12), 
        s[C_TOTAL_BILI] > 12],
        index=range(5))
    s4 = pd.DataFrame([s[C_MEANBP] >= 70, 
        (s[C_MEANBP] < 70) & (s[C_MEANBP] >= 65), 
        (s[C_MEANBP] < 65), 
        (s[C_MAX_DOSE_VASO] > 0) & (s[C_MAX_DOSE_VASO] <= 0.1), 
        s[C_MAX_DOSE_VASO] > 0.1],
        index=range(5))
    s5 = pd.DataFrame(
        [s[C_GCS] > 14, 
        (s[C_GCS] > 12) & (s[C_GCS] <= 14), 
        (s[C_GCS] > 9) & (s[C_GCS] <= 12), 
        (s[C_GCS] > 5) & (s[C_GCS] <= 9), 
        s[C_GCS] <= 5],
        index=range(5))
    s6 = pd.DataFrame(
        [s[C_CREATININE] < 1.2, 
        (s[C_CREATININE] >= 1.2) & (s[C_CREATININE] < 2), 
        (s[C_CREATININE] >= 2) & (s[C_CREATININE] < 3.5), 
        ((s[C_CREATININE] >= 3.5) & (s[C_CREATININE] < 5)) | (s[C_OUTPUT_STEP] < 500 * timestep_resolution / 24),
        (s[C_CREATININE] > 5) | (s[C_OUTPUT_STEP] < 200 * timestep_resolution / 24)], 
        index=range(5))

    ms1 = s1.idxmax(axis=0)
    ms2 = s2.idxmax(axis=0)
    ms3 = s3.idxmax(axis=0)
    ms4 = s4.idxmax(axis=0)
    ms5 = s5.idxmax(axis=0)
    ms6 = s6.idxmax(axis=0)
    
    result = ms1 + ms2 + ms3 + ms4 + ms5 + ms6
    
    if impute_computed:
        d = np.nanmean(result)
        print("Replacing SOFA with average value", d)
        result[pd.isna(result)] = d
    else:
        print("Keeping SOFA as null when it can't be computed")
        
    return result

def compute_sirs(df, impute_computed=True):
    s = df[[C_TEMP_C, C_HR, C_RR, C_PACO2, C_WBC_COUNT]]

    s1 = (s[C_TEMP_C] >= 38) | (s[C_TEMP_C] <= 36)  # count of points for all criteria of SIRS
    s2 = (s[C_HR] > 90)
    s3 = (s[C_RR] >= 20) | (s[C_PACO2] <= 32)
    s4 = (s[C_WBC_COUNT] >= 12) | (s[C_WBC_COUNT] < 4)
    
    result = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)
    
    if impute_computed:
        d = np.nanmean(result)
        print("Replacing SIRS with average value", d)
        result[pd.isna(result)] = d
    else:
        print("Keeping SIRS as null when it can't be computed")
        
    return result

def compute_sapsii(df):
    """ Calculate the SAPSII score provided the dataframe of raw patient features. """
    age_values = np.array([0, 7, 12, 15, 16, 18])
    hr_values = np.array([11, 2, 0, 4, 7])
    bp_values = np.array([13, 5, 0, 2])
    temp_values = np.array([0, 3])
    o2_values = np.array([11, 9, 6])
    output_values = np.array([11, 4, 0])
    bun_values = np.array([0, 6, 10])
    wbc_values = np.array([12, 0, 3])
    k_values = np.array([3, 0, 3])
    na_values = np.array([5, 0, 1])
    hco3_values = np.array([5, 3, 0])
    bili_values = np.array([0, 4, 9])
    gcs_values = np.array([26, 13, 7, 5, 0])
    
    sapsii = np.zeros((df.shape[0],1))
    
    cols = [
        C_AGE, C_HR, C_SYSBP, C_TEMP_C, C_PAO2_FIO2, C_OUTPUT_STEP, C_BUN,
        C_WBC_COUNT, C_POTASSIUM, C_SODIUM, C_HCO3, C_TOTAL_BILI, C_GCS
    ]
    tt = df[cols]
    
    age = np.array([ tt.iloc[:,0]<40, (tt.iloc[:,0]>=40)&(tt.iloc[:,0]<60), (tt.iloc[:,0]>=60)&(tt.iloc[:,0]<70), (tt.iloc[:,0]>=70)&(tt.iloc[:,0]<75), (tt.iloc[:,0]>=75)&(tt.iloc[:,0]<80), tt.iloc[:,0]>=80 ])
    hr = np.array([ tt.iloc[:,1]<40, (tt.iloc[:,1]>=40)&(tt.iloc[:,1]<70), (tt.iloc[:,1]>=70)&(tt.iloc[:,1]<120), (tt.iloc[:,1]>=120)&(tt.iloc[:,1]<160), tt.iloc[:,1]>=160 ])
    bp = np.array([ tt.iloc[:,2]<70, (tt.iloc[:,2]>=70)&(tt.iloc[:,2]<100), (tt.iloc[:,2]>=100)&(tt.iloc[:,2]<200), tt.iloc[:,2]>=200 ])
    temp = np.array([ tt.iloc[:,3]<39, tt.iloc[:,3]>=39 ])
    o2 = np.array([ tt.iloc[:,4]<100, (tt.iloc[:,4]>=100)&(tt.iloc[:,4]<200), tt.iloc[:,4]>=200 ])
    out = np.array([ tt.iloc[:,5]<500, (tt.iloc[:,5]>=500)&(tt.iloc[:,5]<1000), tt.iloc[:,5]>=1000 ])
    bun = np.array([ tt.iloc[:,6]<28, (tt.iloc[:,6]>=28)&(tt.iloc[:,6]<84), tt.iloc[:,6]>=84 ])
    wbc = np.array([ tt.iloc[:,7]<1, (tt.iloc[:,7]>=1)&(tt.iloc[:,7]<20), tt.iloc[:,7]>=20 ])
    k = np.array([ tt.iloc[:,8]<3, (tt.iloc[:,8]>=3)&(tt.iloc[:,8]<5), tt.iloc[:,8]>=5 ])
    na = np.array([ tt.iloc[:,9]<125, (tt.iloc[:,9]>=125)&(tt.iloc[:,9]<145), tt.iloc[:,9]>=145 ])
    hco3 = np.array([ tt.iloc[:,10]<15, (tt.iloc[:,10]>=15)&(tt.iloc[:,10]<20), tt.iloc[:,10]>=20 ])
    bili = np.array([ tt.iloc[:,11]<4, (tt.iloc[:,11]>=4)&(tt.iloc[:,11]<6), tt.iloc[:,11]>=6 ])
    gcs = np.array([ tt.iloc[:,12]<6, (tt.iloc[:,12]>=6)&(tt.iloc[:,12]<9), (tt.iloc[:,12]>=9)&(tt.iloc[:,12]<11), (tt.iloc[:,12]>=11)&(tt.iloc[:,12]<14), tt.iloc[:,12]>=14 ])
    
    for ii in range(df.shape[0]):
        sapsii[ii] = max(age_values[age[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(o2_values[o2[:,ii]]*df.loc[ii,C_MECHVENT], default=0) + max(output_values[out[:,ii]], default=0) + max(bun_values[bun[:,ii]], default=0) + max(wbc_values[wbc[:,ii]], default=0) + max(k_values[k[:,ii]], default=0) + max(na_values[na[:,ii]], default=0) + max(hco3_values[hco3[:,ii]], default=0) + max(bili_values[bili[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0)
    return sapsii.flatten()

def compute_oasis(df):
    """ Calculate the OASIS score provided the dataframe of raw patient features. """
    age_values = np.array([0, 3, 6, 9, 7])
    bp_values = np.array([4, 3, 2, 0, 3])
    gcs_values = np.array([10, 4, 3, 0])
    hr_values = np.array([4, 0, 1, 3, 6])
    rr_values = np.array([10, 1, 0, 1, 6, 9])
    temp_values = np.array([3, 4, 2, 2, 6])
    output_values = np.array([10, 5, 1, 0, 8])
    vent_value = 9
    
    oasis = np.zeros((df.shape[0],1))
    
    cols = [C_AGE, C_MEANBP, C_GCS, C_HR, C_RR, C_TEMP_C, C_OUTPUT_STEP]
    tt = df[cols]
    
    age = np.array([ tt.iloc[:,0]<24, (tt.iloc[:,0]>=24)&(tt.iloc[:,0]<=53), (tt.iloc[:,0]>53)&(tt.iloc[:,0]<=77), (tt.iloc[:,0]>77)&(tt.iloc[:,0]<=89), tt.iloc[:,0]>89 ])
    bp = np.array([ tt.iloc[:,1]<20.65, (tt.iloc[:,1]>=20.65)&(tt.iloc[:,1]<51), (tt.iloc[:,1]>=51)&(tt.iloc[:,1]<61.33), (tt.iloc[:,1]>=61.33)&(tt.iloc[:,1]<143.44), tt.iloc[:,1]>=143.44 ])
    gcs = np.array([ tt.iloc[:,2]<=7, (tt.iloc[:,2]>7)&(tt.iloc[:,1]<14), tt.iloc[:,1]==14, tt.iloc[:,1]>14 ])
    hr = np.array([ tt.iloc[:,3]<33, (tt.iloc[:,3]>=33)&(tt.iloc[:,3]<89), (tt.iloc[:,3]>=89)&(tt.iloc[:,3]<106), (tt.iloc[:,3]>=106)&(tt.iloc[:,3]<=125), tt.iloc[:,3]>125 ])
    rr = np.array([ tt.iloc[:,4]<6, (tt.iloc[:,4]>=6)&(tt.iloc[:,4]<13), (tt.iloc[:,4]>=13)&(tt.iloc[:,4]<22), (tt.iloc[:,4]>=22)&(tt.iloc[:,4]<30), (tt.iloc[:,4]>=30)&(tt.iloc[:,4]<44), tt.iloc[:,4]>=44 ])
    temp = np.array([ tt.iloc[:,5]<33.22, (tt.iloc[:,5]>=33.22)&(tt.iloc[:,5]<35.93), (tt.iloc[:,5]>=35.93)&(tt.iloc[:,5]<36.89), (tt.iloc[:,5]>=36.89)&(tt.iloc[:,5]<=39.88), tt.iloc[:,5]>39.88 ])
    out = np.array([ tt.iloc[:,6]<671.09, (tt.iloc[:,6]>=671.09)&(tt.iloc[:,6]<1427), (tt.iloc[:,6]>=1427)&(tt.iloc[:,6]<=2514), (tt.iloc[:,6]>2514)&(tt.iloc[:,6]<=6896), tt.iloc[:,6]>6896 ])
    vent = (vent_value*df[C_MECHVENT]).values
    
    for ii in range(df.shape[0]):
        oasis[ii] = max(age_values[age[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(rr_values[rr[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(output_values[out[:,ii]], default=0) + vent[ii]
        
    return oasis.flatten()
