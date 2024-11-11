#load package
import pandas as pd
import numpy as np
from missingpy import MissForest #impute missing value
from datetime import timedelta

def merge_data(dailyactivity, wear_time, dailyFitbitActiveZoneMinutes, dailyHRV, fitbitBreathingRate, fitbitSkinTemperature, sleepDay, sleepStageLogInfo, heartrate_15min):
    #rename data file
    dailyactivity.rename(columns = {'ActivityDate':'date'}, inplace=True)
    wear_time.rename(columns = {'Day':'date'}, inplace=True)
    dailyFitbitActiveZoneMinutes.rename(columns = {'Date':'date'}, inplace=True)
    dailyHRV.rename(columns = {'SleepDay':'date'}, inplace=True)
    fitbitBreathingRate.rename(columns = {'SleepDay':'date'}, inplace = True)
    fitbitSkinTemperature.rename(columns = {'SleepDay':'date'}, inplace=True)
    sleepDay.rename(columns = {'SleepDay':'date'}, inplace=True)
    sleepStageLogInfo.rename(columns = {'SleepDay':'date'}, inplace=True)
    heartrate_15min.rename(columns = {'Time':'date'}, inplace=True)

    #Combine to one dataset
    total_data = [wear_time, dailyactivity, dailyFitbitActiveZoneMinutes, dailyHRV, fitbitBreathingRate, 
                fitbitSkinTemperature, sleepDay, sleepStageLogInfo, heartrate_15min]

    for d in total_data:
        d['date'] = pd.to_datetime(d['date'], errors = 'coerce')
        d['date'] = d['date'].dt.date #.strftime('%m/%d/%Y').str.replace('/0', '/')

    #transfer heart_rate_15 min
    heartrate_15min = heartrate_15min.groupby(by = ['Id','date']).mean().reset_index()

    #merge file
    merged1 = pd.merge(wear_time, dailyactivity, on=['Id','date'], how='left')  # inner join only keeps common dates
    merged2 = pd.merge(merged1, sleepDay, on=['Id','date'], how='left')
    merged3 = pd.merge(merged2, sleepStageLogInfo, on=['Id','date'], how='left')
    merged4 = pd.merge(merged3, heartrate_15min, on=['Id','date'], how='left')
    merged5 = pd.merge(merged4, dailyFitbitActiveZoneMinutes, on=['Id','date'], how='left')
    merged6 = pd.merge(merged5, dailyHRV, on=['Id','date'], how='left')
    merged7 = pd.merge(merged6, fitbitBreathingRate, on=['Id','date'], how='left')
    merged_final = pd.merge(merged7, fitbitSkinTemperature, on=['Id','date'], how='left')

    #filter 
    merged_final_10 = merged_final[(merged_final['TotalMinutesWearTime'] >= 10*60)] #10 hour, what's the meaning of four days a week, is this for calendar week or four consecutive days
    merged_0_3000steps = merged_final[(merged_final['TotalMinutesWearTime'] == 0) & (merged_final['TotalSteps'] >= 3000)]
    merged_final = pd.concat([merged_final_10, merged_0_3000steps])
    merged_final = merged_final.sort_values(by=['Id', 'date'])

    #rename ID
    merged_final['Id'] = [items[-3:] for items in merged_final['Id']]
    merged_final['Id'] = [int(digit) for digit in merged_final['Id']]

    return merged_final

def select_survey_data(survey_data):
    #Proprocessing survey data
    survey_data['health_coach_survey_complete'].value_counts()

    #select people who complete health coach session and the second follow-up
    survey_data_clean = survey_data[survey_data['weeks_followup_survey_96ac_complete'] == 2] #87 participants

    #select variables from survey
    uses_vars_of_survey = ['record_id','current_status',
    'demographics_age','demographics_sex','demographics_sexorient','demographics_ethnicity','demographics_immigration',
    'demographics_race___1', 'demographics_race___2', 'demographics_race___3', 'demographics_race___4', 'demographics_race___5', 
    'demographics_race___6', 'demographics_race___7', 'demographics_education', 'demographics_sorority',
    'nervous_v1', 'down_v1', 'calm_v1', 'blue_v1', 'happy_v1',
    'nervous_v2', 'down_v2', 'calm_v2', 'blue_v2', 'happy_v2',
    'weeks_followup_survey_complete','weeks_followup_survey_96ac_complete',
    'weeks_followup_survey_timestamp', 'weeks_followup_survey_96ac_timestamp'
    ]
    select_survey = survey_data_clean[uses_vars_of_survey] #have the demongraphic information

    #rename the name of record_id
    select_survey.rename(columns = {"record_id":'Id'}, inplace=True)

    #reframe the date time format
    select_survey['weeks_followup_survey_timestamp'] = pd.to_datetime(select_survey['weeks_followup_survey_timestamp'], errors = 'coerce' )
    select_survey['weeks_followup_survey_timestamp'] = select_survey['weeks_followup_survey_timestamp'].dt.date
    select_survey['weeks_followup_survey_96ac_timestamp'] = pd.to_datetime(select_survey['weeks_followup_survey_96ac_timestamp'], errors = 'coerce' )
    select_survey['weeks_followup_survey_96ac_timestamp'] = select_survey['weeks_followup_survey_96ac_timestamp'].dt.date
    return select_survey

# 149    151

def impute_missing(data):
    #impute missing value
    total_ids = data['Id'].unique()
    impute_data_frame = pd.DataFrame()

    for i in total_ids:
        imputed_data = data[data['Id'] == i]
        date = imputed_data['date']
        imputed_data_date = imputed_data.set_index('date')
        
        # Drop columns that have all values missing
        non_missing_columns = imputed_data_date.dropna(axis=1, how='all')
        
        # Apply MissForest if there are remaining columns with data
        if not non_missing_columns.empty:
            mf = MissForest()
            mf.fit(non_missing_columns)
            imputed_data_transformed = pd.DataFrame(mf.transform(non_missing_columns), index=non_missing_columns.index, columns=non_missing_columns.columns)
            
            # Re-add the dropped columns as all NaNs
            for col in imputed_data_date.columns:
                if col not in imputed_data_transformed.columns:
                    imputed_data_transformed[col] = 0  # Or any other fill method, like 0

            # Reorder columns to match the original order
            imputed_data_transformed = imputed_data_transformed[imputed_data_date.columns]
            
            # Reorganize the data with date column
            last_data_frame = pd.concat([imputed_data_transformed.reset_index(), date.reset_index(drop=True)], axis=1)

            # Append to the final DataFrame
            impute_data_frame = pd.concat([impute_data_frame, last_data_frame])

    # Check the result
    return impute_data_frame

def extend_time(impute_data_frame, select_survey):
    time_sequency = impute_data_frame.copy()
    all_fitbit_id = time_sequency['Id'].unique()
    new_data_frame = pd.DataFrame()

    for i in range(len(all_fitbit_id)):

        fitbit_id = all_fitbit_id[i]

        sorted_individual = time_sequency[time_sequency['Id'] == fitbit_id].sort_values(by = 'date')
        
        #outcome variable
        individual_survey = select_survey[select_survey['Id'] == fitbit_id]
        followup1_time = individual_survey['weeks_followup_survey_timestamp'].iloc[0] if not individual_survey['weeks_followup_survey_timestamp'].empty else pd.NaT
        followup2_time = individual_survey['weeks_followup_survey_96ac_timestamp'].iloc[0] if not individual_survey['weeks_followup_survey_96ac_timestamp'].empty else pd.NaT
        
        
        start_date = sorted_individual['date'].iloc[0]
        end_date = followup2_time - timedelta(days = 1) # the previous one day of completing the second follow-up, the end day data
        
        full_date_range = pd.date_range(start=start_date, end=end_date)

        sorted_individual.set_index('date', inplace=True)
        
        sorted_individual_reindexed = sorted_individual.reindex(full_date_range).reset_index() #.fillna(0) #reset the wrong index, I need to handle the missing value before fill 0
        sorted_individual_reindexed['Id'] = fitbit_id
        sorted_individual_reindexed.rename(columns = {'index':'date'}, inplace = True)

        new_data_frame = pd.concat([new_data_frame, sorted_individual_reindexed])
    return new_data_frame

def assign_survey_date(time, followup1_time, followup2_time):
    if time < followup1_time:
        return 1
    elif time == followup1_time:
        return 2
    elif followup1_time < time < followup2_time:
        return 3
    elif time == followup2_time:
        return 4
    
    return None

def recode_survey_time(new_data_frame, select_survey):
    new_used_fitbit_data = new_data_frame.copy()
    all_fitbit_id = new_used_fitbit_data['Id'].unique()

    for i in range(len(all_fitbit_id)):

        fitbit_id = all_fitbit_id[i]
        #outcome variable
        individual_survey = select_survey[select_survey['Id'] == fitbit_id]
        
        followup1_time = individual_survey['weeks_followup_survey_timestamp'].iloc[0] if not individual_survey['weeks_followup_survey_timestamp'].empty else pd.NaT
        #print(followup1_time)
        followup2_time = individual_survey['weeks_followup_survey_96ac_timestamp'].iloc[0] if not individual_survey['weeks_followup_survey_96ac_timestamp'].empty else pd.NaT
        followup2_time = followup2_time - timedelta(days=1)
        #print(followup2_time)
        
        if pd.isna(followup1_time) or pd.isna(followup2_time):
            print(f"missing follow-up times for Id{fitbit_id}")
            continue
        #predictors
        individual_fitbit = new_used_fitbit_data[new_used_fitbit_data['Id'] == fitbit_id].copy()

        individual_fitbit['survey_date'] = individual_fitbit.apply(
            lambda row: assign_survey_date(row['date'].date(), followup1_time, followup2_time), axis=1 
            ) #
        new_used_fitbit_data.loc[new_used_fitbit_data['Id'] == fitbit_id, 'survey_date'] = individual_fitbit['survey_date'] #replace with new define
    
    return new_used_fitbit_data

def add_gaussian_noise(time_series, mean=0.0, stddev=1.0):
    """
    Adds Gaussian noise to a time series.
     Options:
     time_series (array-like): A time series to which noise is added.
     mean (float): The average value of the noise. Default is 0.0.
     stddev (float): Standard deviation of noise. Default is 1.0.

     Returns:
     noisy_series (np.array): Time series with added noise.
     """
     # Gaussian noise generation
    noise = np.random.normal(mean, stddev, len(time_series))

    # Adding noise to the original time series
    noisy_series = time_series + noise

    return noisy_series