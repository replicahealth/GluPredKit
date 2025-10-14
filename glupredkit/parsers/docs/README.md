# Data Parsing 

The core dataset should consist of:
- date (index): local datetime
- date_utc: utc datetime 
- CGM [mg/dL]: Glucose values
- bolus [U]: total bolus delivery within the previous five minutes
- basal [U/h]: basal rate within the previous five minutes
- insulin_type: type of insulin used
- carbs [grams]: total carbohydrates intake within the previous five minutes
- is_test (bool): indication of whether it is training or test data
- id (str): subject id indication

- optionals:
  - steps
  - calories burned
  - workout_label
  - workout_intensity




Output is a dataframe, where each subject has data sorted by ascending dates of 5-minute intervals. 




## T1Dexi


To do: 
- Validate the handling of heartrate data
- Add steps data
- Add tests


## DiaTrend

### Limitations and Assumptions 

- Only a subset of 17 subjects have basal data because of an error that happened during data collection. Hence, we only include these subjects in the processing. 
- The age is noted in the dataset as an interval of 10 years. Since our dataset dictionary defines the age as a numeric type, we have used the midpoint value, because we argue that it is more important to preserve the information that the subject is an adult, rather than having the exact age value. 
- It is assumed that subjects using MiniMed 670G is using that with the SmartGuard function active 




### General To Dos

- Add a snapshot of a few rows for each dataset to show the dataset format
- Improve the API for adding ice and iob values, using the loop to python api (with insulin type if available!)
- Document processing decisions and assumptions made
- For each dataset, add a separate table user_data, with information about total daily insulin, demographics, therapy information, etc.











