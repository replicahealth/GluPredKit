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




### T1Dexi


To do: 
- Validate the handling of heartrate data
- Add steps data
- Add tests




### General To Dos

- Add a snapshot of a few rows for each dataset to show the dataset format
- Improve the API for adding ice and iob values, using the loop to python api (with insulin type if available!)
- Document processing decisions and assumptions made
- For each dataset, add a separate table user_data, with information about total daily insulin, demographics, therapy information, etc.











