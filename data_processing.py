import enum
import pandas as pd
import numpy as np
from sklearn import preprocessing

class Data(enum.Enum):
  id_policy = 0
  pol_bonus = 1
  pol_coverage = 2
  pol_duration = 3
  pol_sit_duration = 4
  pol_pay_freq = 5
  pol_payd = 6
  pol_usage = 7
  pol_insee_code = 8
  drv_drv2 = 9
  drv_age1 = 10
  drv_age2 = 11
  drv_sex1 = 12
  drv_sex2 = 13
  drv_age_lic1 = 14
  drv_age_lic2 = 15
  vh_age = 16
  vh_cyl = 17
  vh_din = 18
  vh_fuel = 19
  vh_make = 20
  vh_model = 21
  vh_sale_begin = 22
  vh_sale_end = 23
  vh_speed = 24
  vh_type = 25
  vh_value = 26
  vh_weight = 27
  town_mean_altitude = 28
  town_surface_area = 29
  population = 30
  commune_code = 31
  canton_code = 32
  city_district_code = 33
  regional_department_code = 34
  claim_amount = 35
  made_claim = 36

NUMERICAL = [Data.pol_bonus, Data.pol_duration, Data.pol_sit_duration, Data.drv_age1, Data.drv_age2, Data.drv_age_lic1, Data.drv_age_lic2, Data.vh_age, Data.vh_cyl, Data.vh_din, Data.vh_sale_begin, Data.vh_sale_end, Data.vh_speed, Data.vh_value, Data.vh_weight, Data.town_mean_altitude, Data.town_surface_area, Data.population]

ORDINAL = [Data.pol_coverage, Data.pol_usage]

CATEGORICAL = [Data.pol_payd, Data.drv_drv2, Data.drv_sex1, Data.drv_sex2, Data.vh_fuel, Data.vh_type] 

EXCLUDED = [Data.id_policy, Data.pol_pay_freq, Data.pol_insee_code, Data.vh_make, Data.vh_model, Data.commune_code, Data.canton_code, Data.city_district_code, Data.regional_department_code]

def data_analysis():
  dataset = pd.read_csv('part3_training_data.csv')  
  input_dim = 35

  x = dataset.iloc[:,:input_dim]
  y = dataset.iloc[:,input_dim+1:] # not including claim_amount
  X_raw = x.to_numpy()
  data_count = X_raw.shape[0]
  attr_count = X_raw.shape[1]

  print("=== Unfiltered Data ===")
  print("Num of data points: ", data_count)
  print("Num of attributes: ", attr_count)
  print("\n")
  
  THRESHOLD = 0.1

  print("Proportion of missing values (0 / nan) in each attribute: ")
  removed_att = []
  exceed_threshold_count = 0
  has_missing_val = []
  att_missing_count = {}
  for i in range(attr_count):
    count = 0
    for data in X_raw[:,i]:
      if not data or data != data:
        count += 1
    if count > 0:
      if count > THRESHOLD * data_count:
        exceed_threshold_count += 1
        removed_att.append(i)
      else:
        has_missing_val.append(i)
    if count != 0:
      att_missing_count[Data(i).name] = count
      print(Data(i).name, ": ", round(count/data_count*100, 3))
  print("Num of attributes with missing values > ", THRESHOLD*100,"%: ", exceed_threshold_count)
  print("\n")

  print("Proportion of missing values (0 / nan) in each row of data: ")
  exceed_threshold_count = 0
  exceed_threshold_clean = 0
  for row in range(data_count):
    count = 0
    clean_count = 0
    for i in range(X_raw.shape[1]):
      data = X_raw[row][i]
      if not data or data != data:
        count += 1
        if i not in removed_att:
          clean_count += 1
    if count > THRESHOLD * attr_count:
      exceed_threshold_count += 1
    if clean_count > THRESHOLD * attr_count:
      exceed_threshold_clean += 1
  print("Num of rows with missing values >", THRESHOLD, "%: ", exceed_threshold_count)
  print("Num of rows with missing values >", THRESHOLD, "%", "in attributes that are not removed: ", exceed_threshold_clean)

  print("\n")
  print("=== Filtered Data ===")

  print("Data Types: ")
  print("Ordinal attributes: ")
  for data in ORDINAL:
    print(data.name, ": ", type(X_raw[1][data.value]))
  print("CATEGORICAL attributes: ")
  for data in CATEGORICAL:
    print(data.name, ": ", type(X_raw[1][data.value]))
  print("NUMERICAL attributes: ")
  for data in NUMERICAL:
    print(data.name, ": ", type(X_raw[1][data.value]))
  print("\n")


  print("Attributes chosen to be excluded: ")
  for data in EXCLUDED:
    print(data.name)
  print("\n")
  print("Attributes excluded due to too many missing values: ")
  for i in removed_att:
    print(Data(i).name)

  print("\n")
  print("Num of missing values that have missing values and are not filtered: ")
  for i in has_missing_val:
    if i not in removed_att and i not in [e.value for e in EXCLUDED]:
      if isinstance(X_raw[0][i], (float, int)):
        print(Data(i).name, ":", att_missing_count[Data(i).name])



# data_analysis()