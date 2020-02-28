import enum

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

EXCLUDED = [Data.id_policy, Data.pol_pay_freq, Data.pol_insee_code, Data.commune_code, Data.canton_code, Data.city_district_code, Data.regional_department_code]

# MERGED
VH_MAKE_MODEL = [Data.vh_make, Data.vh_model]
COMMUNE_CANTON_DIST_REG = [Data.commune_code, Data.canton_code, Data.city_district_code, Data.regional_department_code]