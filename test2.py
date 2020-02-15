import pandas as pd
from part2_claim_classifier import *

X_raw = pd.read_csv("part2_training_data.csv", usecols=["drv_age1","vh_age","vh_cyl","vh_din","pol_bonus","vh_sale_begin","vh_sale_end","vh_value","vh_speed","claim_amount"])
y_raw = pd.read_csv("part2_training_data.csv", usecols=["made_claim"])
print(X_raw)
print(y_raw)

cc = ClaimClassifier()


# cc.fit()