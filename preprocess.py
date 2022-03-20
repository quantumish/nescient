
import pandas as pd
import sys
import math

data = pd.read_csv(sys.argv[1])

frontal = data[data["Frontal/Lateral"] == "Frontal"]

frontal_pos = frontal[frontal["Lung Lesion"] == 1.0].sample(frac=1)
frontal_neg = frontal[frontal["Lung Lesion"] == -1.0].sample(frac=1)
frontal_neu = frontal[frontal["Lung Lesion"].isnull()].sample(frac=1)

pd.concat(
    [frontal_pos,
     frontal_neg,
     frontal_neu.head(
         frontal_pos.shape[0]-frontal_neg.shape[0]
     )]
).to_csv("all.csv")
