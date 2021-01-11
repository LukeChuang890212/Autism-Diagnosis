#%%
import pandas as pd 

#%%
data = pd.read_excel("data/差異檢定_ANS_SD_F_P_S_V.xlsx").drop(["participantName", "性別","photo_id", "aoi", "AOI_Visit_SD", "Pupil_Size"], axis=1)
column_order = [i for i in range(1,len(data.columns) - 1)]+[0]
data = data[data.columns[column_order]]
data

#%%
groups = dict(ASD = 1, TD = 0)
data["group"] = data["group"].replace(groups)
data

#%%
emos = set(data["emotion"])
for emo in emos:
    emo_data = data[data["emotion"] == emo]
    print(emo_data)
    emo_data.to_csv("data/"+emo+"_data.csv")


