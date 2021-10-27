import pandas as pd

d = { "name" : [ "Ellie", "Gayatri", "Rebecca", "Rishab" ],
      "year" : [ "Junior", "Junior", "Senior", "Junior" ],
      "major" : [ "Psychology", "CS", "CS", "MechE" ]
    }
df= pd.DataFrame(d)
print(df)

df["college"]=["A1","A2","A3","A4"]
for index, row in df.iterrows():
    print(index)
    print(row)
    print("Name:", row["name"])
    print()
dffromcsv=pd.read_csv("data\politicaldata.csv")
print(dffromcsv)
