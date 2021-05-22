import numpy as np
import pandas as pd

# data = pd.read_csv('./csv/raw_inter.csv')
# data = data.drop_duplicates(['X','Y'])
# data.to_csv("./csv/inter.csv")

# data = pd.read_csv('./csv/grid_inter.csv')
# data = data[['Y','X','OBJECTID','SHAPE_Length']]
# data.to_csv("./csv/grid_inter.csv")

data = pd.read_csv('./csv/vertices.csv')
data = data[['Y','X','OBJECTID','left','top','right','bottom','SHAPE_Length','vertex_index','distance','angle','IMAGE']]
data.to_csv("./csv/vertices.csv")


# data = pd.read_csv('./csv/grid.csv')
# data['IMAGE'] = data['IMAGE'].apply(lambda x: str(x).zfill(6))
# data.to_csv("./csv/grid.csv")