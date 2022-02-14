import pandas as pd
from os import listdir

# # Bob waters his lawn every Tuesday, as long as it didn't rain within two days before.  If it did he will wait until it has been two days since his lawn has recieved water.  If it is cloudy it might have rained.  If the grass is wet it may have been watered or it might be dew.
# df = pd.DataFrame([['Cloudy','Sunday',True],
#                    ['Sunny','Monday',False],
#                    ['Sunny','Tuesday',True],
#                    ['Sunny','Wednesday',False],
#                    ['Sunny','Thursday',False],
#                    ['Sunny','Friday',True],
#                    ['Rain','Saturday',True],
#                    ['Rain','Sunday',True],
#                    ['Sunny','Monday',False],
#                    ['Sunny','Tuesday',True],
#                    ['Cloudy','Wednesday',False],
#                    ['Sunny','Thursday',False],
#                    ['Sunny','Friday',False],
#                    ['Cloudy','Saturday',True],
#                    ['Cloudy','Sunday',False],
#                    ['Rain','Monday',True],
#                    ['Sunny','Tuesday',False],
#                    ['Sunny','Wednesday',True],
#                    ['Cloudy','Thursday',True],
#                    ['Sunny','Friday',False],
#                    ['Sunny','Saturday',False],
#                    ['Cloudy','Sunday',True],
#                    ], columns=['Weather','Day','GrassWet'])
# df.to_csv('../../data/grass_wet.csv',index=False)

data = pd.read_csv('../../data/grass_wet.csv')


class NaiveBayes:
    def __init__(self,data,lapace_smoothing=1):
        self.data = data

    def predict(X):
        # for each class c, calc p(X|c)p(c)/p(X)
        #                   calc log(p(X|c)p(c)/p(X))
        #                   calc log(p(X|c)p(c)/p(X))
        pass
