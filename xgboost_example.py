# Project name : XGBoost fast model eval library in Odin
#
# Description  : This is a library to evaluate XGBoost models in odin without
#                dependencies .
#                The example model that I provide generates a model for 13 input
#                variables of Boston Housing Prices and as 100 trees.
#                It evaluates / predicts in 2.45 micro-seconds on a AMD Ryzen 4700G
#                65 Watt's.
#                So it's very fast!
#                It reads and generates the internal representation of the 228 KByte
#                model from a Python text model dump file in 1.8 ms.
#                After that it makes any number of prediction in 2.45 micro-seconds,
#                after the first evaluation to make the fill the caches with the model
#                data.
# 
#                The output of the example program is as follows:
#
#                ``` 
#                ./xgboost_fast_model_eval.exe
#                xgboost fast model evaluator in Odin begin...
#                Execution duration xme.load_model_from_txt() : 1.896404ms  
#
#                target_predicted_value_correct_1 : 24.019
#                Execution duration xme.xgb_predict() [1] : 3.67µs  
#                predicted_value_1 : 24.019
#
#                target_predicted_value_correct_2 : 41.766
#                Execution duration xme.xgb_predict() [2] : 2.45µs 
#                predicted_value_2 : 41.766
#
#                xgboost fast model evaluator in Odin end...
#                ```
#
# Author       : João Nuno Carvalho
# License      : MIT Open License
# Date         : 2024.01.01


# NOTE: The code below is a copy odf the last example of the following link
#       That I then adapted to my needs.
#       So I could make a model dump and then read it in my Odin code.
#
# From:
#   https://machinelearningmastery.com/xgboost-for-regression/
#

# fit a final xgboost model on the housing dataset and make a prediction
from numpy import asarray
from pandas import read_csv
from xgboost import XGBRegressor
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split dataset into input and output columns
X, y = data[:, :-1], data[:, -1]
# define model
model = XGBRegressor()
# fit model
model.fit(X, y)
# define new data
# Original data
# row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]

# Invented data
row = [0.10632,19.00,3.310,0,0.5380,7.5750,66.20,5.0900,1,297.0,16.30,397.90,3.98]
new_data = asarray([row])
# make a prediction
yhat = model.predict(new_data)
# summarize prediction
print('Predicted: %.3f' % yhat)


# jnc
model_dump = model.get_booster().get_dump()

model_dump_as_string = []
# Append all trees to a single string.
for index, tree_str in enumerate( model_dump ):
#    model_dump_as_string.append( "#" + str(index) + "\n" )
    model_dump_as_string.append( tree_str )


model_dump_as_string = "".join( model_dump_as_string )
# print line by line.
#for tree in model_dump_as_string:
#    print(tree)
#    print()


#  print( model_dump_as_string )

# write a entire file to disk.
with open(".//model_xgboost.txt", 'w') as file:
    file.write(model_dump_as_string)

print( "Model written to disk! \n .//model_xgboost.txt\n" )
