from model_classification import *

# Choose the models you want to run
RUN_NET = True
RUN_NET_WH = True
RUN_NET_WH_AL = True

# You can modify the number of tests for each model
rangetest = 10

listmodels = []
if RUN_NET:
    listmodels.append(Net)
if (RUN_NET_WH):
    listmodels.append(Net_wh)
if (RUN_NET_WH_AL):
    listmodels.append(Net_wh_al)

run_tests(rangetest, listmodels)
