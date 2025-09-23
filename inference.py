##----------Region: Install and imports-----------------------##
import sys
import subprocess

# Installing the same version of sklean, used during the development
# subprocess.check_call([sys.executable, "-m", "conda", "install", "scikit-learn=0.24.1"])
# subprocess.check_call([sys.executable, "-m", "conda", "install","joblib"])
# subprocess.check_call([sys.executable, "-m", "pip", "install","lightgbm"])


import joblib
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm
from lightgbm import LGBMRegressor
import pandas as pd
import argparse
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import boto3
import gc
import io
from io import StringIO
import os
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

pd.set_option('display.max_columns', None)


##-----------Region: Global variables---------------------## 

# bucket_name = 'cbr-cohort-experiments'
# processed_data_prefix ='KPIs_Inference_Processed/'
# models_prefix =''

# kpi_processed_file = 'KPIs_data_processed' + '_' + str(date.today()) + '.csv'


def prediction_on_data(backtesting_model, data, args, dict_models): 
    
    df_to_predict = data[data['backtesting_limit'] == backtesting_model].reset_index().drop(columns = 'index')
    df = df_to_predict.copy()
    
    model_1 = lightgbm.Booster(model_file = os.path.join(args.models_path ,dict_models[backtesting_model][0]))
    model_5 = lightgbm.Booster(model_file = os.path.join(args.models_path ,dict_models[backtesting_model][1]))
    model_8 = lightgbm.Booster(model_file = os.path.join(args.models_path ,dict_models[backtesting_model][2]))

    feature_cols_1 = model_1.feature_name()
    feature_cols_5 = model_5.feature_name()
    feature_cols_8 = model_8.feature_name()
    
    if backtesting_model in [30, 60, 90, 180, 270]:
    
        X_1 = df[feature_cols_1]
        X_5 = df[feature_cols_5]
        X_8 = df[feature_cols_8]

    else:

        col_720 = [col for col in df.columns if '_720' in col]
        col_630 = [col for col in df.columns if '_630' in col]
        col_540 = [col for col in df.columns if '_540' in col]
        col_450 = [col for col in df.columns if '_450' in col]

        col_450_540_630_720 = col_720+col_630+col_540+col_450
        col_540_630_720 = col_720+col_630+col_540
        col_630_720 = col_720+col_630

        # df.loc[(df['cohort_age_backtesting']==360),col_450_540_630_720] = np.nan
        # df.loc[(df['cohort_age_backtesting']==450),col_540_630_720] = np.nan
        # df.loc[(df['cohort_age_backtesting']==540),col_630_720] = np.nan
        # df.loc[(df['cohort_age_backtesting']==630),col_720] = np.nan

        df.loc[(df['cohort_age_backtesting']==360),col_450_540_630_720] =  -99
        df.loc[(df['cohort_age_backtesting']==450),col_540_630_720] =  -99
        df.loc[(df['cohort_age_backtesting']==540),col_630_720] =  -99
        df.loc[(df['cohort_age_backtesting']==630),col_720] = -99

        X = df.copy()
        X_1 = df[feature_cols_1]
        #X_5 = X.rename(columns = {'avg_upfront_price_usd':'avg_upfront_price', 'avg_unlock_price_usd':'avg_unlock_price'})
        X_5 = df[feature_cols_5]
        X_8 = df[feature_cols_8]
    
    # else:
        
    #     X = df.copy()
    #     X_1 = df[feature_cols_1]
    #     X_5 = X.rename(columns = {'avg_upfront_price_usd':'avg_upfront_price', 'avg_unlock_price_usd':'avg_unlock_price'})
    #     X_5 = X_5[feature_cols_5]
    #     X_8 = df[feature_cols_8]
        
    #y_pred = model.predict(X)
    classifiers = {}
    pred_1 = pd.DataFrame(model_1.predict(X_1), columns = [str(0.1)])
    pred_5 = pd.DataFrame(model_5.predict(X_5), columns = [str(0.5)])
    pred_8 = pd.DataFrame(model_8.predict(X_8), columns = [str(0.8)])
    classifiers[str(0.1)] = {'clf': model_1, 'predictions': pred_1}
    classifiers[str(0.5)] = {'clf': model_5, 'predictions': pred_5}
    classifiers[str(0.8)] = {'clf': model_8, 'predictions': pred_8}

    pred_data = pd.DataFrame({'0.1': classifiers['0.1']['predictions']['0.1'],
    '0.5': classifiers['0.5']['predictions']['0.5'],                         
    '0.8': classifiers['0.8']['predictions']['0.8'],'days': backtesting_model})
    
    return pred_data


def fix_predictions(df_to_predict, y_pred_test):
    
    df_to_predict['prediction_0.1'] = y_pred_test['0.1']
    df_to_predict['prediction_0.5'] = y_pred_test['0.5']
    df_to_predict['prediction_0.8'] = y_pred_test['0.8']
    
    
    df_to_predict.loc[df_to_predict['prediction_0.1']<0,'prediction_0.1'] =0
    df_to_predict.loc[df_to_predict['prediction_0.1']>1,'prediction_0.1'] =1
    df_to_predict.loc[df_to_predict['prediction_0.5']<0,'prediction_0.5'] =0
    df_to_predict.loc[df_to_predict['prediction_0.5']>1,'prediction_0.5'] =1
    df_to_predict.loc[df_to_predict['prediction_0.8']<0,'prediction_0.8'] =0
    df_to_predict.loc[df_to_predict['prediction_0.8']>1,'prediction_0.8'] =1
    
    # Ensuring predicted FRR is never less than the last observed FRR
    
    if df_to_predict['backtesting_limit'][0] == 360:
        
        df_to_predict.loc[(df_to_predict['age_in_months']==360) & (df_to_predict['frr_360']>df_to_predict['prediction_0.5']),'prediction_0.5'] = df_to_predict['frr_360']
        df_to_predict.loc[(df_to_predict['age_in_months']==450) & (df_to_predict['frr_450']>df_to_predict['prediction_0.5']),'prediction_0.5'] = df_to_predict['frr_450']
        df_to_predict.loc[(df_to_predict['age_in_months']==540) & (df_to_predict['frr_540']>df_to_predict['prediction_0.5']),'prediction_0.5'] = df_to_predict['frr_540']
        df_to_predict.loc[(df_to_predict['age_in_months']==630) & (df_to_predict['frr_630']>df_to_predict['prediction_0.5']),'prediction_0.5'] = df_to_predict['frr_630']
        df_to_predict.loc[(df_to_predict['age_in_months']==720) & (df_to_predict['frr_720']>df_to_predict['prediction_0.5']),'prediction_0.5'] = df_to_predict['frr_720']
    
    else:
        
        df_to_predict.loc[df_to_predict['frr_'+str(y_pred_test['days'][0])]>df_to_predict['prediction_0.5'],'prediction_0.5'] = df_to_predict['frr_'+str(y_pred_test['days'][0])]
    
    df_to_predict['predicted_revenue_USD_0.1'] = np.round(df_to_predict['prediction_0.1']*df_to_predict['total_follow_on_revenue_current_usd'],2)
    df_to_predict['predicted_revenue_USD_0.5'] = np.round(df_to_predict['prediction_0.5']*df_to_predict['total_follow_on_revenue_current_usd'],2)
    df_to_predict['predicted_revenue_USD_0.8'] = np.round(df_to_predict['prediction_0.8']*df_to_predict['total_follow_on_revenue_current_usd'],2)

    return df_to_predict


def format_predictions(df_to_predict):
    
    df_to_predict = df_to_predict.rename(columns={'prediction_0.1':'frr_prediction_10','prediction_0.5':'frr_prediction_50','prediction_0.8':'frr_prediction_80',
                                       'predicted_revenue_USD_0.1':'predicted_revenue_3_years_10','predicted_revenue_USD_0.5':'predicted_revenue_3_years_50',
                                       'predicted_revenue_USD_0.8':'predicted_revenue_3_years_80'})
    
    df_to_predict.loc[:,"date_uploaded"] = datetime.today().strftime("%Y-%m-%d")
    df_to_predict["pred_prim_key"] = df_to_predict['accounts_group']+"_"+str(df_to_predict['backtesting_limit'][0])
    df_to_predict = df_to_predict.rename(columns = {'backtesting_limit':'backtesting_unit_age_days'})
    
    df_to_predict = df_to_predict[['pred_prim_key', 'accounts_group', 'count_units', 'reg_month',
       'country', 'area', 'primary_product', 'product_group',
       'backtesting_unit_age_days', 'frr_prediction_10', 'frr_prediction_50',
       'frr_prediction_80', 'predicted_revenue_3_years_10',
       'predicted_revenue_3_years_50', 'predicted_revenue_3_years_80',
       'total_follow_on_revenue_current_usd','date_uploaded']]
    
    return df_to_predict


##------------Region: Main---------------------------------##


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="./data/")
    parser.add_argument('--models_path', type=str, default="./models/")
    parser.add_argument('--output_path', type=str, default="./output/")
    parser.add_argument('--output_file_predictions', type=str, default = 'predictions')

    # parser.add_argument("--model_file_30_1", type=str)
    # parser.add_argument("--model_file_30_5", type=str)
    # parser.add_argument("--model_file_30_8", type=str)
    # parser.add_argument("--model_file_60_1", type=str)
    # parser.add_argument("--model_file_60_5", type=str)
    # parser.add_argument("--model_file_60_8", type=str)
    # parser.add_argument("--model_file_90_1", type=str)
    # parser.add_argument("--model_file_90_5", type=str)
    # parser.add_argument("--model_file_90_8", type=str)
    # parser.add_argument("--model_file_180_1", type=str)
    # parser.add_argument("--model_file_180_5", type=str)
    # parser.add_argument("--model_file_180_8", type=str)
    # parser.add_argument("--model_file_270_1", type=str)
    # parser.add_argument("--model_file_270_5", type=str)
    # parser.add_argument("--model_file_270_8", type=str)
    parser.add_argument("--model_file_360_1", type=str)
    parser.add_argument("--model_file_360_5", type=str)
    parser.add_argument("--model_file_360_8", type=str)

    args = parser.parse_args()
    args = parser.parse_args()
    print('Inside Main....................')
    # print("Getting data from s3")
    # bucket = boto3.resource('s3').Bucket(bucket_name)
    
    # Download KPIs(input) data and models

    # model_file_30_1 = args.model_file_30_1
    # model_file_30_5 = args.model_file_30_5
    # model_file_30_8 = args.model_file_30_8
    # model_file_60_1 = args.model_file_60_1
    # model_file_60_5 = args.model_file_60_5
    # model_file_60_8 = args.model_file_60_8
    # model_file_90_1 = args.model_file_90_1
    # model_file_90_5 = args.model_file_90_5
    # model_file_90_8 = args.model_file_90_8
    # model_file_180_1 = args.model_file_180_1
    # model_file_180_5 = args.model_file_180_5
    # model_file_180_8 = args.model_file_180_8
    # model_file_270_1 = args.model_file_270_1
    # model_file_270_5 = args.model_file_270_5
    # model_file_270_8 = args.model_file_270_8
    model_file_360_1 = args.model_file_360_1
    model_file_360_5 = args.model_file_360_5
    model_file_360_8 = args.model_file_360_8

    
    # dict_models = {30: [model_file_30_1, model_file_30_5, model_file_30_8], 60: [model_file_60_1, model_file_60_5, model_file_60_8],
    #                90: [model_file_90_1, model_file_90_5, model_file_90_8], 180: [model_file_180_1, model_file_180_5, model_file_180_8],
    #                270: [model_file_270_1, model_file_270_5, model_file_270_8], 360: [model_file_360_1, model_file_360_5, model_file_360_8]}
    
    dict_models = {360: [model_file_360_1, model_file_360_5, model_file_360_8]}

    # bucket.download_file(models_prefix+model_file_30_1,os.path.join(args.input_path,model_file_30_1))
    # bucket.download_file(models_prefix+model_file_30_5,os.path.join(args.input_path,model_file_30_5))
    # bucket.download_file(models_prefix+model_file_30_8,os.path.join(args.input_path,model_file_30_8))
    # bucket.download_file(models_prefix+model_file_60_1,os.path.join(args.input_path,model_file_60_1))
    # bucket.download_file(models_prefix+model_file_60_5,os.path.join(args.input_path,model_file_60_5))
    # bucket.download_file(models_prefix+model_file_60_8,os.path.join(args.input_path,model_file_60_8))
    # bucket.download_file(models_prefix+model_file_90_1,os.path.join(args.input_path,model_file_90_1))
    # bucket.download_file(models_prefix+model_file_90_5,os.path.join(args.input_path,model_file_90_5))
    # bucket.download_file(models_prefix+model_file_90_8,os.path.join(args.input_path,model_file_90_8))
    # bucket.download_file(models_prefix+model_file_180_1,os.path.join(args.input_path,model_file_180_1))
    # bucket.download_file(models_prefix+model_file_180_5,os.path.join(args.input_path,model_file_180_5))
    # bucket.download_file(models_prefix+model_file_180_8,os.path.join(args.input_path,model_file_180_8))
    # bucket.download_file(models_prefix+model_file_270_1,os.path.join(args.input_path,model_file_270_1))
    # bucket.download_file(models_prefix+model_file_270_5,os.path.join(args.input_path,model_file_270_5))
    # bucket.download_file(models_prefix+model_file_270_8,os.path.join(args.input_path,model_file_270_8))
    # bucket.download_file(models_prefix+model_file_360_1,os.path.join(args.input_path,model_file_360_1))
    # bucket.download_file(models_prefix+model_file_360_5,os.path.join(args.input_path,model_file_360_5))
    # bucket.download_file(models_prefix+model_file_360_8,os.path.join(args.input_path,model_file_360_8))
    
    # bucket.download_file(processed_data_prefix+kpi_processed_file,os.path.join(args.input_path,kpi_processed_file))
    
    print("All data saved in {}".format(os.path.join(args.input_path)))
    
    print("Reading inference data...")
    kpi_processed_file = 'KPIs_data_processed_2025-09-01.csv'
    data = pd.read_csv(os.path.join(args.input_path,kpi_processed_file),low_memory=False)
    
    print("Starting multiprocessing parallel model predictions...")
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # backtesting_model = [30, 60, 90, 180, 270, 360]
        backtesting_model = [360]
        results = [executor.submit(prediction_on_data, backtesting_days, data, args, dict_models) for backtesting_days in backtesting_model]
        print("Completed multiprocessing parallel model predictions...")
        
    for f in concurrent.futures.as_completed(results):
        
        pred_data = pd.DataFrame(data = f.result())
        df = data[data['backtesting_limit'] == pred_data['days'][0]].reset_index().drop(columns = 'index')
        
        print("Starting fixing predictions...")
        df_to_predict = fix_predictions(df, pred_data)
        print("Completed fixing predictions...")
        
        print('Formatting predictions for upload------------')
        df_to_predict = format_predictions(df_to_predict)

        df_to_predict['backtesting_unit_age_days'] = df_to_predict['backtesting_unit_age_days'].astype(int)
        
        print("Saving predictions to {}".format(args.output_path))
        pred_file_name = args.output_file_predictions + "_"+ str(df_to_predict['backtesting_unit_age_days'][0]) +'_days' + "_"+ str(date.today()) +".csv"
        #print('Prediction file name: ',pred_file_name)
        df_to_predict.to_csv(os.path.join(args.output_path,pred_file_name), index=False)