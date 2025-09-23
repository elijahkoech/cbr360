import pandas as pd
import argparse
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import boto3
import gc
import datetime as dt
import io
from io import StringIO
import os

bucket_name = 'cbr-cohort-experiments'
kpi_prefix ='KPIs_Inference/'
kpi_file = 'KPIs_data_predictions.csv'
currency_exchange_prefix = 'Exchange_rate_data/'
currency_exchange_file = 'Exchange_rate_as_on_1st_march_24.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/data/')
parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/')
parser.add_argument('--output_file', type=str, default='KPIs_data_processed')
parser.add_argument('--output_KPI_redshift_file', type=str, default='KPIs_redshift_data')



# def get_cohort_details(df_KPI):
    
#     # Get cohort details
#     cohort_details = df_KPI["accounts_group"].str.split("_", n = 5, expand = True)
    
#     df_KPI["reg_month"] = cohort_details[0]
#     df_KPI["country"] = cohort_details[1]
#     df_KPI["product_group"] = cohort_details[2]
#     df_KPI["area"] = cohort_details[3]
#     df_KPI["primary_product"] = cohort_details[4]
    
#     return df_KPI


def get_cohort_age_in_months(row):
    
    reg_year = int(row['reg_month'].split('-')[0])
    reg_month = int(row['reg_month'].split('-')[1])
    first_dt_of_mon = date(reg_year, reg_month, 1)
    next_month = first_dt_of_mon.replace(day=28) + timedelta(days=4)
    last_dt_of_mon = next_month - timedelta(days=next_month.day)
    # Last day of previous month for reference
    last_day_of_prev_month = date.today().replace(day=1) - timedelta(days=1)
    # Get the relativedelta between two dates
    delta = relativedelta(last_day_of_prev_month, last_dt_of_mon)
    # get months difference
    res_months = delta.months + (delta.years * 12)
    
    return res_months


def preprocessing_common(df_KPI):

    df_KPI['age_in_months'] = df_KPI.apply(lambda row: get_cohort_age_in_months(row), 
                                               axis=1)
    
    df_KPI['avg_planned_repayment_days'].fillna(-1, inplace=True)
    df_KPI['avg_planned_repayment_days'] = df_KPI['avg_planned_repayment_days'].astype(int)

    for limit in [30,60,90,180,270,360,450,540,630,720]:
        df_KPI.loc[df_KPI['frr_'+str(limit)+'']>1, 'frr_'+str(limit)+''] = 1
        df_KPI.loc[(df_KPI['at_risk_rate_'+str(limit)+''].isnull()) &(df_KPI['frr_'+str(limit)+'']==1),'at_risk_rate_'+str(limit)+''] = 0
        
    return df_KPI


def preprocessing_backtesting(df_KPI, limit):
    
    dict_backtesting_age_in_months = { 30 : 1, 60 : 2, 90 : 3, 180 : 6, 270: 9 , 360 : 12, 450: 15, 540: 18, 630: 21, 720: 24}
    
    df_KPI_backtesting = df_KPI.loc[df_KPI['age_in_months']>=dict_backtesting_age_in_months[backtesting_limit]]
    df_KPI_backtesting.loc[:,'backtesting_limit'] = limit
        
    if backtesting_limit == 360: 
        
        df_KPI_backtesting.loc[(df_KPI_backtesting['age_in_months']>=12) & (df_KPI_backtesting['age_in_months']<15),'cohort_age_backtesting'] = 360
        df_KPI_backtesting.loc[(df_KPI_backtesting['age_in_months']>=15) & (df_KPI_backtesting['age_in_months']<18),'cohort_age_backtesting']  = 450
        df_KPI_backtesting.loc[(df_KPI_backtesting['age_in_months']>=18) & (df_KPI_backtesting['age_in_months']<21),'cohort_age_backtesting']= 540
        df_KPI_backtesting.loc[(df_KPI_backtesting['age_in_months']>=21) & (df_KPI_backtesting['age_in_months']<24),'cohort_age_backtesting'] = 630
        df_KPI_backtesting.loc[df_KPI_backtesting['age_in_months']>=24,'cohort_age_backtesting'] = 720
    else:
        df_KPI_backtesting.loc[:,'cohort_age_backtesting'] = limit
        
    df_KPI_backtesting['cohort_age_backtesting'] = df_KPI_backtesting['cohort_age_backtesting'].astype('int64')
                       
        
    return df_KPI_backtesting


def revenue_cal(df_KPI, df_exchange_rate):
    
    currency_country_dict = {'KES':'Kenya','UGX':'Uganda','NGN':'Nigeria','TZS':'Tanzania','MMK':'Myanmar','RMB':'China',
                         'HKD':'Hong Kong','USD':'United States','INR':'India','ZMK':'Zambia','MZN':'Mozambique','XOF':'Togo',
                         'XAF':'Cameroon','MWK':'Malawi'}
    df_exchange_rate['country'] = df_exchange_rate['currency'].map(currency_country_dict)
    
    df_KPI = pd.merge(df_KPI, df_exchange_rate[['country','exchange']], how='left', on='country')
    
    df_KPI['total_follow_on_revenue_usd_cal'] = np.round(df_KPI['total_follow_on_revenue']/df_KPI['exchange'],4)
    df_KPI['total_follow_on_revenue_usd_final'] = np.where(df_KPI['reg_month']<='2024-01', df_KPI['total_follow_on_revenue_usd_cal'],df_KPI['total_follow_on_revenue_usd'])
    
    return df_KPI


def feature_engineering(df_to_predict):
    
    df_to_predict['product_group'].fillna('NA', inplace=True)
    # Creating column for Unlock price
    #df_to_predict['unlock_price_usd'] = df_to_predict['upfront_price_usd'] + df_to_predict['total_follow_on_revenue_current_usd_final']
    df_to_predict['unlock_price_usd'] = df_to_predict['upfront_price_usd'] + df_to_predict['total_follow_on_revenue_usd_final']
    
    # Calculating averge unlock and upfront price
    df_to_predict['avg_upfront_price_usd'] = np.round((df_to_predict['upfront_price_usd']/df_to_predict['count_units']),0)
    df_to_predict['avg_unlock_price_usd'] = np.round((df_to_predict['unlock_price_usd']/df_to_predict['count_units']),0)
    
    # Removing unnecessary columns
    #cols_to_remove = [col for col in df_to_predict.columns if ('repayment_speed_' in col)]
    cols_to_remove = []
    cols_to_remove.append('upfront_price_usd')
    cols_to_remove.append('unlock_price_usd')

    print('columns to remove: ',cols_to_remove)
    
    df_to_predict.drop(cols_to_remove, axis=1, inplace=True)
    
    return df_to_predict


###-------------------------------------Custom methods end here------------------------------------###


# Main flow

if __name__ == "__main__":

    args = parser.parse_args()

    print("Getting data from s3")
    bucket = boto3.resource('s3').Bucket(bucket_name)
    print('KPI file path: ',kpi_prefix+kpi_file)
    
    # Downloan KPIs(input) and target data 
    bucket.download_file(kpi_prefix+kpi_file,os.path.join(args.input_path,kpi_file))
    bucket.download_file(currency_exchange_prefix+currency_exchange_file,os.path.join(args.input_path,currency_exchange_file))
    print("All data saved in {}".format(os.path.join(args.input_path)))

    print("Reading data...")
    df_KPI = pd.read_csv(os.path.join(args.input_path,kpi_file),low_memory=False)
    df_exchange_rate = pd.read_csv(os.path.join(args.input_path,currency_exchange_file),low_memory=False)
    print('All data has been read')
    
    print('Starting preprocessing.....................')
    #df_KPI = get_cohort_details(df_KPI)    
    df_KPI = preprocessing_common(df_KPI)
    print('Preprocessing completed.....................')
    
    print('Starting revenue calculation.....................')
    df_KPI = revenue_cal(df_KPI, df_exchange_rate)
    print('Completed revenue calculation.....................')
    
    print('Starting feature engineering.....................')
    df_KPI = feature_engineering(df_KPI)
    print('Conpleted feature engineering.....................')
    
    df_final = pd.DataFrame()

    for backtesting_limit in [30, 60, 90, 180, 270, 360]:

        print('Processing for {}'.format(backtesting_limit))

        df_KPI_temp = preprocessing_backtesting(df_KPI, backtesting_limit)
        df_final = df_KPI_temp if df_final.shape[0]==0 else pd.concat([df_final, df_KPI_temp])

    df_final.reset_index(inplace=True, drop=True)
    print('Completed preprocessing.....................')
    
    print("Saving preprocessed data to {}".format(args.output_path))
    processed_file_name = args.output_file + "_"+ str(date.today()) +".csv"
    df_final.to_csv(os.path.join(args.output_path,processed_file_name), index = False)
    print("Processing completed and KPIs data saved to S3")