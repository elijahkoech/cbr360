-- Save Target to S3
UNLOAD ('SELECT * FROM redshift_consumer_pdt.cbr_cohort3_target')
TO 's3://bucket_name/cbr_targets.csv'
CREDENTIALS 'aws_access_key_id=*******;aws_secret_access_key=**************'
ALLOWOVERWRITE
PARALLEL OFF
FORMAT AS CSV
DELIMITER ','
HEADER;

-- Save the KPIs to S3
UNLOAD ('SELECT * FROM redshift_consumer_pdt.cbr_cohort3_kpis')
TO 's3://bucket_name/cbr_kpis.csv'
CREDENTIALS 'aws_access_key_id=*******;aws_secret_access_key=**************'
ALLOWOVERWRITE
PARALLEL OFF
FORMAT AS CSV
DELIMITER ','
HEADER;



UNLOAD ($$
select country, to_char(registration_date_utc, 'YYYY-MM') as reg_month, product_group, primary_product, area_derived, count(*) as num_accounts 
from easybuy.accounts_final_pdt as accounts 
where has_payment_plan = true and NOT(LOWER(group_name) LIKE '%cross%') 
AND (NOT (LOWER(accounts.group_name) LIKE '%cross%') OR (LOWER(accounts.group_name) LIKE '%cross%') IS NULL) 
AND (NOT (LEFT (accounts.organization,  21)  <> 'Greenlight Planet DTV' AND  accounts.organization  <> 'Global Community Standards' 
AND LEFT (accounts.organization, 7)   <> 'EasyBuy') OR (LEFT (accounts.organization,  21)  <> 'Greenlight Planet DTV' 
AND  accounts.organization <> 'Global Community Standards' AND LEFT (accounts.organization, 7)   <> 'EasyBuy') IS NULL)  
Group By 1,2,3,4,5
$$)
TO 's3://bucket_name/accounts_moddelling_data.csv'
CREDENTIALS 'aws_access_key_id==*******;;aws_secret_access_key==**************''
ALLOWOVERWRITE
PARALLEL OFF
FORMAT AS CSV
DELIMITER ','
HEADER;
