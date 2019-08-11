set mapred.job.queue.name=eng;

set hive.exec.scratchdir=/tmp/zhouxiaolei2;

use xl_mobile_uv_analysis;

DROP TABLE IF EXISTS boc_bjdq_0711;
CREATE TABLE '${final_data_table}'  AS --boc_bjdq_0711
SELECT 
    A.deviceid,
    A.channel,
    B.reg_date as register_date,
    A.date as boc_submit_date,
    A.order_count as order_count,
(CASE
     WHEN datediff(A.date,B.reg_date) = 0 then 1
     WHEN datediff(A.date,B.reg_date) = 1 then 2
     WHEN datediff(A.date,B.reg_date) = 2 then 3
     WHEN datediff(A.date,B.reg_date) >= 3 and datediff(A.date,B.reg_date) < 6 then 7
     WHEN datediff(A.date,B.reg_date) >= 7 and datediff(A.date,B.reg_date) < 14 then 14
     WHEN datediff(A.date,B.reg_date) >= 14 and datediff(A.date,B.reg_date) < 21 then 21
     WHEN datediff(A.date,B.reg_date) >= 21 and datediff(A.date,B.reg_date) < 30 then 30
     WHEN datediff(A.date,B.reg_date) >= 30 and datediff(A.date,B.reg_date) < 60 then 60
     WHEN datediff(A.date,B.reg_date) >= 60 and datediff(A.date,B.reg_date) < 90 then 90
     WHEN datediff(A.date,B.reg_date) >= 90 then 99
    END
     ) as active_date_flag
FROM
    (SELECT 
       deviceid,
       channel,
       date,
       order_count
     FROM
         xl_mobile_uv_analysis.mobile_active_user_BJDQ
     WHERE
        date between '2016-05-19' and '${end_date}'
     ) A
     
     JOIN
     
     (SELECT
            deviceid,
            reg_date
      FROM
          xl_mobile_uv_analysis.mobile_new_user_BJDQ
      where
          reg_date between '2016-05-19' and '${end_date}'
      ) B
      
      ON
      
      A.deviceid = B.deviceid
order by register_date asc,boc_submit_date asc, active_date_flag asc
;



DROP TABLE IF EXISTS final_report_bjdq_0711;
CREATE TABLE '${final_report}' AS --final_report_bjdq_0711
SELECT 
    A.channel,
    A.register_date,
    B.new_user_total,
    C.daily_active_users as active_users_of_all_time,
    A.active_users_to_reg_date,
    A.BOC_within_1_day,
    A.BOC_within_2_days,
    A.BOC_within_3_days,
    A.BOC_within_7_days,
    A.BOC_within_14_days,
    A.BOC_within_21_days,
    A.BOC_within_30_days,
    A.BOC_within_60_days,
    A.BOC_within_90_days
FROM   
   (
    SELECT
        channel,
        register_date,
        count(distinct deviceid) as active_users_to_reg_date,
        sum(CASE WHEN active_date_flag = 1 THEN order_count ELSE 0 END) as BOC_within_1_day,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 2 THEN order_count ELSE 0 END) as BOC_within_2_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 3 THEN order_count ELSE 0 END) as BOC_within_3_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 7 THEN order_count ELSE 0 END) as BOC_within_7_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 14 THEN order_count ELSE 0 END) as BOC_within_14_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 21 THEN order_count ELSE 0 END) as BOC_within_21_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 30 THEN order_count ELSE 0 END) as BOC_within_30_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 60 THEN order_count ELSE 0 END) as BOC_within_60_days,
        sum(CASE WHEN active_date_flag BETWEEN 1 AND 90 THEN order_count ELSE 0 END) as BOC_within_90_days
    FROM
        xl_mobile_uv_analysis.boc_bjdq_0711 
    GROUP BY
        channel,register_date
    ) A
    
LEFT JOIN

    (SELECT
        AA.reg_date AS register_date,
        BB.cha AS channel,
        count(*) as new_user_total
    FROM
        (SELECT
            distinct
            tmpB.deviceid,
            tmpB.reg_date
        FROM
            xl_mobile_uv_analysis.mobile_new_user_bjdq tmpB
        WHERE
            tmpB.reg_date BETWEEN '2016-05-19' AND '${end_date}'
        ) AA
     JOIN
        (SELECT
            distinct
            tmpC.dvid,
            tmpC.cha,
            tmpC.date
        FROM
             mobile_dw.stage_mobile_bjdq_start_d tmpC
        WHERE
            LENGTH(tmpC.dvid) > 0
            AND
            tmpC.cha in ('c46','c126')
            AND
            tmpC.date BETWEEN '2016-05-19' AND '2016-07-11'
        ) BB
     ON
        AA.deviceid = BB.dvid 
        AND 
        AA.reg_date = BB.date
     GROUP BY
        AA.reg_date,BB.cha
    ) B
     
    ON
        A.channel = B.channel
        AND
        A.register_date = B.register_date

LEFT JOIN

    (SELECT 
       channel,
       date,
       count(distinct deviceid) as daily_active_users
     FROM
         xl_mobile_uv_analysis.mobile_active_user_BJDQ
     WHERE
        date between '2016-05-19' and '2016-07-11'
     GROUP BY
        channel,date
     ) C
    
    ON
    A.channel = C.channel
    AND
    A.register_date = C.date
;      