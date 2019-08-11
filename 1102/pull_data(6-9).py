# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:36:30 2017

@author: finup
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import time
import pyhs2
import sqlparse
import datetime
import pymysql as mdb
from IPython.display import display
import sys
import pickle
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

reload(sys)
sys.setdefaultencoding('utf-8')

sql = '''
    DROP TABLE IF EXISTS qianjin_test.test_yimei0719_track_1;


    DROP TABLE IF EXISTS qianjin_test.test_yimei07191_track_2;


    DROP TABLE IF EXISTS qianjin_test.test_yimei07192_track_3;


    DROP TABLE IF EXISTS qianjin_test.test_waihu0719_track_4;


    DROP TABLE IF EXISTS qianjin_test.test0719_track_5;


    DROP TABLE IF EXISTS qianjin_test.test07191_track_6;


    DROP TABLE IF EXISTS qianjin_test.test0719_app_track_7;


    DROP TABLE IF EXISTS qianjin_test.test0719_waihu_track_8;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_1_track_9;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_2_track_10;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_track_11;


    DROP TABLE IF EXISTS qianjin_test.test0719_lianxu_track_12;


    DROP TABLE IF EXISTS qianjin_test.test07191_final_1_track_13;


    DROP TABLE IF EXISTS qianjin_test.final_train_baiyang_14;


    DROP TABLE IF EXISTS qianjin_test.app_log_action_track_15;


    DROP TABLE IF EXISTS qianjin_test.test_finaltrack_new_16;


    CREATE TABLE qianjin_test.test_finaltrack_new_16 AS
    SELECT *
    FROM qianjin_develop.test_finaltrack_new
    WHERE 0=1;


    CREATE TABLE qianjin_test.test_yimei0719_track_1 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100));


    CREATE TABLE qianjin_test.test_yimei07191_track_2 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int);


    CREATE TABLE qianjin_test.test_yimei07192_track_3 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int);


    CREATE TABLE qianjin_test.test_waihu0719_track_4 (user_id varchar(100), status int, max_call_time TIMESTAMP, min_call_time TIMESTAMP, call_times int);


    CREATE TABLE qianjin_test.test0719_track_5 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int);


    CREATE TABLE qianjin_test.test07191_track_6 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int);


    CREATE TABLE qianjin_test.test0719_app_track_7 (user_Id varchar(100), pno varchar(100), staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100));


    CREATE TABLE qianjin_test.test0719_waihu_track_8 (user_Id varchar(100), pno varchar(100), isactive_aftercall int);


    CREATE TABLE qianjin_test.test0719_gap_1_track_9 (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100));


    CREATE TABLE qianjin_test.test0719_gap_2_track_10 (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100), rank int);


    CREATE TABLE qianjin_test.test0719_gap_track_11 (user_Id varchar(100), pno varchar(100), active_num int, Lastday_before_invest INT);


    CREATE TABLE qianjin_test.test0719_lianxu_track_12 (user_Id varchar(100), pno varchar(100), lianxuday_last INT);


    CREATE TABLE qianjin_test.test07191_final_1_track_13 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double, Get_date_id int);


    CREATE TABLE qianjin_test.final_train_baiyang_14 (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, isinvited int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double);


    CREATE TABLE qianjin_test.app_log_action_track_15 (pid varchar(100), userid int, dateid varchar(20), pageid varchar(100), lab varchar(5), act varchar(200), actiontime varchar(100), exittime varchar(100), clienttype int, devicetype varchar(100));


    INSERT overwrite TABLE qianjin_test.test_yimei0719_track_1
    SELECT DISTINCT b.user_id ,
                    b.mobile,
                    CASE
                        WHEN e.user_id IS NOT NULL THEN '已投资'
                        ELSE '未投资'
                    END IsInvested,
                    b.user_age,
                    b.sex_id,
                    b.create_date_Id,
                    from_unixtime(b.create_time/1000) create_time,
                    e.date_id,
                    from_unixtime(invest_time/1000) invest_time,
                    b.city_id areacode,
                    CASE
                        WHEN e.date_Id IS NULL THEN 'NA'
                        ELSE datediff(concat(substring(date_Id,1,4),'-',substring(date_Id,5,2),'-',substring(date_Id,7,2)),concat(substring(create_date_Id,1,4),'-',substring(create_date_Id,5,2),'-',substring(create_date_Id,7,2)))
                    END plus_date,
                    b.client_type_id,
                    b.channel_Id ,
                    f.pno
    FROM qianjin_mid.qianjin_mid_user_detail_temp b
    LEFT JOIN
      (SELECT *
       FROM qianjin_mid.qianjin_mid_user_invest_detail -- isinvested
       WHERE rank_id = 1
         AND bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') e ON b.user_id = e.user_Id
    LEFT JOIN
      (SELECT DISTINCT user_id,
                       max(CASE WHEN TYPE = 2 THEN pid ELSE pno END) pno
       FROM qianjin_mid.qianjin_mid_app_token -- pno
       WHERE pno IS NOT NULL
         AND bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's'
       GROUP BY user_id) f ON b.user_id = f.user_id
    WHERE concat(substring(b.create_date_id,1,4),'-',substring(b.create_date_id,5,2),'-',substring(b.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
      AND b.channel_categ_id != 73
      AND b.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
      AND b.bdp_type = 's' ;


    INSERT overwrite TABLE qianjin_test.test_yimei07191_track_2
    SELECT DISTINCT a.*,
                    max(CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END) isrecharged,
                    max(b.recharge_status_id) rechargestatus
    FROM test_yimei0719_track_1 a
    LEFT JOIN qianjin_mid.qianjin_mid_user_recharge_detail b ON a.user_id = b.user_id
    AND to_date(concat(substring(a.create_date_Id,1,4),'-',substring(a.create_date_Id,5,2),'-',substring(a.create_date_Id,7,2))) = to_date(from_unixtime(b.recharge_time/1000))
    GROUP BY a.user_Id,
             a.mobile,
             a.isinvested,
             a.user_age,
             a.sex_id,
             a.create_date_id,
             a.create_time,
             a.date_id,
             a.invest_time,
             a.areacode,
             a.plus_date,
             a.client_type_Id,
             a.channel_id,
             a.pno ;


    INSERT overwrite TABLE qianjin_test.test_yimei07192_track_3 -- isinvited
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN a.invest_time IS NOT NULL THEN datediff(concat(substring(a.date_id,1,4),'-',substring(a.date_id,5,2),'-',substring(a.date_id,7,2)),concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
                        ELSE datediff('partition_date', concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
                    END Lastday_invite ,
                    b.invest_amt Lastamt_invite
    FROM
      (SELECT a.user_Id,
              a.mobile,
              a.isinvested,
              a.user_age,
              a.sex_id,
              a.create_date_id,
              a.create_time,
              a.date_id,
              a.invest_time,
              a.areacode,
              a.plus_date,
              a.client_type_Id,
              a.channel_id,
              a.pno,
              a.isrecharged,
              a.rechargestatus,
              b.invited_id,
              CASE
                  WHEN b.invited_Id IS NOT NULL THEN 1
                  ELSE 0
              END IsInvited ,
              max(C.rank_id) rank_id
       FROM test_yimei07191_track_2 a
       JOIN
         (SELECT *
          FROM qianjin_mid.qianjin_mid_user_detail_temp
          WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND bdp_type = 's') b ON a.user_id = b.user_id
       LEFT JOIN
         (SELECT *
          FROM qianjin_mid.qianjin_mid_user_invest_detail
          WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND bdp_type = 's') c ON b.invited_id = c.user_Id
       AND (a.invest_time >= from_unixtime(c.invest_time/1000)
            OR a.invest_time IS NULL)
       GROUP BY a.user_Id,
                a.mobile,
                a.isinvested,
                a.user_age,
                a.sex_id,
                a.create_date_id,
                a.create_time,
                a.date_id,
                a.invest_time,
                a.areacode,
                a.plus_date,
                a.client_type_Id,
                a.channel_id,
                a.pno,
                a.isrecharged,
                a.rechargestatus,
                b.invited_id) a
    LEFT JOIN
      (SELECT *
       FROM qianjin_mid.qianjin_mid_user_invest_detail
       WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') b ON a.invited_id = b.user_id
    AND a.rank_id = b.rank_id;


    INSERT overwrite TABLE qianjin_test.test_waihu0719_track_4 -- call info
    SELECT user_id,
           max(phone_status) status,
           max(from_unixtime(cast(create_time/1000 AS int))) max_call_time ,
           min(from_unixtime(cast(create_time/1000 AS int))) min_call_time,
           count(*) call_times
    FROM
      (SELECT DISTINCT user_id,
                       phone_status,
                       create_time,
                       call_type
       FROM qianjin_crm.qianjin_crm_outcall_work_order
       WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') a
    WHERE call_type = 1
    GROUP BY user_id;


    INSERT overwrite TABLE qianjin_test.test0719_track_5
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN b.call_times IS NOT NULL
                             AND a.invest_time IS NOT NULL
                             AND (b.min_call_time > a.create_time
                                  OR a.invest_time > b.max_call_time) THEN 1
                        WHEN b.call_times IS NOT NULL
                             AND a.invest_time IS NULL
                             AND (b.min_call_time > a.create_time
                                  OR b.max_call_time > a.create_time) THEN 1
                        ELSE 0
                    END callstatus
    FROM qianjin_test.test_yimei07192_track_3 a
    LEFT JOIN qianjin_test.test_waihu0719_track_4 b ON a.user_Id = b.user_id;


    INSERT overwrite TABLE qianjin_test.test07191_track_6
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN b.status = 3
                             AND a.callstatus = 1 THEN '成功'
                        WHEN b.status = 4
                             AND a.callstatus = 1 THEN '预约下次'
                        WHEN b.status = 5
                             AND a.callstatus = 1 THEN '拒访'
                        ELSE '失败'
                    END call_effec ,
                    CASE
                        WHEN a.callstatus = 1 THEN datediff(a.invest_time,max_call_time)
                    END Gap_CalltoInvest ,
                    CASE
                        WHEN a.callstatus = 1
                             AND a.invest_time >= b.max_call_time THEN b.max_call_time
                        WHEN a.callstatus = 1
                             AND a.invest_time < b.max_call_time THEN b.min_call_time
                        WHEN a.callstatus = 1
                             AND a.invest_time IS NULL THEN b.max_call_time
                    END last_calltime ,
                    b.call_times
    FROM qianjin_test.test0719_track_5 a
    LEFT JOIN qianjin_test.test_waihu0719_track_4 b ON a.user_Id = b.user_id ;


    INSERT overwrite TABLE qianjin_test.app_log_action_track_15
    SELECT pid,
           userid,
           dateid,
           pageid,
           lab,
           act,
           actiontime,
           exittime,
           clienttype,
           devicetype
    FROM qianjin.qianjin_app_log b
    WHERE concat(substring(b.bdp_day,1,4),'-',substring(b.bdp_day,5,2),'-',substring(b.bdp_day,7,2)) BETWEEN statistic_dayx AND statistic_dayy ;


    INSERT overwrite TABLE qianjin_test.test0719_app_track_7
    SELECT a.user_id,
           a.pno,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('RechargeActivity','RechargeMainViewController', 'JoinConfirmRechargeNewActivity','BuyConfirmViewController')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_recharge,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('RechargeActivity','RechargeMainViewController', 'JoinConfirmRechargeNewActivity','BuyConfirmViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_recharge,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity3','ProductionDetailDPViewController_3')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p3,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity3','ProductionDetailDPViewController_3')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p3,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity6','ProductionDetailDPViewController_6')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p6,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity6','ProductionDetailDPViewController_6')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p6,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity12','ProductionDetailDPViewController_12')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p12,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity12','ProductionDetailDPViewController_12')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p12,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('IHuoBaoDetailActivity','ProductionDetailCTViewController')
               AND exittime >= actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_ph,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('IHuoBaoDetailActivity','ProductionDetailCTViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_ph,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity1','ProductionDetailDPViewController_1')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p1,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity1','ProductionDetailDPViewController_1')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p1,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('JoinDepositActivity3','BuyDPViewController_3' ,'JoinDepositActivity6','BuyDPViewController_6' ,'JoinDepositActivity12','BuyDPViewController_12' ,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_buy,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('JoinDepositActivity3','BuyDPViewController_3' ,'JoinDepositActivity6','BuyDPViewController_6' ,'JoinDepositActivity12','BuyDPViewController_12' ,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_buy,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('AssetsNewActivity','AssetsMainViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_asset,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('AssetsNewActivity','AssetsMainViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_asset,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('MyRewardActivity','RewardViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_redbag,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('MyRewardActivity','RewardViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_redbag,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('InterActionActivity','InteractionMainViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_hudong,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('InterActionActivity','InteractionMainViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_hudong,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击banner','点击baner') THEN pageid END) staynum_banner,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 1
                      OR c.clienttype =3
                      AND c.lab = 0) THEN pageid END) staynum_icon1,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 2
                      OR c.clienttype =3
                      AND c.lab = 1) THEN pageid END) staynum_icon2,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 3
                      OR c.clienttype =3
                      AND c.lab = 2) THEN pageid END) staynum_icon3,
           max(c.devicetype) devicetype
    FROM qianjin_test.test07191_track_6 a
    LEFT JOIN qianjin_test.app_log_action_track_15 c ON a.pno = c.pid
    AND concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) = concat(substring(c.dateid,1,4),'-',substring(c.dateid,5,2),'-',substring(c.dateid,7,2))
    GROUP BY a.user_id,
             a.pno;


    INSERT overwrite TABLE qianjin_test.test0719_waihu_track_8
    SELECT a.user_id,
           a.pno,
           MAX (CASE
                    WHEN a.last_calltime IS NOT NULL
                         AND a.last_calltime < b.actiontime
                         AND to_date(a.last_calltime) = to_date(b.actiontime)
                         AND pageid !='060000' THEN 1
                    ELSE 0
                END) isactive_aftercall
    FROM qianjin_test.test07191_track_6 a
    LEFT JOIN qianjin_test.app_log_action_track_15 b ON a.pno = b.pid
    GROUP BY a.user_id,
             a.pno;


    INSERT overwrite TABLE qianjin_test.test0719_gap_1_track_9
    SELECT DISTINCT a.user_id,
                    a.pno,
                    a.create_date_Id,
                    a.date_id,
                    CASE
                        WHEN b.actiontime BETWEEN a.create_time AND a.invest_time THEN to_date(b.actiontime)
                        WHEN b.actiontime >= a.create_time
                             AND (a.invest_time IS NULL
                                  OR a.invest_time ='')
                             AND to_date(b.actiontime) <= date_add(a.create_time,30) THEN to_date(b.actiontime)
                    END active_dateid
    FROM qianjin_test.test07191_track_6 a
    LEFT JOIN qianjin_test.app_log_action_track_15 b ON a.pno = b.pid;


    INSERT overwrite TABLE qianjin_test.test0719_gap_2_track_10
    SELECT *,
           row_number() over (partition BY user_id
                              ORDER BY active_dateid) rank
    FROM qianjin_test.test0719_gap_1_track_9
    WHERE active_dateid IS NOT NULL;


    INSERT overwrite TABLE qianjin_test.test0719_gap_track_11
    SELECT user_id,
           pno,
           count(CASE WHEN active_dateid IS NOT NULL THEN user_id END) active_num ,
           MAX(CASE
                   WHEN date_id IS NOT NULL
                        AND active_dateid < concat(substring(date_id,1,4),'-',substring(date_id,5,2),'-',substring(date_id,7,2)) THEN date_id
               END) Lastday_before_invest
    FROM qianjin_test.test0719_gap_2_track_10
    GROUP BY user_id,
             pno;


    INSERT overwrite TABLE qianjin_test.test0719_lianxu_track_12
    SELECT a.user_id,
           a.pno,
           MAX(a.rank) - MAX(CASE
                                 WHEN c.active_Num =1 THEN 1
                                 WHEN (datediff(a.active_dateid,b.active_dateid) !=1
                                       OR b.active_dateid IS NULL)
                                      AND c.active_Num !=1 THEN a.rank
                             END) lianxuday_last
    FROM qianjin_test.test0719_gap_2_track_10 a
    JOIN qianjin_test.test0719_gap_track_11 c ON a.user_Id = c.user_id
    LEFT JOIN qianjin_test.test0719_gap_2_track_10 b ON a.user_id = b.user_id
    AND a.rank = b.rank+1
    GROUP BY a.user_id,
             a.pno,
             c.active_Num;


    INSERT overwrite TABLE qianjin_test.test07191_final_1_track_13
    SELECT a.*,
           b.staytime,
           b.staynum,
           b.staytime_recharge,
           b.staynum_recharge,
           b.staytime_p3,
           b.staynum_p3,
           b.staytime_p6,
           b.staynum_p6,
           b.staytime_p12,
           b.staynum_p12,
           b.staytime_ph,
           b.staynum_ph,
           b.staytime_p1,
           b.staynum_p1,
           b.staytime_buy,
           b.staynum_buy,
           b.staytime_asset,
           b.staynum_asset,
           b.staytime_redbag,
           b.staynum_redbag,
           b.staytime_hudong,
           b.staynum_hudong,
           b.staynum_banner,
           b.staynum_icon1,
           b.staynum_icon2,
           b.staynum_icon3,
           b.devicetype,
           c.isactive_aftercall,
           d.active_num,
           d.Lastday_before_invest,
           e.lianxuday_last,
           f.rate,
           '20170818' AS Get_date_id
    FROM qianjin_test.test07191_track_6 a
    JOIN qianjin_test.test0719_app_track_7 b ON a.user_id = b.user_Id
    JOIN qianjin_test.test0719_waihu_track_8 c ON a.user_id = c.user_id
    LEFT JOIN qianjin_test.test0719_gap_track_11 d ON a.user_id = d.user_id
    LEFT JOIN qianjin_test.test0719_lianxu_track_12 e ON a.user_id = e.user_id
    LEFT JOIN
      (SELECT a.channel_id,
              a.num/b.num rate
       FROM
         (SELECT channel_id,
                 count(DISTINCT a.user_id) num
          FROM qianjin_mid.qianjin_mid_user_detail_temp a
          JOIN qianjin_mid.qianjin_mid_user_invest_detail b ON a.user_id = b.user_id
          AND date_id = create_date_id
          WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
            AND a.bdp_type = 's'
            AND b.bdp_type = 's'
            AND a.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND b.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND b.rank_id = 1
          GROUP BY channel_id) a
       JOIN
         (SELECT channel_id,
                 count(DISTINCT a.user_id) num
          FROM qianjin_mid.qianjin_mid_user_detail_temp a
          WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
            AND a.bdp_type = 's'
            AND a.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
          GROUP BY channel_id) b ON a.channel_id = b.channel_id) f ON a.channel_id = f.channel_id;


    INSERT overwrite TABLE qianjin_test.final_train_baiyang_14
    SELECT a.user_id,
           a.mobile,
           a.isinvested,
           a.user_age,
           a.sex_id,
           a.create_date_id,
           a.create_time,
           a.date_id,
           a.invest_time,
           a.areacode,
           a.plus_date,
           a.client_type_Id,
           a.channel_id,
           a.pno,
           a.isrecharged,
           a.rechargestatus,
           a.isinvited,
           a.lastday_invite,
           a.lastamt_invite,
           a.callstatus,
           a.call_effec,
           a.gap_calltoinvest,
           a.last_calltime,
           a.call_times,
           a.staytime,
           a.staynum,
           a.staytime_recharge,
           a.staynum_recharge,
           a.staytime_p3,
           a.staynum_p3,
           a.staytime_p6,
           a.staynum_p6,
           a.staytime_p12,
           a.staynum_p12,
           a.staytime_ph,
           a.staynum_ph,
           a.staytime_p1,
           a.staynum_p1,
           a.staytime_buy,
           a.staynum_buy,
           a.staytime_asset,
           a.staynum_asset,
           a.staytime_redbag,
           a.staynum_redbag,
           a.staytime_hudong,
           a.staynum_hudong,
           a.staynum_banner,
           a.staynum_icon1,
           a.staynum_icon2,
           a.staynum_icon3,
           a.devicetype,
           a.isactive_aftercall,
           a.active_num,
           a.lastday_before_invest,
           a.lianxuday_last,
           a.rate
    FROM qianjin_test.test07191_final_1_track_13 a;


    DROP TABLE qianjin_test.test_yimei0719_track_1;


    DROP TABLE qianjin_test.test_yimei07191_track_2;


    DROP TABLE qianjin_test.test_yimei07192_track_3;


    DROP TABLE qianjin_test.test_waihu0719_track_4;


    DROP TABLE qianjin_test.test0719_track_5;


    DROP TABLE qianjin_test.test07191_track_6;


    DROP TABLE qianjin_test.test0719_app_track_7;


    DROP TABLE qianjin_test.test0719_waihu_track_8;


    DROP TABLE qianjin_test.test0719_gap_1_track_9;


    DROP TABLE qianjin_test.test0719_gap_2_track_10;


    DROP TABLE qianjin_test.test0719_gap_track_11;


    DROP TABLE qianjin_test.test0719_lianxu_track_12;


    DROP TABLE qianjin_test.test07191_final_1_track_13;


    DROP TABLE qianjin_test.app_log_action_track_15;


    SELECT *
    FROM qianjin_test.final_train_baiyang_14
    '''

sql_train = sql
sql_test = sql.replace('statistic_dayx', 'statistic_dayb').replace('statistic_dayy', 'statistic_daye')
sql_test = sql_test.replace('final_train_baiyang', 'final_test_baiyang').replace('final_train_baiyang', 'final_test_baiyang')

class HiveClient:
    # create connection to hive server2
    MAX_BLOCK_SIZE = 2
    def __init__(self, db_host, user, password, database, port=10000, authMechanism="PLAIN"):
        self.conn = pyhs2.connect(host=db_host,
                                  port=port,
                                  authMechanism=authMechanism,
                                  user=user,
                                  password=password,
                                  database=database,
                                  )
       
    def query(self, sql):
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            val = cursor.fetchall()
#            #print val
            columnNames = [a['columnName'] for a in cursor.getSchema()]
            df = pd.DataFrame(data=val, columns=columnNames)
#            # print df.head()
            return df
            # return cursor.fetch()
    def close(self):
        self.conn.close()

def time_func(delta):
    return (datetime.date.today() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")

def pull_data(sql_train, sql_test):
    print("Begin pulling data...")
    
    # statistic_dayx, statistic_dayy = '"' + time_func(4) + '"', '"' + time_func(4) + '"'
    # statistic_dayb, statistic_daye = '"' + time_func(1) + '"', '"' + time_func(1) + '"'

    # statistic_dayx, statistic_dayy = '''"2017-08-01"''', '''"2017-08-31"'''



    statistic_dayx, statistic_dayy = '''"2017-06-01"''', '''"2017-09-30"'''

    # statistic_dayb, statistic_daye = '''"2017-09-02"''', '''"2017-09-02"'''

    # partition_date_train = '"' + time_func(0) + '"'
    # partition_date_test = '"' + time_func(0) + '"'

    partition_date_train = '''"2017-10-01"'''
    partition_date_label = '''"2017-10-01"'''

    # partition_date_test = '"' + time_func(0) + '"'

    print("Training dates: {}".format(statistic_dayx, statistic_dayy))
    # print("Prediction dates: {}".format(statistic_dayb, statistic_daye))
    # print("Partition date: {}".format(partition_date_test))

    start_time = time.time()
    hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
            password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    
    sql_train = unicode(sql_train,'utf-8')
    parsedsql_train = sqlparse.split(sql_train)
    for elt in enumerate(parsedsql_train):
        sqltmp = elt[1].replace(';','').replace('statistic_dayx',statistic_dayx).replace('statistic_dayy',statistic_dayy).replace('partition_date', partition_date_train).replace('partition_label', partition_date_label).encode('utf8')
        print(sqltmp)
        result = hive_client.query(sqltmp)
        print(result.head())
        print("pulling data is done --- %s seconds ---" % round((time.time() - start_time),2))
        filename = '0923-0922_training' + str(elt[0]) +'.csv' 
        result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
        print("saving data is done --- %s seconds ---" % round((time.time() - start_time),2))
    # hive_client.close()


    # hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
    #         password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')

    # #pull test data    
    # sql_test = unicode(sql_test,'utf-8')
    # parsedsql_test = sqlparse.split(sql_test)
    # for elt in enumerate(parsedsql_test):
    #     sqltmp = elt[1].replace(';','').replace('statistic_dayb',statistic_dayb).replace('statistic_daye',statistic_daye).replace('partition_date', partition_date_test).encode('utf8')
    #     print(sqltmp)
    #     result = hive_client.query(sqltmp)
    #     print(result.head())
    #     print("pulling data is done --- %s seconds ---" % round((time.time() - start_time),2))
    #     filename = 'oot_testing' + str(elt[0]) +'.csv' 
    #     result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
    #     print("saving data is done --- %s seconds ---" % round((time.time() - start_time),2))
    hive_client.close()

    print("Finished pulling data...")

if __name__ == '__main__':
    pull_data(sql_train, sql_test)