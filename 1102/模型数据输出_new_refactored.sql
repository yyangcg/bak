USE qianjin_develop;

 -- create tables

CREATE TABLE qianjin_develop.test_yimei0719_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100));


CREATE TABLE qianjin_develop.test_yimei07191_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int);


CREATE TABLE qianjin_develop.test_yimei07192_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int);


CREATE TABLE qianjin_develop.test_waihu0719_track (user_id varchar(100), status int, max_call_time TIMESTAMP, min_call_time TIMESTAMP, call_times int);


CREATE TABLE qianjin_develop.test0719_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int);


CREATE TABLE qianjin_develop.test07191_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int);


CREATE TABLE qianjin_develop.test0719_app_track (user_Id varchar(100), pno varchar(100), staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100));


CREATE TABLE qianjin_develop.test0719_waihu_track (user_Id varchar(100), pno varchar(100), isactive_aftercall int);


CREATE TABLE qianjin_develop.test0719_gap_1_track (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100));


CREATE TABLE qianjin_develop.test0719_gap_2_track (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100), rank int);


CREATE TABLE qianjin_develop.test0719_gap_track (user_Id varchar(100), pno varchar(100), active_num int, Lastday_before_invest INT);


CREATE TABLE qianjin_develop.test0719_lianxu_track (user_Id varchar(100), pno varchar(100), lianxuday_last INT);


CREATE TABLE qianjin_develop.test07191_final_1_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double, Get_date_id int);


CREATE TABLE qianjin_develop.test07191_final_new_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, isinvited int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double);


CREATE TABLE qianjin_develop.app_log_action_track (pid varchar(100), userid int, dateid varchar(20), pageid varchar(100), lab varchar(5), act varchar(200), actiontime varchar(100), exittime varchar(100), clienttype int, devicetype varchar(100));

 -- begin

INSERT INTO test_yimei0719_track
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
                    
                    ELSE datediff(FROM_UNIXTIME(unix_timestamp(date_Id, 'yyyyMMdd'),'yyyyMMdd')), FROM_UNIXTIME(unix_timestamp(create_date_Id, 'yyyyMMdd'),'yyyyMMdd')
                    -- ELSE datediff(concat(substring(date_Id,1,4),'-',substring(date_Id,5,2),'-',substring(date_Id,7,2)), concat(substring(create_date_Id,1,4),'-',substring(create_date_Id,5,2),'-',substring(create_date_Id,7,2)))
                END plus_date,
                b.client_type_id,
                b.channel_Id ,
                f.pno
FROM qianjin_mid.qianjin_mid_user_detail_temp b
LEFT JOIN
  (SELECT *
   FROM qianjin_mid.qianjin_mid_user_invest_detail
   WHERE rank_id = 1
   AND bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
     -- AND bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
     AND bdp_type = 's') e ON b.user_id = e.user_Id
LEFT JOIN
  (SELECT DISTINCT user_id,
                   max(CASE WHEN TYPE = 2 THEN pid ELSE pno END) pno
   FROM qianjin_mid.qianjin_mid_app_token
   WHERE pno IS NOT NULL
     AND bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
     -- AND bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
     AND bdp_type = 's'
   GROUP BY user_id) f ON b.user_id = f.user_id
  WHERE FROM_UNIXTIME(unix_timestamp(b.create_date_id, 'yyyyMMdd'),'yyyyMMdd') = date_add(current_date(),-1)
-- WHERE concat(substring(b.create_date_id,1,4),'-',(b.create_date_id,5,2),'-',(b.create_date_id,7,2)) = date_add(current_date(),-1)
  AND b.channel_categ_id != 73
  AND b.bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
  -- AND b.bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
  AND b.bdp_type = 's' ;

INSERT INTO test_yimei07191_track
SELECT DISTINCT a.*,
                max(CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END) isrecharged,
                max(b.recharge_status_id) rechargestatus
FROM test_yimei0719_track a
LEFT JOIN qianjin_mid.qianjin_mid_user_recharge_detail b ON a.user_id = b.user_id
AND to_date(FROM_UNIXTIME(unix_timestamp(a.create_date_Id, 'yyyyMMdd'),'yyyyMMdd')) = to_date(from_unixtime(b.recharge_time/1000))
-- AND to_date(concat(substring(a.create_date_Id,1,4),'-',substring(a.create_date_Id,5,2),'-',substring(a.create_date_Id,7,2))) = to_date(from_unixtime(b.recharge_time/1000))
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


INSERT INTO qianjin_develop.test_yimei07192_track
SELECT DISTINCT a.* ,
                CASE
                	WHEN a.invest_time IS NOT NULL THEN datediff(FROM_UNIXTIME(unix_timestamp(a.date_id, 'yyyyMMdd'),'yyyyMMdd'), FROM_UNIXTIME(unix_timestamp(b.date_id, 'yyyyMMdd'),'yyyyMMdd'))
                    -- WHEN a.invest_time IS NOT NULL THEN datediff(concat(substring(a.date_id,1,4),'-',substring(a.date_id,5,2),'-',substring(a.date_id,7,2)),concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
                    ELSE datediff(current_date(), FROM_UNIXTIME(unix_timestamp(b.date_id, 'yyyyMMdd'),'yyyyMMdd'))
                    -- ELSE datediff(current_date(), concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
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
   FROM test_yimei07191_track a
   JOIN
     (SELECT *
      FROM qianjin_mid.qianjin_mid_user_detail_temp
      
      WHERE bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
      -- WHERE bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
        AND bdp_type = 's') b ON a.user_id = b.user_id
   LEFT JOIN
     (SELECT *
      FROM qianjin_mid.qianjin_mid_user_invest_detail
      bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
      -- WHERE bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
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

   
   WHERE bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
   -- WHERE bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
     AND bdp_type = 's') b ON a.invited_id = b.user_id
AND a.rank_id = b.rank_id;

 --  添加电销相关标签，电销数据来自46库 waihu date

INSERT INTO qianjin_develop.test_waihu0719_track
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
   WHERE bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
   -- WHERE bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
     AND bdp_type = 's') a
WHERE call_type = 1
GROUP BY user_id;

 -- is waihu ?

INSERT INTO qianjin_develop.test0719_track
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
FROM qianjin_develop.test_yimei07192_track a
LEFT JOIN qianjin_develop.test_waihu0719_track b ON a.user_Id = b.user_id;

 -- waihu reusult

INSERT INTO qianjin_develop.test07191_track
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
FROM qianjin_develop.test0719_track a
LEFT JOIN qianjin_develop.test_waihu0719_track b ON a.user_Id = b.user_id ;

 -- add app_Log data

INSERT INTO qianjin_develop.app_log_action_track
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
FROM qianjin.qianjin_app_log
--                                                                                    current_date()) ;
-- WHERE concat(substring(b.bdp_day,1,4),'-',(b.bdp_day,5,2),'-',(b.bdp_day,7,2)) IN (date_add(current_date(),-1),
--                                                                                    current_date()) ;
WHERE FROM_UNIXTIME(unix_timestamp(b.bdp_day, 'yyyyMMdd'),'yyyyMMdd')


INSERT INTO qianjin_develop.test0719_app_track
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
FROM qianjin_develop.test07191_track a
LEFT JOIN qianjin_develop.app_log_action_track c ON a.pno = c.pid
-- AND concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) = concat(substring(c.dateid,1,4),'-',substring(c.dateid,5,2),'-',substring(c.dateid,7,2))
AND FROM_UNIXTIME(unix_timestamp(a.create_date_id, 'yyyyMMdd'),'yyyyMMdd') = FROM_UNIXTIME(unix_timestamp(c.dateid, 'yyyyMMdd'),'yyyyMMdd')
GROUP BY a.user_id,
         a.pno;

 -- is login after waihu

INSERT INTO qianjin_develop.test0719_waihu_track
SELECT a.user_id,
       a.pno,
       MAX (CASE
                WHEN a.last_calltime IS NOT NULL
                     AND a.last_calltime < b.actiontime
                     AND to_date(a.last_calltime) = to_date(b.actiontime)
                     AND pageid !='060000' THEN 1
                ELSE 0
            END) isactive_aftercall
FROM qianjin_develop.test07191_track a
LEFT JOIN qianjin_develop.app_log_action_track b ON a.pno = b.pid
GROUP BY a.user_id,
         a.pno;

 -- count gap day

INSERT INTO qianjin_develop.test0719_gap_1_track
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
FROM qianjin_develop.test07191_track a
LEFT JOIN qianjin_develop.app_log_action_track b ON a.pno = b.pid;


INSERT INTO qianjin_develop.test0719_gap_2_track
SELECT *,
       row_number() over (partition BY user_id
                          ORDER BY active_dateid) rank
FROM qianjin_develop.test0719_gap_1_track
WHERE active_dateid IS NOT NULL;


INSERT INTO qianjin_develop.test0719_gap_track
SELECT user_id,
       pno,
       count(CASE WHEN active_dateid IS NOT NULL THEN user_id END) active_num ,
       MAX(CASE
               WHEN date_id IS NOT NULL
                    -- AND active_dateid < concat(substring(date_id,1,4),'-',substring(date_id,5,2),'-',substring(date_id,7,2)) THEN date_id
                    AND active_dateid < FROM_UNIXTIME(unix_timestamp(date_id, 'yyyyMMdd'),'yyyyMMdd') THEN date_id
           END) Lastday_before_invest
FROM qianjin_develop.test0719_gap_2_track
GROUP BY user_id,
         pno;

 -- count lianxu login day

INSERT INTO qianjin_develop.test0719_lianxu_track
SELECT a.user_id,
       a.pno,
       MAX(a.rank) - MAX(CASE
                             WHEN c.active_Num =1 THEN 1
                             WHEN (datediff(a.active_dateid,b.active_dateid) !=1
                                   OR b.active_dateid IS NULL)
                                  AND c.active_Num !=1 THEN a.rank
                         END) lianxuday_last
FROM qianjin_develop.test0719_gap_2_track a
JOIN qianjin_develop.test0719_gap_track c ON a.user_Id = c.user_id
LEFT JOIN qianjin_develop.test0719_gap_2_track b ON a.user_id = b.user_id
AND a.rank = b.rank+1
GROUP BY a.user_id,
         a.pno,
         c.active_Num;

 -- final

INSERT INTO qianjin_develop.test07191_final_1_track
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
FROM qianjin_develop.test07191_track a
JOIN qianjin_develop.test0719_app_track b ON a.user_id = b.user_Id
JOIN qianjin_develop.test0719_waihu_track c ON a.user_id = c.user_id
LEFT JOIN qianjin_develop.test0719_gap_track d ON a.user_id = d.user_id
LEFT JOIN qianjin_develop.test0719_lianxu_track e ON a.user_id = e.user_id
LEFT JOIN
  (SELECT a.channel_id,
          a.num/b.num rate
   FROM
     (SELECT channel_id,
             count(DISTINCT a.user_id) num
      FROM qianjin_mid.qianjin_mid_user_detail_temp a
      JOIN qianjin_mid.qianjin_mid_user_invest_detail b ON a.user_id = b.user_id
      AND date_id = create_date_id
      -- WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN date_add(current_date(),-29) AND date_add(current_date(),-1)
      WHERE FROM_UNIXTIME(unix_timestamp(a.create_date_id, 'yyyyMMdd'),'yyyyMMdd') BETWEEN date_add(current_date(),-29) AND date_add(current_date(),-1)
        AND a.bdp_type = 's'
        AND b.bdp_type = 's'
        AND a.bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
        AND b.bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
        AND b.rank_id = 1
      GROUP BY channel_id) a
   JOIN
     (SELECT channel_id,
             count(DISTINCT a.user_id) num
      FROM qianjin_mid.qianjin_mid_user_detail_temp a
      -- WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN date_add(current_date(),-29) AND date_add(current_date(),-1)
      WHERE FROM_UNIXTIME(unix_timestamp(a.create_date_id, 'yyyyMMdd'),'yyyyMMdd') BETWEEN date_add(current_date(),-29) AND date_add(current_date(),-1)
        AND a.bdp_type = 's'
        AND a.bdp_day = FROM_UNIXTIME(unix_timestamp(current_date(), 'yyyy-MM-dd'),'yyyyMMdd')
      GROUP BY channel_id) b ON a.channel_id = b.channel_id) f ON a.channel_id = f.channel_id;


INSERT INTO qianjin_develop.test07191_final_new_track
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
FROM qianjin_develop.test07191_final_1_track a;


DROP TABLE qianjin_develop.test_yimei0719_track;


DROP TABLE qianjin_develop.test_yimei07191_track;


DROP TABLE qianjin_develop.test_yimei07192_track;


DROP TABLE qianjin_develop.test_waihu0719_track;


DROP TABLE qianjin_develop.test0719_track;


DROP TABLE qianjin_develop.test07191_track;


DROP TABLE qianjin_develop.test0719_app_track;


DROP TABLE qianjin_develop.test0719_waihu_track;


DROP TABLE qianjin_develop.test0719_gap_1_track;


DROP TABLE qianjin_develop.test0719_gap_2_track;


DROP TABLE qianjin_develop.test0719_gap_track;


DROP TABLE qianjin_develop.test0719_lianxu_track;


DROP TABLE qianjin_develop.test07191_final_1_track;


DROP TABLE qianjin_develop.app_log_action_track;

