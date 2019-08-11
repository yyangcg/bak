use qianjin_develop;

-- create tables

create table qianjin_develop.test_yimei0719_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100));


create table qianjin_develop.test_yimei07191_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int);

create table qianjin_develop.test_yimei07192_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int,
invited_id int,
isinvited int,
rank_id int,
lastday_invite int,
lastamt_invite int);


create table qianjin_develop.test_waihu0719_track
(
user_id varchar(100),
status int,
max_call_time timestamp,
min_call_time timestamp,
call_times int);

create table qianjin_develop.test0719_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int,
invited_id int,
isinvited int,
rank_id int,
lastday_invite int,
lastamt_invite int,
callstatus int);

create table qianjin_develop.test07191_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int,
invited_id int,
isinvited int,
rank_id int,
lastday_invite int,
lastamt_invite int,
callstatus int,
call_effec varchar(100),
Gap_CalltoInvest int,
last_calltime timestamp,
call_times int);

create table qianjin_develop.test0719_app_track
(user_Id varchar(100),
pno varchar(100),
staytime INT,
staynum int,
staytime_recharge int,
staynum_recharge int,
staytime_p3 int,
staynum_p3 int,
staytime_p6 int,
staynum_p6 int,
staytime_p12 int,
staynum_p12 int,
staytime_ph int,
staynum_ph int,
staytime_p1 int,
staynum_p1 int,
staytime_buy int,
staynum_buy int,
staytime_asset int,
staynum_asset int,
staytime_redbag int,
staynum_redbag int,
staytime_hudong int,
staynum_hudong int,
staynum_banner int,
staynum_icon1 int,
staynum_icon2 int,
staynum_icon3 int ,
devicetype varchar(100)
);



create table qianjin_develop.test0719_waihu_track
(user_Id varchar(100),
pno varchar(100),
isactive_aftercall int 
);

create table qianjin_develop.test0719_gap_1_track
(user_Id varchar(100),
pno varchar(100),
create_date_id int,
date_id int,
active_dateid varchar(100)
);

create table qianjin_develop.test0719_gap_2_track
(user_Id varchar(100),
pno varchar(100),
create_date_id int,
date_id int,
active_dateid varchar(100),
rank int
);

create table qianjin_develop.test0719_gap_track
(user_Id varchar(100),
pno varchar(100),
active_num int,
Lastday_before_invest INT
);

create table qianjin_develop.test0719_lianxu_track
(user_Id varchar(100),
pno varchar(100),
lianxuday_last INT);

create table qianjin_develop.test07191_final_1_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int,
invited_id int,
isinvited int,
rank_id int,
lastday_invite int,
lastamt_invite int,
callstatus int,
call_effec varchar(100),
Gap_CalltoInvest int,
last_calltime timestamp,
call_times int,
staytime INT,
staynum int,
staytime_recharge int,
staynum_recharge int,
staytime_p3 int,
staynum_p3 int,
staytime_p6 int,
staynum_p6 int,
staytime_p12 int,
staynum_p12 int,
staytime_ph int,
staynum_ph int,
staytime_p1 int,
staynum_p1 int,
staytime_buy int,
staynum_buy int,
staytime_asset int,
staynum_asset int,
staytime_redbag int,
staynum_redbag int,
staytime_hudong int,
staynum_hudong int,
staynum_banner int,
staynum_icon1 int,
staynum_icon2 int,
staynum_icon3 int ,
devicetype varchar(100),
isactive_aftercall int,
active_num int,
Lastday_before_invest INT,
lianxuday_last INT,
rate double,
Get_date_id int);


create table qianjin_develop.test07191_final_new_track
(user_Id varchar(100),
mobile varchar(100),
IsInvested varchar(10),
user_age int,
sex_id int,
create_date_Id int,
create_time timestamp,
date_id int,
invest_time timestamp,
areacode varchar(20),
plus_date int,
client_type_Id int,
channel_id int,
pno varchar(100),
isrecharged int,
rechargestatus int,
isinvited int,
lastday_invite int,
lastamt_invite int,
callstatus int,
call_effec varchar(100),
Gap_CalltoInvest int,
last_calltime timestamp,
call_times int,
staytime INT,
staynum int,
staytime_recharge int,
staynum_recharge int,
staytime_p3 int,
staynum_p3 int,
staytime_p6 int,
staynum_p6 int,
staytime_p12 int,
staynum_p12 int,
staytime_ph int,
staynum_ph int,
staytime_p1 int,
staynum_p1 int,
staytime_buy int,
staynum_buy int,
staytime_asset int,
staynum_asset int,
staytime_redbag int,
staynum_redbag int,
staytime_hudong int,
staynum_hudong int,
staynum_banner int,
staynum_icon1 int,
staynum_icon2 int,
staynum_icon3 int ,
devicetype varchar(100),
isactive_aftercall int,
active_num int,
Lastday_before_invest INT,
lianxuday_last INT,
rate double);



create table qianjin_develop.app_log_action_track
(pid varchar(100),
 userid int,
 dateid varchar(20),
 pageid varchar(100),
 lab varchar(5),
 act varchar(200),
 actiontime varchar(100),
 exittime varchar(100),
 clienttype int,
 devicetype varchar(100)
);


-- begin

insert into test_yimei0719_track
select 
distinct b.user_id
,b.mobile,case when e.user_id is not null then '已投资' else '未投资' end IsInvested,
b.user_age,b.sex_id,
b.create_date_Id, from_unixtime(b.create_time/1000) create_time,
e.date_id, from_unixtime(invest_time/1000) invest_time,
b.city_id areacode,
case when e.date_Id is null then 'NA'
     else datediff(concat(substring(date_Id,1,4),'-',substring(date_Id,5,2),'-',substring(date_Id,7,2)),concat(substring(create_date_Id,1,4),'-',substring(create_date_Id,5,2),'-',substring(create_date_Id,7,2))) end plus_date,
b.client_type_id,b.channel_Id
,f.pno
from  qianjin_mid.qianjin_mid_user_detail_temp  b
left join (select * from qianjin_mid.qianjin_mid_user_invest_detail where rank_id = 1 and bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's') e
on b.user_id = e.user_Id
left join (select distinct user_id,max(case when type = 2 then pid else pno end) pno from qianjin_mid.qianjin_mid_app_token where pno is not null and bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's' group by user_id) f
on b.user_id = f.user_id
where concat(substring(b.create_date_id,1,4),'-',(b.create_date_id,5,2),'-',(b.create_date_id,7,2)) = date_add(current_date(),-1) and b.channel_categ_id != 73 and b.bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and b.bdp_type = 's'  ;

insert into test_yimei07191_track
select distinct a.*,max(case when b.user_id is not null then 1 else 0 end) isrecharged,
max(b.recharge_status_id) rechargestatus
from test_yimei0719_track a
left join qianjin_mid.qianjin_mid_user_recharge_detail b
on a.user_id = b.user_id
and to_date(concat(substring(a.create_date_Id,1,4),'-',substring(a.create_date_Id,5,2),'-',substring(a.create_date_Id,7,2))) = to_date(from_unixtime(b.recharge_time/1000))
group by a.user_Id,a.mobile,a.isinvested,a.user_age,a.sex_id,a.create_date_id,a.create_time,a.date_id,a.invest_time,a.areacode,a.plus_date,a.client_type_Id,a.channel_id,a.pno ;

insert into qianjin_develop.test_yimei07192_track
select distinct a.*
,case when a.invest_time is not null  then datediff(concat(substring(a.date_id,1,4),'-',substring(a.date_id,5,2),'-',substring(a.date_id,7,2)),concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2))) 
  else datediff(current_date(), concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2))) end  Lastday_invite
,b.invest_amt Lastamt_invite
from (
select  a.user_Id,a.mobile,a.isinvested,a.user_age,a.sex_id,a.create_date_id,a.create_time,a.date_id,a.invest_time,a.areacode,a.plus_date,a.client_type_Id,a.channel_id,a.pno,a.isrecharged,a.rechargestatus,b.invited_id,
case when b.invited_Id is not null then 1 else 0 end IsInvited
,max(C.rank_id) rank_id 
from test_yimei07191_track a
join (select * from qianjin_mid.qianjin_mid_user_detail_temp where  bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's')  b
on a.user_id = b.user_id 
left join (select * from qianjin_mid.qianjin_mid_user_invest_detail where bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's') c
on b.invited_id = c.user_Id
and (a.invest_time >= from_unixtime(c.invest_time/1000) or a.invest_time is null)
group by a.user_Id,a.mobile,a.isinvested,a.user_age,a.sex_id,a.create_date_id,a.create_time,a.date_id,a.invest_time,a.areacode,a.plus_date,a.client_type_Id,a.channel_id,a.pno,a.isrecharged,a.rechargestatus,b.invited_id) a
left join (select * from qianjin_mid.qianjin_mid_user_invest_detail where bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's') b
on a.invited_id = b.user_id 
and a.rank_id = b.rank_id;


--  添加电销相关标签，电销数据来自46库 waihu date
insert into qianjin_develop.test_waihu0719_track
select user_id,max(phone_status) status, max(from_unixtime(cast(create_time/1000 as int))) max_call_time
, min(from_unixtime(cast(create_time/1000 as int))) min_call_time,count(*) call_times
from (select distinct user_id,phone_status,create_time,call_type from qianjin_crm.qianjin_crm_outcall_work_order where bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2)) and bdp_type = 's') a
where call_type = 1
group by user_id;

 -- is waihu ?
insert into qianjin_develop.test0719_track
select distinct a.*
,case when b.call_times is not null and a.invest_time is not null and (b.min_call_time > a.create_time or a.invest_time > b.max_call_time) then 1
      when b.call_times is not null and a.invest_time is null and (b.min_call_time > a.create_time or b.max_call_time > a.create_time) then 1
      else 0 end callstatus
from qianjin_develop.test_yimei07192_track a
left join qianjin_develop.test_waihu0719_track b
on a.user_Id = b.user_id;

-- waihu reusult
insert into qianjin_develop.test07191_track
select distinct a.*
,case when b.status = 3 and a.callstatus = 1 then '成功'
      when b.status = 4 and a.callstatus = 1  then '预约下次'
      when b.status = 5 and a.callstatus = 1  then '拒访'
      else '失败' end call_effec
,case when a.callstatus = 1 then datediff(a.invest_time,max_call_time) end Gap_CalltoInvest
,case when a.callstatus = 1 and a.invest_time >= b.max_call_time then b.max_call_time
      when a.callstatus = 1 and a.invest_time < b.max_call_time then b.min_call_time
      when a.callstatus = 1 and a.invest_time is null then b.max_call_time end last_calltime
,b.call_times
from qianjin_develop.test0719_track a
left join qianjin_develop.test_waihu0719_track b
on a.user_Id = b.user_id ;



-- add app_Log data
insert into qianjin_develop.app_log_action_track
select pid,userid,dateid,pageid,lab,act,actiontime,exittime,clienttype,devicetype from qianjin.qianjin_app_log 
where concat(substring(b.bdp_day,1,4),'-',(b.bdp_day,5,2),'-',(b.bdp_day,7,2)) In ( date_add(current_date(),-1),current_date()) ;


insert into qianjin_develop.test0719_app_track
select a.user_id,a.pno,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and exittime>=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime,
count(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and substring(pageid,1,1) != '0' then pageid end) staynum,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('RechargeActivity','RechargeMainViewController',
'JoinConfirmRechargeNewActivity','BuyConfirmViewController') and exittime >=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_recharge,
count(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('RechargeActivity','RechargeMainViewController',
'JoinConfirmRechargeNewActivity','BuyConfirmViewController')and substring(pageid,1,1) != '0' then pageid end) staynum_recharge,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity3','ProductionDetailDPViewController_3') and exittime >=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_p3,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity3','ProductionDetailDPViewController_3')and substring(pageid,1,1) != '0' then pageid end) staynum_p3,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity6','ProductionDetailDPViewController_6') and exittime >=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_p6,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity6','ProductionDetailDPViewController_6')and substring(pageid,1,1) != '0' then pageid end) staynum_p6,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity12','ProductionDetailDPViewController_12') and exittime >=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_p12,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity12','ProductionDetailDPViewController_12')and substring(pageid,1,1) != '0' then pageid end) staynum_p12,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('IHuoBaoDetailActivity','ProductionDetailCTViewController') and exittime >= actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_ph,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('IHuoBaoDetailActivity','ProductionDetailCTViewController')and substring(pageid,1,1) != '0' then pageid end) staynum_ph,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity1','ProductionDetailDPViewController_1') and exittime >=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_p1,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('DPlanActivity1','ProductionDetailDPViewController_1')and substring(pageid,1,1) != '0' then pageid end) staynum_p1,

sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('JoinDepositActivity3','BuyDPViewController_3'
,'JoinDepositActivity6','BuyDPViewController_6'
,'JoinDepositActivity12','BuyDPViewController_12'
,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController') and exittime>=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_buy,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('JoinDepositActivity3','BuyDPViewController_3'
,'JoinDepositActivity6','BuyDPViewController_6'
,'JoinDepositActivity12','BuyDPViewController_12'
,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController')and substring(pageid,1,1) != '0' then pageid end) staynum_buy,


sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('AssetsNewActivity','AssetsMainViewController'
) and exittime>=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_asset,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('AssetsNewActivity','AssetsMainViewController')and substring(pageid,1,1) != '0' then pageid end) staynum_asset,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('MyRewardActivity','RewardViewController'
) and exittime>=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_redbag,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('MyRewardActivity','RewardViewController') and substring(pageid,1,1) != '0' then pageid end) staynum_redbag,
sum(case when  (actiontime <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('InterActionActivity','InteractionMainViewController'
) and exittime>=actiontime then (unix_timestamp(exittime)-unix_timestamp(actiontime)) end) staytime_hudong,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid in ('InterActionActivity','InteractionMainViewController')and substring(pageid,1,1) != '0' then pageid end) staynum_hudong,

count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid ='010000' and act in ('点击banner','点击baner') then pageid end) staynum_banner,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid ='010000' and act in ('点击icon') and (c.clienttype =2 and c.lab = 1 or c.clienttype =3 and c.lab = 0)  then pageid end) staynum_icon1,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid ='010000' and act in ('点击icon') and (c.clienttype =2 and c.lab = 2 or c.clienttype =3 and c.lab = 1)  then pageid end) staynum_icon2,
count(case when  (actiontime  <=invest_time or invest_time is null or a.invest_time ='') and pageid ='010000' and act in ('点击icon') and (c.clienttype =2 and c.lab = 3 or c.clienttype =3 and c.lab = 2)  then pageid end) staynum_icon3,
max(c.devicetype) devicetype

from qianjin_develop.test07191_track a
left join qianjin_develop.app_log_action_track c
on a.pno = c.pid
and concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) =
concat(substring(c.dateid,1,4),'-',substring(c.dateid,5,2),'-',substring(c.dateid,7,2))
group by a.user_id,a.pno;


-- is login after waihu
insert into qianjin_develop.test0719_waihu_track
select a.user_id,a.pno,
max (case when a.last_calltime is not null and a.last_calltime < b.actiontime and to_date(a.last_calltime) = to_date(b.actiontime) and pageid !='060000'
then 1 else 0 end) isactive_aftercall
from qianjin_develop.test07191_track a
left join qianjin_develop.app_log_action_track b
on a.pno = b.pid
group by a.user_id,a.pno;

-- count gap day
insert into qianjin_develop.test0719_gap_1_track
select distinct a.user_id,a.pno,a.create_date_Id,a.date_id,
case when b.actiontime between  a.create_time and a.invest_time  then to_date(b.actiontime)
when b.actiontime >=  a.create_time and (a.invest_time is null or a.invest_time ='') and to_date(b.actiontime) <= date_add(a.create_time,30) then to_date(b.actiontime)
 end active_dateid
from qianjin_develop.test07191_track a
left join qianjin_develop.app_log_action_track b
on a.pno = b.pid;

insert into qianjin_develop.test0719_gap_2_track
select *,row_number() over (partition by user_id ORDER BY active_dateid) rank from qianjin_develop.test0719_gap_1_track where active_dateid is not null;

insert into qianjin_develop.test0719_gap_track
select user_id,pno,count(case when active_dateid is not null then user_id end) active_num
,max(case when date_id is not null and active_dateid < concat(substring(date_id,1,4),'-',substring(date_id,5,2),'-',substring(date_id,7,2)) then date_id end) Lastday_before_invest
from qianjin_develop.test0719_gap_2_track
group by user_id,pno;


-- count lianxu login day
insert into qianjin_develop.test0719_lianxu_track
select a.user_id,a.pno,max(a.rank) - max(case when c.active_Num =1 then 1
 when (datediff(a.active_dateid,b.active_dateid) !=1 or b.active_dateid is null) and c.active_Num !=1  then a.rank end) lianxuday_last
from qianjin_develop.test0719_gap_2_track a
join qianjin_develop.test0719_gap_track c
on a.user_Id = c.user_id
left join qianjin_develop.test0719_gap_2_track b
on a.user_id = b.user_id
and a.rank = b.rank+1
group by a.user_id,a.pno,c.active_Num;


-- final  

insert into qianjin_develop.test07191_final_1_track
select a.*,b.staytime,b.staynum,b.staytime_recharge,b.staynum_recharge,
b.staytime_p3,b.staynum_p3,b.staytime_p6,b.staynum_p6,b.staytime_p12,b.staynum_p12,
b.staytime_ph,b.staynum_ph,b.staytime_p1,b.staynum_p1,b.staytime_buy,b.staynum_buy,
b.staytime_asset,b.staynum_asset,b.staytime_redbag,b.staynum_redbag,
b.staytime_hudong,b.staynum_hudong,b.staynum_banner,b.staynum_icon1,b.staynum_icon2,b.staynum_icon3,b.devicetype,
c.isactive_aftercall,
d.active_num,d.Lastday_before_invest,
e.lianxuday_last,f.rate,'20170818' as Get_date_id
from qianjin_develop.test07191_track a
join qianjin_develop.test0719_app_track b
on a.user_id = b.user_Id
join qianjin_develop.test0719_waihu_track c
on a.user_id = c.user_id
left join qianjin_develop.test0719_gap_track d
on a.user_id = d.user_id
left join qianjin_develop.test0719_lianxu_track e
on a.user_id = e.user_id
left join (select a.channel_id,a.num/b.num rate from (
select channel_id,count(distinct a.user_id) num from qianjin_mid.qianjin_mid_user_detail_temp a
join qianjin_mid.qianjin_mid_user_invest_detail b
on a.user_id = b.user_id
and date_id = create_date_id
where concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) between date_add(current_date(),-29) and date_add(current_date(),-1) and a.bdp_type = 's' and b.bdp_type = 's' 
and a.bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
and b.bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
and b.rank_id = 1
group by channel_id) a
join
(
select channel_id,count(distinct a.user_id) num
from qianjin_mid.qianjin_mid_user_detail_temp a
where concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) between date_add(current_date(),-29) and date_add(current_date(),-1) and a.bdp_type = 's'
and a.bdp_day = concat(substring(current_date(),1,4),substring(current_date(),6,2),substring(current_date(),9,2))
group by channel_id) b
on a.channel_id = b.channel_id) f
on a.channel_id = f.channel_id;


insert into qianjin_develop.test07191_final_new_track
select a.user_id,a.mobile,a.isinvested,a.user_age,a.sex_id,a.create_date_id,a.create_time,
a.date_id,a.invest_time,a.areacode,a.plus_date,a.client_type_Id,a.channel_id,a.pno,
a.isrecharged,a.rechargestatus,a.isinvited,a.lastday_invite,a.lastamt_invite,a.callstatus,
a.call_effec,a.gap_calltoinvest,a.last_calltime,a.call_times,a.staytime,a.staynum,a.staytime_recharge,
a.staynum_recharge,a.staytime_p3,a.staynum_p3,a.staytime_p6,a.staynum_p6,a.staytime_p12,a.staynum_p12,
a.staytime_ph,a.staynum_ph,a.staytime_p1,a.staynum_p1,a.staytime_buy,a.staynum_buy,a.staytime_asset,a.staynum_asset,
a.staytime_redbag,a.staynum_redbag,a.staytime_hudong,a.staynum_hudong,a.staynum_banner,a.staynum_icon1,a.staynum_icon2,
a.staynum_icon3,a.devicetype,a.isactive_aftercall,a.active_num,a.lastday_before_invest,a.lianxuday_last,a.rate from qianjin_develop.test07191_final_1_track a;



drop table qianjin_develop.test_yimei0719_track;
drop table qianjin_develop.test_yimei07191_track;
drop table qianjin_develop.test_yimei07192_track;
drop table qianjin_develop.test_waihu0719_track;
drop table qianjin_develop.test0719_track;
drop table qianjin_develop.test07191_track;
drop table qianjin_develop.test0719_app_track;
drop table qianjin_develop.test0719_waihu_track;
drop table qianjin_develop.test0719_gap_1_track;
drop table qianjin_develop.test0719_gap_2_track;
drop table qianjin_develop.test0719_gap_track;
drop table qianjin_develop.test0719_lianxu_track;
drop table qianjin_develop.test07191_final_1_track;
drop table qianjin_develop.app_log_action_track;



