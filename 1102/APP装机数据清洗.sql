use qianjin_develop;

create table qianjin_develop.test_originaldata
(userid int,
pid varchar(100),
clienttype int,
clientversion varchar(100),
appsinstalled varchar(100),
createtime bigint,
syncjsonkey varchar(500),
bdp_day varchar(100),
bdp_type varchar(10));

create table qianjin_develop.test_dataclear_step1
(userid int,
pid varchar(100),
clienttype int,
clientversion varchar(100),
appsinstalled varchar(100),
createtime bigint);

create table qianjin_develop.test_dataclear_step2
(userid int,
pid varchar(100),
clienttype int,
clientversion varchar(100),
appsinstalled varchar(100),
createtime bigint);

insert into  qianjin_develop.test_originaldata
select * from qianjin_mongodb.qianjin_appclientinstalled
where bdp_day between 20170826 and 20170901;


insert into qianjin_develop.test_dataclear_step1
select distinct a.userid,a.pid,a.clienttype,a.clientversion, 
a.appsinstalled,a.createtime
from qianjin_develop.test_originaldata a
join test_yimei0719_track b
on a.pid = b.pno;

insert into qianjin_develop.test_dataclear_step1
select distinct a.userid,a.pid,a.clienttype,a.clientversion, 
a.appsinstalled,a.createtime
from qianjin_develop.test_originaldata a
join test_yimei0719_track b
on a.userid = b.user_id;

insert into qianjin_develop.test_dataclear_step2
select distinct a.userid,a.pid,a.clienttype,a.clientversion, 
a.appsinstalled,a.createtime
from qianjin_develop.test_dataclear_step1 a;


create table qianjin_develop.test_dataclear_step3 AS
select userid,pid,from_unixtime(createtime/1000) createtime,clienttype,
substring(appsinstalled,1,instr(appsinstalled,',')-1)  appsinstalled
from qianjin_develop.test_dataclear_step2;

create table qianjin_develop.test_dataclear_step4 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,' ') > 0 then  substring(appsinstalled,1,instr(appsinstalled,' ')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step3 ;
 

create table qianjin_develop.test_dataclear_step5 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'-') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'-')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step4 ;

create table qianjin_develop.test_dataclear_step6 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'¡ª¡ª') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'¡ª¡ª')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step5 ;

create table qianjin_develop.test_dataclear_step7 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'¡ª') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'¡ª')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step6 ;

create table qianjin_develop.test_dataclear_step8 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'£¨') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'£¨')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step7 ;

create table qianjin_develop.test_dataclear_step9 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'£¨') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'£¨')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step8 ;

create table qianjin_develop.test_dataclear_step10 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'HD') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'HD')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step9 ;

create table qianjin_develop.test_dataclear_step11 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,'£­') > 0 then  substring(appsinstalled,1,instr(appsinstalled,'£­')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step10 ;

create table qianjin_develop.test_dataclear_step12 AS
select userid,pid,date(createtime) createtime,clienttype,
case when instr(appsinstalled,':') > 0 then  substring(appsinstalled,1,instr(appsinstalled,':')-1)  
else appsinstalled end appsinstalled
from qianjin_develop.test_dataclear_step11 ;


insert into qianjin_develop.test_dataclear_final1 
select a.userid,
a.pid,a.createtime,a.clienttype,a.appsinstalled
,categ,rank,sub_categ,sub_rank,company,active_user_amt
from qianjin_develop.test_dataclear_step12 a 
left join qianjin_develop.qianjin_test_appcateg b
on a.appsinstalled = b.app_name;

select count(*) from qianjin_develop.test_dataclear_final1 limit 100 ;

drop table qianjin_develop.test_originaldata;
drop table qianjin_develop.test_dataclear_step1;
drop table qianjin_develop.test_dataclear_step2;
drop table qianjin_develop.test_dataclear_step3;
drop table qianjin_develop.test_dataclear_step4;
drop table qianjin_develop.test_dataclear_step5;
drop table qianjin_develop.test_dataclear_step6;
drop table qianjin_develop.test_dataclear_step7;
drop table qianjin_develop.test_dataclear_step8;
drop table qianjin_develop.test_dataclear_step9;
drop table qianjin_develop.test_dataclear_step10;
drop table qianjin_develop.test_dataclear_step11;
drop table qianjin_develop.test_dataclear_step12;


create table qianjin_develop.test_dataclear_final as
select distinct * from qianjin_develop.test_dataclear_final1 a

select count(*) from qianjin_develop.test_dataclear_final