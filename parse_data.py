'''
py.exe .\parse_data.py --name dlt --debug 1 --ball_color red --cycle_row 3 --dleft 3
py.exe .\parse_data.py --name dlt --debug 2 --ball_color red --cycle_row 3 --dleft 1
py.exe .\parse_data.py --name kl8 --debug 1 --cycle_row 2   --dleft 0
'''
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as iters
from loguru import logger
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--name',default='dlt',type=str,help="choose lottery name")
parser.add_argument('--ball_color',default='red',type=str,help="choose ball color")
parser.add_argument('--cycle_row',type=int,help="cycle window size")
parser.add_argument('--offset_step',type=int,help="row offset step")
parser.add_argument('--row_offset',type=int,help="row offset")
parser.add_argument('--col_offset',type=int,help="col offset")
parser.add_argument('--debug',default=0,type=int,help="choose lottery name")
parser.add_argument('--dleft',default=0,type=int,help="choose lottery name")
args = parser.parse_args()

def cal_dupzone(ds,col_s,col_e):
    ds_li = ds[col_s:col_e].tolist()
    is_dupzone = len(ds_li) - len(set(ds_li))
    return is_dupzone

def cal_dvalue(ds,col_s,col_e):
    ds_li = ds[col_s:col_e].tolist()
    len_ds_li = len(ds_li) - 1
    dvalue_li = []
    for idx in range(0,len_ds_li):
        #填充差值列表
        dvalue = ds_li[idx+1] - ds_li[idx]
        dvalue_li.append(dvalue)
    return ','.join(map(str,dvalue_li))

def cal_bit(ds,col_s,col_e):
    ds_li = ds[col_s:col_e].tolist()
    bit_str = "".join(map(lambda x: str(x//10%10),ds_li))
    return bit_str

#线性关系计算
def cal_arith_geo_prog(ds,col_s,col_e):
    '''
    xyz--->(z-y)/(y-x) = f
    ds:pandas.core.series.Series
    return:str
    '''
    factor_li = []
    offset_li = []
    ds_li = ds[col_s:col_e].tolist()
    len_ds_li = len(ds_li)
    for idx in range(0,len_ds_li-2):
        #判断整除
        factor = 0
        r = (ds_li[idx+2] - ds_li[idx+1]) % (ds_li[idx+1] - ds_li[idx])
        if r == 0:
            factor = (ds_li[idx+2] - ds_li[idx+1]) // (ds_li[idx+1] - ds_li[idx])
        offset = ds_li[idx+1] - ds_li[idx] * factor
        factor_li.append(factor)
        offset_li.append(offset)
    return ','.join(map(str,factor_li))+'|'+','.join(map(str,offset_li))

def show_numrange_stats_incol(df,col_s,col_e,col_end,t_settings):
    if "numrange" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_numrange_stats_incol")
        return None
    t_settings[args.ball_color]['num_range'].append({'col_s':col_s,'min':df.iloc[:,col_s].min(),'max':df.iloc[:,col_e-1].max()})
    if col_e == col_end and args.debug != 0:
        df_numrange = pd.DataFrame(t_settings[args.ball_color]['num_range'])
        file_name = t_settings[args.ball_color]['debug_data_path'] + 'numrange_debug_ds.csv'
        file_dir = os.path.dirname(file_name)
        os.makedirs(file_dir,exist_ok=True)
        df_numrange.to_csv(file_name,index=False)

#前/后区重叠数
def show_dupzone_stats_inrow(df,col_e,t_settings):
    if "dupzone" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_dupzone_stats_inrow")
        return None 
    col_s = t_settings['red']['start_col_idx']
    df['dupzone'] = df.apply(cal_dupzone,result_type='expand',axis=1,args=(col_s,col_e))
    dup_zone_cnt_dict = df['dupzone'].value_counts().to_dict()
    logger.info("blue zone same as red zone cnt:{}",dup_zone_cnt_dict)

#相邻数差值统计
def show_dvalue_stats_inrow(df,col_s,col_e,t_settings):
    if "dvalue" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_dvalue_stats_inrow")
        return None
    df['dvalue'] = df.apply(cal_dvalue,result_type='expand',axis=1,args=(col_s,col_e))
    dvalue_cnt_ds = df['dvalue'].value_counts()

    #开始绘图
    fig,axes = plt.subplots(1,1,figsize=(7,7))
    fig.canvas.manager.set_window_title('行差序列统计')
    axes.bar(dvalue_cnt_ds.index,dvalue_cnt_ds.values,tick_label=dvalue_cnt_ds.index,color='g',width=0.25)
    axes.set_ylabel("dvalue//cnt")
    for index,val in dvalue_cnt_ds.items():
        axes.text(index,val+1,val,color='red',fontsize=8,ha='center',va='center') 
    axes.grid()
    plt.subplots_adjust(left=0.08,right=0.94,top=1,bottom=0.17)
    #结束绘图

def show_bit_stats_inrow(df,col_s,col_e,t_settings):
    if "bit" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_bit_stats_inrow")
        return None
    df['bit'] = df.apply(cal_bit,result_type='expand',axis=1,args=(col_s,col_e))
    if args.debug != 0:
        bit_cnt_ds = df['bit'].value_counts(normalize=True)
        file_name = t_settings[args.ball_color]['debug_data_path']+'bit_debug_ds_'+str(col_s)+'.csv'
        file_dir = os.path.dirname(file_name)
        os.makedirs(file_dir,exist_ok=True)
        bit_cnt_ds.to_csv(file_name)
    #debug模式，选10bit高频数据前90%?????

#对各行应用cal_arith_geo_prog函数
def show_arith_geo_stats_inrow(df,col_s,col_e,t_settings):
    if "arith_geo" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_arith_geo_stats_inrow")
        return None
    df['arith_geo'] = df.apply(cal_arith_geo_prog,result_type='expand',axis=1,args=(col_s,col_e))

    df_li = []
    for val in df['arith_geo']:
        li_str = val.split('|')
        factor_li = li_str[0].split(',')
        offset_li = li_str[1].split(',')
        fac_off_li = list(zip(factor_li,offset_li))
        zero_idx = np.where(np.array(factor_li) == '0')[0]
        if(len(zero_idx) == len(factor_li)):
            #arith_geo0
            df_li.append([0,0])
        else:
            #arith_geo3,...
            len_arith_geo = 2 +max([len(list(v)) for k,v in iters.groupby(fac_off_li)])
            fac_off_li = list(dict.fromkeys(fac_off_li))

            df_li.append([len_arith_geo,len(fac_off_li),list(map(lambda x:','.join(map(str,x)),fac_off_li))])

    fo_df = pd.DataFrame(df_li,columns=['seqlen','foc','fol'])
    #debug模式，筛选线性关系列数据
    if args.debug != 0:
        file_name = t_settings[args.ball_color]['debug_data_path']+'fo_debug_ds_'+str(col_s)+'.csv'
        file_dir = os.path.dirname(file_name)
        os.makedirs(file_dir,exist_ok=True)
        fo_df[fo_df['seqlen'] != 0].to_csv(file_name,index=False)
    #开始绘图
    fig,axes = plt.subplots(2,1,figsize=(7,7))
    fig.canvas.manager.set_window_title("行线性序列统计")
    #0:seq max len distribution
    hist_min = fo_df['seqlen'].values.min()
    hist_max = fo_df['seqlen'].values.max() + 1
    hist_y,hist_x,hist_patch = axes[0].hist(fo_df['seqlen'],bins=hist_max,range=(hist_min,hist_max),color='green')
    #标注seq max len count
    for i in range(len(hist_patch)):
        axes[0].text(x=hist_x[i],y=hist_y[i]+5,s=int(hist_y[i]),ha='center')
    axes[0].set_xticks(range(hist_min,hist_max))
    axes[0].set_ylabel("seqlen//seqlen_cnt")
    axes[0].grid()

    #1:(factor,offset) class distribution
    hist_min = fo_df['foc'].values.min()
    hist_max = fo_df['foc'].values.max() + 1
    hist_y,hist_x,hist_patch =axes[1].hist(fo_df['foc'],bins=hist_max,range=(hist_min,hist_max),color='green')
    #标注(factor,offset) class count
    for i in range(len(hist_patch)):
        axes[1].text(x=hist_x[i],y=hist_y[i]+5,s=int(hist_y[i]),ha='center')
    axes[1].set_xticks(range(hist_min,hist_max))
    axes[1].set_ylabel("foc//foc_cnt")
    axes[1].grid()
    plt.subplots_adjust(left=0.08,right=0.94,top=1,bottom=0.17)

#统计周期组，对应颜色球出现次数
def show_cycle_line_stats(df,col_s,col_e,t_settings):
    if "cycleline" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_cycle_line_stats")
        return None
    len_data = df.shape[0]
    #行配置
    if not args.cycle_row:
        cycle_row = t_settings[args.ball_color]['cycle_row']
    else:
        cycle_row = args.cycle_row
    if not args.offset_step:
        offset_step = cycle_row
    else:
        offset_step = args.offset_step
    if not args.row_offset:
        row_offset = (len_data - (cycle_row - 1)) % cycle_row
    else:
        row_offset = args.row_offset
    #开始统计
    zone_len_list = []
    zone_dup_list = []
    for i in range(row_offset,len_data,offset_step):
        if i + cycle_row < len_data:
            zone_counts = df.iloc[i:i+cycle_row,col_s:col_e].stack().value_counts()
            len_zone_count = len(zone_counts.values)
            zone_len_list.append(len_zone_count)

            if args.debug != 0:
                #debug mode 执行，重点看周期行对应len data
                if len_zone_count not in [12,13,14]:
                    continue
            if zone_counts[zone_counts>1].values.size == 0:
                zone_dup_list.append('0')
            else:
                #数组转字符串，用于比较
                zone_dup_list.append(''.join(map(str,zone_counts[zone_counts>1].values)))
    #结束统计
                
    #开始绘图
    #number class count
    z_len_ds = pd.Series(zone_len_list)
    z_len_feq_ds = z_len_ds.value_counts()
    #dup number class count
    z_dup_feq_ds = pd.Series(zone_dup_list).value_counts()

    #折线图统计趋势，直方图统计频率
    fig,axes = plt.subplots(3,1,figsize=(7,7))
    fig.canvas.manager.set_window_title(str(cycle_row)+"周期行统计")

    #0:number class count change trend
    axes[0].plot(z_len_ds.index,z_len_ds.values,label='trend',linewidth=0.8,marker='o',markersize=3)
    axes[0].set_ylabel("index//class_num")
    axes[0].set_yticks(range(z_len_ds.min(),z_len_ds.max()+1))
    axes[0].grid()
    axes[0].legend()

    #1:number class count distribution
    axes[1].bar(z_len_feq_ds.index,z_len_feq_ds.values,tick_label=z_len_feq_ds.index,color='g',width=0.25)
    axes[1].set_ylabel("class_num//cnt")
    for index,val in z_len_feq_ds.items():
        axes[1].text(index,val+1,val,color='red',fontsize=8,ha='center',va='center')

    #2: dup number class count distribution
    axes[2].bar(z_dup_feq_ds.index,z_dup_feq_ds.values,color='g',width=0.25)
    axes[2].set_xticklabels(z_dup_feq_ds.index,rotation=80,fontsize=7)
    axes[2].set_ylabel("foucs_len:dup_class//cnt")
    for index,val in z_dup_feq_ds.items():
        axes[2].text(index,val+1,val,color='red',fontsize=8,ha='center',va='center')
    plt.subplots_adjust(left=0.08,right=0.94,top=1,bottom=0.17)
    #结束绘图

#统计重复序列行数
def show_dup_line_stats(df,col_s,col_e,t_settings):
    if "dupline" in t_settings[args.ball_color]["skip"]:
        logger.info("skip process show_dup_line_stats")
        return None
    dup_list = df.columns.tolist()
    dup_index = dup_list[col_s:col_e]
    duplicate = df[df.duplicated(dup_index)]
    logger.info("Project name:{} Zone area:{} Duplicated num:{}--Data total num:{}".format(args.name,args.ball_color,duplicate.shape[0],df.shape[0]))

if __name__ == '__main__':
    if not args.name:
        raise Exception("lottery name is null")
    elif not args.ball_color:
        raise Exception("ball_color is null")
    else:
        df = pd.read_csv(test_config[args.name]['datapath'])
        if args.dleft != 0:
            df = df.iloc[:(len(df) - args.dleft),:]
        t_settings = test_config[args.name]['settings']
        picknum = t_settings[args.ball_color]['picknum']
        col_s = t_settings[args.ball_color]['start_col_idx']
        col_e = t_settings[args.ball_color]['end_col_idx'] + 1
        if picknum > 10:
            picknum = t_settings[args.ball_color]['subpicknum']
            if args.col_offset:
                col_s = args.col_offset
                col_e = col_s+picknum

        for col_s_i in range(col_s,col_e,picknum):
            col_e_i = col_s_i +picknum
            show_dup_line_stats(df,col_s_i,col_e_i,t_settings)

            show_cycle_line_stats(df,col_s_i,col_e_i,t_settings)

            show_arith_geo_stats_inrow(df,col_s_i,col_e_i,t_settings)

            show_bit_stats_inrow(df,col_s_i,col_e_i,t_settings)

            show_dvalue_stats_inrow(df,col_s_i,col_e_i,t_settings)

            show_dupzone_stats_inrow(df,col_e_i,t_settings)

            show_numrange_stats_incol(df,col_s_i,col_e_i,col_e,t_settings)

        if args.debug != 2:
            plt.show()