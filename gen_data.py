'''
py.exe .\gen_data.py --name dlt --ball_color red --debug 1 --dleft 3
py.exe .\gen_data.py --name dlt --ball_color blue --debug 1 --dleft 1 --presel_seq 1,2,3,4,5
'''
import os
import sys
import ast
import time
import random
import argparse
import numpy as np
import pandas as pd
import itertools as iters
from collections import Counter
from itertools import accumulate
from loguru import logger
from config import *

parse = argparse.ArgumentParser()
parse.add_argument('--name',default='dlt',type=str,help='choose lottery name')
parse.add_argument('--ball_color',default='red',type=str,help='choose ball color')
parse.add_argument('--offset_step',type=int,help='row offset step')
parse.add_argument('--debug',default=0,type=int,help='debug data')
parse.add_argument('--dleft',default=0,type=int,help='debug data left: test predict data')
parse.add_argument('--presel_seq',default='',type=str,help='pre select seq')
args = parse.parse_args()

def test_predictdata_dropflag(loopname,test_ds,true_seq,dropflag_dict):
    test_ds_li = test_ds.values.tolist()
    #test test_ds 是否命中？开始
    if args.dleft != 0:
        testindex = -2
        try:
            testindex = test_ds_li.index(tuple(true_seq))
        except:
            testindex = -1
        dropflag_dict[loopname] = testindex
    #test结束

def test_save_predictdata(predict_ds,true_seq,dropflag_dict,col_s,debugpath,t_precap):
    predict_ds_li = predict_ds.values.tolist()

    #save predict_ds
    predict_ds.to_csv(debugpath +'predict_ds_'+str(col_s)+'.csv',header=False,index=False)
    predict_idx = int(float(format((random.uniform(t_precap['cap_range'][0],t_precap['cap_range'][1])),'.9f')) * len(predict_ds))
    logger.info("predict_idx {} value {} in predict_ds.csv",predict_idx,predict_ds_li[predict_idx])

    #test 开始 diff_ds是否命中?
    if args.dleft != 0:
        if isinstance(true_seq,pd.Series):
            logger.info("true_seq is {}",true_seq.values)
        else:
            logger.info("true_seq is {}",true_seq)
        dropflag_li = list(dropflag_dict.values())
        final_index = dropflag_li[-1]
        if args.debug == 2:
            #log 重定向到控制台
            logger.add(sys.stdout,colorize=True)
        if final_index != -1:
            logger.info("you are shooted in predict_ds.csv col_s:{} totalcnt:{} index:{} value:{}",col_s,len(predict_ds),(final_index+1),predict_ds_li[final_index])
        else:
            try:
                first_drop_index = dropflag_li.index(-1)
                logger.info("you are missed!!! col_s:{} drop in {}",col_s,list(dropflag_dict.keys())[first_drop_index])
            except:
                logger.info("you are missed!!! col_s:{} drop in at first time",col_s)
    #test 结束

def drop_dupfrontzone_inline(df,total_ds):
    presel_seq = ()
    drop_idx_li = []
    if args.presel_seq != "":
        presel_seq = tuple(map(int,args.presel_seq.split(',')))
    else:
        r_col_s = test_config[args.name]['settings']['red']['start_col_idx']
        r_col_e = test_config[args.name]['settings']['red']['end_col_idx'] + 1
        presel_seq = tuple(df.iloc[[len(df)-args.dleft],r_col_s:r_col_e].squeeze())
    for idx,val in total_ds.items():
        temp_t = presel_seq+val
        if len(temp_t) - len(set(temp_t)) > 0:
            drop_idx_li.append(idx)
    return drop_idx_li

def drop_fo_inline(test_ds,fo_df,a_g_len_li,foc_li):
    drop_idx_li = []
    ref_foli = []
    fo_df['fol'] = fo_df['fol'].apply(ast.literal_eval)
    for idx in range(len(a_g_len_li)):
        fo_debug_ds = fo_df[(fo_df['seqlen'].isin(a_g_len_li[idx]))&(fo_df['foc'].isin(foc_li[idx]))]['fol'].explode().value_counts(normalize=True)
        rate_sum = 0.0
        sel_idx = ''
        for i,v in fo_debug_ds.items():
            rate_sum += v
            if rate_sum >=0.9:
                sel_idx = i
                break
        ref_foli.append(fo_debug_ds.loc[:sel_idx].index.tolist())

    for idx,val in test_ds.items():
        if fo_inline_need_drop(val,ref_foli,a_g_len_li,foc_li) == 1:
            drop_idx_li.append(idx)
    return drop_idx_li

def drop_dopc_cycle_crossline(ref_ds,test_ds,cycle_cnt_l,cycle_len_l,cycle_dupc_l):
    drop_idx_li = []
    ref_t_li = []
    for cnt_i in iter(cycle_cnt_l):
        ref_t_li.append(tuple(accumulate(ref_ds[-cnt_i:].values))[cnt_i-1])
    len_ref_t_li = len(ref_t_li)

    for idx,val in test_ds.items():
        for index in range(len_ref_t_li):
            temp_t=val+ref_t_li[index]
            if len(set(temp_t)) not in cycle_len_l[index]:
                drop_idx_li.append(idx)
                break
            else:
                if cycle_dupc_l and cycle_dupc_l[index]:
                    dup_class_li = list(Counter(temp_t).values())
                    dup_class_li.sort(reverse=True)
                    dup_class_li = [x for x in dup_class_li if x >1]
                    dup_class_str = ''.join(map(str,dup_class_li))
                    dup_class_str = dup_class_str if dup_class_str else '0'
                    if dup_class_str not in cycle_dupc_l[index]:
                        drop_idx_li.append(idx)
                        break
    return drop_idx_li

def parse_precap_li(array_data,name):
    multi_res = ()
    col_li = []
    if array_data:
        if name == 'cycle_len_dupc_df':
            col_li = ['cnt','seqlen','dupc']
        if name == 'len_foc_inline_df':
            col_li = ['a_g_len','foc']
        
        df = pd.DataFrame(array_data)
        for col in col_li:
            if col in df:
                m_res_temp = (df[col].tolist(),)
            else:
                m_res_temp = ([],)
            multi_res += m_res_temp
    return multi_res

def dvalue_inline_need_drop(testli,ref_dvall):
    need_drop = 1
    test_diff_str = ''.join(map(lambda x:str(x),np.diff(np.array(testli))))
    if ref_dvall.count(test_diff_str) > 0:
        need_drop = 0
    return need_drop

def bit_inline_need_drop(testli,ref_bitl):
    need_drop = 1
    bit_test_str = ''.join(map(lambda x:str(x//10%10),testli))
    if ref_bitl.count(bit_test_str) > 0:
        need_drop = 0
    return need_drop

def fo_inline_need_drop(testli,ref_foli,a_g_len_li,foc_li):
    need_drop = 1
    factor_li = []
    offset_li = []
    len_li = len(testli)
    for idx in range(0,len_li-2):
        #判断整除
        factor = 0
        r = (testli[idx+2] - testli[idx+1]) // (testli[idx+1] - testli[idx])
        offset = testli[idx+1] - testli[idx] * factor
        factor_li.append(factor)
        offset_li.append(offset)
    fac_off_li = list(zip(factor_li,offset_li))

    zero_idx = np.where(np.array(factor_li) == 0)[0]
    if len(zero_idx) == len(fac_off_li):
        len_arith_geo = 0
    else:
        len_arith_geo = 2 + max([len(list(v) for k,v in iters.groupby(fac_off_li))])
    fac_off_li = [n for i, n in enumerate(fac_off_li) if i not in zero_idx]
    fac_off_li = list(map(lambda x:','.join(map(str,x)),dict.fromkeys(fac_off_li)))
    fo_c = len(fac_off_li)
    
    for idx in range(len(a_g_len_li)):
        #只需满足一个
        if len_arith_geo in a_g_len_li[idx] and fo_c in foc_li[idx]:
            if fo_c == 0 or len(set(fac_off_li)) & set(ref_foli[idx]):
                need_drop = 0
                break
    return need_drop

def del_diff_ds_cond(test_ds,ref_ds,true_seq,dropflag_dict,t_precap):
    if 'diff_ds_cond' in t_precap['skip_delcond']:
        logger.info('skip del_diff_ds_cond')
        return test_ds
    #取差集
    diff_ds = test_ds[~test_ds.isin(ref_ds)]
    test_predictdata_dropflag('diff_ds_cond',diff_ds,true_seq,dropflag_dict)
    return diff_ds

def del_cycle_dup_cond(test_ds,ref_ds,true_seq,dropflag_dict,t_precap):
    if 'cycle_dup_cond' in t_precap['skip_delcond']:
        logger.info('skip del_cycle_dup_cond')
        return None
    if args.debug != 0:
        time_start = time.time()
    drop_idx_li = []
    cycle_cnt_l,cycle_len_l,cycle_dupc_l = parse_precap_li(t_precap['cycle_len_dupc_df'],'cycle_len_dupc_df')
    drop_idx_li = drop_dopc_cycle_crossline(ref_ds,test_ds,cycle_cnt_l,cycle_len_l,cycle_dupc_l)
    test_ds.drop(drop_idx_li,inplace=True)
    test_predictdata_dropflag('cycle_len_dupc_df',test_ds,true_seq,dropflag_dict)

    if args.debug != 0:
        time_end = time.time()
        logger.info("loop del_cycle_dup_cond take time:{}".format(time_end - time_start))

def del_fo_cond(test_ds,debugpath,col_s,true_seq,dropflag_dict,t_precap):
    if 'fo_cond' in t_precap['skip_delcond']:
        logger.info('skip del_fo_cond')
        return None
    if args.debug != 0:
        time_start = time.time()
    
    fo_debug_df = pd.read_csv(debugpath + 'fo_debug_ds_' + str(col_s) +'.csv')
    a_g_len_li,foc_li = parse_precap_li(t_precap['len_foc_inline_df'],'len_foc_inline_df')
    drop_idx_li = drop_fo_inline(test_ds,fo_debug_df,a_g_len_li,foc_li)
    test_ds.drop(drop_idx_li,inplace=True)
    test_predictdata_dropflag('fo_cond',test_ds,true_seq,dropflag_dict)

    if args.debug != 0:
        time_end = time.time()
        logger.info("loop del_fo_cond take time:{}".format(time_end - time_start))
    
def del_bit_cond(test_ds,debugpath,col_s,true_seq,dropflag_dict,t_precap):
    if 'bit_cond' in t_precap['skip_delcond']:
        logger.info('skip del_bit_cond')
        return None
    if args.debug != 0:
        time_start = time.time()
    
    drop_idx_li = []
    rate_sum = 0.0
    sel_idx = 0
    bit_debug_df = pd.read_csv(debugpath + 'bit_debug_ds_' + str(col_s) + '.csv',converters = {'bit':str})
    for index in bit_debug_df.index:
        rate_sum += bit_debug_df.loc[index,'proportion']
        if rate_sum >=0.9:
            sel_idx = index
            break
    ref_bitli = bit_debug_df.loc[:sel_idx,'bit'].tolist()

    for idx,val in test_ds.items():
        if bit_inline_need_drop(val,ref_bitli) == 1:
            drop_idx_li.append(idx)

    test_ds.drop(drop_idx_li,inplace=True)
    test_predictdata_dropflag('bit_cond',test_ds,true_seq,dropflag_dict)

    if args.debug != 0:
        time_end = time.time()
        logger.info("loop del_bit_cond take time:{}".format(time_end - time_start))    

def del_dupfz_cond(test_ds,df,true_seq,dropflag_dict):
    if 'dupfz_cond' in t_precap['skip_delcond']:
        logger.info('skip del_dupfz_cond')
        return None    
    drop_idx_li = drop_dupfrontzone_inline(df,test_ds)
    test_ds.drop(drop_idx_li,inplace=True)
    test_predictdata_dropflag('dupfz_cond',test_ds,true_seq,dropflag_dict)

def del_dvalue_cond(test_ds,true_seq,dropflag_dict,t_precap):
    if 'dvalue_cond' in t_precap['skip_delcond']:
        logger.info('skip del_dvalue_cond')
        return None    
    drop_idx_li = []
    for idx,val in test_ds.items():
        if dvalue_inline_need_drop(val,t_precap['dvalue']) == 1:
            drop_idx_li.append(idx)
    test_ds.drop(drop_idx_li,inplace=True)
    test_predictdata_dropflag('dvalue_cond',test_ds,true_seq,dropflag_dict)

def gen_kl8_red_data(df,col_s,col_e,debugpath,t_setting,t_precap):
    pick_num = t_setting['subpicknum']
    range_df = pd.read_csv(debugpath + 'numrange_debug_ds.csv')
    for col_s_i in range(col_s,col_e,pick_num):
        col_e_i = col_s_i + pick_num
        c_index = range_df.loc[range_df['col_s'] == col_s_i].index.tolist()[0]
        c_range = range_df.loc[c_index,'min':'max']
        #全组合
        total_ds = pd.Series(iters.combinations(range(c_range['min'],c_range['max']+1),pick_num))
        #已开出
        ref_ds = df.iloc[:len(df)-args.dleft,col_s_i:col_e_i].apply(lambda x:tuple(x),axis=1)

        gen_normal_data(df,total_ds,ref_ds,col_s_i,col_e_i,debugpath,t_precap)
    '''
    1,2,3,4     5,6,7,8     9,10,11,12
    loop 5
        1.min~max from csv
        2.del 相邻行
        3.del cond cross line dup
        4.del cond arith_geo from csv
        5.del cond bit from csv
        6. choose rand one?
    '''

def gen_normal_data(df,total_ds,ref_ds,col_s,col_e,debugpath,t_precap):
    dropflag_dict = {}
    true_seq = df.iloc[[len(df)-args.dleft],col_s:col_e].squeeze()
    
    #del 不满足condition1
    diff_ds = del_diff_ds_cond(total_ds,ref_ds,true_seq,dropflag_dict,t_precap)

    #del 不满足condition2
    del_cycle_dup_cond(diff_ds,ref_ds,true_seq,dropflag_dict,t_precap)

    #del 不满足condition3
    del_fo_cond(diff_ds,debugpath,col_s,true_seq,dropflag_dict,t_precap)

    #del 不满足condition4
    del_bit_cond(diff_ds,debugpath,col_s,true_seq,dropflag_dict,t_precap)

    #del 不满足condition5
    del_dupfz_cond(diff_ds,df,true_seq,dropflag_dict)

    #del 不满足condition6
    del_dvalue_cond(diff_ds,true_seq,dropflag_dict,t_precap)

    if args.debug != 0:
        test_save_predictdata(diff_ds,true_seq,dropflag_dict,col_s,debugpath,t_precap)

if __name__ == '__main__':
    if not args.name:
        raise Exception("lottery name is null")
    elif not args.ball_color:
        raise Exception("ball_color is null")
    else:
        df = pd.read_csv(test_config[args.name]['datapath'])
        t_setting = test_config[args.name]['settings'][args.ball_color]
        t_precap = test_config[args.name]['precap'][args.ball_color]
        range_max = t_setting['cnt'] + 1
        pick_num = t_setting['picknum']
        col_s = t_setting['start_col_idx']
        col_e = t_setting['end_col_idx'] + 1
        debugpath = t_setting['debug_data_path']
        if not os.path.exists(debugpath):
            os.makedirs(debugpath)
        
        total_ds = None
        ref_ds = None
        if pick_num > 10 and args.name == 'kl8':
            logger.info("total_ds is too large")
            gen_kl8_red_data(df,col_s,col_e,debugpath,t_setting,t_precap)
        else:
            #全组合
            total_ds = pd.Series(iters.combinations(range(1,range_max),pick_num))
            #已开出
            ref_ds = df.iloc[:len(df)-args.dleft,col_s:col_e].apply(lambda x: tuple(x),axis=1)
            gen_normal_data(df,total_ds,ref_ds,col_s,col_e,debugpath,t_precap)