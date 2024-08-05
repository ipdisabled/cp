'''
delft hitindex total rate
index 0         0
index x         y     x/y 
py.exe .\autotest.py  --name dlt --ball_color red --debug 2
'''
import os
import re
import argparse
import subprocess
import pandas as pd
from loguru import logger
from config import *

parse = argparse.ArgumentParser()
parse.add_argument('--name',default='dlt',type=str,help='choose lottery name')
parse.add_argument('--ball_color',default='red',type=str,help='choose ball color')
parse.add_argument('--cycle_row',default=2,type=int,help='cylce window size')
parse.add_argument('--debug',default=2,type=int,help='debug data')
args = parse.parse_args()

if __name__ == '__main__':
    hit_data_li = []
    for index in range(1,101):
        logger.info("testloop {}",index)
        os.system("python parse_data.py --name "+args.name+" --ball_color "+args.ball_color
            +" --cycle_row "+str(args.cycle_row) +" --debug "+str(args.debug)+" --dleft "+str(index))
        result = subprocess.run("python gen_data.py --name "+args.name+" --ball_color "+args.ball_color
            +" --debug "+str(args.debug)+" --dleft "+str(index),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,text=True)
        if result.returncode == 0:
            missed_col_s_li = re.findall('(?<=you are missed!!! col_s:).*?(?= drop in)',result.stdout)
            if missed_col_s_li:
                for ele in missed_col_s_li:
                    hit_data_li.append([index,ele,0,1,0])

            col_s_li = re.findall('(?<=predict_ds.csv col_s:).*?(?= totalcnt:)',result.stdout)
            totalcnt_li = re.findall('(?<=totalcnt:).*?(?= index:)',result.stdout)
            hitindex_li = re.findall('(?<=index:).*?(?= value:)',result.stdout)
            if col_s_li and hitindex_li and totalcnt_li:
                len_col_s = len(col_s_li)
                for idx in range(0,len_col_s):
                    hit = int(hitindex_li[idx])
                    total = int(totalcnt_li[idx])
                    hit_data_li.append([index,col_s_li[idx],hit,total,round(hit/total,9)])
        else:
           logger.info("subprocess run error")

    hit_data_li = [i for n,i in enumerate(hit_data_li) if i not in hit_data_li[:n]]
    df =  pd.DataFrame(hit_data_li,columns=['dleft','cols','hitindex','totalcnt','rate'])
    df.to_csv(test_config[args.name]['settings'][args.ball_color]['debug_data_path'] + 'test_predict_ds.csv',index=0) 