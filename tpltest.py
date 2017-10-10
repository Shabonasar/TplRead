# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:53:39 2017

@author: rnt
"""
import tplread as tp
import pandas as pd

filename1 = '3.tpl'
#t=tp.TplFile(filename1)
# читаем сначала данные плоской модели
tsim = tp.TplParams('NPtest1/')
tsim.read_data()
tsim.calc_data()
# затем читаем данные модели с одним горбом модели

#print(t.df_super)
#print(t.get_matrix(key='USG', pipe='Pipe-3', q_g='5.0', q_l='5000.0', p_end='10.0'))


"""  

tpl_path = '1/'

tpl6 = tp.TplParams(tpl_path)
tpl6.readD10ata()
tp.calcData(tpl6)
#print(tpl6.dfsuper)
"""
print('calc done')

#%%



    
qg=2           # номер значения дебита газа
ql=8           # номер значения дебита жидкости
pp=0           # номер значения дебита участка трубы
p=1            # номер значения дебита давления
k=['USFEXP','USTEXP']   # номер значения ключевого слова
