# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:53:39 2017

@author: rnt
"""
import tplread as tp
import pandas as pd

filename1 = '3.tpl'
#t=tp.TplFile(filename1)
t = tp.TplParams('tpl/')
t.read_data()
t.calc_data()
#print(t.df_super)
#print(t.get_matrix(key='USG', pipe='Pipe-3', q_g='5.0', q_l='5000.0', p_end='10.0'))


"""  

tpl_path = '1/'

tpl6 = tp.TplParams(tpl_path)
tpl6.readD10ata()
tp.calcData(tpl6)
#print(tpl6.dfsuper)

print('calc done')

#%%
print('plottindg start') 

i=0
for ppnm in tpl6.f[0].pipe_list :
    print(ppnm)
    plt.figure(i)
    d=tpl6.dfsuper
    dd=d.loc[d.ppnm==ppnm].loc[d.P_atm==8]
    ddd=dd.pivot_table(values='forceFrac',index='qgas_Mm3day', columns='qliq_m3day')
    ax = sns.heatmap(ddd, cmap="YlGnBu")
    i=i+1
"""    