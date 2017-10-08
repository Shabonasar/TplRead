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
tsim = tp.TplParams('NP2_pl/')
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


def plot_map(pp, pend=10):
    print('Контрольная точка ',pp, '--------------------------------------------------------')
    
    AA = plt.figure(figsize=(17, 6), dpi=70)
    ax1 = AA.add_subplot(121)
    
    #print('Карта для для плоского трубопровода')
    plt.title('Карта для для плоского трубопровода')
    a=tsim.get_matr_ql_qg(pipe=pp,p_end=pend,val='slug_velocity_front')     # расчет карты по данным моделирования
    sns.heatmap(a, annot=True,cmap="Reds", vmin=0, vmax=50)             # построение визуального представления карты
    plt.yticks(rotation="horizontal")
    
    ax2 = AA.add_subplot(122)
    #print('Карта трубопровода с "горбом"') 
    plt.title('Карта трубопровода с "горбом"')
    b=tpl.get_matr_ql_qg(pipe=pp,p_end=pend,val='slug_velocity_front')      # расчет карты по данным моделирования
    sns.heatmap(b, annot=True,cmap="Greens", vmin=0, vmax=50)           # построение визуального представления карты
    plt.yticks(rotation="horizontal")
    AA.show()
    show()
    
    print('Карта отношений степени воздействия')
    plt.title('Карта отношений степени воздействия')
    c = b / a 
    sns.heatmap(c, annot=True,  cmap='Blues', vmin=0, vmax=3)
    show()
    
for pp in tsim.pipe_list:           # строим для всех контрольных точек
    plot_map(pp)
    
    
def ipipe(tpl,pp):
    pipe_ind = tsim.pipe_list.tolist().index(pp)
    return pipe_ind

def plot_trend_example():
    tp.plot_trend(tsim,klist=k,qg_num=qg, ql_num=ql,pipe_num=pp,p_num=p)[400:].plot()
    show()
    tp.plot_trend(tpl,klist=k,qg_num=qg, ql_num=ql,pipe_num=pp,p_num=p)[400:].plot()
    show()
    tp.plot_trend_super(tsim,qg_num=qg, ql_num=ql,pipe_num=pp,p_num=p).plot()
    show()
    
qg=2           # номер значения дебита газа
ql=8           # номер значения дебита жидкости
pp=0           # номер значения дебита участка трубы
p=1            # номер значения дебита давления
k=['USFEXP','USTEXP']   # номер значения ключевого слова

plot_trend_example()