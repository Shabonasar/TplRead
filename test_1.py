# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import tplread as tp
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pylab


pylab.interactive(False)
#%pylab inline
# читаем сначала данные плоской модели
tsim = tp.TplParams('NPsimple/')
tsim.read_data()
tsim.calc_data()
# затем читаем данные модели с одним горбом модели
tpl = tp.TplParams('NPtest1/')
tpl.read_data()
tpl.calc_data()

# print(t.df_super)
def check(t):

    # процедура вывод результатов проверки загрузки данных
    print(" файлов прочитано", len(t.files))
    print('ключевые слова для расчетов      ', t.key_list)
    print('полный список ключевых слов      ', t.key_list_all)
    print('список точек контроля параметров ', t.pipe_list)
    print('список значений давления в трубе ', t.p_end_list)
    print('список значений дебитов жидкости ', t.qliq_list)
    print('список значений дебитов газа     ', t.qgas_list)
    
check(tsim)     # проверим первый блок файлов и выведем результаты
check(tpl) # проверим второй блок файлов и выведем результаты
print()

pp = 'Pipe-5'
pend = 10
print('Контрольная точка ', pp, '--------------------------------------------------------')
    
print('Карта для для плоского трубопровода')   
a = tsim.get_matr_ql_qg(pipe = pp, p_end = pend, val = 'mech')     # расчет карты по данным моделирования
sns.heatmap(a,cmap="Reds", vmin=0, vmax=1)      

    
print('Карта трубопровода с "горбом"')   
b=tpl.get_matr_ql_qg(pipe=pp,p_end=pend,val='mech')      # расчет карты по данным моделирования
sns.heatmap(b,cmap="Greens", vmin=0, vmax=10)           # построение визуального представления карты

    
print('Карта отношений степени воздействия')   
c = b / a 
sns.heatmap(c, annot=True,  cmap='Blues', vmin=0, vmax=3)

def comparison_pipe():
    print('\nСравнение')

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'


def plot_map(pp, pend=10):
    print('Контрольная точка ', pp, '--------------------------------------------------------')

    AA = plt.figure(figsize=(16, 6), dpi=100)
    ax1 = AA.add_subplot(121)
    #plt.axes()
    plt.title('Карта для для плоского трубопровода')
    a = tsim.get_matr_ql_qg(pipe=pp, p_end=pend, val='slug_delta_holdup')  # расчет карты по данным моделирования
    sns.heatmap(a, cmap="Reds", annot=True, vmin=0, vmax=1, linewidths=.5)
    plt.yticks(rotation="horizontal")

    ax2 = AA.add_subplot(122)
    plt.title('Карта трубопровода с "горбом"')
    b = tpl.get_matr_ql_qg(pipe=pp, p_end=pend, val='slug_delta_holdup')  # расчет карты по данным моделирования
    sns.heatmap(b, cmap="Reds", annot=True, vmin=0, vmax=1, linewidths=.5)  # построение визуального представления карты
    plt.yticks(rotation="horizontal")
    AA.show()

    BB = plt.figure(figsize=(8, 6), dpi=100)
    ax3 = BB.add_subplot(111)
    plt.title('Карта отношений степени воздействия')
    c = b / a
    sns.heatmap(c, annot=True, cmap='Blues', vmin=0, vmax=3, linewidths=.5)
    plt.yticks(rotation="horizontal")
    BB.show()

plot_map('Pipe-5', pend=10)
    
