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

def plot_map_comparsion(pp, pend=10, value='mech', prnt=False):
    """
    Вывод карт для сравнения

    :param pp: - название узла, к примеру 'Pipe-5'
    :param pend:  - давление?
    :param value: - параметры для вывода карт: 'mech' - перемещение, 'slug_delta_holdup',
                                               'slug_holdup', 'film_holdup', 'slug_velocity'
    :return:
    """

    if value == 'mech':
        vmax = 10
    else:
        vmax = 1

    a = tsim.get_matr_ql_qg(pipe=pp, p_end=pend, val=value)   # расчет карты по данным моделирования
    b = tpl.get_matr_ql_qg(pipe=pp, p_end=pend, val=value)    # расчет карты по данным моделирования
    c = b / a

    # Нахождение максимального элемента на карте плоского трубопровода
    a_max = a.max().max()             # максимальное значение
    a_i = a.max(axis=1).idxmax()      # дебит по жидкости
    a_j = a.max(axis=0).idxmax()      # дебит по газу
    a_i_idx = a.index.get_loc(a_i)    # индекс дебита по жидкости
    a_j_idx = a.columns.get_loc(a_j)  # индекс дебита по газу

    # Нахождение максимального элемента на карте трубопровода с перепадами высот
    b_max = b.max().max()             # максимальное значение
    b_i = b.max(axis=1).idxmax()      # дебит по жидкости
    b_j = b.max(axis=0).idxmax()      # дебит по газу
    b_i_idx = b.index.get_loc(b_i)    # индекс дебита по жидкости
    b_j_idx = b.columns.get_loc(b_j)  # индекс дебита по газу

    if prnt == True:                  # вывод пояснительной записки и графиков
        print('\n\n//--------------------------------------------------------',
              '\nРасчет карт для: '
              '\n - контрольной точки - \"', pp,
              '\"\n - даление - \"', pend,
              '\"\n - карта - \"', value)
        print("Максимальное значение на карте \"плоского трубопровода\" = ", a_max, ", при:"
                                                                                    "\n - скорости жидкости = ", a_i,
              "\n - скорости газа = ", a_j)
        print("\nМаксимальное значение на карте трубопровода с \"горбом\" = ", b_max, ", при:"
                                                                                      "\n - скорости жидкости = ", b_i,
              "\n - скорости газа = ", b_j)

        AA = plt.figure(figsize=(16, 6), dpi=100)
        ax1 = AA.add_subplot(121)
        plt.title('Карта для для плоского трубопровода в точке - ' + str(pp))
        sns.heatmap(a, cmap="Reds", annot=True, vmin=0, vmax=vmax, linewidths=.5)
        plt.yticks(rotation="horizontal")

        ax2 = AA.add_subplot(122)
        plt.title('Карта трубопровода с "горбом" в точке - ' + str(pp))
        sns.heatmap(b, cmap="Reds", annot=True, vmin=0, vmax=vmax, linewidths=.5)    # построение визуального представления карты
        plt.yticks(rotation="horizontal")
        AA.show()

        if value == 'mech':           # построение карты отношений
            BB = plt.figure(figsize=(8, 6), dpi=70)
            ax3 = BB.add_subplot(111)
            plt.title('Карта отношений степени воздействияв точке - ' + str(pp))
            sns.heatmap(c, annot=True, cmap='Blues', vmin=0, vmax=3, linewidths=.5)
            plt.yticks(rotation="horizontal")
            BB.show()
    else:
        tp.plot_trend(tsim, klist='HOLEXP', qg_num=a_j_idx, ql_num=a_i_idx, pipe_num=pp, p_num=pend)[1300:].plot()
        plt.show()


#for pp in tsim.pipe_list:             # построение карт потока для всех узлов
#    plot_map_comparsion(pp, prnt=True)
plot_map_comparsion('Pipe-5', prnt=True)


def ipipe(tpl,pp):
    pipe_ind = tsim.pipe_list.tolist().index(pp)
    return pipe_ind


cc = tp.plot_trend      (tsim, klist=['HOLEXP'], qg_num=2, ql_num=0, pipe_num=ipipe(tsim,'Pipe-5'), p_num=0)[1300:]
dd = tp.plot_trend_super(tsim,                   qg_num=2, ql_num=0, pipe_num=ipipe(tsim,'Pipe-5'), p_num=0)[1300:]


CC = plt.figure(figsize=(16, 6), dpi=80)
ax1 = CC.add_subplot(121)
plt.title('Название - ' + str(pp))
sns.set(style="darkgrid")
gammas = sns.load_dataset("gammas")
plt.plot(cc)

ax2 = CC.add_subplot(122)
plt.plot(dd)
#CC.legend(bbox_to_anchor=(1.01, 0.5, 0.4, .1), loc='center', ncol=1, mode="expand", borderaxespad=0.,fancybox=True, shadow=False,frameon = False)
CC.show()


def plot_trend_comparsion(tp_1, tp_2, klist=['HOLEXP'], qg_num=0, ql_num=0, pipe_num=ipipe(tsim, 'Pipe-5'), p_num=0):
    """

    :param tp_1: - Трубопровод без перепадов высот
    :param tp_2: - Трубопровод с перепадами высот
    :param klist: - Ключевые слова
    :param qg_num: - НОМЕР значения дебита газа
    :param ql_num: - НОМЕР значения дебита жидкости
    :param pipe_num: - название узла
    :param p_num: - Номер значения давления
    :return:
    """

    if klist == ['MECH']:
        cc = tp.plot_trend_super(tp_1, qg_num=qg_num, ql_num=ql_num, pipe_num=pipe_num, p_num=p_num)[1300:]
        dd = tp.plot_trend_super(tp_2, qg_num=qg_num, ql_num=ql_num, pipe_num=pipe_num, p_num=p_num)[1300:]
    else:
        cc = tp.plot_trend(tp_1, klist=klist, qg_num=qg_num, ql_num=ql_num, pipe_num=pipe_num, p_num=p_num)[1300:]
        dd = tp.plot_trend(tp_2, klist=klist, qg_num=qg_num, ql_num=ql_num, pipe_num=pipe_num, p_num=p_num)[1300:]

    CC = plt.figure(figsize=(16, 6), dpi=80)
    CC.suptitle(cc._info_axis._data.real[0])
    sns.set(style="darkgrid")
    ax1 = CC.add_subplot(121)
    plt.title('Для плоского трубопровода')
    plt.plot(cc, label="test1")

    ax2 = CC.add_subplot(122)
    plt.title('Для трубопровода с перепадом высот')
    plt.plot(dd, label="test2")
    # CC.legend()
    CC.show()

plot_trend_comparsion(tsim, tpl, klist=['HOL'], qg_num=0, ql_num=0, pipe_num=ipipe(tsim, 'Pipe-5'), p_num=0)


