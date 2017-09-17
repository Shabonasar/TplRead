# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:54:56 2017

@author: Khabibullin Rinat

анализ данных сгенерированных в OLGA 

"""

import math
import pyfas as fa
import pandas as pd
from glob import glob
import numpy as np
import re


class TplFile(fa.Tpl):
    """
    read and parse one tpl file
    """

    def __init__(self, filename):
        super().__init__(filename)
        self.data_all = 0
        self.data_trends = 0
        self.data_trends_summary = 0

        self.q_liq_m3day = 0
        self.q_gas_Mm3day = 0
        self.p_atm = 10
        self.var_num = 0
        self.file_name = filename
        self.tpl_split = re.compile(r'\'*\s*\'', re.IGNORECASE)
 #       self.dict = {}
        self.df = pd.DataFrame()

        params = self.filter_data('PIPE:')  # get all trends with PIPE:
        for i in params:
            params.update({i: self.tpl_split.split(params[i])})
        self.fd = pd.DataFrame(params, index=['key', 'sect', 'branch', 'model', 'pipe', 'pipe_num', 'nr', '1',
                                              'dim', 'msg', '2']).T
        self.pipe_list = self.fd['pipe_num'].unique()  # список труб
        self.key_list = self.fd['key'].unique()  # список ключевых слов
        self.extract_all()

    def extract_all(self):
        """
        extract all columns to DataFrame
        :return: df attribute filled with file data
        """
        """read all data from file"""
        self.data_all = np.loadtxt(self.abspath,
                                   skiprows=self._attributes['data_idx'] + 1,
                                   unpack=True)
        """put in in DataFrame for further manipulations"""
        self.df = pd.DataFrame(self.data_all).T
        dic = self.trends
        for v in dic:
            dic.update({v: dic[v].replace("'", " ")})
        dic.update({0: 'time'})
        self.df.columns = dic.values()

    def get_trend(self, key_list, pipe_list):
        """
        method to extract trends for some pipe and set of keys
        :param key_list:
        :param pipe_list:
        :return:
        """

        self.data_trends = pd.DataFrame(index=self.df.index)
        key1 = 'start'
        for pipe in pipe_list:
            for key in key_list:
                try:
                    key1 = key + ":" + pipe
                    self.data_trends[key1] = self.df.filter(like=key+" ").filter(like=pipe).values
                except Exception:
                    print('Error ' + key1)
        """calculate summary data on trends extracted"""
        self.data_trends_summary = pd.DataFrame()
        self.data_trends_summary['mean'] = self.data_trends.mean()
        self.data_trends_summary['min'] = self.data_trends[20:].min()
        self.data_trends_summary['max'] = self.data_trends.max()
        # spl = re.compile(r':', re.IGNORECASE)
        self.data_trends_summary['key'] = self.data_trends.columns.str.split(pat=':').str[0]
        self.data_trends_summary['point'] = self.data_trends.columns.str.split(":").str[1]
        self.data_trends_summary['q_liq'] = self.q_liq_m3day
        self.data_trends_summary['q_gas'] = self.q_gas_Mm3day
        self.data_trends_summary['p_end'] = self.p_atm
        return self.data_trends


"""
    def trend_min(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].min()

    def trend_max(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].max()

    def trend_mean(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].mean()


    def read_points(self):
        '''
        reads all the data for all points  with ppnm (pipename)
        obsolete
        '''
        i = 0
        for ppnm in self.pipe_list:
            self.indlist = list(self.fd[self.fd['pipenum'] == ppnm].T)  # get index list with needed point
            self.tpl = fa.Tpl(self.fName)  # reread file and clear all data
            for ind in self.indlist:
                self.tpl.extract(ind)
            self.dfm = pd.DataFrame(self.tpl.data, index=self.tpl.time)
            self.dfm.columns = list(self.key_list)
            self.dict.update({ppnm: self.dfm})
            self.df.set_value(i, 'pipename', ppnm)
            #     self.df.set_value(i,'key_list',[self.key_list])
            i = i + 1
            #     print(ppnm + '  done')
        #        self.dfm['pipeName'] = ppnm   # add column with pipe name

    def getTrend(self, ppnm, keyname):
        '''
        return trend by point name and keyname 
        '''
        return self.dict[ppnm][keyname]

    def getMin(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].min()

    def getMax(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].max()

    def getMean(self, ppnm, keyname):
        return self.getTrend(ppnm, keyname)[20:].mean()

"""


class TplParams:
    """
    class read all data from parametric study
    and allows to manipulate it 
    """

    def __init__(self, pathname):
        """
        set path
        read files
        estimate parameters variation 
        """
        self.tpl_path = pathname
        self.file_name_mask = '*.tpl'
        self.files = glob(self.tpl_path + self.file_name_mask)
        print(self.files)
        self.count_files = 0  # number of cases loaded
        self.file_num = 0
        self.file_list = {}
        self.qliqlist = set()
        self.qgaslist = set()
        self.plist = set()
        self.df = pd.DataFrame()
        self.dfsuper = pd.DataFrame()
        self.key_list = ['HOLEXP', 'USFEXP', 'USL', 'USG']
        self.pipe_list = ['Pipe-3']
        self.flSplit = re.compile(r'[-,\s,\.]',re.IGNORECASE)
        """
        параметры трубы
        """
        self.ID_mm = 800  # внутренний диамет трубы
        self.densliq_kgm3 = 800  # плотность жидкости
        self.weight_kgm = 200  # удельный вес трубы

    def read_data(self):
        for file in self.files:  # iterating through all files
            par = self.flSplit.split(file)
            #print(par)
            fl_read = TplFile(file)  # read file
            fl_read.q_liq_m3day = float(par[2])
            fl_read.q_gas_Mm3day = float(par[4])
            fl_read.p_atm = float(par[6])
            fl_read.get_trend(self.key_list, self.pipe_list)
            self.file_list.update({self.file_num: fl_read})  # put reader object to dictionary
            print(file + ' read done')
            self.file_num = self.file_num + 1
        df_list = []
        for df in self.file_list.values():
            df_list.append(df.data_trends_summary)
        self.df = pd.concat(df_list)
        print('read done')

    def collect_data(self):
        return 1

    def calcData(self):
        i = 0
        for index, row in self.df.iterrows():
            f = self.f[row['fnum']]
            for ppnm in f.pipelist:
                self.dfsuper.set_value(i, 'index', index)
                self.dfsuper.set_value(i, 'ppnm', ppnm)
                self.dfsuper.set_value(i, 'qliq_m3day', f.qliq_m3day)
                self.dfsuper.set_value(i, 'qgas_Mm3day', f.qgas_Mm3day)
                self.dfsuper.set_value(i, 'P_atm', f.P_atm)
                """
                читаем и преобразуем данные по содержанию жидкости в потоке
                """
                usf_min = f.getMin(ppnm, 'USFEXP')
                usf_max = f.getMax(ppnm, 'USFEXP')
                usf_mean = f.getMean(ppnm, 'USFEXP')

                ust_min = f.getMin(ppnm, 'USFEXP')
                ust_max = f.getMax(ppnm, 'USFEXP')
                ust_mean = f.getMean(ppnm, 'USFEXP')

                uslug_min = (usf_min + ust_min) / 2
                uslug_max = (usf_max + ust_max) / 2
                uslug_mean = (usf_mean + ust_mean) / 2

                self.dfsuper.set_value(i, 'HOLEXP min', f.getMin(ppnm, 'HOLEXP'))
                self.dfsuper.set_value(i, 'HOLEXP max', f.getMax(ppnm, 'HOLEXP'))
                self.dfsuper.set_value(i, 'HOLEXP mean', f.getMean(ppnm, 'HOLEXP'))

                hslug = f.getMax(ppnm, 'HOLEXP')
                hfilm = f.getMin(ppnm, 'HOLEXP')
                """
                читаем и преобразуем данные по скорости движения фронта пробки
                """
                self.dfsuper.set_value(i, 'USlug min', uslug_min)
                self.dfsuper.set_value(i, 'USlug max', uslug_max)
                self.dfsuper.set_value(i, 'USlug mean', uslug_mean)

                """
                читаем и преобразуем данные по скорости движения жидкости
                """

                usl_min = f.getMin(ppnm, 'USL')
                usl_max = f.getMax(ppnm, 'USL')
                usl_mean = f.getMean(ppnm, 'USL')

                usg_min = f.getMin(ppnm, 'USG')
                usg_max = f.getMax(ppnm, 'USG')
                usg_mean = f.getMean(ppnm, 'USG')

                usm_min = usl_min + usg_min
                usm_max = usl_max + usg_max
                usm_mean = usl_mean + usg_mean

                self.dfsuper.set_value(i, 'USL min', usl_min)
                self.dfsuper.set_value(i, 'USL max', usl_max)
                self.dfsuper.set_value(i, 'USL mean', usl_mean)
                """
                читаем и преобразуем данные по скорости движения газа
                """
                self.dfsuper.set_value(i, 'USG min', usg_min)
                self.dfsuper.set_value(i, 'USG max', usg_max)
                self.dfsuper.set_value(i, 'USG mean', usg_mean)

                self.dfsuper.set_value(i, 'USM min', usm_min)
                self.dfsuper.set_value(i, 'USM max', usm_max)
                self.dfsuper.set_value(i, 'USM mean', usm_mean)

                forceFrac = forceFraction(uslug_max, self.densliq_kgm3, self.ID_mm, self.weight_kgm, Hlslug=hslug,
                                          Hlfilm=hfilm)
                self.dfsuper.set_value(i, 'forceFrac', forceFrac)

                i = i + 1


def calcData(tpl):
    """
    процедура для построения карт
    """
    i = 0
    for index, row in tpl.df.iterrows():
        f = tpl.f[row['fnum']]
        for ppnm in f.pipelist:
            tpl.dfsuper.set_value(i, 'index', index)
            tpl.dfsuper.set_value(i, 'ppnm', ppnm)
            tpl.dfsuper.set_value(i, 'qliq_m3day', f.qliq_m3day)
            tpl.dfsuper.set_value(i, 'qgas_Mm3day', f.qgas_Mm3day)
            tpl.dfsuper.set_value(i, 'P_atm', f.P_atm)
            """
            читаем и преобразуем данные по содержанию жидкости в потоке
            """
            usf_min = f.getMin(ppnm, 'USFEXP')
            usf_max = f.getMax(ppnm, 'USFEXP')
            usf_mean = f.getMean(ppnm, 'USFEXP')

            ust_min = f.getMin(ppnm, 'USFEXP')
            ust_max = f.getMax(ppnm, 'USFEXP')
            ust_mean = f.getMean(ppnm, 'USFEXP')

            uslug_min = (usf_min + ust_min) / 2
            uslug_max = (usf_max + ust_max) / 2
            uslug_mean = (usf_mean + ust_mean) / 2

            tpl.dfsuper.set_value(i, 'HOLEXP min', f.getMin(ppnm, 'HOLEXP'))
            tpl.dfsuper.set_value(i, 'HOLEXP max', f.getMax(ppnm, 'HOLEXP'))
            tpl.dfsuper.set_value(i, 'HOLEXP mean', f.getMean(ppnm, 'HOLEXP'))

            hslug = f.getMax(ppnm, 'HOLEXP')
            hfilm = f.getMin(ppnm, 'HOLEXP')
            """
            читаем и преобразуем данные по скорости движения фронта пробки
            """
            tpl.dfsuper.set_value(i, 'USlug min', uslug_min)
            tpl.dfsuper.set_value(i, 'USlug max', uslug_max)
            tpl.dfsuper.set_value(i, 'USlug mean', uslug_mean)

            """
            читаем и преобразуем данные по скорости движения жидкости
            """

            usl_min = f.getMin(ppnm, 'USL')
            usl_max = f.getMax(ppnm, 'USL')
            usl_mean = f.getMean(ppnm, 'USL')

            usg_min = f.getMin(ppnm, 'USG')
            usg_max = f.getMax(ppnm, 'USG')
            usg_mean = f.getMean(ppnm, 'USG')

            usm_min = usl_min + usg_min
            usm_max = usl_max + usg_max
            usm_mean = usl_mean + usg_mean

            tpl.dfsuper.set_value(i, 'USL min', usl_min)
            tpl.dfsuper.set_value(i, 'USL max', usl_max)
            tpl.dfsuper.set_value(i, 'USL mean', usl_mean)
            """
            читаем и преобразуем данные по скорости движения газа
            """
            tpl.dfsuper.set_value(i, 'USG min', usg_min)
            tpl.dfsuper.set_value(i, 'USG max', usg_max)
            tpl.dfsuper.set_value(i, 'USG mean', usg_mean)

            tpl.dfsuper.set_value(i, 'USM min', usm_min)
            tpl.dfsuper.set_value(i, 'USM max', usm_max)
            tpl.dfsuper.set_value(i, 'USM mean', usm_mean)

            forceFrac = forceFraction(uslug_max, tpl.densliq_kgm3, tpl.ID_mm, tpl.weight_kgm, Hlslug=hslug,
                                      Hlfilm=hfilm)
            tpl.dfsuper.set_value(i, 'forceFrac', forceFrac)

            i = i + 1


def elbowForce_kN(vel_ms, rho_kgm3=800, ID_mm=800, Theta_deg=90, Hlslug=1, Hlfilm=0.5):
    """
    Расчет усилия действующего на сгиб трубы
    """
    DLF = 1
    Area_m2 = 3.14 / 4 * (ID_mm / 1000) ** 2
    Theta_rad = Theta_deg / 180 * 3.14
    xForce_N = DLF * rho_kgm3 * (Hlslug - Hlfilm) * vel_ms ** 2 * Area_m2 * math.sin(
        Theta_rad)  # (2 * (1 - Cos(Theta_rad))) ^ (0.5)
    return xForce_N / 1000


def critForce_kN(weight_kgm=200, rho_kgm3=800, ID_mm=800, Hlslug=1, Hlfilm=0.5, lengthPipe_m=15):
    """
    Расчет критического усилия исходя из трения трубы об опору
    """
    Area_m2 = 3.14 / 4 * (ID_mm / 1000) ** 2
    weight_fluid_sl_kgm = Area_m2 * rho_kgm3 * (Hlslug)
    #    weight_fluid_flm_kgm = Area_m2 * rho_kgm3 * (Hlfilm)
    critForce_kN = (weight_kgm + weight_fluid_sl_kgm) * 9.81 * lengthPipe_m / 1000
    friction = 0.3  # ' ñòàëü - ñòàëü
    return critForce_kN * friction


def forceFraction(vel_ms, rho_kgm3=800, ID_mm=800, weight_kgm=200, Theta_deg=90, Hlslug=1, Hlfilm=0.5, lengthPipe_m=15):
    return elbowForce_kN(vel_ms, rho_kgm3, ID_mm, Theta_deg, Hlslug, Hlfilm) / critForce_kN(weight_kgm, rho_kgm3, ID_mm,
                                                                                            Hlslug, Hlfilm,
                                                                                            lengthPipe_m)

#        print(i,tplSplit.split(params[i]))
