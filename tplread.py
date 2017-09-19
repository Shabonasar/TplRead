# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:54:56 2017

@author: Khabibullin Rinat

анализ данных сгенерированных в OLGA 

"""

# import pyfas as fa

import os
import pandas as pd
from glob import glob
import numpy as np
import re


"""
Tpl class   from pyfas 
"""


class Tpl:
    """
    Data extraction for tpl files (OLGA >= 6.0)
    """
    def __init__(self, fname):
        """
        Initialize the tpl attributes
        """
        if fname.endswith(".tpl") is False:
            raise ValueError("not a tpl file")
        self.fname = fname.split(os.sep)[-1]
        self.path = os.sep.join(fname.split(os.sep)[:-1])
        if self.path == '':
            self.abspath = self.fname
        else:
            self.abspath = self.path+os.sep+self.fname
        self._attributes = {}
        self.data = {}
        self.label = {}
        self.trends = {}
        self.time = ""
        with open(self.abspath) as fobj:
            for idx, line in enumerate(fobj):
                if 'CATALOG' in line:
                    self._attributes['CATALOG'] = idx
                    self._attributes['nvars'] = idx+1
                if 'TIME SERIES' in line:
                    self._attributes['data_idx'] = idx
                    break
                if 'CATALOG' in self._attributes:
                    adj_idx = idx-self._attributes['CATALOG']-1
                    if adj_idx > 0:
                        self.trends[adj_idx] = line

    def filter_data(self, pattern=''):
        """
        Filter available varaibles
        """
        filtered_trends = {}
        with open(self.abspath) as fobj:
            for idx, line in enumerate(fobj):
                variable_idx = idx-self._attributes['CATALOG']-1
                if 'TIME SERIES' in line:
                    break
                if pattern in line and variable_idx > 0:
                    filtered_trends[variable_idx] = line
        return filtered_trends

    def extract(self, variable_idx):
        """
        Extract a specific variable
        """
        self.time = np.loadtxt(self.abspath,
                               skiprows=self._attributes['data_idx']+1,
                               unpack=True, usecols=(0,))
        data = np.loadtxt(self.abspath,
                          skiprows=self._attributes['data_idx']+1,
                          unpack=True,
                          usecols=(variable_idx,))
        with open(self.abspath) as fobj:
            for idx, line in enumerate(fobj):
                if idx == 1 + variable_idx+self._attributes['CATALOG']:
                    try:
                        self.data[variable_idx] = data[:len(self.time)]
                    except TypeError:
                        self.data[variable_idx] = data.base
                    self.label[variable_idx] = line.replace("\'",
                                                            '').replace("\n",
                                                                        "")
                    break

    def to_excel(self, *args):
        """
        Dump all the data to excel, fname and path can be passed as args
        """
        path = os.getcwd()
        fname = self.fname.replace(".tpl", "_tpl") + ".xlsx"
        idxs = self.filter_data("")
        for idx in idxs:
            self.extract(idx)
        data_df = pd.DataFrame(self.data)
        data_df.columns = self.label.values()
        data_df.insert(0, "Time [s]", self.time)
        if len(args) > 0 and args[0] != "":
            path = args[0]
            if os.path.exists(path) == False:
                os.mkdir(path)
            data_df.to_excel(path + os.sep + fname)
        else:
            data_df.to_excel(self.path + os.sep + fname)


class TplFile(Tpl):
    """
    read and parse one tpl file
    """

    def __init__(self, filename):
        super().__init__(filename)
        self.data_all = 0
        self.data_trends = pd.DataFrame()
        self.data_trends_summary = 0
        self.q_liq_m3day = 0
        self.q_gas_Mm3day = 0
        self.p_atm = 10
        self.file_name = filename
        self.tpl_split = re.compile(r'\'*\s*\'', re.IGNORECASE)
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
        dic = {}
        dic.update({0: 'time'})
        for v in self.trends:
            dic.update({v: self.trends[v].replace("'", " ")})
        self.df.columns = dic.values()
        self.df.index = self.df.time.round(0)

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
        return self.data_trends

    def get_trend_summary(self):
        """calculate summary data on trends extracted"""
        self.data_trends_summary = pd.DataFrame()
        self.data_trends_summary['mean'] = self.data_trends.mean()
        self.data_trends_summary['min'] = self.data_trends[20:].min()
        self.data_trends_summary['max'] = self.data_trends.max()
        self.data_trends_summary['key'] = self.data_trends.columns.str.split(pat=':').str[0]
        self.data_trends_summary['point'] = self.data_trends.columns.str.split(":").str[1]
        self.data_trends_summary['q_liq'] = self.q_liq_m3day
        self.data_trends_summary['q_gas'] = self.q_gas_Mm3day
        self.data_trends_summary['p_end'] = self.p_atm
        return self.data_trends_summary

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
        self.plist = []
        self.df = pd.DataFrame()
        self.df_super = pd.DataFrame()
        self.key_list = ['HOLEXP', 'USFEXP', 'USL', 'USG']
        self.pipe_list = ['Pipe-3']
        self.flSplit = re.compile(r'[-,\s.]', re.IGNORECASE)
        self.num_table = pd.DataFrame()
        """
        параметры трубы
        """
        self.ID_mm = 800  # внутренний диамет трубы
        self.dens_liq_kgm3 = 800  # плотность жидкости
        self.weight_kgm = 200  # удельный вес трубы

    def read_data(self):
        """
        read all files
        tries to get params from file names
        :return:
        """
        self.num_table = pd.DataFrame(columns=['num', 'q_liq', 'q_gas', 'p_end'])
        self.file_num = 0
        for file in self.files:                                     # iterating through all files
            try:
                par = self.flSplit.split(file)
                fl_read = TplFile(file)                             # read file
                fl_read.q_liq_m3day = float(par[2])
                fl_read.q_gas_Mm3day = float(par[4])
                fl_read.p_atm = float(par[6])
                fl_read.get_trend(self.key_list, self.pipe_list)
                fl_read.get_trend_summary()
                self.file_list.update({self.file_num: fl_read})     # put reader object to dictionary
                self.plist.append(fl_read.p_atm)
                self.num_table.loc[self.file_num] = [self.file_num, fl_read.q_liq_m3day,
                                                     fl_read.q_gas_Mm3day, fl_read.p_atm]
                self.file_num = self.file_num + 1
                print(file + ' read done')
            except Exception:
                print(file + ' read failed')
                # here can be some potential error - need better exception handler
        df_list = []
        for df in self.file_list.values():
            df_list.append(df.data_trends_summary)
        self.df = pd.concat(df_list)
        print('read done')

    def calc_data(self):
        """
        recalculate main DataFrame into table with
        row - some point (file) with point, q_liq, q_gas, p and slug_vel mech_factor params
        :return:
        """
        agr_type = ['mean', 'min', 'max']
        self.df_super = pd.pivot_table(self.df,
                                       values=['mean', 'max', 'min'],
                                       index=['point', 'q_liq', 'q_gas', 'p_end'],
                                       columns=['key'])
        self.df_super = self.df_super.reset_index()
        """
        calc additional data needed to calculate all params
        """
        ''' superficial mixture velocity - min, mean, max '''
        for agr in agr_type:
            self.df_super[agr, 'USM'] = self.df_super[agr, 'USG'] + self.df_super[agr, 'USL']
        """ slug velocity """
        self.df_super['slug_velocity'] = self.df_super['max', 'USFEXP']
        self.df_super['slug_holdup'] = self.df_super['max', 'HOLEXP'] - self.df_super['min', 'HOLEXP']
        self.df_super['mech'] = force_fraction(self.df_super['max', 'USM'],
                                               800 * self.df_super['slug_holdup'])
        return 1

    def get_matr_ql_qg(self, pipe=0, p_end=0, val='mech'):

        """
        convert data read to some format
        :param pipe:
        :param p_end:
        :param val:
        :return:
        """
        if type(pipe) == int:
            pipe = self.pipe_list[pipe]
        if type(p_end) == int:
            p_end = self.plist[p_end]                            
        df1 = pd.pivot_table(self.df_super, index=['point', 'p_end', 'q_liq'], columns=['q_gas'], values=val)
        df1.columns = df1.columns.droplevel(0)
        df1 = df1.reset_index()
        df1.index = df1.q_liq
        df1 = df1[df1.point == pipe]
        df1 = df1[df1.p_end == p_end]
        df1 = df1.drop(['point', 'p_end', 'q_liq'], 1)
        return df1
    
    def get_number_tpl(self, q_liq_list, q_gas_list, p_end_list):
        nt = self.num_table
        out = nt[(nt.q_liq.isin(q_liq_list)) & (nt.q_gas.isin(q_gas_list)) & 
                 (nt.p_end.isin(p_end_list))]['num'].values
        return out
    
    def get_trend(self, q_liq_list, q_gas_list, p_end_list, point_list, key_list):
        n = self.get_number_tpl(q_liq_list, q_gas_list, p_end_list)
        out = pd.DataFrame()
        for file_num in n:
            a = self.file_list[file_num].get_trend(key_list, point_list)
            a.columns = [a.columns[0] + ' :q_liq ' + str(self.file_list[file_num].q_liq_m3day)
                                      + ' :q_gas ' + str(self.file_list[file_num].q_gas_Mm3day)
                                      + ' :p_end ' + str(self.file_list[file_num].p_atm)]
            out = pd.concat([out, a], axis=1)
        return out


def elbow_force_kN(vel_ms, rho_kgm3=800, id_mm=800, theta_deg=90, holdup_slug=1, holdup_film=0.5):
    """
    Расчет усилия действующего на сгиб трубы
    """
    DLF = 1
    area_m2 = 3.14 / 4 * (id_mm / 1000) ** 2
    theta_rad = theta_deg / 180 * 3.14
    x_force__n = DLF * rho_kgm3 * (holdup_slug - holdup_film) * vel_ms ** 2 * area_m2 * np.sin(theta_rad)
    return x_force__n / 1000


def crit_force_kN(weight_kgm=200, rho_kgm3=800, id_mm=800, holdup_slug=1, length_pipe_m=15):
    """
    Расчет критического усилия исходя из трения трубы об опору
    """
    area_m2 = 3.14 / 4 * (id_mm / 1000) ** 2
    weight_fluid_sl_kgm = area_m2 * rho_kgm3 * holdup_slug
    crit_force_k_n = (weight_kgm + weight_fluid_sl_kgm) * 9.81 * length_pipe_m / 1000
    friction = 0.3
    return crit_force_k_n * friction


def force_fraction(vel_ms, rho_kgm3=800, id_mm=800, weight_kgm=200, theta_deg=90,
                   holdup_slug=1, holdup_film=0.5, length_pipe_m=15):

    ef = elbow_force_kN(vel_ms, rho_kgm3, id_mm, theta_deg, holdup_slug, holdup_film)
    cf = crit_force_kN(weight_kgm, rho_kgm3, id_mm, holdup_slug, length_pipe_m)
    return ef / cf
