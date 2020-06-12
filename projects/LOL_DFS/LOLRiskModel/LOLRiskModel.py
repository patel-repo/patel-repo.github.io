import pandas as pd
import glob
import os
import lol_utils as lu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


COV_ORDER = ['Blue - team',
            'Blue - top',
            'Blue - jng',
            'Blue - mid',
            'Blue - bot',
            'Blue - sup',
            'Red - team',
            'Red - top',
            'Red - jng',
            'Red - mid',
            'Red - bot',
            'Red - sup']

class LOLRiskModel(object):
    def __init__(self, asof=None, league_filters=None):
        self.elixir_df = LOLRiskModel.read_oracle_elixir_data(asof, league_filters)


    @staticmethod
    def read_oracle_elixir_data(asof=None, league_filters=None):
        '''
        Reads in the latest file from Oracle's Elixier dir.
        :param asof: as of date for historical backtesting
        :return: historical data df
        '''

        list_of_files = glob.glob('OracleElixir/*')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        df = pd.read_csv(latest_file)
        df['date'] = pd.to_datetime(df['date'])

        if asof is not None:
            df = df[df['date'] < pd.to_datetime(asof)]
        if league_filters is not None:
            df = df[df['league'].apply(lambda x: x in league_filters)]

        return df

    def generate_cov_matrix(self):
        self.elixir_df = lu.compute_fanduel_scores(self.elixir_df)
        self.elixir_df['label'] = self.elixir_df['side'] + ' - ' + self.elixir_df['position']

        rez = {}
        gg = self.elixir_df.groupby('gameid')
        for gameid, group in gg:
            temp = group.set_index('label')
            rez[gameid] = temp['FD_points']
        data_df = pd.DataFrame(rez).T.reindex(columns = COV_ORDER)

        ax = sns.heatmap(data_df.corr(), vmin=-1, vmax=1, annot=True)
        plt.yticks(rotation=0)
        ax.xaxis.tick_top()  # x axis on top
        ax.xaxis.set_label_position('top')
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='left'
        )
        plt.show()

        return data_df

    def historical_point_averages(self):
        self.elixir_df['player_name'] = self.elixir_df['player'].combine_first(self.elixir_df['team'])
        return self.elixir_df.groupby('player_name').apply(lambda x: x['FD_points'].mean())



def test():
    rm = LOLRiskModel(league_filters=['LCS'])
    cov = rm.generate_cov_matrix()
    return rm.historical_point_averages()



