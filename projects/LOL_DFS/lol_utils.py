import pandas as pd
import numpy as np


UTIL_SCORING = {'player':
                    {
                        'assists': 2,
                        'kills': 3,
                        'deaths': -1,
                        'total cs': 0.02,
                        # 'games_not_played': 20
                    },

                'team':
                    {
                        'barons': 3,
                        'dragons': 2,
                        'firstblood': 2,
                        'towers': 1,
                        'result': 2,
                        'win_under_30': 2,
                        # 'games_not_played': 15
                    }
               }

def compute_fanduel_scores(elixir_df):
    '''
    Compute fantasy scores for each player using the FanDuel formula.

    CURRENTLY DOES NOT HANDLE BEST OF 3/5. NEED TO ADD.

    :param elixir_df: Historical data df
    :return:
    '''

    gg = elixir_df.groupby(['gameid'])
    player_scoring = pd.Series(UTIL_SCORING['player'])
    team_scoring = pd.Series(UTIL_SCORING['team'])

    rez = []

    for gameid, group in gg:
        players = group[group['player'].apply(lambda x: x == x)]
        teams = group[group['player'].apply(lambda x: x != x)]
        teams['game_under_30'] = (teams['gamelength'] / 60.).apply(lambda x: 1 if x < 30 else 0)
        teams['win_under_30'] = teams['result'] * teams['game_under_30']

        players['FD_points'] = players.reindex(columns = player_scoring.index).dot(player_scoring)
        teams['FD_points'] = teams.reindex(columns = team_scoring.index).dot(team_scoring)

        rez.append(players)
        rez.append(teams)

    return pd.concat(rez)
