import numpy as np
import pandas as pd

class ProcessData:
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    def __init__(self, df):
        self.clean_dic = self.process_dataframe(df)

    def set_time_index(self, df):
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.set_index('date')
        df = df.sort_index()
        return df

    def make_multiseries_dictionary(self, df):
        grouped = df.groupby(['country','store','product'])
        grouped_dataframes = {key: group for key, group in grouped}
        return grouped_dataframes
    
    def check_index_dataframes(self, dic):
        for key in dic.keys():
            dic[key] = dic[key].asfreq('D')
            dic[key] = dic[key].sort_index()
        return dic

    def fill_with_mean_neighbor(self, series):
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):  # Verify whether it is NaN
                # Get previous and following value
                prev_val = series.iloc[i - 1] if i > 0 else np.nan
                next_val = series.iloc[i + 1] if i < len(series) - 1 else np.nan
                # Calculate average
                neighbors = [val for val in [prev_val, next_val] if not pd.isna(val)]
                series.iloc[i] = np.mean(neighbors) if neighbors else np.nan
        return series
    
    def handle_nans(self, multi_dict):
        for group in multi_dict.values():
            if group.isnull().any(axis=1).sum() == len(group):
                group.fillna(0, inplace=True)
            elif group.isnull().any(axis=1).sum() > 0:
                group.apply(self.fill_with_mean_neighbor)
        return multi_dict
    
    def process_dataframe(self, df):
        time_index_df = self.set_time_index(df)
        multi_dic = self.make_multiseries_dictionary(time_index_df)
        checked_multi_dic = self.check_index_dataframes(multi_dic)
        clean_dic = self.handle_nans(checked_multi_dic)
        return clean_dic
    
    def rebuild_dataframe(self, dic):
        df = pd.concat(dic.values())
        sorted_df = df.sort_values('id')
        return sorted_df
    
    




