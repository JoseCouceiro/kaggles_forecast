import numpy as np
import pandas as pd

class PreProcessData:
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    def __init__(self, df_train, df_test):
        self.date_index_dfs = self.process_dataframe(df_train, df_test)
        self.clean_dic = self.get_dic(self.date_index_dfs)

    def set_time_index(self, df):
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.set_index('date')
        df = df.sort_index()
        return df

    def make_multiseries_dictionary(self, df1, df2):
        grouped_df1 = df1.groupby(['country', 'store', 'product'])
        grouped_df2 = df2.groupby(['country', 'store', 'product'])

        combined_dict = {
            key: [group1, grouped_df2.get_group(key)]
            for key, group1 in grouped_df1
            if key in grouped_df2.groups 
        }
        return combined_dict
    
    def check_index_dataframes(self, dic):
        for key in dic.keys():
            dic[key][0] = dic[key][0].asfreq('D')
            dic[key][0] = dic[key][0].sort_index()
            dic[key][1] = dic[key][1].asfreq('D')
            dic[key][1] = dic[key][1].sort_index()
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
        for lst in multi_dict.values():
            for group in lst:
                if group.isnull().any(axis=1).sum() == len(group):
                    group.fillna(0, inplace=True)
                elif group.isnull().any(axis=1).sum() > 0:
                    group.apply(self.fill_with_mean_neighbor)
        return multi_dict
    
    def process_dataframe(self, df_train, df_test):
        train_index_df = self.set_time_index(df_train)
        test_index_df = self.set_time_index(df_test)
        return train_index_df, test_index_df
    
    def get_dic(self, tup):
        multi_dic = self.make_multiseries_dictionary(tup[0], tup[1])
        checked_multi_dic = self.check_index_dataframes(multi_dic)
        clean_dic = self.handle_nans(checked_multi_dic)
        return clean_dic
    
class PostProcessData:

    def verify_index(self, dic):
        faulty_df = []
        for lst in dic.values():
            for df in lst:
                start_date = df.index.min()
                end_date = df.index.max()
                complete_date_range = pd.date_range(start=start_date, end=end_date, freq=df.index.freq)
                is_index_complete = (df.index == complete_date_range).all()
                if not is_index_complete:
                    faulty_df.append(df)
        print(f'Number of dataframes with uncomplete index: ', len(faulty_df))

    def train_test_split(self, df, steps):
        data_train = df[:-steps]
        data_test  = df[-steps:]
        print(
            f"Train dates : {data_train.index.min()} --- "
            f"{data_train.index.max()}  (n={len(data_train)})"
        )
        print(
            f"Test dates  : {data_test.index.min()} --- "
            f"{data_test.index.max()}  (n={len(data_test)})"
        )
        return data_train, data_test
    
    def rebuild_dataframe(self, dic):
        df = pd.concat(dic.values())
        sorted_df = df.sort_values('id')
        return sorted_df
    
class ProcessMultiseries:

    def group_df(self, grouped_dic):
        reorg_df = pd.DataFrame({'_'.join(key):df['num_sold'] for key, df in grouped_dic.items()}) 
        return reorg_df
    
    


