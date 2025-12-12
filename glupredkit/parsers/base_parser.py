import numpy as np


class BaseParser:
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        """
        Optional additional parameters such as api username and key.

        Returns four dataframes on the format specified in the README.
        """
        raise NotImplementedError("Metric not implemented!")

    def __repr__(self):
        return self.__class__.__name__

    def remove_outliers_and_following(self, df, remove_cols: list, lower_treshold=0, upper_treshold=50, extend_lim=12*8):
        """
        This function takes a feature, and removes an outlier based on upper and lower criterion, including the
        following hours of data.

        extend_lim is the number of samples that will be removed following the outlier.
        """
        # Remove current and following 8 hrs of insulin if outlier
        for dose_col in remove_cols:
            if dose_col in df.columns:
                bad_idx = df.index[(df[dose_col] < lower_treshold) | (df[dose_col] > upper_treshold)]
                if len(bad_idx) > 0:
                    print(f"Warning: Subject {df} has {len(bad_idx)} outlier {dose_col} values. "
                          "We set the value and the following eight hours of data to nan.")
                    rows_to_nan = []
                    for idx in bad_idx:
                        loc = df.index.get_loc(idx)  # safe unless duplicates exist
                        rows_to_nan.extend(range(loc, loc + extend_lim))
                    rows_to_nan = [i for i in rows_to_nan if i < len(df)]
                    insulin_col = df.columns.get_loc(dose_col)
                    df.iloc[rows_to_nan, insulin_col] = np.nan
        return df

