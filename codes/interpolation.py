import pandas as pd
import numpy as np
import itertools
from .categorization import *

def interpolate(combined_df, fix_elements, threshold=5):
    """
    Interpolate missing performance values by averaging available data while fixing certain elements.
        method A (with category A')
        fine-tuning model B (with model size B')
        on dataset C (with task type C')
    Fix some of subset of {A, A', B, B', C, C'} and take the avg
    If not enough data for a criteria (eg A), fix its ' (eg A') instead
    eg. Interpolating (x): HiZOO fine-tuning llama2-7b on the SST-2 dataset
        fix_elements = ("A", "B'", "C'")
        x = mean(HiZOO fine-tuning large model on sentiment analysis dataset)

    Parameters:
    - combined_df (pd.DataFrame): Merged DataFrame containing all method types.
    - fix_elements (tuple): Elements to fix during interpolation.
    - threshold (int, optional): Minimum number of available data points required to refine filtering. Default is 5.

    Returns:
    - pd.DataFrame: The DataFrame with interpolated missing values.
    """

    assert isinstance(combined_df, pd.DataFrame)
    assert isinstance(fix_elements, list) and len(fix_elements) <= 6
    assert all([element in ("A", "A'", "B", "B'", "C", "C'") for element in fix_elements])
    assert isinstance(threshold, int) and threshold > 0

    META_INFO = ["Method", "Model", "method_type", "model_type"]

    new_df = combined_df.copy()
    for i, row in combined_df.iterrows():
        method, model = row.iloc[0], row.iloc[1]
        method_type = row['method_type']
        model_type = get_model_size(model)

        for j, entry in enumerate(row[2:]):
            if isinstance(entry, str) or not np.isnan(entry):
                continue

            dataset_name = combined_df.columns[j+2]
            dataset_type = get_dataset_type(dataset_name)
            sample_df = combined_df

            # filter by general category
            if "A'" in fix_elements:
                sample_df_experiment = sample_df[sample_df['method_type'] == method_type]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment

            if "B'" in fix_elements:
                sample_df_experiment = sample_df[sample_df['model_type'] == model_type]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment

            if "C'" in fix_elements:
                good_columns = [get_dataset_type(col) == dataset_type or col in META_INFO
                                for col in combined_df.columns]
                sample_df_experiment = sample_df.iloc[:, good_columns]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment

            # filter by specific if possible
            if "A" in fix_elements:
                sample_df_experiment = sample_df[sample_df['Method'] == method]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment
            if "B" in fix_elements:
                sample_df_experiment = sample_df[sample_df['Model'] == model]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment
            if "C" in fix_elements:
                sample_df_experiment = sample_df.loc[:, META_INFO+[dataset_name]]
                if sample_df_experiment.drop(columns=META_INFO).apply(lambda row: row.notna().sum(), axis=1).sum() >= threshold:
                    sample_df = sample_df_experiment

            # calculate
            new_df.iloc[i, j+2] = sample_df.drop(columns=META_INFO).mean(skipna=True).mean(skipna=True)
            if np.isnan(new_df.iloc[i, j+2]):
                print(i, j+2)
                print(sample_df.iloc[:, 2:])
                return new_df
    return new_df