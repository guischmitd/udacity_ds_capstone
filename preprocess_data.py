import numpy as np
import pandas as pd

cat_cols = pd.read_csv('data/clean/cat_cols.csv', header=None)
dias_atts = pd.read_csv('data/clean/dias_atts.csv')
cols_to_drop = pd.read_csv('data/clean/cols_to_drop.csv', header=None)


def replace_invalid(value) -> float:
    """Tries to cast a value as float, otherwise returns NaN"""

    if pd.isna(value):
        # If value is already NaN, keep it as is
        return value
    
    else:
        try:
            out = float(value)
        except Exception as e:
            # If cast raises an error, return NaN
            out = np.nan

        return out


def replace_unknown_with_na(df_, dias_atts=dias_atts):
    """
        Using the attribute description file (dias_atts), replaces values
        that indicate unknown data with np.NaN.
        Returns a copy of the dataframe after replacing.
    """

    df = df_.copy()
    unknown_mapping = dias_atts[~dias_atts['isna'].isna()].copy()

    unknown_mapping_dict = {}
    grouped = unknown_mapping.groupby('Attribute')

    for g in grouped.groups:
        nan_vals = (grouped.get_group(g)['Value'].map(str)+',').sum().strip(',').split(',')
        nan_vals = [int(val) for val in nan_vals]
        unknown_mapping_dict[g] = {k: np.nan for k in nan_vals}

    # Extra feature `nan` mapping found after looking at the histograms
    unknown_mapping_dict['ALTERSKATEGORIE_FEIN'] = {0: np.nan}
    
    for col in unknown_mapping_dict:
        if col in df.columns:
            df[col] = df[col].replace(unknown_mapping_dict[col])

    return df


def impute_missing_with_mode(df):
    """Fills NaN values in a dataframe using the mode of each column"""
    return df.fillna(df.mode().iloc[0])


def optimize_dataframe(df):
    """
        Tries to downcast all numeric columns of a dataframe to the lowest
        memory consuming integer type possible. Returns a copy of the dataframe
        after performing optimizations.
        
        *Warning:* Should only be used on dataframes with discrete numeric columns
    """
    df_ = df.copy()
    for c in df.columns:
        df_[c] = pd.to_numeric(df[c], errors='ignore', downcast='integer')
    
    return df_


def preprocess(df, drop_na_rows=False):
    print('\tRaw data info:')
    df.info()

    # Replace invalid values (strings in float columns)
    df[['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']] = df[['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']].applymap(replace_invalid)

    # Parse datetime cols
    df['EINGEFUEGT_AM'] = pd.to_datetime(df['EINGEFUEGT_AM'])
    
    # Set `LNR` as the index
    df = df.set_index('LNR', drop=True)

    # Replace "Unknown" values with NaN (as described in the attribute description files)
    df = replace_unknown_with_na(df)
    
    print('\n*** Dropping columns/rows and imputing missing values...')
    # Drop columns listed in the `cols_to_drop.csv` file
    for col in cols_to_drop[0].tolist():
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Drop rows with more than 5% missing features (if requested)
    if drop_na_rows:
        df_na_percent_row = df.isna().mean(axis=1)
        df = df.drop(df_na_percent_row[df_na_percent_row > 0.05].index)

    # Impute missing values
    print(f'\tMissing values in df before imputing: {df.isna().sum().sum()}')
        # Impute the ALTER_KIND cols with `0` for every NaN
    kinder_cols = ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4']
    df[kinder_cols] = df[kinder_cols].fillna(0)
        # Impute the remaining cols with the mode
    df = impute_missing_with_mode(df)
    print(f'\tMissing values in df now: {df.isna().sum().sum()}')

    # Perform optimization (downcasting) and load to file
    print('\n\tBefore optimization:')
    df.info()
    df = optimize_dataframe(df)
    print('\n\tAfter optimization:')
    df.info()

    return df


if __name__=='__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type=str, help='Path to the input csv file.', required=True)
    ap.add_argument('-o', '--output', type=str, help='Path to the output parquet file.')
    ap.add_argument('-dnr', '--dropnarows', action='store_true', help='Whether to drop rows with more than 5% NaN values.')

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = args.output or in_path.parent / (in_path.stem + '.parquet')
    print(f'DNR set to {args.dropnarows}')

    print('\n*** Reading and parsing dataframe...')
    df = pd.read_csv(in_path, sep=';')
    df = preprocess(df, drop_na_rows=bool(args.dropnarows))

    df.to_parquet(out_path)
    print('\n *** Preprocessed dataset saved to disk ***')


