import numpy as np
import pandas as pd 
import pickle


# load onehot encoder
with open("preprocessors/onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# load scaler
with open("preprocessors/standard-scalen.pkl", "rb") as f:
    scaler = pickle.load(f)


# TODO
COLUMNS_TO_REMOVE = [
    
    "Over18","EmployeeCount", 'EmployeeNumber', 'StandardHours',
    'EducationField_Human Resources',
    'EducationField_Other',
    'Education',
     'JobRole_Human Resources',
     'EducationField_Marketing',
     'JobRole_Research Scientist',
     'TrainingTimesLastYear',
     'BusinessTravel_Travel_Frequently',
     'MaritalStatus_Divorced',
     'PerformanceRating',
     'JobRole_Laboratory Technician',
     'MaritalStatus_Single',
     'MonthlyRate',
     'MeanYearIncome',
     'EducationField_Medical',
     'Gender_Male',
     'EducationField_Technical Degree',
     'OverYearIncomeAvg',
     'YearsSinceLastPromotion',
     'StockOptionLevel',
     'JobRole_Sales Representative',
     'EducationField_Life Sciences',
     'RelationshipSatisfaction'

]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"

]


def preprocess(sample: dict) -> np.ndarray:
    sample_df = pd.DataFrame(sample, index=[0])

    sample_df = create_features(sample_df)
    sample_df = encode_columns(sample_df)
    
    sample_df = drop_columns(sample_df)    
    scaled_sample_values = scale(sample_df.values)
    scaled_sample_values = scaled_sample_values.reshape(1, -1)
    return scaled_sample_values


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_REMOVE)


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe with one-hot encoded columns
    encoded_df = pd.DataFrame(onehot_encoder.transform(df[COLUMNS_TO_ONEHOT_ENCODE]).toarray())
    # set new column names
    column_names = onehot_encoder.get_feature_names(COLUMNS_TO_ONEHOT_ENCODE)
    encoded_df.columns = column_names
    # drop raw columns, and add one-hot encoded columns instead
    df = df.drop(columns=COLUMNS_TO_ONEHOT_ENCODE, axis=1)
    df = df.join(encoded_df)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # create AvgAttrition feature
    df["AvgAttrition"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    # create MeanYearIncome
    meanYearIncome = df.groupby("TotalWorkingYears")["MonthlyIncome"].mean()
    df.groupby("TotalWorkingYears")["MonthlyIncome"].count()
    
    df["MeanYearIncome"] = df.apply(
    lambda x: meanYearIncome[x["TotalWorkingYears"]], axis=1)
    #create OverYearIncomeAvg
    df['OverYearIncomeAvg'] = df.apply(
        lambda x: 1 if x['MonthlyIncome'] > x['MeanYearIncome'] else 0, axis=1) 
    # YearsAtCompany convert into categories
    bins = pd.IntervalIndex.from_tuples([(-1,5),(5,10),(10,15),(15,100)])
    cat_YearsAtCompany = pd.cut(df["YearsAtCompany"].to_list(), bins)
    cat_YearsAtCompany.categories = [0,1,2,3]
    df["YearsAtCompanyCat"] = cat_YearsAtCompany
    df.drop(columns=["YearsAtCompany"], inplace=True, axis=1)
    
    # TODO

    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
