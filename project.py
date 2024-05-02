import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Missing Value Imputation Methods
def impute_central_tendency(df, feature, choice):
    if choice == 1:
        central_tendency = df[feature].mean()
    elif choice == 2:
        central_tendency = df[feature].median()
    elif choice == 3:
        central_tendency = df[feature].mode().iloc[0]
    return df[feature].fillna(central_tendency)

def impute_random_sample(df, feature):
    df[feature+"_random"] = df[feature]
    random_sample = df[feature].dropna().sample(df[feature].isnull().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature+'_random'] = random_sample
    return df

def impute_nan_importance(df, feature, choice):
    if choice == 1:
        central_tendency = df[feature].mean()
    elif choice == 2:
        central_tendency = df[feature].median()
    elif choice == 3:
        central_tendency = df[feature].mode().iloc[0]
    df[feature + '_filled_with_' + str(choice)] = df[feature].fillna(central_tendency)
    return df

def impute_end_distribution(df, feature):
    extreme = df[feature].mean() + 3*df[feature].std()
    df[feature+"_end_dist"] = df[feature].fillna(extreme)
    return df

def impute_frequent(df, feature):
    frequent = df[feature].mode()[0]
    df[feature+'_frequent'] = df[feature].fillna(frequent)
    return df

def impute_frequent_importance(df, feature):
    df[feature+'_frequent_importance'] = np.where(df[feature].isnull(), 1, 0)
    frequent = df[feature].mode()[0]
    df[feature+'_frequent'] = df[feature].fillna(frequent)
    return df

# Handling Categorical Values
def encode_target_guided(df, feature, target):
    ordinal_labels = df.groupby([feature])[target].mean().sort_values().index
    enu = {k: i for i, k in enumerate(ordinal_labels, 0)}
    df[feature+'_encode_targuided'] = df[feature].map(enu)
    return df

def encode_mean(df, feature, target):
    mean_ord = df.groupby([feature])[target].mean().to_dict()
    df[feature+'_encode_mean'] = df[feature].map(mean_ord)
    return df

# Transformations
def log_trans(df, feature):
    df[feature+'_log'] = np.log(df[feature])
    return df

def reci_trans(df, feature):
    df[feature+'_reci'] = 1/df[feature]
    return df

def sqrt_trans(df, feature):
    df[feature+'_sqrt'] = df[feature]**(1/2)
    return df

def expo_trans(df, feature):
    df[feature+'_expo'] = df[feature]**(1/1.2)
    return df

# Outlier Removal
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Model Training and Evaluation
def train_test_model(X, y, model_type):
    if model_type == 'Regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Support Vector Regression': SVR()
        }
    elif model_type == 'Classification':
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Support Vector Classifier': SVC()
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_type == 'Regression':
            results[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R^2': r2_score(y_test, y_pred)
            }
        elif model_type == 'Classification':
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Classification Report': classification_report(y_test, y_pred)
            }

    return results

# Streamlit App
st.title('Data Preprocessing and Model Evaluation App')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    selected_tasks = st.sidebar.multiselect('Select Preprocessing Tasks', ('Missing Value Imputation', 'Handling Categorical Values', 'Transformations', 'Outlier Removal'))

    if 'Missing Value Imputation' in selected_tasks:
        sub_task = st.sidebar.selectbox(
            'Select a missing value imputation method:',
            ('Fill using central tendency', 'Random Sampling', 'Imputing with Importance', 'End of Distribution', 'Frequent Categories', 'Frequent with Importance'))

        if sub_task == 'Fill using central tendency':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            choice = st.sidebar.selectbox('Choose a measure of central tendency:', ('Mean', 'Median', 'Mode'))
            if choice == 'Mean':
                choice = 1
            elif choice == 'Median':
                choice = 2
            elif choice == 'Mode':
                choice = 3
            df[feature] = impute_central_tendency(df, feature, choice)

        elif sub_task == 'Random Sampling':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            df[feature] = impute_random_sample(df, feature)

        elif sub_task == 'Imputing with Importance':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            choice = st.sidebar.selectbox('Choose a measure of central tendency:', ('Mean', 'Median', 'Mode'))
            if choice == 'Mean':
                choice = 1
            elif choice == 'Median':
                choice = 2
            elif choice == 'Mode':
                choice = 3
            df[feature] = impute_nan_importance(df, feature, choice)

        elif sub_task == 'End of Distribution':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            df[feature] = impute_end_distribution(df, feature)

        elif sub_task == 'Frequent Categories':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            df[feature] = impute_frequent(df, feature)

        elif sub_task == 'Frequent with Importance':
            feature = st.sidebar.selectbox('Select feature to impute:', df.columns)
            df[feature] = impute_frequent_importance(df, feature)

        st.write(df.head())

    if 'Handling Categorical Values' in selected_tasks:
        sub_task = st.sidebar.selectbox(
            'Select a categorical values handling method:',
            ('Target Guided', 'Mean Encoded'))

        if sub_task == 'Target Guided':
            feature = st.sidebar.selectbox('Select categorical feature:', df.select_dtypes(include='object').columns)
            target_variable = st.sidebar.selectbox('Select the target variable:', df.columns)
            df = encode_target_guided(df, feature, target_variable)

        elif sub_task == 'Mean Encoded':
            feature = st.sidebar.selectbox('Select categorical feature:', df.select_dtypes(include='object').columns)
            target_variable = st.sidebar.selectbox('Select the target variable:', df.columns)
            df = encode_mean(df, feature, target_variable)

        st.write(df.head())

    if 'Transformations' in selected_tasks:
        sub_task = st.sidebar.selectbox(
            'Select a transformation method:',
            ('Logarithmic', 'Reciprocal', 'Square Root', 'Exponential'))

        if sub_task == 'Logarithmic':
            feature = st.sidebar.selectbox('Select feature for transformation:', df.select_dtypes(include='number').columns)
            df = log_trans(df, feature)

        elif sub_task == 'Reciprocal':
            feature = st.sidebar.selectbox('Select feature for transformation:', df.select_dtypes(include='number').columns)
            df = reci_trans(df, feature)

        elif sub_task == 'Square Root':
            feature = st.sidebar.selectbox('Select feature for transformation:', df.select_dtypes(include='number').columns)
            df = sqrt_trans(df, feature)

        elif sub_task == 'Exponential':
            feature = st.sidebar.selectbox('Select feature for transformation:', df.select_dtypes(include='number').columns)
            df = expo_trans(df, feature)

        st.write(df.head())

    if 'Outlier Removal' in selected_tasks:
        df = remove_outliers(df)
        st.write(df.head())

    if st.sidebar.checkbox('Model Training and Evaluation'):
        target_variable = st.sidebar.selectbox('Select the target variable:', df.columns)
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        model_type = st.sidebar.selectbox('Select Model Type:', ('Regression', 'Classification'))

        if st.button('Train/Test Model'):
            results = train_test_model(X, y, model_type)
            st.write(results)

    user_input_section = st.sidebar.checkbox('Input your own data')
    if user_input_section:
        st.subheader('Input Your Own Data')
        user_data = {}
        for column in df.columns:
            user_input = st.sidebar.text_input(f'Enter value for {column}')
            user_data[column] = user_input

        if st.button('Predict'):
            user_df = pd.DataFrame(user_data, index=[0])
            user_df = user_df.astype(df.dtypes.to_dict())
            if model_type == 'Regression':
                prediction = model.predict(user_df)
                st.write(f'Predicted Value: {prediction}')
            elif model_type == 'Classification':
                prediction = model.predict(user_df)
                st.write(f'Predicted Class: {prediction}')
