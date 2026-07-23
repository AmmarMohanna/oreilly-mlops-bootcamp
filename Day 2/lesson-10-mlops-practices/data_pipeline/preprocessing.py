import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    # Load both datasets with proper column names and missing value handling
    df_old = pd.read_csv("data/adult.data", header=None, names=columns, na_values=" ?")
    df_new = pd.read_csv("data/adult.newdata", header=None, names=columns, na_values=" ?")
    data = pd.concat([df_old, df_new], ignore_index=True)
    
    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Normalize whitespace in the UCI categorical columns.
    string_columns = data.select_dtypes(include="object").columns
    data[string_columns] = data[string_columns].apply(lambda col: col.str.strip())
    
    # Impute missing values
    missing_cols = ["workclass", "occupation", "native-country"]
    imputer = SimpleImputer(strategy="most_frequent")
    data[missing_cols] = imputer.fit_transform(data[missing_cols])
    
    
    # Encode target variables
    #print("Unique values in income before cleaning:", data["income"].unique())
    data["income"] = data["income"].str.replace(".", "", regex=False)
    #print("Cleaned income values:", data["income"].unique())
    data["income"] = data["income"].map({"<=50K": 0, ">50K": 1})
    
    # Features and target
    X = data.drop("income", axis=1)
    y = data["income"]

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Preprocessing pipeline
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    

    return X_train_res, X_test, y_train_res, y_test, preprocessor

# Run the preprocessing function

if __name__ == "__main__":
    load_and_preprocess_data()
