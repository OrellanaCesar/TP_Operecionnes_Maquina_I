import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.impute import KNNImputer
from typing import Tuple

# ============================================================================
# CONFIGURATION - Update these before starting
# ============================================================================
TARGET_COL = "stroke"
TEST_SIZE = 0.2
RANDOM_STATE = 13
SEED = 42

def split_and_store_data(input_path: str,
                         train_path: str, 
                         test_path: str,
                         storage_options: dict):
    """Lee el CSV crudo, divide en Train/Test y guarda como Parquet en S3."""
    
    # obtengo .csv
    df = pd.read_csv(input_path, storage_options=storage_options)
    # Eliminar columna ID (no aporta información predictiva) y 
    # eliminar el único registro con gender='Other' (outlier extremo)
    df = clean_data_id_gender(df)

    # Split estratificado: mantener la proporción de clases en train y test
    df_train, df_test = get_train_test_split(df, 0.2,SEED)
    
    # 3. Transformar: Aplicar tu imputación KNN
    # Como tu función impute_bmi_knn filtra internamente por 'cols_knn', 
    # es seguro pasarle el DataFrame completo sin que afecte a la columna 'stroke'.
    df_train, df_test = impute_bmi_knn(df_train, df_test)


    # 4. Cargar (Load) de nuevo al Data Lake
    df_train.to_parquet(train_path, index=False, storage_options=storage_options)
    df_test.to_parquet(test_path, index=False, storage_options=storage_options)
    
    return True



def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets.

    Uses a FIXED random_state so every team member gets the exact same split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df_train, df_test: tuple of pd.DataFrame / pd.Series
    """

    # 2. Dividir (Estratificado)
    # Aquí el split devuelve 2 DataFrames completos (features + target juntos)
    train_df, test_df, = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
    )
    print(
        f"Split: train={train_df.shape[0]} rows"
        f"(test_size={test_size}, seed={random_state})"
    )
    return train_df, test_df



def clean_data_id_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza inicial del dataset eliminando el atributo id y el unico registro
    con genero Other .

    Parameters
    ----------
    df : pd.DataFrame
        El DataFrame original que contiene los datos crudos.

    Returns
    -------
    pd.DataFrame
        Un nuevo DataFrame sin la columna 'id'.
    """
    if 'id' in df.columns:
        # Usamos copy() para evitar el SettingWithCopyWarning de pandas
        df = df.drop(columns=['id']).copy()
        print(f"Columna 'id' eliminada. Shape resultante: {df.shape}")
    else:
        print("La columna 'id' no se encontró en el DataFrame.")
    # Eliminar el único registro con gender='Other' (outlier extremo)
    df = df[df['gender'] != 'Other'].reset_index(drop=True)   
    return df


def impute_bmi_knn(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputa los valores faltantes de la columna 'bmi' utilizando KNN, evitando la fuga de datos.

    Utiliza un subconjunto de características numéricas relevantes para encontrar los 
    vecinos más cercanos. El imputer se entrena únicamente con el conjunto de 
    entrenamiento (X_train) para preservar la integridad del conjunto de prueba (X_test).

    Parameters
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento con posibles valores nulos en 'bmi'.
    X_test : pd.DataFrame
        Conjunto de prueba con posibles valores nulos en 'bmi'.

    Returns
    -------
    X_train : pd.DataFrame
        DataFrame de entrenamiento con los valores de 'bmi' imputados.
    X_test : pd.DataFrame
        DataFrame de prueba con los valores de 'bmi' imputados.

    Notes
    -----
    Se utiliza la configuración `weights='distance'`, lo que significa que los vecinos 
    más cercanos tienen una mayor influencia en el valor imputado que los más lejanos.
    Las columnas utilizadas para el cálculo son: 'age', 'avg_glucose_level', 'bmi', 
    'hypertension' y 'heart_disease'.
    """
    cols_knn = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']

    print(f"NaN en bmi (X_train) antes de KNN: {X_train['bmi'].isnull().sum()}")
    print(f"NaN en bmi (X_test)  antes de KNN: {X_test['bmi'].isnull().sum()}")
    
    # weights='distance' es excelente para capturar mejor la similitud local
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # Fit SOLO en X_train — transform en ambos por separado para evitar Data Leakage
    X_train[cols_knn] = knn_imputer.fit_transform(X_train[cols_knn])
    X_test[cols_knn]  = knn_imputer.transform(X_test[cols_knn])
    
    print(f"\nNaN en bmi (X_train) después de KNN: {X_train['bmi'].isnull().sum()}")
    print(f"NaN en bmi (X_test)  después de KNN: {X_test['bmi'].isnull().sum()}")

    return X_train, X_test