import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:yourpassword@localhost/diabetes_readmission")
#                                      ^^^^  ^^^^^^^^^^^
#                                      your MySQL username and password

def load_data():
    return pd.read_sql("SELECT * FROM diabetic_data", engine)