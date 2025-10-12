import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL connection parameters
db_params = {
    "database": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT')
}

def fetch_data():
    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
    query = "SELECT incident_duration, data_leaked, detection_time FROM security_incidents"
    df = pd.read_sql_query(query, engine)
    return df

def plot_distributions(df):
    # Create directory if it doesn't exist
    os.makedirs('assets/images', exist_ok=True)

    # Plot Incident Duration
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='incident_duration', kde=True, color='skyblue')
    plt.title('Distribution of Incident Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.savefig('assets/images/incident_duration.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Data Leaked
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='data_leaked', kde=True, color='lightgreen')
    plt.title('Distribution of Data Leaked')
    plt.xlabel('Data Leaked (MB)')
    plt.ylabel('Frequency')
    plt.savefig('assets/images/data_leaked.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Detection Time
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='detection_time', kde=True, color='salmon')
    plt.title('Distribution of Detection Time')
    plt.xlabel('Detection Time (hours)')
    plt.ylabel('Frequency')
    plt.savefig('assets/images/detection_time.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Fetch data from PostgreSQL
    df = fetch_data()
    
    # Generate and save plots
    plot_distributions(df)
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()