import numpy as np
import psycopg2
from datetime import datetime
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

# Generate realistic data
def generate_security_incidents(n_samples=30):
    # Generate incident durations (convert to standard Python int)
    incident_durations = [int(x) for x in np.random.normal(60, 20, n_samples)]
    incident_durations = [max(1, x) for x in incident_durations]  # ensure positive values
    
    # Generate data leaked (convert to float)
    data_leaked = [float(round(x, 2)) for x in np.random.lognormal(0, 1, n_samples) * 2]
    
    # Generate detection times (convert to float)
    detection_times = [float(round(x, 2)) for x in np.random.exponential(2, n_samples)]
    
    return list(zip(incident_durations, data_leaked, detection_times))

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # Generate and insert data
    incidents = generate_security_incidents()
    
    for incident in incidents:
        cur.execute("""
            INSERT INTO security_incidents 
            (incident_duration, data_leaked, detection_time)
            VALUES (%s, %s, %s)
        """, incident)
    
    conn.commit()
    print(f"Successfully inserted {len(incidents)} records")

except Exception as e:
    print(f"Error: {e}")
finally:
    if conn:
        cur.close()
        conn.close()