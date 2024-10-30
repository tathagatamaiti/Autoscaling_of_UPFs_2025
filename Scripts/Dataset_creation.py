import pandas as pd
import numpy as np

np.random.seed(42)

# Generate PDU Sessions Dataset
num_pdu_sessions = 50
pdu_ids = [f'pdu_{i}' for i in range(num_pdu_sessions)]
start_times = np.sort(np.random.choice(range(0, 100), size=num_pdu_sessions, replace=False))  # Random start times
activity_durations = np.random.randint(1, 10, size=num_pdu_sessions)  # Random activity durations
end_times = start_times + activity_durations  # End times based on start times and durations
rates = np.random.randint(100, 1000, size=num_pdu_sessions)  # Random rates in bits per second
latencies = np.random.uniform(0.1, 1.0, size=num_pdu_sessions)  # Random latencies in seconds

# Create PDU DataFrame
pdu_data = pd.DataFrame({
    'id': pdu_ids,
    'start_time': start_times,
    'end_time': end_times,
    'rate': rates,
    'latency': latencies,
})

# Generate UPF Instances Dataset
num_upf_instances = 5
upf_ids = [f'upf_{i}' for i in range(num_upf_instances)]
workload_factors = np.random.uniform(0.01, 0.1, size=num_upf_instances)  # Workload factors
cpu_capacities = np.random.randint(1000, 5000, size=num_upf_instances)  # CPU capacities in cycles

# Create UPF DataFrame
upf_data = pd.DataFrame({
    'id': upf_ids,
    'workload_factor': workload_factors,
    'cpu_capacity': cpu_capacities,
})

# Save datasets to CSV files
pdu_data.to_csv("pdu_data.csv", index=False)
upf_data.to_csv("upf_data.csv", index=False)
