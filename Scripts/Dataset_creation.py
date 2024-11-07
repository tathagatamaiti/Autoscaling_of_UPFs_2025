import pandas as pd
import random

# Seed for reproducibility
random.seed(42)

# Generate PDU sessions dataset with sequential start times
def generate_pdu_sessions(filename='pdu_sessions.csv', num_pdus=2):
    pdu_data = []
    for i in range(1, num_pdus + 1):
        start = i * 10  # Sequential start times
        end = start + random.randint(1, 50)  # Random duration added to start time
        rate = random.randint(10, 20)  # Rate
        latency = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])  # Latency requirement

        # Append to the dataset
        pdu_data.append({
            'id': i,
            'start': start,
            'end': end,
            'latency': latency,
            'rate': rate
        })

    # Convert to DataFrame and save as CSV
    df_pdus = pd.DataFrame(pdu_data)
    df_pdus.to_csv(filename, index=False)
    print(f"PDU sessions dataset saved as '{filename}'")


# Generate UPF instances dataset
def generate_upf_instances(filename='upf_instances.csv', num_upfs=1):
    upf_data = []
    for i in range(1, num_upfs + 1):
        workload_factor = 1  # Workload factor
        cpu_capacity = 200  # CPU capacity

        # Append to the dataset
        upf_data.append({
            'instance_id': f'u{i}',
            'workload_factor': workload_factor,
            'cpu_capacity': cpu_capacity
        })

    # Convert to DataFrame and save as CSV
    df_upfs = pd.DataFrame(upf_data)
    df_upfs.to_csv(filename, index=False)
    print(f"UPF instances dataset saved as '{filename}'")


# Run the data generation functions
generate_pdu_sessions()
generate_upf_instances()
