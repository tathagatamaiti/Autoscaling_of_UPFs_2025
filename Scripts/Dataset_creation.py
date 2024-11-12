import pandas as pd
import random

# Seed for reproducibility
random.seed(42)

# Generate PDU sessions dataset with specified corner cases
def generate_pdu_sessions_with_corner_cases(filename='pdu_sessions.csv', num_pdus=5):
    pdu_data = []
    for i in range(1, num_pdus + 1):
        if i == 1 or i == 2:  # PDU 1 and 2 start at the same time
            start = 10
        elif i == 3:  # PDU 3 starts when PDU 2 terminates
            start = 20
        else:
            start = i * 10

        if i == 2:  # PDU 2 terminates at the same time PDU 3 starts
            end = 20
        elif i == 4 or i == 5:  # PDU 4 and 5 terminate at the same time
            end = 60
        else:
            end = start + random.randint(1, 50)  # Random duration

        rate = random.randint(10, 20)
        latency = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])

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
    print(f"PDU sessions dataset with corner cases saved as '{filename}'")

# Generate UPF instances dataset
def generate_upf_instances(filename='upf_instances.csv', num_upfs=2):
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

# Run the data generation functions with corner cases
generate_pdu_sessions_with_corner_cases()
generate_upf_instances()
