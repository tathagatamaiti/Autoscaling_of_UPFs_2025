import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
merged_data = pd.read_csv('offline_solution.csv')
pdu_sessions = pd.read_csv('pdu_sessions.csv')

# Merge observed latency data with required latency for each PDU session
merged_data = merged_data.merge(pdu_sessions[['id', 'latency']], left_on='PDU_session', right_on='id', how='left')
merged_data.rename(columns={'latency': 'Required_latency'}, inplace=True)

# Calculate the average observed latency for each PDU session
average_observed_latency = merged_data.groupby('PDU_session')['Observed_latency'].mean()

# Calculate the normalized averaged latency for each PDU session
normalized_average_latency = average_observed_latency / merged_data.groupby('PDU_session')['Required_latency'].first()

# Plot the normalized time-averaged latency for each PDU session
plt.figure(figsize=(20, 10))
plt.scatter(normalized_average_latency.index, normalized_average_latency.values, marker='o')
plt.axhline(y=1, color='r', linestyle='--', label='Normalized Latency Bound')
plt.xlabel('PDU Session', fontsize=20)
plt.ylabel('Normalized Average Latency', fontsize=20)
plt.title('Normalized Average Latency for Each PDU Session', fontsize=25)
plt.xticks(normalized_average_latency.index)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)

#Save plot
plt.savefig('PDU_latency.png', format='png')
