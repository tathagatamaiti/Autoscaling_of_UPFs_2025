import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
merged_data = pd.read_csv('offline_solution.csv')
pdu_sessions = pd.read_csv('pdu_sessions.csv')

# Merge observed latency data with required latency for each PDU session
merged_data = merged_data.merge(pdu_sessions[['id', 'latency']], left_on='PDU_session', right_on='id', how='left')
merged_data.rename(columns={'latency': 'Required_latency'}, inplace=True)

# Calculate the normalized latency for each time instance of each PDU session
merged_data['Normalized_latency'] = merged_data['Observed_latency'] / merged_data['Required_latency']

# Identify the UPF instance with the maximum CPU share for each PDU session at each time instance
merged_data['Max_CPU_UPF'] = merged_data.groupby('PDU_session')['CPU_share'].transform(lambda x: merged_data.loc[x.idxmax(), 'UPF_instance'])

# Plot the normalized latency over time for each PDU session with legend including the UPF instance
plt.figure(figsize=(20, 10))
for pdu_session, session_data in merged_data.groupby('PDU_session'):
    plt.scatter(session_data['Time_instance'], session_data['Normalized_latency'], label=f'PDU {pdu_session} (UPF: {session_data["Max_CPU_UPF"].iloc[0]})')

plt.axhline(y=1, color='r', linestyle='--', label='Normalized Latency Bound')
plt.xlabel('Time Instance', fontsize=20)
plt.ylabel('Normalized Latency', fontsize=20)
plt.title('Normalized Latency Over Time for Each PDU Session', fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=10, loc='lower left')
plt.grid(True)

# Save plot
plt.savefig('Normalized_Latency_PDU_Time.png', format='png')

