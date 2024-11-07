import pandas as pd

# Load the dataset
df = pd.read_csv('offline_solution.csv')

# Filter rows where Admission_status is 1 and CPU_share > 0
filtered_df = df[(df['Admission_status'] == 1) & (df['CPU_share'] != 0)]

# Group by PDU_session and average CPU_share and Time_instance
averaged_df = filtered_df.groupby('PDU_session', as_index=False).agg({
    #'UPF_instance': 'first',
    'Time_instance': 'mean',       # Calculate the mean of Time_instance for each PDU_session
    #'Admission_status': 'first',
    #'UPF_active': 'first',
    #'Anchoring': 'first',
    'CPU_share': 'mean'            # Calculate the mean of CPU_share for each PDU_session
})

# Sort the dataset by 'PDU_session' for ordered output
ordered_df = averaged_df.sort_values(by=['PDU_session']).reset_index(drop=True)

# Save the final dataset to a new CSV file
ordered_df.to_csv('cleaned_data.csv', index=False)
