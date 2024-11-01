import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Load datasets with pandas
pdu_data = pd.read_csv('pdu_sessions.csv') # Columns: id, start, end, latency, rate
upf_data = pd.read_csv('upf_instances.csv') # Columns: instance id, workload factor, CPU capacity

# Define dictionaries
pdu_dict = pdu_data.set_index('id').to_dict()  # Convert PDU data to dictionary
upf_dict = upf_data.set_index('instance_id').to_dict()  # Convert UPF data to dictionary

# Initialize lists to store results of each PDU arrival
results = []

# Loop through each PDU session as they arrive
for index, pdu in pdu_data.iterrows():
    # Identify the new PDU session `q`
    q = pdu['id']
    tau_q = pdu['start']  # Arrival time of PDU `q`

    # Set Aq: All sessions active at the time of `q`'s arrival
    Aq = list(
        pdu_data[(pdu_data['start'] <= tau_q) & (pdu_data['end'] >= tau_q)]['id']
    ) + [q]

    # Set Uq: UPF instances that are active at `q`'s arrival time
    active_upfs = set()
    for result in results:
        if result['Time'] <= tau_q and result['UPF_active'] == 1:
            active_upfs.add(result['UPF_instance'])
    Uq = list(active_upfs)

    # Create model
    model = ConcreteModel()

    # Define Sets
    model.Aq = Set(initialize=Aq, ordered=True)  # Set of active PDU sessions at q's arrival
    model.U = Set(initialize=upf_data['instance_id'].tolist())  # Set of UPF instances
    model.Uq = Set(initialize=Uq, ordered=True)  # Active UPFs as an ordered set
    model.U_inactive = model.U - model.Uq  # Set of inactive UPF instances

    # Define Parameters
    model.τ = Param(model.Aq, initialize={j: pdu_dict['start'][j] for j in model.Aq})  # Start time of PDU session
    model.ϵ = Param(model.Aq,
                    initialize={j: pdu_dict['end'][j] - pdu_dict['start'][j] for j in model.Aq})  # Activity time
    model.l = Param(model.Aq, initialize={j: pdu_dict['latency'][j] for j in model.Aq})  # Latency requirement
    model.r = Param(model.Aq, initialize={j: pdu_dict['rate'][j] for j in model.Aq})  # Data rate
    model.C = Param(model.U, initialize=upf_dict['cpu_capacity'])  # CPU capacity of UPF instance
    model.w = Param(model.U, initialize=upf_dict['workload_factor'])  # Workload factor of UPF instance

    # Define Variables
    model.x = Var(model.U, within=Binary)  # Binary indicating if a UPF instance is active
    model.y = Var(model.Aq, model.U, within=Binary)  # Binary indicating if PDU session j is anchored to UPF instance i
    model.s = Var(model.Aq, model.U, within=NonNegativeReals, bounds=(0, None))  # CPU share allocated
    model.s_total = Var(model.U, within=NonNegativeReals, bounds=(0, None))  # Total CPU share per UPF instance

    # Objective function
    model.obj = Objective(expr=sum(model.x[i] for i in model.U), sense=minimize)

    # Constraints
    # Constraint (18): Only one new UPF instance can be activated from U_complement_q
    def single_new_upf_constraint(model):
        return sum(model.x[i] for i in model.U_inactive) <= 1
    model.single_new_upf_constraint = Constraint(rule=single_new_upf_constraint)

    # Constraint (19): Ensure if i is not deployed, q cannot be anchored to it
    M1 = 1e9 # Large constant
    def anchor_constraint(model, j, i):
        return model.y[j, i] <= M1 * model.x[i]
    model.anchor_constraint = Constraint(model.Aq, model.U, rule=anchor_constraint)

    # Constraint (20): If q is anchored to an inactive UPF, activate that UPF instance
    def upf_activation_constraint(model, i):
        return model.x[i] >= model.y[q, i]
    model.upf_activation_constraint = Constraint(model.U, rule=upf_activation_constraint)

    # Constraint (21): Total CPU share at i must be allocated to anchored PDU sessions
    def cpu_allocation_constraint(model, i):
        return sum(model.s[j, i] for j in model.Aq) <= model.C[i]
    model.cpu_allocation_constraint = Constraint(model.Uq, rule=cpu_allocation_constraint)

    # Constraint (22): CPU share allocation for PDU session anchored to UPF instance
    def cpu_share_upper_bound_constraint(model, j, i):
        return model.s[j, i] <= model.C[i] * model.y[j, i]
    model.cpu_share_upper_bound_constraint = Constraint(model.Aq, model.U, rule=cpu_share_upper_bound_constraint)

    # Constraint (23): CPU share allocation lower bound when PDU is anchored
    def cpu_share_lower_bound_constraint(model, j, i):
        return model.s[j, i] >= model.C[i] * model.y[j, i]
    model.cpu_share_lower_bound_constraint = Constraint(model.Aq, model.U, rule=cpu_share_lower_bound_constraint)

    # Constraint (24): Link total CPU share to number of anchored PDU sessions
    def total_cpu_share_constraint(model, i):
        return model.s_total[i] <= model.C[i]
    model.total_cpu_share_constraint = Constraint(model.U, rule=total_cpu_share_constraint)

    # Constraint (25): Ensure latency is within limits for anchored PDU sessions
    def latency_constraint(model, j, i):
        return model.w[i] * model.r[j] * model.y[j, i] <= model.s[j, i] * model.l[j]
    model.latency_constraint = Constraint(model.Aq, model.U, rule=latency_constraint)

    # Constraint (26): Upper bound on CPU share when PDU is not anchored
    M2 = 1e9
    def upper_bound_cpu_constraint(model, j, i):
        return model.s_total[i] <= M2 * (model.y[j, i])
    model.upper_bound_cpu_constraint = Constraint(model.Aq, model.U, rule=upper_bound_cpu_constraint)

    # Solve the model
    solver = SolverFactory('glpk')
    solver.solve(model, tee=True)

    # Save results for each UPF instance and time instance
    for i in model.U:
        results.append({
            'PDU_session': q,
            'UPF_instance': i,
            'Time': tau_q,
            'UPF_active': model.x[i].value,
            'Total_CPU_Share': model.s_total[i].value if model.s_total[i].value else 0
        })

# Save all results to a CSV file
df_output = pd.DataFrame(results)
df_output.to_csv('online_solution.csv', index=False)
