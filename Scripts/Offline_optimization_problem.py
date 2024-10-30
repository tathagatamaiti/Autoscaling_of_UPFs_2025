import pandas as pd
from pyomo.environ import *
from pyomo.environ import Binary, NonNegativeReals
from pyomo.opt import SolverFactory

# Load datasets with pandas
pdu_data = pd.read_csv("pdu_data.csv")  # Columns: id, start_time, end_time, rate, latency, instance_id
upf_data = pd.read_csv("upf_data.csv")  # Columns: id, workload_factor, cpu_capacity, memory, network_connectivity

# Create sets and dictionaries from datasets
A = pdu_data['id'].tolist()  # Set of PDU sessions
U = upf_data['id'].tolist()  # Set of UPF instances

# Parameters for each PDU and UPF session based on input data
start_time = dict(zip(pdu_data['id'], pdu_data['start_time']))
end_time = dict(zip(pdu_data['id'], pdu_data['end_time']))
rate = dict(zip(pdu_data['id'], pdu_data['rate']))
latency = dict(zip(pdu_data['id'], pdu_data['latency']))
cpu_capacity = dict(zip(upf_data['id'], upf_data['cpu_capacity']))
workload_factor = dict(zip(upf_data['id'], upf_data['workload_factor']))

# Generate the time set T based on unique start and end times
T = sorted(set(pdu_data['start_time']).union(set(pdu_data['end_time'])))

# Create model
model = ConcreteModel()

# Define Sets
model.A = Set(initialize=A)
model.U = Set(initialize=U)
model.T = Set(initialize=T)

# Define Parameters
model.start_time = Param(model.A, initialize=start_time)
model.end_time = Param(model.A, initialize=end_time)
model.rate = Param(model.A, initialize=rate)
model.latency = Param(model.A, initialize=latency)
model.cpu_capacity = Param(model.U, initialize=cpu_capacity)
model.workload_factor = Param(model.U, initialize=workload_factor)

# Define Variables
model.z = Var(model.A, within=Binary)  # Admittance of PDU session j
model.x = Var(model.U, model.T, within=Binary)  # Activity of UPF instance i at time n
model.y = Var(model.A, model.U, within=Binary)  # Assignment of PDU j to UPF i
model.s = Var(model.A, model.U, model.T, within=NonNegativeReals)  # CPU share of i assigned to j at time n

# Objective function
gamma = 0.0001  # Penalty factor
model.obj = Objective(
    expr=sum(model.z[j] for j in model.A) - gamma * sum(model.x[i, n] for i in model.U for n in model.T),
    sense=maximize
)

# Constraints
# Constraint (3): zj <= sum(y[j, i] for i in model.U)
def admittance_constraint(model, j):
    return model.z[j] <= sum(model.y[j, i] for i in model.U)
model.admittance_constraint = Constraint(model.A, rule=admittance_constraint)

# Constraint (4): zj >= y[j, i]
def reverse_admittance_constraint(model, j, i):
    return model.z[j] >= model.y[j, i]
model.reverse_admittance_constraint = Constraint(model.A, model.U, rule=reverse_admittance_constraint)

# Constraint (5): sum(y[j, i] for i in model.U) <= 1
def single_assignment_constraint(model, j):
    return sum(model.y[j, i] for i in model.U) <= 1
model.single_assignment_constraint = Constraint(model.A, rule=single_assignment_constraint)

# Constraint (6): y[j, i] <= M1 * x[i, n]
M1 = 1e9  # Large constant
def deployment_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.y[j, i] <= M1 * model.x[i, n]
    else:
        return Constraint.Skip
model.deployment_constraint = Constraint(model.A, model.U, model.T, rule=deployment_constraint)

# Constraint (7): x[i, n] >= y[j, i]
def continuity_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.x[i, n] >= model.y[j, i]
    else:
        return Constraint.Skip
model.continuity_constraint = Constraint(model.A, model.U, model.T, rule=continuity_constraint)

# Constraint (8): x[i, n] <= sum(y[j, i] for j in model.A)
def scale_in_constraint(model, i, n):
    return model.x[i, n] <= sum(model.y[j, i] for j in model.A if model.start_time[j] <= n <= model.end_time[j])
model.scale_in_constraint = Constraint(model.U, model.T, rule=scale_in_constraint)

# Constraint (9): sum(s[j, i, n] for j in A if start_time[j] <= n <= end_time[j]) == cpu_capacity[i]
def cpu_capacity_constraint(model, i, n):
    active_pdus = [j for j in model.A if model.start_time[j] <= n <= model.end_time[j]]
    if active_pdus:
        return sum(model.s[j, i, n] for j in active_pdus) == model.cpu_capacity[i]
    return Constraint.Skip
model.cpu_capacity_constraint = Constraint(model.U, model.T, rule=cpu_capacity_constraint)

# Constraint (10): s[j, i, n] <= (cpu_capacity[i] / sum(y[j, i] for j in A)) * y[j, i]
def equal_cpu_share_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.s[j, i, n] <= (model.cpu_capacity[i] / sum(model.y[k, i] for k in model.A)) * model.y[j, i]
    return Constraint.Skip
model.equal_cpu_share_constraint = Constraint(model.A, model.U, model.T, rule=equal_cpu_share_constraint)

# Constraint (11): s[j, i, n] >= (cpu_capacity[i] / sum(y[j, i] for j in A)) * y[j, i]
def equal_cpu_share_lower_bound_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.s[j, i, n] >= (model.cpu_capacity[i] / sum(model.y[k, i] for k in model.A)) * model.y[j, i]
    return Constraint.Skip
model.equal_cpu_share_lower_bound_constraint = Constraint(model.A, model.U, model.T, rule=equal_cpu_share_lower_bound_constraint)

# Constraint (12): sum(s[j, i, n] for j in A) == cpu_capacity[i]
def cpu_share_sum_constraint(model, i, n):
    active_pdus = [j for j in model.A if model.start_time[j] <= n <= model.end_time[j]]
    if active_pdus:
        return sum(model.s[j, i, n] for j in active_pdus) == model.cpu_capacity[i]
    return Constraint.Skip
model.cpu_share_sum_constraint = Constraint(model.U, model.T, rule=cpu_share_sum_constraint)

# Constraint (13): workload_factor[i] * rate[j] * y[j, i] <= s[j, i, n] * latency[j]
def latency_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.workload_factor[i] * model.rate[j] * model.y[j, i] <= model.s[j, i, n] * model.latency[j]
    return Constraint.Skip
model.latency_constraint = Constraint(model.A, model.U, model.T, rule=latency_constraint)

# Constraint (14): s[j, i, n] <= M2 * y[j, i]
M2 = 1e9  # Large constant
def upper_bound_cpu_constraint(model, j, i, n):
    if model.start_time[j] <= n <= model.end_time[j]:
        return model.s[j, i, n] <= M2 * model.y[j, i]
    return Constraint.Skip
model.upper_bound_cpu_constraint = Constraint(model.A, model.U, model.T, rule=upper_bound_cpu_constraint)

# Solving the model
solver = SolverFactory("glpk")
results = []
for n in model.T:
    # Update constraints and solve the model at each time instance n
    result = solver.solve(model)
    # Collect results
    results.append({
        "time": n,
        "PDU_sessions_admitted": sum(model.z[j]() for j in model.A),
        "UPFs_active": sum(model.x[i, n]() for i in model.U),
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("upf_scaling_results.csv", index=False)
