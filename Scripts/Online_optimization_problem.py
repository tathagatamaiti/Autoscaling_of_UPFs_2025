import pandas as pd
from pyomo.environ import *
from pyomo.environ import Binary, NonNegativeReals
from pyomo.opt import SolverFactory

# Load datasets with pandas
pdu_data = pd.read_csv("pdu_data.csv")  # Columns: id, start_time, end_time, rate, latency
upf_data = pd.read_csv("upf_data.csv")  # Columns: id, workload_factor, cpu_capacity

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

# Dynamically identify the latest arriving PDU session as q
next_pdu = pdu_data[pdu_data['start_time'] == pdu_data['start_time'].max()]  # Select the latest arriving session
new_pdu_id = next_pdu.iloc[0]['id']  # ID of the new PDU session
tau_q = next_pdu.iloc[0]['start_time']  # Arrival time of q

# Calculate sets Aq and Uq based on the arrival time tau_q
Aq = [j for j in A if start_time[j] <= tau_q <= end_time[j]]  # Set of active PDU sessions when q arrives
Uq = [i for i in U if any(start_time[j] <= tau_q for j in Aq)]  # UPFs active at time tau_q
U_complement_q = list(set(U) - set(Uq))  # UPF instances that are not active when q arrives

# Create the model
model = ConcreteModel()

# Define Sets
model.Aq = Set(initialize=Aq)  # Set of PDU sessions active when q arrives
model.U = Set(initialize=U)  # Set of all UPF instances
model.Uq = Set(initialize=Uq)  # Set of UPF instances already active at q arrival
model.U_complement_q = Set(initialize=U_complement_q)  # Set of inactive UPFs at q arrival

# Define Parameters
model.start_time = Param(model.Aq, initialize=start_time)
model.end_time = Param(model.Aq, initialize=end_time)
model.rate = Param(model.Aq, initialize=rate)
model.latency = Param(model.Aq, initialize=latency)
model.cpu_capacity = Param(model.U, initialize=cpu_capacity)
model.workload_factor = Param(model.U, initialize=workload_factor)

# Define Variables
model.x = Var(model.U, within=Binary)  # Activity of UPF instance i
model.y = Var(model.Aq, model.U, within=Binary)  # Assignment of PDU j to UPF i
model.s = Var(model.Aq, model.U, within=NonNegativeReals)  # CPU share of i assigned to j
model.si = Var(model.U, within=NonNegativeReals)  # Total CPU share for each UPF i

# Objective function to minimize number of new UPF instances activated
model.obj = Objective(expr=sum(model.x[i] for i in model.U), sense=minimize)

# Constraints
# Constraint (18): Only one new UPF instance can be activated from U_complement_q
def single_new_upf_constraint(model):
    return sum(model.x[i] for i in model.U_complement_q) <= 1
model.single_new_upf_constraint = Constraint(rule=single_new_upf_constraint)

# Constraint (19): Ensure if i is not deployed, q cannot be anchored to it
M1 = 1e9  # Large constant
def anchor_constraint(model, j, i):
    return model.y[j, i] <= M1 * model.x[i]
model.anchor_constraint = Constraint(model.Aq, model.U, rule=anchor_constraint)

# Constraint (20): If q is anchored to an inactive UPF, activate that UPF instance
def upf_activation_constraint(model, j, i):
    return model.x[i] >= model.y[j, i]
model.upf_activation_constraint = Constraint(model.Aq, model.U, rule=upf_activation_constraint)

# Constraint (21): Total CPU share at i must be allocated to anchored PDU sessions
def cpu_allocation_constraint(model, i):
    return sum(model.s[j, i] for j in model.Aq) == model.cpu_capacity[i]
model.cpu_allocation_constraint = Constraint(model.Uq, rule=cpu_allocation_constraint)

# Constraint (22): CPU share allocation for PDU session anchored to UPF instance
def cpu_share_upper_bound_constraint(model, j, i):
    return model.s[j, i] <= model.si[i] * model.y[j, i]
model.cpu_share_upper_bound_constraint = Constraint(model.Aq, model.U, rule=cpu_share_upper_bound_constraint)

# Constraint (23): CPU share allocation lower bound when PDU is anchored
def cpu_share_lower_bound_constraint(model, j, i):
    return model.s[j, i] >= model.si[i] * model.y[j, i]
model.cpu_share_lower_bound_constraint = Constraint(model.Aq, model.U, rule=cpu_share_lower_bound_constraint)

# Constraint (24): Link total CPU share to number of anchored PDU sessions
def total_cpu_share_constraint(model, i):
    return sum(model.si[i] for j in model.Aq) == model.cpu_capacity[i]
model.total_cpu_share_constraint = Constraint(model.U, rule=total_cpu_share_constraint)

# Constraint (25): Ensure latency is within limits for anchored PDU sessions
def latency_constraint(model, j, i):
    return model.workload_factor[i] * model.rate[j] * model.y[j, i] <= model.s[j, i] * model.latency[j]
model.latency_constraint = Constraint(model.Aq, model.U, rule=latency_constraint)

# Constraint (26): Upper bound on si when PDU is not anchored
M2 = 1e9  # Large constant
def upper_bound_cpu_constraint(model, j, i):
    return model.si[i] <= M2 * model.y[j, i]
model.upper_bound_cpu_constraint = Constraint(model.Aq, model.U, rule=upper_bound_cpu_constraint)

# Solving the model
solver = SolverFactory("glpk")
result = solver.solve(model)

# Collect results
results = {
    "active_UPFs": [i for i in model.U if model.x[i]() > 0],
    "assigned_PDUs": [(j, i) for j in model.Aq for i in model.U if model.y[j, i]() > 0],
    "cpu_shares": {(j, i): model.s[j, i]() for j in model.Aq for i in model.U if model.s[j, i]() > 0},
}

# Convert to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("upf_scaling_online_results.csv", index=False)
