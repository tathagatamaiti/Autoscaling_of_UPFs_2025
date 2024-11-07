import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Load datasets with pandas
pdu_data = pd.read_csv('pdu_sessions.csv')  # Columns: id, start, end, latency, rate
upf_data = pd.read_csv('upf_instances.csv')  # Columns: instance id, workload factor, CPU capacity

# Create model
model = ConcreteModel()

# Define Sets
time_points = sorted(set(pdu_data['start']).union(set(pdu_data['end'])))
model.T = Set(initialize=time_points, ordered=True)  # Set of time instances
model.A = Set(initialize=pdu_data['id'].tolist())  # Set of PDU sessions
model.U = Set(initialize=upf_data['instance_id'].tolist())  # Set of all UPF instances

# Define Parameters
pdu_dict = pdu_data.set_index('id').to_dict()  # Convert PDU data to dictionary
upf_dict = upf_data.set_index('instance_id').to_dict()  # Convert UPF data to dictionary

model.τ = Param(model.A, initialize=pdu_dict['start'])  # Start time of PDU session
model.ϵ = Param(model.A, initialize={j: pdu_dict['end'][j] - pdu_dict['start'][j] for j in model.A})  # Activity time of PDU
model.l = Param(model.A, initialize=pdu_dict['latency'])  # Latency requirement of PDU session
model.r = Param(model.A, initialize=pdu_dict['rate'])  # Data rate of PDU session
model.C = Param(model.U, initialize=upf_dict['cpu_capacity'])  # CPU capacity of UPF instance
model.w = Param(model.U, initialize=upf_dict['workload_factor'])  # Workload factor of UPF instance

# Define Variables
model.z = Var(model.A, within=Binary)  # Binary indicating if a PDU session is admitted
model.x = Var(model.U, model.T, within=Binary)  # Binary for UPF instance active status at time n
model.y = Var(model.A, model.U, within=Binary)  # Binary indicating if PDU session j is anchored to UPF instance i
model.s = Var(model.A, model.U, model.T, within=NonNegativeReals)  # CPU share allocated

# Auxiliary variables
model.h = Var(model.U, model.T, within=NonNegativeIntegers)  # Number of PDU sessions anchored to UPF at time n

# Objective function
gamma = 0.00001  # Penalty factor
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
def reverse_admittance_constraint(model, j):
    return model.z[j] >= sum(model.y[j, i] for i in model.U)
model.reverse_admittance_constraint = Constraint(model.A, rule=reverse_admittance_constraint)

# Constraint (5): sum(y[j, i] for i in model.U) <= 1
def single_assignment_constraint(model, j):
    return sum(model.y[j, i] for i in model.U) <= 1
model.single_assignment_constraint = Constraint(model.A, rule=single_assignment_constraint)

# Constraint (6): y[j, i] <= M1 * x[i, n]
M1 = 1e9  # Large constant
model.deployment_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in model.T:
            if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                model.deployment_constraint.add(model.y[j, i] <= M1 * model.x[i, n])

# Constraint (7): x[i, n] >= y[j, i]
model.continuity_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in model.T:
            if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                model.continuity_constraint.add(model.x[i, n] >= model.y[j, i])

# Constraint (8): x[i, n] <= sum(y[j, i] for j in model.A)
def scale_in_constraint(model, i, n):
    return model.x[i, n] <= sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j])
model.scale_in_constraint = Constraint(model.U, model.T, rule=scale_in_constraint)

# Constraint (9): sum(s[j, i, n] for j in A if start_time[j] <= n <= end_time[j]) == cpu_capacity[i]*x[i,n]
def cpu_capacity_constraint(model, i, n):
    return sum(model.s[j, i, n] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]) == model.C[i]*model.x[i,n]
model.cpu_allocation_constraint = Constraint(model.U, model.T, rule=cpu_capacity_constraint)

# Constraint (11): s[j, i, n] <= (cpu_capacity[i] for j in A)
model.equal_cpu_share_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in model.T:
            if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                model.equal_cpu_share_constraint.add(model.s[j, i, n] <= model.C[i])

# Constraint (12): Link the number of active sessions to the anchored sessions
def cpu_link_constraint(model, i, n):
    return model.h[i, n] == sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j])
model.cpu_link_constraint = Constraint(model.U, model.T, rule=cpu_link_constraint)

# Constraint (13): workload_factor[i] * rate[j] * y[j, i] <= s[j, i, n] * latency[j]
model.latency_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in model.T:
            if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                model.latency_constraint.add(
                    model.w[i] * model.r[j] * model.y[j, i] <= model.s[j, i, n] * model.l[j]
                )

# Constraint (14): s[j, i, n] <= M2 * y[j, i]
M2 = 1e9  # Another large constant
model.upper_bound_cpu_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in model.T:
            if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                model.upper_bound_cpu_constraint.add(model.s[j, i, n] <= M2 * model.y[j, i])

# # Define sin as a variable
# model.sin = Var(model.U, model.T, within=NonNegativeReals)
#
# # Define active_indicator as a binary variable
# model.active_indicator = Var(model.U, model.T, within=Binary)
#
# # Initialize sin_constraint as a ConstraintList
# model.sin_constraint = ConstraintList()
#
# # Link active_indicator to the presence of active sessions
# for i in model.U:
#     for n in model.T:
#         # Ensure active_indicator is 1 if there are any active sessions anchored to UPF i at time n
#         model.sin_constraint.add(
#             sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]) <= model.active_indicator[i, n] * len(model.A)
#         )
#         # Ensure active_indicator is 0 if there are no active sessions
#         model.sin_constraint.add(
#             sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]) >= model.active_indicator[i, n]
#         )
#
# # Define sin based on active_indicator
# for i in model.U:
#     for n in model.T:
#         active_sessions = sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j])
#
#         # When active_indicator is 1, sin = C_i / active_sessions
#         model.sin_constraint.add(model.sin[i, n] <= model.C[i] / (active_sessions + (1 - model.active_indicator[i, n])))
#
#         # When active_indicator is 0, sin should be 0
#         model.sin_constraint.add(model.sin[i, n] <= model.C[i] * model.active_indicator[i, n])
#
# # Constraint: sjin <= sin * yji
# model.lower_cpu_share_constraint = ConstraintList()
# for j in model.A:
#     for i in model.U:
#         for n in model.T:
#             if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
#                 model.lower_cpu_share_constraint.add(model.s[j, i, n] <= model.sin[i, n] * model.y[j, i])
#
# # Constraint: sjin >= sin * yji
# model.upper_cpu_share_constraint = ConstraintList()
# for j in model.A:
#     for i in model.U:
#         for n in model.T:
#             if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
#                 model.upper_cpu_share_constraint.add(model.s[j, i, n] >= model.sin[i, n] * model.y[j, i])

# Solve the model
solver = SolverFactory('scip', executable='/home/tmaiti/Downloads/SCIPOptSuite-9.1.1-Linux/bin/scip')
results = solver.solve(model, tee=True)

# Objective value
print(f"Objective value: {model.obj.expr()}")

# Solver Status and Termination Condition
print("Solver Status:", results.solver.status)
print("Solution Status:", results.solver.termination_condition)

# Retrieve variable values
print("Variable Values:")
for v in model.component_objects(Var, active=True):
    varobject = getattr(model, str(v))
    print(f"Variable {v} values:")
    for index in varobject:
        print(f"   {index} : {varobject[index].value}")

# Save results to CSV
output = []
for j in model.A:
    for i in model.U:
        for n in model.T:
            if model.s[j, i, n].value and model.y[j, i].value:
                observed_latency = (model.w[i] * model.r[j] * model.y[j, i].value) / model.s[j, i, n].value
            else:
                observed_latency = None
            output.append({
                'PDU_session': j,
                'UPF_instance': i,
                'Time_instance': n,
                'Admission_status': model.z[j].value,
                'UPF_active': model.x[i, n].value,
                'Anchoring': model.y[j, i].value,
                'CPU_share': model.s[j, i, n].value,
                'Observed_latency': observed_latency
            })

# Output results to a CSV file
results_df = pd.DataFrame(output)
results_df.to_csv('offline_solution.csv', index=False)
