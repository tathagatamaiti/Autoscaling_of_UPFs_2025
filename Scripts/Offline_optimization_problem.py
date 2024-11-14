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

# Create a sparse time set for each PDU session based on its active time range
pdu_time_map = {j: list(range(int(pdu_data.loc[pdu_data['id'] == j, 'start'].values[0]), int(pdu_data.loc[pdu_data['id'] == j, 'end'].values[0]) + 1)) for j in model.A}

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

# Define s only over relevant indices
# Create an index set for s where (j, i, n) are valid combinations
s_index = []
for j in model.A:
    for i in model.U:
        for n in pdu_time_map[j]:
            s_index.append((j, i, n))
model.s_index = Set(initialize=s_index, dimen=3)
model.s = Var(model.s_index, within=NonNegativeReals)  # CPU share allocated

# Auxiliary variable h[i, n], only for time points where any PDU is active
active_time_points = sorted(set(n for times in pdu_time_map.values() for n in times))
model.active_T = Set(initialize=active_time_points, ordered=True)
model.h = Var(model.U, model.active_T, within=NonNegativeIntegers)  # Number of PDU sessions anchored to UPF at time n

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

# Constraint (4): zj >= sum(y[j, i] for i in model.U)
def reverse_admittance_constraint(model, j):
    return model.z[j] >= sum(model.y[j, i] for i in model.U)
model.reverse_admittance_constraint = Constraint(model.A, rule=reverse_admittance_constraint)

# Constraint (5): sum(y[j, i] for i in model.U) <= 1
def single_assignment_constraint(model, j):
    return sum(model.y[j, i] for i in model.U) <= 1
model.single_assignment_constraint = Constraint(model.A, rule=single_assignment_constraint)

# Helper function to check if an index is valid for model.x
def is_valid_index(var, *index):
    try:
        var[index]
        return True
    except KeyError:
        return False


# Constraint (6): y[j, i] <= M1 * x[i, n] only for sparse times
M1 = 1e9  # Large constant
model.deployment_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in pdu_time_map[j]:
            # Use helper function to check if (i, n) is a valid index for model.x
            if is_valid_index(model.x, i, n):
                model.deployment_constraint.add(model.y[j, i] <= M1 * model.x[i, n])

# Constraint (7): x[i, n] >= y[j, i] only for sparse times
model.continuity_constraint = ConstraintList()
for j in model.A:
    for i in model.U:
        for n in pdu_time_map[j]:
            # Use helper function to check if (i, n) is a valid index for model.x
            if is_valid_index(model.x, i, n):
                model.continuity_constraint.add(model.x[i, n] >= model.y[j, i])

# Constraint (8): x[i, n] <= sum(y[j, i] for j where n in pdu_time_map[j])
def scale_in_constraint(model, i, n):
    if is_valid_index(model.x, i, n):
        return model.x[i, n] <= sum(model.y[j, i] for j in model.A if n in pdu_time_map[j])
    else:
        return Constraint.Skip  # Skip if (i, n) is not a valid index
model.scale_in_constraint = Constraint(model.U, model.T, rule=scale_in_constraint)

# Constraint (9): sum(s[j, i, n] for j where (j, i, n) in s_index) == C[i] * x[i, n]
def cpu_capacity_constraint(model, i, n):
    relevant_j = [j for j in model.A if (j, i, n) in model.s_index]
    if is_valid_index(model.x, i, n):
        return sum(model.s[j, i, n] for j in relevant_j) == model.C[i] * model.x[i, n]
    else:
        return Constraint.Skip
model.cpu_allocation_constraint = Constraint(model.U, model.T, rule=cpu_capacity_constraint)


# Constraint (11): s[j, i, n] <= C[i]
model.equal_cpu_share_constraint = ConstraintList()
for j, i, n in model.s_index:
    model.equal_cpu_share_constraint.add(model.s[j, i, n] <= model.C[i])

# Constraint (12): h[i, n] == sum(y[j, i] for j where n in pdu_time_map[j])
def cpu_link_constraint(model, i, n):
    return model.h[i, n] == sum(model.y[j, i] for j in model.A if n in pdu_time_map[j])
model.cpu_link_constraint = Constraint(model.U, model.active_T, rule=cpu_link_constraint)

# Constraint (13): w[i] * r[j] * y[j, i] <= s[j, i, n] * l[j]
model.latency_constraint = ConstraintList()
for j, i, n in model.s_index:
    model.latency_constraint.add(
        model.w[i] * model.r[j] * model.y[j, i] <= model.s[j, i, n] * model.l[j]
    )

# Constraint (14): s[j, i, n] <= M2 * y[j, i]
M2 = 1e9  # Another large constant
model.upper_bound_cpu_constraint = ConstraintList()
for j, i, n in model.s_index:
    model.upper_bound_cpu_constraint.add(model.s[j, i, n] <= M2 * model.y[j, i])

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
# Retrieve variable values and prepare output data
output = []
for j in model.A:
    for i in model.U:
        y_value = model.y[j, i].value if model.y[j, i].value is not None else 0
        for n in pdu_time_map[j]:
            # Check if (i, n) is a valid index for model.x before accessing it
            x_value = model.x[i, n].value if is_valid_index(model.x, i, n) and model.x[i, n].value is not None else 0
            z_value = model.z[j].value if model.z[j].value is not None else 0
            key = (j, i, n)
            # Check if (j, i, n) is a valid index for model.s before accessing it
            s_value = model.s[key].value if key in model.s_index and model.s[key].value is not None else 0
            if s_value > 0 and y_value > 0:
                observed_latency = (model.w[i] * model.r[j] * y_value) / s_value
            else:
                observed_latency = None
            output.append({
                'PDU_session': j,
                'UPF_instance': i,
                'Time_instance': n,
                'Admission_status': z_value,
                'UPF_active': x_value,
                'Anchoring': y_value,
                'CPU_share': s_value,
                'Observed_latency': observed_latency
            })

# Output results to a CSV file
results_df = pd.DataFrame(output)
results_df.to_csv('offline_solution.csv', index=False)

