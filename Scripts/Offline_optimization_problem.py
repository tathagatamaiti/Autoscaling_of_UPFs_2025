import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory

# Load datasets
pdu_data = pd.read_csv('pdu_sessions.csv')  # Columns: id, start, end, latency, rate
upf_data = pd.read_csv('upf_instances.csv')  # Columns: instance_id, workload_factor, CPU_capacity

# Sort PDU data by start time
pdu_data = pdu_data.sort_values(by="start")

# Model parameters
time_step = 10  # Time step for intervals
time_horizon = max(pdu_data["end"])

# Prepare results storage
results_list = []

# Track assignments across intervals
pdu_to_upf_assignment = {}

# Loop through time intervals
for t_start in range(0, time_horizon + 1, time_step):
    t_end = t_start + time_step

    # Filter PDU sessions active in the current time window
    active_pdu = pdu_data[(pdu_data["start"] < t_end) & (pdu_data["end"] >= t_start)]
    active_sessions = active_pdu["id"].tolist()

    if not active_sessions:
        continue  # Skip intervals with no active sessions

    # Create model
    model = ConcreteModel()

    # Define Sets
    model.A = Set(initialize=active_sessions)  # Active PDU sessions
    model.U = Set(initialize=upf_data['instance_id'].tolist())  # UPF instances
    model.T = Set(initialize=list(range(t_start, t_end)))  # Time points in the current interval

    # Define Parameters
    pdu_dict = active_pdu.set_index('id').to_dict()
    upf_dict = upf_data.set_index('instance_id').to_dict()

    model.τ = Param(model.A, initialize=pdu_dict['start'])  # Start time of PDU session
    model.ϵ = Param(model.A, initialize={j: pdu_dict['end'][j] - pdu_dict['start'][j] for j in model.A})  # Duration
    model.l = Param(model.A, initialize=pdu_dict['latency'])  # Latency requirement
    model.r = Param(model.A, initialize=pdu_dict['rate'])  # Data rate
    model.C = Param(model.U, initialize=upf_dict['cpu_capacity'])  # CPU capacity
    model.w = Param(model.U, initialize=upf_dict['workload_factor'])  # Workload factor

    # Define Variables
    model.z = Var(model.A, within=Binary)  # Admission decision
    model.x = Var(model.U, model.T, within=Binary)  # UPF active status
    model.y = Var(model.A, model.U, within=Binary)  # PDU anchoring decision
    model.s = Var(model.A, model.U, model.T, within=NonNegativeReals)  # CPU share allocated

    # Auxiliary Variables
    model.h = Var(model.U, model.T, within=NonNegativeIntegers)  # Number of PDU sessions anchored to UPF at time n

    # Objective function
    gamma = 0.00001  # Penalty factor
    model.obj = Objective(
        expr=sum(model.z[j] for j in model.A) - gamma * sum(model.x[i, n] for i in model.U for n in model.T),
        sense=maximize
    )

    # Constraints
    def admittance_constraint(model, j):
        return model.z[j] <= sum(model.y[j, i] for i in model.U)
    model.admittance_constraint = Constraint(model.A, rule=admittance_constraint)

    def reverse_admittance_constraint(model, j):
        return model.z[j] >= sum(model.y[j, i] for i in model.U)
    model.reverse_admittance_constraint = Constraint(model.A, rule=reverse_admittance_constraint)

    def single_assignment_constraint(model, j):
        return sum(model.y[j, i] for i in model.U) <= 1
    model.single_assignment_constraint = Constraint(model.A, rule=single_assignment_constraint)

    M1 = 1e9  # Large constant
    model.deployment_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                    model.deployment_constraint.add(model.y[j, i] <= M1 * model.x[i, n])

    model.continuity_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                    model.continuity_constraint.add(model.x[i, n] >= model.y[j, i])

    def scale_in_constraint(model, i, n):
        return model.x[i, n] <= sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j])
    model.scale_in_constraint = Constraint(model.U, model.T, rule=scale_in_constraint)

    def cpu_capacity_constraint(model, i, n):
        return sum(model.s[j, i, n] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]) == model.C[i] * model.x[i, n]
    model.cpu_allocation_constraint = Constraint(model.U, model.T, rule=cpu_capacity_constraint)

    model.equal_cpu_share_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                    model.equal_cpu_share_constraint.add(model.s[j, i, n] <= model.C[i])

    def cpu_link_constraint(model, i, n):
        return model.h[i, n] == sum(model.y[j, i] for j in model.A if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j])
    model.cpu_link_constraint = Constraint(model.U, model.T, rule=cpu_link_constraint)

    model.latency_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                    model.latency_constraint.add(
                        model.w[i] * model.r[j] * model.y[j, i] <= model.s[j, i, n] * model.l[j]
                    )

    M2 = 1e9  # Another large constant
    model.upper_bound_cpu_constraint = ConstraintList()
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if n >= model.τ[j] and n <= model.τ[j] + model.ϵ[j]:
                    model.upper_bound_cpu_constraint.add(model.s[j, i, n] <= M2 * model.y[j, i])

    # Enforce consistent anchoring across intervals
    for j in active_sessions:
        if j in pdu_to_upf_assignment:
            assigned_upf = pdu_to_upf_assignment[j]
            for i in model.U:
                if i != assigned_upf:
                    model.y[j, i].fix(0)  # Prevent assignment to other UPFs
                else:
                    model.y[j, i].fix(1)  # Maintain assignment

    # Solve model
    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=False)

    # Store assignments for consistency in future intervals
    for j in model.A:
        for i in model.U:
            if model.y[j, i].value > 0.9:
                pdu_to_upf_assignment[j] = i

    # Store results
    for j in model.A:
        for i in model.U:
            for n in model.T:
                if model.s[j, i, n].value and model.y[j, i].value:
                    observed_latency = (model.w[i] * model.r[j] * model.y[j, i].value) / model.s[j, i, n].value
                else:
                    observed_latency = None
                output = {
                    "Interval_Start": t_start,
                    "Interval_End": t_end,
                    "PDU_Session": j,
                    "UPF_Instance": i,
                    "Time": n,
                    "Admission_status": model.z[j].value,
                    "UPF_active": model.x[i, n].value,
                    "Anchoring": model.y[j, i].value,
                    "CPU_share": model.s[j, i, n].value,
                    "Observed_latency": observed_latency
                }
                results_list.append(output)

# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv('offline_solution_intervals_consistent.csv', index=False)
