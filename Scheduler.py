import pandas as pd
import numpy as np

from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    lpSum,
    LpInteger,
    LpStatus,
    value
)

# -------------------------------------------------------
# LOAD INPUT FILES
# -------------------------------------------------------

forecast_df = pd.read_csv("data/client_forecast.csv")
rules_df = pd.read_csv("data/wfm_rules.csv")

# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------

forecast_df['date'] = pd.to_datetime(
    forecast_df['date']
)

# -------------------------------------------------------
# EXTRACT WFM RULES
# -------------------------------------------------------

rules = dict(
    zip(
        rules_df['parameter'],
        rules_df['value']
    )
)

SHRINKAGE = float(rules['shrinkage'])
OCCUPANCY = float(rules['occupancy'])
MAX_SHIFT_HOURS = int(rules['max_shift_hours'])
MIN_SHIFT_HOURS = int(rules['min_shift_hours'])

# -------------------------------------------------------
# SHIFT MASTER
# -------------------------------------------------------

SHIFT_OPTIONS = {
    "S1": {
        "start": "09:00",
        "end": "18:00",
        "coverage_hours": [9,10,11,12,13,14,15,16,17]
    },
    "S2": {
        "start": "10:00",
        "end": "19:00",
        "coverage_hours": [10,11,12,13,14,15,16,17,18]
    },
    "S3": {
        "start": "11:00",
        "end": "20:00",
        "coverage_hours": [11,12,13,14,15,16,17,18,19]
    },
    "S4": {
        "start": "12:00",
        "end": "21:00",
        "coverage_hours": [12,13,14,15,16,17,18,19,20]
    }
}

# -------------------------------------------------------
# GENERATE REQUIRED HEADCOUNT
# -------------------------------------------------------

AHT = 300

forecast_df['required_headcount'] = np.ceil(
    (
        forecast_df['forecasted_volume'] * AHT
    )
    /
    (
        3600 * OCCUPANCY * (1 - SHRINKAGE)
    )
).astype(int)

# Extract hour from interval
forecast_df['hour'] = (
    forecast_df['interval']
    .str
    .split(':')
    .str[0]
    .astype(int)
)

# -------------------------------------------------------
# CREATE OPTIMIZATION MODEL
# -------------------------------------------------------

model = LpProblem(
    "FLOW_Scheduler_Optimization",
    LpMinimize
)

# -------------------------------------------------------
# DECISION VARIABLES
# Number of employees assigned to each shift
# -------------------------------------------------------

shift_variables = {
    shift: LpVariable(
        shift,
        lowBound=0,
        cat=LpInteger
    )
    for shift in SHIFT_OPTIONS.keys()
}

# -------------------------------------------------------
# OBJECTIVE FUNCTION
# Minimize total workforce allocation
# -------------------------------------------------------

model += lpSum(
    shift_variables[shift]
    for shift in SHIFT_OPTIONS.keys()
)

# -------------------------------------------------------
# COVERAGE CONSTRAINTS
# Ensure every interval meets required HC
# -------------------------------------------------------

for _, row in forecast_df.iterrows():

    interval_hour = row['hour']
    required_hc = row['required_headcount']

    covering_shifts = []

    for shift_name, shift_info in SHIFT_OPTIONS.items():

        if interval_hour in shift_info['coverage_hours']:
            covering_shifts.append(
                shift_variables[shift_name]
            )

    model += (
        lpSum(covering_shifts)
        >=
        required_hc,
        f"Coverage_{interval_hour}"
    )

# -------------------------------------------------------
# BUSINESS RULE CONSTRAINTS
# -------------------------------------------------------

# Minimum staffing in each shift
model += shift_variables['S1'] >= 2
model += shift_variables['S2'] >= 2

# Balance late shifts
model += (
    shift_variables['S3']
    +
    shift_variables['S4']
    >=
    4
)

# -------------------------------------------------------
# SOLVE MODEL
# -------------------------------------------------------

model.solve()

# -------------------------------------------------------
# PRINT SOLVER STATUS
# -------------------------------------------------------

print("\n-------------------------------------")
print("OPTIMIZATION STATUS")
print("-------------------------------------")

print("Solver Status :", LpStatus[model.status])

# -------------------------------------------------------
# EXTRACT RESULTS
# -------------------------------------------------------

schedule_results = []

for shift_name, variable in shift_variables.items():

    shift_info = SHIFT_OPTIONS[shift_name]

    allocated_hc = int(variable.varValue)

    schedule_results.append({
        'shift_id': shift_name,
        'shift_start': shift_info['start'],
        'shift_end': shift_info['end'],
        'allocated_headcount': allocated_hc
    })

# -------------------------------------------------------
# CONVERT RESULTS TO DATAFRAME
# -------------------------------------------------------

schedule_df = pd.DataFrame(schedule_results)

print("\n-------------------------------------")
print("OPTIMIZED SHIFT ALLOCATION")
print("-------------------------------------")

print(schedule_df)

# -------------------------------------------------------
# VALIDATION CHECK
# -------------------------------------------------------

validation_output = []

for _, row in forecast_df.iterrows():

    interval_hour = row['hour']
    required_hc = row['required_headcount']

    allocated = 0

    for shift_name, shift_info in SHIFT_OPTIONS.items():

        if interval_hour in shift_info['coverage_hours']:

            allocated += int(
                shift_variables[shift_name].varValue
            )

    validation_output.append({
        'interval': row['interval'],
        'required_headcount': required_hc,
        'allocated_headcount': allocated,
        'status': (
            'Balanced'
            if allocated >= required_hc
            else 'Understaffed'
        )
    })

validation_df = pd.DataFrame(validation_output)

print("\n-------------------------------------")
print("SCHEDULER VALIDATION")
print("-------------------------------------")

print(validation_df.head())

# -------------------------------------------------------
# KPI SUMMARY
# -------------------------------------------------------

total_required = (
    forecast_df['required_headcount']
    .sum()
)

total_allocated = (
    schedule_df['allocated_headcount']
    .sum()
)

utilization = round(
    (
        total_required /
        total_allocated
    ) * 100,
    2
)

print("\n-------------------------------------")
print("SCHEDULER KPI SUMMARY")
print("-------------------------------------")

print("Total Required HC :", total_required)
print("Total Allocated HC:", total_allocated)
print("Utilization %     :", utilization)

# -------------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------------

schedule_df.to_csv(
    "outputs/optimized_schedule.csv",
    index=False
)

validation_df.to_csv(
    "outputs/scheduler_validation.csv",
    index=False
)

print("\nScheduler outputs saved successfully.")
