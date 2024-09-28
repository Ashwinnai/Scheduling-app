import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyworkforce.scheduling import MinAbsDifference, MinRequiredResources
import random

# Set page config
st.set_page_config(page_title="Shift Schedule Generator", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px;
        color: #fafafa;
        font-size: 14px;
        font-weight: 400;
        align-items: center;
        justify-content: center;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def generate_random_requirements(num_days, num_periods):
    return [[random.randint(1, 20) for _ in range(num_days)] for _ in range(num_periods)]

def create_shifts_coverage(shift_names, periods):
    return {shift: [0] * periods for shift in shift_names}

def solve_scheduling_problem(method, num_days, periods, shifts_coverage, required_resources, max_period_concurrency, max_shift_concurrency, costs):
    required_resources = list(map(list, zip(*required_resources)))
    
    scheduler_class = MinAbsDifference if method == "MinAbsDifference" else MinRequiredResources
    scheduler = scheduler_class(
        num_days=num_days,
        periods=periods,
        shifts_coverage=shifts_coverage,
        required_resources=required_resources,
        max_period_concurrency=max_period_concurrency,
        max_shift_concurrency=max_shift_concurrency,
        cost_dict=costs
    )
    
    return scheduler.solve()

def calculate_costs(solution, costs):
    daily_costs = {}
    shift_costs = {}
    total_cost = 0

    for item in solution['resources_shifts']:
        day = item['day']
        shift = item['shift']
        resources = item['resources']
        cost = resources * costs[shift]
        
        daily_costs[day] = daily_costs.get(day, 0) + cost
        shift_costs[shift] = shift_costs.get(shift, 0) + cost
        
        total_cost += cost

    return daily_costs, shift_costs, total_cost

def display_results(solution, required_resources, num_days, periods, shift_names, shifts_coverage, costs):
    daily_costs, shift_costs, total_cost = calculate_costs(solution, costs)

    st.metric("Total Cost", f"${total_cost:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost per Day")
        cost_per_day_df = pd.DataFrame.from_dict(daily_costs, orient='index', columns=['Cost'])
        cost_per_day_df.index.name = 'Day'
        st.dataframe(cost_per_day_df.style.highlight_max(axis=0))

    with col2:
        st.subheader("Cost per Shift")
        cost_per_shift_df = pd.DataFrame.from_dict(shift_costs, orient='index', columns=['Cost'])
        cost_per_shift_df.index.name = 'Shift'
        st.dataframe(cost_per_shift_df.style.highlight_max(axis=0))

    st.subheader("Generated Schedule")
    schedule_df = pd.DataFrame(solution['resources_shifts'])
    schedule_df = schedule_df.pivot(index='day', columns='shift', values='resources')
    schedule_df.index.name = "Day"
    st.dataframe(schedule_df.style.highlight_max(axis=1))

    st.subheader("Resource Requirements")
    req_df = pd.DataFrame(required_resources, index=[f"Period {i+1}" for i in range(periods)],
                          columns=[f"Day {i+1}" for i in range(num_days)])
    st.dataframe(req_df.style.highlight_max(axis=1))

    st.subheader("Requirements vs. Generated Schedule (Interactive)")
    fig = make_subplots(rows=num_days, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=[f"Day {i+1}" for i in range(num_days)])

    for day in range(num_days):
        fig.add_trace(
            go.Scatter(x=list(range(periods)), y=[row[day] for row in required_resources],
                       mode='lines+markers', name=f'Required (Day {day+1})', line=dict(color='blue')),
            row=day+1, col=1
        )
        
        scheduled = [sum(schedule_df.loc[day, shift] * shifts_coverage[shift][period] for shift in shift_names)
                     for period in range(periods)]
        fig.add_trace(
            go.Scatter(x=list(range(periods)), y=scheduled,
                       mode='lines+markers', name=f'Scheduled (Day {day+1})', line=dict(color='red')),
            row=day+1, col=1
        )

    fig.update_layout(height=200*num_days, title_text="Requirements vs. Generated Schedule per Day",
                      showlegend=False)
    fig.update_xaxes(title_text="Period")
    fig.update_yaxes(title_text="Number of Resources")

    st.plotly_chart(fig, use_container_width=True)

# Main app
st.title("Shift Schedule Generator")

# Sidebar
st.sidebar.header("📊 Input Parameters")

with st.sidebar.expander("📅 Schedule Parameters", expanded=True):
    num_days = st.number_input("Number of days to schedule", min_value=1, max_value=14, value=7)
    periods = st.number_input("Number of periods per day", min_value=1, max_value=48, value=24)
    num_shifts = st.number_input("Number of shifts", min_value=1, max_value=5, value=3)

shift_names = [f"Shift_{i+1}" for i in range(num_shifts)]
shifts_coverage = create_shifts_coverage(shift_names, periods)

with st.sidebar.expander("⏰ Shift Coverage", expanded=True):
    for shift in shift_names:
        st.subheader(shift)
        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input(f"Start period", min_value=0, max_value=periods-1, value=0, key=f"start_{shift}")
        with col2:
            end = st.number_input(f"End period", min_value=start+1, max_value=periods, value=periods, key=f"end_{shift}")
        shifts_coverage[shift][start:end] = [1] * (end - start)

with st.sidebar.expander("🔢 Resource Limits", expanded=True):
    max_period_concurrency = st.number_input("Max resources per period", min_value=1, value=30)
    max_shift_concurrency = st.number_input("Max resources per shift", min_value=1, value=25)

with st.sidebar.expander("💰 Shift Costs", expanded=True):
    costs = {shift: st.number_input(f"Cost for {shift}", min_value=0.1, value=1.0, step=0.1) for shift in shift_names}

# Main content
tab1, tab2, tab3 = st.tabs(["📋 Resource Requirements", "🧮 Generate Schedules", "ℹ️ About"])

with tab1:
    st.header("📋 Resource Requirements")
    requirements_option = st.radio("Choose how to set requirements:", ("Generate Random", "Input Manual"))

    if requirements_option == "Generate Random":
        required_resources = generate_random_requirements(num_days, periods)
        st.info("Random requirements have been generated. Go to the 'Generate Schedules' tab to see the results.")
    else:
        st.write("Enter the required resources for each day and period:")
        df = pd.DataFrame(index=[f"Period {i+1}" for i in range(periods)],
                          columns=[f"Day {i+1}" for i in range(num_days)])
        df = df.fillna(5)
        edited_df = st.data_editor(df)
        required_resources = edited_df.values.tolist()

with tab2:
    st.header("🧮 Generate Schedules")
    if st.button("Generate Schedules", key="generate_button"):
        with st.spinner("Generating schedules..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Min required Method")
                solution_abs = solve_scheduling_problem(
                    "MinAbsDifference", num_days, periods, shifts_coverage, required_resources,
                    max_period_concurrency, max_shift_concurrency, costs
                )
                
                if solution_abs['status'] == 'OPTIMAL':
                    st.success("Schedule generated successfully!")
                    display_results(solution_abs, required_resources, num_days, periods, shift_names, shifts_coverage, costs)
                else:
                    st.error("Unable to generate a feasible schedule. Try adjusting the parameters.")
            
            with col2:
                st.subheader("Min Required per interval Resources Method")
                solution_req = solve_scheduling_problem(
                    "MinRequiredResources", num_days, periods, shifts_coverage, required_resources,
                    max_period_concurrency, max_shift_concurrency, costs
                )
                
                if solution_req['status'] == 'OPTIMAL':
                    st.success("Schedule generated successfully!")
                    display_results(solution_req, required_resources, num_days, periods, shift_names, shifts_coverage, costs)
                else:
                    st.error("Unable to generate a feasible schedule. Try adjusting the parameters.")

with tab3:
    st.header("ℹ️ About the App")
    st.write("""
    This app generates optimal schedules by two methods. 
    
    To use the app:
    1. Set the input parameters in the sidebar.
    2. Choose how to set resource requirements in the 'Resource Requirements' tab.
    3. Generate schedules in the 'Generate Schedules' tab.
    4. Compare the results of both methods.

    For more information contact Ashwin Nair
    """)

# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #0e1117;">
        <p>|Developed with ❤️ by Ashwin Nair | 
    </div>
    """, unsafe_allow_html=True)
