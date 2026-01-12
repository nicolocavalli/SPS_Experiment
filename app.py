import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional: enables map click events (recommended)
try:
    from streamlit_plotly_events import plotly_events  # pip install streamlit-plotly-events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


# -----------------------------
# Fixed (assumed-correct) power constants
# -----------------------------
Z_ALPHA = 1.96   # two-sided 5%
Z_BETA = 0.84    # 80% power

# Costs
VAT_RATE = 0.22
MGMT_FEE_EUR = 2000.0

# Cost rule (per your instruction):
# - all countries default to 2.60 EUR per complete (net)
# - Tunisia special (high cost) kept at 8.40 net
DEFAULT_PPC_NET = 2.60
SPECIAL_PPC_NET = {
    "Tunisia": 8.40
}

# -----------------------------
# Power / MDE helper functions
# -----------------------------
def effective_n(N: float, m: int, rho: float) -> float:
    """Effective independent observations for repeated measures with ICC rho."""
    return (N * m) / (1 + (m - 1) * rho)

def mde_two_group_from_neff(neff: float, sigma: float, p1: float, p2: float) -> float:
    """Two-group MDE on continuous DV given effective n and group shares."""
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

def mde_two_group(N: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    neff = effective_n(N, m, rho)
    return mde_two_group_from_neff(neff, sigma, p1, p2)

def mde_group_difference(N_A: float, N_B: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    """MDE for difference-in-effects between two groups (interaction test)."""
    neff_A = effective_n(N_A, m, rho)
    neff_B = effective_n(N_B, m, rho)
    var_A = (1/(neff_A*p1)) + (1/(neff_A*p2))
    var_B = (1/(neff_B*p1)) + (1/(neff_B*p2))
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt(var_A + var_B)

# -----------------------------
# Experiment-specific share settings (as requested)
# -----------------------------
def shares_experiment_1_three_year_bins():
    """
    Experiment 1: age profile 7–75 (inclusive), uniform.
    69 possible ages. A 3-year bin has 3/69 probability.
    For an adjacent-bin comparison: p1=p2=3/69.
    """
    return (3/69, 3/69)

def shares_experiment_2_around_18_three_year_bins():
    """
    Experiment 2: ages 15–25 (11 values), uniform.
    3-year bins around 18: 15–17 vs 18–20 => p1=p2=3/11.
    """
    return (3/11, 3/11)

def shares_experiment_3_health_timing():
    """
    Experiment 3: timing prior vs during, 50/50 in code.
    """
    return (0.5, 0.5)

# -----------------------------
# Cost functions
# -----------------------------
def ppc_net(country_name: str) -> float:
    return SPECIAL_PPC_NET.get(country_name, DEFAULT_PPC_NET)

def compute_costs(selected_samples: dict) -> dict:
    """
    selected_samples: {country_name: N}
    Returns dict with net panel, management, VAT, gross total.
    """
    net_panel = sum(N * ppc_net(c) for c, N in selected_samples.items())
    subtotal_net = net_panel + MGMT_FEE_EUR
    vat = subtotal_net * VAT_RATE
    gross_total = subtotal_net + vat
    return {
        "net_panel": net_panel,
        "mgmt_fee": MGMT_FEE_EUR,
        "vat": vat,
        "gross_total": gross_total
    }

# -----------------------------
# UI setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("Country Selection, Samples, Costs & Power (MDE)")

# Sidebar: statistical assumptions + m per module
st.sidebar.header("Statistical assumptions (fixed knobs)")
sigma = st.sidebar.slider("Outcome SD (σ) on 1–7 scale", 1.0, 2.5, 1.8, 0.1)
rho = st.sidebar.slider("ICC (ρ) within respondent", 0.0, 0.6, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Vignettes per respondent (m) by experiment/module")

m1 = st.sidebar.slider("m₁: Experiment 1 (Age profile 7–75)", 1, 4, 2, 1)
m2 = st.sidebar.slider("m₂: Experiment 2 (Adulthood 15–25 + arrival)", 1, 4, 2, 1)
m3 = st.sidebar.slider("m₃: Experiment 3 (Health)", 1, 4, 2, 1)

st.sidebar.markdown("---")
st.sidebar.header("Comparison mode")
mode = st.sidebar.radio(
    "MDE output mode",
    ["Pooled (all selected countries)", "Group comparison (A vs B)"],
    index=0
)

# -----------------------------
# Build a world-ish map using Plotly gapminder country ISO-3 list (offline)
# -----------------------------
gap = px.data.gapminder()
gap2007 = gap[gap["year"] == 2007].copy()

# Create a stable country list with ISO-3 codes and readable names
COUNTRIES = gap2007[["country", "iso_alpha"]].drop_duplicates().sort_values("country").reset_index(drop=True)

# First-best start (your earlier recommended set) mapped onto available gapminder countries
# Note: gapminder uses "Tunisia" and "Turkey", "Germany", "Italy", "Greece", "Poland", "Sweden"
FIRST_BEST = {
    "Italy": 969,
    "Greece": 969,
    "Germany": 969,
    "Sweden": 969,
    "Poland": 969,
    "Turkey": 969,
    "Tunisia": 400,
}

# Session state for selection + sample sizes
if "selected" not in st.session_state:
    st.session_state.selected = dict(FIRST_BEST)

def toggle_country(country_name: str):
    if country_name in st.session_state.selected:
        # Toggle off (acts like double-click deselect)
        st.session_state.selected.pop(country_name, None)
    else:
        # Toggle on with a default N
        st.session_state.selected[country_name] = 900

# -----------------------------
# MAP (main visualization) - on top
# -----------------------------
st.subheader("World map (click to toggle selection)")

# Create a map dataframe with N values (0 if not selected)
map_df = COUNTRIES.copy()
map_df["N"] = map_df["country"].map(lambda c: st.session_state.selected.get(c, 0))

fig = px.choropleth(
    map_df,
    locations="iso_alpha",
    color="N",
    hover_name="country",
    color_continuous_scale="Blues",
    range_color=(0, max(map_df["N"].max(), 1)),
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# Layout: map + outputs beside it
map_col, out_col = st.columns([2.1, 1.0], vertical_alignment="top")

with map_col:
    if HAS_PLOTLY_EVENTS:
        selected_points = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=520,
            override_width="100%",
            key="map_events"
        )
        # If clicked, toggle country
        if selected_points:
            # plotly returns point data; we stored hover_name as country
            # Depending on plotly version, this may be under "hovertext" or "customdata"
            pt = selected_points[0]
            clicked_country = pt.get("hovertext") or pt.get("text")
            if clicked_country:
                toggle_country(clicked_country)
                st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.info("Tip: install `streamlit-plotly-events` to enable click-to-select on the map.")

with out_col:
    st.subheader("Outputs")

    selected_samples = dict(st.session_state.selected)
    # Compute costs (VAT on top)
    costs = compute_costs(selected_samples)

    # Experiment-specific MDEs (shares fixed per instructions)
    p1_e1, p2_e1 = shares_experiment_1_three_year_bins()            # Exp 1: 3-year bins
    p1_e2, p2_e2 = shares_experiment_2_around_18_three_year_bins()  # Exp 2: 15–17 vs 18–20
    p1_e3, p2_e3 = shares_experiment_3_health_timing()              # Exp 3: timing prior vs during

    # Pooled or group comparison
    total_N = sum(selected_samples.values())

    if mode.startswith("Pooled"):
        mde_e1 = mde_two_group(total_N, m1, rho, sigma, p1_e1, p2_e1)
        mde_e2 = mde_two_group(total_N, m2, rho, sigma, p1_e2, p2_e2)
        mde_e3 = mde_two_group(total_N, m3, rho, sigma, p1_e3, p2_e3)

        st.markdown("### COSTS (VAT on top)")
        st.write(f"Net panel: €{costs['net_panel']:,.0f}")
        st.write(f"Management fee: €{costs['mgmt_fee']:,.0f}")
        st.write(f"VAT (22%): €{costs['vat']:,.0f}")
        st.write(f"**Total (gross): €{costs['gross_total']:,.0f}**")

        st.markdown("### MDE by experiment (pooled)")
        st.write(f"**Exp 1 (Age 7–75, 3-year bin vs 3-year bin):** {mde_e1:.3f} scale points")
        st.write(f"**Exp 2 (15–17 vs 18–20 around 18):** {mde_e2:.3f} scale points")
        st.write(f"**Exp 3 (Health timing prior vs during):** {mde_e3:.3f} scale points")

    else:
        st.markdown("### Define groups for A vs B comparison")
        sel_list = sorted(list(selected_samples.keys()))
        group_A = st.multiselect("Group A countries", sel_list, default=sel_list[: max(1, len(sel_list)//2)])
        group_B = [c for c in sel_list if c not in group_A]

        N_A = sum(selected_samples[c] for c in group_A)
        N_B = sum(selected_samples[c] for c in group_B)

        if N_A == 0 or N_B == 0:
            st.warning("Select at least one country in Group A and leave at least one in Group B.")
        else:
            mde_e1 = mde_group_difference(N_A, N_B, m1, rho, sigma, p1_e1, p2_e1)
            mde_e2 = mde_group_difference(N_A, N_B, m2, rho, sigma, p1_e2, p2_e2)
            mde_e3 = mde_group_difference(N_A, N_B, m3, rho, sigma, p1_e3, p2_e3)

            st.markdown("### COSTS (VAT on top)")
            st.write(f"Net panel: €{costs['net_panel']:,.0f}")
            st.write(f"Management fee: €{costs['mgmt_fee']:,.0f}")
            st.write(f"VAT (22%): €{costs['vat']:,.0f}")
            st.write(f"**Total (gross): €{costs['gross_total']:,.0f}**")

            st.markdown("### MDE by experiment (Group A vs Group B difference-in-effects)")
            st.write(f"**Exp 1 (Age 7–75, 3-year bin vs 3-year bin):** {mde_e1:.3f} scale points")
            st.write(f"**Exp 2 (15–17 vs 18–20 around 18):** {mde_e2:.3f} scale points")
            st.write(f"**Exp 3 (Health timing prior vs during):** {mde_e3:.3f} scale points")

            st.caption(f"Group A total N: {N_A:,}  •  Group B total N: {N_B:,}")

    st.markdown("---")
    st.caption(f"Selected countries: {len(selected_samples)}  •  Total N: {total_N:,}")
    st.caption("Note: MDEs are in **1–7 scale points** (not percentage points).")

# -----------------------------
# Country dashboard (below the map)
# -----------------------------
st.subheader("Country dashboard (sample-size sliders)")

if not st.session_state.selected:
    st.info("Click countries on the map to add them.")
else:
    dash = []
    for c, N in sorted(st.session_state.selected.items()):
        dash.append((c, N, ppc_net(c)))
    dash_df = pd.DataFrame(dash, columns=["Country", "N", "Net cost per complete (€)"])
    st.dataframe(dash_df, use_container_width=True, hide_index=True)

    st.markdown("### Adjust sample sizes")
    # Put sliders in two columns for compactness
    slider_cols = st.columns(2)
    i = 0
    for c in sorted(st.session_state.selected.keys()):
        col = slider_cols[i % 2]
        with col:
            newN = st.slider(
                f"{c} — N",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.selected[c]),
                step=50,
                key=f"slider_{c}"
            )
            if newN == 0:
                # Treat as deselect
                st.session_state.selected.pop(c, None)
                st.rerun()
            else:
                st.session_state.selected[c] = int(newN)
        i += 1

st.markdown("---")
st.caption(
    "If map clicks are not working: `pip install streamlit-plotly-events` and rerun. "
    "Selection toggles on click; clicking again deselects (double-click behavior)."
)
