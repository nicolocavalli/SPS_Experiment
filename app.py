import math
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Fixed power constants
# -----------------------------
Z_ALPHA = 1.96   # two-sided 5%
Z_BETA = 0.84    # 80% power

# -----------------------------
# Cost constants (VAT on top)
# -----------------------------
VAT_RATE = 0.22
MGMT_FEE_EUR = 2000.0

DEFAULT_PPC_NET = 2.60
SPECIAL_PPC_NET = {"Tunisia": 8.40}

def ppc_net(country_name: str) -> float:
    return SPECIAL_PPC_NET.get(country_name, DEFAULT_PPC_NET)

def compute_costs(selected_samples: dict) -> dict:
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
# Power / MDE helpers
# -----------------------------
def effective_n(N: float, m: int, rho: float) -> float:
    return (N * m) / (1 + (m - 1) * rho)

def mde_two_group(N: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    neff = effective_n(N, m, rho)
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

def mde_group_difference(N_A: float, N_B: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    neff_A = effective_n(N_A, m, rho)
    neff_B = effective_n(N_B, m, rho)
    var_A = (1/(neff_A*p1)) + (1/(neff_A*p2))
    var_B = (1/(neff_B*p1)) + (1/(neff_B*p2))
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt(var_A + var_B)

# -----------------------------
# Experiment-specific shares (fixed per your request)
# -----------------------------
def shares_exp1_three_year_bins():
    # Age 7–75 inclusive => 69 ages. A 3-year bin => 3/69
    return (3/69, 3/69)

def shares_exp2_around18_three_year_bins():
    # Age 15–25 inclusive => 11 ages. 15–17 vs 18–20 => 3/11
    return (3/11, 3/11)

def shares_exp3_health_timing():
    # prior vs during => 0.5 / 0.5
    return (0.5, 0.5)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Country Selection, Sample Sizes, Costs & MDE Explorer")

# Sidebar: assumptions
st.sidebar.header("Statistical assumptions")
sigma = st.sidebar.slider("Outcome SD (σ) on 1–7 scale", 1.0, 2.5, 1.8, 0.1)
rho = st.sidebar.slider("ICC (ρ) within respondent", 0.0, 0.6, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Vignettes per respondent (m) by experiment")
m1 = st.sidebar.slider("m₁: Experiment 1 (Age profile 7–75)", 1, 6, 2, 1)
m2 = st.sidebar.slider("m₂: Experiment 2 (Adulthood 15–25)", 1, 6, 2, 1)
m3 = st.sidebar.slider("m₃: Experiment 3 (Health)", 1, 6, 2, 1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "MDE output mode",
    ["Pooled (all selected countries)", "Group comparison (A vs B)"],
    index=0
)

# Offline country universe using plotly's built-in gapminder list
gap = px.data.gapminder()
gap2007 = gap[gap["year"] == 2007].copy()
COUNTRIES = gap2007[["country", "iso_alpha"]].drop_duplicates().sort_values("country").reset_index(drop=True)

# First-best starting selection (your preferred set)
FIRST_BEST = {
    "Italy": 969,
    "Greece": 969,
    "Germany": 969,
    "Sweden": 969,
    "Poland": 969,
    "Turkey": 969,
    "Tunisia": 400,
}

if "selected" not in st.session_state:
    st.session_state.selected = dict(FIRST_BEST)

# --- Country add/remove control (type-to-search)
st.subheader("Country selection (type to search)")
all_country_names = COUNTRIES["country"].tolist()

selected_now = st.multiselect(
    "Start typing a country name to add/remove it:",
    options=all_country_names,
    default=sorted(list(st.session_state.selected.keys()))
)

# Sync session state to multiselect (preserve existing Ns where possible)
new_selected = {}
for c in selected_now:
    if c in st.session_state.selected:
        new_selected[c] = st.session_state.selected[c]
    else:
        # default N for new countries
        new_selected[c] = 900
        if c == "Tunisia":
            new_selected[c] = 400
st.session_state.selected = new_selected

selected_samples = dict(st.session_state.selected)

# --- Map + outputs row (map on top, main visualization)
st.subheader("World map and outputs")
map_col, out_col = st.columns([2.2, 1.0], vertical_alignment="top")

# Build map dataframe with selected N (0 otherwise)
map_df = COUNTRIES.copy()
sel_set = set(selected_samples.keys())
map_df["N"] = map_df["country"].map(lambda c: selected_samples.get(c, 0))

with map_col:
    fig = px.choropleth(
        map_df,
        locations="iso_alpha",
        color="N",
        hover_name="country",
        color_continuous_scale="Blues",
        range_color=(0, max(map_df["N"].max(), 1)),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with out_col:
    st.subheader("Outputs")

    costs = compute_costs(selected_samples)
    total_N = sum(selected_samples.values())

    p1_e1, p2_e1 = shares_exp1_three_year_bins()
    p1_e2, p2_e2 = shares_exp2_around18_three_year_bins()
    p1_e3, p2_e3 = shares_exp3_health_timing()

    st.markdown("### COSTS (VAT on top)")
    st.write(f"Net panel: €{costs['net_panel']:,.0f}")
    st.write(f"Management fee: €{costs['mgmt_fee']:,.0f}")
    st.write(f"VAT (22%): €{costs['vat']:,.0f}")
    st.write(f"**Total (gross): €{costs['gross_total']:,.0f}**")

    st.markdown("### MDE by experiment")
    if mode.startswith("Pooled"):
        mde_e1 = mde_two_group(total_N, m1, rho, sigma, p1_e1, p2_e1)
        mde_e2 = mde_two_group(total_N, m2, rho, sigma, p1_e2, p2_e2)
        mde_e3 = mde_two_group(total_N, m3, rho, sigma, p1_e3, p2_e3)

        st.write(f"**Exp 1 (Age 7–75, 3-year bin vs 3-year bin):** {mde_e1:.3f} scale points")
        st.write(f"**Exp 2 (15–17 vs 18–20 around 18):** {mde_e2:.3f} scale points")
        st.write(f"**Exp 3 (Health timing prior vs during):** {mde_e3:.3f} scale points")
    else:
        sel_list = sorted(list(selected_samples.keys()))
        st.markdown("#### Group A vs Group B (difference-in-effects)")
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

            st.write(f"**Exp 1 (Age 7–75, 3-year bin vs 3-year bin):** {mde_e1:.3f} scale points")
            st.write(f"**Exp 2 (15–17 vs 18–20 around 18):** {mde_e2:.3f} scale points")
            st.write(f"**Exp 3 (Health timing prior vs during):** {mde_e3:.3f} scale points")
            st.caption(f"Group A total N: {N_A:,}  •  Group B total N: {N_B:,}")

    st.markdown("---")
    st.caption(f"Selected countries: {len(selected_samples)}  •  Total N: {total_N:,}")
    st.caption("MDEs are in **1–7 scale points** (not percentage points).")

# --- Dashboard below
st.subheader("Country dashboard (sample-size sliders)")

if not selected_samples:
    st.info("Use the country selector above to add countries.")
else:
    dash_rows = []
    for c, N in sorted(selected_samples.items()):
        dash_rows.append((c, N, ppc_net(c)))
    dash_df = pd.DataFrame(dash_rows, columns=["Country", "N", "Net cost per complete (€)"])
    st.dataframe(dash_df, use_container_width=True, hide_index=True)

    st.markdown("### Adjust sample sizes (set to 0 to remove)")
    cols = st.columns(2)
    i = 0
    for c in sorted(list(selected_samples.keys())):
        with cols[i % 2]:
            newN = st.slider(
                f"{c} — N",
                min_value=0,
                max_value=5000,
                value=int(selected_samples[c]),
                step=50,
                key=f"slider_{c}"
            )
            if newN == 0:
                st.session_state.selected.pop(c, None)
                st.rerun()
            else:
                st.session_state.selected[c] = int(newN)
        i += 1
