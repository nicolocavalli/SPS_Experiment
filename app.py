import math
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Fixed power constants
# -----------------------------
Z_ALPHA = 1.96   # two-sided 5%
Z_BETA = 0.84    # 80% power
Z_SUM = Z_ALPHA + Z_BETA

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
    """Effective independent observations for repeated measures with ICC rho."""
    return (N * m) / (1 + (m - 1) * rho)

def mde_diff_means(N: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    """
    MDE for a two-group difference in means on continuous DV.
    p1, p2 are shares of observations in each group (within the module).
    """
    neff = effective_n(N, m, rho)
    return Z_SUM * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

def mde_did_interaction(N: float, m: int, rho: float, sigma: float, p_cells: dict) -> float:
    """
    MDE for a 2x2 difference-in-differences / interaction coefficient.
    p_cells = dict with four cell shares that sum to 1:
      keys can be anything; values are shares.
    SE(interaction) ≈ sigma * sqrt(sum_i 1/(neff * p_i))
    """
    neff = effective_n(N, m, rho)
    s = 0.0
    for p in p_cells.values():
        s += 1.0 / (neff * p)
    return Z_SUM * sigma * math.sqrt(s)

def mde_group_difference_two_group(N_A: float, N_B: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    """
    MDE for difference-in-effects between Group A and Group B for a two-group contrast.
    """
    neff_A = effective_n(N_A, m, rho)
    neff_B = effective_n(N_B, m, rho)
    var_A = (1/(neff_A*p1)) + (1/(neff_A*p2))
    var_B = (1/(neff_B*p1)) + (1/(neff_B*p2))
    return Z_SUM * sigma * math.sqrt(var_A + var_B)

def mde_group_difference_did(N_A: float, N_B: float, m: int, rho: float, sigma: float, p_cells: dict) -> float:
    """
    MDE for difference-in-interaction (DiD coefficient) between Group A and Group B.
    """
    neff_A = effective_n(N_A, m, rho)
    neff_B = effective_n(N_B, m, rho)

    sA = sum(1.0/(neff_A * p) for p in p_cells.values())
    sB = sum(1.0/(neff_B * p) for p in p_cells.values())
    return Z_SUM * sigma * math.sqrt(sA + sB)

# -----------------------------
# Estimands implied by your clarified "main tests"
# -----------------------------
def shares_vignette1_local_3yr_bins():
    """
    VIGNETTE 1 main test:
    local age-bin effect: adjacent 3-year bins over age 7–75.
    Age is uniform over 69 integer ages. A 3-year bin has share 3/69.
    We compare one 3-year bin vs an adjacent 3-year bin => p1=p2=3/69.
    """
    return (3/69, 3/69)

def cells_vignette2_arrived_minor_vs_adult_interaction():
    """
    VIGNETTE 2 main test (your clarification):
    NOT "arrived 5 years ago vs recently".
    Instead: "ARRIVED as a MINOR vs ARRIVED as an ADULT", which is identified as an interaction.

    Operationalization for the main test:
    - Use symmetric 3-year bins around 18 in CURRENT age: 15–17 vs 18–20
    - Arrival timing randomized: recently vs 5-years ago
    - DiD/interaction interpretation:
        (Effect of arrival_timing among 18–20) - (Effect of arrival_timing among 15–17)

    Under uniform age 15–25 (11 ages) and arrival_timing 50/50:
      P(current in 15–17) = 3/11
      P(current in 18–20) = 3/11
      Each bin × arrival_timing cell share = (3/11)*0.5 = 3/22
    The remaining ages (21–25) are excluded from this main estimand window.
    In analysis you'd restrict to the 15–20 window (or weight accordingly).
    For MDE planning, we treat the estimand as being estimated on that restricted sample.
    """
    # Four equal cells within the restricted window
    return {
        "15_17_recent": 3/22,
        "15_17_5years": 3/22,
        "18_20_recent": 3/22,
        "18_20_5years": 3/22,
    }

def shares_vignette3_conjoint_factors():
    """
    VIGNETTE 3 main test:
    typical conjoint-style factor effects (AMCE-style) on acceptance.

    We report MDEs for a small set of *core* contrasts that match your design:
    - Timing (prior vs during): 0.5 vs 0.5
    - Communicable vs non-communicable:
        communicable = {TB, HIV} => 2/6 = 1/3
        non-communicable = remaining 4/6 = 2/3
    - Age category within health vignette: 25 vs 45 (1/3 vs 1/3) as a representative pairwise contrast
      (25 vs 65 or 45 vs 65 would have the same shares under uniform {25,45,65}.)
    """
    return {
        "timing_prior_vs_during": (0.5, 0.5),
        "communicable_vs_non": (1/3, 2/3),
        "age25_vs_age45": (1/3, 1/3),
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Country Selection, Sample Sizes, Costs & MDE Explorer (Estimand-Aligned)")

st.sidebar.header("Statistical assumptions")
sigma = st.sidebar.slider("Outcome SD (σ) on 1–7 scale", 1.0, 2.5, 1.8, 0.1)
rho = st.sidebar.slider("ICC (ρ) within respondent", 0.0, 0.6, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Vignettes per respondent (m) by vignette module")
m1 = st.sidebar.slider("m₁: Vignette 1 (Age 7–75)", 1, 6, 1, 1)
m2 = st.sidebar.slider("m₂: Vignette 2 (Transition/adulthood 15–25)", 1, 6, 1, 1)
m3 = st.sidebar.slider("m₃: Vignette 3 (Health conjoint)", 1, 6, 2, 1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "MDE output mode",
    ["Pooled (all selected countries)", "Group comparison (A vs B)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "MDEs are reported in **1–7 scale points**.\n"
    "Vignette 2 MDE is for a **DiD/interaction** capturing "
    "\"arrived as minor vs adult\" around 18 using 15–17 vs 18–20."
)

# Offline country universe using plotly's built-in gapminder list
gap = px.data.gapminder()
gap2007 = gap[gap["year"] == 2007].copy()
COUNTRIES = gap2007[["country", "iso_alpha"]].drop_duplicates().sort_values("country").reset_index(drop=True)

# First-best starting selection
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

# Country add/remove control (type-to-search)
st.subheader("Country selection (type to search)")
all_country_names = COUNTRIES["country"].tolist()

selected_now = st.multiselect(
    "Start typing a country name to add/remove it:",
    options=all_country_names,
    default=sorted(list(st.session_state.selected.keys()))
)

# Sync session state to multiselect (preserve Ns where possible)
new_selected = {}
for c in selected_now:
    if c in st.session_state.selected:
        new_selected[c] = st.session_state.selected[c]
    else:
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

    # Vignette 1: local adjacent 3-year bins over 7–75
    p1_v1, p2_v1 = shares_vignette1_local_3yr_bins()

    # Vignette 2: arrived-minor vs adult around 18 is a DiD/interaction on 15–20 window
    cells_v2 = cells_vignette2_arrived_minor_vs_adult_interaction()

    # Vignette 3: conjoint-style contrasts
    v3_contrasts = shares_vignette3_conjoint_factors()

    st.markdown("### COSTS (VAT on top)")
    st.write(f"Net panel: €{costs['net_panel']:,.0f}")
    st.write(f"Management fee: €{costs['mgmt_fee']:,.0f}")
    st.write(f"VAT (22%): €{costs['vat']:,.0f}")
    st.write(f"**Total (gross): €{costs['gross_total']:,.0f}**")

    st.markdown("### MDE by vignette")

    if mode.startswith("Pooled"):
        # Vignette 1
        mde_v1 = mde_diff_means(total_N, m1, rho, sigma, p1_v1, p2_v1)

        # Vignette 2 (DiD interaction)
        mde_v2 = mde_did_interaction(total_N, m2, rho, sigma, cells_v2)

        # Vignette 3 contrasts
        mde_v3_timing = mde_diff_means(total_N, m3, rho, sigma, *v3_contrasts["timing_prior_vs_during"])
        mde_v3_comm = mde_diff_means(total_N, m3, rho, sigma, *v3_contrasts["communicable_vs_non"])
        mde_v3_age = mde_diff_means(total_N, m3, rho, sigma, *v3_contrasts["age25_vs_age45"])

        st.write(f"**Vignette 1 (Age 7–75): adjacent 3-year bins (e.g., 7–9 vs 10–12)** → MDE: **{mde_v1:.3f}**")
        st.write(f"**Vignette 2 (Transition): ARRIVED minor vs ARRIVED adult (DiD around 18 using 15–17 vs 18–20)** → MDE: **{mde_v2:.3f}**")
        st.write("**Vignette 3 (Health conjoint): factor-level MDEs**")
        st.write(f"• Timing (prior vs during): {mde_v3_timing:.3f}")
        st.write(f"• Communicable vs non-communicable: {mde_v3_comm:.3f}")
        st.write(f"• Age category (25 vs 45): {mde_v3_age:.3f}")

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
            # Vignette 1: group difference in local 3-year-bin effect
            mde_v1 = mde_group_difference_two_group(N_A, N_B, m1, rho, sigma, p1_v1, p2_v1)

            # Vignette 2: group difference in DiD interaction
            mde_v2 = mde_group_difference_did(N_A, N_B, m2, rho, sigma, cells_v2)

            # Vignette 3: group difference in conjoint contrasts
            mde_v3_timing = mde_group_difference_two_group(N_A, N_B, m3, rho, sigma, *v3_contrasts["timing_prior_vs_during"])
            mde_v3_comm = mde_group_difference_two_group(N_A, N_B, m3, rho, sigma, *v3_contrasts["communicable_vs_non"])
            mde_v3_age = mde_group_difference_two_group(N_A, N_B, m3, rho, sigma, *v3_contrasts["age25_vs_age45"])

            st.write(f"**Vignette 1 (Age 7–75): adjacent 3-year bins — Group-diff MDE**: **{mde_v1:.3f}**")
            st.write(f"**Vignette 2 (Arrived minor vs adult, DiD around 18) — Group-diff MDE**: **{mde_v2:.3f}**")
            st.write("**Vignette 3 (Health conjoint): group-diff MDEs**")
            st.write(f"• Timing (prior vs during): {mde_v3_timing:.3f}")
            st.write(f"• Communicable vs non-communicable: {mde_v3_comm:.3f}")
            st.write(f"• Age category (25 vs 45): {mde_v3_age:.3f}")

            st.caption(f"Group A total N: {N_A:,}  •  Group B total N: {N_B:,}")

    st.markdown("---")
    st.caption(f"Selected countries: {len(selected_samples)}  •  Total N: {total_N:,}")
    st.caption("All MDEs are in **1–7 scale points** (not percentage points).")

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
