import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional: enables map click events
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
# - all countries default to 2.60 EUR net
# - Tunisia special (high cost) net 8.40
DEFAULT_PPC_NET = 2.60
SPECIAL_PPC_NET = {"Tunisia": 8.40}


# -----------------------------
# Power / MDE helper functions
# -----------------------------
def effective_n(N: float, m: int, rho: float) -> float:
    return (N * m) / (1 + (m - 1) * rho)

def mde_two_group_from_neff(neff: float, sigma: float, p1: float, p2: float) -> float:
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

def mde_two_group(N: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    neff = effective_n(N, m, rho)
    return mde_two_group_from_neff(neff, sigma, p1, p2)

def mde_group_difference(N_A: float, N_B: float, m: int, rho: float, sigma: float, p1: float, p2: float) -> float:
    neff_A = effective_n(N_A, m, rho)
    neff_B = effective_n(N_B, m, rho)
    var_A = (1/(neff_A*p1)) + (1/(neff_A*p2))
    var_B = (1/(neff_B*p1)) + (1/(neff_B*p2))
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt(var_A + var_B)


# -----------------------------
# Experiment-specific shares
# -----------------------------
def shares_experiment_1_three_year_bins():
    # Exp 1: age 7–75 inclusive => 69 values; 3-year bin => 3/69
    return (3/69, 3/69)

def shares_experiment_2_around_18_three_year_bins():
    # Exp 2: age 15–25 inclusive => 11 values; 15–17 vs 18–20 => 3/11
    return (3/11, 3/11)

def shares_experiment_3_health_timing():
    # Exp 3: timing prior vs during => 50/50
    return (0.5, 0.5)


# -----------------------------
# Cost functions
# -----------------------------
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
# UI setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("Country Selection, Samples, Costs & Power (MDE)")

st.sidebar.header("Statistical assumptions")
sigma = st.sidebar.slider("Outcome SD (σ) on 1–7 scale", 1.0, 2.5, 1.8, 0.1)
rho = st.sidebar.slider("ICC (ρ) within respondent", 0.0, 0.6, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Vignettes per respondent (m) by experiment/module")
m1 = st.sidebar.slider("m₁: Exp 1 (Age profile 7–75)", 1, 4, 2, 1)
m2 = st.sidebar.slider("m₂: Exp 2 (Adulthood 15–25)", 1, 4, 2, 1)
m3 = st.sidebar.slider("m₃: Exp 3 (Health)", 1, 4, 2, 1)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "MDE output mode",
    ["Pooled (all selected countries)", "Group comparison (A vs B)"],
    index=0
)

# -----------------------------
# Country universe (offline)
# -----------------------------
gap = px.data.gapminder()
gap2007 = gap[gap["year"] == 2007].copy()
COUNTRIES = gap2007[["country", "iso_alpha"]].drop_duplicates().sort_values("country").reset_index(drop=True)
ISO_TO_COUNTRY = dict(zip(COUNTRIES["iso_alpha"], COUNTRIES["country"]))

# First-best starting point
FIRST_BEST = {
    "Italy": 969,
    "Greece": 969,
    "Germany": 969,
    "Sweden": 969,
    "Poland": 969,
    "Turkey": 969,
    "Tunisia": 400,
}

# Session state
if "selected" not in st.session_state:
    st.session_state.selected = dict(FIRST_BEST)

def toggle_country(country_name: str):
    # Click again de-selects (toggle)
    if country_name in st.session_state.selected:
        st.session_state.selected.pop(country_name, None)
    else:
        st.session_state.selected[country_name] = 900

def normalize_clicked_country(pt: dict, map_df: pd.DataFrame) -> str | None:
    """
    Robustly extract the clicked country name from plotly_events payload.
    Usually pt["location"] is ISO3, pt["hovertext"] is country, but varies.
    """
    if not pt:
        return None
    # Try hovertext
    if isinstance(pt.get("hovertext"), str) and pt["hovertext"]:
        return pt["hovertext"]
    # Try location (ISO3)
    loc = pt.get("location")
    if isinstance(loc, str) and loc in ISO_TO_COUNTRY:
        return ISO_TO_COUNTRY[loc]
    # Try pointNumber lookup from map_df
    pn = pt.get("pointNumber")
    if isinstance(pn, int) and 0 <= pn < len(map_df):
        return map_df.iloc[pn]["country"]
    return None


# -----------------------------
# Map (top, main visualization)
# -----------------------------
st.subheader("World map (click to toggle; click again to de-select)")

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

map_col, out_col = st.columns([2.2, 1.0], vertical_alignment="top")

with map_col:
    if HAS_PLOTLY_EVENTS:
        # Make truly clickable
        pts = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=520,
            override_width="100%",
            key="map_events"
        )
        if pts:
            clicked = normalize_clicked_country(pts[0], map_df)
            if clicked:
                toggle_country(clicked)
                st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.warning(
            "Map clicks require `streamlit-plotly-events`.\n\n"
            "Install: `pip install streamlit-plotly-events` (and add it to requirements.txt)."
        )

        # Fallback: still allow adding/removing countries easily
        st.markdown("### Fallback (since map clicks are unavailable)")
        fallback_pick = st.multiselect(
            "Add/remove countries here:",
            options=COUNTRIES["country"].tolist(),
            default=sorted(list(st.session_state.selected.keys()))
        )
        # Sync selection to session_state
        new_selected = {c: st.session_state.selected.get(c, 900) for c in fallback_pick}
        # Preserve Tunisia default if newly added
        for c in new_selected:
            if c == "Tunisia" and c not in st.session_state.selected:
                new_selected[c] = 400
        st.session_state.selected = new_selected

with out_col:
    st.subheader("Outputs")

    selected_samples = dict(st.session_state.selected)
    costs = compute_costs(selected_samples)

    p1_e1, p2_e1 = shares_experiment_1_three_year_bins()
    p1_e2, p2_e2 = shares_experiment_2_around_18_three_year_bins()
    p1_e3, p2_e3 = shares_experiment_3_health_timing()

    total_N = sum(selected_samples.values())

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
        st.markdown("#### Group A vs Group B (difference-in-effects)")
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

            st.write(f"**Exp 1 (Age 7–75, 3-year bin vs 3-year bin):** {mde_e1:.3f} scale points")
            st.write(f"**Exp 2 (15–17 vs 18–20 around 18):** {mde_e2:.3f} scale points")
            st.write(f"**Exp 3 (Health timing prior vs during):** {mde_e3:.3f} scale points")
            st.caption(f"Group A total N: {N_A:,}  •  Group B total N: {N_B:,}")

    st.markdown("---")
    st.caption(f"Selected countries: {len(selected_samples)}  •  Total N: {total_N:,}")
    st.caption("MDEs are in **1–7 scale points** (not percentage points).")


# -----------------------------
# Country dashboard (below)
# -----------------------------
st.subheader("Country dashboard (sample-size sliders)")

if not st.session_state.selected:
    st.info("Click countries on the map (or use fallback selector) to add them.")
else:
    dash = []
    for c, N in sorted(st.session_state.selected.items()):
        dash.append((c, N, ppc_net(c)))
    dash_df = pd.DataFrame(dash, columns=["Country", "N", "Net cost per complete (€)"])
    st.dataframe(dash_df, use_container_width=True, hide_index=True)

    st.markdown("### Adjust sample sizes")
    slider_cols = st.columns(2)
    i = 0
    for c in sorted(list(st.session_state.selected.keys())):
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
                st.session_state.selected.pop(c, None)
                st.rerun()
            else:
                st.session_state.selected[c] = int(newN)
        i += 1

st.markdown("---")
if not HAS_PLOTLY_EVENTS:
    st.info("To enable map clicking: `pip install streamlit-plotly-events` and rerun.")
else:
    st.caption("Map is clickable: click to toggle select/de-select.")
