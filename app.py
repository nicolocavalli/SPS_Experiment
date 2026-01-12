
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math

# -----------------------------
# Power / MDE helper functions
# -----------------------------

Z_ALPHA = 1.96   # two-sided 5%
Z_BETA = 0.84    # 80% power

def effective_n(N, m, rho):
    return (N * m) / (1 + (m - 1) * rho)

def mde_two_group(N, m, rho, sigma, p1, p2):
    neff = effective_n(N, m, rho)
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

def mde_pooled(countries, m, rho, sigma, p1, p2):
    total_N = sum(countries.values())
    neff = effective_n(total_N, m, rho)
    return (Z_ALPHA + Z_BETA) * sigma * math.sqrt((1/(neff*p1)) + (1/(neff*p2)))

# -----------------------------
# Country cost data (net prices)
# -----------------------------

COUNTRY_DATA = {
    "Italy":    {"ppc": 2.50, "iso": "ITA"},
    "Greece":   {"ppc": 2.50, "iso": "GRC"},
    "Germany":  {"ppc": 2.60, "iso": "DEU"},
    "Sweden":   {"ppc": 2.60, "iso": "SWE"},
    "Poland":   {"ppc": 2.50, "iso": "POL"},
    "Turkey":   {"ppc": 2.90, "iso": "TUR"},
    "Tunisia":  {"ppc": 8.40, "iso": "TUN"},
    "France":   {"ppc": 2.60, "iso": "FRA"},
    "Spain":    {"ppc": 2.50, "iso": "ESP"},
    "USA":      {"ppc": 2.60, "iso": "USA"},
    "Mexico":   {"ppc": 2.60, "iso": "MEX"},
}

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(layout="wide")
st.title("Vignette Design, Power, and Cost Explorer")

st.markdown("""
This app helps explore **country selection, sample sizes, costs (VAT on top)** and
**minimum detectable effects (MDEs)** for the refugee vignette experiments.

**Outcome unit:** 1–7 acceptance scale  
(MDE = smallest detectable difference in scale points)
""")

# Sidebar parameters
st.sidebar.header("Statistical assumptions")

sigma = st.sidebar.slider("Outcome SD (σ)", 1.0, 2.5, 1.8, 0.1)
rho = st.sidebar.slider("ICC (ρ)", 0.0, 0.6, 0.30, 0.05)
m_transition = st.sidebar.slider("Transition vignettes per respondent (m)", 1, 4, 2, 1)

st.sidebar.markdown("---")
st.sidebar.header("Age-bin definition")

age_bin = st.sidebar.selectbox(
    "Age comparison",
    ["17 vs 18", "16–17 vs 18–19", "15–17 vs 18–20", "Minor (15–17) vs Adult (18–25)"]
)

if age_bin == "17 vs 18":
    p1 = p2 = 1/11
elif age_bin == "16–17 vs 18–19":
    p1 = p2 = 2/11
elif age_bin == "15–17 vs 18–20":
    p1 = p2 = 3/11
else:
    p1, p2 = 3/11, 8/11

# Initial first-best sample sizes
default_samples = {
    "Italy": 969,
    "Greece": 969,
    "Germany": 969,
    "Sweden": 969,
    "Poland": 969,
    "Turkey": 969,
    "Tunisia": 400,
}

st.subheader("Country selection and sample sizes")

countries_selected = st.multiselect(
    "Select countries",
    list(COUNTRY_DATA.keys()),
    default=list(default_samples.keys())
)

samples = {}
for c in countries_selected:
    default_n = default_samples.get(c, 900)
    samples[c] = st.slider(f"{c} sample size (N)", 100, 2000, default_n, 50)

# -----------------------------
# Cost calculations (VAT on top)
# -----------------------------

net_panel_cost = sum(samples[c] * COUNTRY_DATA[c]["ppc"] for c in samples)
management_fee = 2000
subtotal_net = net_panel_cost + management_fee
vat = 0.22 * subtotal_net
total_cost = subtotal_net + vat

# -----------------------------
# Power calculations
# -----------------------------

pooled_mde = mde_pooled(samples, m_transition, rho, sigma, p1, p2)

# Per-country MDEs
per_country_mde = {
    c: mde_two_group(samples[c], m_transition, rho, sigma, p1, p2)
    for c in samples
}

# -----------------------------
# Outputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Costs (EUR)")
    st.write(f"Net panel cost: €{net_panel_cost:,.0f}")
    st.write(f"Management fee: €{management_fee:,.0f}")
    st.write(f"VAT (22%): €{vat:,.0f}")
    st.write(f"**Total cost (gross): €{total_cost:,.0f}**")

with col2:
    st.subheader("Power / MDE")
    st.write(f"**Pooled MDE ({age_bin}): {pooled_mde:.3f} scale points**")
    st.markdown("Per-country MDEs:")
    for c, v in per_country_mde.items():
        st.write(f"{c}: {v:.3f}")

# -----------------------------
# World map
# -----------------------------

st.subheader("Country map")

map_df = pd.DataFrame({
    "country": list(samples.keys()),
    "iso": [COUNTRY_DATA[c]["iso"] for c in samples],
    "N": list(samples.values())
})

fig = px.choropleth(
    map_df,
    locations="iso",
    color="N",
    hover_name="country",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig, use_container_width=True)
