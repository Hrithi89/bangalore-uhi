# ============================================================
# app.py — Bangalore Urban Heat Island Analyser
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore UHI Analyser",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    monthly_stats      = joblib.load('data/monthly_stats.pkl')
    daily_lookup       = joblib.load('data/daily_lookup.pkl')
    trend_data         = joblib.load('data/trend_data.pkl')
    poly_trend_data    = joblib.load('data/poly_trend_data.pkl')
    festival_data      = joblib.load('data/festival_data.pkl')
    model_kmeans       = joblib.load('data/model_kmeans.pkl')
    scaler_c           = joblib.load('data/scaler_c.pkl')
    cluster_map        = joblib.load('data/cluster_map.pkl')
    risk_config        = joblib.load('data/risk_config.pkl')
    summary            = joblib.load('data/summary.pkl')
    model_rf_hi        = joblib.load('data/model_rf_heatindex.pkl')
    scaler_rf          = joblib.load('data/scaler_rf.pkl')
    feature_importance = joblib.load('data/feature_importance.pkl')
    return (monthly_stats, daily_lookup, trend_data, poly_trend_data,
            festival_data, model_kmeans, scaler_c, cluster_map,
            risk_config, summary, model_rf_hi, scaler_rf,
            feature_importance)

@st.cache_data
def load_data():
    df = pd.read_csv('data/bangalore_weather_final.csv')
    return df[df['year'] != 2024]  # drop incomplete 2024

(monthly_stats, daily_lookup, trend_data, poly_trend_data,
 festival_data, model_kmeans, scaler_c, cluster_map,
 risk_config, summary, model_rf_hi, scaler_rf,
 feature_importance) = load_models()

df = load_data()

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #060e1a;
    color: #e8f0f7;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 2rem 2rem; max-width: 1200px; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.6rem;
    font-weight: 800; line-height: 1.15;
    background: linear-gradient(135deg, #00d4aa, #7ab8e8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #7a9bb5; font-size: 0.95rem; margin-bottom: 1.5rem; }

.card {
    background: #0a1628; border: 1px solid #1a3050;
    border-radius: 12px; padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.card-title {
    font-size: 0.7rem; font-weight: 600; color: #00d4aa;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.card-val {
    font-family: 'Syne', sans-serif; font-size: 1.7rem;
    font-weight: 700; color: #e8f0f7;
}
.card-sub { font-size: 0.75rem; color: #7a9bb5; margin-top: 0.2rem; }

.sec-header {
    font-size: 0.7rem; font-weight: 600; color: #00d4aa;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding-bottom: 0.4rem; border-bottom: 1px solid #1a3050;
    margin-bottom: 0.8rem;
}

.result-card {
    background: #0a1628; border: 1px solid #1a3050;
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    margin-bottom: 0.8rem;
}
.result-val {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 700;
}
.result-label { font-size: 0.75rem; color: #7a9bb5; margin-top: 0.2rem; }

.alert-safe    { background:rgba(0,212,170,0.08);  border:1px solid #00d4aa; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.8rem; }
.alert-caution { background:rgba(239,159,39,0.10); border:1px solid #EF9F27; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.8rem; }
.alert-danger  { background:rgba(255,140,66,0.10); border:1px solid #FF8C42; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.8rem; }
.alert-extreme { background:rgba(216,90,48,0.12);  border:1px solid #D85A30; border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.8rem; }

[data-testid="stSelectbox"] > div > div {
    background: #0f2035 !important; border: 1px solid #1a3050 !important;
    border-radius: 8px !important; color: #e8f0f7 !important;
}
[data-testid="stSlider"] > div > div > div { background: #00d4aa !important; }
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    color: #7a9bb5 !important; font-size: 0.82rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00d4aa !important; border-bottom: 2px solid #00d4aa !important;
}
.stButton > button {
    background: #0a1628 !important; color: #7a9bb5 !important;
    border: 1px solid #1a3050 !important; border-radius: 20px !important;
    font-weight: 400 !important; width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #112240 !important; color: #00d4aa !important;
    border-color: #00d4aa !important;
}
[data-testid="stButton-nav_active"] > button {
    background: #00d4aa !important; color: #060e1a !important;
    border-color: #00d4aa !important; font-weight: 600 !important;
}
[data-testid="stMetric"] {
    background: #0a1628; border: 1px solid #1a3050;
    border-radius: 10px; padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #7a9bb5 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e8f0f7 !important; font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
RISK_COLORS = {'Safe':'#00d4aa','Caution':'#EF9F27',
               'Danger':'#FF8C42','Extreme':'#D85A30'}
RISK_EMOJIS = {'Safe':'🟢','Caution':'🟡','Danger':'🟠','Extreme':'🔴'}
FEST_MONTHS = {1,3,4,8,10,11,12}

def get_risk_label(hi):
    if hi < 32:   return 'Safe'
    elif hi < 35: return 'Caution'
    elif hi < 38: return 'Danger'
    else:         return 'Extreme'

def calculate_health_risk(heat_index, humidity, visibility,
                           wind_speed, is_festival, profile):
    weights = risk_config['weights']
    ranges  = risk_config['ranges']
    mult    = risk_config['profile_multipliers'].get(profile, 1.0)
    def norm(v, mn, mx): return (v - mn) / (mx - mn + 1e-9)
    score = (
        norm(heat_index, *ranges['heat_index']) * weights['heat_index'] +
        norm(humidity,   *ranges['humidity'])   * weights['humidity']   +
        (1 - norm(visibility, *ranges['visibility'])) * weights['visibility'] +
        (1 - norm(wind_speed, *ranges['wind_speed'])) * weights['wind_speed'] +
        (1.0 if is_festival else 0.0)                 * weights['festival']
    ) * 100 * mult
    return round(min(score, 100), 1)

def plt_dark(fig, ax):
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0a1628')
    ax.tick_params(colors='#7a9bb5', labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#1a3050')
    ax.yaxis.grid(True, color='#1a3050', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

def get_forecast_30(start_date):
    rows = []
    for i in range(30):
        d     = start_date + datetime.timedelta(days=i)
        stats = monthly_stats.loc[d.month]
        rows.append({
            'date'           : d,
            'label'          : d.strftime('%b %d'),
            'heat_index_mean': stats['heat_index_mean'],
            'hi_upper'       : stats['hi_upper'],
            'hi_lower'       : stats['hi_lower'],
            'temp_mean'      : stats['temp_mean'],
            'risk_label'     : stats['risk_label'],
        })
    return pd.DataFrame(rows)

# Use polynomial projection values as they are more accurate
poly_temp_2030 = poly_trend_data['poly_temp_2030']
poly_r2        = poly_trend_data['poly_r2']

# ── Navigation ────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

nav_cols = st.columns([2.5, 1, 1, 1, 1])
with nav_cols[0]:
    st.markdown('<span style="font-family:Syne,sans-serif;font-size:1rem;'
                'font-weight:700;color:#00d4aa;">🌡️ Bangalore UHI Analyser</span>',
                unsafe_allow_html=True)

pages     = ['🏠 Home', '📅 Forecast', '🔍 Insights', '❤️ Health Risk']
page_keys = ['Home', 'Forecast', 'Insights', 'Health Risk']
for col, pg, key in zip(nav_cols[1:], pages, page_keys):
    with col:
        # Active page gets special key so CSS can target it
        btn_key = 'nav_active' if st.session_state.page == key else f'nav_{key}'
        if st.button(pg, key=btn_key, use_container_width=True):
            st.session_state.page = key

st.markdown('<hr style="border:none;border-top:1px solid #1a3050;margin:0.2rem 0 1.2rem;">',
            unsafe_allow_html=True)

page = st.session_state.page

# ════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════
if page == 'Home':

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="hero-title">Is Bangalore<br>Getting Hotter?</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">10 years of real weather data · '
                    'SDG 12 & SDG 13 · Climate action for Bangalore</p>',
                    unsafe_allow_html=True)

        # Quick stats — using polynomial projection (more accurate)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'''<div class="card" style="padding:0.8rem 1rem;">
                <div class="card-title">Avg Heat Index</div>
                <div class="card-val" style="font-size:1.4rem;">
                {summary["avg_heat_index"]}°C</div>
                <div class="card-sub">2014–2023 average</div>
            </div>''', unsafe_allow_html=True)
        with s2:
            st.markdown(f'''<div class="card" style="padding:0.8rem 1rem;">
                <div class="card-title">2030 Projection</div>
                <div class="card-val" style="font-size:1.4rem;color:#D85A30;">
                {poly_temp_2030}°C</div>
                <div class="card-sub">Polynomial model forecast</div>
            </div>''', unsafe_allow_html=True)
        with s3:
            st.markdown(f'''<div class="card" style="padding:0.8rem 1rem;">
                <div class="card-title">Extreme Days</div>
                <div class="card-val" style="font-size:1.4rem;color:#EF9F27;">
                {summary["extreme_days_pct"]}%</div>
                <div class="card-sub">Heat Index above 35°C</div>
            </div>''', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="sec-header">Analyse any day — get instant results</p>',
                    unsafe_allow_html=True)

        # ── Simple 3 inputs only ──────────────────────────
        sel_month = st.selectbox(
            "Which month do you want to check?",
            options=list(range(1, 13)),
            format_func=lambda x: MONTH_NAMES[x-1],
            key='home_month')

        profile = st.selectbox(
            "Who are you?",
            list(risk_config['profile_multipliers'].keys()),
            key='home_profile')

        is_festival = st.radio(
            "Is today a festival or public holiday?",
            options=[0, 1],
            format_func=lambda x: "Yes 🎉" if x == 1 else "No",
            horizontal=True,
            key='home_fest')

        analyse_btn = st.button("🔍 Analyse This Month", key='home_analyse')

    with right:
        if analyse_btn:
            # Get monthly averages automatically
            stats      = monthly_stats.loc[sel_month]
            hi_mean    = stats['heat_index_mean']
            humidity   = stats['humidity_mean']
            visibility = stats['visibility_mean']
            wind_speed = stats['wind_mean']
            temp       = stats['temp_mean']
            season     = (3 if sel_month in [3,4,5,6] else
                          2 if sel_month in [7,8,9] else
                          1 if sel_month in [10,11] else 0)

            # RF Heat Index prediction using monthly averages
            dew_point   = stats.get('dew_mean', 18.0) if 'dew_mean' in stats else 18.0
            cloud_cover = stats['cloud_mean']
            rf_input    = np.array([[temp, dew_point, humidity,
                                      wind_speed, 0.0, visibility,
                                      cloud_cover, sel_month, season,
                                      int(is_festival)]])
            rf_scaled = scaler_rf.transform(rf_input)
            hi_pred   = round(float(model_rf_hi.predict(rf_scaled)[0]), 1)

            # Health risk
            health_score = calculate_health_risk(
                hi_pred, humidity, visibility,
                wind_speed, is_festival, profile)
            risk       = get_risk_label(hi_pred)
            advice     = risk_config['profile_advice'][profile][risk]
            risk_color = RISK_COLORS[risk]
            alert_cls  = f'alert-{risk.lower()}'

            # Heat zone
            c_input = scaler_c.transform(
                np.array([[temp, hi_pred, humidity, sel_month]]))
            zone_id = model_kmeans.predict(c_input)[0]
            zone    = cluster_map.get(int(zone_id), 'Unknown')
            zone_colors = {'Cool':'#00d4aa','Warm':'#EF9F27',
                           'Hot':'#FF8C42','Extreme':'#D85A30'}
            zc = zone_colors.get(zone, '#7a9bb5')

            st.markdown('<p class="sec-header">Results</p>',
                        unsafe_allow_html=True)

            # Results — 2 columns
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f'''<div class="result-card">
                    <div class="result-val" style="color:{risk_color};">
                    {hi_pred}°C</div>
                    <div class="result-label">Predicted Heat Index</div>
                </div>''', unsafe_allow_html=True)
            with r2:
                st.markdown(f'''<div class="result-card">
                    <div class="result-val" style="color:{risk_color};
                    font-size:1.4rem;">{RISK_EMOJIS[risk]} {risk}</div>
                    <div class="result-label">Heat Risk Level</div>
                </div>''', unsafe_allow_html=True)

            r3, r4 = st.columns(2)
            with r3:
                st.markdown(f'''<div class="result-card">
                    <div class="result-val" style="color:{risk_color};
                    font-size:1.5rem;">{health_score}/100</div>
                    <div class="result-label">Health Risk Score</div>
                </div>''', unsafe_allow_html=True)
            with r4:
                st.markdown(f'''<div class="result-card">
                    <div class="result-val" style="color:{zc};
                    font-size:1.5rem;">{zone}</div>
                    <div class="result-label">Heat Zone (K-Means)</div>
                </div>''', unsafe_allow_html=True)

            # Advice
            st.markdown(f'''<div class="{alert_cls}">
                <strong style="color:{risk_color};">
                Advice for {profile}:</strong><br>
                <span style="font-size:0.88rem;">{advice}</span>
            </div>''', unsafe_allow_html=True)

            # Monthly context
            st.markdown(f'''<div class="card" style="padding:0.8rem 1rem;">
                <div class="card-title">
                {MONTH_NAMES[sel_month-1]} historical averages</div>
                <div style="display:flex;gap:1.5rem;margin-top:0.4rem;">
                    <div>
                        <div style="font-size:1rem;font-weight:600;
                        color:#e8f0f7;">{temp:.1f}°C</div>
                        <div style="font-size:0.72rem;color:#7a9bb5;">
                        Temperature</div>
                    </div>
                    <div>
                        <div style="font-size:1rem;font-weight:600;
                        color:#e8f0f7;">{humidity:.0f}%</div>
                        <div style="font-size:0.72rem;color:#7a9bb5;">
                        Humidity</div>
                    </div>
                    <div>
                        <div style="font-size:1rem;font-weight:600;
                        color:#e8f0f7;">{wind_speed:.1f} km/h</div>
                        <div style="font-size:0.72rem;color:#7a9bb5;">
                        Wind Speed</div>
                    </div>
                    <div>
                        <div style="font-size:1rem;font-weight:600;
                        color:#e8f0f7;">{visibility:.1f} km</div>
                        <div style="font-size:0.72rem;color:#7a9bb5;">
                        Visibility</div>
                    </div>
                </div>
            </div>''', unsafe_allow_html=True)

        else:
            st.markdown('''<div style="text-align:center;padding:5rem 1rem;
                color:#7a9bb5;">
                <div style="font-size:3.5rem;">🌡️</div>
                <p style="font-family:Syne,sans-serif;font-size:1.1rem;
                color:#e8f0f7;margin-top:1rem;">Select a month & your profile</p>
                <p style="font-size:0.85rem;margin-top:0.4rem;">Click
                <strong style="color:#00d4aa;">Analyse This Month</strong>
                to see Heat Index, Health Risk and Heat Zone</p>
            </div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# FORECAST PAGE
# ════════════════════════════════════════════════════════════
elif page == 'Forecast':

    st.markdown('<p class="sec-header">30-Day Heat Forecast — Bangalore</p>',
                unsafe_allow_html=True)

    start_date  = st.date_input(
        "Forecast start date",
        value=datetime.date.today(),
        min_value=datetime.date.today())

    forecast    = get_forecast_30(start_date)
    risk_counts = forecast['risk_label'].value_counts()
    peak_day    = forecast.loc[forecast['heat_index_mean'].idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("🟢 Safe Days",    risk_counts.get('Safe', 0))
    with m2: st.metric("🟡 Caution Days", risk_counts.get('Caution', 0))
    with m3: st.metric("🟠 Danger Days",  risk_counts.get('Danger', 0))
    with m4: st.metric("🔴 Extreme Days", risk_counts.get('Extreme', 0))

    peak_risk = get_risk_label(peak_day['heat_index_mean'])
    st.markdown(
        f'<div class="alert-{peak_risk.lower()}" style="margin:0.8rem 0;">'
        f'{RISK_EMOJIS[peak_risk]} <strong>Peak heat: '
        f'{peak_day["date"].strftime("%b %d")}</strong> — '
        f'Predicted Heat Index {peak_day["heat_index_mean"]:.1f}°C · {peak_risk}'
        f'</div>', unsafe_allow_html=True)

    # Forecast chart with confidence bands
    fig, ax = plt.subplots(figsize=(12, 4.5))
    plt_dark(fig, ax)
    x      = range(30)
    means  = forecast['heat_index_mean'].tolist()
    uppers = forecast['hi_upper'].tolist()
    lowers = forecast['hi_lower'].tolist()
    colors = [RISK_COLORS[r] for r in forecast['risk_label']]

    ax.fill_between(x, lowers, uppers, alpha=0.12,
                    color='#7ab8e8', label='Confidence band (±1 std)')
    ax.plot(x, means, color='#00d4aa', linewidth=2, zorder=3)
    for xi, yi, ci in zip(x, means, colors):
        ax.scatter(xi, yi, color=ci, s=40, zorder=4,
                   edgecolors='#0a1628', linewidth=0.5)
    ax.axhline(y=35, color='#D85A30', linestyle='--',
               linewidth=1, alpha=0.8, label='Extreme heat threshold (35°C)')
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(forecast['label'].tolist()[::3],
                       rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Heat Index (°C)', color='#7a9bb5', fontsize=9)
    ax.set_title(f'30-Day Heat Forecast from {start_date.strftime("%B %d, %Y")}',
                 color='#e8f0f7', fontsize=10, pad=10)
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in RISK_COLORS.items()]
    ax.legend(handles=patches, fontsize=7.5, facecolor='#0a1628',
              labelcolor='#7a9bb5', edgecolor='#1a3050', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Simple text summary instead of table
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">What to expect this month</p>',
                unsafe_allow_html=True)
    dominant_risk = risk_counts.index[0]
    dominant_color = RISK_COLORS[dominant_risk]
    st.markdown(f'''<div class="card">
        <p style="color:#7a9bb5;font-size:0.9rem;line-height:1.8;margin:0;">
        Over the next 30 days starting
        <strong style="color:#e8f0f7;">
        {start_date.strftime("%B %d")}</strong>,
        Bangalore is expected to see mostly
        <strong style="color:{dominant_color};">{dominant_risk}</strong>
        heat conditions.
        The peak heat day is
        <strong style="color:#D85A30;">
        {peak_day["date"].strftime("%B %d")}</strong>
        with a predicted Heat Index of
        <strong style="color:#D85A30;">
        {peak_day["heat_index_mean"]:.1f}°C</strong>.
        Safe days: {risk_counts.get("Safe", 0)} ·
        Caution days: {risk_counts.get("Caution", 0)} ·
        Danger days: {risk_counts.get("Danger", 0)} ·
        Extreme days: {risk_counts.get("Extreme", 0)}.
        </p>
    </div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# INSIGHTS PAGE
# ════════════════════════════════════════════════════════════
elif page == 'Insights':

    st.markdown('<p class="sec-header">Climate Insights — Bangalore 2014–2030</p>',
                unsafe_allow_html=True)

    tab_climate, tab_festival, tab_models = st.tabs([
        '📈 Climate Story', '🎉 Festival Impact', '🤖 Model Comparison'
    ])

    # ── Climate Story ─────────────────────────────────────────
    with tab_climate:

        c1, c2, c3, c4 = st.columns(4)
        items = [
            ('Avg Heat Index',    f"{summary['avg_heat_index']}°C",  '2014–2023'),
            ('2030 Temp (Poly)',  f"{poly_temp_2030}°C",             'Polynomial projection'),
            ('2030 Heat Index',   f"{summary['hi_2030']}°C",         'Projected'),
            ('Extreme Days',      f"{summary['extreme_days_pct']}%", 'Heat Index > 35°C'),
        ]
        for col, (t, v, s) in zip([c1,c2,c3,c4], items):
            with col:
                st.markdown(f'''<div class="card">
                    <div class="card-title">{t}</div>
                    <div class="card-val" style="font-size:1.3rem;">{v}</div>
                    <div class="card-sub">{s}</div>
                </div>''', unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            # Linear vs Polynomial comparison
            fig, ax = plt.subplots(figsize=(6, 3.8))
            plt_dark(fig, ax)
            yearly = trend_data['yearly_temp']
            future = trend_data['future_years']
            lin_p  = trend_data['temp_projection']
            poly_p = poly_trend_data['poly_projection']

            ax.scatter(yearly['year'], yearly['Temperature'],
                       color='#e8f0f7', s=40, zorder=3, label='Actual data')
            ax.plot(future, lin_p, color='#00d4aa', linewidth=1.8,
                    label=f"Linear (R²={trend_data['lr_r2']})")
            ax.plot(future, poly_p, color='#EF9F27', linestyle='--',
                    linewidth=1.8,
                    label=f"Polynomial (R²={poly_trend_data['poly_r2']})")
            ax.set_ylabel('Temperature (°C)', color='#7a9bb5', fontsize=8)
            ax.set_title('Linear vs Polynomial Regression — 2030 Projection',
                         color='#e8f0f7', fontsize=9, pad=8)
            ax.legend(fontsize=7.5, facecolor='#0a1628',
                      labelcolor='#7a9bb5', edgecolor='#1a3050')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(f"Polynomial Regression fits better (R²={poly_r2}) — "
                       f"projects Bangalore reaching {poly_temp_2030}°C by 2030")

        with cr:
            # Monthly heat pattern
            fig, ax = plt.subplots(figsize=(6, 3.8))
            plt_dark(fig, ax)
            monthly_hi = df.groupby('month')['Heat Index'].mean()
            bar_cols   = [RISK_COLORS[get_risk_label(v)] for v in monthly_hi]
            ax.bar(MONTH_NAMES, monthly_hi.values,
                   color=bar_cols, edgecolor='none')
            ax.axhline(y=35, color='#D85A30', linestyle='--',
                       linewidth=1, alpha=0.8, label='Extreme threshold (35°C)')
            ax.set_ylabel('Avg Heat Index (°C)', color='#7a9bb5', fontsize=8)
            ax.set_title('Monthly Heat Index — Bangalore (2014–2023)',
                         color='#e8f0f7', fontsize=9, pad=8)
            patches = [mpatches.Patch(color=c, label=l)
                       for l, c in RISK_COLORS.items()]
            ax.legend(handles=patches, fontsize=7.5, facecolor='#0a1628',
                      labelcolor='#7a9bb5', edgecolor='#1a3050')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # SDG story — using polynomial numbers
        st.markdown(f'''<div class="card">
            <div class="card-title">SDG 13 — Climate Action</div>
            <p style="color:#7a9bb5;font-size:0.88rem;line-height:1.8;margin:0;">
            Bangalore's temperature trend shows a clear warming pattern when
            modelled with Polynomial Regression (R²={poly_r2}), which captures
            the curve of urban growth better than a straight line.
            At this rate, Bangalore's average temperature could reach
            <strong style="color:#D85A30;">{poly_temp_2030}°C by 2030</strong>.
            Extreme heat days already account for
            <strong style="color:#e8f0f7;">{summary["extreme_days_pct"]}%</strong>
            of all days — a direct consequence of urban heat island effect,
            irresponsible consumption (SDG 12), and climate change (SDG 13).
            </p>
        </div>''', unsafe_allow_html=True)

    # ── Festival Impact ───────────────────────────────────────
    with tab_festival:

        fest = festival_data

        # Only show if festival impact is meaningful
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(f'''<div class="card">
                <div class="card-title">Festival Days HI</div>
                <div class="card-val" style="color:#EF9F27;">
                {fest["fest_avg_hi"]}°C</div>
                <div class="card-sub">{fest["fest_count"]} festival days</div>
            </div>''', unsafe_allow_html=True)
        with f2:
            st.markdown(f'''<div class="card">
                <div class="card-title">Normal Days HI</div>
                <div class="card-val" style="color:#00d4aa;">
                {fest["non_fest_avg_hi"]}°C</div>
                <div class="card-sub">{fest["non_fest_count"]} normal days</div>
            </div>''', unsafe_allow_html=True)
        with f3:
            impact_val  = fest["fest_impact"]
            impact_color = '#D85A30' if impact_val > 0 else '#00d4aa'
            impact_sign  = '+' if impact_val > 0 else ''
            st.markdown(f'''<div class="card">
                <div class="card-title">Overall Impact</div>
                <div class="card-val" style="color:{impact_color};">
                {impact_sign}{impact_val}°C</div>
                <div class="card-sub">festival vs normal days</div>
            </div>''', unsafe_allow_html=True)

        fl, fr = st.columns(2)
        with fl:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            plt_dark(fig, ax)
            ax.bar(['Festival Days', 'Normal Days'],
                   [fest['fest_avg_hi'], fest['non_fest_avg_hi']],
                   color=['#EF9F27', '#00d4aa'],
                   edgecolor='none', width=0.4)
            ax.set_ylabel('Avg Heat Index (°C)', color='#7a9bb5', fontsize=8)
            ax.set_title('Festival vs Normal Days Heat Index',
                         color='#e8f0f7', fontsize=9, pad=8)
            for i, v in enumerate([fest['fest_avg_hi'], fest['non_fest_avg_hi']]):
                ax.text(i, v + 0.05, f'{v}°C',
                        ha='center', color='#e8f0f7', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with fr:
            # Only show months with valid (non-NaN) impact
            monthly_fest = fest['monthly_fest'].dropna()
            # Only positive impact months for SDG 12 story
            positive_months = monthly_fest[monthly_fest['Impact'] > 0]

            if not positive_months.empty:
                month_labels = [MONTH_NAMES[m-1] for m in positive_months.index]
                fig, ax = plt.subplots(figsize=(5, 3.5))
                plt_dark(fig, ax)
                ax.bar(month_labels, positive_months['Impact'].values,
                       color='#D85A30', edgecolor='none')
                ax.set_ylabel('Heat Impact (°C)', color='#7a9bb5', fontsize=8)
                ax.set_title('Months Where Festivals Raise Heat',
                             color='#e8f0f7', fontsize=9, pad=8)
                for i, v in enumerate(positive_months['Impact'].values):
                    ax.text(i, v + 0.01, f'+{v}°C',
                            ha='center', color='#e8f0f7', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown(f'''<div class="card">
            <div class="card-title">SDG 12 — Responsible Consumption</div>
            <p style="color:#7a9bb5;font-size:0.88rem;line-height:1.8;margin:0;">
            During major Bangalore festivals — Ugadi, Diwali, Ganesh Chaturthi,
            Kannada Rajyotsava, IPL matches — the city sees measurably different
            heat patterns. Excess lighting, firecrackers, generators, and heavy
            traffic are all forms of irresponsible energy consumption that
            contribute to the Urban Heat Island effect.
            This is the SDG 12 story told through data.
            </p>
        </div>''', unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────
    with tab_models:

        st.markdown('<p class="sec-header">Algorithm Performance Comparison</p>',
                    unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        fi = feature_importance

        with mc1:
            # Regression model comparison
            fig, ax = plt.subplots(figsize=(5, 3.2))
            plt_dark(fig, ax)
            model_names = ['Linear\nRegression', 'Polynomial\nRegression',
                           'Random Forest\nRegressor']
            r2_vals = [trend_data['lr_r2'], poly_r2, fi['rf_r2']]
            bar_c   = ['#1a3050', '#EF9F27', '#00d4aa']
            bars = ax.bar(model_names, r2_vals, color=bar_c, edgecolor='none')
            ax.set_ylim(0, 1)
            ax.set_ylabel('R² Score', color='#7a9bb5', fontsize=8)
            ax.set_title('Regression Model Comparison (R² Score)',
                         color='#e8f0f7', fontsize=9, pad=8)
            for i, v in enumerate(r2_vals):
                ax.text(i, v + 0.01, f'{v:.3f}',
                        ha='center', color='#e8f0f7', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(f"Random Forest best for prediction (R²={fi['rf_r2']}) · "
                       f"CV Mean R²={fi['cv_mean']} ± {fi['cv_std']} (5-fold)")

        with mc2:
            # Feature importance
            imp_series = pd.Series(fi['importances']).sort_values()
            fig, ax    = plt.subplots(figsize=(5, 3.2))
            plt_dark(fig, ax)
            bar_colors = ['#00d4aa' if i == len(imp_series)-1
                          else '#1a3050' for i in range(len(imp_series))]
            ax.barh(imp_series.index, imp_series.values,
                    color=bar_colors, edgecolor='none')
            ax.set_xlabel('Importance Score', color='#7a9bb5', fontsize=8)
            ax.set_title('What drives Heat Index the most?',
                         color='#e8f0f7', fontsize=9, pad=8)
            for i, v in enumerate(imp_series.values):
                ax.text(v + 0.001, i, f'{v:.3f}',
                        va='center', color='#e8f0f7', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Temperature is the dominant feature — "
                       "followed by Dew Point and Humidity")

        # CV scores chart
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="sec-header">5-Fold Cross Validation — '
                    'Random Forest Regressor</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        plt_dark(fig, ax)
        cv_scores = fi['cv_scores']
        fold_names = [f'Fold {i+1}' for i in range(len(cv_scores))]
        ax.bar(fold_names, cv_scores, color='#00d4aa',
               edgecolor='none', width=0.4)
        ax.axhline(y=fi['cv_mean'], color='#EF9F27', linestyle='--',
                   linewidth=1.5, label=f"Mean R²={fi['cv_mean']}")
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel('R² Score', color='#7a9bb5', fontsize=8)
        ax.set_title('Cross Validation Scores — consistent across all folds',
                     color='#e8f0f7', fontsize=9, pad=8)
        ax.legend(fontsize=8, facecolor='#0a1628',
                  labelcolor='#7a9bb5', edgecolor='#1a3050')
        for i, v in enumerate(cv_scores):
            ax.text(i, v + 0.001, f'{v:.3f}',
                    ha='center', color='#e8f0f7', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(f"R² stays between {min(cv_scores):.3f} and "
                   f"{max(cv_scores):.3f} across all folds — "
                   f"model is stable and not overfitting")

# ════════════════════════════════════════════════════════════
# HEALTH RISK PAGE
# ════════════════════════════════════════════════════════════
elif page == 'Health Risk':

    st.markdown('<p class="sec-header">Personalised Daily Health Risk</p>',
                unsafe_allow_html=True)

    hl, hr = st.columns([1, 1.5])

    with hl:
        hr_profile = st.selectbox(
            "Who are you?",
            list(risk_config['profile_multipliers'].keys()),
            key='hr_profile')

        hr_date  = st.date_input("Which day?",
                                  value=datetime.date.today(),
                                  key='hr_date')
        hr_month = hr_date.month
        hr_stats = monthly_stats.loc[hr_month]
        hr_hi    = float(hr_stats['heat_index_mean'])
        hr_hum   = float(hr_stats['humidity_mean'])
        hr_vis   = float(hr_stats['visibility_mean'])
        hr_wind  = float(hr_stats['wind_mean'])

        custom = st.toggle("Customise weather values", value=False)
        if custom:
            hr_hi   = st.slider("Heat Index (°C)", 25.0, 45.0, hr_hi, 0.1)
            hr_hum  = st.slider("Humidity (%)", 10.0, 99.0, hr_hum, 0.5)
            hr_vis  = st.slider("Visibility (km)", 0.6, 6.2, hr_vis, 0.1)
            hr_wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, hr_wind, 0.5)

        hr_fest = 1 if hr_month in FEST_MONTHS else 0
        hr_btn  = st.button("Calculate My Risk", key='hr_btn')

    with hr:
        if hr_btn:
            score   = calculate_health_risk(hr_hi, hr_hum, hr_vis,
                                             hr_wind, hr_fest, hr_profile)
            risk    = get_risk_label(hr_hi)
            advice  = risk_config['profile_advice'][hr_profile][risk]
            color   = RISK_COLORS[risk]
            alert_c = f'alert-{risk.lower()}'

            st.markdown(f'''<div class="card" style="text-align:center;
                margin-bottom:1rem;">
                <div class="card-title">Your Health Risk Score</div>
                <div style="font-family:Syne,sans-serif;font-size:4.5rem;
                font-weight:800;color:{color};line-height:1;">{score}</div>
                <div style="font-size:1rem;color:{color};margin-bottom:0.3rem;">
                / 100</div>
                <div style="font-size:0.85rem;color:#7a9bb5;">
                {RISK_EMOJIS[risk]} {risk} · {hr_profile}</div>
            </div>''', unsafe_allow_html=True)

            st.markdown(f'''<div class="{alert_c}">
                <strong style="color:{color};">What this means for you:</strong><br>
                <span style="font-size:0.88rem;">{advice}</span>
            </div>''', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            hm1, hm2, hm3, hm4 = st.columns(4)
            with hm1: st.metric("Heat Index",  f"{hr_hi:.1f}°C")
            with hm2: st.metric("Humidity",    f"{hr_hum:.0f}%")
            with hm3: st.metric("Visibility",  f"{hr_vis:.1f} km")
            with hm4: st.metric("Wind Speed",  f"{hr_wind:.1f} km/h")

            st.markdown('<br>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4 = st.columns(4)
            scale = [
                ('🟢 Safe',    '< 32°C',  '#00d4aa'),
                ('🟡 Caution', '32–35°C', '#EF9F27'),
                ('🟠 Danger',  '35–38°C', '#FF8C42'),
                ('🔴 Extreme', '> 38°C',  '#D85A30'),
            ]
            for col, (label, rng, c) in zip([sc1,sc2,sc3,sc4], scale):
                with col:
                    bw = '2px' if label.split()[1] == risk else '1px'
                    st.markdown(f'''<div class="card"
                        style="border-color:{c};border-width:{bw};
                        text-align:center;padding:0.7rem;">
                        <div style="color:{c};font-weight:600;
                        font-size:0.85rem;">{label}</div>
                        <div style="color:#7a9bb5;font-size:0.75rem;">
                        {rng}</div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div style="text-align:center;padding:4rem 1rem;
                color:#7a9bb5;">
                <div style="font-size:3rem;">❤️</div>
                <p style="font-family:Syne,sans-serif;font-size:1rem;
                color:#e8f0f7;margin-top:0.8rem;">Select your profile</p>
                <p style="font-size:0.85rem;">Choose who you are and click
                <strong style="color:#00d4aa;">Calculate My Risk</strong></p>
            </div>''', unsafe_allow_html=True)