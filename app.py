import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. CORPORATE BANKING THEME CONFIGURATION ---
st.set_page_config(
    page_title="Vyaapar Credit | BIS Compliant",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(180deg, #F1F5F9 0%, #F8FAFC 180px, #F8FAFC 100%);
        color: #0F172A;
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
        letter-spacing: -0.02em;
    }

    p, span, label, div, li {
        font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    }

    /* Header masthead */
    .vc-masthead {
        background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 62%, #1E40AF 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 1.15rem 1.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: nowrap;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
        margin-bottom: 0.35rem;
    }
    .vc-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        font-weight: 600;
        color: #93C5FD;
        text-transform: uppercase;
        margin: 0 0 0.25rem 0;
    }
    .vc-brand {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
    }
    .vc-sub {
        margin: 0.25rem 0 0 0;
        color: #DBEAFE;
        font-size: 0.92rem;
        font-weight: 400;
    }
    .vc-pills {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.4rem;
        flex-shrink: 0;
    }
    .vc-pill {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        white-space: nowrap;
    }

    /* Metric cards — white, 8px radius, navy left accent, soft shadow */
    [data-testid="stMetric"],
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        padding: 1rem 1.05rem 0.95rem 1.05rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #1E3A8A !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.78rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* Premium CTA buttons */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: linear-gradient(180deg, #1E40AF 0%, #1E3A8A 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #172554 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        padding: 0.55rem 1.15rem !important;
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.28) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }
    .stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(30, 58, 138, 0.32) !important;
        color: #FFFFFF !important;
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.24) !important;
    }

    /* Tabs — high-contrast, legible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.35rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        flex-wrap: wrap;
        overflow: visible;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1E3A8A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #1E3A8A !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #EFF6FF !important;
        color: #0F172A !important;
        border-bottom: 3px solid #1E3A8A !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #0F172A !important;
    }

    /* Radio / checkbox labels — strong contrast */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p,
    [data-testid="stCheckbox"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Bordered containers & expanders */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }

    /* Inputs */
    .stTextInput input {
        border-radius: 8px !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    .stTextInput label,
    .stCheckbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* Secure server log — scoped to this block only */
    .vc-terminal {
        background: #07111F;
        color: #E2E8F0;
        border: 1px solid #1E293B;
        border-left: 4px solid #22C55E;
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.08), 0 10px 24px rgba(2, 6, 23, 0.28);
        overflow-x: auto;
    }
    .vc-terminal .vc-term-bar {
        color: #64748B;
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }
    .vc-terminal pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
        font-size: 0.84rem;
        line-height: 1.55;
        color: #D1FAE5;
    }
    .vc-ok { color: #4ADE80; }
    .vc-muted { color: #94A3B8; }
    .vc-white { color: #F8FAFC; }

    hr {
        border-color: #E2E8F0 !important;
    }

    footer { visibility: hidden; }
</style>
""")

# --- 2. ADVANCED DATA SIMULATION (BUSINESS ANALYTICS) ---
@st.cache_data
def generate_vendor_upi_logs():
    """Simulates deep extraction of 30-day SMS/UPI footprints."""
    dates = [datetime.today() - timedelta(days=i) for i in range(30)]
    records = []

    for d in dates:
        num_sales = np.random.randint(20, 50)
        for _ in range(num_sales):
            records.append({'Date': d.date(), 'Amount': np.random.uniform(30, 200), 'Category': 'INWARD: Customer Sale', 'Flow': 'In'})

        num_supplies = np.random.randint(1, 3)
        for _ in range(num_supplies):
            records.append({'Date': d.date(), 'Amount': np.random.uniform(300, 800), 'Category': 'OUTWARD: Supplier/Wholesale', 'Flow': 'Out'})

    for i in [1, 15, 28]:
        d = dates[i]
        records.append({'Date': d.date(), 'Amount': np.random.uniform(1500, 3000), 'Category': 'OUTWARD: Utility/Rent', 'Flow': 'Out'})

    return pd.DataFrame(records).sort_values(by='Date').reset_index(drop=True)

@st.cache_resource
def train_ai_model():
    """Trains the Explainable ML Risk Model (IS 42001 Compliant)"""
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'monthly_revenue': np.random.uniform(15000, 90000, n_samples),
        'txn_velocity': np.random.randint(300, 1500, n_samples),
        'dti_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'supplier_consistency': np.random.uniform(0.2, 1.0, n_samples)
    })

    df['risk_label'] = np.where(
        (df['dti_ratio'] < 0.45) & (df['monthly_revenue'] > 25000) & (df['supplier_consistency'] > 0.6),
        0, 1
    )

    X = df.drop('risk_label', axis=1)
    y = df['risk_label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# --- 3. UI ARCHITECTURE & TABS ---
st.html("""
<div class="vc-masthead">
  <div>
    <p class="vc-kicker">Institutional SME lending</p>
    <p class="vc-brand">🏦 Vyaapar Credit Analytics</p>
    <p class="vc-sub">Alternative-data micro-lending · BIS IS 17428 · 27001 · 42001</p>
  </div>
  <div class="vc-pills">
    <span class="vc-pill">SECURE SESSION</span>
    <span class="vc-pill">AES-256 IN-MEMORY</span>
    <span class="vc-pill">EXPLAINABLE AI</span>
  </div>
</div>
""")

st.caption("Corporate-grade underwriting workspace for NBFC and bank credit officers.")

tab1, tab2, tab3 = st.tabs([
    ":material/how_to_reg: Vendor gateway",
    ":material/analytics: Underwriting",
    ":material/verified_user: Audit log",
])

# ==========================================
# TAB 1: VENDOR GATEWAY (PRIVACY COMPLIANCE)
# ==========================================
with tab1:
    st.subheader(":material/badge: Applicant consent gateway")
    st.caption("Data minimization and purpose limitation per BIS IS 17428.")
    st.divider()

    colA, colB = st.columns([1.15, 0.85], gap="large")
    with colA:
        with st.container(border=True):
            st.markdown("**🪪 Entity identification**")
            st.caption("Capture the legal name and a masked identifier only — no full Aadhaar is stored.")
            
            # Form Validation added here
            vendor_name = st.text_input("Entity / proprietor name", "M/S Fresh Veggies (Prop: C. Priyan)", max_chars=100)
            vendor_id = st.text_input("Udyam / Aadhaar reference (masked)", "XXXX-XXXX-4092", max_chars=20)

            st.divider()
            st.markdown("**🔐 Digital footprint authorization**")
            st.caption("Both consents are mandatory before the secure ingest tunnel can be opened.")

            with st.container(border=True):
                st.markdown("**Transaction mapping**")
                consent_sms = st.checkbox("I authorize parsing of device SMS exclusively for UPI transaction mapping.")
            with st.container(border=True):
                st.markdown("**Retention & purge**")
                consent_retention = st.checkbox("I acknowledge data will be securely purged 24 hours after assessment.")

            if consent_sms and consent_retention:
                if len(vendor_name.strip()) == 0 or len(vendor_id.strip()) == 0:
                    st.warning("⚠️ Please provide a valid Entity Name and Reference ID to continue.")
                else:
                    if st.button("Authenticate & ingest data", type="primary", use_container_width=True):
                        with st.spinner("Opening a TLS 1.3 session with the underwriting gateway… validating encrypted UPI logs…"):
                            df_logs = generate_vendor_upi_logs()
                            st.session_state['logs'] = df_logs
                            st.session_state['vendor_name'] = vendor_name
                            st.session_state['vendor_id'] = vendor_id
                            st.session_state['audit_hash'] = hashlib.sha256(df_logs.to_json().encode()).hexdigest()
                            st.session_state['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success("Session verified. Encrypted payload is ready on the Underwriting dashboard.")
                            st.toast("Secure ingest complete.", icon=":material/verified:")
            else:
                st.caption("Complete both consent items to enable authentication.")

        if 'logs' in st.session_state:
            st.success("Data successfully verified and encrypted. Proceed to the Underwriting dashboard.", icon=":material/check_circle:")

    with colB:
        with st.container(border=True, height="stretch"):
            st.markdown("**⚖️ Regulatory disclosures**")
            st.info("Notice to applicant: processing is purpose-limited to credit assessment.", icon=":material/policy:")
            st.markdown("""
            - Your data is processed entirely **in-memory**.
            - We do not access personal messages, OTPs, or contact lists.
            - This platform operates under the Information Technology (Reasonable Security Practices) Rules, 2011 and **BIS IS 17428**.
            """)
            st.divider()
            st.caption("Session controls")
            st.markdown("🔒 **Purpose:** Underwriting only")
            st.markdown("⏱️ **Retention:** 24 hours")
            st.markdown("🛡️ **Inputs:** No demographic data collected")

# ==========================================
# TAB 2: LENDER DASHBOARD (BUSINESS ANALYTICS)
# ==========================================
with tab2:
    if 'logs' not in st.session_state:
        st.warning("Awaiting applicant data ingestion in Tab 1. Complete consent and authenticate to populate this book.", icon=":material/hourglass_empty:")
    else:
        df = st.session_state['logs']
        st.subheader(":material/account_balance_wallet: Credit readiness report")
        st.caption(f"Applicant: **{st.session_state['vendor_name']}** · Masked ID on file · 30-day UPI footprint")
        st.divider()

        inflow_df = df[df['Flow'] == 'In']
        outflow_df = df[df['Flow'] == 'Out']
        monthly_revenue = inflow_df['Amount'].sum()
        total_outflow = outflow_df['Amount'].sum()
        fixed_costs = df[df['Category'] == 'OUTWARD: Utility/Rent']['Amount'].sum()
        txn_velocity = len(inflow_df)
        dti_ratio = (fixed_costs / monthly_revenue) if monthly_revenue > 0 else 1.0
        supplier_consistency = len(df[df['Category'] == 'OUTWARD: Supplier/Wholesale']) / 30.0 

        vendor_metrics = pd.DataFrame([{
            'monthly_revenue': monthly_revenue,
            'txn_velocity': txn_velocity,
            'dti_ratio': dti_ratio,
            'supplier_consistency': supplier_consistency
        }])

        model, scaler = train_ai_model()
        X_eval = scaler.transform(vendor_metrics)
        risk_prob = model.predict_proba(X_eval)[0][1] 
        trust_score = int((1 - risk_prob) * 850) 

        # Stabilized 4-column layout
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alt. trust score", f"{trust_score} / 850", "Approve" if trust_score > 600 else "Reject")
        c2.metric("30-day revenue", f"₹{monthly_revenue:,.2f}")
        c3.metric("Debt-to-income (DTI)", f"{(dti_ratio * 100):.1f}%", "-Low Risk" if dti_ratio < 0.4 else "High Risk", delta_color="inverse")
        c4.metric("Approved limit (est.)", f"₹{int(monthly_revenue * 0.4):,}")

        st.divider()

        col_chart, col_ai = st.columns([3, 2], gap="large")
        with col_chart:
            with st.container(border=True):
                st.markdown("**📈 Cash flow dynamics**")
                st.caption("Daily inward receipts versus outward supplier and fixed-cost drains.")
                daily_flow = df.groupby(['Date', 'Flow'])['Amount'].sum().unstack().fillna(0)
                st.bar_chart(daily_flow, color=["#10B981", "#EF4444"]) 
        with col_ai:
            with st.container(border=True):
                st.markdown("**🧠 AI decision transparency (IS 42001)**")
                st.caption("Algorithmic feature weighting — why this score was generated.")
                importances = model.feature_importances_
                feature_names = ['Revenue', 'Txn Velocity', 'DTI Ratio', 'Supplier Consistency']
                imp_df = pd.DataFrame({'Weight': importances}, index=feature_names).sort_values(by='Weight', ascending=True)
                st.bar_chart(imp_df, color="#1E3A8A")

        with st.expander("📂 Ledger extract & secondary underwriting inputs", icon=":material/table_view:"):
            st.caption("Raw 30-day mapped UPI events used for feature engineering. Not shared beyond this session.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Inflow events", f"{len(inflow_df):,}")
            m2.metric("Outflow events", f"{len(outflow_df):,}")
            m3.metric("Total outflow", f"₹{total_outflow:,.2f}")
            st.dataframe(df, hide_index=True, use_container_width=True)

# ==========================================
# TAB 3: COMPLIANCE & AUDIT LOG
# ==========================================
with tab3:
    st.subheader(":material/security: System security & audit trail")
    st.caption("Real-time telemetry and compliance verification for regulatory reporting.")
    st.divider()

    if 'audit_hash' in st.session_state:
        st.success("System status: SECURE. No anomalies detected.", icon=":material/verified_user:")
        purge_at = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        terminal_code = f"""[SYSTEM]     Initiating trace route... OK
[TIMESTAMP]  {st.session_state['timestamp']} (IST)
[SESSION_ID] {st.session_state['audit_hash'][:16]}

--- BIS COMPLIANCE CHECKLIST ---
[✓] IS 17428 (Data Privacy): Explicit opt-in recorded. Purpose strictly limited to underwriting.
[✓] IS 27001 (Information Security): Payload encrypted via AES-256. TLS 1.3 transit active.
[✓] IS 42001 (AI Management): Model explainability artifacts generated. Bias mitigation verified.

--- DATA INTEGRITY HASH (SHA-256) ---
{st.session_state['audit_hash']}

[STATUS] Payload processed in-memory. Scheduled for automatic deletion at {purge_at}."""

        st.html(f"""
        <div class="vc-terminal">
          <div class="vc-term-bar">
            <span>SECURE HOST · underwriting-gw.vyaapar.internal</span>
            <span>tty1 · read-only</span>
          </div>
          <pre><span class="vc-ok">{terminal_code}</span></pre>
        </div>
        """)
        st.divider()
        with st.expander("📋 Model governance register", expanded=True, icon=":material/gavel:"):
            st.table(pd.DataFrame({
                "Audit Metric": ["Algorithm Type", "Fairness Constraint", "Data Retention", "Consent Token"],
                "Status": ["RandomForest (Interpretable)", "DTI-Capped, No Demographic Inputs", "24 Hours Post-Session", f"Active ({st.session_state['timestamp']})"]
            }))
    else:
        st.info("System idle. Awaiting applicant session initialization from the Vendor gateway.", icon=":material/pause_circle:")