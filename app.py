import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64

# Set page configuration
st.set_page_config(page_title="🧠 ML Mini Project: Customer Segmentation", layout="wide")

# Title
st.title("🧠 ML Mini Project: Customer Segmentation")

# =========================
# Customer Segmentation
# =========================
tab1, = st.tabs(["📊 Customer Segmentation"])

with tab1:
    st.header("Customer Segmentation (K-Means Clustering)")

    uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"], key="customer")
    if uploaded_file is None:
        st.warning("Please upload a customer data CSV file.")
        st.stop()

    try:
        customer_df = pd.read_csv(uploaded_file)
        st.subheader("Raw Customer Data")
        st.dataframe(customer_df)

        # Select features
        features = st.multiselect(
            "Select features for clustering",
            customer_df.columns.tolist(),
            default=["Annual Income (k$)", "Spending Score (1-100)"] if 
            "Annual Income (k$)" in customer_df.columns else customer_df.columns[:2]
        )
        if len(features) < 2:
            st.warning("Please select at least two features for clustering.")
            st.stop()

        # Scale features
        X = customer_df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow Method
        st.subheader("Elbow Method for Optimal Clusters")
        inertias = []
        K = range(1, 11)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots()
        sns.lineplot(x=list(K), y=inertias, ax=ax)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)

        # Select K
        k = st.slider("Select number of clusters (K)", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Plot clusters
        st.subheader("Customer Segmentation Clusters")
        fig, ax = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette('tab10', k)
        for cluster in range(k):
            cluster_data = customer_df[customer_df['Cluster'] == cluster]
            ax.scatter(
                cluster_data[features[0]],
                cluster_data[features[1]],
                label=f"Cluster {cluster + 1}",
                s=50,
                color=palette[cluster]
            )

        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='magenta', marker='X', label='Centroids')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.legend()
        st.pyplot(fig)

        # Cluster summary
        st.subheader("Cluster Summary")
        cluster_summary = customer_df.groupby('Cluster')[features].mean().round(2)
        st.dataframe(cluster_summary)

        # Display clustered data
        st.subheader("Clustered Data")
        st.dataframe(customer_df)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# =========================
# Sidebar: PDF Report
# =========================
st.sidebar.header("📄 Export Report")

if 'cluster_summary' not in locals():
    st.sidebar.warning("⚠️ Please complete clustering before generating the report.")
    st.stop()

if st.sidebar.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Customer Segmentation Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Number of Clusters: {k}", ln=True)
    pdf.ln(5)

    for cluster in range(k):
        pdf.cell(200, 8, txt=f"Cluster {cluster+1}:", ln=True)
        for col in features:
            avg = cluster_summary.loc[cluster][col]
            pdf.cell(200, 8, txt=f" - Avg {col}: {avg}", ln=True)
        pdf.ln(4)

    pdf.output("Customer_Segmentation_Report.pdf")

    with open("Customer_Segmentation_Report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="Customer_Segmentation_Report.pdf">📥 Download Report</a>'
        st.sidebar.markdown(download_link, unsafe_allow_html=True)
    st.sidebar.success("✅ Report Generated Successfully!")


