# 🧠 Customer Segmentation using K-Means Clustering

This project performs **Customer Segmentation** using **K-Means Clustering** and provides visual analysis through an interactive **Streamlit Dashboard**.  
Users can upload a dataset, select features, analyze clusters, and export a **PDF report** of the results.

---

## 📌 Features

| Feature | Description |
|--------|-------------|
| **CSV Upload** | Upload your customer dataset in `.csv` format |
| **Feature Selection** | Choose the features used to form clusters |
| **Standard Scaling** | The data is normalized before clustering |
| **Elbow Method** | Helps determine the optimal number of clusters |
| **K-Means Clustering** | Creates customer groups based on similarity |
| **Cluster Visualization** | Plot clusters and centroids on a scatter plot |
| **Cluster Summary** | View per-cluster average feature values |
| **PDF Export** | Download a professional segmentation report |

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib / Seaborn
- FPDF (for PDF Report)

---

## 📂 Project Setup

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```
2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
---

## 📁 Dataset Format

Your dataset should be a CSV file containing numeric customer attributes such as:

| CustomerID | Age | Annual Income (k$) | Spending Score (1-100) |
|------------|-----|-------------------|------------------------|
| 001        | 19  | 15                | 39                     |
| 002        | 21  | 15                | 81                     |

> You may select **any 2+ numeric columns** during runtime.


## 🎯 How It Works
- Upload dataset  
- Select the features (e.g., Income & Spending Score)  
- View Elbow Curve and choose best number of clusters K  
- Visualize cluster scatter plot with centroids  
- Analyze cluster summary row-wise  
- Export PDF Report (optional)

  
## 📄 Sample Output (Cluster Interpretation)

Cluster 1 → Low income, low spending  
Cluster 2 → High income, high spending  
Cluster 3 → High income, low spending  
Cluster 4 → Low income, high spending  

This helps businesses identify different customer types and plan targeted strategies.

  
## 📤 PDF Report Example Includes:
Number of clusters  
Cluster size  
Average feature values  
Summary insights  

  
## Useful For:
✅ Business presentations   
✅ Sales optimization  
✅ Understanding customer behavior  
✅ Academic project submission  
✅ Marketing strategy planning
