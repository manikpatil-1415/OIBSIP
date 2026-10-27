import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# TASK 2: CUSTOMER SEGMENTATION
# =========================

# Load dataset
file_path = "Online Retail.xlsx"
df = pd.read_excel(file_path)

# Inspect dataset
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled/invalid transactions
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Create Total Purchase Value
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# =========================
# RFM ANALYSIS
# =========================

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

print("\nRFM Dataset:")
print(rfm.head())

print("\nDescriptive Statistics:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# =========================
# FEATURE SELECTION
# =========================

features = rfm[["Recency", "Frequency", "Monetary"]]

# Remove extreme outliers using IQR
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1

mask = ~((features < (Q1 - 1.5 * IQR)) |
         (features > (Q3 + 1.5 * IQR))).any(axis=1)

rfm_clean = rfm[mask].copy()

features = rfm_clean[["Recency", "Frequency", "Monetary"]]

# =========================
# STANDARDIZATION
# =========================

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# =========================
# ELBOW METHOD
# =========================

inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# =========================
# K-MEANS CLUSTERING
# =========================

# K=4 is used for customer segmentation
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

rfm_clean["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nCluster Counts:")
print(rfm_clean["Cluster"].value_counts().sort_index())

# =========================
# CLUSTER PROFILE
# =========================

cluster_profile = rfm_clean.groupby("Cluster")[[
    "Recency", "Frequency", "Monetary"
]].mean().round(2)

print("\nCluster Profile:")
print(cluster_profile)

# =========================
# SCATTER PLOT 1
# =========================

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=rfm_clean,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="viridis",
    s=60
)
plt.title("Customer Clusters: Frequency vs Monetary")
plt.xlabel("Purchase Frequency")
plt.ylabel("Monetary Value")
plt.show()

# =========================
# SCATTER PLOT 2
# =========================

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=rfm_clean,
    x="Recency",
    y="Frequency",
    hue="Cluster",
    palette="viridis",
    s=60
)
plt.title("Customer Clusters: Recency vs Frequency")
plt.xlabel("Recency")
plt.ylabel("Purchase Frequency")
plt.show()

# =========================
# BAR CHART
# =========================

cluster_counts = rfm_clean["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
plt.title("Number of Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

# =========================
# MARKETING INSIGHTS
# =========================

print("\nMARKETING INSIGHTS")

for cluster in sorted(rfm_clean["Cluster"].unique()):
    row = cluster_profile.loc[cluster]

    print(f"\nCluster {cluster}:")
    print(f"Average Recency: {row['Recency']:.2f} days")
    print(f"Average Frequency: {row['Frequency']:.2f}")
    print(f"Average Monetary Value: {row['Monetary']:.2f}")

    if row["Frequency"] >= cluster_profile["Frequency"].median() and row["Monetary"] >= cluster_profile["Monetary"].median():
        print("Customer Type: High-value / Loyal Customers")
        print("Marketing Action: Reward loyalty, provide exclusive offers and premium products.")

    elif row["Recency"] <= cluster_profile["Recency"].median() and row["Frequency"] < cluster_profile["Frequency"].median():
        print("Customer Type: Recent / Potential Customers")
        print("Marketing Action: Use personalised recommendations and offers to increase repeat purchases.")

    elif row["Recency"] > cluster_profile["Recency"].median() and row["Monetary"] < cluster_profile["Monetary"].median():
        print("Customer Type: At-risk / Low-value Customers")
        print("Marketing Action: Send reactivation campaigns, discounts and reminders.")

    else:
        print("Customer Type: Regular Customers")
        print("Marketing Action: Maintain engagement with targeted promotions and cross-selling.")