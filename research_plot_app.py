import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from io import StringIO
import PyPDF2
import docx

st.set_page_config(page_title="Universal Plot App", layout="wide")

# Layout
left, right = st.columns([2, 3])

with left:
    st.title("📊 Universal Data Plotter")
    uploaded_file = st.file_uploader("📁 Upload file (CSV, Excel, PDF, Word, or Text)", type=["csv", "xlsx", "xls", "txt", "pdf", "docx"])

# Function to handle various file types
def read_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif file.name.endswith(".txt"):
        stringio = StringIO(file.getvalue().decode("utf-8"))
        text = stringio.read()
        return pd.DataFrame({"Text": text.splitlines()})
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return pd.DataFrame({"PDF Text": text.splitlines()})
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return pd.DataFrame({"Word Text": paragraphs})
    return None

if uploaded_file:
    df = read_uploaded_file(uploaded_file)

    if df is not None and not df.empty:
        with right:
            st.subheader("📄 Data Preview")
            st.dataframe(df, use_container_width=True)

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        with left:
            if numeric_cols:
                st.subheader("⚙️ Plot Configuration")

                x_suggest = df[numeric_cols].nunique().idxmax()
                y_suggest = df[numeric_cols].var().idxmax()

                x_axis = st.selectbox("X-axis", options=numeric_cols, index=numeric_cols.index(x_suggest))
                y_axis = st.selectbox("Y-axis", options=[col for col in numeric_cols if col != x_axis], index=numeric_cols.index(y_suggest) if y_suggest != x_axis else 0)

                hue = st.selectbox("Group by (optional)", options=["None"] + category_cols)
                hue = None if hue == "None" else hue

                chart_type = st.selectbox("📈 Choose chart type", ["Line", "Bar", "Scatter", "Pie"])
                apply_fit = st.checkbox("Apply Curve Fitting (Line/Scatter only)", value=False)
                fit_degree = st.slider("Polynomial Degree", 1, 5, 2) if apply_fit else None

                dark = st.checkbox("🌙 Dark Mode", value=False)
                title = st.text_input("Plot Title", f"{y_axis} vs {x_axis}")
                xlabel = st.text_input("X-axis Label", x_axis)
                ylabel = st.text_input("Y-axis Label", y_axis)

                if st.button("Generate Plot"):
                    sns.set_style("darkgrid" if dark else "whitegrid")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if chart_type == "Line":
                        if hue:
                            sns.lineplot(data=df, x=x_axis, y=y_axis, hue=hue, marker="o", ax=ax)
                        else:
                            sns.lineplot(data=df, x=x_axis, y=y_axis, marker="o", ax=ax)
                    elif chart_type == "Bar":
                        if hue:
                            sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
                        else:
                            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
                    elif chart_type == "Scatter":
                        if hue:
                            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue, ax=ax)
                        else:
                            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                    elif chart_type == "Pie":
                        if x_axis in category_cols:
                            counts = df[x_axis].value_counts()
                            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')

                    if apply_fit and chart_type in ["Line", "Scatter"]:
                        X = df[[x_axis]].values
                        y = df[y_axis].values
                        poly = PolynomialFeatures(degree=fit_degree)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                        y_pred = model.predict(poly.transform(X_range))
                        ax.plot(X_range, y_pred, linestyle="--", label="Fit")

                    if chart_type != "Pie":
                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        ax.legend()
                        ax.grid(True)

                    st.pyplot(fig)

            else:
                st.warning("No numeric columns detected. This file might not be graphable.")

    else:
        st.error("❌ Unable to read or parse the file.")
