import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Relative Growth")

alt.data_transformers.disable_max_rows()

def add_leading_zeroes(x):
    if pd.isna(x):
        x = 0
    return "{:02d}".format(int(x))

astro_url = "https://drive.google.com/uc?export=download&id=1GySlfSGMIt0LZb_XCgP29DaqPL2aCISI"
tsne_url = "https://drive.google.com/uc?export=download&id=1AlqzyJQSxfK2MJGVdQriZfBtnGrzDzVS"
names_url = "https://drive.google.com/uc?export=download&id=1s6T-5KchhgOnoCX16aMYGtJ1_TiU_hqm"

data = pd.read_csv(astro_url, index_col=0)
data['years'] = data['years'].fillna(0)
data.years = data.years.astype(int)
data = data.rename(columns={"years": "Year"})

df = pd.read_csv(tsne_url, encoding="utf8")
df = df.rename(columns={
    "Topic Name (Post Forced)": "Cluster",
    "x": "TSNE-x",
    "y": "TSNE-y",
    "title": "AbstractTitle",
    "abstract": "Abstract"
})
df["Topic (Post Forced)"] = df["Topic (Post Forced)"].fillna(0).astype(int)
df = pd.merge(df, data, on=['AbstractTitle'], suffixes=("_df", None))
df = df.drop(columns=df.filter(regex="_df$").columns)
df['Index'] = np.arange(1, df.shape[0] + 1)
df["Cluster"] = df["Topic (Post Forced)"].apply(add_leading_zeroes)

bt60_names = pd.read_csv(names_url)
bt60_names = bt60_names.rename(columns={"title": "AbstractTitle"})
bt60_names["Topic (Post Forced)"] = bt60_names["Topic (Post Forced)"].fillna(0).astype(int)
df = pd.merge(
    df,
    bt60_names[["AbstractTitle", "GPT_Names", "Topic (Post Forced)"]],
    on=["AbstractTitle", "Topic (Post Forced)"],
    how="left"
)

df = df.rename(columns={"GPT_Names": "TopicName"})
df["TopicName"] = df["TopicName"].fillna("Topic " + df["Topic (Post Forced)"].astype(str))
df["TopicName"] = df["TopicName"].apply(lambda x: x if len(x) <= 50 else x[:47] + "...")


topic_growth = (
    df.groupby(["TopicName", "Year"])
    .size()
    .reset_index(name="AbstractsPerYear")
)


def calc_smoothed_growth(g):
    g = g[g["AbstractsPerYear"] > 0]
    if len(g) < 2:
        return 0
    X = g[["Year"]]
    y = np.log(g["AbstractsPerYear"])
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    return np.exp(slope) - 1  

smoothed_growth = (
    topic_growth.groupby("TopicName")
    .apply(calc_smoothed_growth)
    .reset_index(name="RelativeGrowthRate")
)

df = df.merge(smoothed_growth, on="TopicName", how="left")
df["RelativeGrowthRate"] = df["RelativeGrowthRate"].fillna(0.0)


max_abs_growth = float(np.abs(df["RelativeGrowthRate"]).max())
if max_abs_growth == 0 or np.isclose(max_abs_growth, 0.0):
    max_abs_growth = 1e-6

color_scale = alt.Scale(
    domain=[-max_abs_growth, 0.0, max_abs_growth],
    range=["#4575b4", "#762a83", "#d73027"],
)

final_chart = (
    alt.Chart(df)
    .mark_circle(size=25, opacity=0.9)
    .encode(
        x=alt.X("TSNE-x:Q", title="t-SNE x"),
        y=alt.Y("TSNE-y:Q", title="t-SNE y"),
        color=alt.Color(
            "RelativeGrowthRate:Q",
            scale=color_scale,
            title="Avg Relative Growth (% change per year)",
            legend=alt.Legend(
                orient="right",
                title="Average Growth Rate",
                titleFontSize=13,
                labelFontSize=11,
                labelLimit=250,
                gradientLength=200,
                direction="vertical",
                gradientThickness=20
            )
        ),
        tooltip=[
            alt.Tooltip("AbstractTitle:N", title="Abstract Title"),
            alt.Tooltip("TopicName:N", title="Topic Name"),
            alt.Tooltip("RelativeGrowthRate:Q", title="Avg. % Growth/Year", format=".2%"),
            alt.Tooltip("Year:Q", title="Year")
        ],
    )
    .properties(width=700, height=1000)
    .configure_title(fontSize=18, anchor="start")
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)

st.title("Relative Growth Figure (% change)")
st.altair_chart(final_chart, use_container_width=True)
