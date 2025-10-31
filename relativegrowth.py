import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import os

st.set_page_config(layout="wide", page_title="Relative growth")

alt.data_transformers.disable_max_rows()

def add_leading_zeroes(x):
    if pd.isna(x):
        x = 0
    return "{:02d}".format(int(x))

astro_url = "https://drive.google.com/uc?id=1GySlfSGMIt0LZb_XCgP29DaqPL2aCISI"
tsne_url = "https://drive.google.com/uc?id=1AlqzyJQSxfK2MJGVdQriZfBtnGrzDzVS"
names_url = "https://drive.google.com/uc?id=1s6T-5KchhgOnoCX16aMYGtJ1_TiU_hqm"

data = pd.read_csv(astro_url, index_col=0)
data['years'] = data['years'].fillna(0)
data.years = data.years.astype(int)
data = data.rename(columns={"years": "Year"})

df = pd.read_csv(tsne_url, encoding="utf8", errors='ignore')
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

growth_df = (
    df.groupby(["TopicName", "Year"])
    .size()
    .reset_index(name="Count")
    .sort_values(["TopicName", "Year"])
)

growth_df["GrowthRate"] = growth_df.groupby("TopicName")["Count"].diff()
growth_df["GrowthRate"] = growth_df["GrowthRate"].fillna(0)

avg_growth = (
    growth_df.groupby("TopicName")["GrowthRate"]
    .mean()
    .reset_index(name="AvgGrowthRate")
)

df = df.merge(avg_growth, on="TopicName", how="left")
df["AvgGrowthRate"] = df["AvgGrowthRate"].fillna(0)

color_scale = alt.Scale(
    domain=[df["AvgGrowthRate"].min(), 0, df["AvgGrowthRate"].max()],
    range=["#4575b4", "#762a83", "#d73027"]
)

chart = (
    alt.Chart(df)
    .mark_circle(size=25, opacity=0.9)
    .encode(
        x=alt.X('TSNE-x:Q', title='TSNE-x'),
        y=alt.Y('TSNE-y:Q', title='TSNE-y'),
        color=alt.Color(
            'AvgGrowthRate:Q',
            title='Topic Growth Rate (Δ abstracts / year)',
            scale=color_scale,
            legend=alt.Legend(
                title="Growth Rate (Δ abstracts / year)",
                titleFontSize=12,
                labelFontSize=10
            )
        ),
        tooltip=[
            alt.Tooltip('AbstractTitle', title='Abstract Title'),
            alt.Tooltip('TopicName:N', title='Topic Name'),
            alt.Tooltip('AvgGrowthRate:Q', title='Growth Rate (Δ abstracts / year)', format=".2f"),
            alt.Tooltip('Year:Q', title='Year')
        ]
    )
    .properties(
        title='Relative Growth',
        width=900,
        height=700
    )
    .configure_title(
        fontSize=18,
        anchor='start'
    )
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        grid=True
    )
    .configure_view(strokeWidth=0)
)

st.title("Relative growth")
st.altair_chart(chart, use_container_width=True)
