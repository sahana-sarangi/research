import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Relative Growth figure", layout="wide")

st.title("Relative Growth (% change)")

astro_url = "https://drive.google.com/uc?export=download&id=1GySlfSGMIt0LZb_XCgP29DaqPL2aCISI"
tsne_url = "https://drive.google.com/uc?export=download&id=1AlqzyJQSxfK2MJGVdQriZfBtnGrzDzVS"
names_url = "https://drive.google.com/uc?export=download&id=1s6T-5KchhgOnoCX16aMYGtJ1_TiU_hqm"

astro = pd.read_csv(astro_url, encoding="utf8")
tsne = pd.read_csv(tsne_url, encoding="utf8")
names = pd.read_csv(names_url, encoding="utf8")

df = tsne.merge(names, on="Topic", how="left").merge(astro, on="Topic", how="left")
df = df.sort_values(["Topic", "Year"])
df["RelativeGrowth"] = df.groupby("Topic")["AbstractCount"].pct_change() * 100
df["RelativeGrowth"] = df["RelativeGrowth"].fillna(0)

min_growth = df["RelativeGrowth"].min()
max_growth = df["RelativeGrowth"].max()

chart = (
    alt.Chart(df)
    .mark_circle(size=70, opacity=0.75)
    .encode(
        x=alt.X("TSNE-x:Q", axis=alt.Axis(title="t-SNE X")),
        y=alt.Y("TSNE-y:Q", axis=alt.Axis(title="t-SNE Y")),
        color=alt.Color(
            "RelativeGrowth:Q",
            scale=alt.Scale(domain=[min_growth, 0, max_growth], range=["blue", "purple", "red"]),
            legend=alt.Legend(title="% Growth in Abstracts per Year"),
        ),
        tooltip=[
            alt.Tooltip("TopicName:N", title="Topic Name"),
            alt.Tooltip("Year:O", title="Year"),
            alt.Tooltip("AbstractTitle:N", title="Abstract Title"),
            alt.Tooltip("RelativeGrowth:Q", title="Relative Growth (%)", format=".2f"),
        ],
    )
    .properties(width=900, height=600)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)
