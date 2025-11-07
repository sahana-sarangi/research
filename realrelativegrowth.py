import streamlit as st
import pandas as pd
import altair as alt
import requests
import io

st.set_page_config(page_title="Relative Growth Figure", layout="wide")
st.title("Relative Growth (% change)")

@st.cache_data
def load_csv_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to download file with id {file_id}")
        st.stop()
    return pd.read_csv(io.BytesIO(response.content), encoding="utf8", errors="ignore")

astro_id = "1GySlfSGMIt0LZb_XCgP29DaqPL2aCISI"
tsne_id  = "1AlqzyJQSxfK2MJGVdQriZfBtnGrzDzVS"
names_id = "1s6T-5KchhgOnoCX16aMYGtJ1_TiU_hqm"

astro = load_csv_from_gdrive(astro_id)
tsne = load_csv_from_gdrive(tsne_id)
names = load_csv_from_gdrive(names_id)

for df in [astro, tsne, names]:
    df.columns = [c.strip() for c in df.columns]

if "Topic" in tsne.columns and "Topic" in names.columns:
    df = tsne.merge(names, on="Topic", how="left").merge(astro, on="Topic", how="left")
elif "TopicName" in tsne.columns:
    df = tsne.merge(names, on="TopicName", how="left").merge(astro, on="TopicName", how="left")
else:
    st.error("Could not find a common merge key like 'Topic' or 'TopicName'.")
    st.stop()

if "Year" not in df.columns:
    st.error("Missing 'Year' column in dataset.")
    st.stop()

if "AbstractCount" not in df.columns:
    if "Count" in df.columns:
        df = df.rename(columns={"Count": "AbstractCount"})
    else:
        st.error("Missing 'AbstractCount' column in dataset.")
        st.stop()

df = df.sort_values(["TopicName", "Year"])
df["RelativeGrowth"] = df.groupby("TopicName")["AbstractCount"].pct_change() * 100
df["RelativeGrowth"] = df["RelativeGrowth"].fillna(0)

min_g = df["RelativeGrowth"].min()
max_g = df["RelativeGrowth"].max()

color_scale = alt.Scale(
    domain=[min_g, 0, max_g],
    range=["#4575b4", "#762a83", "#d73027"]
)

chart = (
    alt.Chart(df)
    .mark_circle(size=60, opacity=0.8)
    .encode(
        x=alt.X("TSNE-x:Q", title="t-SNE X"),
        y=alt.Y("TSNE-y:Q", title="t-SNE Y"),
        color=alt.Color(
            "RelativeGrowth:Q",
            scale=color_scale,
            legend=alt.Legend(
                title="Relative Growth (% change per year)",
                titleFontSize=12,
                labelFontSize=10
            )
        ),
        tooltip=[
            alt.Tooltip("TopicName:N", title="Topic Name"),
            alt.Tooltip("Year:O", title="Year"),
            alt.Tooltip("AbstractCount:Q", title="# of Abstracts"),
            alt.Tooltip("RelativeGrowth:Q", title="Relative Growth (%)", format=".2f")
        ]
    )
    .properties(width=900, height=650)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)
