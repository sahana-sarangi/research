import pandas as pd
import numpy as np
import altair as alt
import os

DATA_PATH = "./" 


alt.data_transformers.disable_max_rows()


def add_leading_zeroes(x):
    if pd.isna(x):
        x = 0
    return "{:02d}".format(int(x))



data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0)
data["years"] = data["years"].fillna(0).astype(int)
data = data.rename(columns={"years": "Year"})

df = pd.read_csv(os.path.join(DATA_PATH, "updated_fine_tuned_tsne60.csv"))
df = df.rename(columns={
    "Topic Name (Post Forced)": "Cluster",
    "x": "TSNE-x",
    "y": "TSNE-y",
    "title": "AbstractTitle",
    "abstract": "Abstract"
})
df["Topic (Post Forced)"] = df["Topic (Post Forced)"].fillna(0).astype(int)


df = pd.merge(df, data, on="AbstractTitle", suffixes=("_df", None))
df = df.drop(columns=df.filter(regex="_df$").columns)
df["Index"] = np.arange(1, df.shape[0] + 1)
df["Cluster"] = df["Topic (Post Forced)"].apply(add_leading_zeroes)


bt60_names = pd.read_csv(os.path.join(DATA_PATH, "updated_fine_tuned_tnse60_w_names_final_ver.csv"))
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
    .reset_index(name="Count")
    .sort_values(["TopicName", "Year"])
)


topic_growth["PrevCount"] = topic_growth.groupby("TopicName")["Count"].shift(1)
topic_growth["GrowthRate"] = (
    (topic_growth["Count"] - topic_growth["PrevCount"]) / topic_growth["PrevCount"]
)
topic_growth["GrowthRate"] = topic_growth["GrowthRate"].replace([np.inf, -np.inf], np.nan)
topic_growth["GrowthRate"] = topic_growth["GrowthRate"].fillna(0)


avg_growth = (
    topic_growth.groupby("TopicName")["GrowthRate"].mean().reset_index(name="AvgGrowthRate")
)


df = df.merge(avg_growth, on="TopicName", how="left")


color_scale = alt.Scale(
    domain=[df["AvgGrowthRate"].min(), 0, df["AvgGrowthRate"].max()],
    range=["#4575b4", "#762a83", "#d73027"]  # blue → purple → red
)


final_chart = (
    alt.Chart(df)
    .mark_circle(size=25, opacity=0.9)
    .encode(
        x=alt.X("TSNE-x:Q", title="TSNE-x"),
        y=alt.Y("TSNE-y:Q", title="TSNE-y"),
        color=alt.Color(
            "AvgGrowthRate:Q",
            title="Average Growth per Topic (Δ abstracts/year)",
            scale=color_scale,
            legend=alt.Legend(
                orient="right",
                title="Topic Growth Trend",
                titleFontSize=12,
                labelFontSize=10,
                labelExpr="""
                    datum.label == '−0.5' ? 'Strong Decline'
                    : datum.label == '0' ? 'Stable'
                    : datum.label == '0.5' ? 'Strong Growth'
                    : datum.label
                """
            )
        ),
        tooltip=[
            alt.Tooltip("AbstractTitle", title="Abstract Title"),
            alt.Tooltip("TopicName:N", title="Topic Name"),
            alt.Tooltip("AvgGrowthRate:Q", title="Average Growth Rate", format=".2f"),
            alt.Tooltip("Year:Q", title="Year"),
        ]
    )
    .properties(
        title="Relative growth",
        width=800,
        height=700
    )
    .configure_title(fontSize=18, anchor="start")
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)


output_file_name = "tsne_relative_growth.html"
final_chart.save(output_file_name)
print(f"{output_file_name}")


try:
    import streamlit as st
    st.title("Topic Growth Visualization")
    st.altair_chart(final_chart, use_container_width=True)
except ImportError:
    print("Streamlit not detected — chart saved to HTML instead.")
