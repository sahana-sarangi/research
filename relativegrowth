from google.colab import drive
import pandas as pd
import numpy as np
import altair as alt
import os


drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Research/bycolor')
print(f"Current working directory: {os.getcwd()}")


alt.data_transformers.disable_max_rows()


DATA_PATH = '/content/drive/My Drive/Research/bycolor'


def add_leading_zeroes(x):
    if pd.isna(x):
        x = 0
    return "{:02d}".format(int(x))


data = pd.read_csv(os.path.join(DATA_PATH, "updated_astro_dataset60.csv"), index_col=0)
data['years'] = data['years'].fillna(0)
data['years'] = data['years'].astype(int)
data = data.rename(columns={"years": "Year"})

file_path_tsne = os.path.join(DATA_PATH, "updated_fine_tuned_tsne60.csv")
with open(file_path_tsne, encoding="utf8", errors='ignore') as temp_f:
    df = pd.read_csv(temp_f)
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


file_path_names = os.path.join(DATA_PATH, "updated_fine_tuned_tnse60_w_names_final_ver.csv")
bt60_names = pd.read_csv(file_path_names)
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


growth_rates = (
    topic_growth.groupby("TopicName")
    .apply(lambda g: np.polyfit(g["Year"], g["AbstractsPerYear"], 1)[0] if len(g) > 1 else 0)
    .reset_index(name="GrowthRate")
)

df = df.merge(growth_rates, on="TopicName", how="left")
df["GrowthRate"] = df["GrowthRate"].fillna(0.0)


max_abs_growth = float(np.abs(df["GrowthRate"]).max())
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
            "GrowthRate:Q",
            scale=color_scale,
            title="Topic Growth Rate (abstracts per year)",
            legend=alt.Legend(
                orient="right",
                title="Growth Trend",
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
            alt.Tooltip("GrowthRate:Q", title="Growth Rate (Δ abstracts/year)", format=".2f"),
            alt.Tooltip("Year:Q", title="Year")
        ],
    )
    .properties(
        title="Relative Growth of Topics",
        width=850,
        height=700
    )
    .configure_title(fontSize=18, anchor="start")
    .configure_axis(labelFontSize=12, titleFontSize=14, grid=True)
    .configure_view(strokeWidth=0)
)


output_file_name = "tsne_relative_growth_red_purple_blue.html"
final_chart.save(output_file_name)
print(f"Chart saved as {output_file_name}")

final_chart
