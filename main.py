from pythermalcomfort.models import pmv_ppd, set_tmp, two_nodes
from pythermalcomfort.utilities import v_relative, clo_dynamic
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import warnings
from scipy import stats
import matplotlib.style
import matplotlib as mpl
from scipy import stats

warnings.filterwarnings("ignore")


def preprocess_comfort_db_data(limit, import_csv):

    if import_csv:
        return pd.read_csv(r"./Data/DBII_preprocessed.csv")

    # import DB II data
    df_db2 = pd.read_csv(r"./Data/DatabaseI&II_20180703.csv", encoding="unicode_escape")

    rename_cols = {
        "Thermal sensation": "TSV",
        "Air temperature (°C)": "Ta",
        "Radiant temperature (°C)": "Tr",
        "Relative humidity (%)": "Rh",
        "Air velocity (m/s)": "V",
        "Operative temperature (°C)": "To",
    }
    df_db2.rename(columns=rename_cols, inplace=True)

    # remove entries which do not have all the data
    tot_entries = df_db2.shape[0]

    # I am keeping entries without Tr otherwise I would lose too many data points
    df_db2 = (
        df_db2.dropna(subset=["Ta", "V", "Rh", "Met", "Clo", "TSV"])
        .reset_index(drop=True)
        .copy()
    )

    def entries_removed(initial, new_df):
        print(
            f"Entries: {new_df.shape[0]} -- Entries removed: "
            f"{initial - new_df.shape[0]}"
        )

    entries_removed(tot_entries, df_db2)

    # only process a sub-set of data
    if limit:
        df_db2 = df_db2.head(limit).copy()

    df_db2 = df_db2.reset_index(drop=True).copy()

    # calculate PMV ISO, PMV SET and PMV*
    results = []
    for ix, row in df_db2.iterrows():

        # remove entries outside applicability limits
        if row["Tr"] != row["Tr"]:
            if row["To"] != row["To"]:
                row["Tr"] = row["Ta"]
            else:
                row["Tr"] = 2 * row["To"] - row["Ta"]

        vr = v_relative(v=row["V"], met=row["Met"])
        clo_d = clo_dynamic(clo=row["Clo"], met=row["Met"])
        pmv_ashrae = pmv_ppd(
            tdb=row["Ta"],
            tr=row["Tr"],
            vr=vr,
            rh=row["Rh"],
            met=row["Met"],
            clo=clo_d,
            standard="ashrae",
        )["pmv"]
        pmv_iso = pmv_ppd(
            tdb=row["Ta"],
            tr=row["Tr"],
            vr=vr,
            rh=row["Rh"],
            met=row["Met"],
            clo=clo_d,
            standard="iso",
        )["pmv"]
        # pmv_set = two_nodes(
        #     tdb=row["Ta"],
        #     tr=row["Tr"],
        #     v=row["V"],
        #     rh=row["Rh"],
        #     met=row["Met"],
        #     clo=row["Clo"],
        # )["pmv_set"]
        # set_py = set_tmp(
        #     tdb=row["Ta"],
        #     tr=row["Tr"],
        #     v=row["V"],
        #     rh=row["Rh"],
        #     met=row["Met"],
        #     clo=row["Clo"],
        # )
        results.append(
            {
                "index": ix,
                "pmv_ashrae": pmv_ashrae,
                "pmv_iso": pmv_iso,
                "Tr_est": row["Tr"]
                # "pmv_set": pmv_set,
                # "set_py": set_py,
            }
        )

    df_ = pd.DataFrame(results)

    df_ = pd.concat([df_db2, df_], axis=1, sort=False)

    df_.to_csv(r"./Data/DBII_preprocessed.csv", index=False)

    return df_


def filter_data(df_):

    # remove entries outside the Standards' applicability limits
    df_ = df_[(df_["Ta"] >= 10) & (df_["Ta"] <= 30)]
    df_ = df_[(df_["Tr_est"] >= 10) & (df_["Tr_est"] <= 40)]
    df_ = df_[(df_["V"] >= 0) & (df_["V"] <= 1)]
    df_ = df_[(df_["Clo"] >= 0) & (df_["Clo"] <= 1.5)]
    df_ = df_[(df_["Met"] >= 1) & (df_["Met"] <= 4)]
    # todo add more filters based on age and location of the study
    df_ = df_[(df_["pmv_iso"] > -2) & (df_["pmv_iso"] < 2)]
    df_ = df_[(df_["pmv_ashrae"] > -2) & (df_["pmv_ashrae"] < 2)]
    df_ = df_[(df_["TSV"] > -2) & (df_["TSV"] < 2)]
    # todo analyse the data for V > 0.1 since there is where you can see most of the diff

    return df_


def calculate_new_indices(df_):

    df_["pmv_iso_round"] = df_["pmv_iso"].round()
    df_["pmv_ashrae_round"] = df_["pmv_ashrae"].round()
    df_["PMV_round"] = df_["PMV"].round()
    df_["TSV_round"] = df_["TSV"].round()
    df_["diff_iso"] = df_[["TSV", "pmv_iso"]].diff(axis=1)["pmv_iso"]
    df_["diff_ash"] = df_[["TSV", "pmv_ashrae"]].diff(axis=1)["pmv_ashrae"]

    # check delta between the PMV I calculated and the one Toby did
    df_["PMV - pmv_iso"] = df_["PMV"] - df_["pmv_iso"]
    df_["PMV - pmv_ashrae"] = df_["PMV"] - df_["pmv_ashrae"]
    # df_["SET - set_py"] = df_["SET"] - df_["set_py"]
    df_["TSV_round - pmv_ashrae_round"] = df_["TSV"] - df_["pmv_ashrae_round"]

    return df_


def bar_chart(data, ind="tsv"):
    f, axs = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    map_model_name = {"pmv_iso_round": r"PMV", "pmv_ashrae_round": r"PMV$_{CE}$"}

    for ix, model in enumerate(["pmv_iso_round", "pmv_ashrae_round"]):
        if ind == "pmv":
            _df = data.groupby(["TSV_round", model])[model].count().unstack("TSV_round")
            x = model
            x_label = "PMV"
        else:
            _df = data.groupby(["TSV_round", model])["TSV_round"].count().unstack(model)
            x = "TSV_round"
            x_label = "TSV"
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        df_plot = df_rel.reset_index()
        df_plot[x] = pd.to_numeric(df_plot[x], downcast="integer")
        hist = df_plot.plot(
            x=x,
            kind="bar",
            stacked=True,
            mark_right=True,
            width=0.95,
            rot=0,
            legend=False,
            ax=axs[ix],
            colormap="coolwarm",
        )
        axs[ix].set(xlabel=x_label, ylabel="Percentage [%]")
        axs[ix].set_title(map_model_name[model], y=1.06)
        sns.despine(ax=axs[ix], left=True, bottom=True)

        # show accuracy
        if np.all(df_rel.index == df_rel.columns):
            diagonal = pd.Series(np.diag(df_rel), index=df_rel.index)

            for ix_s, value in diagonal.items():
                print(ix_s, value)
                if value != value:
                    value = 0
                axs[ix].text(
                    ix_s + 2,
                    110,
                    f"{value:.0f}%",
                    va="center",
                    ha="center",
                )

        for ix_s, value in enumerate(data.groupby(["TSV_round"])["TSV_round"].count()):
            axs[ix].text(
                ix_s,
                103,
                f"{value:.0f}",
                va="center",
                ha="center",
            )

        # add percentages
        for index, row in df_rel.fillna(0).iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                if el > 7:
                    axs[ix].text(
                        index + 2,
                        cum_sum + el / 2,
                        f"{int(round(el, 0))}%",
                        va="center",
                        ha="center",
                    )
                cum_sum += el
                print(cum_sum)

    if data.V.min() == 0:
        sm = plt.cm.ScalarMappable(
            cmap="rainbow", norm=plt.Normalize(vmin=-2.5, vmax=+2.5)
        )
        cmap = mpl.cm.rainbow
        bounds = np.linspace(-2.5, 2.5, 6)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.get_cmap("rainbow", 5)
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
            ticks=np.linspace(-2, 2, 5),
            ax=axs,
            orientation="horizontal",
            aspect=70,
        )
        cbar.ax.set_xticklabels(
            [
                "Cool (-2)",
                "Slightly Cool (-1)",
                "Neutral (0)",
                "Slightly Warm (1)",
                "Warm (2)",
            ]
        )
        cbar.outline.set_visible(False)
    plt.savefig(f"./Manuscript/Figures/bar_plot_{ind}_Vmin_{data.V.min()}.png", dpi=300)


def distributions_pmv(v_lower=False):
    # check difference in distribution for V > 0.1
    variables = ["pmv_iso", "pmv_ashrae", "TSV"]
    f, axs = plt.subplots(len(variables), 1, sharex=True)
    if v_lower:
        data = df[df["V"] > 0.1]
    else:
        data = df.copy()
    for ix, var in enumerate(variables):
        sns.histplot(data=data, x=var, kde=True, stat="density", ax=axs[ix])
        axs[ix].set(title=var)
    plt.tight_layout()


def scatter_plot(ind="tsv", x_jitter=0):
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    for ix, model in enumerate(["pmv_iso", "pmv_ashrae"]):
        if ind == "pmv":
            sns.regplot(data=df, x=df[model], y="TSV", ax=axs[ix], x_jitter=0.1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                y=df["TSV"], x=df[model]
            )
        else:
            sns.regplot(data=df, y=df[model], x="TSV", ax=axs[ix], x_jitter=x_jitter)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=df["TSV"], y=df[model]
            )

        slope, intercept, r_value, p_value, std_err = [
            round(x, 2) for x in [slope, intercept, r_value, p_value, std_err]
        ]

        axs[ix].text(
            0.1,
            0.1,
            f"{slope =}, {intercept =},\n{r_value =}, {p_value =}",
            transform=axs[ix].transAxes,
        )

        sns.lineplot(x=[-6, 6], y=[-6, 6], ax=axs[ix])
        sns.lineplot(x=df["TSV"], y=intercept + df["TSV"] * slope, ax=axs[ix])
        # sns.scatterplot(data=df, y=df[model], x="TSV", s=5, color=".15", ax=axs[ix])
        # sns.histplot(data=df, y=df[model], x="TSV", bins=50, pthresh=.1, cmap="mako", ax=axs[ix])
        # sns.histplot(data=df, y=df[model], x="TSV", bins=50, cmap="mako", ax=axs[ix])
        # sns.kdeplot(data=df, y=df[model], x="TSV", levels=5, color="w", linewidths=1)
        axs[ix].set(ylim=(-2.5, 2.5), xlim=(-2.5, 2.5), ylabel="")
        axs[ix].set_aspect("equal", adjustable="box")
        axs[ix].set(title=model)
        sns.despine(bottom=True, left=True)

    plt.tight_layout()
    plt.savefig("./Manuscript/Figures/scatter_tsv_pmv.png", dpi=300)


def plot_error_prediction(data, ind="tsv"):
    # f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(8.0, 3))
    # for ix, model in enumerate(["diff_iso", "diff_ash"]):
    #     if ind == "pmv":
    #         sns.violinplot(data=df, y="TSV_round", x=df[model], ax=axs[ix], size="data")
    #     else:
    #         sns.violinplot(data=df, x="TSV_round", y=df[model], ax=axs[ix], size="data")
    #     axs[ix].set_aspect("equal", adjustable="box")
    #     axs[ix].set(title=model)
    #     axs[ix].fill_between([-0.5, 6.5], 0.5, -0.5, color="red", alpha=0.5)
    # plt.tight_layout()

    f, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(8.0, 3))
    _df = (
        data[["TSV_round", "diff_iso", "diff_ash"]]
        .set_index("TSV_round")
        .stack()
        .reset_index()
    )
    _df.columns = ["TSV", "model", "delta"]
    _df["model"] = _df["model"].map({"diff_iso": "PMV", "diff_ash": r"PMV$_{CE}$"})
    _df["TSV"] = pd.to_numeric(_df["TSV"], downcast="integer")
    sns.violinplot(
        data=_df,
        x="TSV",
        y="delta",
        size="data",
        split=True,
        hue="model",
        inner="quartile",
        color="gray",
    )
    axs.set(ylabel="Prediction error")
    sns.despine(bottom=True, left=True)
    plt.legend(frameon=False, loc=3)

    acceptable_error = 0.5
    # t-test
    for ix, tsv_vote in enumerate(_df["TSV"].sort_values().unique()):
        print(tsv_vote)
        sample_1 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV")]["delta"]
        sample_2 = _df[(_df["TSV"] == tsv_vote) & (_df["model"] == "PMV$_{CE}$")][
            "delta"
        ]
        p = round(stats.ttest_ind(sample_1, sample_2).pvalue, 2)
        if p < 0.01:
            text_p = r"$p$ < 0.01"
        else:
            text_p = r"$p$ = " + str(p)
        axs.text(ix, 5, text_p, ha="center")
        # perc_1 = round(sum(sample_1.abs() <= acceptable_error) / sample_1.shape[0] * 100)
        # perc_2 = round(sum(sample_2.abs() <= acceptable_error) / sample_1.shape[0] * 100)
        # axs.text(ix, -0.5, f"{perc_2}%", va="center")
        # axs.text(ix, 0.5, f"{perc_1}%", ha="right", va="center")

    axs.fill_between(
        [-0.5, 4.5], acceptable_error, -acceptable_error, color="red", alpha=0.5
    )

    plt.savefig(
        f"./Manuscript/Figures/prediction_error_Vmin_{data.V.min()}.png", dpi=300
    )


def plot_distribution_variable():
    f, axs = plt.subplots(1, 6, constrained_layout=True, figsize=(8, 3))

    for ix, var in enumerate(["Ta", "Tr", "V", "Clo", "Met", "Rh"]):
        # _df = df[[var, "Year"]]
        # _df["Year"] = 0
        # min, max = df[var].min(), df[var].max()
        # _df = _df.append({var: 999, "Year": 1}, ignore_index=True)
        # _df["Var"] = var_names[var]
        # sns.violinplot(
        #     x=var,
        #     y="Var",
        #     ax=axs[ix],
        #     hue="Year",
        #     split=True,
        #     data=_df,
        #     color="lightgray",
        #     inner="quartiles",
        # )
        sns.boxenplot(y=var, data=df, ax=axs[ix], color="lightgray")
        # axs[ix].set(xlabel=var_units[var], xlim=(min, max), ylabel="")
        axs[ix].set(ylabel="", xlabel=f"{var} ({var_units[var]})")
        # axs[ix].legend_.remove()
    sns.despine(bottom=True, left=True)
    plt.savefig("./Manuscript/Figures/dist_input_data.png", dpi=300)


if __name__ == "__main__":

    plt.close("all")
    sns.set_context("paper")
    mpl.rcParams["figure.figsize"] = [8.0, 3.5]
    sns.set_theme(style="whitegrid")

    var_names = {
        "Ta": r"$t_{db}$",
        "Tr": r"$\overline{t_{r}}$",
        "V": r"$V$",
        "Rh": r"$RH$",
        "Clo": r"$I_{cl}$",
        "Met": r"$M$",
    }

    var_units = {
        "Ta": r"$^{\circ}$C",
        "Tr": r"$^{\circ}$C",
        "V": r"m/s",
        "Rh": r"%",
        "Clo": r"clo",
        "Met": r"met",
    }

    # import data
    df = preprocess_comfort_db_data(limit=False, import_csv=True)

    df = filter_data(df_=df)
    df = calculate_new_indices(df_=df)

    print(df[df.V > 0.1].shape)
    print(df[df.V > 0.4].shape)
    print(df[df.V > 0.8].shape)
    print(df.shape[0])
    print(df[~df.TSV.isin(range(-3, 3))].shape[0])
    df_tpv = df.dropna(subset=["Thermal preference"])
    print(
        df_tpv[
            (df_tpv.TSV_round.isin([-1, 1]))
            & (df_tpv["Thermal preference"] == "no change")
        ].shape[0]
    )
    print(df_tpv.shape[0])

if __name__ == "__plot_figure__":

    plot_distribution_variable()

    # bar_chart(ind="pmv")
    bar_chart(data=df, ind="tsv")
    bar_chart(data=df[df.V > 0.1], ind="tsv")

    distributions_pmv(v_lower=False)
    # distributions_pmv(v_lower=0.1)

    # scatter_plot(ind="pmv")
    scatter_plot("tsv")

    # todo add counts per boxplot
    plot_error_prediction(data=df, ind="tsv")
    plot_error_prediction(data=df[df.V > 0.1], ind="tsv")

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ix, model in enumerate(["pmv_iso", "pmv_ashrae"]):
        sns.regplot(
            data=df,
            y=df[model],
            x="TSV",
            ax=axs[ix],
            x_ci="sd",
            x_estimator=np.mean,
            robust=True,
            ci=None,
        )
        x, y = df["TSV"].values, df[model].values
        x = sm.add_constant(x, prepend=False)
        lr = sm.OLS(y, x)
        results = lr.fit()
        print(results.summary())
        sns.lineplot([-3, 3], [-3, 3], ax=axs[ix])
        axs[ix].set(ylim=(-4, 4), xlim=(-4, 4))
        axs[ix].set_aspect("equal", adjustable="box")
        axs[ix].set(title=model)
    plt.tight_layout()
