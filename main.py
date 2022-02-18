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

warnings.filterwarnings("ignore")


def preprocess_comfort_db_data(limit, import_csv):

    if import_csv:
        return pd.read_csv(r"./Data/DBII_preprocessed.csv")

    # import DB II data
    df_db2 = pd.read_csv(r"./Data/DatabaseI&II_20180703.csv", encoding="unicode_escape")
    # df_db2 = pd.read_csv(r"./Data/db-for-analysis.csv", encoding="unicode_escape")

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
                row["Tr"] = (row["Ta"] + row["To"]) / 2

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
    df_ = df_[(df_["pmv_iso"] > -3.5) & (df_["pmv_iso"] < 3.5)]
    df_ = df_[(df_["pmv_ashrae"] > -3.5) & (df_["pmv_ashrae"] < 3.5)]
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


if __name__ == "__main__":

    # import data
    df = preprocess_comfort_db_data(limit=False, import_csv=True)

    df = filter_data(df_=df)
    df = calculate_new_indices(df_=df)

    plt.close("all")

    sns.set_context("paper")
    mpl.rcParams["figure.figsize"] = [8.0, 6.0]

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    for ix, model in enumerate(["pmv_iso_round", "pmv_ashrae_round"]):
        _df = df.groupby(["TSV_round", model])["TSV_round"].count().unstack(model)
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        hist = df_rel.reset_index().plot(
            x="TSV_round",
            kind="bar",
            stacked=True,
            mark_right=True,
            width=0.95,
            rot=0,
            legend=False,
            ax=axs[ix],
            colormap="rainbow",
        )
        axs[ix].set(title=model)
        sns.despine(ax=axs[ix], left=True, bottom=True)

        if np.all(df_rel.index == df_rel.columns):
            diagonal = pd.Series(np.diag(df_rel), index=df_rel.index)

            for ix_s, value in diagonal.items():
                print(ix_s, value)
                if value != value:
                    value = 0
                axs[ix].text(
                    ix_s + 3,
                    105,
                    f"{value:.0f}%",
                    va="center",
                    ha="center",
                )

        for ix_s, value in enumerate(df.groupby(["TSV_round"])["TSV_round"].count()):
            axs[ix].text(
                ix_s,
                102,
                f"#{value:.0f}",
                va="center",
                ha="center",
            )

        # add percentages
        for index, row in df_rel.iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                print(ixe)
                if el > 7:
                    axs[ix].text(
                        index + 3,
                        cum_sum + el / 2,
                        f"{int(round(el, 0))}%",
                        va="center",
                        ha="center",
                    )
                cum_sum += el

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
    for ix, model in enumerate(["pmv_iso_round", "pmv_ashrae_round"]):
        _df = df.groupby(["TSV_round", model])[model].count().unstack("TSV_round")
        df_total = _df.sum(axis=1)
        df_rel = _df.div(df_total, 0) * 100
        hist = df_rel.reset_index().plot(
            x=model,
            kind="bar",
            stacked=True,
            mark_right=True,
            width=0.95,
            rot=0,
            legend=False,
            ax=axs[ix],
            colormap="rainbow",
        )
        axs[ix].set(title=model)

        if np.all(df_rel.index == df_rel.columns):
            diagonal = pd.Series(np.diag(df_rel), index=df_rel.index)

            for ix_s, value in diagonal.items():
                print(ix_s, value)
                if value != value:
                    value = 0
                axs[ix].text(
                    ix_s + 3,
                    105,
                    f"{value:.0f}%",
                    va="center",
                    ha="center",
                )

        for ix_s, value in enumerate(df.groupby([model])[model].count()):
            axs[ix].text(
                ix_s,
                102,
                f"#{value:.0f}",
                va="center",
                ha="center",
            )

        # add percentages
        df_rel.fillna(0, inplace=True)
        for index, row in df_rel.iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                if el > 7:
                    print(el)
                    axs[ix].text(
                        index + 3,
                        cum_sum + el / 2,
                        f"{int(round(el, 0))}%",
                        va="center",
                        ha="center",
                    )
                cum_sum += el

    # variables = ["pmv_iso", "pmv_ashrae", "TSV"]
    # f, axs = plt.subplots(len(variables), 1, sharex=True)
    # for ix, var in enumerate(variables):
    #     sns.histplot(data=df, x=var, kde=True, stat="density", ax=axs[ix])
    #     axs[ix].set(title=var)
    # plt.tight_layout()

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ix, model in enumerate(["pmv_iso", "pmv_ashrae"]):
        sns.regplot(data=df, y=df[model], x="TSV", ax=axs[ix], x_jitter=0.1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=df["TSV"], y=df[model]
        )
        sns.lineplot(x=[-6, 6], y=[-6, 6], ax=axs[ix])
        sns.lineplot(x=df["TSV"], y=intercept + df["TSV"] * slope, ax=axs[ix])
        # sns.scatterplot(data=df, y=df[model], x="TSV", s=5, color=".15", ax=axs[ix])
        # sns.histplot(data=df, y=df[model], x="TSV", bins=50, pthresh=.1, cmap="mako", ax=axs[ix])
        # sns.histplot(data=df, y=df[model], x="TSV", bins=50, cmap="mako", ax=axs[ix])
        # sns.kdeplot(data=df, y=df[model], x="TSV", levels=5, color="w", linewidths=1)
        axs[ix].set(ylim=(-3.5, 3.5), xlim=(-3.5, 3.5))
        axs[ix].set_aspect("equal", adjustable="box")
        axs[ix].set(title=model)
    plt.tight_layout()

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ix, model in enumerate(["pmv_iso", "pmv_ashrae"]):
        sns.regplot(data=df, x=df[model], y="TSV", ax=axs[ix], x_jitter=0.1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            y=df["TSV"], x=df[model]
        )
        slope, intercept, r_value, p_value, std_err = [
            round(x, 2) for x in [slope, intercept, r_value, p_value, std_err]
        ]
        sns.lineplot(x=[-3, 3], y=[-3, 3], ax=axs[ix])
        axs[ix].text(
            0.1,
            0.1,
            f"{slope =}, {intercept =}, {r_value =}, {p_value =}, {std_err =}",
            transform=axs[ix].transAxes,
        )
        sns.lineplot(x=df["TSV"], y=intercept + df["TSV"] * slope, ax=axs[ix])
        # sns.scatterplot(data=df, x=df[model], x="TSV", s=5, color=".15", ax=axs[ix])
        # sns.histplot(data=df, x=df[model], x="TSV", bins=50, pthresh=.1, cmap="mako", ax=axs[ix])
        # sns.histplot(data=df, x=df[model], x="TSV", bins=50, cmap="mako", ax=axs[ix])
        # sns.kdeplot(data=df, x=df[model], x="TSV", levels=5, color="w", linewidths=1)
        axs[ix].set(ylim=(-3, 3), xlim=(-3, 3))
        axs[ix].set_aspect("equal", adjustable="box")
        axs[ix].set(title=model)
    plt.tight_layout()

    # plot discrepancy
    # todo add counts per boxplot

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ix, model in enumerate(["diff_iso", "diff_ash"]):
        sns.boxenplot(data=df, x="TSV_round", y=df[model], ax=axs[ix])
        axs[ix].set_aspect("equal", adjustable="box")
        axs[ix].set(title=model)
        axs[ix].fill_between([-0.5, 6.5], 0.5, -0.5, color="red", alpha=0.5)
    plt.tight_layout()

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

    f, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    sns.histplot(data=df, x="SET", kde=True, stat="density", ax=ax[0])
    sns.histplot(data=df, x="set_py", kde=True, stat="density", ax=ax[1])
    sns.histplot(data=df, x="SET - set_py", kde=True, stat="density", ax=ax[2])
    ax[0].set(title="SET")
    ax[1].set(title="set_py")
    ax[2].set(title="SET - set_py", xlabel="")

    # df.to_csv(
    #     r"C:\Users\sbbfti\Google Drive\Shared File - can be deleted\db-results.csv",
    #     index=False,
    # )
    #
    # pmv_ppd(
    #     tdb=20.8,
    #     tr=20.8,
    #     vr=v_relative(0.05, 1),
    #     rh=59,
    #     met=1,
    #     clo=clo_dynamic(0.64, 1.0),
    #     standard="ashrae",
    # )
    #
    # round(-2.51)
