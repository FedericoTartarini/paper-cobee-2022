from scipy import optimize
from pythermalcomfort.models import pmv_ppd, set_tmp
from pythermalcomfort.utilities import v_relative, clo_dynamic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.close("all")


mean_ta = 27
sd_ta = 0.4
mean_rh = 50
sd_rh = 2.5
mean_v = 0.4
sd_v = 0.05
mean_met = 1.05
sd_met = 0.05
mean_clo = 0.64
sd_clo = 0.05

number_points = 1000
t_range = np.random.normal(mean_ta, sd_ta, number_points)
rh_range = np.random.normal(mean_rh, sd_rh, number_points)
v_range = np.random.normal(mean_v, sd_v, number_points)
v_range = [v if v > 0 else 0 for v in v_range]
met_range = np.random.normal(mean_met, sd_met, number_points)
clo_range = np.random.normal(mean_clo, sd_clo, number_points)


results = []

# for t in t_range:
#     for rh in rh_range:
#         for v in v_range:
#             for met in met_range:
#                 for clo in clo_range:
#                     results.append([t, rh, v, met, clo, pmv_ppd(t, t, v, rh, met, clo)['pmv']])

for iterations in range(1, 1000):
    t = t_range[np.random.randint(number_points)]
    rh_arr = rh_range[np.random.randint(number_points)]
    v = v_range[np.random.randint(number_points)]
    met_arr = met_range[np.random.randint(number_points)]
    clo = clo_range[np.random.randint(number_points)]
    v_r = v_relative(v, met_arr)
    clo_d = clo_dynamic(clo, met_arr)
    results.append(
        [
            t,
            rh_arr,
            v,
            met_arr,
            clo,
            clo_d,
            v_r,
            pmv_ppd(t, t, v_r, rh_arr, met_arr, clo_d, standard="ISO")["pmv"],
            pmv_ppd(t, t, v_r, rh_arr, met_arr, clo_d, standard="ASHRAE")["pmv"],
        ]
    )

df = pd.DataFrame(
    results,
    columns=["t", "rh", "v", "met", "clo", "clo_d", "v_r", "pmv_iso", "pmv_ashrae"],
)
df["delta_iso_ashrae"] = df["pmv_iso"] - df["pmv_ashrae"]

# plot_distribution(df["pmv"])


def plot_mean_sd(_ax, data):
    line = _ax.lines[0]
    mean = data.mean()
    std = data.std()
    xs = line.get_xdata()
    ys = line.get_ydata()
    height = np.interp(mean, xs, ys)
    xy = [val for val in zip(xs, ys) if (val[0] > mean - std) & (val[0] < mean + std)]
    x = [x[0] for x in xy]
    y = [y[1] for y in xy]
    _ax.vlines(mean, 0, height, color="k", ls=":")
    _ax.fill_between(x, 0, y, facecolor="k", alpha=0.8)
    _ax.text(
        0.85,
        0.7,
        r"$\mu$" + f"={mean:.2f}\nSD={std:.2f}",
        ha="center",
        va="center",
        transform=_ax.transAxes,
    )


def remove_label(_ax, var):
    _ax.set(xlabel="")
    map_var_names = {
        "t": r"$t$ (°C)",
        "tr": r"$t_r$ (°C)",
        "rh": "$RH$ (%)",
        "vr": r"$v_r$ (m/s)",
        "met": "$M$ (met)",
        "clo_d": r"$I_{cl} (clo)$",
        "pmv_iso": "PMV",
        "pmv_ashrae": r"PMV$_{CE}$",
    }
    _ax.text(
        0.1,
        0.7,
        map_var_names[var],
        ha="center",
        va="center",
        transform=_ax.transAxes,
    )


# sns.displot(data=df, x="t", y="pmv", kind="kde", rug=True)
f, ax = plt.subplots(4, 2, sharey=True, constrained_layout=True)
ax = ax.flat
sns.histplot(data=df, x="t", kde=True, stat="probability", ax=ax[0])
plot_mean_sd(_ax=ax[0], data=df["t"].values)
remove_label(_ax=ax[0], var="t")
sns.histplot(data=df, x="t", kde=True, stat="probability", ax=ax[1])
plot_mean_sd(_ax=ax[1], data=df["t"].values)
remove_label(_ax=ax[1], var="tr")
sns.histplot(data=df, x="rh", kde=True, stat="probability", ax=ax[2])
plot_mean_sd(_ax=ax[2], data=df["rh"].values)
remove_label(_ax=ax[2], var="rh")
# sns.histplot(
#     data=df, x="v", kde=True, stat="probability", ax=ax[3], fill=False, color="tab:red"
# )
# plot_mean_sd(_ax=ax[3], data=df["v"].values)
sns.histplot(data=df, x="v_r", kde=True, stat="probability", ax=ax[3])
plot_mean_sd(_ax=ax[3], data=df["v_r"].values)
remove_label(_ax=ax[3], var="vr")
sns.histplot(data=df, x="met", kde=True, stat="probability", ax=ax[4])
plot_mean_sd(_ax=ax[4], data=df["met"].values)
remove_label(_ax=ax[4], var="met")
# sns.histplot(
#     data=df,
#     x="clo",
#     kde=True,
#     stat="probability",
#     ax=ax[5],
#     fill=False,
#     color="tab:red",
# )
# plot_mean_sd(_ax=ax[5], data=df["clo"].values)
sns.histplot(data=df, x="clo_d", kde=True, stat="probability", ax=ax[5])
plot_mean_sd(_ax=ax[5], data=df["clo_d"].values)
remove_label(_ax=ax[5], var="clo_d")
# ax[3].text(0.05, 0.6, "blue - v_r\n red - v", transform=ax[3].transAxes)
# ax[5].text(0.05, 0.6, "blue - clo_d\n red - clo", transform=ax[5].transAxes)

# f, ax = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
sns.histplot(data=df, x="pmv_iso", kde=True, stat="probability", ax=ax[6])
plot_mean_sd(_ax=ax[6], data=df["pmv_iso"].values)
remove_label(_ax=ax[6], var="pmv_iso")
sns.histplot(data=df, x="pmv_ashrae", kde=True, stat="probability", ax=ax[7])
plot_mean_sd(_ax=ax[7], data=df["pmv_ashrae"].values)
remove_label(_ax=ax[7], var="pmv_ashrae")
plt.savefig("./Manuscript/Figures/error_analysis.png", dpi=300)


# sns.histplot(data=df, x="delta_iso_ashrae", kde=True, stat="density", ax=ax[2])
# ax[0].set(title="pmv_iso")
# ax[1].set(title="pmv_ashrae", xlabel="PMV")
# ax[2].set(title="pmv iso - pmv ashrae", xlabel="")

results = []
pmv_target_value = 0.0
# find the temperature for a specific pmv value
for iterations in range(1, 1000):
    rh_arr = rh_range[np.random.randint(number_points)]
    v = v_range[np.random.randint(number_points)]
    met_arr = met_range[np.random.randint(number_points)]
    clo = clo_range[np.random.randint(number_points)]
    v_r = v_relative(v, met_arr)
    clo_d = clo_dynamic(clo, met_arr)

    def function_iso(x):
        return (
            pmv_ppd(x, x, v_r, rh_arr, met_arr, clo_d, standard="ISO")["pmv"]
            - pmv_target_value
        )

    def function_ashrae(x):
        return (
            pmv_ppd(x, x, v_r, rh_arr, met_arr, clo_d, standard="ASHRAE")["pmv"]
            - pmv_target_value
        )

    try:
        t_iso = optimize.brentq(function_iso, 20, 40)
        t_ashrae = optimize.brentq(function_ashrae, 20, 40)
    except ValueError:
        t_iso = 9999
        t_ashrae = 9999

    results.append(
        [
            t_iso,
            t_ashrae,
            rh_arr,
            v,
            met_arr,
            clo,
            clo_d,
            v_r,
        ]
    )

df = pd.DataFrame(
    results,
    columns=["t_iso", "t_ashrae", "rh", "v", "met", "clo", "clo_d", "v_r"],
)
df["delta_iso_ashrae"] = df["t_iso"] - df["t_ashrae"]

f, ax = plt.subplots(3, 1, constrained_layout=True)
sns.histplot(data=df, x="t_iso", kde=True, stat="density", ax=ax[0])
sns.histplot(data=df, x="t_ashrae", kde=True, stat="density", ax=ax[1])
sns.histplot(data=df, x="delta_iso_ashrae", kde=True, stat="density", ax=ax[2])
ax[0].set(
    title=f"t_iso - pmv_target = {pmv_target_value}",
    xlim=(df["t_iso"].min(), df["t_ashrae"].max()),
)
ax[1].set(
    title=f"t_ashrae - pmv_target = {pmv_target_value}",
    xlim=(df["t_iso"].min(), df["t_ashrae"].max()),
)
ax[2].set(title="t_iso - t_ashrae", xlabel="")
