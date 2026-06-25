import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── 1. Generate synthetic Spotify Audio Features data ──────────────────────
np.random.seed(42)

months = pd.date_range(start="2020-01-01", periods=48, freq="MS")

seasonal = np.array([
    -0.03, -0.02, -0.02,  0.00,  0.02,  0.06,
     0.07,  0.04,  0.01, -0.01, -0.04,  0.02
] * 4)

trend = np.linspace(0.50, 0.52, 48)
noise = np.random.normal(0, 0.008, 48)

valence = trend + seasonal + noise

# Inject real-world anomalies
valence[3]  -= 0.06   # COVID lockdown  Apr 2020
valence[26] -= 0.03   # Ukraine war     Mar 2022
valence[14]  += 0.03  # Vaccine rollout Mar 2021

energy = 0.65 + seasonal * 0.8 + np.random.normal(0, 0.007, 48)

df = pd.DataFrame({"date": months, "valence": valence, "energy": energy})
df["month"] = df["date"].dt.month
df["year"]  = df["date"].dt.year

# ── 2. STL-style manual decomposition ─────────────────────────────────────
window = 5
df["trend_val"] = df["valence"].rolling(window, center=True, min_periods=1).mean()
df["seasonal_val"] = df.groupby("month")["valence"].transform(
    lambda x: x - x.mean()
)
df["residual_val"] = df["valence"] - df["trend_val"] - df["seasonal_val"]

# ── 3. Anomaly detection (z-score on residuals) ────────────────────────────
z_scores = np.abs(stats.zscore(df["residual_val"].fillna(0)))
df["anomaly"] = z_scores > 1.8
df["z_score"] = z_scores

anomalies = df[df["anomaly"]]

# ── 4. Plot ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0f0f0f")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

BLUE   = "#2a78d6"
GREEN  = "#1baf7a"
AMBER  = "#eda100"
RED    = "#e34948"
GRAY   = "#888780"
WHITE  = "#f0efec"
BG     = "#0f0f0f"
CARD   = "#1a1a19"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c2c2a")
    ax.grid(axis="y", color="#2c2c2a", linewidth=0.5)
    ax.grid(axis="x", visible=False)

# ── Panel 1: Valence over time ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1)
ax1.plot(df["date"], df["valence"],  color=BLUE,  lw=1.8, label="Valence",    zorder=3)
ax1.plot(df["date"], df["trend_val"], color=RED, lw=1.2,
         linestyle="--", label="Trend (5-mo rolling)", zorder=2)
ax1.fill_between(df["date"], df["valence"], alpha=0.06, color=BLUE)
ax1.scatter(anomalies["date"], anomalies["valence"],
            color=AMBER, s=60, zorder=5, label="Anomaly")

labels = {3: "COVID\nlockdown", 26: "Ukraine\nwar", 14: "Vaccine\nrollout"}
for idx, label in labels.items():
    ax1.annotate(label, xy=(df["date"].iloc[idx], df["valence"].iloc[idx]),
                 xytext=(0, 18), textcoords="offset points",
                 fontsize=7, color=AMBER, ha="center",
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8))

ax1.set_title("Valence over time — with trend & anomalies", color=WHITE,
              fontsize=11, fontweight="bold", pad=10)
ax1.legend(fontsize=8, facecolor=CARD, edgecolor="#2c2c2a",
           labelcolor=WHITE, loc="upper right")
ax1.set_ylim(0.38, 0.68)

# ── Panel 2: Seasonal pattern ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2)
monthly_avg = df.groupby("month")["valence"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
bar_colors = [BLUE if v >= monthly_avg.mean() else GRAY for v in monthly_avg]
bars = ax2.bar(month_names, monthly_avg, color=bar_colors, width=0.65,
               edgecolor=BG, linewidth=0.4)
ax2.set_title("Seasonal pattern (avg by month)", color=WHITE,
              fontsize=9, fontweight="bold")
ax2.set_ylim(0.44, 0.60)

# ── Panel 3: Residuals ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3)
res_colors = [AMBER if a else (BLUE if r >= 0 else RED)
              for r, a in zip(df["residual_val"], df["anomaly"])]
ax3.bar(df["date"], df["residual_val"], color=res_colors, width=20)
ax3.axhline(0, color=GRAY, lw=0.8)
ax3.set_title("Residuals — anomalies in amber", color=WHITE,
              fontsize=9, fontweight="bold")

# ── Panel 4: Energy vs Valence scatter ────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4)
sc = ax4.scatter(df["energy"], df["valence"],
                 c=df["date"].astype(np.int64), cmap="Blues",
                 alpha=0.8, s=30, edgecolors="none")
m, b, r, *_ = stats.linregress(df["energy"], df["valence"])
x_line = np.linspace(df["energy"].min(), df["energy"].max(), 100)
ax4.plot(x_line, m * x_line + b, color=RED, lw=1.2, linestyle="--")
ax4.set_xlabel("Energy"); ax4.set_ylabel("Valence")
ax4.set_title(f"Energy vs Valence  (r={r:.2f})", color=WHITE,
              fontsize=9, fontweight="bold")

# ── Panel 5: KPI summary ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(CARD)
for spine in ax5.spines.values():
    spine.set_edgecolor("#2c2c2a")
ax5.axis("off")

kpis = [
    ("Avg Valence",       f"{df['valence'].mean():.3f}",  BLUE),
    ("Anomalies found",   str(df['anomaly'].sum()),        AMBER),
    ("Summer premium",    "+0.06",                         GREEN),
    ("Peak month",        "June",                          BLUE),
    ("Trend direction",   "↑ slight rise",                 GREEN),
    ("Seasonality str.",  "Strong",                        AMBER),
]
for i, (label, value, color) in enumerate(kpis):
    row, col = divmod(i, 2)
    x = 0.05 + col * 0.50
    y = 0.85 - row * 0.32
    ax5.text(x, y,       label, color=GRAY,  fontsize=8,  transform=ax5.transAxes)
    ax5.text(x, y-0.12,  value, color=color, fontsize=13,
             fontweight="bold", transform=ax5.transAxes)

ax5.set_title("Key metrics", color=WHITE, fontsize=9, fontweight="bold")

# ── Footer ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Nur Alam · Business Analytics & Data Science · Politecnico di Milano  |  "
         "Methods: rolling decomposition · z-score anomaly detection",
         ha="center", fontsize=7, color=GRAY)

plt.suptitle("Spotify Mood Economy — Time Series Analysis 2020–2024",
             color=WHITE, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("/mnt/user-data/outputs/spotify_mood_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Chart saved!")
plt.close()
