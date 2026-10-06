"""
Penn Electric Racing - sponsorship correlation heatmap (Seaborn).

Run
---
    python3 correlation_heatmap.py
    python3 correlation_heatmap.py --static --save overview.png
    python3 correlation_heatmap.py --static --view breakdown --save breakdown.png
    python3 correlation_heatmap.py --csv path/to/PER_sponsorship_dataset.csv

Requires: pandas, numpy, seaborn, matplotlib   (pip install pandas seaborn matplotlib)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Colours from the project brief: negative = red, none = white, positive = green
RED_WHITE_GREEN = LinearSegmentedColormap.from_list(
    "per_red_white_green", ["#b2182b", "#e8836f", "#ffffff", "#7cc47f", "#1b7837"])

# ---------------------------------------------------------------------------
# Variables: (label, stage, family, only-exists-when)
#   family  = options of the same question (can't be compared with each other)
#   only    = variable only exists when another variable is yes
# ---------------------------------------------------------------------------
VARIABLES = [
    ("Company size", "Company", None, None),
    ("Tier: Local", "Company", "tier", None),
    ("Tier: Mid-size", "Company", "tier", None),
    ("Tier: Corporate", "Company", "tier", None),
    ("Warmth of first contact", "Outreach", None, None),
    ("Channel: Cold email", "Outreach", "chan", None),
    ("Channel: Referral", "Outreach", "chan", None),
    ("Channel: Event", "Outreach", "chan", None),
    ("Channel: LinkedIn", "Outreach", "chan", None),
    ("Channel: Inbound", "Outreach", "chan", None),
    ("Follow-ups before reply", "Outreach", None, None),
    ("Days between follow-ups", "Outreach", None, None),
    ("We asked for a call", "Outreach", None, None),
    ("They offered a call", "Outreach", None, None),
    ("Team emails sent", "Outreach", None, None),
    ("Responded", "Result", None, None),
    ("Days to first response", "Result", None, "Responded"),
    ("Closed", "Result", None, None),
    ("Cash in deal", "Result", "deal", "Closed"),
    ("In-kind deal", "Result", "deal", "Closed"),
    ("Discount deal", "Result", "deal", "Closed"),
    ("Deal value ($)", "Result", None, "Closed"),
    ("Renewed next season", "Result", None, "Closed"),
]
ORDER = [v[0] for v in VARIABLES]
INFO = {v[0]: {"stage": v[1], "family": v[2], "only": v[3]} for v in VARIABLES}
DERIVED = {"Company size": "tier", "Warmth of first contact": "chan"}

VIEWS = {
    "overview": {
        "vars": ["Company size", "Warmth of first contact", "Follow-ups before reply",
                 "Days between follow-ups", "We asked for a call", "Responded",
                 "Days to first response", "Closed", "Cash in deal", "Deal value ($)",
                 "Renewed next season"],
        "title": "PER sponsorship: every step against every other",
    },
    "breakdown": {
        "y": ["Responded", "Days to first response", "Closed", "Cash in deal", "In-kind deal",
              "Discount deal", "Deal value ($)", "Renewed next season"],
        "x": ["Tier: Local", "Tier: Mid-size", "Tier: Corporate", "Channel: Cold email",
              "Channel: Referral", "Channel: Event", "Channel: LinkedIn", "Channel: Inbound"],
        "title": "PER sponsorship: results by tier and channel",
    },
}


def not_meaningful(a, b):
    """True when the pair is related by construction, so r would not be a finding."""
    A, B = INFO[a], INFO[b]
    if a == b:
        return True
    if A["family"] and A["family"] == B["family"]:
        return True                               # e.g. Tier: Local vs Tier: Corporate
    if (DERIVED.get(a) and DERIVED.get(a) == B["family"]) or (DERIVED.get(b) and DERIVED.get(b) == A["family"]):
        return True                               # Company size vs a tier option
    if A["only"] == b or B["only"] == a:
        return True                               # Deal value vs Closed
    if (A["only"] == "Closed" and b == "Responded") or (B["only"] == "Closed" and a == "Responded"):
        return True                               # closed deals had all replied
    return False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def build_features(raw):
    """One numeric column per variable (yes/no -> 1/0)."""
    f = pd.DataFrame(index=raw.index)
    tier, chan = raw["Sponsor_Tier"], raw["Outreach_Channel"]
    f["Company size"] = tier.map({"Local": 1, "Mid-size": 2, "Corporate": 3})
    for t in ["Local", "Mid-size", "Corporate"]:
        f[f"Tier: {t}"] = (tier == t).astype(float)
    f["Warmth of first contact"] = chan.map(
        {"Cold email": 0, "Referral": 1, "Event": 1, "LinkedIn": 1, "Inbound": 2})  # Existing -> blank
    for c in ["Cold email", "Referral", "Event", "LinkedIn", "Inbound"]:
        f[f"Channel: {c}"] = (chan == c).astype(float)
    num = lambda col: pd.to_numeric(raw[col], errors="coerce")
    f["Follow-ups before reply"] = num("Followups_Before_Reply")
    f["Days between follow-ups"] = num("Avg_Days_Between_Followups")
    f["We asked for a call"] = num("Asked_For_Call")
    f["They offered a call"] = num("Company_Offered_Call")
    f["Team emails sent"] = num("Team_Emails_Sent")
    f["Responded"] = num("Responded")
    f["Days to first response"] = num("Days_To_First_Response")
    closed = num("Closed")
    f["Closed"] = closed
    deal = raw["Deal_Type"]
    only_closed = lambda s: np.where(closed == 1, s.astype(float), np.nan)
    f["Cash in deal"] = only_closed(deal.isin(["Monetary", "Mixed"]))
    f["In-kind deal"] = only_closed(deal.isin(["In-kind", "Mixed"]))
    f["Discount deal"] = only_closed(deal == "Discount")
    f["Deal value ($)"] = num("Deal_Value_USD")
    f["Renewed next season"] = num("Renewed_Next_Season")
    return f


def usable(df, cols, min_group=3):
    """Keep columns that vary; yes/no columns need at least `min_group` of each answer."""
    keep = []
    for c in cols:
        v = df[c].dropna()
        if v.nunique() < 2:
            continue
        if set(v.unique()) <= {0.0, 1.0} and min(int(v.sum()), int((v == 0).sum())) < min_group:
            continue
        keep.append(c)
    return keep


def apply_filters(raw, tier="All", channel="All", exclude_existing=True):
    keep = pd.Series(True, index=raw.index)
    if tier != "All":
        keep &= raw["Sponsor_Tier"] == tier
    if channel != "All":
        keep &= raw["Outreach_Channel"] == channel
    if exclude_existing and channel != "Existing":
        keep &= raw["Outreach_Channel"] != "Existing"
    return raw[keep]


# ---------------------------------------------------------------------------
# Correlation + heatmap (extends the example function)
# ---------------------------------------------------------------------------
def correlation_table(df, x_vars, y_vars, method="pearson", min_periods=8):
    """(r, n, n/a-mask) tables with y_vars as rows and x_vars as columns."""
    cols = list(dict.fromkeys(list(y_vars) + list(x_vars)))
    corr = df[cols].corr(method=method, min_periods=min_periods).loc[y_vars, x_vars]
    notna = df[cols].notna().astype(int)
    n = notna.T.dot(notna).loc[y_vars, x_vars]
    na = pd.DataFrame([[not_meaningful(y, x) for x in x_vars] for y in y_vars],
                      index=y_vars, columns=x_vars)
    corr = corr.mask(na)
    return corr, n, na


def _draw(ax, corr, na, tri_mask, method, cbar_ax=None):
    mask = na.values | tri_mask
    sns.heatmap(corr, mask=mask, annot=True, fmt="+.2f",
                annot_kws={"size": 8 if corr.shape[1] > 9 else 9},
                cmap=RED_WHITE_GREEN, vmin=-1, vmax=1, center=0,
                linewidths=0.8, linecolor="white", ax=ax, cbar_ax=cbar_ax,
                cbar_kws={"label": f"{method.title()} correlation (r)"})
    # label pairs that are true by definition, instead of leaving them blank
    for i, j in zip(*np.where(na.values & ~tri_mask)):
        ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=7, color="#8a938e")
    # "blank = not enough data" cells (fewer than 8 companies) stay empty
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=40, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")


def plot_correlation_heatmap(
    df,
    method="pearson",       # 'pearson', 'spearman', or 'kendall'
    mask_upper=True,        # square view: hide the redundant upper triangle
    figsize=(11, 9),
    title="PER Sponsorship Correlations",
    min_periods=8,          # need at least this many companies with both values
    save_path=None,
    x_vars=None,            # columns on the x-axis (default: all, in pipeline order)
    y_vars=None,            # rows on the y-axis (default: same as x)
    show=True,
    n_companies=None,
):
    """Compute and plot a correlation heatmap. Returns the correlation table."""
    x_vars = usable(df, list(x_vars or ORDER))
    y_vars = usable(df, list(y_vars or x_vars))
    square = mask_upper and x_vars == y_vars
    if square:  # like the Titanic example, minus the empty first row / last column
        y_vars, x_vars = x_vars[1:], x_vars[:-1]
    corr, n, na = correlation_table(df, x_vars, y_vars, method, min_periods)
    tri = np.zeros(corr.shape, dtype=bool)
    if square:
        tri = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    _draw(ax, corr, na, tri, method)
    ax.set_title(f"{title}  ·  {n_companies or len(df)} companies", loc="left", fontsize=12)
    plt.figtext(0.01, 0.01, "Green = positive, white = none, red = negative. n/a = true by definition. "
                "Blank = fewer than 8 companies.", fontsize=8, color="#555")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return corr


# ---------------------------------------------------------------------------
# Interactive window (matplotlib widgets)
# ---------------------------------------------------------------------------
def interactive_heatmap(raw):
    from matplotlib.widgets import CheckButtons, RadioButtons

    state = {"view": "overview", "method": "pearson", "tier": "All", "channel": "All",
             "exclude": True}
    fig = plt.figure(figsize=(16, 10))
    heat_ax = fig.add_axes([0.33, 0.22, 0.55, 0.7])
    cbar_ax = fig.add_axes([0.9, 0.22, 0.012, 0.7])

    def radio(rect, title, labels):
        a = fig.add_axes(rect)
        a.set_title(title, fontsize=9, loc="left")
        r = RadioButtons(a, labels, active=0)
        for t in r.labels:
            t.set_fontsize(9)
        return r

    r_view = radio([0.02, 0.83, 0.13, 0.1], "View", ["Overview", "By tier & channel"])
    r_method = radio([0.02, 0.68, 0.13, 0.11], "Method", ["pearson", "spearman", "kendall"])
    r_tier = radio([0.02, 0.5, 0.13, 0.14], "Sponsor tier", ["All", "Local", "Mid-size", "Corporate"])
    r_chan = radio([0.02, 0.25, 0.13, 0.21], "Channel",
                   ["All", "Cold email", "Referral", "Event", "Inbound", "Existing"])
    c_ex = CheckButtons(fig.add_axes([0.02, 0.16, 0.15, 0.06]), ["Exclude existing sponsors"], [True])
    c_ex.labels[0].set_fontsize(9)
    readout = fig.text(0.33, 0.07, "Hover a cell to see r and how many companies it is based on.", fontsize=10)
    fig.text(0.33, 0.03, "Green = positive, white = none, red = negative. n/a = true by definition "
             "(e.g. two tiers, or deal value vs closed). Blank = fewer than 8 companies.",
             fontsize=8, color="#555")

    def redraw(*_):
        heat_ax.clear()
        cbar_ax.clear()
        d = apply_filters(raw, state["tier"], state["channel"], state["exclude"])
        feats = build_features(d)
        if state["view"] == "overview":
            vs = usable(feats, VIEWS["overview"]["vars"])
            ys, xs, square = vs[1:], vs[:-1], True
        else:
            ys, xs, square = usable(feats, VIEWS["breakdown"]["y"]), usable(feats, VIEWS["breakdown"]["x"]), False
        if not xs or not ys or len(d) < 8:
            heat_ax.text(0.5, 0.5, "Not enough data for this filter - widen it.",
                         ha="center", va="center", transform=heat_ax.transAxes)
            heat_ax.set_axis_off()
            fig.canvas.draw_idle()
            return
        corr, n, na = correlation_table(feats, xs, ys, state["method"])
        tri = np.triu(np.ones(corr.shape, dtype=bool), k=1) if square else np.zeros(corr.shape, dtype=bool)
        _draw(heat_ax, corr, na, tri, state["method"], cbar_ax=cbar_ax)
        heat_ax.set_title(f"{VIEWS[state['view']]['title']}  ·  {len(d)} companies", loc="left")
        heat_ax._per = (corr, n, na, tri)
        fig.canvas.draw_idle()

    def on_hover(event):
        if event.inaxes is not heat_ax or not hasattr(heat_ax, "_per") or event.xdata is None:
            return
        corr, n, na, tri = heat_ax._per
        col, row = int(event.xdata), int(event.ydata)
        if row >= corr.shape[0] or col >= corr.shape[1] or tri[row, col]:
            return
        y, x, r = corr.index[row], corr.columns[col], corr.iat[row, col]
        if na.iat[row, col]:
            text = f"{y}  ×  {x}:  n/a - related by definition, not a finding"
        elif pd.isna(r):
            text = f"{y}  ×  {x}:  not enough data (n = {n.iat[row, col]})"
        else:
            a = abs(r)
            strength = "strong" if a >= .5 else "moderate" if a >= .3 else "weak" if a >= .1 else "no clear"
            sign = ("positive " if r > 0 else "negative ") if a >= .1 else ""
            text = f"{y}  ×  {x}:  r = {r:+.2f}  ({strength} {sign}relationship)   ·   n = {n.iat[row, col]} companies"
        readout.set_text(text)
        fig.canvas.draw_idle()

    def setter(key, transform=lambda v: v):
        def f(label):
            state[key] = transform(label)
            redraw()
        return f

    r_view.on_clicked(setter("view", lambda v: "overview" if v == "Overview" else "breakdown"))
    r_method.on_clicked(setter("method"))
    r_tier.on_clicked(setter("tier"))
    r_chan.on_clicked(setter("channel"))
    c_ex.on_clicked(lambda _: (state.update(exclude=not state["exclude"]), redraw()))
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig._per_widgets = [r_view, r_method, r_tier, r_chan, c_ex]  # keep references alive
    redraw()
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(here, "PER_sponsorship_dataset.csv"))
    p.add_argument("--static", action="store_true", help="plain seaborn figure, no controls")
    p.add_argument("--view", choices=list(VIEWS), default="overview")
    p.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson")
    p.add_argument("--tier", default="All")
    p.add_argument("--channel", default="All")
    p.add_argument("--include-existing", action="store_true")
    p.add_argument("--save", default=None, help="save the static figure to this file")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Dataset not found: {args.csv}\nPass --csv /path/to/PER_sponsorship_dataset.csv")
    raw_df = load_dataset(args.csv)

    if args.static:
        d = apply_filters(raw_df, args.tier, args.channel, not args.include_existing)
        feats = build_features(d)
        v = VIEWS[args.view]
        corr_matrix = plot_correlation_heatmap(
            feats, method=args.method,
            x_vars=v.get("vars", v.get("x")), y_vars=v.get("vars", v.get("y")),
            title=v["title"], figsize=(11, 9) if args.view == "overview" else (12, 7),
            save_path=args.save, show=args.save is None, n_companies=len(d),
        )
        print(corr_matrix.round(2).to_string())
    else:
        interactive_heatmap(raw_df)
        plt.show()
