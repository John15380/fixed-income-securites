"""Export original-paper numerical checks and frozen-validation research figures."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache'/'matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(output=None):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    r = pd.read_csv(ROOT/'output'/'literature_replication'/'published_comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, name, title in zip(axes, ['FB in CP05 Table 2', 'CP05 restricted', 'LN09 LN6 released'],
                              ['FB: CP05 Table 2 (updated prices)', 'CP05: restricted common factor', 'LN09: released original factors']):
        q = r[r.model == name]
        x = np.arange(4)
        ax.bar(x-.18, 100*q.published, .36, label='Published', color='#afbfce')
        ax.bar(x+.18, 100*q.reproduced, .36, label='Reproduced', color='#175e70')
        ax.set_xticks(x, ['2y', '3y', '4y', '5y']); ax.set_title(title, fontsize=10)
        ax.set_ylabel('R-squared (%)' if name != 'LN09 LN6 released' else 'Adjusted R-squared (%)')
        ax.set_ylim(0, 135*max(q.published.max(), q.reproduced.max()))
        ax.legend(frameon=False, loc='upper left')
    fig.savefig(ROOT/'assets'/'literature_replication.png', dpi=180)
    plt.close(fig)

    folder = Path(output) if output is not None else ROOT/'output'/'validated_extension'
    scores = pd.read_csv(folder/'all_forecast_scores.csv')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    names = ['CP05', 'LN6', 'CP15', 'CP_EMA6', 'Full_OLS', 'PCA5_OLS', 'EN_real_cal', 'EN_GC_cal', 'XGB_real_cal', 'XGB_GC_cal', 'ValidationSelected']
    for j, panel in enumerate(['core_revised', 'headline_vintage']):
        ax = axes[0, j]
        q = scores[(scores.panel == panel) & (scores.horizon == 12) & (scores.period == 'test')]
        values = np.array([[q[(q.model == name) & (q.maturity == str(n))].r2_pct.iloc[0] for n in [2, 3, 4, 5]] for name in names])
        ax.imshow(values, cmap='RdBu', vmin=-40, vmax=40, aspect='auto')
        ax.set_yticks(range(len(names)), names); ax.set_xticks(range(4), ['2y', '3y', '4y', '5y'])
        ax.set_title(panel+': annual OOS R2 (%)')
        for row in range(len(names)):
            for col in range(4):
                v = values[row, col]
                label = f'{v:.1f}' if abs(v) < 1000 else f'{v:.1e}'
                ax.text(col, row, label, ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 20 else '#202c37')
        ax = axes[1, j]
        t = pd.read_csv(folder/panel/'h1'/'test_trades.csv')
        for name, label, color in [('XGB_GC_cal/frozen', 'XGB + copula, calibrated', '#186f72'),
                                   ('CP15/frozen', 'CP15 cycle allocation', '#b87324'),
                                   ('PriorEW3/frozen', 'Previously retained EW3', '#8060a3'),
                                   ('EH/frozen', 'Historical-mean allocation', '#727e8a'),
                                   ('BuyHold', 'Passive basket', '#ba4953')]:
            g = t[t.model == name].sort_values('purchase')
            dates = pd.PeriodIndex([g.purchase.iloc[0], *g.sale.tolist()], freq='M').to_timestamp()
            wealth = np.r_[1., np.cumprod(1+g.return_net)]
            ax.plot(dates, wealth, label=label, color=color, linewidth=1.8)
        ax.set_ylabel('Net wealth (initial = 1)'); ax.set_title(panel+': non-overlapping MONTHLY trades')
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.15)
    fig.suptitle('Frozen-validation extension | annual forecasts and monthly trading are distinct targets\nHeatmap colors clipped at +/-40%; annotations show the actual values', fontsize=12)
    fig.savefig(ROOT/'assets'/'validated_extension.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
