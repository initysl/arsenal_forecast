import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load simulation results
# EPL Title Race
epl_data = {
    'Arsenal': 65.1,
    'Man City': 23.8,
    'Level': 11.1
}

# UCL
ucl_data = pd.read_csv('outputs/simulations/ucl_probability.txt', sep=': ', names=['metric', 'value'], engine='python')
arsenal_ucl_win = float(ucl_data[ucl_data['metric'] == 'Arsenal UCL Win Probability']['value'].values[0].strip('%'))
arsenal_ucl_advance = float(ucl_data[ucl_data['metric'] == 'Arsenal advance from semi-final']['value'].values[0].strip('%'))

print("Creating visualizations...")

# Figure 1: EPL Title Race
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Arsenal FC 2025/26 Season Forecast', fontsize=16, fontweight='bold')

# 1. EPL Title Probabilities
ax1 = axes[0, 0]
colors = ['#EF0107', '#6CABDD', '#CCCCCC']  # Arsenal red, City blue, Gray
bars = ax1.bar(['Arsenal', 'Man City', 'Tied on Points'], 
               [epl_data['Arsenal'], epl_data['Man City'], epl_data['Level']], 
               color=colors, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax1.set_ylabel('Probability (%)', fontsize=12)
ax1.set_title('Premier League Title Winner', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# 2. UCL Probabilities
ax2 = axes[0, 1]
ucl_outcomes = ['Win UCL', 'Advance\nfrom Semi', 'Eliminated']
ucl_probs = [arsenal_ucl_win, arsenal_ucl_advance - arsenal_ucl_win, 100 - arsenal_ucl_advance]
colors_ucl = ['#EF0107', '#FDB913', '#023474']  # Arsenal red, Yellow, Navy

bars2 = ax2.bar(ucl_outcomes, ucl_probs, color=colors_ucl, edgecolor='black', linewidth=1.5)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel('Probability (%)', fontsize=12)
ax2.set_title('Champions League Progression', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)

# 3. Combined Trophy Scenarios
ax3 = axes[1, 0]

# Calculate combined probabilities
p_epl_only = (epl_data['Arsenal'] / 100) * (1 - arsenal_ucl_win / 100)
p_ucl_only = (1 - epl_data['Arsenal'] / 100) * (arsenal_ucl_win / 100)
p_double = (epl_data['Arsenal'] / 100) * (arsenal_ucl_win / 100)
p_neither = (1 - epl_data['Arsenal'] / 100) * (1 - arsenal_ucl_win / 100)

scenarios = ['Double\n(Both)', 'EPL Only', 'UCL Only', 'No Trophies']
probs = [p_double * 100, p_epl_only * 100, p_ucl_only * 100, p_neither * 100]
colors_scenarios = ['#FFD700', '#EF0107', '#023474', '#999999']  # Gold, Red, Navy, Gray

bars3 = ax3.bar(scenarios, probs, color=colors_scenarios, edgecolor='black', linewidth=1.5)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax3.set_ylabel('Probability (%)', fontsize=12)
ax3.set_title('Multi-Trophy Scenarios', fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(probs) * 1.2)
ax3.grid(axis='y', alpha=0.3)

# 4. Summary Stats Table
ax4 = axes[1, 1]
ax4.axis('off')

summary_data = [
    ['Metric', 'Probability'],
    ['Win EPL Title', f"{epl_data['Arsenal']:.1f}%"],
    ['Win UCL Trophy', f"{arsenal_ucl_win:.1f}%"],
    ['Win Both (Double)', f"{p_double * 100:.1f}%"],
    ['Win At Least One', f"{(p_double + p_epl_only + p_ucl_only) * 100:.1f}%"],
    ['', ''],
    ['Expected EPL Points', '81.2 ± 2.4'],
    ['Man City Expected', '79.2 ± 2.8'],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#023474')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax4.set_title('Season Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/figures/season_forecast.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/figures/season_forecast.png")

# Figure 2: Scenario Waterfall
fig2, ax = plt.subplots(figsize=(10, 6))

scenarios_detailed = [
    'Current\nPosition',
    'Win EPL\nTitle',
    'Win UCL\nTrophy', 
    'Win\nDouble'
]

values = [
    100,  # Start
    epl_data['Arsenal'],
    arsenal_ucl_win,
    p_double * 100
]

colors_waterfall = ['#CCCCCC', '#EF0107', '#023474', '#FFD700']

x_pos = np.arange(len(scenarios_detailed))
bars = ax.bar(x_pos, values, color=colors_waterfall, edgecolor='black', linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios_detailed, fontsize=11)
ax.set_ylabel('Probability (%)', fontsize=12)
ax.set_title('Arsenal 2025/26: Path to Glory', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/scenario_waterfall.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/figures/scenario_waterfall.png")

plt.show()

print("\nVisualization complete!")
print(f"\nKey Findings:")
print(f"- Arsenal has a {epl_data['Arsenal']:.1f}% chance to win the Premier League")
print(f"- Arsenal has a {arsenal_ucl_win:.1f}% chance to win the Champions League")
print(f"- Arsenal has a {p_double * 100:.1f}% chance to win BOTH trophies (the Double)")
print(f"- Arsenal has a {(p_double + p_epl_only + p_ucl_only) * 100:.1f}% chance to win at least one trophy")