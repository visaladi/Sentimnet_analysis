import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def save_coin_chart(coin, flows, save_dir):
    total_mentions = len(flows)
    pos_mentions = sum(1 for v in flows if v > 0)
    neg_mentions = sum(1 for v in flows if v < 0)
    pos_pct = round(pos_mentions / total_mentions * 100, 2) if total_mentions else 0
    neg_pct = round(100 - pos_pct, 2)

    total_flow = sum(flows)
    pos_flow = sum(v for v in flows if v > 0)
    neg_flow = sum(v for v in flows if v < 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    ax.set_title(f"{coin} Community Mentions", color='white')

    # Bar chart
    ax.bar(['Positive', 'Negative'], [pos_pct, neg_pct], color=['limegreen', 'red'])

    # Add value labels
    ax.text(0, pos_pct + 1, f'{pos_pct:.2f}%', color='white', ha='center')
    ax.text(1, neg_pct + 1, f'{neg_pct:.2f}%', color='white', ha='center')

    # Add trader insight box
    trader_box = f"Winning: {pos_mentions} (+${int(pos_flow):,})\nLosing: {neg_mentions} (-${int(abs(neg_flow)):,})\nOffloading: {total_mentions} (${int(abs(total_flow)):,})"
    ax.text(2.5, 50, trader_box, color='white', va='center', bbox=dict(facecolor='#222222', edgecolor='white'))

    ax.set_ylim(0, 120)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{coin}.png"), facecolor=fig.get_facecolor())
    plt.close()
