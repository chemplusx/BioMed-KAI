import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the figure style parameters
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# Create sample data for each model
def create_sample_data(mean, sd, n=100):
    return np.random.normal(mean, sd, n)

# Data for Treatment suggestion performance (plot a)
eval_metrics_kl_divergence = {
    'OS': {'data': create_sample_data(15.89, 4.539), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(14.81, 2.001), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(6, 2.447), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(21.06, 10.75), 'color': '#d62728', 'marker': 'D'}
}

eval_metrics_sim = {
    'OS': {'data': create_sample_data(80.0388, 1.9101), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(79.01, 2.001), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(81.33, 1.02), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(77.98, 3.01), 'color': '#d62728', 'marker': 'D'}
}


eval_metrics_hpi = {
    'OS': {'data': create_sample_data(80.0388, 1.9101), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(79.01, 2.001), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(81.33, 1.02), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(77.98, 3.01), 'color': '#d62728', 'marker': 'D'}
}

# Data for BERT Score metrics
bert_score_precision = {
    'OS': {'data': create_sample_data(90.24, 1.6), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(90.22, 2.6), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(88.35, 1.1), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(89.20, 3.16), 'color': '#d62728', 'marker': 'D'}
}

bert_score_recall = {
    'OS': {'data': create_sample_data(88.5, 1.8), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(87.8, 2.2), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(86.9, 1.4), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(87.5, 2.8), 'color': '#d62728', 'marker': 'D'}
}

bert_score_f1 = {
    'OS': {'data': create_sample_data(89.3, 1.5), 'color': '#1f77b4', 'marker': 'o'},
    'D': {'data': create_sample_data(89.0, 2.4), 'color': '#2ca02c', 'marker': 's'},
    'IDM': {'data': create_sample_data(87.6, 1.2), 'color': '#ff7f0e', 'marker': '^'},
    'MA': {'data': create_sample_data(88.3, 3.0), 'color': '#d62728', 'marker': 'D'}
}

# Create figure with complex subplot layout
fig = plt.figure(figsize=(15, 10))

# Create GridSpec with 2 rows and 3 columns
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Create the top subplot (spans all columns)
ax1 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax13 = fig.add_subplot(gs[0, 2])

# Create the bottom three subplots
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

def create_boxplot(data_dict, ax, title, metric="Score", zero_shot_models=['OS', 'MA'], show_legend=False):
    # Prepare data and colors
    data = {k: v['data'] for k, v in data_dict.items()}
    colors = [v['color'] for v in data_dict.values()]
    markers = [v['marker'] for v in data_dict.values()]
    
    # Convert data to long format
    df_long = pd.DataFrame(data)
    df_melted = df_long.melt(var_name='Model', value_name=metric)
    
    # Create box plot
    bp = sns.boxplot(data=df_melted, x='Model', y=metric, ax=ax, 
                palette=colors, width=0.7, fliersize=5)
    
    # Add scatter points for each data point
    for i, (model, values) in enumerate(data.items()):
        ax.scatter([i] * len(values), values, 
                  color=data_dict[model]['color'],
                  marker=data_dict[model]['marker'],
                  alpha=0.5, s=30)
    
    # Customize plot
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)  # Set rotation to 0
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Remove x-axis labels
    ax.set_xticklabels([])
    
    # Add mean ± SD annotations
    for i, (model, values) in enumerate(data.items()):
        mean = np.mean(values)
        sd = np.std(values)
        ax.text(i, ax.get_ylim()[1], f'{mean:.1f}±{sd:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Add zero-shot box
    if zero_shot_models:
        rect = plt.Rectangle(
            (len(data) - len(zero_shot_models) - 0.5, ax.get_ylim()[0]),
            len(zero_shot_models), ax.get_ylim()[1] - ax.get_ylim()[0],
            fill=False, color='red', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(len(data) - len(zero_shot_models)/2 - 0.5, 
                ax.get_ylim()[1], 'Zero-shot', 
                ha='center', va='bottom', color='red', fontsize=10)
    
    # Add legend with full model names
    if show_legend:
        legend_labels = {
            'OS': 'OneShot Knowledge',
            'D': 'Diagnostic',
            'IDM': 'In-Depth Medical Knowledge',
            'MA': 'Med-Alpaca'
        }
        legend_elements = [plt.Line2D([0], [0], color=v['color'], marker=v['marker'],
                                    label=legend_labels[k], markersize=8, linestyle='None')
                         for k, v in data_dict.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

# Create the box plots
create_boxplot(eval_metrics_sim, ax1, 'Semantic Relevance')
create_boxplot(eval_metrics_kl_divergence, ax12, 'KL Divergence')
create_boxplot(eval_metrics_hpi, ax13, 'Holistic Performace Index (HPI)', show_legend=True)
create_boxplot(bert_score_precision, ax2, 'BERT Score - Precision')
create_boxplot(bert_score_recall, ax3, 'BERT Score - Recall')
create_boxplot(bert_score_f1, ax4, 'BERT Score - F1')

# Adjust layout
plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('medical_performance_plots.png', dpi=300, bbox_inches='tight')