import plotly.graph_objects as go
import pandas as pd

# Define the categories and values for each model
categories = ['BERT Precision', 'BERT Recall', 'BERT F1', 'Rogue Precision', 'Rogue Recall', 'Rogue F1', 'BLEU']

# Data for each model
metrics = {
    'OneShot Knowledge': [0.90251,0.90228,0.9023949853,0.52395,0.651651,0.5808646666,0.1351],
    'Diagnostic Knowledge': [0.90251,0.90228,0.9022411089,0.5105,0.605,0.5537,0.1351],
    'In-Depth Medical Knowledge': [0.8990908245,0.8692042928,0.8835346324,0.624,0.655,0.6391,0.1079211581],
    'Med Alpaca Queries': [0.8847833495,0.9000967346,0.8919868455,0.604,0.63,0.6167,0.1079211581],
    # 'HuatuoGPT': [1.2, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1]
}

# Create figure
fig = go.Figure()

# Colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Add traces for each model
for idx, (metric, values) in enumerate(metrics.items()):
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        name=metric,
        line=dict(color=colors[idx]),
        # fill='tonext' if metric == 'In-Depth Medical Knowledge' else None  # Only fill for GPT-3.5-turbo as shown in image
    ))

# Update layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.2,
            showline=True,
            gridcolor='lightgray'
        ),
        angularaxis=dict(
            direction='clockwise',
            period=7
        )
    ),
    showlegend=True,
    title='Metrics Comparison for Different Datasets',
    width=800,
    height=800
)

# Show the plot
fig.show()

# Optional: Save the plot
# fig.write_html("radar_plot.html")  # Interactive HTML
# fig.write_image("radar_plot.png")  # Static image (requires kaleido package)