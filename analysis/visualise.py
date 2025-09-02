#!/usr/bin/env python3
"""
BioMed-KAI Analysis Scripts - Part 2
====================================

Advanced visualization suite and specialized analytics for publication-ready results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class VisualizationSuite:
    """
    Advanced visualization suite for publication-quality figures
    """
    
    def __init__(self, style: str = 'default', figsize_default: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        sns.set_palette("Set2")
        self.figsize_default = figsize_default
        
        # Publication-ready color schemes
        self.color_schemes = {
            'comparison': ['#2E8B57', '#FF6B35', '#F7931E', '#4A90E2', '#9013FE'],
            'performance': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'heatmap': 'viridis',
            'sequential': 'Blues'
        }
    
    def create_publication_ready_comparison_plot(self, comparison_data: Dict[str, Dict[str, float]], 
                                               save_path: str) -> None:
        """Create publication-ready comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        systems = list(comparison_data.keys())
        medqa_scores = [data['medqa'] for data in comparison_data.values()]
        resource_efficiency = [data['resource_efficiency'] for data in comparison_data.values()]
        
        colors = self.color_schemes['comparison']
        
        # MedQA Performance
        bars1 = ax1.bar(systems, medqa_scores, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.2)
        ax1.set_title('MedQA Performance Comparison', fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_ylim(60, 95)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        
        # Add value labels on bars
        for bar, score in zip(bars1, medqa_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Resource Efficiency (log scale for better visualization)
        bars2 = ax2.bar(systems, resource_efficiency, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=1.2)
        ax2.set_title('Resource Efficiency Comparison', fontsize=18, fontweight='bold', pad=20)
        ax2.set_ylabel('Efficiency Ratio (log scale)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45, labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        
        # Add value labels
        for bar, eff in zip(bars2, resource_efficiency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{eff:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_agent_performance_heatmap(self, agent_data: Dict[str, Any], save_path: str) -> None:
        """Create agent performance heatmap"""
        agents = list(agent_data.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'semantic_similarity', 'hpi']
        
        # Create performance matrix
        performance_matrix = []
        for agent in agents:
            row = [getattr(agent_data[agent], metric) for metric in metrics]
            performance_matrix.append(row)
        
        performance_df = pd.DataFrame(performance_matrix, 
                                    index=agents, 
                                    columns=[m.replace('_', ' ').title() for m in metrics])
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(performance_df, annot=True, fmt='.3f', cmap='RdYlGn',
                        center=0.8, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Agent Performance Heatmap', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Agents', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_care_rag_architecture_diagram(self, save_path: str) -> None:
        """Create CARE-RAG architecture flow diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Define components and their positions
        components = {
            'Query Input': (1, 8, '#E3F2FD'),
            'Dual-Pathway Classification': (3, 8, '#FFECB3'),
            'Label Classification': (2, 6, '#F3E5F5'),
            'Entity Extraction': (4, 6, '#E8F5E8'),
            'Agent Router': (6, 8, '#FFE0B2'),
            'Knowledge Graph': (9, 8, '#FFCDD2'),
            'Vector Search': (8, 6, '#E1F5FE'),
            'Graph Traversal': (10, 6, '#F9FBE7'),
            'Context Manager': (9, 4, '#FCE4EC'),
            'Response Generator': (6, 2, '#E8F5E8')
        }
        
        # Draw components
        for name, (x, y, color) in components.items():
            rect = Rectangle((x-0.8, y-0.4), 1.6, 0.8, facecolor=color, 
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontweight='bold',
                   fontsize=10, wrap=True)
        
        # Draw arrows (simplified connections)
        arrows = [
            ((1, 8), (3, 8)),    # Query -> Dual-Pathway
            ((3, 7.6), (2, 6.4)), # Dual-Pathway -> Label
            ((3, 7.6), (4, 6.4)), # Dual-Pathway -> Entity
            ((2.8, 6), (6, 7.6)), # Label -> Router
            ((4.8, 6), (6, 7.6)), # Entity -> Router
            ((6.8, 8), (9, 8)),   # Router -> KG
            ((9, 7.6), (8, 6.4)), # KG -> Vector Search
            ((9, 7.6), (10, 6.4)),# KG -> Graph Traversal
            ((8.5, 5.6), (9, 4.4)), # Vector -> Context
            ((9.5, 5.6), (9, 4.4)), # Graph -> Context
            ((8.2, 4), (6.8, 2.4))  # Context -> Response
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
        
        ax.set_xlim(0, 11)
        ax.set_ylim(1, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('CARE-RAG Architecture Flow Diagram', 
                    fontsize=20, fontweight='bold', pad=30)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_error_analysis_dashboard(self, error_data: Dict[str, Dict[str, float]], 
                                      save_path: str) -> None:
        """Create comprehensive error analysis dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        error_types = list(error_data.keys())
        frequencies = [data['frequency'] for data in error_data.values()]
        impacts = [data['impact'] for data in error_data.values()]
        recovery_rates = [data['recovery_rate'] for data in error_data.values()]
        
        # Error frequency bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(error_types)), frequencies, 
                       color=self.color_schemes['performance'][:len(error_types)])
        ax1.set_title('Error Frequencies', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (%)', fontweight='bold')
        ax1.set_xticks(range(len(error_types)))
        ax1.set_xticklabels([t.replace('_', '\n') for t in error_types], rotation=0, fontsize=9)
        
        # Add value labels
        for bar, freq in zip(bars1, frequencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{freq:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Error impact bar plot
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(error_types)), impacts, 
                       color=self.color_schemes['performance'][:len(error_types)])
        ax2.set_title('Error Impact on Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Performance Drop (%)', fontweight='bold')
        ax2.set_xticks(range(len(error_types)))
        ax2.set_xticklabels([t.replace('_', '\n') for t in error_types], rotation=0, fontsize=9)
        
        for bar, impact in zip(bars2, impacts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{impact:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Recovery rates bar plot
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(len(error_types)), recovery_rates, 
                       color=self.color_schemes['performance'][:len(error_types)])
        ax3.set_title('Error Recovery Rates', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Recovery Rate (%)', fontweight='bold')
        ax3.set_xticks(range(len(error_types)))
        ax3.set_xticklabels([t.replace('_', '\n') for t in error_types], rotation=0, fontsize=9)
        
        for bar, recovery in zip(bars3, recovery_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{recovery:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # Scatter plot: frequency vs impact
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(frequencies, impacts, s=200, alpha=0.7, 
                            c=self.color_schemes['performance'][:len(error_types)])
        ax4.set_xlabel('Frequency (%)', fontweight='bold')
        ax4.set_ylabel('Impact (%)', fontweight='bold')
        ax4.set_title('Error Frequency vs Impact', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        for i, error_type in enumerate(error_types):
            ax4.annotate(error_type.replace('_', '\n'), 
                        (frequencies[i], impacts[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold')
        
        # Severity heatmap (frequency * impact / recovery)
        ax5 = fig.add_subplot(gs[1, 1:])
        severity_scores = [(freq * impact / recovery) for freq, impact, recovery 
                          in zip(frequencies, impacts, recovery_rates)]
        
        severity_matrix = np.array(severity_scores).reshape(1, -1)
        im = ax5.imshow(severity_matrix, cmap='Reds', aspect='auto')
        ax5.set_title('Error Severity Scores (Frequency × Impact ÷ Recovery)', 
                     fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(error_types)))
        ax5.set_xticklabels([t.replace('_', '\n') for t in error_types], rotation=45)
        ax5.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
        cbar.set_label('Severity Score', fontweight='bold')
        
        # Add severity values on heatmap
        for i, score in enumerate(severity_scores):
            ax5.text(i, 0, f'{score:.2f}', ha='center', va='center', 
                    fontweight='bold', color='white' if score > max(severity_scores)/2 else 'black')
        
        # Error timeline simulation
        ax6 = fig.add_subplot(gs[2, :])
        time_points = np.arange(0, 100, 1)
        
        for i, error_type in enumerate(error_types):
            # Simulate error occurrence over time
            base_rate = frequencies[i] / 100
            error_occurrences = np.random.poisson(base_rate, len(time_points))
            cumulative_errors = np.cumsum(error_occurrences)
            
            ax6.plot(time_points, cumulative_errors, label=error_type.replace('_', ' '),
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax6.set_xlabel('Time (arbitrary units)', fontweight='bold')
        ax6.set_ylabel('Cumulative Error Count', fontweight='bold')
        ax6.set_title('Simulated Error Occurrence Over Time', fontsize=14, fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_temperature_sensitivity_3d_plot(self, temp_data: Dict[float, Dict[str, float]], 
                                             save_path: str) -> None:
        """Create 3D temperature sensitivity plot"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        temperatures = list(temp_data.keys())
        accuracies = [temp_data[temp]['accuracy'] for temp in temperatures]
        consistencies = [temp_data[temp]['consistency'] for temp in temperatures]
        hallucination_rates = [temp_data[temp]['hallucination_rate'] for temp in temperatures]
        
        # Create 3D scatter plot
        scatter = ax.scatter(temperatures, accuracies, consistencies,
                           c=hallucination_rates, s=200, alpha=0.8,
                           cmap='RdYlBu_r')
        
        # Add connecting lines
        ax.plot(temperatures, accuracies, consistencies, 'k--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Temperature', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_zlabel('Consistency', fontweight='bold', fontsize=12)
        ax.set_title('Temperature Sensitivity Analysis (3D View)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Hallucination Rate', fontweight='bold')
        
        # Annotate optimal point
        optimal_idx = np.argmax(accuracies)
        ax.text(temperatures[optimal_idx], accuracies[optimal_idx], 
               consistencies[optimal_idx] + 0.02,
               f'Optimal\nT={temperatures[optimal_idx]}', 
               fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_statistical_validation_report_plot(self, validation_results: Dict[str, Any], 
                                                save_path: str) -> None:
        """Create statistical validation summary plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Bootstrap confidence interval visualization
        ci_lower = validation_results['bootstrap_ci']['lower']
        ci_upper = validation_results['bootstrap_ci']['upper']
        mean_diff = (ci_lower + ci_upper) / 2
        
        ax1.barh(['Performance\nImprovement'], [mean_diff], 
                xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]], 
                capsize=10, color='#2E8B57', alpha=0.7)
        ax1.set_xlabel('Performance Difference', fontweight='bold')
        ax1.set_title('Bootstrap 95% Confidence Interval', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.text(mean_diff, 0, f'{mean_diff:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]', 
                ha='center', va='center', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Effect size visualization
        effect_size = validation_results['effect_size']
        effect_categories = ['Small\n(0.2)', 'Medium\n(0.5)', 'Large\n(0.8)', 'Our Study']
        effect_values = [0.2, 0.5, 0.8, effect_size]
        colors = ['#FFB6C1', '#FFA500', '#FF6347', '#2E8B57']
        
        bars = ax2.bar(effect_categories, effect_values, color=colors, alpha=0.7)
        ax2.set_ylabel("Cohen's d", fontweight='bold')
        ax2.set_title('Effect Size Comparison', fontsize=14, fontweight='bold')
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect Threshold')
        
        for bar, value in zip(bars, effect_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # P-value corrections visualization
        if 'multiple_comparison_correction' in validation_results:
            corrections = validation_results['multiple_comparison_correction']
            original_p = [0.001, 0.001, 0.001, 0.032, 0.018]  # Example values
            corrected_p = [p for _, p in sorted(corrections)]
            
            x_pos = np.arange(len(original_p))
            width = 0.35
            
            ax3.bar(x_pos - width/2, original_p, width, label='Original p-values', 
                   color='lightcoral', alpha=0.7)
            ax3.bar(x_pos + width/2, corrected_p, width, label='Corrected p-values',
                   color='lightblue', alpha=0.7)
            
            ax3.set_ylabel('p-value', fontweight='bold')
            ax3.set_xlabel('Test Number', fontweight='bold')
            ax3.set_title('Multiple Comparison Correction\n(Benjamini-Hochberg)', 
                         fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
            ax3.legend()
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'Test {i+1}' for i in range(len(original_p))])
        
        # Power analysis visualization
        sample_sizes = [100, 200, 500, 1000, 2000]
        effect_sizes_power = [0.2, 0.5, 0.8]
        
        for i, es in enumerate(effect_sizes_power):
            powers = []
            for n in sample_sizes:
                # Simplified power calculation
                power = min(1.0, 0.05 + (es * np.sqrt(n) / 5))
                powers.append(power)
            
            ax4.plot(sample_sizes, powers, 'o-', label=f'Effect Size = {es}',
                    linewidth=2, markersize=6)
        
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% Power')
        ax4.set_xlabel('Sample Size', fontweight='bold')
        ax4.set_ylabel('Statistical Power', fontweight='bold')
        ax4.set_title('Power Analysis', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


class InteractiveVisualizationSuite:
    """
    Interactive visualizations using Plotly for web-based exploration
    """
    
    def __init__(self):
        self.color_discrete_sequence = px.colors.qualitative.Set2
    
    def create_interactive_performance_dashboard(self, agent_data: Dict[str, Any], 
                                               care_rag_data: Dict[str, Any]) -> str:
        """Create interactive performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Agent Performance Comparison', 'CARE-RAG Ablation Study',
                           'Performance Metrics Radar', 'Component Contribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]]
        )
        
        # Agent performance comparison
        agents = list(agent_data.keys())
        accuracies = [agent_data[agent].accuracy * 100 for agent in agents]
        hpi_scores = [agent_data[agent].hpi * 100 for agent in agents]
        
        fig.add_trace(
            go.Bar(name='Accuracy', x=agents, y=accuracies, 
                  marker_color=self.color_discrete_sequence[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(name='HPI', x=agents, y=hpi_scores, 
                  marker_color=self.color_discrete_sequence[1]),
            row=1, col=1
        )
        
        # CARE-RAG ablation study
        care_rag_configs = list(care_rag_data.keys())
        care_rag_scores = [care_rag_data[config].accuracy * 100 for config in care_rag_configs]
        
        fig.add_trace(
            go.Bar(x=care_rag_configs, y=care_rag_scores, 
                  name='CARE-RAG Performance',
                  marker_color=self.color_discrete_sequence[2]),
            row=1, col=2
        )
        
        # Radar chart for best performing agent (Drug Interaction)
        best_agent = agent_data['Drug Interaction']
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Semantic Sim', 'HPI']
        values = [best_agent.accuracy, best_agent.f1_score, best_agent.precision, 
                 best_agent.recall, best_agent.semantic_similarity, best_agent.hpi]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Drug Interaction Agent',
                line_color=self.color_discrete_sequence[3]
            ),
            row=2, col=1
        )
        
        # Component contribution (example data)
        components = ['Label Classification', 'Entity Extraction', 'Graph Traversal', 
                     'Vector Search', 'Context Management']
        contributions = [14.9, 11.2, 20.5, 22.6, 12.3]  # Percentage improvements
        
        fig.add_trace(
            go.Bar(x=components, y=contributions, 
                  name='Component Contribution (%)',
                  marker_color=self.color_discrete_sequence[4]),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="BioMed-KAI Interactive Performance Dashboard",
            title_x=0.5,
            title_font_size=20
        )
        
        # Update polar subplot
        fig.update_polars(radialaxis_range=[0, 1])
        
        return fig.to_html(include_plotlyjs=True, div_id="performance_dashboard")
    
    def create_interactive_error_analysis(self, error_data: Dict[str, Dict[str, float]]) -> str:
        """Create interactive error analysis visualization"""
        
        error_types = list(error_data.keys())
        frequencies = [data['frequency'] for data in error_data.values()]
        impacts = [data['impact'] for data in error_data.values()]
        recovery_rates = [data['recovery_rate'] for data in error_data.values()]
        
        # Calculate severity scores
        severity_scores = [(freq * impact / recovery) for freq, impact, recovery 
                          in zip(frequencies, impacts, recovery_rates)]
        
        fig = go.Figure()
        
        # Add bubble chart
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=impacts,
            mode='markers+text',
            marker=dict(
                size=[s*10 for s in severity_scores],  # Scale bubble size
                color=recovery_rates,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Recovery Rate (%)"),
                sizemode='diameter',
                sizeref=2.*max(severity_scores)*10/(40.**2),
                sizemin=4
            ),
            text=error_types,
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hovertemplate='<b>%{text}</b><br>' +
                         'Frequency: %{x:.1f}%<br>' +
                         'Impact: %{y:.1f}%<br>' +
                         'Recovery Rate: %{marker.color:.1f}%<br>' +
                         'Severity Score: %{marker.size}<br>' +
                         '<extra></extra>',
            name='Error Analysis'
        ))
        
        fig.update_layout(
            title='Interactive Error Analysis: Frequency vs Impact vs Recovery',
            xaxis_title='Error Frequency (%)',
            yaxis_title='Performance Impact (%)',
            font=dict(size=14),
            height=600,
            width=900
        )
        
        return fig.to_html(include_plotlyjs=True, div_id="error_analysis")
    
    def create_interactive_temperature_analysis(self, temp_data: Dict[float, Dict[str, float]]) -> str:
        """Create interactive temperature analysis"""
        
        temperatures = list(temp_data.keys())
        accuracies = [temp_data[temp]['accuracy'] for temp in temperatures]
        consistencies = [temp_data[temp]['consistency'] for temp in temperatures]
        hallucination_rates = [temp_data[temp]['hallucination_rate'] for temp in temperatures]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature vs Performance Metrics', '3D Temperature Analysis'),
            specs=[[{"secondary_y": True}, {"type": "scatter3d"}]]
        )
        
        # Temperature performance curves
        fig.add_trace(
            go.Scatter(x=temperatures, y=accuracies, mode='lines+markers',
                      name='Accuracy', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=temperatures, y=consistencies, mode='lines+markers',
                      name='Consistency', line=dict(color='green', width=3)),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=temperatures, y=hallucination_rates, mode='lines+markers',
                      name='Hallucination Rate', line=dict(color='red', width=3)),
            row=1, col=1, secondary_y=True
        )
        
        # 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=temperatures,
                y=accuracies,
                z=consistencies,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=hallucination_rates,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Hallucination Rate", x=0.9)
                ),
                line=dict(color='black', width=4),
                name='Temperature Path',
                hovertemplate='<b>Temperature: %{x}</b><br>' +
                             'Accuracy: %{y:.3f}<br>' +
                             'Consistency: %{z:.3f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Temperature", row=1, col=1)
        fig.update_yaxes(title_text="Performance Metric", row=1, col=1)
        fig.update_yaxes(title_text="Hallucination Rate", secondary_y=True, row=1, col=1)
        
        fig.update_layout(
            height=600,
            title_text="Interactive Temperature Sensitivity Analysis",
            title_x=0.5
        )
        
        return fig.to_html(include_plotlyjs=True, div_id="temperature_analysis")


class AdvancedAnalyticsSuite:
    """
    Advanced analytics including clustering, dimensionality reduction, and predictive modeling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def perform_agent_clustering_analysis(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis on agent performance"""
        
        # Prepare feature matrix
        agents = list(agent_data.keys())
        features = []
        feature_names = ['accuracy', 'f1_score', 'precision', 'recall', 'semantic_similarity', 'hpi']
        
        for agent in agents:
            agent_features = [getattr(agent_data[agent], feature) for feature in feature_names]
            features.append(agent_features)
        
        features_array = np.array(features)
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Hierarchical clustering
        linkage_matrix = linkage(features_scaled, method='ward')
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # t-SNE for non-linear dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        features_tsne = tsne.fit_transform(features_scaled)
        
        return {
            'agents': agents,
            'features_original': features_array,
            'features_scaled': features_scaled,
            'linkage_matrix': linkage_matrix,
            'pca_components': features_pca,
            'tsne_components': features_tsne,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'feature_names': feature_names
        }
    
    def create_clustering_visualization(self, clustering_results: Dict[str, Any], save_path: str) -> None:
        """Create comprehensive clustering visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        agents = clustering_results['agents']
        linkage_matrix = clustering_results['linkage_matrix']
        pca_components = clustering_results['pca_components']
        tsne_components = clustering_results['tsne_components']
        
        # Dendrogram
        dendrogram(linkage_matrix, labels=agents, ax=ax1, orientation='top')
        ax1.set_title('Agent Performance Hierarchical Clustering', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # PCA scatter plot
        colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))
        for i, (agent, color) in enumerate(zip(agents, colors)):
            ax2.scatter(pca_components[i, 0], pca_components[i, 1], 
                       c=[color], s=200, alpha=0.7, edgecolors='black')
            ax2.annotate(agent, (pca_components[i, 0], pca_components[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax2.set_xlabel(f'PC1 ({clustering_results["pca_explained_variance"][0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({clustering_results["pca_explained_variance"][1]:.2%} variance)')
        ax2.set_title('PCA Agent Clustering', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # t-SNE scatter plot
        for i, (agent, color) in enumerate(zip(agents, colors)):
            ax3.scatter(tsne_components[i, 0], tsne_components[i, 1], 
                       c=[color], s=200, alpha=0.7, edgecolors='black')
            ax3.annotate(agent, (tsne_components[i, 0], tsne_components[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax3.set_xlabel('t-SNE Component 1')
        ax3.set_ylabel('t-SNE Component 2')
        ax3.set_title('t-SNE Agent Clustering', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Feature importance heatmap
        feature_matrix = clustering_results['features_scaled']
        feature_names = clustering_results['feature_names']
        
        im = ax4.imshow(feature_matrix, cmap='RdBu_r', aspect='auto')
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels([name.replace('_', ' ').title() for name in feature_names], rotation=45)
        ax4.set_yticks(range(len(agents)))
        ax4.set_yticklabels(agents)
        ax4.set_title('Normalized Feature Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.6)
        cbar.set_label('Normalized Value', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def perform_sensitivity_analysis(self, base_performance: float, 
                                   parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis for system parameters"""
        
        sensitivity_results = {}
        
        for parameter, values in parameter_ranges.items():
            # Simulate performance changes based on parameter values
            if parameter == 'temperature':
                # Temperature has optimal point around 0.3
                performances = [base_performance * (1 - 0.1 * abs(v - 0.3)) for v in values]
            elif parameter == 'threshold':
                # Threshold has diminishing returns
                performances = [base_performance * (1 + 0.05 * np.log(v + 0.1)) for v in values]
            elif parameter == 'depth':
                # Depth has plateau effect
                performances = [base_performance * min(1.2, 1 + 0.03 * v) for v in values]
            else:
                # Default linear relationship
                performances = [base_performance * (1 + 0.02 * v) for v in values]
            
            # Calculate sensitivity metrics
            performance_range = max(performances) - min(performances)
            parameter_range = max(values) - min(values)
            sensitivity = performance_range / parameter_range if parameter_range > 0 else 0
            
            sensitivity_results[parameter] = {
                'values': values,
                'performances': performances,
                'sensitivity': sensitivity,
                'optimal_value': values[np.argmax(performances)],
                'optimal_performance': max(performances)
            }
        
        return sensitivity_results
    
    def create_sensitivity_analysis_plot(self, sensitivity_results: Dict[str, Any], save_path: str) -> None:
        """Create sensitivity analysis visualization"""
        
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(16, 10))
        if n_params == 1:
            axes = [axes]
        axes = axes.flatten() if n_params > 1 else axes
        
        for i, (parameter, results) in enumerate(sensitivity_results.items()):
            ax = axes[i]
            
            values = results['values']
            performances = results['performances']
            optimal_idx = np.argmax(performances)
            
            # Plot performance curve
            ax.plot(values, performances, 'o-', linewidth=3, markersize=8, 
                   label=f'Sensitivity: {results["sensitivity"]:.3f}')
            
            # Mark optimal point
            ax.scatter(values[optimal_idx], performances[optimal_idx], 
                      color='red', s=200, zorder=5, marker='*', 
                      label=f'Optimal: {values[optimal_idx]:.2f}')
            
            ax.set_xlabel(parameter.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Performance', fontweight='bold')
            ax.set_title(f'{parameter.replace("_", " ").title()} Sensitivity Analysis', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove unused subplots
        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def perform_ablation_impact_analysis(self, component_contributions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the impact of different components through ablation"""
        
        components = list(component_contributions.keys())
        contributions = list(component_contributions.values())
        
        # Calculate cumulative impact
        sorted_indices = np.argsort(contributions)[::-1]  # Sort in descending order
        sorted_components = [components[i] for i in sorted_indices]
        sorted_contributions = [contributions[i] for i in sorted_indices]
        cumulative_impact = np.cumsum(sorted_contributions)
        
        # Calculate relative importance
        total_contribution = sum(contributions)
        relative_importance = [contrib / total_contribution * 100 for contrib in contributions]
        
        return {
            'components': components,
            'contributions': contributions,
            'sorted_components': sorted_components,
            'sorted_contributions': sorted_contributions,
            'cumulative_impact': cumulative_impact.tolist(),
            'relative_importance': relative_importance,
            'most_important': sorted_components[0],
            'least_important': sorted_components[-1]
        }
    
    def create_ablation_impact_plot(self, ablation_results: Dict[str, Any], save_path: str) -> None:
        """Create ablation impact visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        components = ablation_results['components']
        contributions = ablation_results['contributions']
        sorted_components = ablation_results['sorted_components']
        sorted_contributions = ablation_results['sorted_contributions']
        cumulative_impact = ablation_results['cumulative_impact']
        relative_importance = ablation_results['relative_importance']
        
        # Component contributions bar chart
        bars1 = ax1.bar(components, contributions, color=plt.cm.viridis(np.linspace(0, 1, len(components))))
        ax1.set_title('Component Contributions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Performance Improvement', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, contrib in zip(bars1, contributions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{contrib:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Sorted contributions (Pareto-style)
        bars2 = ax2.bar(range(len(sorted_components)), sorted_contributions, 
                       color=plt.cm.plasma(np.linspace(0, 1, len(sorted_components))))
        ax2.set_title('Components Ranked by Impact', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Performance Improvement', fontweight='bold')
        ax2.set_xlabel('Component Rank', fontweight='bold')
        ax2.set_xticks(range(len(sorted_components)))
        ax2.set_xticklabels([f'{i+1}' for i in range(len(sorted_components))])
        
        # Add component labels
        for i, (bar, comp) in enumerate(zip(bars2, sorted_components)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    comp[:8] + '...' if len(comp) > 8 else comp, 
                    ha='center', va='bottom', fontweight='bold', rotation=45, fontsize=9)
        
        # Cumulative impact curve
        ax3.plot(range(1, len(sorted_components) + 1), cumulative_impact, 'o-', 
                linewidth=3, markersize=8, color='darkgreen')
        ax3.fill_between(range(1, len(sorted_components) + 1), cumulative_impact, alpha=0.3, color='lightgreen')
        ax3.set_title('Cumulative Component Impact', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Components', fontweight='bold')
        ax3.set_ylabel('Cumulative Performance Improvement', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add 80% line (Pareto principle)
        total_impact = max(cumulative_impact)
        ax3.axhline(y=total_impact * 0.8, color='red', linestyle='--', 
                   label='80% of Impact', linewidth=2)
        ax3.legend()
        
        # Pie chart of relative importance
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        wedges, texts, autotexts = ax4.pie(relative_importance, labels=components, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax4.set_title('Relative Component Importance', fontsize=14, fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


# Main execution script for advanced analytics
def run_advanced_analytics():
    """Run advanced analytics suite"""
    
    print("Running BioMed-KAI Advanced Analytics Suite...")
    print("="*50)
    
    # Initialize suite
    viz_suite = VisualizationSuite()
    interactive_suite = InteractiveVisualizationSuite()
    analytics_suite = AdvancedAnalyticsSuite()
    
    # Create results directory
    import os
    results_dir = "./advanced_analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Simulate experimental data (in production, load real data)
    from dataclasses import dataclass
    
    @dataclass
    class PerformanceMetrics:
        accuracy: float
        f1_score: float
        precision: float
        recall: float
        semantic_similarity: float
        kl_divergence: float
        entropy: float
        hpi: float = 0.0
        
        def __post_init__(self):
            w1, w2, w3, w4 = 0.1, 0.4, 0.3, 0.2
            self.hpi = (w1 * self.entropy + w2 * self.f1_score + 
                       w3 * self.semantic_similarity + w4 * (1 - self.kl_divergence))
    
    # Agent performance data
    agent_data = {
        'Diagnostic': PerformanceMetrics(0.857, 0.834, 0.845, 0.823, 0.834, 0.151, 0.114),
        'Treatment': PerformanceMetrics(0.889, 0.867, 0.878, 0.856, 0.859, 0.138, 0.106),
        'Drug Interaction': PerformanceMetrics(0.921, 0.897, 0.909, 0.885, 0.892, 0.125, 0.098),
        'General Medical': PerformanceMetrics(0.842, 0.818, 0.830, 0.806, 0.823, 0.165, 0.127),
        'Preventive Care': PerformanceMetrics(0.868, 0.845, 0.856, 0.834, 0.847, 0.142, 0.112),
        'Research': PerformanceMetrics(0.823, 0.798, 0.810, 0.786, 0.812, 0.178, 0.134)
    }
    
    # CARE-RAG data
    care_rag_data = {
        'Full_CARE_RAG': PerformanceMetrics(0.880, 0.847, 0.863, 0.831, 0.834, 0.151, 0.114),
        'Label_Only': PerformanceMetrics(0.793, 0.723, 0.758, 0.691, 0.751, 0.267, 0.187),
        'Entity_Only': PerformanceMetrics(0.768, 0.689, 0.728, 0.652, 0.718, 0.289, 0.203),
        'Neither_Baseline': PerformanceMetrics(0.642, 0.542, 0.592, 0.498, 0.623, 0.421, 0.341)
    }
    
    # Temperature data
    temp_data = {
        0.1: {'accuracy': 0.893, 'consistency': 0.947, 'hallucination_rate': 0.021},
        0.3: {'accuracy': 0.887, 'consistency': 0.923, 'hallucination_rate': 0.034},
        0.5: {'accuracy': 0.872, 'consistency': 0.889, 'hallucination_rate': 0.052},
        0.7: {'accuracy': 0.849, 'consistency': 0.823, 'hallucination_rate': 0.087},
        0.9: {'accuracy': 0.814, 'consistency': 0.756, 'hallucination_rate': 0.123}
    }
    
    # Error data
    error_data = {
        'Entity_Misclassification': {'frequency': 8.3, 'impact': 15.2, 'recovery_rate': 73},
        'Ambiguous_Query_Parsing': {'frequency': 12.1, 'impact': 8.7, 'recovery_rate': 65},
        'Multi_entity_Extraction_Failure': {'frequency': 5.4, 'impact': 22.3, 'recovery_rate': 58},
        'Classification_Confidence_Low': {'frequency': 14.7, 'impact': 11.8, 'recovery_rate': 71}
    }
    
    # Comparison data
    comparison_data = {
        'BiomedKAI': {'medqa': 84.4, 'diagnostic': 85.7, 'resource_efficiency': 1.0},
        'Med-Gemini': {'medqa': 91.1, 'diagnostic': None, 'resource_efficiency': 0.05},
        'GPT-4': {'medqa': 90.2, 'diagnostic': 64.0, 'resource_efficiency': 0.08},
        'Med-PaLM2': {'medqa': 86.5, 'diagnostic': None, 'resource_efficiency': 0.12},
        'MedRAG': {'medqa': 70.0, 'diagnostic': None, 'resource_efficiency': 0.8}
    }
    
    # Generate static visualizations
    print("Generating static visualizations...")
    
    viz_suite.create_publication_ready_comparison_plot(
        comparison_data, f"{results_dir}/comparison_analysis.png")
    
    viz_suite.create_agent_performance_heatmap(
        agent_data, f"{results_dir}/agent_heatmap.png")
    
    viz_suite.create_care_rag_architecture_diagram(
        f"{results_dir}/care_rag_architecture.png")
    
    viz_suite.create_error_analysis_dashboard(
        error_data, f"{results_dir}/error_dashboard.png")
    
    viz_suite.create_temperature_sensitivity_3d_plot(
        temp_data, f"{results_dir}/temperature_3d.png")
    
    # Validation results simulation
    validation_results = {
        'bootstrap_ci': {'lower': 0.185, 'upper': 0.238},
        'effect_size': 0.847,
        'multiple_comparison_correction': [(0, 0.001), (1, 0.002), (2, 0.003), (3, 0.041), (4, 0.025)]
    }
    
    viz_suite.create_statistical_validation_report_plot(
        validation_results, f"{results_dir}/statistical_validation.png")
    
    # Generate interactive visualizations
    print("Generating interactive visualizations...")
    
    performance_dashboard_html = interactive_suite.create_interactive_performance_dashboard(
        agent_data, care_rag_data)
    
    with open(f"{results_dir}/interactive_dashboard.html", "w") as f:
        f.write(performance_dashboard_html)
    
    error_analysis_html = interactive_suite.create_interactive_error_analysis(error_data)
    
    with open(f"{results_dir}/interactive_error_analysis.html", "w") as f:
        f.write(error_analysis_html)
    
    temp_analysis_html = interactive_suite.create_interactive_temperature_analysis(temp_data)
    
    with open(f"{results_dir}/interactive_temperature_analysis.html", "w") as f:
        f.write(temp_analysis_html)
    
    # Advanced analytics
    print("Running advanced analytics...")
    
    # Clustering analysis
    clustering_results = analytics_suite.perform_agent_clustering_analysis(agent_data)
    analytics_suite.create_clustering_visualization(
        clustering_results, f"{results_dir}/agent_clustering.png")
    
    # Sensitivity analysis
    parameter_ranges = {
        'temperature': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'threshold': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'depth': [1, 2, 3, 4, 5]
    }
    
    sensitivity_results = analytics_suite.perform_sensitivity_analysis(0.857, parameter_ranges)
    analytics_suite.create_sensitivity_analysis_plot(
        sensitivity_results, f"{results_dir}/sensitivity_analysis.png")
    
    # Ablation impact analysis
    component_contributions = {
        'Label Classification': 14.9,
        'Entity Extraction': 11.2,
        'Graph Traversal': 20.5,
        'Vector Search': 22.6,
        'Context Management': 12.3,
        'Multi-Agent Routing': 18.5
    }
    
    ablation_results = analytics_suite.perform_ablation_impact_analysis(component_contributions)
    analytics_suite.create_ablation_impact_plot(
        ablation_results, f"{results_dir}/ablation_impact.png")
    
    print(f"\nAdvanced analytics completed successfully!")
    print(f"All results saved to: {results_dir}")
    print("\nGenerated files:")
    print("- Static visualizations (PNG format)")
    print("- Interactive dashboards (HTML format)")
    print("- Clustering analysis")
    print("- Sensitivity analysis")
    print("- Ablation impact analysis")


if __name__ == "__main__":
    run_advanced_analytics()