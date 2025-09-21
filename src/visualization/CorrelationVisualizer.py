import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CorrelationVisualizer:
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        self.figures = {}
        self.statistical_results = {}

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def analyze_correlation_matrix(self, data, asset_names=None, title="Asset Correlation Analysis"):
        """
        Comprehensive correlation matrix analysis with multiple correlation measures
        """
        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(data.shape[1])]

        df = pd.DataFrame(data, columns=asset_names)

        # Calculate different correlation measures
        pearson_corr = df.corr(method='pearson')
        spearman_corr = df.corr(method='spearman')
        kendall_corr = df.corr(method='kendall')

        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Pearson correlation heatmap
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[0,0], cbar_kws={"shrink": .8})
        axes[0,0].set_title('Pearson Correlation Matrix')

        # Spearman correlation heatmap
        sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[0,1], cbar_kws={"shrink": .8})
        axes[0,1].set_title('Spearman Correlation Matrix')

        # Correlation difference heatmap
        corr_diff = np.abs(pearson_corr - spearman_corr)
        sns.heatmap(corr_diff, mask=mask, annot=True, cmap='Reds',
                   square=True, ax=axes[1,0], cbar_kws={"shrink": .8})
        axes[1,0].set_title('|Pearson - Spearman| Difference\n(Non-linearity Indicator)')

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(pearson_corr)
        eigenvals = eigenvals[::-1]  # Sort descending

        axes[1,1].bar(range(len(eigenvals)), eigenvals, alpha=0.7)
        axes[1,1].set_title('Correlation Matrix Eigenvalues')
        axes[1,1].set_xlabel('Principal Component')
        axes[1,1].set_ylabel('Eigenvalue')
        axes[1,1].axhline(y=1, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Store results
        self.figures['correlation_analysis'] = fig
        self.statistical_results['correlations'] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs
        }

        # Detect significant correlations
        significant_correlations = self._detect_significant_correlations(pearson_corr, spearman_corr)

        return significant_correlations

    def _detect_significant_correlations(self, pearson_corr, spearman_corr, threshold=0.3):
        """
        Detect significant correlations and unusual patterns
        """
        n_assets = len(pearson_corr)
        significant_pairs = []

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]

                # Check for significant correlation
                if abs(pearson_val) > threshold:
                    # Check for non-linearity
                    non_linear = abs(pearson_val - spearman_val) > 0.2

                    significant_pairs.append({
                        'asset_1': pearson_corr.index[i],
                        'asset_2': pearson_corr.columns[j],
                        'pearson_correlation': pearson_val,
                        'spearman_correlation': spearman_val,
                        'difference': abs(pearson_val - spearman_val),
                        'is_significant': True,
                        'is_non_linear': non_linear,
                        'strength': self._classify_correlation_strength(abs(pearson_val))
                    })

        return significant_pairs

    def _classify_correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr < 0.3:
            return 'weak'
        elif abs_corr < 0.5:
            return 'moderate'
        elif abs_corr < 0.7:
            return 'strong'
        else:
            return 'very_strong'

    def analyze_rolling_correlations(self, data1, data2, window=50, asset1_name="Asset1", asset2_name="Asset2"):
        """
        Analyze time-varying correlations between two assets
        """
        # Calculate rolling correlations
        df = pd.DataFrame({'asset1': data1, 'asset2': data2})
        rolling_corr = df['asset1'].rolling(window=window).corr(df['asset2'])

        # Detect regime changes
        corr_changes = np.abs(np.diff(rolling_corr.dropna()))
        change_threshold = np.percentile(corr_changes, 95)
        change_points = np.where(corr_changes > change_threshold)[0]

        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Time series data
        axes[0].plot(data1, label=asset1_name, alpha=0.7)
        axes[0].plot(data2, label=asset2_name, alpha=0.7)
        axes[0].set_title(f'{asset1_name} vs {asset2_name} - Time Series')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Rolling correlation
        axes[1].plot(rolling_corr.index, rolling_corr.values, linewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Significance threshold')
        axes[1].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)

        # Mark regime changes
        for cp in change_points:
            if cp < len(rolling_corr):
                axes[1].axvline(x=rolling_corr.index[cp], color='orange', alpha=0.7)

        axes[1].set_title(f'Rolling Correlation ({window}-period window)')
        axes[1].set_ylabel('Correlation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Correlation change magnitude
        axes[2].plot(corr_changes, alpha=0.7)
        axes[2].axhline(y=change_threshold, color='red', linestyle='--',
                       label=f'95th percentile: {change_threshold:.3f}')
        axes[2].set_title('Correlation Change Magnitude')
        axes[2].set_ylabel('|Δ Correlation|')
        axes[2].set_xlabel('Time Period')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Store results
        self.figures['rolling_correlation'] = fig

        # Statistical analysis
        regime_stats = self._analyze_correlation_regimes(rolling_corr, change_points)

        return {
            'rolling_correlations': rolling_corr,
            'change_points': change_points,
            'change_threshold': change_threshold,
            'regime_statistics': regime_stats
        }

    def _analyze_correlation_regimes(self, rolling_corr, change_points):
        """Analyze different correlation regimes"""
        regimes = []
        start_idx = 0

        for cp in change_points:
            if cp > start_idx:
                regime_data = rolling_corr.iloc[start_idx:cp]
                if len(regime_data) > 5:  # Minimum regime length
                    regimes.append({
                        'start': start_idx,
                        'end': cp,
                        'length': len(regime_data),
                        'mean_correlation': regime_data.mean(),
                        'std_correlation': regime_data.std(),
                        'min_correlation': regime_data.min(),
                        'max_correlation': regime_data.max()
                    })
                start_idx = cp

        # Last regime
        if start_idx < len(rolling_corr):
            regime_data = rolling_corr.iloc[start_idx:]
            regimes.append({
                'start': start_idx,
                'end': len(rolling_corr),
                'length': len(regime_data),
                'mean_correlation': regime_data.mean(),
                'std_correlation': regime_data.std(),
                'min_correlation': regime_data.min(),
                'max_correlation': regime_data.max()
            })

        return regimes

    def analyze_tail_dependencies(self, data, asset_names=None, quantiles=[0.05, 0.95]):
        """
        Analyze tail dependencies between assets
        """
        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(data.shape[1])]

        n_assets = data.shape[1]
        n_quantiles = len(quantiles)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Tail Dependency Analysis', fontsize=16, fontweight='bold')

        # Calculate tail dependencies
        tail_deps = {}

        for i, q in enumerate(quantiles):
            tail_deps[f'quantile_{q}'] = np.zeros((n_assets, n_assets))

            for asset1 in range(n_assets):
                for asset2 in range(n_assets):
                    if asset1 != asset2:
                        tail_dep = self._calculate_tail_dependence(
                            data[:, asset1], data[:, asset2], quantile=q)
                        tail_deps[f'quantile_{q}'][asset1, asset2] = tail_dep

        # Plot lower tail dependencies (5th percentile)
        lower_tail = pd.DataFrame(tail_deps['quantile_0.05'],
                                 index=asset_names, columns=asset_names)
        sns.heatmap(lower_tail, annot=True, cmap='Blues', center=0,
                   square=True, ax=axes[0,0], cbar_kws={"shrink": .8})
        axes[0,0].set_title('Lower Tail Dependencies (5th percentile)')

        # Plot upper tail dependencies (95th percentile)
        upper_tail = pd.DataFrame(tail_deps['quantile_0.95'],
                                 index=asset_names, columns=asset_names)
        sns.heatmap(upper_tail, annot=True, cmap='Reds', center=0,
                   square=True, ax=axes[0,1], cbar_kws={"shrink": .8})
        axes[0,1].set_title('Upper Tail Dependencies (95th percentile)')

        # Asymmetric tail dependencies
        asymmetric_tail = upper_tail - lower_tail
        sns.heatmap(asymmetric_tail, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[1,0], cbar_kws={"shrink": .8})
        axes[1,0].set_title('Tail Asymmetry (Upper - Lower)')

        # Distribution of tail dependencies
        all_upper = upper_tail.values[upper_tail.values != 0]
        all_lower = lower_tail.values[lower_tail.values != 0]

        axes[1,1].hist(all_lower, alpha=0.7, label='Lower tail', bins=20)
        axes[1,1].hist(all_upper, alpha=0.7, label='Upper tail', bins=20)
        axes[1,1].set_title('Distribution of Tail Dependencies')
        axes[1,1].set_xlabel('Tail Dependence Coefficient')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()

        plt.tight_layout()

        self.figures['tail_dependencies'] = fig

        # Identify significant tail dependencies
        significant_tail_deps = self._identify_significant_tail_deps(
            lower_tail, upper_tail, threshold=0.2)

        return significant_tail_deps

    def _calculate_tail_dependence(self, x, y, quantile=0.05):
        """Calculate tail dependence coefficient"""
        n = len(x)

        if quantile < 0.5:  # Lower tail
            threshold_x = np.percentile(x, quantile * 100)
            threshold_y = np.percentile(y, quantile * 100)

            joint_exceedances = np.sum((x <= threshold_x) & (y <= threshold_y))
            marginal_exceedances = min(np.sum(x <= threshold_x), np.sum(y <= threshold_y))
        else:  # Upper tail
            threshold_x = np.percentile(x, quantile * 100)
            threshold_y = np.percentile(y, quantile * 100)

            joint_exceedances = np.sum((x >= threshold_x) & (y >= threshold_y))
            marginal_exceedances = min(np.sum(x >= threshold_x), np.sum(y >= threshold_y))

        if marginal_exceedances == 0:
            return 0

        return joint_exceedances / marginal_exceedances

    def _identify_significant_tail_deps(self, lower_tail, upper_tail, threshold=0.2):
        """Identify significant tail dependencies"""
        significant_deps = []

        n_assets = len(lower_tail)

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                lower_val = lower_tail.iloc[i, j]
                upper_val = upper_tail.iloc[i, j]

                if max(abs(lower_val), abs(upper_val)) > threshold:
                    significant_deps.append({
                        'asset_1': lower_tail.index[i],
                        'asset_2': lower_tail.columns[j],
                        'lower_tail_dependence': lower_val,
                        'upper_tail_dependence': upper_val,
                        'asymmetry': upper_val - lower_val,
                        'max_dependence': max(abs(lower_val), abs(upper_val)),
                        'is_asymmetric': abs(upper_val - lower_val) > 0.1
                    })

        return significant_deps

    def detect_volatility_clustering(self, returns_data, asset_names=None):
        """
        Detect and visualize volatility clustering patterns
        """
        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(returns_data.shape[1])]

        n_assets = returns_data.shape[1]

        fig, axes = plt.subplots(n_assets, 2, figsize=(15, 4*n_assets))
        if n_assets == 1:
            axes = axes.reshape(1, -1)

        clustering_results = {}

        for i, asset_name in enumerate(asset_names):
            returns = returns_data[:, i]
            abs_returns = np.abs(returns)
            squared_returns = returns ** 2

            # Calculate autocorrelations of squared returns
            max_lag = min(20, len(returns) // 4)
            autocorrs = []
            lags = range(1, max_lag + 1)

            for lag in lags:
                corr = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
                autocorrs.append(corr)

            # Plot returns and squared returns
            axes[i, 0].plot(returns, alpha=0.7, linewidth=1)
            axes[i, 0].set_title(f'{asset_name} - Returns')
            axes[i, 0].set_ylabel('Returns')
            axes[i, 0].grid(True, alpha=0.3)

            # Plot autocorrelations of squared returns
            axes[i, 1].bar(lags, autocorrs, alpha=0.7)
            axes[i, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[i, 1].axhline(y=1.96/np.sqrt(len(returns)), color='red',
                              linestyle='--', alpha=0.5, label='95% confidence')
            axes[i, 1].axhline(y=-1.96/np.sqrt(len(returns)), color='red',
                              linestyle='--', alpha=0.5)
            axes[i, 1].set_title(f'{asset_name} - Volatility Clustering (Squared Returns ACF)')
            axes[i, 1].set_xlabel('Lag')
            axes[i, 1].set_ylabel('Autocorrelation')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

            # Store results
            significant_lags = [lag for lag, corr in zip(lags, autocorrs)
                              if abs(corr) > 1.96/np.sqrt(len(returns))]

            clustering_results[asset_name] = {
                'autocorrelations': dict(zip(lags, autocorrs)),
                'significant_lags': significant_lags,
                'has_clustering': len(significant_lags) > 0,
                'clustering_strength': np.mean([abs(autocorrs[lag-1]) for lag in significant_lags]) if significant_lags else 0
            }

        plt.tight_layout()
        self.figures['volatility_clustering'] = fig

        return clustering_results

    def create_interactive_correlation_network(self, correlation_matrix, asset_names=None, threshold=0.3):
        """
        Create an interactive network visualization of significant correlations
        """
        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(len(correlation_matrix))]

        # Create network data
        edges = []
        nodes = []

        # Add nodes
        for i, name in enumerate(asset_names):
            nodes.append({
                'id': i,
                'label': name,
                'size': 20
            })

        # Add edges for significant correlations
        n_assets = len(correlation_matrix)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                corr_val = correlation_matrix.iloc[i, j] if hasattr(correlation_matrix, 'iloc') else correlation_matrix[i, j]

                if abs(corr_val) > threshold:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': abs(corr_val),
                        'correlation': corr_val,
                        'color': 'red' if corr_val > 0 else 'blue',
                        'width': abs(corr_val) * 10
                    })

        # Create Plotly network visualization
        edge_x = []
        edge_y = []
        edge_info = []

        # Position nodes in a circle
        import math
        for edge in edges:
            source_angle = 2 * math.pi * edge['source'] / n_assets
            target_angle = 2 * math.pi * edge['target'] / n_assets

            x0, y0 = math.cos(source_angle), math.sin(source_angle)
            x1, y1 = math.cos(target_angle), math.sin(target_angle)

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{asset_names[edge['source']]} - {asset_names[edge['target']]}: {edge['correlation']:.3f}")

        # Node positions
        node_x = [math.cos(2 * math.pi * i / n_assets) for i in range(n_assets)]
        node_y = [math.sin(2 * math.pi * i / n_assets) for i in range(n_assets)]

        # Create the plot
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='gray'),
                                hoverinfo='none',
                                mode='lines'))

        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                text=asset_names,
                                textposition="middle center",
                                hovertext=[f"{name}<br>Connections: {sum(1 for e in edges if e['source'] == i or e['target'] == i)}"
                                          for i, name in enumerate(asset_names)],
                                marker=dict(size=30,
                                          color='lightblue',
                                          line=dict(width=2, color='darkblue'))))

        fig.update_layout(title='Asset Correlation Network',
                         titlefont_size=16,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[ dict(
                             text="Connections show correlations > " + str(threshold),
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002,
                             xanchor='left', yanchor='bottom',
                             font=dict(color='gray', size=12)
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        self.figures['correlation_network'] = fig

        return fig

    def generate_anomaly_report(self, data, asset_names=None):
        """
        Generate comprehensive anomaly detection report
        """
        if asset_names is None:
            asset_names = [f'Asset_{i+1}' for i in range(data.shape[1])]

        anomaly_results = {}

        for i, asset_name in enumerate(asset_names):
            asset_data = data[:, i]

            # Detect outliers using multiple methods
            outliers_iqr = self._detect_outliers_iqr(asset_data)
            outliers_zscore = self._detect_outliers_zscore(asset_data)
            outliers_isolation = self._detect_outliers_isolation_forest(asset_data)

            # Jump detection
            jumps = self._detect_jumps(asset_data)

            # Regime changes
            regime_changes = self._detect_regime_changes(asset_data)

            anomaly_results[asset_name] = {
                'outliers_iqr': outliers_iqr,
                'outliers_zscore': outliers_zscore,
                'outliers_isolation': outliers_isolation,
                'jumps': jumps,
                'regime_changes': regime_changes,
                'total_anomalies': len(outliers_iqr) + len(jumps) + len(regime_changes)
            }

        # Create visualization
        self._visualize_anomalies(data, asset_names, anomaly_results)

        return anomaly_results

    def _detect_outliers_iqr(self, data, factor=1.5):
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append({
                    'index': i,
                    'value': value,
                    'type': 'lower' if value < lower_bound else 'upper',
                    'deviation': abs(value - (upper_bound if value > upper_bound else lower_bound))
                })

        return outliers

    def _detect_outliers_zscore(self, data, threshold=3):
        """Detect outliers using Z-score method"""
        mean = np.mean(data)
        std = np.std(data)

        outliers = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append({
                    'index': i,
                    'value': value,
                    'z_score': z_score,
                    'deviation': abs(value - mean)
                })

        return outliers

    def _detect_outliers_isolation_forest(self, data, contamination=0.1):
        """Detect outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(contamination=contamination, random_state=42)
        outliers_pred = clf.fit_predict(data.reshape(-1, 1))

        outliers = []
        for i, pred in enumerate(outliers_pred):
            if pred == -1:  # Outlier
                outliers.append({
                    'index': i,
                    'value': data[i],
                    'anomaly_score': clf.score_samples(data[i].reshape(1, -1))[0]
                })

        return outliers

    def _detect_jumps(self, data, threshold_factor=3):
        """Detect sudden jumps in the data"""
        if len(data) < 2:
            return []

        returns = np.diff(data) / data[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        threshold = threshold_factor * std_return

        jumps = []
        for i, ret in enumerate(returns):
            if abs(ret - mean_return) > threshold:
                jumps.append({
                    'index': i + 1,
                    'return': ret,
                    'magnitude': abs(ret - mean_return),
                    'direction': 'up' if ret > mean_return else 'down'
                })

        return jumps

    def _detect_regime_changes(self, data, window=50):
        """Detect regime changes using rolling statistics"""
        if len(data) < 2 * window:
            return []

        rolling_mean = pd.Series(data).rolling(window=window).mean()
        rolling_std = pd.Series(data).rolling(window=window).std()

        # Detect significant changes in mean and volatility
        mean_changes = np.abs(np.diff(rolling_mean.dropna()))
        std_changes = np.abs(np.diff(rolling_std.dropna()))

        mean_threshold = np.percentile(mean_changes, 95)
        std_threshold = np.percentile(std_changes, 95)

        regime_changes = []

        for i in range(len(mean_changes)):
            if mean_changes[i] > mean_threshold or std_changes[i] > std_threshold:
                regime_changes.append({
                    'index': i + window,
                    'mean_change': mean_changes[i],
                    'std_change': std_changes[i],
                    'type': 'mean' if mean_changes[i] > mean_threshold else 'volatility'
                })

        return regime_changes

    def _visualize_anomalies(self, data, asset_names, anomaly_results):
        """Visualize detected anomalies"""
        n_assets = len(asset_names)

        fig, axes = plt.subplots(n_assets, 1, figsize=(15, 4*n_assets))
        if n_assets == 1:
            axes = [axes]

        for i, asset_name in enumerate(asset_names):
            asset_data = data[:, i]
            results = anomaly_results[asset_name]

            # Plot the time series
            axes[i].plot(asset_data, alpha=0.7, linewidth=1, label='Data')

            # Mark outliers
            for outlier in results['outliers_iqr']:
                axes[i].scatter(outlier['index'], outlier['value'],
                               color='red', s=50, alpha=0.7, marker='o')

            # Mark jumps
            for jump in results['jumps']:
                axes[i].axvline(x=jump['index'], color='orange',
                               linestyle='--', alpha=0.7)

            # Mark regime changes
            for regime_change in results['regime_changes']:
                axes[i].axvline(x=regime_change['index'], color='purple',
                               linestyle=':', alpha=0.7)

            axes[i].set_title(f'{asset_name} - Anomaly Detection\n'
                             f'Outliers: {len(results["outliers_iqr"])}, '
                             f'Jumps: {len(results["jumps"])}, '
                             f'Regime Changes: {len(results["regime_changes"])}')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        axes[-1].set_xlabel('Time')
        plt.tight_layout()

        self.figures['anomaly_detection'] = fig

    def save_all_figures(self, format='png', dpi=300):
        """Save all generated figures"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        saved_files = []

        for name, fig in self.figures.items():
            filename = f"{self.output_dir}/{name}.{format}"

            if hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(filename)
            else:  # Matplotlib figure
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')

            saved_files.append(filename)
            print(f"Saved: {filename}")

        return saved_files

    def generate_summary_report(self):
        """Generate a summary report of all analyses"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'analyses_performed': list(self.figures.keys()),
            'statistical_results': self.statistical_results,
            'figures_generated': len(self.figures)
        }

        return report

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_assets = 5

    # Create correlated data with some interesting patterns
    base_data = np.random.randn(n_samples, n_assets)

    # Add correlations
    correlation_matrix = np.array([
        [1.0, 0.7, 0.3, -0.2, 0.1],
        [0.7, 1.0, 0.5, -0.1, 0.2],
        [0.3, 0.5, 1.0, 0.1, -0.3],
        [-0.2, -0.1, 0.1, 1.0, 0.6],
        [0.1, 0.2, -0.3, 0.6, 1.0]
    ])

    # Apply correlation structure
    L = np.linalg.cholesky(correlation_matrix)
    correlated_data = base_data @ L.T

    # Add some anomalies and regime changes
    correlated_data[100:110, 0] += 3  # Add outliers to first asset
    correlated_data[500:600, 1] *= 2  # Regime change in second asset

    # Initialize visualizer
    viz = CorrelationVisualizer()
    asset_names = ['Stock_A', 'Stock_B', 'Bond_C', 'Commodity_D', 'Currency_E']

    # Perform analyses
    print("Performing correlation analysis...")
    significant_corrs = viz.analyze_correlation_matrix(correlated_data, asset_names)

    print("Analyzing rolling correlations...")
    rolling_results = viz.analyze_rolling_correlations(
        correlated_data[:, 0], correlated_data[:, 1],
        asset1_name=asset_names[0], asset2_name=asset_names[1])

    print("Analyzing tail dependencies...")
    tail_deps = viz.analyze_tail_dependencies(correlated_data, asset_names)

    print("Detecting volatility clustering...")
    clustering_results = viz.detect_volatility_clustering(
        np.diff(correlated_data, axis=0), asset_names)

    print("Generating anomaly report...")
    anomaly_results = viz.generate_anomaly_report(correlated_data, asset_names)

    print("Creating correlation network...")
    network_fig = viz.create_interactive_correlation_network(
        pd.DataFrame(correlation_matrix, index=asset_names, columns=asset_names),
        asset_names)

    # Save all figures
    print("Saving figures...")
    saved_files = viz.save_all_figures()

    # Generate summary report
    summary = viz.generate_summary_report()
    print("\nAnalysis Summary:")
    print(f"- Figures generated: {summary['figures_generated']}")
    print(f"- Analyses performed: {', '.join(summary['analyses_performed'])}")
    print(f"- Significant correlations found: {len(significant_corrs)}")
    print(f"- Files saved: {len(saved_files)}")