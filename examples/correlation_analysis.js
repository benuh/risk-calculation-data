/**
 * Correlation Analysis Example
 *
 * This example demonstrates the unsupervised correlation analysis
 * and anomaly detection capabilities of the platform.
 */

const AutomatedCorrelationReporter = require('../src/reports/AutomatedCorrelationReporter');
const CorrelationAnalyzer = require('../src/analysis/CorrelationAnalyzer');

function generateSampleCorrelatedData(n = 500, correlationMatrix = null) {
  // Default correlation matrix if none provided
  if (!correlationMatrix) {
    correlationMatrix = [
      [1.00, 0.75, 0.45, -0.30, 0.20],
      [0.75, 1.00, 0.60, -0.25, 0.15],
      [0.45, 0.60, 1.00, -0.10, 0.35],
      [-0.30, -0.25, -0.10, 1.00, -0.40],
      [0.20, 0.15, 0.35, -0.40, 1.00]
    ];
  }

  // Generate independent random data
  const randomData = Array.from({length: n}, () =>
    Array.from({length: correlationMatrix.length}, () => Math.random() - 0.5)
  );

  // Apply correlation structure using Cholesky decomposition
  const L = choleskyDecomposition(correlationMatrix);
  const correlatedData = randomData.map(row => {
    return matrixVectorMultiply(L, row);
  });

  // Add some interesting patterns

  // 1. Regime change in correlations (middle period)
  const regimeStart = Math.floor(n * 0.4);
  const regimeEnd = Math.floor(n * 0.6);
  for (let i = regimeStart; i < regimeEnd; i++) {
    // Increase correlation between assets 0 and 1
    const factor = 1.5;
    correlatedData[i][1] = correlatedData[i][0] * 0.9 + correlatedData[i][1] * 0.1 * factor;
  }

  // 2. Add outliers/jumps
  const numOutliers = Math.floor(n * 0.02); // 2% outliers
  for (let i = 0; i < numOutliers; i++) {
    const idx = Math.floor(Math.random() * n);
    const assetIdx = Math.floor(Math.random() * correlationMatrix.length);
    correlatedData[idx][assetIdx] += (Math.random() > 0.5 ? 3 : -3);
  }

  // 3. Volatility clustering in one asset
  for (let i = Math.floor(n * 0.7); i < Math.floor(n * 0.85); i++) {
    correlatedData[i][2] *= 2; // Higher volatility period
  }

  return correlatedData;
}

function choleskyDecomposition(matrix) {
  const n = matrix.length;
  const L = Array.from({length: n}, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      if (i === j) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[j][k] * L[j][k];
        }
        L[i][j] = Math.sqrt(matrix[i][i] - sum);
      } else {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (matrix[i][j] - sum) / L[j][j];
      }
    }
  }

  return L;
}

function matrixVectorMultiply(matrix, vector) {
  return matrix.map(row =>
    row.reduce((sum, val, idx) => sum + val * (vector[idx] || 0), 0)
  );
}

async function correlationAnalysisExample() {
  console.log('='.repeat(60));
  console.log('CORRELATION ANALYSIS EXAMPLE');
  console.log('='.repeat(60));

  // Initialize analyzer
  const analyzer = new CorrelationAnalyzer();
  const reporter = new AutomatedCorrelationReporter();

  // Asset names for the example
  const assetNames = ['Tech_Stock', 'Financial_Stock', 'Bond_ETF', 'Gold_ETF', 'Real_Estate'];

  console.log('\n1. GENERATING SAMPLE DATA');
  console.log('-'.repeat(30));

  // Generate correlated data with interesting patterns
  const correlatedData = generateSampleCorrelatedData(1000);

  // Convert to dataset format
  const datasets = assetNames.map((name, assetIndex) => ({
    name: name,
    values: correlatedData.map(row => row[assetIndex]),
    timestamps: Array.from({length: correlatedData.length}, (_, i) =>
      new Date(Date.now() - (correlatedData.length - i) * 24 * 60 * 60 * 1000))
  }));

  console.log(`Generated ${datasets.length} assets with ${datasets[0].values.length} observations each`);

  console.log('\n2. BASIC CORRELATION ANALYSIS');
  console.log('-'.repeat(30));

  try {
    // Perform pairwise correlation analysis
    const results = analyzer.performUnsupervisedCorrelationAnalysis(datasets);

    console.log(`Found ${results.significantCorrelations.length} significant correlations`);
    console.log(`Detected ${results.regimeChanges.filter(r => r.hasRegimeChange).length} regime changes`);
    console.log(`Identified ${results.nonLinearRelationships.filter(r => r.isNonLinear).length} non-linear relationships`);
    console.log(`Found ${results.tailDependencies.filter(t => t.hasTailDependence).length} tail dependencies`);

    // Show top correlations
    console.log('\nTop Significant Correlations:');
    results.significantCorrelations
      .sort((a, b) => Math.abs(b.pearsonCorrelation) - Math.abs(a.pearsonCorrelation))
      .slice(0, 5)
      .forEach(corr => {
        console.log(`  ${corr.asset1} - ${corr.asset2}: ρ = ${corr.pearsonCorrelation.toFixed(3)} ` +
                   `(p = ${corr.pearsonPValue.toFixed(4)})`);
      });

  } catch (error) {
    console.error('Error in basic correlation analysis:', error.message);
  }

  console.log('\n3. REGIME CHANGE DETECTION');
  console.log('-'.repeat(30));

  try {
    // Analyze specific pair for regime changes
    const asset1Data = datasets[0].values;
    const asset2Data = datasets[1].values;

    const pairAnalysis = analyzer.analyzePairwiseRelationship(
      { name: datasets[0].name, values: asset1Data },
      { name: datasets[1].name, values: asset2Data }
    );

    if (pairAnalysis.hasRegimeChange) {
      console.log(`Regime changes detected between ${datasets[0].name} and ${datasets[1].name}:`);
      console.log(`  Number of regimes: ${pairAnalysis.regimeAnalysis.regimes.length}`);
      console.log(`  Correlation volatility: ${pairAnalysis.regimeAnalysis.volatility.toFixed(3)}`);
      console.log(`  Change points: ${pairAnalysis.regimeAnalysis.changePoints.length}`);

      // Show regime details
      pairAnalysis.regimeAnalysis.regimes.forEach((regime, i) => {
        console.log(`    Regime ${i + 1}: Mean ρ = ${regime.meanCorrelation.toFixed(3)}, ` +
                   `Length = ${regime.length}, Volatility = ${regime.volatility.toFixed(3)}`);
      });
    } else {
      console.log('No significant regime changes detected in sample pair');
    }

  } catch (error) {
    console.error('Error in regime change detection:', error.message);
  }

  console.log('\n4. ANOMALY DETECTION');
  console.log('-'.repeat(30));

  try {
    // Detect anomalies in the datasets
    const anomalies = analyzer.detectAnomalousPatterns(datasets);

    console.log(`Anomalies detected in ${anomalies.length} assets:`);

    anomalies.forEach(anomaly => {
      console.log(`\n  ${anomaly.dataset}:`);
      console.log(`    Outliers: ${anomaly.outliers.length}`);

      if (anomaly.volatilityClusters.isSignificant) {
        console.log(`    Volatility clustering: Yes (${anomaly.volatilityClusters.significantLags.length} significant lags)`);
      }

      if (anomaly.jumpPatterns.hasJumps) {
        console.log(`    Jump patterns: ${anomaly.jumpPatterns.jumps.length} jumps detected`);
        console.log(`    Jump intensity: ${(anomaly.jumpPatterns.jumpIntensity * 100).toFixed(2)}%`);
      }

      if (anomaly.seasonalPatterns.hasAnomalies) {
        console.log(`    Seasonal anomalies: Detected`);
      }
    });

  } catch (error) {
    console.error('Error in anomaly detection:', error.message);
  }

  console.log('\n5. TAIL DEPENDENCY ANALYSIS');
  console.log('-'.repeat(30));

  try {
    // Analyze tail dependencies
    const results = analyzer.performUnsupervisedCorrelationAnalysis(datasets);
    const tailDeps = results.tailDependencies.filter(t => t.hasTailDependence);

    if (tailDeps.length > 0) {
      console.log(`Found ${tailDeps.length} significant tail dependencies:`);

      tailDeps.forEach(dep => {
        const upperTail = dep.tailDependence.upperTail || 0;
        const lowerTail = dep.tailDependence.lowerTail || 0;
        const asymmetry = Math.abs(upperTail - lowerTail);

        console.log(`  ${dep.asset1} - ${dep.asset2}:`);
        console.log(`    Upper tail: ${upperTail.toFixed(3)}`);
        console.log(`    Lower tail: ${lowerTail.toFixed(3)}`);
        console.log(`    Asymmetry: ${asymmetry.toFixed(3)} ${asymmetry > 0.1 ? '(High)' : '(Low)'}`);
      });
    } else {
      console.log('No significant tail dependencies found in sample data');
    }

  } catch (error) {
    console.error('Error in tail dependency analysis:', error.message);
  }

  console.log('\n6. COMPREHENSIVE AUTOMATED REPORT');
  console.log('-'.repeat(30));

  try {
    // Generate comprehensive automated report
    const options = {
      includeVisualizations: false, // Set to true if you want to generate Python plots
      outputFormat: 'json',
      correlationThreshold: 0.3,
      significanceLevel: 0.05
    };

    console.log('Generating comprehensive analysis report...');

    const report = await reporter.generateComprehensiveReport(datasets, options);

    // Display executive summary
    console.log('\nEXECUTIVE SUMMARY:');
    console.log(`  Overall Risk Level: ${report.executiveSummary.overallRiskLevel.toUpperCase()}`);
    console.log(`  Key Findings: ${report.executiveSummary.keyFindings.length}`);

    if (report.executiveSummary.keyFindings.length > 0) {
      report.executiveSummary.keyFindings.forEach((finding, i) => {
        console.log(`    ${i + 1}. ${finding}`);
      });
    }

    // Critical alerts
    if (report.executiveSummary.criticalAlerts.length > 0) {
      console.log('\n  CRITICAL ALERTS:');
      report.executiveSummary.criticalAlerts.forEach(alert => {
        console.log(`    ${alert.type}: ${alert.message}`);
      });
    }

    // Recommended actions
    if (report.executiveSummary.recommendedActions.length > 0) {
      console.log('\n  RECOMMENDED ACTIONS:');
      report.executiveSummary.recommendedActions.forEach(action => {
        console.log(`    ${action.priority}: ${action.action} (${action.timeline})`);
      });
    }

    console.log(`\nDetailed report saved to: ${reporter.reportDir}`);

  } catch (error) {
    console.error('Error generating comprehensive report:', error.message);
  }

  console.log('\n7. RISK MANAGEMENT INSIGHTS');
  console.log('-'.repeat(30));

  try {
    // Generate risk management insights based on findings
    const results = analyzer.performUnsupervisedCorrelationAnalysis(datasets);

    const highCorrelations = results.significantCorrelations.filter(c =>
      Math.abs(c.pearsonCorrelation) > 0.7).length;

    const regimeChanges = results.regimeChanges.filter(r =>
      r.hasRegimeChange).length;

    const nonLinearRelationships = results.nonLinearRelationships.filter(r =>
      r.isNonLinear).length;

    console.log('Risk Management Implications:');

    if (highCorrelations > 0) {
      console.log(`  • High correlations (${highCorrelations}): Concentration risk present`);
      console.log('    → Consider diversification strategies');
    }

    if (regimeChanges > 0) {
      console.log(`  • Regime changes (${regimeChanges}): Correlation instability detected`);
      console.log('    → Implement dynamic hedging strategies');
    }

    if (nonLinearRelationships > 0) {
      console.log(`  • Non-linear relationships (${nonLinearRelationships}): Standard models may be inadequate`);
      console.log('    → Consider advanced risk models');
    }

    const anomalyCount = results.anomalousPatterns.length;
    if (anomalyCount > 0) {
      console.log(`  • Anomalous patterns (${anomalyCount} assets): Enhanced monitoring needed`);
      console.log('    → Implement real-time anomaly detection');
    }

  } catch (error) {
    console.error('Error generating risk insights:', error.message);
  }

  console.log('\n' + '='.repeat(60));
  console.log('CORRELATION ANALYSIS EXAMPLE COMPLETED');
  console.log('='.repeat(60));
  console.log('\nTo run with visualizations:');
  console.log('1. Install Python dependencies: pip install -r requirements.txt');
  console.log('2. Set includeVisualizations: true in the options');
  console.log('3. Check the visualizations/ directory for generated plots');
}

// Run the example
if (require.main === module) {
  correlationAnalysisExample().catch(error => {
    console.error('Example failed:', error);
    process.exit(1);
  });
}

module.exports = {
  correlationAnalysisExample,
  generateSampleCorrelatedData
};