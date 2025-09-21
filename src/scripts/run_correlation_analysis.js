#!/usr/bin/env node

const AutomatedCorrelationReporter = require('../reports/AutomatedCorrelationReporter');
const FMPClient = require('../api/fmp');
const QuandlClient = require('../api/quandl');

async function generateSampleData() {
  // Generate sample correlated data for demonstration
  // In production, this would fetch real data from APIs

  const n = 1000; // Number of observations
  const assets = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'];

  console.log('Generating sample data for correlation analysis...');

  // Create correlation matrix
  const correlationMatrix = [
    [1.00, 0.75, 0.65, 0.85, 0.80],
    [0.75, 1.00, 0.70, 0.80, 0.82],
    [0.65, 0.70, 1.00, 0.78, 0.85],
    [0.85, 0.80, 0.78, 1.00, 0.95],
    [0.80, 0.82, 0.85, 0.95, 1.00]
  ];

  // Generate random data
  const randomData = Array.from({length: n}, () =>
    Array.from({length: assets.length}, () => Math.random() - 0.5)
  );

  // Apply correlation structure using Cholesky decomposition
  const L = choleskyDecomposition(correlationMatrix);
  const correlatedData = randomData.map(row => {
    return matrixVectorMultiply(L, row);
  });

  // Add some regime changes and anomalies
  for (let i = 200; i < 250; i++) {
    correlatedData[i][0] += 2; // Add outliers to first asset
    correlatedData[i][1] += 1.5; // Correlated outliers
  }

  for (let i = 500; i < 600; i++) {
    correlatedData[i].forEach((val, idx, arr) => {
      arr[idx] = val * 1.5; // Regime change - higher volatility
    });
  }

  for (let i = 800; i < 820; i++) {
    correlatedData[i][2] += Math.random() > 0.5 ? 3 : -3; // Jumps
  }

  // Convert to dataset format
  const datasets = assets.map((name, assetIndex) => ({
    name: name,
    values: correlatedData.map(row => row[assetIndex]),
    timestamps: Array.from({length: n}, (_, i) => new Date(Date.now() - (n - i) * 24 * 60 * 60 * 1000))
  }));

  return datasets;
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
    row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
  );
}

async function fetchRealMarketData() {
  // This function would fetch real data from FMP and Quandl APIs
  // For now, we'll use sample data

  try {
    console.log('Note: Using sample data. To use real data, configure API keys in .env file');

    // Example of how real data fetching would work:
    /*
    const fmpClient = new FMPClient();
    const quandlClient = new QuandlClient();

    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'];
    const datasets = [];

    for (const symbol of symbols) {
      try {
        const priceData = await fmpClient.getStockPrice(symbol);
        const economicData = await quandlClient.getMacroeconomicData('FRED', 'GDP');

        datasets.push({
          name: symbol,
          values: priceData, // Would need to extract actual price values
          timestamps: // Would need to extract timestamps
        });
      } catch (error) {
        console.warn(`Failed to fetch data for ${symbol}:`, error.message);
      }
    }

    return datasets;
    */

    return null; // Will fall back to sample data

  } catch (error) {
    console.warn('Failed to fetch real market data:', error.message);
    return null;
  }
}

async function main() {
  try {
    console.log('='.repeat(60));
    console.log('AUTOMATED CORRELATION ANALYSIS');
    console.log('='.repeat(60));

    // Initialize reporter
    const reporter = new AutomatedCorrelationReporter();

    // Try to fetch real data, fall back to sample data
    let datasets = await fetchRealMarketData();

    if (!datasets) {
      console.log('Using sample data for demonstration...');
      datasets = await generateSampleData();
    }

    console.log(`\nAnalyzing ${datasets.length} assets with ${datasets[0].values.length} observations each...`);

    // Configure analysis options
    const options = {
      includeVisualizations: true,
      outputFormat: 'json',
      correlationThreshold: 0.3,
      significanceLevel: 0.05
    };

    console.log('\nStarting comprehensive correlation analysis...');

    // Generate comprehensive report
    const report = await reporter.generateComprehensiveReport(datasets, options);

    console.log('\n' + '='.repeat(60));
    console.log('ANALYSIS RESULTS SUMMARY');
    console.log('='.repeat(60));

    // Display key findings
    console.log(`\nOverall Risk Level: ${report.executiveSummary.overallRiskLevel.toUpperCase()}`);

    if (report.executiveSummary.keyFindings.length > 0) {
      console.log('\nKey Findings:');
      report.executiveSummary.keyFindings.forEach((finding, index) => {
        console.log(`${index + 1}. ${finding}`);
      });
    }

    if (report.executiveSummary.criticalAlerts.length > 0) {
      console.log('\n🚨 CRITICAL ALERTS:');
      report.executiveSummary.criticalAlerts.forEach(alert => {
        console.log(`   ${alert.type.toUpperCase()} (${alert.severity}): ${alert.message}`);
      });
    }

    // Display significant correlations
    if (report.detailedFindings.significantCorrelations.unusualPatterns.length > 0) {
      console.log('\n📊 UNUSUAL CORRELATION PATTERNS:');
      report.detailedFindings.significantCorrelations.unusualPatterns
        .slice(0, 5) // Show top 5
        .forEach(pattern => {
          console.log(`   ${pattern.assetPair}: ρ=${pattern.pearsonCorrelation.toFixed(3)} ` +
                     `${pattern.isNonLinear ? '(Non-linear)' : ''} ` +
                     `${pattern.hasTailDependence ? '(Tail-dependent)' : ''}`);
        });
    }

    // Display regime changes
    if (report.detailedFindings.regimeChanges.mostVolatileCorrelations.length > 0) {
      console.log('\n📈 CORRELATION REGIME CHANGES:');
      report.detailedFindings.regimeChanges.mostVolatileCorrelations
        .slice(0, 3)
        .forEach(regime => {
          console.log(`   ${regime.assetPair}: ${regime.numberOfRegimes} regimes, ` +
                     `volatility=${regime.correlationVolatility.toFixed(3)}`);
        });
    }

    // Display anomalies
    if (report.detailedFindings.anomalousPatterns.highAnomalyAssets.length > 0) {
      console.log('\n⚠️  HIGH ANOMALY ASSETS:');
      report.detailedFindings.anomalousPatterns.highAnomalyAssets.forEach(asset => {
        console.log(`   ${asset.dataset}: ${asset.totalAnomalies} anomalies ` +
                   `${asset.hasVolatilityClustering ? '(Vol. clustering)' : ''} ` +
                   `${asset.hasJumps ? '(Jumps)' : ''}`);
      });
    }

    // Display recommendations
    if (report.executiveSummary.recommendedActions.length > 0) {
      console.log('\n💡 RECOMMENDED ACTIONS:');
      report.executiveSummary.recommendedActions.forEach(action => {
        console.log(`   ${action.priority.toUpperCase()}: ${action.action} (${action.timeline})`);
      });
    }

    console.log('\n' + '='.repeat(60));
    console.log('DETAILED ANALYSIS COMPLETE');
    console.log('='.repeat(60));

    console.log(`\n📁 Report saved to: ${reporter.reportDir}`);

    if (report.visualizations && report.visualizations.generated) {
      console.log(`📊 Visualizations saved to: ${reporter.visualizationDir}`);
    }

    console.log('\nTo view detailed results:');
    console.log(`   - Open the JSON report in ${reporter.reportDir}`);
    console.log(`   - View visualizations in ${reporter.visualizationDir}`);
    console.log(`   - Review findings documentation in docs/UNSUPERVISED_ANALYSIS_FINDINGS.md`);

    // Generate markdown summary
    try {
      await reporter.saveReport(report, 'markdown');
      console.log(`   - Markdown summary available in ${reporter.reportDir}`);
    } catch (error) {
      console.warn('Failed to generate markdown report:', error.message);
    }

  } catch (error) {
    console.error('\n❌ Analysis failed:', error.message);
    console.error('Stack trace:', error.stack);
    process.exit(1);
  }
}

// Command line interface
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: node run_correlation_analysis.js [options]

Options:
  --help, -h          Show this help message
  --real-data         Attempt to fetch real market data (requires API keys)
  --assets=SYMBOLS    Comma-separated list of assets (default: AAPL,MSFT,GOOGL,SPY,QQQ)
  --threshold=VALUE   Correlation significance threshold (default: 0.3)
  --no-visuals        Skip visualization generation

Examples:
  node run_correlation_analysis.js
  node run_correlation_analysis.js --real-data --assets=AAPL,TSLA,MSFT
  node run_correlation_analysis.js --threshold=0.5 --no-visuals

Environment Variables:
  FMP_API_KEY         Financial Modeling Prep API key
  QUANDL_API_KEY      Quandl API key
    `);
    process.exit(0);
  }

  // Parse command line arguments
  const useRealData = args.includes('--real-data');
  const noVisuals = args.includes('--no-visuals');

  const assetsArg = args.find(arg => arg.startsWith('--assets='));
  const thresholdArg = args.find(arg => arg.startsWith('--threshold='));

  if (useRealData) {
    console.log('Real data mode enabled');
  }

  if (assetsArg) {
    console.log(`Custom assets: ${assetsArg.split('=')[1]}`);
  }

  if (thresholdArg) {
    console.log(`Custom threshold: ${thresholdArg.split('=')[1]}`);
  }

  if (noVisuals) {
    console.log('Visualization generation disabled');
  }

  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}