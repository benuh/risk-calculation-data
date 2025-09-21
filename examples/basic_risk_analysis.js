/**
 * Basic Risk Analysis Example
 *
 * This example demonstrates how to perform basic risk analysis
 * using the Risk Calculation Data Platform.
 */

const RiskCalculator = require('../src/calculators/RiskCalculator');
const AdvancedRiskCalculator = require('../src/calculators/AdvancedRiskCalculator');
const FMPClient = require('../src/api/fmp');

async function basicRiskAnalysisExample() {
  console.log('='.repeat(50));
  console.log('BASIC RISK ANALYSIS EXAMPLE');
  console.log('='.repeat(50));

  // Initialize calculators
  const riskCalc = new RiskCalculator();
  const advancedRiskCalc = new AdvancedRiskCalculator();

  // Sample price data (in practice, this would come from FMP API)
  const samplePrices = [
    100, 102, 98, 105, 103, 101, 99, 104, 106, 102,
    108, 105, 107, 103, 109, 111, 108, 106, 110, 112,
    108, 105, 103, 107, 109, 106, 104, 108, 110, 107
  ];

  const marketPrices = [
    1000, 1020, 980, 1050, 1030, 1010, 990, 1040, 1060, 1020,
    1080, 1050, 1070, 1030, 1090, 1110, 1080, 1060, 1100, 1120,
    1080, 1050, 1030, 1070, 1090, 1060, 1040, 1080, 1100, 1070
  ];

  console.log('\n1. BASIC RISK METRICS');
  console.log('-'.repeat(30));

  try {
    // Calculate basic volatility
    const volatility = riskCalc.calculateVolatility(samplePrices);
    console.log(`Annualized Volatility: ${(volatility * 100).toFixed(2)}%`);

    // Calculate beta
    const beta = riskCalc.calculateBeta(samplePrices, marketPrices);
    console.log(`Beta Coefficient: ${beta.toFixed(3)}`);

    // Calculate Sharpe ratio
    const sharpeRatio = riskCalc.calculateSharpeRatio(samplePrices);
    console.log(`Sharpe Ratio: ${sharpeRatio.toFixed(3)}`);

    // Calculate VaR
    const var95 = riskCalc.calculateValueAtRisk(samplePrices, 0.95);
    console.log(`VaR (95%): ${(var95 * 100).toFixed(2)}%`);

    // Calculate Expected Shortfall
    const es95 = riskCalc.calculateExpectedShortfall(samplePrices, 0.95);
    console.log(`Expected Shortfall (95%): ${(es95 * 100).toFixed(2)}%`);

    // Calculate Maximum Drawdown
    const maxDrawdown = riskCalc.calculateMaxDrawdown(samplePrices);
    console.log(`Maximum Drawdown: ${(maxDrawdown * 100).toFixed(2)}%`);

  } catch (error) {
    console.error('Error in basic calculations:', error.message);
  }

  console.log('\n2. ADVANCED RISK METRICS');
  console.log('-'.repeat(30));

  try {
    // Generate returns for advanced calculations
    const returns = [];
    for (let i = 1; i < samplePrices.length; i++) {
      returns.push((samplePrices[i] - samplePrices[i-1]) / samplePrices[i-1]);
    }

    // GARCH volatility
    try {
      const garchResults = advancedRiskCalc.calculateGarchVolatility(returns);
      console.log(`GARCH Volatility: ${(garchResults.currentVolatility * 100).toFixed(2)}%`);
    } catch (error) {
      console.log('GARCH calculation requires more data points');
    }

    // Cornish-Fisher VaR
    const cfVar = advancedRiskCalc.calculateCornishFisherVaR(returns);
    console.log(`Cornish-Fisher VaR (95%): ${(cfVar * 100).toFixed(2)}%`);

    // Higher moments
    const skewness = advancedRiskCalc.calculateSkewness(returns);
    const kurtosis = advancedRiskCalc.calculateKurtosis(returns);
    console.log(`Skewness: ${skewness.toFixed(3)}`);
    console.log(`Excess Kurtosis: ${kurtosis.toFixed(3)}`);

  } catch (error) {
    console.error('Error in advanced calculations:', error.message);
  }

  console.log('\n3. COMPREHENSIVE RISK ASSESSMENT');
  console.log('-'.repeat(30));

  try {
    // Generate returns
    const returns = [];
    for (let i = 1; i < samplePrices.length; i++) {
      returns.push((samplePrices[i] - samplePrices[i-1]) / samplePrices[i-1]);
    }

    const marketReturns = [];
    for (let i = 1; i < marketPrices.length; i++) {
      marketReturns.push((marketPrices[i] - marketPrices[i-1]) / marketPrices[i-1]);
    }

    // Comprehensive risk metrics
    const comprehensiveMetrics = advancedRiskCalc.calculateAdvancedRiskMetrics(
      'SAMPLE_ASSET',
      returns,
      marketReturns
    );

    console.log('Comprehensive Risk Report:');
    console.log(`  Symbol: ${comprehensiveMetrics.symbol}`);
    console.log(`  Volatility: ${(comprehensiveMetrics.volatility * 100).toFixed(2)}%`);
    console.log(`  Beta: ${comprehensiveMetrics.beta?.toFixed(3) || 'N/A'}`);
    console.log(`  VaR (95%): ${(comprehensiveMetrics.valueAtRisk * 100).toFixed(2)}%`);
    console.log(`  Expected Shortfall: ${(comprehensiveMetrics.expectedShortfall * 100).toFixed(2)}%`);
    console.log(`  Skewness: ${comprehensiveMetrics.skewness?.toFixed(3) || 'N/A'}`);
    console.log(`  Excess Kurtosis: ${comprehensiveMetrics.kurtosis?.toFixed(3) || 'N/A'}`);

  } catch (error) {
    console.error('Error in comprehensive assessment:', error.message);
  }

  console.log('\n4. PORTFOLIO RISK ANALYSIS');
  console.log('-'.repeat(30));

  try {
    // Sample portfolio
    const portfolioWeights = [0.4, 0.3, 0.3];
    const assetReturns = [
      returns,
      returns.map(r => r * 0.8 + 0.001), // Correlated but different asset
      returns.map(r => -r * 0.3 + 0.0005) // Negatively correlated asset
    ];

    // Simple correlation matrix
    const correlationMatrix = [
      [1.0, 0.7, -0.3],
      [0.7, 1.0, -0.2],
      [-0.3, -0.2, 1.0]
    ];

    // Component VaR
    const componentVar = advancedRiskCalc.calculateComponentVaR(
      portfolioWeights,
      assetReturns,
      correlationMatrix
    );

    console.log('Portfolio Risk Analysis:');
    console.log(`  Portfolio VaR: ${(componentVar.portfolioVaR * 100).toFixed(2)}%`);
    console.log('  Component Contributions:');
    componentVar.componentVaRs.forEach((comp, i) => {
      console.log(`    Asset ${i+1}: ${(comp.contribution * 100).toFixed(1)}% ` +
                 `(weight: ${(comp.weight * 100).toFixed(1)}%)`);
    });

  } catch (error) {
    console.error('Error in portfolio analysis:', error.message);
  }

  console.log('\n' + '='.repeat(50));
  console.log('EXAMPLE COMPLETED');
  console.log('='.repeat(50));
}

// Example with real API data (requires API keys)
async function realDataExample() {
  console.log('\n5. REAL DATA EXAMPLE (requires API keys)');
  console.log('-'.repeat(30));

  try {
    // This would require valid API keys in .env file
    const fmpClient = new FMPClient();

    console.log('Note: This example requires valid FMP API keys');
    console.log('To run with real data:');
    console.log('1. Add your FMP_API_KEY to .env file');
    console.log('2. Uncomment the API calls below');

    /*
    // Example of real API usage (uncomment when you have API keys):

    const symbol = 'AAPL';
    const priceData = await fmpClient.getStockPrice(symbol);
    const profileData = await fmpClient.getCompanyProfile(symbol);

    console.log(`Real data for ${symbol}:`);
    console.log(`Current Price: $${priceData[0]?.price || 'N/A'}`);
    console.log(`Company: ${profileData[0]?.companyName || 'N/A'}`);
    console.log(`Sector: ${profileData[0]?.sector || 'N/A'}`);

    // Calculate risk metrics with real data
    if (priceData && priceData.length > 30) {
      const prices = priceData.map(p => p.price);
      const riskCalc = new RiskCalculator();

      const volatility = riskCalc.calculateVolatility(prices);
      const var95 = riskCalc.calculateValueAtRisk(prices, 0.95);

      console.log(`Volatility: ${(volatility * 100).toFixed(2)}%`);
      console.log(`VaR (95%): ${(var95 * 100).toFixed(2)}%`);
    }
    */

  } catch (error) {
    console.log('Real data example requires API configuration');
    console.log('Error:', error.message);
  }
}

// Run the example
if (require.main === module) {
  basicRiskAnalysisExample()
    .then(() => realDataExample())
    .catch(error => {
      console.error('Example failed:', error);
      process.exit(1);
    });
}

module.exports = {
  basicRiskAnalysisExample,
  realDataExample
};