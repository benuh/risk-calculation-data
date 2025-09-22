/**
 * Stress Testing Example
 *
 * This example demonstrates comprehensive stress testing capabilities
 * including historical scenarios, Monte Carlo simulations, and custom shock testing.
 */

const StressTestingFramework = require('../src/stress-testing/StressTestingFramework');

// Sample portfolio for testing
function createSamplePortfolio() {
  return {
    name: 'Diversified Portfolio',
    totalValue: 1000000, // $1M portfolio
    assets: [
      {
        name: 'US Large Cap Equities',
        type: 'equity',
        price: 100,
        quantity: 3000,
        weight: 0.30,
        volatility: 0.16,
        currency: 'USD',
        beta: 1.0
      },
      {
        name: 'International Developed Equities',
        type: 'equity',
        price: 50,
        quantity: 4000,
        weight: 0.20,
        volatility: 0.18,
        currency: 'EUR',
        beta: 0.9
      },
      {
        name: 'Emerging Market Equities',
        type: 'equity',
        price: 25,
        quantity: 4000,
        weight: 0.10,
        volatility: 0.25,
        currency: 'USD',
        beta: 1.3
      },
      {
        name: 'US Treasury Bonds',
        type: 'bond',
        price: 100,
        quantity: 2500,
        weight: 0.25,
        volatility: 0.04,
        duration: 7,
        currency: 'USD'
      },
      {
        name: 'Corporate Bonds',
        type: 'bond',
        price: 95,
        quantity: 1000,
        weight: 0.10,
        volatility: 0.06,
        duration: 5,
        currency: 'USD',
        creditRating: 'BBB'
      },
      {
        name: 'Commodities',
        type: 'commodity',
        price: 200,
        quantity: 250,
        weight: 0.05,
        volatility: 0.22,
        currency: 'USD'
      }
    ],
    correlationMatrix: [
      [1.00, 0.75, 0.85, -0.20, 0.10, 0.15],
      [0.75, 1.00, 0.80, -0.15, 0.05, 0.20],
      [0.85, 0.80, 1.00, -0.10, 0.15, 0.25],
      [-0.20, -0.15, -0.10, 1.00, 0.40, -0.05],
      [0.10, 0.05, 0.15, 0.40, 1.00, 0.10],
      [0.15, 0.20, 0.25, -0.05, 0.10, 1.00]
    ],
    peakValue: 1000000
  };
}

async function stressTestingExample() {
  console.log('='.repeat(60));
  console.log('STRESS TESTING FRAMEWORK EXAMPLE');
  console.log('='.repeat(60));

  // Initialize stress testing framework
  const stressTest = new StressTestingFramework({
    monteCarloIterations: 5000,
    confidenceLevels: [0.90, 0.95, 0.99, 0.995],
    shockMagnitudes: [2, 3, 4, 5]
  });

  // Create sample portfolio
  const portfolio = createSamplePortfolio();

  console.log('\\nPORTFOLIO SUMMARY:');
  console.log('-'.repeat(30));
  console.log(`Total Value: $${portfolio.totalValue.toLocaleString()}`);
  console.log(`Number of Assets: ${portfolio.assets.length}`);
  console.log('Asset Allocation:');
  portfolio.assets.forEach(asset => {
    console.log(`  ${asset.name}: ${(asset.weight * 100).toFixed(1)}% ($${(asset.price * asset.quantity).toLocaleString()})`);
  });

  console.log('\\n1. HISTORICAL SCENARIO TESTING');
  console.log('-'.repeat(40));

  try {
    // Run historical scenarios only
    const historicalResults = await stressTest.runHistoricalScenarios(portfolio);

    console.log(`\\nTested ${historicalResults.count} historical scenarios:`);
    console.log('\\nTop 3 Most Damaging Scenarios:');

    const sortedScenarios = historicalResults.scenarios
      .sort((a, b) => a.pnlPercent - b.pnlPercent)
      .slice(0, 3);

    sortedScenarios.forEach((scenario, index) => {
      console.log(`\\n${index + 1}. ${scenario.scenario.toUpperCase()}:`);
      console.log(`   Description: ${scenario.description}`);
      console.log(`   Portfolio Impact: ${scenario.pnlPercent.toFixed(2)}%`);
      console.log(`   Dollar Impact: $${scenario.pnl.toFixed(0)}`);
      console.log(`   Final Value: $${scenario.shockedValue.toFixed(0)}`);
      console.log(`   Max Drawdown: ${(scenario.maxDrawdown * 100).toFixed(2)}%`);
    });

    console.log(`\\nAverage Historical Impact: ${historicalResults.averageImpact.toFixed(2)}%`);

  } catch (error) {
    console.error('Error in historical scenario testing:', error.message);
  }

  console.log('\\n2. MONTE CARLO STRESS TESTING');
  console.log('-'.repeat(40));

  try {
    const monteCarloResults = await stressTest.runMonteCarloStressTest(portfolio);

    console.log(`\\nMonte Carlo Simulation Results (${monteCarloResults.iterations.toLocaleString()} iterations):`);
    console.log('\\nValue at Risk (VaR):');
    Object.entries(monteCarloResults.percentiles).forEach(([level, value]) => {
      console.log(`  ${level}: ${value.toFixed(2)}%`);
    });

    console.log(`\\nStatistical Summary:`);
    console.log(`  Average P&L: ${monteCarloResults.averagePnL.toFixed(2)}%`);
    console.log(`  Standard Deviation: ${monteCarloResults.standardDeviation.toFixed(2)}%`);
    console.log(`  Best Case: ${monteCarloResults.bestCase.toFixed(2)}%`);
    console.log(`  Worst Case: ${monteCarloResults.worstCase.toFixed(2)}%`);

  } catch (error) {
    console.error('Error in Monte Carlo stress testing:', error.message);
  }

  console.log('\\n3. EXTREME VALUE TESTING');
  console.log('-'.repeat(40));

  try {
    const extremeResults = await stressTest.runExtremeValueTest(portfolio);

    console.log(`\\nExtreme Value Test Results (${extremeResults.count} scenarios):`);
    console.log('\\nWorst Case by Shock Type:');

    Object.entries(extremeResults.worstCaseByType).forEach(([shockType, result]) => {
      console.log(`  ${shockType.replace('_', ' ').toUpperCase()}: ${result.pnlPercent.toFixed(2)}%`);
    });

    console.log('\\nExtreme Scenario Summary:');
    extremeResults.summary.forEach(summary => {
      console.log(`  ${summary.shockType.replace('_', ' ').toUpperCase()}:`);
      console.log(`    Worst Case: ${summary.worstCase.toFixed(2)}%`);
      console.log(`    Average Impact: ${summary.averageImpact.toFixed(2)}%`);
      console.log(`    Tests Run: ${summary.testCount}`);
    });

  } catch (error) {
    console.error('Error in extreme value testing:', error.message);
  }

  console.log('\\n4. CORRELATION BREAKDOWN TESTING');
  console.log('-'.repeat(40));

  try {
    const correlationResults = await stressTest.runCorrelationBreakdownTest(portfolio);

    console.log(`\\nCorrelation Breakdown Test Results:`);
    console.log('Impact of Correlation Changes:');

    correlationResults.results.forEach(result => {
      console.log(`  Correlation Shock ${result.correlationShock >= 0 ? '+' : ''}${result.correlationShock}: ${result.pnlPercent.toFixed(2)}%`);
    });

    console.log(`\\nDiversification Analysis:`);
    const diversAnalysis = correlationResults.diversificationAnalysis;
    console.log(`  Correlation Sensitivity: ${diversAnalysis.correlationSensitivity.toFixed(3)}`);
    console.log(`  Diversification Benefit Loss: ${diversAnalysis.diversificationBenefitLoss.toFixed(2)}%`);

  } catch (error) {
    console.error('Error in correlation breakdown testing:', error.message);
  }

  console.log('\\n5. CUSTOM SCENARIO TESTING');
  console.log('-'.repeat(40));

  try {
    // Add custom scenarios
    stressTest.addCustomScenario('fed_rate_shock', 'Federal Reserve Emergency Rate Hike', {
      bonds: -0.15,
      equities: -0.20,
      credit_spreads: 0.02,
      currencies: { USD: 0.10, EUR: -0.05 }
    });

    stressTest.addCustomScenario('tech_bubble_burst', 'Technology Sector Bubble Burst', {
      equities: -0.40,
      bonds: 0.05,
      volatility_shock: 2.0,
      correlation_breakdown: true
    });

    stressTest.addCustomScenario('supply_chain_crisis', 'Global Supply Chain Crisis', {
      commodities: 0.30,
      equities: -0.15,
      bonds: -0.05,
      currencies: { USD: 0.05 }
    });

    const customResults = await stressTest.runCustomScenarios(portfolio);

    console.log(`\\nCustom Scenario Results (${customResults.count} scenarios):`);
    customResults.scenarios.forEach(scenario => {
      console.log(`\\n  ${scenario.scenario.replace('_', ' ').toUpperCase()}:`);
      console.log(`    Description: ${scenario.description}`);
      console.log(`    Impact: ${scenario.pnlPercent.toFixed(2)}%`);
      console.log(`    Dollar Impact: $${scenario.pnl.toFixed(0)}`);
    });

    console.log(`\\nWorst Custom Scenario: ${customResults.worstCase.scenario} (${customResults.worstCase.pnlPercent.toFixed(2)}%)`);

  } catch (error) {
    console.error('Error in custom scenario testing:', error.message);
  }

  console.log('\\n6. COMPREHENSIVE STRESS TEST');
  console.log('-'.repeat(40));

  try {
    // Run comprehensive stress test
    const comprehensiveResults = await stressTest.runComprehensiveStressTest(portfolio, {
      includeHistorical: true,
      includeCustom: true,
      includeMonteCarlo: true,
      includeExtremeValue: true,
      includeCorrelationBreakdown: true
    });

    console.log('\\nCOMPREHENSIVE STRESS TEST SUMMARY:');
    console.log(`Overall Risk Level: ${comprehensiveResults.summary.overallRiskLevel}`);

    if (comprehensiveResults.summary.worstCaseScenario) {
      console.log(`\\nWorst Case Scenario Overall:`);
      console.log(`  Scenario: ${comprehensiveResults.summary.worstCaseScenario.scenario || 'Monte Carlo Simulation'}`);
      console.log(`  Impact: ${comprehensiveResults.summary.worstCaseScenario.pnlPercent.toFixed(2)}%`);
    }

    if (comprehensiveResults.summary.statisticalSummary) {
      console.log(`\\nStatistical Risk Metrics:`);
      const stats = comprehensiveResults.summary.statisticalSummary;
      console.log(`  VaR (95%): ${stats.var95?.toFixed(2) || 'N/A'}%`);
      console.log(`  VaR (99%): ${stats.var99?.toFixed(2) || 'N/A'}%`);
      console.log(`  Expected Shortfall: ${stats.expectedShortfall?.toFixed(2) || 'N/A'}%`);
    }

    console.log(`\\nKEY RECOMMENDATIONS:`);
    comprehensiveResults.recommendations.forEach((rec, index) => {
      console.log(`${index + 1}. [${rec.priority}] ${rec.action}`);
      console.log(`   Category: ${rec.category}`);
      console.log(`   Timeline: ${rec.timeline}`);
      console.log(`   Rationale: ${rec.description}\\n`);
    });

    // Export results
    console.log('\\nEXPORTING RESULTS:');
    const testId = Array.from(stressTest.results.keys())[0];
    const executiveSummary = stressTest.exportStressTestResults(testId, 'summary');
    console.log(executiveSummary);

  } catch (error) {
    console.error('Error in comprehensive stress testing:', error.message);
  }

  console.log('\\n7. RISK MANAGEMENT IMPLICATIONS');
  console.log('-'.repeat(40));

  try {
    console.log('Based on stress test results, consider the following:');
    console.log('\\nIMMEDIATE ACTIONS:');
    console.log('• Review position sizes and concentration risk');
    console.log('• Consider implementing protective hedging strategies');
    console.log('• Evaluate correlation assumptions in risk models');

    console.log('\\nMONITORING RECOMMENDATIONS:');
    console.log('• Set up real-time stress testing alerts');
    console.log('• Monitor correlation changes between asset classes');
    console.log('• Track early warning indicators for identified scenarios');

    console.log('\\nPORTFOLIO ADJUSTMENTS:');
    console.log('• Consider reducing exposure to highly correlated assets');
    console.log('• Evaluate adding uncorrelated or negatively correlated assets');
    console.log('• Review rebalancing frequency during stress periods');

  } catch (error) {
    console.error('Error generating risk management implications:', error.message);
  }

  console.log('\\n' + '='.repeat(60));
  console.log('STRESS TESTING EXAMPLE COMPLETED');
  console.log('='.repeat(60));
  console.log('\\nNote: This example uses simulated data.');
  console.log('In production, use actual market data and');
  console.log('calibrated risk models for accurate results.');
}

// Run the example
if (require.main === module) {
  stressTestingExample().catch(error => {
    console.error('Stress testing example failed:', error);
    process.exit(1);
  });
}

module.exports = {
  stressTestingExample,
  createSamplePortfolio
};