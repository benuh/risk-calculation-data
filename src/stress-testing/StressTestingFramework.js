/**
 * Advanced Stress Testing Framework
 *
 * This module provides comprehensive stress testing capabilities including
 * historical scenarios, Monte Carlo simulations, extreme value testing,
 * and custom shock scenarios for portfolio risk assessment.
 */

class StressTestingFramework {
  constructor(config = {}) {
    this.config = {
      confidenceLevels: [0.90, 0.95, 0.99, 0.999],
      monteCarloIterations: 10000,
      timeHorizon: 252, // 1 year in trading days
      shockMagnitudes: [2, 3, 4, 5], // Standard deviations
      correlationShocks: [-0.5, 0, 0.5, 1.0],
      ...config
    };

    this.historicalScenarios = new Map();
    this.customScenarios = new Map();
    this.results = new Map();
  }

  // Historical Scenario Testing
  addHistoricalScenario(name, description, shocks) {
    this.historicalScenarios.set(name, {
      name,
      description,
      shocks,
      date: new Date(),
      type: 'historical'
    });
  }

  initializeHistoricalScenarios() {
    // 2008 Financial Crisis
    this.addHistoricalScenario('2008_crisis', '2008 Financial Crisis', {
      equities: -0.50,
      bonds: -0.15,
      commodities: -0.35,
      currencies: { USD: 0.15, EUR: -0.20, JPY: 0.10 },
      credit_spreads: 0.40,
      volatility_shock: 2.5
    });

    // 2020 COVID-19 Pandemic
    this.addHistoricalScenario('covid_2020', 'COVID-19 Pandemic March 2020', {
      equities: -0.35,
      bonds: 0.05,
      commodities: -0.25,
      oil: -0.60,
      currencies: { USD: 0.05, EUR: -0.10, GBP: -0.15 },
      credit_spreads: 0.30,
      volatility_shock: 3.0
    });

    // 1998 LTCM / Russian Crisis
    this.addHistoricalScenario('ltcm_1998', 'LTCM / Russian Crisis 1998', {
      equities: -0.25,
      bonds: 0.10,
      emerging_markets: -0.45,
      currencies: { RUB: -0.70, USD: 0.10 },
      credit_spreads: 0.25,
      volatility_shock: 2.0
    });

    // 2011 European Debt Crisis
    this.addHistoricalScenario('eu_debt_2011', 'European Debt Crisis 2011', {
      equities: -0.30,
      bonds: { us_treasury: 0.15, eu_sovereign: -0.20 },
      currencies: { EUR: -0.15, CHF: 0.20, USD: 0.08 },
      credit_spreads: 0.35,
      volatility_shock: 2.2
    });

    // Black Monday 1987
    this.addHistoricalScenario('black_monday_1987', 'Black Monday 1987', {
      equities: -0.22,
      bonds: 0.05,
      volatility_shock: 4.0,
      correlation_breakdown: true
    });
  }

  // Custom Scenario Testing
  addCustomScenario(name, description, shocks) {
    this.customScenarios.set(name, {
      name,
      description,
      shocks,
      date: new Date(),
      type: 'custom'
    });
  }

  // Comprehensive Stress Testing
  async runComprehensiveStressTest(portfolio, options = {}) {
    const {
      includeHistorical = true,
      includeCustom = true,
      includeMonteCarlo = true,
      includeExtremeValue = true,
      includeCorrelationBreakdown = true
    } = options;

    console.log('Starting Comprehensive Stress Testing...');

    const results = {
      portfolio: portfolio,
      timestamp: new Date(),
      scenarios: {},
      summary: {},
      recommendations: []
    };

    // Initialize historical scenarios if not already done
    if (this.historicalScenarios.size === 0) {
      this.initializeHistoricalScenarios();
    }

    try {
      // Historical Scenario Testing
      if (includeHistorical) {
        console.log('Running historical scenario tests...');
        results.scenarios.historical = await this.runHistoricalScenarios(portfolio);
      }

      // Custom Scenario Testing
      if (includeCustom && this.customScenarios.size > 0) {
        console.log('Running custom scenario tests...');
        results.scenarios.custom = await this.runCustomScenarios(portfolio);
      }

      // Monte Carlo Stress Testing
      if (includeMonteCarlo) {
        console.log('Running Monte Carlo stress tests...');
        results.scenarios.monteCarlo = await this.runMonteCarloStressTest(portfolio);
      }

      // Extreme Value Testing
      if (includeExtremeValue) {
        console.log('Running extreme value tests...');
        results.scenarios.extremeValue = await this.runExtremeValueTest(portfolio);
      }

      // Correlation Breakdown Testing
      if (includeCorrelationBreakdown) {
        console.log('Running correlation breakdown tests...');
        results.scenarios.correlationBreakdown = await this.runCorrelationBreakdownTest(portfolio);
      }

      // Generate summary and recommendations
      results.summary = this.generateStressSummary(results.scenarios);
      results.recommendations = this.generateRecommendations(results.summary);

      this.results.set(`stress_test_${Date.now()}`, results);

      console.log('Comprehensive stress testing completed.');
      return results;

    } catch (error) {
      console.error('Error in comprehensive stress testing:', error);
      throw error;
    }
  }

  async runHistoricalScenarios(portfolio) {
    const results = [];

    for (const [scenarioName, scenario] of this.historicalScenarios) {
      const result = await this.applyScenarioShocks(portfolio, scenario);
      results.push({
        scenario: scenarioName,
        description: scenario.description,
        ...result
      });
    }

    return {
      count: results.length,
      scenarios: results,
      worstCase: this.findWorstCaseScenario(results),
      averageImpact: this.calculateAverageImpact(results)
    };
  }

  async runCustomScenarios(portfolio) {
    const results = [];

    for (const [scenarioName, scenario] of this.customScenarios) {
      const result = await this.applyScenarioShocks(portfolio, scenario);
      results.push({
        scenario: scenarioName,
        description: scenario.description,
        ...result
      });
    }

    return {
      count: results.length,
      scenarios: results,
      worstCase: this.findWorstCaseScenario(results),
      averageImpact: this.calculateAverageImpact(results)
    };
  }

  async runMonteCarloStressTest(portfolio) {
    const iterations = this.config.monteCarloIterations;
    const results = [];

    for (let i = 0; i < iterations; i++) {
      // Generate random shocks
      const shocks = this.generateRandomShocks(portfolio);
      const result = await this.applyShocksToPortfolio(portfolio, shocks);

      results.push({
        iteration: i + 1,
        shocks,
        portfolioValue: result.portfolioValue,
        pnl: result.pnl,
        pnlPercent: result.pnlPercent,
        maxDrawdown: result.maxDrawdown
      });
    }

    // Calculate percentiles
    const pnlValues = results.map(r => r.pnlPercent).sort((a, b) => a - b);
    const percentiles = {};

    this.config.confidenceLevels.forEach(level => {
      const index = Math.floor((1 - level) * pnlValues.length);
      percentiles[`VaR_${(level * 100).toFixed(0)}%`] = pnlValues[index];
    });

    return {
      iterations,
      percentiles,
      averagePnL: pnlValues.reduce((sum, val) => sum + val, 0) / pnlValues.length,
      standardDeviation: this.calculateStandardDeviation(pnlValues),
      worstCase: Math.min(...pnlValues),
      bestCase: Math.max(...pnlValues),
      results: results.slice(0, 100) // Store only first 100 for memory efficiency
    };
  }

  async runExtremeValueTest(portfolio) {
    const results = [];

    // Test extreme scenarios based on different shock magnitudes
    for (const shockMagnitude of this.config.shockMagnitudes) {
      const extremeShocks = this.generateExtremeShocks(portfolio, shockMagnitude);

      for (const [shockType, shockData] of Object.entries(extremeShocks)) {
        const result = await this.applyShocksToPortfolio(portfolio, { [shockType]: shockData });

        results.push({
          shockType,
          shockMagnitude,
          shockValue: shockData,
          portfolioValue: result.portfolioValue,
          pnl: result.pnl,
          pnlPercent: result.pnlPercent,
          affectedAssets: result.affectedAssets
        });
      }
    }

    return {
      count: results.length,
      results,
      worstCaseByType: this.groupWorstCaseByShockType(results),
      summary: this.summarizeExtremeValueResults(results)
    };
  }

  async runCorrelationBreakdownTest(portfolio) {
    const results = [];

    for (const correlationShock of this.config.correlationShocks) {
      // Test what happens when correlations increase dramatically
      const result = await this.applyCorrelationShock(portfolio, correlationShock);

      results.push({
        correlationShock,
        description: this.getCorrelationShockDescription(correlationShock),
        portfolioValue: result.portfolioValue,
        pnl: result.pnl,
        pnlPercent: result.pnlPercent,
        diversificationBenefit: result.diversificationBenefit,
        concentrationRisk: result.concentrationRisk
      });
    }

    return {
      count: results.length,
      results,
      worstCase: this.findWorstCaseScenario(results),
      diversificationAnalysis: this.analyzeDiversificationBreakdown(results)
    };
  }

  // Shock Application Methods
  async applyScenarioShocks(portfolio, scenario) {
    return await this.applyShocksToPortfolio(portfolio, scenario.shocks);
  }

  async applyShocksToPortfolio(portfolio, shocks) {
    const originalValue = this.calculatePortfolioValue(portfolio);
    const shockedPortfolio = this.deepClone(portfolio);

    // Apply shocks to each asset class
    this.applyAssetClassShocks(shockedPortfolio, shocks);

    // Apply volatility shocks if specified
    if (shocks.volatility_shock) {
      this.applyVolatilityShock(shockedPortfolio, shocks.volatility_shock);
    }

    // Apply correlation changes if specified
    if (shocks.correlation_breakdown) {
      this.applyCorrelationBreakdown(shockedPortfolio);
    }

    const shockedValue = this.calculatePortfolioValue(shockedPortfolio);
    const pnl = shockedValue - originalValue;
    const pnlPercent = (pnl / originalValue) * 100;

    return {
      originalValue,
      shockedValue: shockedValue,
      pnl,
      pnlPercent,
      maxDrawdown: this.calculateMaxDrawdown(shockedPortfolio),
      affectedAssets: this.identifyAffectedAssets(portfolio, shockedPortfolio),
      riskMetrics: this.calculateStressedRiskMetrics(shockedPortfolio)
    };
  }

  applyAssetClassShocks(portfolio, shocks) {
    if (!portfolio.assets) return;

    portfolio.assets.forEach(asset => {
      const assetClass = this.getAssetClass(asset);
      let shockValue = 0;

      // Apply direct asset class shocks
      if (shocks[assetClass]) {
        shockValue = typeof shocks[assetClass] === 'number' ?
          shocks[assetClass] : shocks[assetClass][asset.subclass] || 0;
      }

      // Apply currency shocks if applicable
      if (shocks.currencies && asset.currency) {
        shockValue += shocks.currencies[asset.currency] || 0;
      }

      // Apply credit spread shocks if applicable
      if (shocks.credit_spreads && asset.type === 'bond') {
        const creditImpact = -shocks.credit_spreads * (asset.duration || 5);
        shockValue += creditImpact;
      }

      // Apply the shock
      if (shockValue !== 0) {
        asset.price = asset.price * (1 + shockValue);
        asset.shockApplied = shockValue;
      }
    });
  }

  applyVolatilityShock(portfolio, volatilityMultiplier) {
    if (!portfolio.assets) return;

    portfolio.assets.forEach(asset => {
      if (asset.volatility) {
        asset.volatility *= volatilityMultiplier;
      }
    });
  }

  applyCorrelationBreakdown(portfolio) {
    // Simulate correlation breakdown by increasing correlations towards 1
    if (portfolio.correlationMatrix) {
      const matrix = portfolio.correlationMatrix;
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
          if (i !== j) {
            // Move correlations towards 1 (or -1 if negative)
            matrix[i][j] = matrix[i][j] > 0 ?
              Math.min(0.95, matrix[i][j] * 1.5) :
              Math.max(-0.95, matrix[i][j] * 1.5);
          }
        }
      }
    }
  }

  async applyCorrelationShock(portfolio, correlationShock) {
    const shockedPortfolio = this.deepClone(portfolio);

    if (shockedPortfolio.correlationMatrix) {
      const matrix = shockedPortfolio.correlationMatrix;
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
          if (i !== j) {
            // Apply uniform correlation shock
            matrix[i][j] = Math.max(-0.99, Math.min(0.99,
              matrix[i][j] + correlationShock));
          }
        }
      }
    }

    return await this.applyShocksToPortfolio(portfolio, {});
  }

  // Shock Generation Methods
  generateRandomShocks(portfolio) {
    const shocks = {};

    // Generate random equity shock
    shocks.equities = (Math.random() - 0.5) * 0.6; // ±30%

    // Generate random bond shock
    shocks.bonds = (Math.random() - 0.5) * 0.2; // ±10%

    // Generate random commodity shock
    shocks.commodities = (Math.random() - 0.5) * 0.8; // ±40%

    // Generate random volatility shock
    shocks.volatility_shock = 0.5 + Math.random() * 2; // 0.5x to 2.5x

    // Generate random currency shocks
    shocks.currencies = {
      USD: (Math.random() - 0.5) * 0.3,
      EUR: (Math.random() - 0.5) * 0.3,
      JPY: (Math.random() - 0.5) * 0.3
    };

    return shocks;
  }

  generateExtremeShocks(portfolio, shockMagnitude) {
    const baseVolatility = 0.15; // Assume 15% base volatility
    const shockSize = shockMagnitude * baseVolatility;

    return {
      equity_crash: -shockSize,
      bond_crash: -shockSize * 0.5,
      commodity_crash: -shockSize * 1.2,
      currency_crisis: shockSize * 0.8,
      credit_crisis: shockSize * 0.6,
      liquidity_crisis: -shockSize * 0.7,
      volatility_spike: shockMagnitude
    };
  }

  // Analysis and Calculation Methods
  calculatePortfolioValue(portfolio) {
    if (!portfolio.assets) return 0;

    return portfolio.assets.reduce((total, asset) => {
      return total + (asset.price * asset.quantity * (asset.weight || 1));
    }, 0);
  }

  calculateMaxDrawdown(portfolio) {
    // Simplified max drawdown calculation
    // In practice, this would require time series data
    const currentValue = this.calculatePortfolioValue(portfolio);
    const peakValue = portfolio.peakValue || currentValue;
    return (peakValue - currentValue) / peakValue;
  }

  calculateStressedRiskMetrics(portfolio) {
    const value = this.calculatePortfolioValue(portfolio);
    const avgVolatility = this.calculatePortfolioVolatility(portfolio);

    return {
      portfolioValue: value,
      volatility: avgVolatility,
      var95: value * avgVolatility * 1.645, // Simplified VaR calculation
      expectedShortfall: value * avgVolatility * 2.326
    };
  }

  calculatePortfolioVolatility(portfolio) {
    if (!portfolio.assets) return 0;

    // Simplified portfolio volatility calculation
    return portfolio.assets.reduce((sum, asset) => {
      return sum + (asset.volatility || 0.15) * (asset.weight || 1 / portfolio.assets.length);
    }, 0);
  }

  calculateStandardDeviation(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
  }

  // Result Analysis Methods
  findWorstCaseScenario(results) {
    return results.reduce((worst, current) => {
      return (current.pnlPercent < worst.pnlPercent) ? current : worst;
    }, results[0]);
  }

  calculateAverageImpact(results) {
    const totalImpact = results.reduce((sum, result) => sum + result.pnlPercent, 0);
    return totalImpact / results.length;
  }

  groupWorstCaseByShockType(results) {
    const grouped = {};

    results.forEach(result => {
      if (!grouped[result.shockType] || result.pnlPercent < grouped[result.shockType].pnlPercent) {
        grouped[result.shockType] = result;
      }
    });

    return grouped;
  }

  summarizeExtremeValueResults(results) {
    const shockTypes = [...new Set(results.map(r => r.shockType))];

    return shockTypes.map(type => {
      const typeResults = results.filter(r => r.shockType === type);
      const worstCase = Math.min(...typeResults.map(r => r.pnlPercent));
      const averageImpact = typeResults.reduce((sum, r) => sum + r.pnlPercent, 0) / typeResults.length;

      return {
        shockType: type,
        worstCase,
        averageImpact,
        testCount: typeResults.length
      };
    });
  }

  analyzeDiversificationBreakdown(results) {
    return {
      correlationSensitivity: this.calculateCorrelationSensitivity(results),
      diversificationBenefitLoss: this.calculateDiversificationLoss(results),
      concentrationRisk: this.assessConcentrationRisk(results)
    };
  }

  calculateCorrelationSensitivity(results) {
    // Measure how portfolio performance changes with correlation
    const correlationImpacts = results.map(r => ({
      correlation: r.correlationShock,
      impact: r.pnlPercent
    }));

    // Simple linear regression to measure sensitivity
    const n = correlationImpacts.length;
    const sumX = correlationImpacts.reduce((sum, p) => sum + p.correlation, 0);
    const sumY = correlationImpacts.reduce((sum, p) => sum + p.impact, 0);
    const sumXY = correlationImpacts.reduce((sum, p) => sum + p.correlation * p.impact, 0);
    const sumXX = correlationImpacts.reduce((sum, p) => sum + p.correlation * p.correlation, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  calculateDiversificationLoss(results) {
    const baseline = results.find(r => r.correlationShock === 0);
    const worstCase = results.reduce((worst, current) =>
      current.pnlPercent < worst.pnlPercent ? current : worst);

    if (baseline && worstCase) {
      return worstCase.pnlPercent - baseline.pnlPercent;
    }
    return 0;
  }

  assessConcentrationRisk(results) {
    return results.map(result => ({
      correlationLevel: result.correlationShock,
      concentrationRisk: result.concentrationRisk || 'unknown',
      riskLevel: this.categorizeRiskLevel(result.pnlPercent)
    }));
  }

  categorizeRiskLevel(pnlPercent) {
    if (pnlPercent > -5) return 'LOW';
    if (pnlPercent > -15) return 'MEDIUM';
    if (pnlPercent > -30) return 'HIGH';
    return 'EXTREME';
  }

  // Summary and Recommendations
  generateStressSummary(scenarios) {
    const summary = {
      overallRiskLevel: 'MEDIUM',
      worstCaseScenario: null,
      keyVulnerabilities: [],
      resilientAreas: [],
      statisticalSummary: {}
    };

    // Find overall worst case
    let worstCase = { pnlPercent: 0 };

    Object.values(scenarios).forEach(scenarioGroup => {
      if (scenarioGroup.worstCase && scenarioGroup.worstCase.pnlPercent < worstCase.pnlPercent) {
        worstCase = scenarioGroup.worstCase;
      }
    });

    summary.worstCaseScenario = worstCase;

    // Determine overall risk level
    if (worstCase.pnlPercent < -30) summary.overallRiskLevel = 'EXTREME';
    else if (worstCase.pnlPercent < -15) summary.overallRiskLevel = 'HIGH';
    else if (worstCase.pnlPercent < -5) summary.overallRiskLevel = 'MEDIUM';
    else summary.overallRiskLevel = 'LOW';

    // Statistical summary
    if (scenarios.monteCarlo) {
      summary.statisticalSummary = {
        var95: scenarios.monteCarlo.percentiles['VaR_95%'],
        var99: scenarios.monteCarlo.percentiles['VaR_99%'],
        expectedShortfall: scenarios.monteCarlo.percentiles['VaR_99%'] * 1.2,
        averageStressLoss: scenarios.monteCarlo.averagePnL
      };
    }

    return summary;
  }

  generateRecommendations(summary) {
    const recommendations = [];

    // Risk level based recommendations
    if (summary.overallRiskLevel === 'EXTREME' || summary.overallRiskLevel === 'HIGH') {
      recommendations.push({
        priority: 'HIGH',
        category: 'HEDGING',
        action: 'Implement protective hedging strategies immediately',
        timeline: 'Immediate',
        description: 'Portfolio shows extreme vulnerability to stress scenarios'
      });

      recommendations.push({
        priority: 'HIGH',
        category: 'DIVERSIFICATION',
        action: 'Reduce concentration in vulnerable asset classes',
        timeline: '1-2 weeks',
        description: 'Rebalance portfolio to reduce single points of failure'
      });
    }

    // Monte Carlo based recommendations
    if (summary.statisticalSummary && summary.statisticalSummary.var95 < -10) {
      recommendations.push({
        priority: 'MEDIUM',
        category: 'RISK_MANAGEMENT',
        action: 'Review and adjust position sizing',
        timeline: '1 month',
        description: 'High VaR indicates potential for significant losses'
      });
    }

    // General recommendations
    recommendations.push({
      priority: 'LOW',
      category: 'MONITORING',
      action: 'Implement regular stress testing schedule',
      timeline: 'Ongoing',
      description: 'Regular stress testing helps identify emerging risks'
    });

    return recommendations;
  }

  // Utility Methods
  getAssetClass(asset) {
    // Simple asset class mapping
    if (asset.type === 'stock' || asset.type === 'equity') return 'equities';
    if (asset.type === 'bond' || asset.type === 'fixed_income') return 'bonds';
    if (asset.type === 'commodity') return 'commodities';
    if (asset.type === 'currency' || asset.type === 'fx') return 'currencies';
    return 'other';
  }

  getCorrelationShockDescription(shock) {
    if (shock === -0.5) return 'Strong negative correlation shock';
    if (shock === 0) return 'No correlation change (baseline)';
    if (shock === 0.5) return 'Moderate positive correlation shock';
    if (shock === 1.0) return 'Extreme positive correlation shock';
    return `Correlation shock: ${shock}`;
  }

  identifyAffectedAssets(originalPortfolio, shockedPortfolio) {
    const affected = [];

    if (originalPortfolio.assets && shockedPortfolio.assets) {
      for (let i = 0; i < originalPortfolio.assets.length; i++) {
        const original = originalPortfolio.assets[i];
        const shocked = shockedPortfolio.assets[i];

        if (shocked.shockApplied && Math.abs(shocked.shockApplied) > 0.01) {
          affected.push({
            asset: original.name || `Asset_${i}`,
            originalPrice: original.price,
            shockedPrice: shocked.price,
            shockApplied: shocked.shockApplied,
            impact: ((shocked.price - original.price) / original.price) * 100
          });
        }
      }
    }

    return affected;
  }

  deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  // Export and Reporting
  exportStressTestResults(testId, format = 'json') {
    const results = this.results.get(testId);
    if (!results) {
      throw new Error(`Stress test results not found for ID: ${testId}`);
    }

    switch (format.toLowerCase()) {
    case 'json':
      return JSON.stringify(results, null, 2);
    case 'csv':
      return this.convertToCSV(results);
    case 'summary':
      return this.generateExecutiveSummary(results);
    default:
      throw new Error(`Unsupported export format: ${format}`);
    }
  }

  convertToCSV(results) {
    // Simplified CSV conversion for scenario results
    let csv = 'Scenario,Description,PnL%,Portfolio Value,Max Drawdown\n';

    if (results.scenarios.historical) {
      results.scenarios.historical.scenarios.forEach(scenario => {
        csv += `${scenario.scenario},${scenario.description},${scenario.pnlPercent.toFixed(2)},${scenario.shockedValue.toFixed(2)},${scenario.maxDrawdown.toFixed(2)}\n`;
      });
    }

    return csv;
  }

  generateExecutiveSummary(results) {
    return `
STRESS TESTING EXECUTIVE SUMMARY
================================

Test Date: ${results.timestamp.toISOString()}
Overall Risk Level: ${results.summary.overallRiskLevel}

WORST CASE SCENARIO:
- Scenario: ${results.summary.worstCaseScenario?.scenario || 'N/A'}
- Impact: ${results.summary.worstCaseScenario?.pnlPercent?.toFixed(2) || 'N/A'}%

KEY RECOMMENDATIONS:
${results.recommendations.map(rec => `- ${rec.action} (${rec.priority} priority)`).join('\n')}

STATISTICAL SUMMARY:
- VaR (95%): ${results.summary.statisticalSummary?.var95?.toFixed(2) || 'N/A'}%
- VaR (99%): ${results.summary.statisticalSummary?.var99?.toFixed(2) || 'N/A'}%
- Expected Shortfall: ${results.summary.statisticalSummary?.expectedShortfall?.toFixed(2) || 'N/A'}%
`;
  }

  // Cleanup
  clearResults() {
    this.results.clear();
  }

  clearScenarios() {
    this.historicalScenarios.clear();
    this.customScenarios.clear();
  }
}

module.exports = StressTestingFramework;