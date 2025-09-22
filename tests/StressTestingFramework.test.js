/**
 * Stress Testing Framework Tests
 */

const StressTestingFramework = require('../src/stress-testing/StressTestingFramework');

describe('StressTestingFramework', () => {
  let stressTest;
  let samplePortfolio;

  beforeEach(() => {
    stressTest = new StressTestingFramework({
      monteCarloIterations: 100, // Reduced for faster testing
      confidenceLevels: [0.90, 0.95],
      shockMagnitudes: [2, 3]
    });

    samplePortfolio = {
      name: 'Test Portfolio',
      totalValue: 100000,
      assets: [
        {
          name: 'US Equity',
          type: 'equity',
          price: 100,
          quantity: 300,
          weight: 0.3,
          volatility: 0.15,
          currency: 'USD'
        },
        {
          name: 'EU Bond',
          type: 'bond',
          price: 95,
          quantity: 500,
          weight: 0.475,
          volatility: 0.05,
          duration: 5,
          currency: 'EUR'
        },
        {
          name: 'Gold',
          type: 'commodity',
          price: 1800,
          quantity: 12.5,
          weight: 0.225,
          volatility: 0.20,
          currency: 'USD'
        }
      ],
      correlationMatrix: [
        [1.0, -0.2, 0.1],
        [-0.2, 1.0, 0.05],
        [0.1, 0.05, 1.0]
      ],
      peakValue: 100000
    };
  });

  describe('Historical Scenario Testing', () => {
    test('should initialize historical scenarios correctly', () => {
      stressTest.initializeHistoricalScenarios();
      expect(stressTest.historicalScenarios.size).toBeGreaterThan(0);

      const crisis2008 = stressTest.historicalScenarios.get('2008_crisis');
      expect(crisis2008).toBeDefined();
      expect(crisis2008.shocks).toHaveProperty('equities');
      expect(crisis2008.shocks.equities).toBeLessThan(0);
    });

    test('should run historical scenarios', async () => {
      const results = await stressTest.runHistoricalScenarios(samplePortfolio);

      expect(results).toHaveProperty('count');
      expect(results).toHaveProperty('scenarios');
      expect(results).toHaveProperty('worstCase');
      expect(results).toHaveProperty('averageImpact');

      expect(results.count).toBeGreaterThan(0);
      expect(results.scenarios).toHaveLength(results.count);
      expect(typeof results.averageImpact).toBe('number');
    });

    test('should identify worst case scenario', async () => {
      const results = await stressTest.runHistoricalScenarios(samplePortfolio);
      const worstCase = results.worstCase;

      expect(worstCase).toHaveProperty('pnlPercent');
      expect(worstCase.pnlPercent).toBeLessThanOrEqual(0);

      // Verify it's actually the worst
      const allImpacts = results.scenarios.map(s => s.pnlPercent);
      const minImpact = Math.min(...allImpacts);
      expect(worstCase.pnlPercent).toBe(minImpact);
    });
  });

  describe('Monte Carlo Stress Testing', () => {
    test('should run Monte Carlo simulations', async () => {
      const results = await stressTest.runMonteCarloStressTest(samplePortfolio);

      expect(results).toHaveProperty('iterations');
      expect(results).toHaveProperty('percentiles');
      expect(results).toHaveProperty('averagePnL');
      expect(results).toHaveProperty('standardDeviation');

      expect(results.iterations).toBe(100);
      expect(results.percentiles).toHaveProperty('VaR_90%');
      expect(results.percentiles).toHaveProperty('VaR_95%');
      expect(typeof results.averagePnL).toBe('number');
      expect(results.standardDeviation).toBeGreaterThan(0);
    });

    test('should calculate VaR percentiles correctly', async () => {
      const results = await stressTest.runMonteCarloStressTest(samplePortfolio);
      const var90 = results.percentiles['VaR_90%'];
      const var95 = results.percentiles['VaR_95%'];

      expect(var95).toBeLessThanOrEqual(var90); // 95% VaR should be more negative
      expect(typeof var90).toBe('number');
      expect(typeof var95).toBe('number');
    });
  });

  describe('Extreme Value Testing', () => {
    test('should run extreme value tests', async () => {
      const results = await stressTest.runExtremeValueTest(samplePortfolio);

      expect(results).toHaveProperty('count');
      expect(results).toHaveProperty('results');
      expect(results).toHaveProperty('worstCaseByType');
      expect(results).toHaveProperty('summary');

      expect(results.count).toBeGreaterThan(0);
      expect(Array.isArray(results.results)).toBe(true);
      expect(Array.isArray(results.summary)).toBe(true);
    });

    test('should test different shock magnitudes', async () => {
      const results = await stressTest.runExtremeValueTest(samplePortfolio);
      const shockMagnitudes = [...new Set(results.results.map(r => r.shockMagnitude))];

      expect(shockMagnitudes.length).toBeGreaterThan(1);
      expect(shockMagnitudes).toContain(2);
      expect(shockMagnitudes).toContain(3);
    });
  });

  describe('Correlation Breakdown Testing', () => {
    test('should test correlation breakdown scenarios', async () => {
      const results = await stressTest.runCorrelationBreakdownTest(samplePortfolio);

      expect(results).toHaveProperty('count');
      expect(results).toHaveProperty('results');
      expect(results).toHaveProperty('diversificationAnalysis');

      expect(results.count).toBeGreaterThan(0);
      expect(results.diversificationAnalysis).toHaveProperty('correlationSensitivity');
    });

    test('should measure correlation sensitivity', async () => {
      const results = await stressTest.runCorrelationBreakdownTest(samplePortfolio);
      const sensitivity = results.diversificationAnalysis.correlationSensitivity;

      expect(typeof sensitivity).toBe('number');
      expect(isFinite(sensitivity)).toBe(true);
    });
  });

  describe('Comprehensive Stress Testing', () => {
    test('should run comprehensive stress test', async () => {
      const results = await stressTest.runComprehensiveStressTest(samplePortfolio, {
        includeHistorical: true,
        includeCustom: false,
        includeMonteCarlo: true,
        includeExtremeValue: true,
        includeCorrelationBreakdown: true
      });

      expect(results).toHaveProperty('portfolio');
      expect(results).toHaveProperty('scenarios');
      expect(results).toHaveProperty('summary');
      expect(results).toHaveProperty('recommendations');

      expect(results.scenarios).toHaveProperty('historical');
      expect(results.scenarios).toHaveProperty('monteCarlo');
      expect(results.scenarios).toHaveProperty('extremeValue');
      expect(results.scenarios).toHaveProperty('correlationBreakdown');

      expect(Array.isArray(results.recommendations)).toBe(true);
    });

    test('should generate appropriate recommendations', async () => {
      const results = await stressTest.runComprehensiveStressTest(samplePortfolio);
      const recommendations = results.recommendations;

      expect(recommendations.length).toBeGreaterThan(0);
      recommendations.forEach(rec => {
        expect(rec).toHaveProperty('priority');
        expect(rec).toHaveProperty('category');
        expect(rec).toHaveProperty('action');
        expect(rec).toHaveProperty('timeline');
        expect(['HIGH', 'MEDIUM', 'LOW']).toContain(rec.priority);
      });
    });
  });

  describe('Custom Scenarios', () => {
    test('should add and run custom scenarios', async () => {
      stressTest.addCustomScenario('test_scenario', 'Test Scenario', {
        equities: -0.3,
        bonds: -0.1,
        commodities: 0.1
      });

      expect(stressTest.customScenarios.size).toBe(1);

      const results = await stressTest.runCustomScenarios(samplePortfolio);
      expect(results.count).toBe(1);
      expect(results.scenarios[0].scenario).toBe('test_scenario');
    });

    test('should handle multiple custom scenarios', async () => {
      stressTest.addCustomScenario('scenario1', 'Scenario 1', { equities: -0.2 });
      stressTest.addCustomScenario('scenario2', 'Scenario 2', { bonds: -0.15 });

      const results = await stressTest.runCustomScenarios(samplePortfolio);
      expect(results.count).toBe(2);
    });
  });

  describe('Shock Application', () => {
    test('should apply asset class shocks correctly', async () => {
      const shocks = {
        equities: -0.2,
        bonds: -0.1,
        commodities: 0.05
      };

      const result = await stressTest.applyShocksToPortfolio(samplePortfolio, shocks);

      expect(result).toHaveProperty('originalValue');
      expect(result).toHaveProperty('shockedValue');
      expect(result).toHaveProperty('pnl');
      expect(result).toHaveProperty('pnlPercent');

      expect(result.originalValue).toBeGreaterThan(0);
      expect(result.pnl).toBeLessThan(0); // Should be negative due to equity shock
      expect(result.pnlPercent).toBeLessThan(0);
    });

    test('should handle volatility shocks', async () => {
      const shocks = {
        volatility_shock: 2.0
      };

      const result = await stressTest.applyShocksToPortfolio(samplePortfolio, shocks);
      expect(result).toHaveProperty('riskMetrics');
      expect(result.riskMetrics.volatility).toBeGreaterThan(0);
    });

    test('should apply correlation breakdown', async () => {
      const shocks = {
        correlation_breakdown: true
      };

      await stressTest.applyShocksToPortfolio(samplePortfolio, shocks);
      // Test should complete without errors
      expect(true).toBe(true);
    });
  });

  describe('Result Export and Analysis', () => {
    test('should export results in different formats', async () => {
      const results = await stressTest.runComprehensiveStressTest(samplePortfolio);
      const testId = Array.from(stressTest.results.keys())[0];

      const jsonExport = stressTest.exportStressTestResults(testId, 'json');
      expect(typeof jsonExport).toBe('string');
      expect(() => JSON.parse(jsonExport)).not.toThrow();

      const csvExport = stressTest.exportStressTestResults(testId, 'csv');
      expect(typeof csvExport).toBe('string');
      expect(csvExport).toContain('Scenario,Description');

      const summaryExport = stressTest.exportStressTestResults(testId, 'summary');
      expect(typeof summaryExport).toBe('string');
      expect(summaryExport).toContain('STRESS TESTING EXECUTIVE SUMMARY');
    });

    test('should handle invalid export requests', () => {
      expect(() => {
        stressTest.exportStressTestResults('invalid_id', 'json');
      }).toThrow();

      const results = stressTest.runComprehensiveStressTest(samplePortfolio);
      const testId = Array.from(stressTest.results.keys())[0];

      expect(() => {
        stressTest.exportStressTestResults(testId, 'invalid_format');
      }).toThrow();
    });
  });

  describe('Edge Cases and Error Handling', () => {
    test('should handle empty portfolio', async () => {
      const emptyPortfolio = {
        name: 'Empty Portfolio',
        totalValue: 0,
        assets: [],
        correlationMatrix: [],
        peakValue: 0
      };

      const result = await stressTest.applyShocksToPortfolio(emptyPortfolio, { equities: -0.2 });
      expect(result.originalValue).toBe(0);
      expect(result.pnl).toBe(0);
    });

    test('should handle portfolio with single asset', async () => {
      const singleAssetPortfolio = {
        name: 'Single Asset Portfolio',
        totalValue: 10000,
        assets: [samplePortfolio.assets[0]],
        correlationMatrix: [[1.0]],
        peakValue: 10000
      };

      const result = await stressTest.applyShocksToPortfolio(singleAssetPortfolio, { equities: -0.1 });
      expect(result.originalValue).toBeGreaterThan(0);
      expect(result.pnl).toBeLessThan(0);
    });

    test('should handle extreme shock values', async () => {
      const extremeShocks = {
        equities: -0.99, // 99% loss
        volatility_shock: 10.0
      };

      const result = await stressTest.applyShocksToPortfolio(samplePortfolio, extremeShocks);
      expect(result.pnlPercent).toBeLessThan(-50); // Should be a very large loss
      expect(isFinite(result.pnlPercent)).toBe(true);
    });
  });

  describe('Utility Functions', () => {
    test('should classify risk levels correctly', () => {
      expect(stressTest.categorizeRiskLevel(-2)).toBe('LOW');
      expect(stressTest.categorizeRiskLevel(-8)).toBe('MEDIUM');
      expect(stressTest.categorizeRiskLevel(-20)).toBe('HIGH');
      expect(stressTest.categorizeRiskLevel(-35)).toBe('EXTREME');
    });

    test('should identify asset classes correctly', () => {
      expect(stressTest.getAssetClass({ type: 'equity' })).toBe('equities');
      expect(stressTest.getAssetClass({ type: 'bond' })).toBe('bonds');
      expect(stressTest.getAssetClass({ type: 'commodity' })).toBe('commodities');
      expect(stressTest.getAssetClass({ type: 'unknown' })).toBe('other');
    });

    test('should calculate portfolio value correctly', () => {
      const value = stressTest.calculatePortfolioValue(samplePortfolio);
      const expectedValue = samplePortfolio.assets.reduce((sum, asset) =>
        sum + (asset.price * asset.quantity * asset.weight), 0
      );
      expect(value).toBeCloseTo(expectedValue, 2);
    });
  });
});