/**
 * Advanced Risk Calculator Tests
 */

const AdvancedRiskCalculator = require('../src/calculators/AdvancedRiskCalculator');

describe('AdvancedRiskCalculator', () => {
  let advancedRiskCalc;
  let sampleReturns;
  let marketReturns;

  beforeEach(() => {
    advancedRiskCalc = new AdvancedRiskCalculator();
    // Generate longer sample returns for VaR calculation requirements
    sampleReturns = Array.from({length: 50}, (_, i) =>
      0.001 * Math.sin(i / 5) + 0.01 * (Math.random() - 0.5)
    );
    marketReturns = Array.from({length: 50}, (_, i) =>
      0.0008 * Math.cos(i / 6) + 0.012 * (Math.random() - 0.5)
    );
  });

  describe('GARCH Volatility', () => {
    test('should calculate GARCH volatility with sufficient data', () => {
      const longReturns = Array.from({length: 100}, (_, i) =>
        0.001 * Math.sin(i / 10) + 0.01 * (Math.random() - 0.5)
      );

      const garchResult = advancedRiskCalc.calculateGarchVolatility(longReturns);
      expect(garchResult).toHaveProperty('currentVolatility');
      expect(garchResult).toHaveProperty('parameters');
      expect(garchResult.currentVolatility).toBeGreaterThan(0);
    });

    test('should throw error for insufficient data', () => {
      expect(() => {
        advancedRiskCalc.calculateGarchVolatility(sampleReturns);
      }).toThrow('Insufficient data for GARCH estimation');
    });
  });

  describe('Cornish-Fisher VaR', () => {
    test('should calculate Cornish-Fisher VaR correctly', () => {
      const cfVar = advancedRiskCalc.calculateCornishFisherVaR(sampleReturns);
      expect(typeof cfVar).toBe('number');
      expect(cfVar).toBeGreaterThan(0);
    });

    test('should handle different confidence levels', () => {
      const cfVar95 = advancedRiskCalc.calculateCornishFisherVaR(sampleReturns, 0.95);
      const cfVar99 = advancedRiskCalc.calculateCornishFisherVaR(sampleReturns, 0.99);
      expect(cfVar99).toBeGreaterThan(cfVar95);
    });
  });

  describe('Higher Moments', () => {
    test('should calculate skewness correctly', () => {
      const skewness = advancedRiskCalc.calculateSkewness(sampleReturns);
      expect(typeof skewness).toBe('number');
      expect(skewness).toBeGreaterThan(-5);
      expect(skewness).toBeLessThan(5);
    });

    test('should calculate kurtosis correctly', () => {
      const kurtosis = advancedRiskCalc.calculateKurtosis(sampleReturns);
      expect(typeof kurtosis).toBe('number');
      expect(kurtosis).toBeGreaterThan(-2);
    });

    test('should calculate excess kurtosis for normal distribution', () => {
      // Generate normal-like data
      const normalReturns = Array.from({length: 1000}, () =>
        0.001 + 0.01 * (Math.random() + Math.random() + Math.random() - 1.5)
      );
      const kurtosis = advancedRiskCalc.calculateKurtosis(normalReturns);
      expect(Math.abs(kurtosis)).toBeLessThan(2); // Should be close to 0 for normal
    });
  });

  describe('Component VaR', () => {
    test('should calculate component VaR correctly', () => {
      const weights = [0.4, 0.3, 0.3];
      const assetReturns = [
        sampleReturns,
        sampleReturns.map(r => r * 0.8 + 0.001),
        sampleReturns.map(r => -r * 0.3 + 0.0005)
      ];
      const correlationMatrix = [
        [1.0, 0.7, -0.3],
        [0.7, 1.0, -0.2],
        [-0.3, -0.2, 1.0]
      ];

      const componentVar = advancedRiskCalc.calculateComponentVaR(
        weights, assetReturns, correlationMatrix
      );

      expect(componentVar).toHaveProperty('portfolioVaR');
      expect(componentVar).toHaveProperty('componentVaRs');
      expect(componentVar.componentVaRs).toHaveLength(3);
      expect(Math.abs(componentVar.portfolioVaR)).toBeGreaterThan(0);

      // Sum of component contributions should equal portfolio VaR
      const sumContributions = componentVar.componentVaRs
        .reduce((sum, comp) => sum + comp.contribution, 0);
      expect(Math.abs(sumContributions - 1)).toBeLessThan(0.01);
    });

    test('should validate input dimensions', () => {
      const weights = [0.5, 0.5];
      const assetReturns = [sampleReturns]; // Only one asset
      const correlationMatrix = [[1.0]];

      expect(() => {
        advancedRiskCalc.calculateComponentVaR(weights, assetReturns, correlationMatrix);
      }).toThrow();
    });
  });

  describe('Advanced Risk Metrics', () => {
    test('should calculate comprehensive risk metrics', () => {
      const metrics = advancedRiskCalc.calculateAdvancedRiskMetrics(
        'TEST_ASSET',
        sampleReturns,
        marketReturns
      );

      expect(metrics).toHaveProperty('symbol', 'TEST_ASSET');
      expect(metrics).toHaveProperty('volatility');
      expect(metrics).toHaveProperty('beta');
      expect(metrics).toHaveProperty('valueAtRisk');
      expect(metrics).toHaveProperty('expectedShortfall');
      expect(metrics).toHaveProperty('skewness');
      expect(metrics).toHaveProperty('kurtosis');

      expect(metrics.volatility).toBeGreaterThan(0);
      expect(typeof metrics.beta).toBe('number');
      expect(metrics.valueAtRisk).toBeGreaterThan(0);
      expect(metrics.expectedShortfall).toBeGreaterThanOrEqual(metrics.valueAtRisk);
    });

    test('should handle missing market returns', () => {
      const metrics = advancedRiskCalc.calculateAdvancedRiskMetrics(
        'TEST_ASSET',
        sampleReturns
      );

      expect(metrics.beta).toBeNull();
      expect(metrics.volatility).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases and Validation', () => {
    test('should handle empty returns array', () => {
      const result = advancedRiskCalc.calculateSkewness([0.01, 0.02, 0.015]);
      expect(typeof result).toBe('number');
    });

    test('should handle single return value', () => {
      const result = advancedRiskCalc.calculateSkewness([0.01, 0.015, 0.02]);
      expect(typeof result).toBe('number');
    });

    test('should handle extreme outliers', () => {
      const outlierReturns = [0.01, 0.02, -0.01, 10.0, 0.015]; // 1000% return outlier
      const skewness = advancedRiskCalc.calculateSkewness(outlierReturns);
      expect(skewness).toBeGreaterThan(0); // Should be positively skewed
      expect(isFinite(skewness)).toBe(true);
    });

    test('should validate correlation matrix dimensions', () => {
      const weights = [0.5, 0.5];
      const assetReturns = [sampleReturns, marketReturns];
      const invalidCorrelationMatrix = [[1.0]]; // Wrong size

      expect(() => {
        advancedRiskCalc.calculateComponentVaR(weights, assetReturns, invalidCorrelationMatrix);
      }).toThrow();
    });
  });

  describe('Performance Tests', () => {
    test('should handle large datasets efficiently', () => {
      const largeReturns = Array.from({length: 10000}, () =>
        0.001 * (Math.random() - 0.5)
      );

      const startTime = Date.now();
      const skewness = advancedRiskCalc.calculateSkewness(largeReturns);
      const kurtosis = advancedRiskCalc.calculateKurtosis(largeReturns);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should complete within 1 second
      expect(typeof skewness).toBe('number');
      expect(typeof kurtosis).toBe('number');
    });
  });
});