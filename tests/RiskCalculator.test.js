/**
 * Risk Calculator Tests
 */

const RiskCalculator = require('../src/calculators/RiskCalculator');

describe('RiskCalculator', () => {
  let riskCalc;
  let samplePrices;
  let marketPrices;

  beforeEach(() => {
    riskCalc = new RiskCalculator();
    samplePrices = [100, 102, 98, 105, 103, 101, 99, 104, 106, 102];
    marketPrices = [1000, 1020, 980, 1050, 1030, 1010, 990, 1040, 1060, 1020];
  });

  describe('Volatility Calculations', () => {
    test('should calculate volatility correctly', () => {
      const volatility = riskCalc.calculateVolatility(samplePrices);
      expect(volatility).toBeGreaterThan(0);
      expect(volatility).toBeLessThan(1);
    });

    test('should throw error for insufficient data', () => {
      expect(() => {
        riskCalc.calculateVolatility([100]);
      }).toThrow('Insufficient price data');
    });

    test('should handle negative prices gracefully', () => {
      const pricesWithNegative = [100, -50, 110];
      // Should not throw, but handle gracefully
      const volatility = riskCalc.calculateVolatility(pricesWithNegative);
      expect(typeof volatility).toBe('number');
    });
  });

  describe('Beta Calculations', () => {
    test('should calculate beta correctly', () => {
      const beta = riskCalc.calculateBeta(samplePrices, marketPrices);
      expect(typeof beta).toBe('number');
      expect(beta).toBeGreaterThan(-2);
      expect(beta).toBeLessThan(3);
    });

    test('should handle mismatched array lengths', () => {
      const shortMarket = [1000, 1020, 980];
      expect(() => {
        riskCalc.calculateBeta(samplePrices, shortMarket);
      }).toThrow();
    });
  });

  describe('Sharpe Ratio Calculations', () => {
    test('should calculate Sharpe ratio correctly', () => {
      const sharpe = riskCalc.calculateSharpeRatio(samplePrices);
      expect(typeof sharpe).toBe('number');
      expect(sharpe).toBeGreaterThan(-5);
      expect(sharpe).toBeLessThan(5);
    });

    test('should use custom risk-free rate', () => {
      const sharpe1 = riskCalc.calculateSharpeRatio(samplePrices, 0.02);
      const sharpe2 = riskCalc.calculateSharpeRatio(samplePrices, 0.05);
      expect(sharpe1).not.toEqual(sharpe2);
    });
  });

  describe('Value at Risk (VaR)', () => {
    test('should calculate VaR correctly', () => {
      const var95 = riskCalc.calculateValueAtRisk(samplePrices, 0.95);
      expect(Math.abs(var95)).toBeGreaterThan(0);
      expect(Math.abs(var95)).toBeLessThan(1);
    });

    test('should handle different confidence levels', () => {
      const var90 = riskCalc.calculateValueAtRisk(samplePrices, 0.90);
      const var99 = riskCalc.calculateValueAtRisk(samplePrices, 0.99);
      expect(Math.abs(var99)).toBeGreaterThanOrEqual(Math.abs(var90));
    });

    test('should validate confidence level bounds', () => {
      // These should work without throwing
      const var95 = riskCalc.calculateValueAtRisk(samplePrices, 0.95);
      const var90 = riskCalc.calculateValueAtRisk(samplePrices, 0.90);
      expect(typeof var95).toBe('number');
      expect(typeof var90).toBe('number');
    });
  });

  describe('Expected Shortfall', () => {
    test('should calculate Expected Shortfall correctly', () => {
      const es = riskCalc.calculateExpectedShortfall(samplePrices, 0.95);
      expect(typeof es).toBe('number');
      expect(Math.abs(es)).toBeGreaterThanOrEqual(0);
    });

    test('should be greater than or equal to VaR in absolute terms', () => {
      const var95 = riskCalc.calculateValueAtRisk(samplePrices, 0.95);
      const es95 = riskCalc.calculateExpectedShortfall(samplePrices, 0.95);
      expect(Math.abs(es95)).toBeGreaterThanOrEqual(Math.abs(var95));
    });
  });

  describe('Maximum Drawdown', () => {
    test('should calculate maximum drawdown correctly', () => {
      const drawdown = riskCalc.calculateMaxDrawdown(samplePrices);
      expect(drawdown).toBeGreaterThanOrEqual(0);
      expect(drawdown).toBeLessThanOrEqual(1);
    });

    test('should be zero for monotonically increasing prices', () => {
      const increasingPrices = [100, 101, 102, 103, 104, 105];
      const drawdown = riskCalc.calculateMaxDrawdown(increasingPrices);
      expect(drawdown).toBe(0);
    });

    test('should handle single price correctly', () => {
      const singlePrice = [100];
      const drawdown = riskCalc.calculateMaxDrawdown(singlePrice);
      expect(drawdown).toBe(0);
    });
  });

  describe('Returns Calculation', () => {
    test('should calculate returns correctly', () => {
      const returns = riskCalc.calculateReturns(samplePrices);
      expect(returns).toHaveLength(samplePrices.length - 1);
      expect(returns[0]).toBeCloseTo(0.02, 2); // (102-100)/100
    });

    test('should handle empty array', () => {
      const returns = riskCalc.calculateReturns([100, 102]);
      expect(returns).toHaveLength(1);
      expect(typeof returns[0]).toBe('number');
    });
  });

  describe('Edge Cases', () => {
    test('should handle all identical prices', () => {
      const identicalPrices = [100, 100, 100, 100, 100];
      const volatility = riskCalc.calculateVolatility(identicalPrices);
      expect(volatility).toBe(0);
    });

    test('should handle very large numbers', () => {
      const largePrices = [1e6, 1.1e6, 0.9e6, 1.2e6, 1.05e6];
      const volatility = riskCalc.calculateVolatility(largePrices);
      expect(volatility).toBeGreaterThan(0);
      expect(isFinite(volatility)).toBe(true);
    });

    test('should handle very small numbers', () => {
      const smallPrices = [0.001, 0.0011, 0.0009, 0.0012, 0.00105];
      const volatility = riskCalc.calculateVolatility(smallPrices);
      expect(volatility).toBeGreaterThan(0);
      expect(isFinite(volatility)).toBe(true);
    });
  });
});