/**
 * Portfolio Optimizer Tests
 */

const PortfolioOptimizer = require('../src/optimization/PortfolioOptimizer');

describe('PortfolioOptimizer', () => {
  let optimizer;
  let sampleData;

  beforeEach(() => {
    optimizer = new PortfolioOptimizer();

    // Create sample data
    sampleData = {
      returns: [
        [0.01, 0.02, -0.01, 0.03, 0.005], // Asset 1
        [0.015, -0.01, 0.025, 0.01, -0.005], // Asset 2
        [-0.005, 0.03, 0.01, -0.015, 0.02] // Asset 3
      ],
      covariance: [
        [0.0004, 0.0002, -0.0001],
        [0.0002, 0.0006, 0.0001],
        [-0.0001, 0.0001, 0.0005]
      ],
      expectedReturns: [0.008, 0.012, 0.006]
    };
  });

  describe('Mean-Variance Optimization', () => {
    test('should optimize portfolio using mean-variance approach', async () => {
      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        targetReturn: 0.01
      });

      expect(result).toHaveProperty('weights');
      expect(result).toHaveProperty('expectedReturn');
      expect(result).toHaveProperty('volatility');
      expect(result).toHaveProperty('sharpeRatio');

      expect(result.weights).toHaveLength(3);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);
      expect(result.weights.every(w => w >= -0.01 && w <= 1.01)).toBe(true);
    });

    test('should handle minimum variance optimization', async () => {
      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'minimum-variance'
      });

      expect(result.weights).toHaveLength(3);
      expect(result.volatility).toBeGreaterThan(0);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);
    });

    test('should handle maximum Sharpe ratio optimization', async () => {
      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'max-sharpe',
        riskFreeRate: 0.02
      });

      expect(result).toHaveProperty('sharpeRatio');
      expect(result.sharpeRatio).toBeGreaterThan(-5);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);
    });
  });

  describe('Risk Parity Optimization', () => {
    test('should calculate risk parity weights', async () => {
      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'risk-parity'
      });

      expect(result.weights).toHaveLength(3);
      expect(result.weights.every(w => w > 0)).toBe(true);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);

      // Risk contributions should be approximately equal
      const riskContributions = result.riskContributions || [];
      if (riskContributions.length > 0) {
        const avgContribution = riskContributions.reduce((sum, c) => sum + c, 0) / riskContributions.length;
        const maxDeviation = Math.max(...riskContributions.map(c => Math.abs(c - avgContribution)));
        expect(maxDeviation).toBeLessThan(0.1); // Within 10% of average
      }
    });
  });

  describe('Black-Litterman Optimization', () => {
    test('should perform Black-Litterman optimization with views', async () => {
      const views = [
        { asset: 0, expectedReturn: 0.015, confidence: 0.5 },
        { asset: 1, expectedReturn: 0.008, confidence: 0.3 }
      ];

      const result = await optimizer.blackLittermanOptimization(sampleData, {
        views,
        tau: 0.025,
        riskAversion: 3
      });

      expect(result).toHaveProperty('weights');
      expect(result).toHaveProperty('adjustedReturns');
      expect(result).toHaveProperty('posteriorCovariance');

      expect(result.weights).toHaveLength(3);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);
      expect(result.adjustedReturns).toHaveLength(3);
    });

    test('should handle empty views', async () => {
      const result = await optimizer.blackLittermanOptimization(sampleData, {
        views: [],
        tau: 0.025,
        riskAversion: 3
      });

      expect(result.weights).toHaveLength(3);
      expect(result.adjustedReturns).toEqual(sampleData.expectedReturns);
    });
  });

  describe('Multi-Objective Optimization', () => {
    test('should perform multi-objective optimization', async () => {
      const objectives = [
        { type: 'return', maximize: true, weight: 0.6 },
        { type: 'volatility', minimize: true, weight: 0.4 }
      ];

      const result = await optimizer.multiObjectiveOptimization(sampleData, {
        objectives,
        populationSize: 50,
        generations: 20
      });

      expect(result).toHaveProperty('weights');
      expect(result).toHaveProperty('objectives');
      expect(result).toHaveProperty('paretoFront');

      expect(result.weights).toHaveLength(3);
      expect(Math.abs(result.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);
    });

    test('should validate objective definitions', async () => {
      const invalidObjectives = [
        { type: 'invalid_type', maximize: true, weight: 1.0 }
      ];

      await expect(optimizer.multiObjectiveOptimization(sampleData, {
        objectives: invalidObjectives
      })).rejects.toThrow();
    });
  });

  describe('Portfolio Performance Analysis', () => {
    test('should calculate portfolio performance metrics', () => {
      const weights = [0.4, 0.3, 0.3];
      const performance = optimizer.calculatePortfolioPerformance(weights, sampleData);

      expect(performance).toHaveProperty('expectedReturn');
      expect(performance).toHaveProperty('volatility');
      expect(performance).toHaveProperty('sharpeRatio');
      expect(performance).toHaveProperty('beta');

      expect(performance.expectedReturn).toBeGreaterThan(0);
      expect(performance.volatility).toBeGreaterThan(0);
      expect(typeof performance.sharpeRatio).toBe('number');
    });

    test('should handle edge case weights', () => {
      const extremeWeights = [1.0, 0.0, 0.0]; // All in first asset
      const performance = optimizer.calculatePortfolioPerformance(extremeWeights, sampleData);

      expect(performance.expectedReturn).toBeCloseTo(sampleData.expectedReturns[0], 3);
      expect(performance.volatility).toBeCloseTo(Math.sqrt(sampleData.covariance[0][0]), 3);
    });
  });

  describe('Constraints Handling', () => {
    test('should respect weight constraints', async () => {
      const constraints = {
        minWeight: 0.1,
        maxWeight: 0.6,
        longOnly: true
      };

      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        targetReturn: 0.01,
        constraints
      });

      expect(result.weights.every(w => w >= 0.09 && w <= 0.61)).toBe(true);
      expect(result.weights.every(w => w >= 0)).toBe(true);
    });

    test('should handle sector constraints', async () => {
      const constraints = {
        sectorLimits: {
          'tech': { assets: [0, 1], maxWeight: 0.7 },
          'finance': { assets: [2], maxWeight: 0.4 }
        }
      };

      const result = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        targetReturn: 0.01,
        constraints
      });

      expect(result.weights[0] + result.weights[1]).toBeLessThanOrEqual(0.71);
      expect(result.weights[2]).toBeLessThanOrEqual(0.41);
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid input data', async () => {
      const invalidData = {
        returns: [[0.01]], // Single asset, single return
        covariance: [[0.0004]],
        expectedReturns: [0.008]
      };

      await expect(optimizer.optimizePortfolio(invalidData, {
        method: 'mean-variance'
      })).rejects.toThrow();
    });

    test('should handle mismatched dimensions', async () => {
      const mismatchedData = {
        returns: sampleData.returns,
        covariance: [[0.0004, 0.0002], [0.0002, 0.0006]], // 2x2 instead of 3x3
        expectedReturns: sampleData.expectedReturns
      };

      await expect(optimizer.optimizePortfolio(mismatchedData, {
        method: 'mean-variance'
      })).rejects.toThrow();
    });

    test('should handle singular covariance matrix', async () => {
      const singularData = {
        returns: sampleData.returns,
        covariance: [
          [0.0004, 0.0004, 0.0004], // Singular matrix (identical columns)
          [0.0004, 0.0004, 0.0004],
          [0.0004, 0.0004, 0.0004]
        ],
        expectedReturns: sampleData.expectedReturns
      };

      await expect(optimizer.optimizePortfolio(singularData, {
        method: 'mean-variance'
      })).rejects.toThrow();
    });
  });

  describe('Optimization Parameters', () => {
    test('should handle different risk aversion levels', async () => {
      const lowRiskAversion = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        riskAversion: 1
      });

      const highRiskAversion = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        riskAversion: 10
      });

      // Higher risk aversion should result in lower portfolio volatility
      expect(highRiskAversion.volatility).toBeLessThanOrEqual(lowRiskAversion.volatility);
    });

    test('should handle different target returns', async () => {
      const lowTarget = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        targetReturn: 0.005
      });

      const highTarget = await optimizer.optimizePortfolio(sampleData, {
        method: 'mean-variance',
        targetReturn: 0.015
      });

      // Higher target return should result in higher volatility
      expect(highTarget.volatility).toBeGreaterThanOrEqual(lowTarget.volatility);
    });
  });
});