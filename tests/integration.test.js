/**
 * Integration Tests
 *
 * Tests that verify the interaction between different components
 * of the risk calculation platform.
 */

const RiskCalculator = require('../src/calculators/RiskCalculator');
const AdvancedRiskCalculator = require('../src/calculators/AdvancedRiskCalculator');
const PortfolioOptimizer = require('../src/optimization/PortfolioOptimizer');
const StressTestingFramework = require('../src/stress-testing/StressTestingFramework');
const CopulaModeling = require('../src/models/CopulaModeling');

describe('Integration Tests', () => {
  let riskCalc;
  let advancedRiskCalc;
  let optimizer;
  let stressTest;
  let copulaModel;
  let samplePortfolio;

  beforeEach(() => {
    riskCalc = new RiskCalculator();
    advancedRiskCalc = new AdvancedRiskCalculator();
    optimizer = new PortfolioOptimizer();
    stressTest = new StressTestingFramework({
      monteCarloIterations: 100,
      confidenceLevels: [0.95]
    });
    copulaModel = new CopulaModeling();

    // Create integrated sample portfolio
    samplePortfolio = createIntegratedPortfolio();
  });

  function createIntegratedPortfolio() {
    // Generate correlated asset returns
    const n = 252; // One year of daily returns
    const assets = ['US_Equity', 'EU_Bond', 'Gold', 'Tech_Stock'];
    const correlations = [
      [1.0, -0.2, 0.1, 0.8],
      [-0.2, 1.0, 0.05, -0.3],
      [0.1, 0.05, 1.0, 0.2],
      [0.8, -0.3, 0.2, 1.0]
    ];

    const returns = generateCorrelatedReturns(n, correlations);
    const prices = returns.map(assetReturns => {
      let price = 100;
      return assetReturns.map(ret => {
        price *= (1 + ret);
        return price;
      });
    });

    return {
      name: 'Integrated Test Portfolio',
      assets: [
        {
          name: 'US_Equity',
          type: 'equity',
          prices: prices[0],
          returns: returns[0],
          weight: 0.4,
          volatility: calculateVolatility(returns[0]),
          currency: 'USD'
        },
        {
          name: 'EU_Bond',
          type: 'bond',
          prices: prices[1],
          returns: returns[1],
          weight: 0.3,
          volatility: calculateVolatility(returns[1]),
          duration: 5,
          currency: 'EUR'
        },
        {
          name: 'Gold',
          type: 'commodity',
          prices: prices[2],
          returns: returns[2],
          weight: 0.2,
          volatility: calculateVolatility(returns[2]),
          currency: 'USD'
        },
        {
          name: 'Tech_Stock',
          type: 'equity',
          prices: prices[3],
          returns: returns[3],
          weight: 0.1,
          volatility: calculateVolatility(returns[3]),
          currency: 'USD'
        }
      ],
      correlationMatrix: correlations,
      totalValue: 1000000
    };
  }

  function generateCorrelatedReturns(n, correlations) {
    const numAssets = correlations.length;
    const returns = Array.from({length: numAssets}, () => []);

    for (let i = 0; i < n; i++) {
      const independent = Array.from({length: numAssets}, () => generateNormal());

      // Apply correlation structure (simplified Cholesky)
      for (let j = 0; j < numAssets; j++) {
        let correlated = 0;
        for (let k = 0; k <= j; k++) {
          correlated += Math.sqrt(correlations[j][k]) * independent[k];
        }

        // Transform to realistic return with different volatilities
        const volatilities = [0.16, 0.04, 0.20, 0.25]; // Different asset class volatilities
        const means = [0.0008, 0.0003, 0.0005, 0.001]; // Different expected returns

        returns[j].push(means[j] + (volatilities[j] / Math.sqrt(252)) * correlated);
      }
    }

    return returns;
  }

  function generateNormal() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function calculateVolatility(returns) {
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance * 252); // Annualized
  }

  describe('End-to-End Risk Analysis Workflow', () => {
    test('should perform complete risk analysis workflow', async () => {
      // Step 1: Basic risk metrics
      const basicMetrics = samplePortfolio.assets.map(asset => {
        const volatility = riskCalc.calculateVolatility(asset.prices);
        const var95 = riskCalc.calculateValueAtRisk(asset.prices, 0.95);
        const maxDrawdown = riskCalc.calculateMaxDrawdown(asset.prices);

        return {
          asset: asset.name,
          volatility,
          var95,
          maxDrawdown
        };
      });

      expect(basicMetrics).toHaveLength(4);
      basicMetrics.forEach(metric => {
        expect(metric.volatility).toBeGreaterThan(0);
        expect(Math.abs(metric.var95)).toBeGreaterThan(0);
        expect(metric.maxDrawdown).toBeGreaterThanOrEqual(0);
      });

      // Step 2: Advanced risk metrics
      const advancedMetrics = samplePortfolio.assets.map(asset => {
        return advancedRiskCalc.calculateAdvancedRiskMetrics(
          asset.name,
          asset.returns,
          samplePortfolio.assets[0].returns // Use first asset as market proxy
        );
      });

      expect(advancedMetrics).toHaveLength(4);
      advancedMetrics.forEach(metric => {
        expect(metric.skewness).toBeDefined();
        expect(metric.kurtosis).toBeDefined();
        expect(metric.expectedShortfall).toBeGreaterThanOrEqual(metric.valueAtRisk);
      });

      // Step 3: Portfolio optimization
      const portfolioData = {
        returns: samplePortfolio.assets.map(asset => asset.returns),
        expectedReturns: samplePortfolio.assets.map(asset =>
          asset.returns.reduce((sum, r) => sum + r, 0) / asset.returns.length * 252
        ),
        covariance: calculateCovarianceMatrix(samplePortfolio.assets.map(asset => asset.returns))
      };

      const optimizedPortfolio = await optimizer.optimizePortfolio(portfolioData, {
        method: 'mean-variance',
        targetReturn: 0.08
      });

      expect(optimizedPortfolio.weights).toHaveLength(4);
      expect(Math.abs(optimizedPortfolio.weights.reduce((sum, w) => sum + w, 0) - 1)).toBeLessThan(0.01);

      // Step 4: Stress testing
      const portfolioForStressTesting = {
        name: 'Test Portfolio',
        totalValue: 1000000,
        assets: samplePortfolio.assets.map((asset, i) => ({
          name: asset.name,
          type: asset.type,
          price: asset.prices[asset.prices.length - 1],
          quantity: optimizedPortfolio.weights[i] * 1000000 / asset.prices[asset.prices.length - 1],
          weight: optimizedPortfolio.weights[i],
          volatility: asset.volatility,
          currency: asset.currency
        })),
        correlationMatrix: samplePortfolio.correlationMatrix
      };

      const stressResults = await stressTest.runComprehensiveStressTest(portfolioForStressTesting, {
        includeHistorical: true,
        includeMonteCarlo: true,
        includeExtremeValue: false,
        includeCorrelationBreakdown: true
      });

      expect(stressResults.scenarios.historical).toBeDefined();
      expect(stressResults.scenarios.monteCarlo).toBeDefined();
      expect(stressResults.summary.overallRiskLevel).toBeDefined();
      expect(stressResults.recommendations.length).toBeGreaterThan(0);

      // Step 5: Copula modeling for dependency analysis
      const bivariateReturns = [
        samplePortfolio.assets[0].returns,
        samplePortfolio.assets[1].returns
      ];

      const copulaResult = copulaModel.estimateCopula(bivariateReturns, 'gaussian');

      expect(copulaResult.parameters.correlationMatrix).toBeDefined();
      expect(copulaResult.dependenceMetrics.kendallTau).toBeDefined();
      expect(copulaResult.goodnessOfFit.aic).toBeDefined();

      // Verify integration: correlation should be consistent across methods
      const basicCorr = calculateCorrelation(
        samplePortfolio.assets[0].returns,
        samplePortfolio.assets[1].returns
      );
      const copulaCorr = copulaResult.parameters.correlationMatrix[0][1];

      expect(Math.abs(basicCorr - copulaCorr)).toBeLessThan(0.2); // Should be reasonably close

      // Return comprehensive results
      return {
        basicMetrics,
        advancedMetrics,
        optimizedPortfolio,
        stressResults,
        copulaResult
      };
    });
  });

  describe('Cross-Component Validation', () => {
    test('should validate consistency between basic and advanced risk metrics', () => {
      const asset = samplePortfolio.assets[0];

      const basicVar = riskCalc.calculateValueAtRisk(asset.prices, 0.95);
      const advancedMetrics = advancedRiskCalc.calculateAdvancedRiskMetrics(
        asset.name,
        asset.returns
      );

      // Both should calculate similar VaR values
      expect(Math.abs(basicVar - advancedMetrics.valueAtRisk)).toBeLessThan(0.05);

      // Expected Shortfall should be greater than VaR
      expect(advancedMetrics.expectedShortfall).toBeGreaterThanOrEqual(advancedMetrics.valueAtRisk);
    });

    test('should validate portfolio optimization against stress testing', async () => {
      const portfolioData = {
        returns: samplePortfolio.assets.map(asset => asset.returns),
        expectedReturns: samplePortfolio.assets.map(asset =>
          asset.returns.reduce((sum, r) => sum + r, 0) / asset.returns.length * 252
        ),
        covariance: calculateCovarianceMatrix(samplePortfolio.assets.map(asset => asset.returns))
      };

      // Optimize for minimum variance
      const minVarPortfolio = await optimizer.optimizePortfolio(portfolioData, {
        method: 'minimum-variance'
      });

      // Optimize for maximum Sharpe ratio
      const maxSharpePortfolio = await optimizer.optimizePortfolio(portfolioData, {
        method: 'max-sharpe',
        riskFreeRate: 0.02
      });

      // Min variance portfolio should have lower volatility
      expect(minVarPortfolio.volatility).toBeLessThanOrEqual(maxSharpePortfolio.volatility);

      // Max Sharpe portfolio should have higher Sharpe ratio
      expect(maxSharpePortfolio.sharpeRatio).toBeGreaterThanOrEqual(minVarPortfolio.sharpeRatio);
    });

    test('should validate copula modeling against correlation analysis', () => {
      const returns1 = samplePortfolio.assets[0].returns;
      const returns2 = samplePortfolio.assets[3].returns; // Should be highly correlated (both equities)

      const pearsonCorr = calculateCorrelation(returns1, returns2);
      const kendallTau = calculateKendallTau(returns1, returns2);

      const copulaResult = copulaModel.estimateCopula([returns1, returns2], 'gaussian');
      const copulaCorr = copulaResult.parameters.correlationMatrix[0][1];
      const copulaTau = copulaResult.dependenceMetrics.kendallTau;

      // Correlations should be consistent
      expect(Math.abs(pearsonCorr - copulaCorr)).toBeLessThan(0.3);
      expect(Math.abs(kendallTau - copulaTau)).toBeLessThan(0.2);

      // Both should indicate positive correlation for equity assets
      expect(pearsonCorr).toBeGreaterThan(0);
      expect(copulaCorr).toBeGreaterThan(0);
    });
  });

  describe('Performance and Scalability', () => {
    test('should handle multiple optimization methods efficiently', async () => {
      const portfolioData = {
        returns: samplePortfolio.assets.map(asset => asset.returns),
        expectedReturns: samplePortfolio.assets.map(asset =>
          asset.returns.reduce((sum, r) => sum + r, 0) / asset.returns.length * 252
        ),
        covariance: calculateCovarianceMatrix(samplePortfolio.assets.map(asset => asset.returns))
      };

      const methods = ['mean-variance', 'minimum-variance', 'max-sharpe', 'risk-parity'];
      const startTime = Date.now();

      const results = await Promise.all(
        methods.map(method =>
          optimizer.optimizePortfolio(portfolioData, { method })
            .catch(error => ({ error: error.message, method }))
        )
      );

      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(10000); // Should complete within 10 seconds

      const successfulResults = results.filter(r => !r.error);
      expect(successfulResults.length).toBeGreaterThan(2); // At least 3 methods should succeed
    });

    test('should handle stress testing with multiple scenarios efficiently', async () => {
      const portfolioForStressTesting = {
        name: 'Performance Test Portfolio',
        totalValue: 1000000,
        assets: samplePortfolio.assets.slice(0, 3).map((asset, i) => ({
          name: asset.name,
          type: asset.type,
          price: asset.prices[asset.prices.length - 1],
          quantity: 1000,
          weight: 1/3,
          volatility: asset.volatility,
          currency: asset.currency
        })),
        correlationMatrix: samplePortfolio.correlationMatrix.slice(0, 3).map(row => row.slice(0, 3))
      };

      const startTime = Date.now();

      const results = await stressTest.runComprehensiveStressTest(portfolioForStressTesting, {
        includeHistorical: true,
        includeMonteCarlo: true,
        includeExtremeValue: true,
        includeCorrelationBreakdown: true
      });

      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(15000); // Should complete within 15 seconds
      expect(results.scenarios.historical.count).toBeGreaterThan(0);
      expect(results.scenarios.monteCarlo.iterations).toBeGreaterThan(0);
    });
  });

  describe('Data Consistency and Quality', () => {
    test('should maintain data consistency across transformations', () => {
      const originalReturns = samplePortfolio.assets[0].returns;
      const originalLength = originalReturns.length;
      const originalMean = originalReturns.reduce((sum, r) => sum + r, 0) / originalReturns.length;

      // Transform through copula
      const uniformData = copulaModel.transformToUniform([originalReturns]);

      expect(uniformData[0]).toHaveLength(originalLength);
      expect(uniformData[0].every(u => u > 0 && u < 1)).toBe(true);

      // Calculate portfolio-level metrics
      const portfolioReturns = samplePortfolio.assets.map(asset => asset.returns);
      const portfolioWeights = samplePortfolio.assets.map(asset => asset.weight);

      const weightedReturns = portfolioReturns[0].map((_, i) =>
        portfolioReturns.reduce((sum, assetReturns, j) =>
          sum + assetReturns[i] * portfolioWeights[j], 0
        )
      );

      const portfolioVolatility = calculateVolatility(weightedReturns);
      expect(portfolioVolatility).toBeGreaterThan(0);
      expect(portfolioVolatility).toBeLessThan(1); // Should be reasonable
    });

    test('should handle edge cases gracefully', async () => {
      // Test with extreme correlations
      const extremePortfolio = {
        assets: [
          { returns: [0.1, -0.1, 0.1, -0.1], weight: 0.5 },
          { returns: [-0.1, 0.1, -0.1, 0.1], weight: 0.5 } // Perfect negative correlation
        ]
      };

      const portfolioData = {
        returns: extremePortfolio.assets.map(asset => asset.returns),
        expectedReturns: [0.0, 0.0],
        covariance: [[0.01, -0.01], [-0.01, 0.01]]
      };

      // Should handle extreme negative correlation
      const result = await optimizer.optimizePortfolio(portfolioData, {
        method: 'minimum-variance'
      }).catch(error => ({ error: error.message }));

      // Either succeeds or fails gracefully
      expect(result).toBeDefined();
      if (!result.error) {
        expect(result.weights).toHaveLength(2);
      }
    });
  });

  // Helper functions
  function calculateCovarianceMatrix(returnsMatrix) {
    const n = returnsMatrix.length;
    const covariance = Array(n).fill().map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const returns1 = returnsMatrix[i];
        const returns2 = returnsMatrix[j];
        const mean1 = returns1.reduce((sum, r) => sum + r, 0) / returns1.length;
        const mean2 = returns2.reduce((sum, r) => sum + r, 0) / returns2.length;

        let cov = 0;
        for (let k = 0; k < returns1.length; k++) {
          cov += (returns1[k] - mean1) * (returns2[k] - mean2);
        }
        covariance[i][j] = cov / (returns1.length - 1) * 252; // Annualized
      }
    }

    return covariance;
  }

  function calculateCorrelation(x, y) {
    const n = x.length;
    const meanX = x.reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let sumX2 = 0;
    let sumY2 = 0;

    for (let i = 0; i < n; i++) {
      const diffX = x[i] - meanX;
      const diffY = y[i] - meanY;
      numerator += diffX * diffY;
      sumX2 += diffX * diffX;
      sumY2 += diffY * diffY;
    }

    return numerator / Math.sqrt(sumX2 * sumY2);
  }

  function calculateKendallTau(u, v) {
    const n = u.length;
    let concordant = 0;
    let discordant = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const sign1 = Math.sign(u[i] - u[j]);
        const sign2 = Math.sign(v[i] - v[j]);

        if (sign1 * sign2 > 0) concordant++;
        else if (sign1 * sign2 < 0) discordant++;
      }
    }

    return (concordant - discordant) / (concordant + discordant);
  }
});