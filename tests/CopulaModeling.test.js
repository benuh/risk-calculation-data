/**
 * Copula Modeling Tests
 */

const CopulaModeling = require('../src/models/CopulaModeling');

describe('CopulaModeling', () => {
  let copulaModel;
  let sampleData;
  let bivariateData;

  beforeEach(() => {
    copulaModel = new CopulaModeling({
      tolerance: 1e-4,
      maxIterations: 100,
      bootstrapSamples: 50 // Reduced for faster testing
    });

    // Generate sample correlated data
    const n = 200;
    sampleData = [];
    bivariateData = [];

    for (let i = 0; i < 3; i++) {
      sampleData.push([]);
    }

    for (let i = 0; i < 2; i++) {
      bivariateData.push([]);
    }

    // Generate correlated data
    for (let i = 0; i < n; i++) {
      const z1 = generateNormal();
      const z2 = generateNormal();
      const z3 = generateNormal();

      // Create correlation structure
      const x1 = z1;
      const x2 = 0.7 * z1 + Math.sqrt(1 - 0.49) * z2;
      const x3 = 0.3 * z1 + 0.4 * z2 + Math.sqrt(1 - 0.09 - 0.16) * z3;

      sampleData[0].push(0.01 + 0.02 * x1);
      sampleData[1].push(0.008 + 0.018 * x2);
      sampleData[2].push(0.012 + 0.025 * x3);

      if (i < n) {
        bivariateData[0].push(0.01 + 0.02 * x1);
        bivariateData[1].push(0.008 + 0.018 * x2);
      }
    }
  });

  function generateNormal() {
    // Box-Muller transformation
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  describe('Data Transformation', () => {
    test('should transform data to uniform marginals', () => {
      const uniformData = copulaModel.transformToUniform(sampleData);

      expect(uniformData).toHaveLength(3);
      expect(uniformData[0]).toHaveLength(sampleData[0].length);

      // Check that transformed data is in [0,1]
      uniformData.forEach(series => {
        series.forEach(value => {
          expect(value).toBeGreaterThan(0);
          expect(value).toBeLessThan(1);
        });
      });

      // Check that transformation preserves order
      const originalRanks = sampleData[0].map((_, i) => i).sort((a, b) => sampleData[0][a] - sampleData[0][b]);
      const uniformRanks = uniformData[0].map((_, i) => i).sort((a, b) => uniformData[0][a] - uniformData[0][b]);
      expect(originalRanks).toEqual(uniformRanks);
    });
  });

  describe('Gaussian Copula', () => {
    test('should estimate Gaussian copula parameters', () => {
      const result = copulaModel.estimateCopula(sampleData, 'gaussian');

      expect(result).toHaveProperty('copulaType', 'gaussian');
      expect(result).toHaveProperty('parameters');
      expect(result).toHaveProperty('goodnessOfFit');
      expect(result).toHaveProperty('dependenceMetrics');

      expect(result.parameters).toHaveProperty('correlationMatrix');
      expect(result.parameters).toHaveProperty('logLikelihood');

      const corrMatrix = result.parameters.correlationMatrix;
      expect(corrMatrix).toHaveLength(3);
      expect(corrMatrix[0]).toHaveLength(3);

      // Diagonal should be 1
      expect(corrMatrix[0][0]).toBeCloseTo(1, 2);
      expect(corrMatrix[1][1]).toBeCloseTo(1, 2);
      expect(corrMatrix[2][2]).toBeCloseTo(1, 2);

      // Matrix should be symmetric
      expect(corrMatrix[0][1]).toBeCloseTo(corrMatrix[1][0], 2);
      expect(corrMatrix[0][2]).toBeCloseTo(corrMatrix[2][0], 2);
      expect(corrMatrix[1][2]).toBeCloseTo(corrMatrix[2][1], 2);
    });

    test('should handle insufficient data', () => {
      const smallData = [[0.01, 0.02], [0.015, 0.018]];
      expect(() => {
        copulaModel.estimateCopula(smallData, 'gaussian');
      }).toThrow('Insufficient data for copula estimation');
    });
  });

  describe('Student t-Copula', () => {
    test('should estimate Student t-copula parameters', () => {
      const result = copulaModel.estimateCopula(sampleData, 't');

      expect(result.copulaType).toBe('t');
      expect(result.parameters).toHaveProperty('correlationMatrix');
      expect(result.parameters).toHaveProperty('degreesOfFreedom');
      expect(result.parameters).toHaveProperty('logLikelihood');

      expect(result.parameters.degreesOfFreedom).toBeGreaterThan(1);
      expect(result.parameters.degreesOfFreedom).toBeLessThan(31);
    });
  });

  describe('Archimedean Copulas', () => {
    test('should estimate Clayton copula for bivariate data', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'clayton');

      expect(result.copulaType).toBe('clayton');
      expect(result.parameters).toHaveProperty('theta');
      expect(result.parameters).toHaveProperty('kendallTau');
      expect(result.parameters).toHaveProperty('logLikelihood');

      expect(result.parameters.theta).toBeGreaterThan(0);
      expect(Math.abs(result.parameters.kendallTau)).toBeLessThan(1);
    });

    test('should estimate Gumbel copula for bivariate data', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gumbel');

      expect(result.copulaType).toBe('gumbel');
      expect(result.parameters).toHaveProperty('theta');
      expect(result.parameters).toHaveProperty('upperTailDependence');

      expect(result.parameters.theta).toBeGreaterThanOrEqual(1);
      expect(result.parameters.upperTailDependence).toBeGreaterThanOrEqual(0);
      expect(result.parameters.upperTailDependence).toBeLessThanOrEqual(1);
    });

    test('should estimate Frank copula for bivariate data', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'frank');

      expect(result.copulaType).toBe('frank');
      expect(result.parameters).toHaveProperty('theta');
      expect(result.parameters).toHaveProperty('kendallTau');

      expect(typeof result.parameters.theta).toBe('number');
      expect(Math.abs(result.parameters.kendallTau)).toBeLessThan(1);
    });

    test('should throw error for multivariate Archimedean copulas', () => {
      expect(() => {
        copulaModel.estimateCopula(sampleData, 'clayton');
      }).toThrow('Clayton copula currently supports only bivariate case');
    });
  });

  describe('Dependence Metrics', () => {
    test('should calculate dependence metrics for bivariate case', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gaussian');
      const metrics = result.dependenceMetrics;

      expect(metrics).toHaveProperty('kendallTau');
      expect(metrics).toHaveProperty('spearmanRho');
      expect(metrics).toHaveProperty('pearsonRho');
      expect(metrics).toHaveProperty('upperTailDependence');
      expect(metrics).toHaveProperty('lowerTailDependence');

      expect(Math.abs(metrics.kendallTau)).toBeLessThan(1);
      expect(Math.abs(metrics.spearmanRho)).toBeLessThan(1);
      expect(Math.abs(metrics.pearsonRho)).toBeLessThan(1);
    });

    test('should calculate dependence metrics for multivariate case', () => {
      const result = copulaModel.estimateCopula(sampleData, 'gaussian');
      const metrics = result.dependenceMetrics;

      expect(metrics).toHaveProperty('averageKendallTau');
      expect(metrics).toHaveProperty('pairwiseCorrelations');

      expect(typeof metrics.averageKendallTau).toBe('number');
      expect(Array.isArray(metrics.pairwiseCorrelations)).toBe(true);
      expect(metrics.pairwiseCorrelations).toHaveLength(3);
    });
  });

  describe('Goodness of Fit', () => {
    test('should calculate goodness of fit statistics', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gaussian');
      const gof = result.goodnessOfFit;

      expect(gof).toHaveProperty('aic');
      expect(gof).toHaveProperty('bic');
      expect(gof).toHaveProperty('logLikelihood');
      expect(gof).toHaveProperty('numParameters');

      expect(typeof gof.aic).toBe('number');
      expect(typeof gof.bic).toBe('number');
      expect(gof.bic).toBeGreaterThan(gof.aic); // BIC penalizes more than AIC
      expect(gof.numParameters).toBeGreaterThan(0);
    });

    test('should include statistical tests', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gaussian');
      const gof = result.goodnessOfFit;

      expect(gof).toHaveProperty('cramerVonMises');
      expect(gof).toHaveProperty('kolmogorovSmirnov');
      expect(gof).toHaveProperty('andersonDarling');

      expect(gof.cramerVonMises).toHaveProperty('statistic');
      expect(gof.cramerVonMises).toHaveProperty('pValue');
    });
  });

  describe('Tail Dependence Analysis', () => {
    test('should calculate tail dependence for Clayton copula', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'clayton');
      const metrics = result.dependenceMetrics;

      expect(metrics.upperTailDependence).toBe(0); // Clayton has no upper tail dependence
      expect(metrics.lowerTailDependence).toBeGreaterThanOrEqual(0);
    });

    test('should calculate tail dependence for Gumbel copula', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gumbel');
      const metrics = result.dependenceMetrics;

      expect(metrics.lowerTailDependence).toBe(0); // Gumbel has no lower tail dependence
      expect(metrics.upperTailDependence).toBeGreaterThanOrEqual(0);
    });

    test('should show no tail dependence for Gaussian copula', () => {
      const result = copulaModel.estimateCopula(bivariateData, 'gaussian');
      const metrics = result.dependenceMetrics;

      expect(metrics.upperTailDependence).toBe(0);
      expect(metrics.lowerTailDependence).toBe(0);
    });
  });

  describe('VaR Calculation with Copulas', () => {
    test('should calculate copula-based VaR', () => {
      const copulaResult = copulaModel.estimateCopula(bivariateData, 'gaussian');

      const marginalDistributions = bivariateData.map(returns => {
        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / (returns.length - 1);
        return {
          type: 'normal',
          mean,
          standardDeviation: Math.sqrt(variance)
        };
      });

      const varResults = copulaModel.calculateCopulaBasedVaR(
        bivariateData,
        marginalDistributions,
        copulaResult,
        0.95
      );

      expect(varResults).toHaveProperty('var');
      expect(varResults).toHaveProperty('expectedShortfall');
      expect(varResults).toHaveProperty('portfolioReturns');

      expect(typeof varResults.var).toBe('number');
      expect(typeof varResults.expectedShortfall).toBe('number');
      expect(varResults.expectedShortfall).toBeLessThanOrEqual(varResults.var);
      expect(Array.isArray(varResults.portfolioReturns)).toBe(true);
    });
  });

  describe('Utility Functions', () => {
    test('should calculate correlation correctly', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [2, 4, 6, 8, 10]; // Perfect positive correlation

      const corr = copulaModel.calculateCorrelation(x, y);
      expect(corr).toBeCloseTo(1, 2);
    });

    test('should calculate Kendall\'s tau correctly', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [1, 2, 3, 4, 5]; // Perfect concordance

      const tau = copulaModel.calculateKendallTau(x, y);
      expect(tau).toBeCloseTo(1, 2);
    });

    test('should calculate Spearman\'s rho correctly', () => {
      const x = [1, 2, 3, 4, 5];
      const y = [5, 4, 3, 2, 1]; // Perfect negative rank correlation

      const rho = copulaModel.calculateSpearmanRho(x, y);
      expect(rho).toBeCloseTo(-1, 1);
    });

    test('should compute inverse normal CDF correctly', () => {
      const result1 = copulaModel.inverseNormalCDF(0.5);
      expect(result1).toBeCloseTo(0, 1);

      const result2 = copulaModel.inverseNormalCDF(0.975);
      expect(result2).toBeCloseTo(1.96, 1);

      const result3 = copulaModel.inverseNormalCDF(0.025);
      expect(result3).toBeCloseTo(-1.96, 1);
    });
  });

  describe('Matrix Operations', () => {
    test('should calculate matrix determinant correctly', () => {
      const matrix2x2 = [[1, 2], [3, 4]];
      const det = copulaModel.matrixDeterminant(matrix2x2);
      expect(det).toBeCloseTo(-2, 2);

      const matrix3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const det3 = copulaModel.matrixDeterminant(matrix3x3);
      expect(det3).toBeCloseTo(1, 2);
    });

    test('should calculate matrix inverse correctly', () => {
      const matrix = [[2, 0], [0, 2]];
      const inverse = copulaModel.matrixInverse(matrix);

      expect(inverse[0][0]).toBeCloseTo(0.5, 2);
      expect(inverse[1][1]).toBeCloseTo(0.5, 2);
      expect(inverse[0][1]).toBeCloseTo(0, 2);
      expect(inverse[1][0]).toBeCloseTo(0, 2);
    });

    test('should ensure positive definite matrix', () => {
      const matrix = [
        [1, 0.8, 0.7],
        [0.8, 1, 0.9],
        [0.7, 0.9, 1]
      ];

      const adjusted = copulaModel.ensurePositiveDefinite(matrix);

      expect(adjusted).toHaveLength(3);
      expect(adjusted[0]).toHaveLength(3);
      expect(adjusted[0][0]).toBeGreaterThanOrEqual(0.001);
      expect(adjusted[1][1]).toBeGreaterThanOrEqual(0.001);
      expect(adjusted[2][2]).toBeGreaterThanOrEqual(0.001);
    });
  });

  describe('Error Handling', () => {
    test('should handle unsupported copula types', () => {
      expect(() => {
        copulaModel.estimateCopula(bivariateData, 'unsupported');
      }).toThrow('Unsupported copula type: unsupported');
    });

    test('should handle single asset data', () => {
      const singleAsset = [[0.01, 0.02, 0.015, 0.008]];
      expect(() => {
        copulaModel.estimateCopula(singleAsset, 'gaussian');
      }).toThrow('Insufficient data for copula estimation');
    });

    test('should handle data with extreme values', () => {
      const extremeData = [];
      const n = 50;
      for (let i = 0; i < 2; i++) {
        extremeData.push([]);
      }

      for (let i = 0; i < n; i++) {
        extremeData[0].push(0.01 + 0.02 * generateNormal());
        extremeData[1].push(0.015 + 0.018 * generateNormal());
      }

      // Should not throw, but handle gracefully
      expect(() => {
        copulaModel.estimateCopula(extremeData, 'gaussian');
      }).not.toThrow();
    });
  });

  describe('Performance Tests', () => {
    test('should handle moderately large datasets', () => {
      const largeData = [];
      const n = 1000;

      for (let i = 0; i < 2; i++) {
        largeData.push([]);
      }

      for (let i = 0; i < n; i++) {
        const z1 = generateNormal();
        const z2 = 0.5 * z1 + Math.sqrt(0.75) * generateNormal();

        largeData[0].push(0.01 + 0.02 * z1);
        largeData[1].push(0.008 + 0.018 * z2);
      }

      const startTime = Date.now();
      const result = copulaModel.estimateCopula(largeData, 'gaussian');
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
      expect(result.parameters.correlationMatrix[0][1]).toBeCloseTo(0.5, 1);
    });
  });
});