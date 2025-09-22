/**
 * Copula Modeling Example
 *
 * This example demonstrates advanced copula-based dependency modeling
 * for financial risk assessment and portfolio optimization.
 */

const CopulaModeling = require('../src/models/CopulaModeling');

// Generate sample correlated data for demonstration
function generateCorrelatedReturns(n = 1000, correlation = 0.7) {
  const returns1 = [];
  const returns2 = [];
  const returns3 = [];

  for (let i = 0; i < n; i++) {
    // Generate independent normal variables
    const z1 = generateStandardNormal();
    const z2 = generateStandardNormal();
    const z3 = generateStandardNormal();

    // Create correlation structure
    const x1 = z1;
    const x2 = correlation * z1 + Math.sqrt(1 - correlation * correlation) * z2;
    const x3 = 0.5 * z1 + 0.3 * z2 + Math.sqrt(1 - 0.5*0.5 - 0.3*0.3) * z3;

    // Transform to returns (with different volatilities and means)
    returns1.push(0.0005 + 0.015 * x1); // Asset 1: 1.5% volatility
    returns2.push(-0.0002 + 0.020 * x2); // Asset 2: 2.0% volatility
    returns3.push(0.0008 + 0.025 * x3); // Asset 3: 2.5% volatility
  }

  return [returns1, returns2, returns3];
}

function generateStandardNormal() {
  // Box-Muller transformation
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Generate data with different dependency structures
function generateNonLinearDependentData(n = 1000) {
  const returns1 = [];
  const returns2 = [];

  for (let i = 0; i < n; i++) {
    const u1 = Math.random();
    const u2 = Math.random();

    // Clayton copula-like structure (lower tail dependence)
    let v1, v2;
    if (Math.random() < 0.3) { // 30% chance of extreme negative correlation
      v1 = u1;
      v2 = Math.pow(Math.pow(u1, -0.5) + Math.pow(u2, -0.5) - 1, -2);
    } else {
      v1 = u1;
      v2 = u2;
    }

    // Transform to returns
    const norm1 = inverseNormalCDF(v1);
    const norm2 = inverseNormalCDF(Math.max(0.001, Math.min(0.999, v2)));

    returns1.push(0.0003 + 0.018 * norm1);
    returns2.push(0.0001 + 0.022 * norm2);
  }

  return [returns1, returns2];
}

function inverseNormalCDF(p) {
  // Simplified inverse normal CDF
  if (p <= 0) return -6;
  if (p >= 1) return 6;

  const a = [0, -3.969683028665376e+01, 2.209460984245205e+02];
  const b = [0, -5.447609879822406e+01, 1.615858368580409e+02];

  if (p < 0.5) {
    const q = Math.sqrt(-2 * Math.log(p));
    return -(a[1] + a[2] * q) / (1 + b[1] * q + b[2] * q * q);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return (a[1] + a[2] * q) / (1 + b[1] * q + b[2] * q * q);
  }
}

async function copulaModelingExample() {
  console.log('='.repeat(60));
  console.log('COPULA MODELING EXAMPLE');
  console.log('='.repeat(60));

  // Initialize copula modeling framework
  const copulaModel = new CopulaModeling({
    tolerance: 1e-6,
    maxIterations: 1000,
    confidenceLevels: [0.90, 0.95, 0.99],
    bootstrapSamples: 500
  });

  console.log('\\n1. GAUSSIAN COPULA ESTIMATION');
  console.log('-'.repeat(40));

  try {
    // Generate correlated data
    const correlatedData = generateCorrelatedReturns(1000, 0.6);

    console.log('Generated 3 assets with 1,000 observations each');
    console.log('Asset correlations:');
    console.log(`  Asset 1 - Asset 2: ${calculateCorrelation(correlatedData[0], correlatedData[1]).toFixed(3)}`);
    console.log(`  Asset 1 - Asset 3: ${calculateCorrelation(correlatedData[0], correlatedData[2]).toFixed(3)}`);
    console.log(`  Asset 2 - Asset 3: ${calculateCorrelation(correlatedData[1], correlatedData[2]).toFixed(3)}`);

    // Estimate Gaussian copula
    const gaussianResult = copulaModel.estimateCopula(correlatedData, 'gaussian');

    console.log('\\nGaussian Copula Results:');
    console.log('Correlation Matrix:');
    gaussianResult.parameters.correlationMatrix.forEach((row, i) => {
      const rowStr = row.map(val => val.toFixed(3)).join('  ');
      console.log(`  [${rowStr}]`);
    });

    console.log(`\\nLog-Likelihood: ${gaussianResult.parameters.logLikelihood.toFixed(2)}`);
    console.log(`AIC: ${gaussianResult.goodnessOfFit.aic.toFixed(2)}`);
    console.log(`BIC: ${gaussianResult.goodnessOfFit.bic.toFixed(2)}`);

    console.log('\\nDependence Metrics:');
    console.log(`  Average Kendall's Tau: ${gaussianResult.dependenceMetrics.averageKendallTau.toFixed(3)}`);

  } catch (error) {
    console.error('Error in Gaussian copula estimation:', error.message);
  }

  console.log('\\n2. STUDENT T-COPULA ESTIMATION');
  console.log('-'.repeat(40));

  try {
    const correlatedData = generateCorrelatedReturns(800, 0.5);

    // Estimate t-copula
    const tCopulaResult = copulaModel.estimateCopula(correlatedData, 't');

    console.log('Student t-Copula Results:');
    console.log(`Degrees of Freedom: ${tCopulaResult.parameters.degreesOfFreedom}`);
    console.log(`Log-Likelihood: ${tCopulaResult.parameters.logLikelihood.toFixed(2)}`);
    console.log(`AIC: ${tCopulaResult.goodnessOfFit.aic.toFixed(2)}`);

    console.log('\\nCorrelation Matrix:');
    tCopulaResult.parameters.correlationMatrix.forEach((row, i) => {
      const rowStr = row.map(val => val.toFixed(3)).join('  ');
      console.log(`  [${rowStr}]`);
    });

  } catch (error) {
    console.error('Error in t-copula estimation:', error.message);
  }

  console.log('\\n3. ARCHIMEDEAN COPULA COMPARISON');
  console.log('-'.repeat(40));

  try {
    // Generate bivariate data with tail dependence
    const bivariateTailData = generateNonLinearDependentData(1200);

    console.log('Generated bivariate data with potential tail dependence');
    console.log(`Linear Correlation: ${calculateCorrelation(bivariateTailData[0], bivariateTailData[1]).toFixed(3)}`);
    console.log(`Kendall's Tau: ${calculateKendallTau(bivariateTailData[0], bivariateTailData[1]).toFixed(3)}`);

    // Test different Archimedean copulas
    const copulaTypes = ['clayton', 'gumbel', 'frank'];
    const results = {};

    for (const copulaType of copulaTypes) {
      try {
        const result = copulaModel.estimateCopula(bivariateTailData, copulaType);
        results[copulaType] = result;

        console.log(`\\n${copulaType.toUpperCase()} COPULA:`);
        console.log(`  Parameter (θ): ${result.parameters.theta.toFixed(3)}`);
        console.log(`  Log-Likelihood: ${result.parameters.logLikelihood.toFixed(2)}`);
        console.log(`  AIC: ${result.goodnessOfFit.aic.toFixed(2)}`);
        console.log(`  Kendall's Tau: ${result.parameters.kendallTau.toFixed(3)}`);

        if (result.parameters.upperTailDependence !== undefined) {
          console.log(`  Upper Tail Dependence: ${result.parameters.upperTailDependence.toFixed(3)}`);
        }

        if (result.dependenceMetrics.lowerTailDependence !== undefined) {
          console.log(`  Lower Tail Dependence: ${result.dependenceMetrics.lowerTailDependence.toFixed(3)}`);
        }

      } catch (error) {
        console.log(`\\n${copulaType.toUpperCase()} COPULA: Estimation failed (${error.message})`);
      }
    }

    // Compare models using AIC
    console.log('\\nMODEL COMPARISON (AIC):');
    const aicResults = Object.entries(results)
      .map(([type, result]) => ({ type, aic: result.goodnessOfFit.aic }))
      .sort((a, b) => a.aic - b.aic);

    aicResults.forEach((result, index) => {
      console.log(`  ${index + 1}. ${result.type.toUpperCase()}: AIC = ${result.aic.toFixed(2)}`);
    });

    if (aicResults.length > 0) {
      console.log(`\\nBest fitting model: ${aicResults[0].type.toUpperCase()}`);
    }

  } catch (error) {
    console.error('Error in Archimedean copula comparison:', error.message);
  }

  console.log('\\n4. TAIL DEPENDENCE ANALYSIS');
  console.log('-'.repeat(40));

  try {
    // Create data with extreme dependencies
    const extremeData = [];
    const n = 1000;

    // Asset 1: Base returns
    const asset1 = Array.from({length: n}, () => 0.0005 + 0.02 * generateStandardNormal());

    // Asset 2: High correlation during stress periods
    const asset2 = asset1.map((ret, i) => {
      if (ret < -0.03) { // Stress period (3% daily loss)
        return ret * 1.2 + 0.01 * generateStandardNormal(); // High correlation
      } else {
        return 0.0003 + 0.018 * generateStandardNormal(); // Normal correlation
      }
    });

    const tailDependenceData = [asset1, asset2];

    console.log('Analyzing tail dependence characteristics:');
    console.log(`Sample size: ${n} observations`);

    // Estimate different copulas to capture tail dependence
    const tailResults = {};

    try {
      const gaussianTail = copulaModel.estimateCopula(tailDependenceData, 'gaussian');
      tailResults.gaussian = gaussianTail;
      console.log('\\nGaussian Copula (no tail dependence):');
      console.log(`  Upper Tail Dependence: ${gaussianTail.dependenceMetrics.upperTailDependence.toFixed(3)}`);
      console.log(`  Lower Tail Dependence: ${gaussianTail.dependenceMetrics.lowerTailDependence.toFixed(3)}`);
    } catch (error) {
      console.log('\\nGaussian Copula: Estimation failed');
    }

    try {
      const claytonTail = copulaModel.estimateCopula(tailDependenceData, 'clayton');
      tailResults.clayton = claytonTail;
      console.log('\\nClayton Copula (lower tail dependence):');
      console.log(`  Parameter θ: ${claytonTail.parameters.theta.toFixed(3)}`);
      console.log(`  Upper Tail Dependence: ${claytonTail.dependenceMetrics.upperTailDependence.toFixed(3)}`);
      console.log(`  Lower Tail Dependence: ${claytonTail.dependenceMetrics.lowerTailDependence.toFixed(3)}`);
    } catch (error) {
      console.log('\\nClayton Copula: Estimation failed');
    }

    try {
      const gumbelTail = copulaModel.estimateCopula(tailDependenceData, 'gumbel');
      tailResults.gumbel = gumbelTail;
      console.log('\\nGumbel Copula (upper tail dependence):');
      console.log(`  Parameter θ: ${gumbelTail.parameters.theta.toFixed(3)}`);
      console.log(`  Upper Tail Dependence: ${gumbelTail.dependenceMetrics.upperTailDependence.toFixed(3)}`);
      console.log(`  Lower Tail Dependence: ${gumbelTail.dependenceMetrics.lowerTailDependence.toFixed(3)}`);
    } catch (error) {
      console.log('\\nGumbel Copula: Estimation failed');
    }

    // Analyze extreme correlations
    console.log('\\nEXTREME EVENT ANALYSIS:');
    const extremeThreshold = -0.02; // 2% daily loss threshold
    const extremeIndices = asset1.map((ret, i) => ret < extremeThreshold ? i : -1).filter(i => i >= 0);

    if (extremeIndices.length > 5) {
      const extremeCorr = calculateCorrelation(
        extremeIndices.map(i => asset1[i]),
        extremeIndices.map(i => asset2[i])
      );

      const normalIndices = asset1.map((ret, i) => ret >= extremeThreshold ? i : -1).filter(i => i >= 0);
      const normalCorr = calculateCorrelation(
        normalIndices.slice(0, 100).map(i => asset1[i]),
        normalIndices.slice(0, 100).map(i => asset2[i])
      );

      console.log(`  Extreme periods correlation: ${extremeCorr.toFixed(3)}`);
      console.log(`  Normal periods correlation: ${normalCorr.toFixed(3)}`);
      console.log(`  Correlation increase during stress: ${((extremeCorr - normalCorr) * 100).toFixed(1)}%`);
      console.log(`  Number of extreme events: ${extremeIndices.length}`);
    }

  } catch (error) {
    console.error('Error in tail dependence analysis:', error.message);
  }

  console.log('\\n5. COPULA-BASED VAR CALCULATION');
  console.log('-'.repeat(40));

  try {
    const portfolioData = generateCorrelatedReturns(500, 0.4);

    // Estimate best-fitting copula
    const portfolioCopula = copulaModel.estimateCopula(portfolioData, 'gaussian');

    console.log('Calculating VaR using copula-based approach...');

    // Simple marginal distributions (assume normal for demonstration)
    const marginalDistributions = portfolioData.map(assetReturns => {
      const mean = assetReturns.reduce((sum, ret) => sum + ret, 0) / assetReturns.length;
      const variance = assetReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / (assetReturns.length - 1);

      return {
        type: 'normal',
        mean,
        standardDeviation: Math.sqrt(variance)
      };
    });

    // Calculate VaR using copula simulation
    const varResults = copulaModel.calculateCopulaBasedVaR(
      portfolioData,
      marginalDistributions,
      portfolioCopula,
      0.95
    );

    console.log('\\nCopula-based Risk Metrics:');
    console.log(`  VaR (95%): ${(varResults.var * 100).toFixed(2)}%`);
    console.log(`  Expected Shortfall: ${(varResults.expectedShortfall * 100).toFixed(2)}%`);

    // Compare with traditional correlation-based VaR
    const avgReturn = marginalDistributions.reduce((sum, dist) => sum + dist.mean, 0) / marginalDistributions.length;
    const avgVol = marginalDistributions.reduce((sum, dist) => sum + dist.standardDeviation, 0) / marginalDistributions.length;
    const traditionalVaR = avgReturn - 1.645 * avgVol; // 95% VaR

    console.log(`\\nComparison with traditional VaR:`);
    console.log(`  Traditional VaR (95%): ${(traditionalVaR * 100).toFixed(2)}%`);
    console.log(`  Copula VaR difference: ${((varResults.var - traditionalVaR) * 100).toFixed(2)}%`);

  } catch (error) {
    console.error('Error in copula-based VaR calculation:', error.message);
  }

  console.log('\\n6. RISK MANAGEMENT IMPLICATIONS');
  console.log('-'.repeat(40));

  try {
    console.log('Key insights from copula analysis:');

    console.log('\\nDEPENDENCE STRUCTURE:');
    console.log('• Linear correlation may underestimate extreme dependencies');
    console.log('• Tail dependence can significantly impact portfolio risk');
    console.log('• Different copulas capture different dependency patterns');

    console.log('\\nRISK MANAGEMENT APPLICATIONS:');
    console.log('• Stress testing with realistic dependency structures');
    console.log('• Portfolio optimization considering tail dependencies');
    console.log('• Dynamic hedging based on changing dependencies');

    console.log('\\nMODEL SELECTION GUIDELINES:');
    console.log('• Use AIC/BIC for model comparison');
    console.log('• Consider economic intuition about dependencies');
    console.log('• Validate with out-of-sample performance');

    console.log('\\nPORTFOLIO IMPLICATIONS:');
    console.log('• Diversification benefits may disappear in crises');
    console.log('• Consider asymmetric dependency patterns');
    console.log('• Monitor correlation regime changes');

  } catch (error) {
    console.error('Error generating implications:', error.message);
  }

  console.log('\\n' + '='.repeat(60));
  console.log('COPULA MODELING EXAMPLE COMPLETED');
  console.log('='.repeat(60));
  console.log('\\nNote: This example demonstrates copula concepts');
  console.log('using simulated data. In practice, use historical');
  console.log('market data and robust estimation procedures.');
}

// Helper functions
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

// Run the example
if (require.main === module) {
  copulaModelingExample().catch(error => {
    console.error('Copula modeling example failed:', error);
    process.exit(1);
  });
}

module.exports = {
  copulaModelingExample,
  generateCorrelatedReturns,
  generateNonLinearDependentData
};