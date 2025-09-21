const { RiskMetrics } = require('../models/FinancialData');

class AdvancedRiskCalculator {
  constructor() {
    this.riskFreeRate = 0.02;
    this.confidenceLevel = 0.95;
    this.tradingDaysPerYear = 252;
  }

  setRiskFreeRate(rate) {
    this.riskFreeRate = rate;
  }

  setConfidenceLevel(level) {
    this.confidenceLevel = level;
  }

  calculateGarchVolatility(returns, alpha = 0.1, beta = 0.85, omega = 0.01) {
    if (!returns || returns.length < 50) {
      throw new Error('Insufficient data for GARCH calculation (minimum 50 observations)');
    }

    const n = returns.length;
    const variances = new Array(n);

    variances[0] = omega / (1 - alpha - beta);

    for (let t = 1; t < n; t++) {
      variances[t] = omega + alpha * Math.pow(returns[t-1], 2) + beta * variances[t-1];
    }

    const currentVolatility = Math.sqrt(variances[n-1] * this.tradingDaysPerYear);
    return {
      currentVolatility,
      variances,
      garchParameters: { alpha, beta, omega }
    };
  }

  calculateExpectedShortfall(returns, confidenceLevel = this.confidenceLevel) {
    if (!returns || returns.length < 30) {
      throw new Error('Insufficient data for Expected Shortfall calculation');
    }

    const sortedReturns = [...returns].sort((a, b) => a - b);
    const cutoffIndex = Math.floor((1 - confidenceLevel) * sortedReturns.length);
    const tailReturns = sortedReturns.slice(0, cutoffIndex);

    if (tailReturns.length === 0) {
      return 0;
    }

    const expectedShortfall = tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length;
    return expectedShortfall;
  }

  calculateCornishFisherVaR(returns, confidenceLevel = this.confidenceLevel) {
    if (!returns || returns.length < 30) {
      throw new Error('Insufficient data for Cornish-Fisher VaR calculation');
    }

    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = this.calculateVariance(returns);
    const skewness = this.calculateSkewness(returns);
    const kurtosis = this.calculateKurtosis(returns);

    const z = this.inverseNormalCDF(1 - confidenceLevel);

    const cornishFisherZ = z +
      (z*z - 1) * skewness / 6 +
      (z*z*z - 3*z) * (kurtosis - 3) / 24 -
      (2*z*z*z - 5*z) * skewness*skewness / 36;

    return mean + Math.sqrt(variance) * cornishFisherZ;
  }

  calculateSkewness(returns) {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const n = returns.length;

    const numerator = returns.reduce((sum, ret) => {
      return sum + Math.pow(ret - mean, 3);
    }, 0) / n;

    const denominator = Math.pow(this.calculateVariance(returns), 1.5);

    return numerator / denominator;
  }

  calculateKurtosis(returns) {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const n = returns.length;

    const numerator = returns.reduce((sum, ret) => {
      return sum + Math.pow(ret - mean, 4);
    }, 0) / n;

    const denominator = Math.pow(this.calculateVariance(returns), 2);

    return numerator / denominator;
  }

  calculateVariance(returns) {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    return returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / (returns.length - 1);
  }

  inverseNormalCDF(p) {
    if (p <= 0 || p >= 1) {
      throw new Error('Probability must be between 0 and 1');
    }

    const a0 = -3.969683028665376e+01;
    const a1 =  2.209460984245205e+02;
    const a2 = -2.759285104469687e+02;
    const a3 =  1.383577518672690e+02;
    const a4 = -3.066479806614716e+01;
    const a5 =  2.506628277459239e+00;

    const b1 = -5.447609879822406e+01;
    const b2 =  1.615858368580409e+02;
    const b3 = -1.556989798598866e+02;
    const b4 =  6.680131188771972e+01;
    const b5 = -1.328068155288572e+01;

    const c0 = -7.784894002430293e-03;
    const c1 = -3.223964580411365e-01;
    const c2 = -2.400758277161838e+00;
    const c3 = -2.549732539343734e+00;
    const c4 =  4.374664141464968e+00;
    const c5 =  2.938163982698783e+00;

    const d1 =  7.784695709041462e-03;
    const d2 =  3.224671290700398e-01;
    const d3 =  2.445134137142996e+00;
    const d4 =  3.754408661907416e+00;

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let x;

    if (p < pLow) {
      const q = Math.sqrt(-2 * Math.log(p));
      x = (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    } else if (p <= pHigh) {
      const q = p - 0.5;
      const r = q * q;
      x = (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
    } else {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }

    return x;
  }

  calculateComponentVaR(portfolioWeights, assetReturns, correlationMatrix, confidenceLevel = this.confidenceLevel) {
    if (!portfolioWeights || !assetReturns || !correlationMatrix) {
      throw new Error('Missing required data for Component VaR calculation');
    }

    const n = portfolioWeights.length;

    if (assetReturns.length !== n || correlationMatrix.length !== n) {
      throw new Error('Dimension mismatch in Component VaR inputs');
    }

    const portfolioVariance = this.calculatePortfolioVariance(portfolioWeights, assetReturns, correlationMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance);

    const portfolioVaR = this.inverseNormalCDF(1 - confidenceLevel) * portfolioVolatility;

    const componentVaRs = [];

    for (let i = 0; i < n; i++) {
      let marginalVaR = 0;

      for (let j = 0; j < n; j++) {
        const assetVolatilityI = Math.sqrt(this.calculateVariance(assetReturns[i]));
        const assetVolatilityJ = Math.sqrt(this.calculateVariance(assetReturns[j]));

        marginalVaR += portfolioWeights[j] * correlationMatrix[i][j] * assetVolatilityI * assetVolatilityJ;
      }

      marginalVaR = marginalVaR / portfolioVolatility * this.inverseNormalCDF(1 - confidenceLevel);

      const componentVaR = portfolioWeights[i] * marginalVaR;
      componentVaRs.push({
        asset: i,
        weight: portfolioWeights[i],
        componentVaR: componentVaR,
        contribution: componentVaR / portfolioVaR
      });
    }

    return {
      portfolioVaR,
      componentVaRs,
      totalContribution: componentVaRs.reduce((sum, comp) => sum + comp.contribution, 0)
    };
  }

  calculatePortfolioVariance(weights, assetReturns, correlationMatrix) {
    let portfolioVariance = 0;
    const n = weights.length;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const volI = Math.sqrt(this.calculateVariance(assetReturns[i]));
        const volJ = Math.sqrt(this.calculateVariance(assetReturns[j]));

        portfolioVariance += weights[i] * weights[j] * volI * volJ * correlationMatrix[i][j];
      }
    }

    return portfolioVariance;
  }

  calculateStressTestScenarios(baseReturns, scenarios) {
    if (!baseReturns || !scenarios) {
      throw new Error('Missing required data for stress testing');
    }

    const results = {};

    scenarios.forEach(scenario => {
      const { name, shocks } = scenario;
      const stressedReturns = baseReturns.map((returns, index) => {
        if (shocks[index]) {
          return returns.map(ret => ret + shocks[index]);
        }
        return returns;
      });

      const portfolioReturns = this.calculatePortfolioReturns(stressedReturns, scenario.weights || null);

      results[name] = {
        portfolioReturns,
        var95: this.calculateValueAtRisk(portfolioReturns, 0.95),
        expectedShortfall: this.calculateExpectedShortfall(portfolioReturns, 0.95),
        maxDrawdown: this.calculateMaxDrawdown(portfolioReturns.map((ret, i, arr) =>
          arr.slice(0, i + 1).reduce((prod, r) => prod * (1 + r), 1)
        ))
      };
    });

    return results;
  }

  calculatePortfolioReturns(assetReturns, weights = null) {
    if (!weights) {
      weights = new Array(assetReturns.length).fill(1 / assetReturns.length);
    }

    const minLength = Math.min(...assetReturns.map(returns => returns.length));
    const portfolioReturns = [];

    for (let t = 0; t < minLength; t++) {
      let portfolioReturn = 0;
      for (let i = 0; i < assetReturns.length; i++) {
        portfolioReturn += weights[i] * assetReturns[i][t];
      }
      portfolioReturns.push(portfolioReturn);
    }

    return portfolioReturns;
  }

  calculateValueAtRisk(returns, confidenceLevel = this.confidenceLevel) {
    if (!returns || returns.length < 30) {
      throw new Error('Insufficient data for VaR calculation');
    }

    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);

    return sortedReturns[index];
  }

  calculateMaxDrawdown(prices) {
    if (!prices || prices.length < 2) {
      throw new Error('Insufficient price data for drawdown calculation');
    }

    let maxDrawdown = 0;
    let peak = prices[0];

    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > peak) {
        peak = prices[i];
      }

      const drawdown = (peak - prices[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown;
  }

  calculateTailDependence(returns1, returns2, confidenceLevel = 0.95) {
    if (!returns1 || !returns2 || returns1.length !== returns2.length) {
      throw new Error('Invalid data for tail dependence calculation');
    }

    const n = returns1.length;
    const threshold = 1 - confidenceLevel;

    const var1 = this.calculateValueAtRisk(returns1, confidenceLevel);
    const var2 = this.calculateValueAtRisk(returns2, confidenceLevel);

    let jointExceedances = 0;
    let marginalExceedances1 = 0;
    let marginalExceedances2 = 0;

    for (let i = 0; i < n; i++) {
      const exceedance1 = returns1[i] <= var1;
      const exceedance2 = returns2[i] <= var2;

      if (exceedance1) marginalExceedances1++;
      if (exceedance2) marginalExceedances2++;
      if (exceedance1 && exceedance2) jointExceedances++;
    }

    const lowerTailDependence = jointExceedances / Math.min(marginalExceedances1, marginalExceedances2);

    return {
      lowerTailDependence,
      jointExceedances,
      marginalExceedances1,
      marginalExceedances2,
      threshold
    };
  }

  calculateAdvancedRiskMetrics(symbol, returns, marketReturns = null, benchmarkReturns = null) {
    const riskMetrics = new RiskMetrics({
      symbol: symbol,
      calculationDate: new Date(),
      confidenceLevel: this.confidenceLevel
    });

    riskMetrics.volatility = Math.sqrt(this.calculateVariance(returns) * this.tradingDaysPerYear);
    riskMetrics.valueAtRisk = this.calculateValueAtRisk(returns, this.confidenceLevel);
    riskMetrics.expectedShortfall = this.calculateExpectedShortfall(returns, this.confidenceLevel);

    try {
      const garchResults = this.calculateGarchVolatility(returns);
      riskMetrics.garchVolatility = garchResults.currentVolatility;
    } catch (error) {
      console.warn(`GARCH calculation failed for ${symbol}: ${error.message}`);
    }

    try {
      riskMetrics.cornishFisherVaR = this.calculateCornishFisherVaR(returns, this.confidenceLevel);
      riskMetrics.skewness = this.calculateSkewness(returns);
      riskMetrics.kurtosis = this.calculateKurtosis(returns);
    } catch (error) {
      console.warn(`Higher moment calculations failed for ${symbol}: ${error.message}`);
    }

    if (marketReturns && marketReturns.length === returns.length) {
      try {
        riskMetrics.beta = this.calculateBeta(returns, marketReturns);
        riskMetrics.tailDependence = this.calculateTailDependence(returns, marketReturns);
      } catch (error) {
        console.warn(`Market-related calculations failed for ${symbol}: ${error.message}`);
      }
    }

    if (benchmarkReturns && benchmarkReturns.length === returns.length) {
      const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
      const benchmarkMeanReturn = benchmarkReturns.reduce((sum, ret) => sum + ret, 0) / benchmarkReturns.length;

      riskMetrics.trackingError = Math.sqrt(this.calculateVariance(
        returns.map((ret, i) => ret - benchmarkReturns[i])
      ) * this.tradingDaysPerYear);

      riskMetrics.informationRatio = (meanReturn - benchmarkMeanReturn) * this.tradingDaysPerYear / riskMetrics.trackingError;
    }

    return riskMetrics;
  }

  calculateBeta(assetReturns, marketReturns) {
    if (!assetReturns || !marketReturns || assetReturns.length !== marketReturns.length) {
      throw new Error('Invalid data for beta calculation');
    }

    const covariance = this.calculateCovariance(assetReturns, marketReturns);
    const marketVariance = this.calculateVariance(marketReturns);

    return covariance / marketVariance;
  }

  calculateCovariance(returns1, returns2) {
    if (returns1.length !== returns2.length) {
      throw new Error('Return arrays must have the same length');
    }

    const mean1 = returns1.reduce((sum, ret) => sum + ret, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, ret) => sum + ret, 0) / returns2.length;

    const covariance = returns1.reduce((sum, ret, i) => {
      return sum + (ret - mean1) * (returns2[i] - mean2);
    }, 0) / (returns1.length - 1);

    return covariance;
  }
}

module.exports = AdvancedRiskCalculator;