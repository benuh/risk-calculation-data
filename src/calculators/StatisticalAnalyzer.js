class StatisticalAnalyzer {
  constructor() {
    this.significanceLevel = 0.05;
    this.confidenceLevel = 0.95;
  }

  setSignificanceLevel(level) {
    this.significanceLevel = level;
    this.confidenceLevel = 1 - level;
  }

  calculateDescriptiveStatistics(data) {
    if (!data || data.length === 0) {
      throw new Error('Data array cannot be empty');
    }

    const n = data.length;
    const sortedData = [...data].sort((a, b) => a - b);

    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
    const standardDeviation = Math.sqrt(variance);

    const median = n % 2 === 0
      ? (sortedData[n/2 - 1] + sortedData[n/2]) / 2
      : sortedData[Math.floor(n/2)];

    const q1Index = Math.floor((n + 1) / 4);
    const q3Index = Math.floor(3 * (n + 1) / 4);
    const q1 = sortedData[q1Index - 1];
    const q3 = sortedData[q3Index - 1];
    const iqr = q3 - q1;

    const skewness = this.calculateSkewness(data);
    const kurtosis = this.calculateKurtosis(data);

    return {
      count: n,
      mean,
      median,
      mode: this.calculateMode(data),
      variance,
      standardDeviation,
      minimum: Math.min(...data),
      maximum: Math.max(...data),
      range: Math.max(...data) - Math.min(...data),
      q1,
      q3,
      iqr,
      skewness,
      kurtosis,
      standardError: standardDeviation / Math.sqrt(n)
    };
  }

  calculateMode(data) {
    const frequency = {};
    let maxFreq = 0;
    let modes = [];

    data.forEach(value => {
      frequency[value] = (frequency[value] || 0) + 1;
      if (frequency[value] > maxFreq) {
        maxFreq = frequency[value];
      }
    });

    for (const value in frequency) {
      if (frequency[value] === maxFreq) {
        modes.push(parseFloat(value));
      }
    }

    return modes.length === Object.keys(frequency).length ? null : modes;
  }

  calculateSkewness(data) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);

    const skewness = data.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / stdDev, 3);
    }, 0) / n;

    return skewness;
  }

  calculateKurtosis(data) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);

    const kurtosis = data.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / stdDev, 4);
    }, 0) / n;

    return kurtosis - 3; // Excess kurtosis
  }

  performNormalityTest(data) {
    if (!data || data.length < 8) {
      throw new Error('Insufficient data for normality test (minimum 8 observations)');
    }

    const jarqueBeraTest = this.jarqueBeraTest(data);
    const shapiroWilkTest = data.length <= 5000 ? this.shapiroWilkTest(data) : null;

    return {
      jarqueBera: jarqueBeraTest,
      shapiroWilk: shapiroWilkTest,
      recommendation: this.interpretNormalityResults(jarqueBeraTest, shapiroWilkTest)
    };
  }

  jarqueBeraTest(data) {
    const n = data.length;
    const skewness = this.calculateSkewness(data);
    const kurtosis = this.calculateKurtosis(data);

    const jbStatistic = (n / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis, 2) / 4);
    const criticalValue = 5.991; // Chi-square critical value with 2 df at 95% confidence
    const pValue = 1 - this.chiSquareCDF(jbStatistic, 2);

    return {
      statistic: jbStatistic,
      pValue,
      criticalValue,
      isNormal: pValue > this.significanceLevel,
      interpretation: pValue > this.significanceLevel ?
        'Data appears to be normally distributed' :
        'Data does not appear to be normally distributed'
    };
  }

  shapiroWilkTest(data) {
    if (data.length > 5000) {
      return null; // Too computationally intensive for large samples
    }

    const n = data.length;
    const sortedData = [...data].sort((a, b) => a - b);

    // Simplified Shapiro-Wilk implementation
    // Note: This is an approximation - full implementation requires extensive coefficient tables
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);

    let numerator = 0;
    for (let i = 0; i < Math.floor(n/2); i++) {
      const coeff = this.approximateShapiroWilkCoefficient(i, n);
      numerator += coeff * (sortedData[n-1-i] - sortedData[i]);
    }

    const wStatistic = Math.pow(numerator, 2) / ((n - 1) * variance);
    const isNormal = wStatistic > 0.90; // Simplified threshold

    return {
      statistic: wStatistic,
      isNormal,
      interpretation: isNormal ?
        'Data appears to be normally distributed' :
        'Data does not appear to be normally distributed'
    };
  }

  approximateShapiroWilkCoefficient(i, n) {
    // Simplified approximation of Shapiro-Wilk coefficients
    return Math.sqrt(2) * this.inverseNormalCDF((i + 0.5) / n) /
           Math.sqrt(n * Math.PI / 2);
  }

  chiSquareCDF(x, df) {
    // Simplified chi-square CDF implementation
    if (x <= 0) return 0;
    if (df === 1) return 2 * this.normalCDF(Math.sqrt(x)) - 1;
    if (df === 2) return 1 - Math.exp(-x / 2);

    // Approximation for other degrees of freedom
    const mean = df;
    const variance = 2 * df;
    const normalizedX = (x - mean) / Math.sqrt(variance);
    return this.normalCDF(normalizedX);
  }

  normalCDF(z) {
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  erf(x) {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  inverseNormalCDF(p) {
    if (p <= 0 || p >= 1) {
      throw new Error('Probability must be between 0 and 1');
    }

    // Beasley-Springer-Moro algorithm
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

    if (p < 0.5) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (((((a0*q+a1)*q+a2)*q+a3)*q+a4)*q+a5) / ((((b1*q+b2)*q+b3)*q+b4)*q+b5);
    } else {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((a0*q+a1)*q+a2)*q+a3)*q+a4)*q+a5) / ((((b1*q+b2)*q+b3)*q+b4)*q+b5);
    }
  }

  interpretNormalityResults(jarqueBera, shapiroWilk) {
    if (jarqueBera.isNormal && (shapiroWilk === null || shapiroWilk.isNormal)) {
      return {
        conclusion: 'normal',
        recommendation: 'Data appears normally distributed. Parametric tests are appropriate.',
        confidence: 'high'
      };
    } else if (!jarqueBera.isNormal && shapiroWilk && !shapiroWilk.isNormal) {
      return {
        conclusion: 'non-normal',
        recommendation: 'Data is not normally distributed. Consider non-parametric tests or data transformation.',
        confidence: 'high'
      };
    } else {
      return {
        conclusion: 'uncertain',
        recommendation: 'Mixed results. Consider additional normality tests or examine data distribution visually.',
        confidence: 'medium'
      };
    }
  }

  performCorrelationAnalysis(data1, data2, method = 'pearson') {
    if (!data1 || !data2 || data1.length !== data2.length) {
      throw new Error('Data arrays must have the same length and be non-empty');
    }

    if (data1.length < 3) {
      throw new Error('Insufficient data for correlation analysis (minimum 3 observations)');
    }

    let correlation, pValue, interpretation;

    switch (method.toLowerCase()) {
    case 'pearson':
      ({ correlation, pValue } = this.pearsonCorrelation(data1, data2));
      break;
    case 'spearman':
      ({ correlation, pValue } = this.spearmanCorrelation(data1, data2));
      break;
    case 'kendall':
      ({ correlation, pValue } = this.kendallCorrelation(data1, data2));
      break;
    default:
      throw new Error('Unsupported correlation method. Use: pearson, spearman, or kendall');
    }

    interpretation = this.interpretCorrelation(correlation, pValue);

    return {
      method,
      correlation,
      pValue,
      significance: pValue < this.significanceLevel,
      interpretation,
      strength: this.classifyCorrelationStrength(Math.abs(correlation))
    };
  }

  pearsonCorrelation(x, y) {
    const n = x.length;
    const meanX = x.reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < n; i++) {
      const deltaX = x[i] - meanX;
      const deltaY = y[i] - meanY;
      numerator += deltaX * deltaY;
      denomX += deltaX * deltaX;
      denomY += deltaY * deltaY;
    }

    const correlation = numerator / Math.sqrt(denomX * denomY);

    // Calculate t-statistic for significance test
    const tStatistic = correlation * Math.sqrt((n - 2) / (1 - correlation * correlation));
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStatistic), n - 2));

    return { correlation, pValue };
  }

  spearmanCorrelation(x, y) {
    const n = x.length;
    const ranksX = this.calculateRanks(x);
    const ranksY = this.calculateRanks(y);

    return this.pearsonCorrelation(ranksX, ranksY);
  }

  kendallCorrelation(x, y) {
    const n = x.length;
    let concordant = 0;
    let discordant = 0;

    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const signX = Math.sign(x[j] - x[i]);
        const signY = Math.sign(y[j] - y[i]);

        if (signX * signY > 0) {
          concordant++;
        } else if (signX * signY < 0) {
          discordant++;
        }
      }
    }

    const totalPairs = n * (n - 1) / 2;
    const correlation = (concordant - discordant) / totalPairs;

    // Approximate p-value calculation
    const variance = (2 * (2 * n + 5)) / (9 * n * (n - 1));
    const zScore = correlation / Math.sqrt(variance);
    const pValue = 2 * (1 - this.normalCDF(Math.abs(zScore)));

    return { correlation, pValue };
  }

  calculateRanks(data) {
    const indexed = data.map((value, index) => ({ value, index }));
    indexed.sort((a, b) => a.value - b.value);

    const ranks = new Array(data.length);
    let currentRank = 1;

    for (let i = 0; i < indexed.length; i++) {
      if (i > 0 && indexed[i].value !== indexed[i - 1].value) {
        currentRank = i + 1;
      }
      ranks[indexed[i].index] = currentRank;
    }

    return ranks;
  }

  tCDF(t, df) {
    // Simplified t-distribution CDF
    if (df > 30) {
      return this.normalCDF(t);
    }

    // Student's t-distribution approximation
    const x = t / Math.sqrt(df);
    return 0.5 + (x / (1 + x * x / df)) / Math.PI;
  }

  interpretCorrelation(correlation, pValue) {
    let interpretation = '';

    if (pValue >= this.significanceLevel) {
      interpretation = 'No significant correlation detected';
    } else {
      const direction = correlation > 0 ? 'positive' : 'negative';
      const strength = this.classifyCorrelationStrength(Math.abs(correlation));
      interpretation = `Significant ${direction} ${strength} correlation`;
    }

    return interpretation;
  }

  classifyCorrelationStrength(absCorrelation) {
    if (absCorrelation < 0.1) return 'negligible';
    if (absCorrelation < 0.3) return 'weak';
    if (absCorrelation < 0.5) return 'moderate';
    if (absCorrelation < 0.7) return 'strong';
    return 'very strong';
  }

  performTimeSeriesAnalysis(data, timestamps = null) {
    if (!data || data.length < 10) {
      throw new Error('Insufficient data for time series analysis (minimum 10 observations)');
    }

    const n = data.length;
    const basicStats = this.calculateDescriptiveStatistics(data);

    // Calculate first differences
    const differences = [];
    for (let i = 1; i < n; i++) {
      differences.push(data[i] - data[i - 1]);
    }

    // Augmented Dickey-Fuller test for stationarity
    const adfTest = this.augmentedDickeyFullerTest(data);

    // Autocorrelation analysis
    const autocorrelations = this.calculateAutocorrelations(data, Math.min(20, Math.floor(n / 4)));

    // Trend analysis
    const trendAnalysis = this.analyzeTrend(data, timestamps);

    return {
      basicStatistics: basicStats,
      stationarityTest: adfTest,
      autocorrelations,
      trendAnalysis,
      firstDifferences: {
        mean: differences.reduce((sum, val) => sum + val, 0) / differences.length,
        variance: this.calculateVariance(differences)
      },
      recommendations: this.generateTimeSeriesRecommendations(adfTest, autocorrelations)
    };
  }

  augmentedDickeyFullerTest(data) {
    // Simplified ADF test implementation
    const n = data.length;
    const y = data.slice(1);
    const x = data.slice(0, -1);

    // Calculate regression: Δy = α + βy_{t-1} + ε
    const differences = [];
    for (let i = 1; i < n; i++) {
      differences.push(data[i] - data[i - 1]);
    }

    const { slope, pValue } = this.simpleLinearRegression(x, differences);

    return {
      testStatistic: slope,
      pValue,
      isStationary: pValue < 0.05,
      interpretation: pValue < 0.05 ?
        'Series appears to be stationary' :
        'Series appears to have a unit root (non-stationary)'
    };
  }

  simpleLinearRegression(x, y) {
    const n = x.length;
    const meanX = x.reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      numerator += (x[i] - meanX) * (y[i] - meanY);
      denominator += (x[i] - meanX) * (x[i] - meanX);
    }

    const slope = numerator / denominator;
    const intercept = meanY - slope * meanX;

    // Calculate R-squared and standard error
    let ssRes = 0;
    let ssTot = 0;

    for (let i = 0; i < n; i++) {
      const predicted = intercept + slope * x[i];
      ssRes += (y[i] - predicted) * (y[i] - predicted);
      ssTot += (y[i] - meanY) * (y[i] - meanY);
    }

    const rSquared = 1 - (ssRes / ssTot);
    const standardError = Math.sqrt(ssRes / (n - 2));

    // Simplified p-value calculation
    const tStatistic = slope / (standardError / Math.sqrt(denominator));
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStatistic), n - 2));

    return {
      slope,
      intercept,
      rSquared,
      standardError,
      pValue
    };
  }

  calculateAutocorrelations(data, maxLag) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / n;

    const autocorrelations = [];

    for (let lag = 0; lag <= maxLag; lag++) {
      let covariance = 0;
      let count = 0;

      for (let i = lag; i < n; i++) {
        covariance += (data[i] - mean) * (data[i - lag] - mean);
        count++;
      }

      covariance /= count;
      const autocorr = covariance / variance;

      autocorrelations.push({
        lag,
        value: autocorr,
        significant: Math.abs(autocorr) > 1.96 / Math.sqrt(n)
      });
    }

    return autocorrelations;
  }

  analyzeTrend(data, timestamps = null) {
    const n = data.length;
    const x = timestamps || Array.from({ length: n }, (_, i) => i);

    const regression = this.simpleLinearRegression(x, data);

    const trendDirection = regression.slope > 0 ? 'increasing' :
      regression.slope < 0 ? 'decreasing' : 'flat';

    const trendStrength = Math.abs(regression.slope) * Math.sqrt(regression.rSquared);

    return {
      slope: regression.slope,
      rSquared: regression.rSquared,
      direction: trendDirection,
      strength: trendStrength,
      significant: regression.pValue < 0.05,
      interpretation: `${trendDirection} trend ${regression.pValue < 0.05 ? '(significant)' : '(not significant)'}`
    };
  }

  generateTimeSeriesRecommendations(stationarityTest, autocorrelations) {
    const recommendations = [];

    if (!stationarityTest.isStationary) {
      recommendations.push('Consider differencing the series to achieve stationarity');
      recommendations.push('Apply unit root tests after differencing');
    }

    const significantLags = autocorrelations.filter(ac => ac.lag > 0 && ac.significant);
    if (significantLags.length > 0) {
      recommendations.push(`Significant autocorrelations detected at lags: ${significantLags.map(ac => ac.lag).join(', ')}`);
      recommendations.push('Consider ARIMA modeling for forecasting');
    }

    if (stationarityTest.isStationary && significantLags.length === 0) {
      recommendations.push('Series appears to be white noise - suitable for simple forecasting methods');
    }

    return recommendations;
  }

  calculateVariance(data) {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (data.length - 1);
  }
}

module.exports = StatisticalAnalyzer;