class CorrelationAnalyzer {
  constructor() {
    this.significanceThreshold = 0.05;
    this.correlationThreshold = 0.3;
    this.minObservations = 50;
  }

  performUnsupervisedCorrelationAnalysis(datasets) {
    const results = {
      timestamp: new Date(),
      significantCorrelations: [],
      anomalousPatterns: [],
      regimeChanges: [],
      nonLinearRelationships: [],
      tailDependencies: [],
      structuralBreaks: []
    };

    for (let i = 0; i < datasets.length; i++) {
      for (let j = i + 1; j < datasets.length; j++) {
        const analysis = this.analyzePairwiseRelationship(datasets[i], datasets[j]);

        if (analysis.isSignificant) {
          results.significantCorrelations.push(analysis);
        }

        if (analysis.hasRegimeChange) {
          results.regimeChanges.push(analysis);
        }

        if (analysis.isNonLinear) {
          results.nonLinearRelationships.push(analysis);
        }

        if (analysis.hasTailDependence) {
          results.tailDependencies.push(analysis);
        }
      }
    }

    results.anomalousPatterns = this.detectAnomalousPatterns(datasets);
    results.structuralBreaks = this.detectStructuralBreaks(datasets);

    return results;
  }

  analyzePairwiseRelationship(data1, data2) {
    if (!data1.values || !data2.values || data1.values.length !== data2.values.length) {
      throw new Error('Invalid data for pairwise analysis');
    }

    const n = data1.values.length;
    if (n < this.minObservations) {
      return { isSignificant: false, reason: 'Insufficient observations' };
    }

    const pearsonCorr = this.calculatePearsonCorrelation(data1.values, data2.values);
    const spearmanCorr = this.calculateSpearmanCorrelation(data1.values, data2.values);
    const kendallTau = this.calculateKendallTau(data1.values, data2.values);

    const tailDependence = this.calculateTailDependence(data1.values, data2.values);
    const rollingCorrelations = this.calculateRollingCorrelations(data1.values, data2.values, 50);
    const regimeAnalysis = this.analyzeCorrelationRegimes(rollingCorrelations);

    const nonLinearTest = this.testNonLinearRelationship(data1.values, data2.values);

    return {
      asset1: data1.name || 'Asset1',
      asset2: data2.name || 'Asset2',
      pearsonCorrelation: pearsonCorr.correlation,
      pearsonPValue: pearsonCorr.pValue,
      spearmanCorrelation: spearmanCorr.correlation,
      spearmanPValue: spearmanCorr.pValue,
      kendallTau: kendallTau.correlation,
      kendallPValue: kendallTau.pValue,
      isSignificant: Math.abs(pearsonCorr.correlation) > this.correlationThreshold &&
                     pearsonCorr.pValue < this.significanceThreshold,
      tailDependence: tailDependence,
      hasTailDependence: tailDependence.upperTail > 0.2 || tailDependence.lowerTail > 0.2,
      rollingCorrelations: rollingCorrelations,
      regimeAnalysis: regimeAnalysis,
      hasRegimeChange: regimeAnalysis.hasSignificantChange,
      nonLinearTest: nonLinearTest,
      isNonLinear: nonLinearTest.isSignificant,
      observationCount: n
    };
  }

  calculatePearsonCorrelation(x, y) {
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

    const tStatistic = correlation * Math.sqrt((n - 2) / (1 - correlation * correlation));
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStatistic), n - 2));

    return { correlation, pValue, tStatistic };
  }

  calculateSpearmanCorrelation(x, y) {
    const ranksX = this.calculateRanks(x);
    const ranksY = this.calculateRanks(y);
    return this.calculatePearsonCorrelation(ranksX, ranksY);
  }

  calculateKendallTau(x, y) {
    const n = x.length;
    let concordant = 0;
    let discordant = 0;
    let ties = 0;

    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const signX = Math.sign(x[j] - x[i]);
        const signY = Math.sign(y[j] - y[i]);

        if (signX === 0 || signY === 0) {
          ties++;
        } else if (signX * signY > 0) {
          concordant++;
        } else {
          discordant++;
        }
      }
    }

    const totalPairs = n * (n - 1) / 2;
    const tau = (concordant - discordant) / (totalPairs - ties);

    const variance = (2 * (2 * n + 5)) / (9 * n * (n - 1));
    const zScore = tau / Math.sqrt(variance);
    const pValue = 2 * (1 - this.normalCDF(Math.abs(zScore)));

    return { correlation: tau, pValue, zScore };
  }

  calculateTailDependence(x, y, quantile = 0.95) {
    const n = x.length;
    const thresholdX = this.calculateQuantile(x, quantile);
    const thresholdY = this.calculateQuantile(y, quantile);

    let upperJoint = 0;
    let upperMarginalX = 0;
    let upperMarginalY = 0;

    const lowerThresholdX = this.calculateQuantile(x, 1 - quantile);
    const lowerThresholdY = this.calculateQuantile(y, 1 - quantile);

    let lowerJoint = 0;
    let lowerMarginalX = 0;
    let lowerMarginalY = 0;

    for (let i = 0; i < n; i++) {
      if (x[i] > thresholdX && y[i] > thresholdY) upperJoint++;
      if (x[i] > thresholdX) upperMarginalX++;
      if (y[i] > thresholdY) upperMarginalY++;

      if (x[i] < lowerThresholdX && y[i] < lowerThresholdY) lowerJoint++;
      if (x[i] < lowerThresholdX) lowerMarginalX++;
      if (y[i] < lowerThresholdY) lowerMarginalY++;
    }

    const upperTail = upperJoint / Math.min(upperMarginalX, upperMarginalY);
    const lowerTail = lowerJoint / Math.min(lowerMarginalX, lowerMarginalY);

    return {
      upperTail: isNaN(upperTail) ? 0 : upperTail,
      lowerTail: isNaN(lowerTail) ? 0 : lowerTail,
      upperJointEvents: upperJoint,
      lowerJointEvents: lowerJoint,
      quantile: quantile
    };
  }

  calculateRollingCorrelations(x, y, window) {
    const rollingCorrs = [];

    for (let i = window; i <= x.length; i++) {
      const windowX = x.slice(i - window, i);
      const windowY = y.slice(i - window, i);

      const corr = this.calculatePearsonCorrelation(windowX, windowY);
      rollingCorrs.push({
        index: i,
        correlation: corr.correlation,
        pValue: corr.pValue
      });
    }

    return rollingCorrs;
  }

  analyzeCorrelationRegimes(rollingCorrelations) {
    if (rollingCorrelations.length < 10) {
      return { hasSignificantChange: false, regimes: [] };
    }

    const correlations = rollingCorrelations.map(r => r.correlation);
    const mean = correlations.reduce((sum, val) => sum + val, 0) / correlations.length;
    const std = Math.sqrt(correlations.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / correlations.length);

    const changePoints = [];
    const threshold = 2 * std;

    for (let i = 1; i < correlations.length; i++) {
      if (Math.abs(correlations[i] - correlations[i-1]) > threshold) {
        changePoints.push({
          index: i,
          fromCorrelation: correlations[i-1],
          toCorrelation: correlations[i],
          change: correlations[i] - correlations[i-1]
        });
      }
    }

    const regimes = this.identifyRegimes(correlations, changePoints);

    return {
      hasSignificantChange: changePoints.length > 0,
      changePoints: changePoints,
      regimes: regimes,
      volatility: std,
      meanCorrelation: mean
    };
  }

  identifyRegimes(correlations, changePoints) {
    if (changePoints.length === 0) {
      return [{
        start: 0,
        end: correlations.length - 1,
        meanCorrelation: correlations.reduce((sum, val) => sum + val, 0) / correlations.length,
        volatility: Math.sqrt(correlations.reduce((sum, val) => {
          const mean = correlations.reduce((s, v) => s + v, 0) / correlations.length;
          return sum + Math.pow(val - mean, 2);
        }, 0) / correlations.length)
      }];
    }

    const regimes = [];
    let start = 0;

    for (let i = 0; i < changePoints.length; i++) {
      const end = changePoints[i].index;
      const regimeData = correlations.slice(start, end);

      if (regimeData.length > 0) {
        const mean = regimeData.reduce((sum, val) => sum + val, 0) / regimeData.length;
        const variance = regimeData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / regimeData.length;

        regimes.push({
          start: start,
          end: end,
          meanCorrelation: mean,
          volatility: Math.sqrt(variance),
          length: regimeData.length
        });
      }

      start = end;
    }

    const finalRegimeData = correlations.slice(start);
    if (finalRegimeData.length > 0) {
      const mean = finalRegimeData.reduce((sum, val) => sum + val, 0) / finalRegimeData.length;
      const variance = finalRegimeData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / finalRegimeData.length;

      regimes.push({
        start: start,
        end: correlations.length - 1,
        meanCorrelation: mean,
        volatility: Math.sqrt(variance),
        length: finalRegimeData.length
      });
    }

    return regimes;
  }

  testNonLinearRelationship(x, y) {
    const pearsonCorr = this.calculatePearsonCorrelation(x, y);
    const spearmanCorr = this.calculateSpearmanCorrelation(x, y);

    const difference = Math.abs(spearmanCorr.correlation - pearsonCorr.correlation);
    const isSignificant = difference > 0.2;

    const mutualInformation = this.calculateMutualInformation(x, y);

    return {
      pearsonCorrelation: pearsonCorr.correlation,
      spearmanCorrelation: spearmanCorr.correlation,
      difference: difference,
      isSignificant: isSignificant,
      mutualInformation: mutualInformation,
      interpretation: isSignificant ?
        'Significant non-linear relationship detected' :
        'Relationship appears to be linear'
    };
  }

  calculateMutualInformation(x, y, bins = 10) {
    const n = x.length;

    const xBins = this.createBins(x, bins);
    const yBins = this.createBins(y, bins);

    const jointCounts = new Array(bins).fill(null).map(() => new Array(bins).fill(0));
    const xCounts = new Array(bins).fill(0);
    const yCounts = new Array(bins).fill(0);

    for (let i = 0; i < n; i++) {
      const xBin = this.findBin(x[i], xBins);
      const yBin = this.findBin(y[i], yBins);

      if (xBin >= 0 && xBin < bins && yBin >= 0 && yBin < bins) {
        jointCounts[xBin][yBin]++;
        xCounts[xBin]++;
        yCounts[yBin]++;
      }
    }

    let mutualInfo = 0;

    for (let i = 0; i < bins; i++) {
      for (let j = 0; j < bins; j++) {
        if (jointCounts[i][j] > 0 && xCounts[i] > 0 && yCounts[j] > 0) {
          const pxy = jointCounts[i][j] / n;
          const px = xCounts[i] / n;
          const py = yCounts[j] / n;

          mutualInfo += pxy * Math.log2(pxy / (px * py));
        }
      }
    }

    return mutualInfo;
  }

  detectAnomalousPatterns(datasets) {
    const anomalies = [];

    datasets.forEach((dataset, index) => {
      if (!dataset.values || dataset.values.length < this.minObservations) return;

      const outliers = this.detectOutliers(dataset.values);
      const volatilityClusters = this.detectVolatilityClustering(dataset.values);
      const jumpPatterns = this.detectJumpDiffusion(dataset.values);
      const seasonalPatterns = this.detectSeasonalAnomalies(dataset.values, dataset.timestamps);

      if (outliers.length > 0 || volatilityClusters.isSignificant ||
          jumpPatterns.hasJumps || seasonalPatterns.hasAnomalies) {
        anomalies.push({
          dataset: dataset.name || `Dataset_${index}`,
          outliers: outliers,
          volatilityClusters: volatilityClusters,
          jumpPatterns: jumpPatterns,
          seasonalPatterns: seasonalPatterns
        });
      }
    });

    return anomalies;
  }

  detectOutliers(values, method = 'iqr') {
    const sortedValues = [...values].sort((a, b) => a - b);
    const n = sortedValues.length;

    const q1 = sortedValues[Math.floor(n * 0.25)];
    const q3 = sortedValues[Math.floor(n * 0.75)];
    const iqr = q3 - q1;

    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    const outliers = [];

    values.forEach((value, index) => {
      if (value < lowerBound || value > upperBound) {
        outliers.push({
          index: index,
          value: value,
          type: value < lowerBound ? 'lower' : 'upper',
          deviationFromBound: value < lowerBound ?
            Math.abs(value - lowerBound) : Math.abs(value - upperBound)
        });
      }
    });

    return outliers;
  }

  detectVolatilityClustering(values) {
    if (values.length < 10) return { isSignificant: false };

    const returns = [];
    for (let i = 1; i < values.length; i++) {
      returns.push((values[i] - values[i-1]) / values[i-1]);
    }

    const squaredReturns = returns.map(r => r * r);
    const autocorrelations = [];

    for (let lag = 1; lag <= Math.min(10, Math.floor(returns.length / 4)); lag++) {
      const corr = this.calculateLaggedCorrelation(squaredReturns, lag);
      autocorrelations.push({
        lag: lag,
        correlation: corr.correlation,
        pValue: corr.pValue,
        isSignificant: corr.pValue < 0.05
      });
    }

    const significantLags = autocorrelations.filter(ac => ac.isSignificant);

    return {
      isSignificant: significantLags.length > 0,
      autocorrelations: autocorrelations,
      significantLags: significantLags,
      interpretation: significantLags.length > 0 ?
        'Volatility clustering detected' : 'No significant volatility clustering'
    };
  }

  detectJumpDiffusion(values) {
    if (values.length < 20) return { hasJumps: false };

    const returns = [];
    for (let i = 1; i < values.length; i++) {
      returns.push((values[i] - values[i-1]) / values[i-1]);
    }

    const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / returns.length);

    const jumpThreshold = 3 * std;
    const jumps = [];

    returns.forEach((ret, index) => {
      if (Math.abs(ret - mean) > jumpThreshold) {
        jumps.push({
          index: index + 1,
          return: ret,
          deviation: Math.abs(ret - mean),
          type: ret > mean ? 'positive' : 'negative'
        });
      }
    });

    const jumpIntensity = jumps.length / returns.length;

    return {
      hasJumps: jumps.length > 0,
      jumps: jumps,
      jumpIntensity: jumpIntensity,
      threshold: jumpThreshold,
      interpretation: jumps.length > 0 ?
        `${jumps.length} potential jumps detected (${(jumpIntensity * 100).toFixed(2)}% of observations)` :
        'No significant jumps detected'
    };
  }

  detectSeasonalAnomalies(values, timestamps) {
    if (!timestamps || timestamps.length !== values.length || values.length < 52) {
      return { hasAnomalies: false, reason: 'Insufficient data or missing timestamps' };
    }

    const weeklyReturns = this.groupByWeek(values, timestamps);
    const monthlyReturns = this.groupByMonth(values, timestamps);

    const weeklyAnomaly = this.testSeasonalEffect(weeklyReturns);
    const monthlyAnomaly = this.testSeasonalEffect(monthlyReturns);

    return {
      hasAnomalies: weeklyAnomaly.isSignificant || monthlyAnomaly.isSignificant,
      weeklyEffect: weeklyAnomaly,
      monthlyEffect: monthlyAnomaly
    };
  }

  detectStructuralBreaks(datasets) {
    const breaks = [];

    datasets.forEach((dataset, index) => {
      if (!dataset.values || dataset.values.length < 50) return;

      const breakPoints = this.cuspSumTest(dataset.values);

      if (breakPoints.length > 0) {
        breaks.push({
          dataset: dataset.name || `Dataset_${index}`,
          breakPoints: breakPoints,
          interpretation: `${breakPoints.length} potential structural break(s) detected`
        });
      }
    });

    return breaks;
  }

  cuspSumTest(values) {
    const n = values.length;
    const mean = values.reduce((sum, val) => sum + val, 0) / n;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    const cusumStats = [];
    let cumulativeSum = 0;

    for (let i = 0; i < n; i++) {
      cumulativeSum += (values[i] - mean) / std;
      cusumStats.push({
        index: i,
        cusum: cumulativeSum,
        normalized: cumulativeSum / Math.sqrt(i + 1)
      });
    }

    const criticalValue = 1.36; // 5% significance level for CUSUM test
    const breakPoints = [];

    cusumStats.forEach((stat, index) => {
      if (Math.abs(stat.normalized) > criticalValue) {
        breakPoints.push({
          index: index,
          cusumValue: stat.cusum,
          normalizedValue: stat.normalized,
          isSignificant: true
        });
      }
    });

    return breakPoints;
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

  calculateQuantile(data, quantile) {
    const sorted = [...data].sort((a, b) => a - b);
    const index = quantile * (sorted.length - 1);

    if (index === Math.floor(index)) {
      return sorted[index];
    } else {
      const lower = sorted[Math.floor(index)];
      const upper = sorted[Math.ceil(index)];
      return lower + (upper - lower) * (index - Math.floor(index));
    }
  }

  calculateLaggedCorrelation(series, lag) {
    if (series.length <= lag) {
      return { correlation: 0, pValue: 1 };
    }

    const x = series.slice(0, -lag);
    const y = series.slice(lag);

    return this.calculatePearsonCorrelation(x, y);
  }

  groupByWeek(values, timestamps) {
    // Implementation would group data by week
    // Simplified for demonstration
    return {};
  }

  groupByMonth(values, timestamps) {
    // Implementation would group data by month
    // Simplified for demonstration
    return {};
  }

  testSeasonalEffect(groupedData) {
    // Implementation would test for seasonal effects
    // Simplified for demonstration
    return { isSignificant: false };
  }

  createBins(data, numBins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;

    const bins = [];
    for (let i = 0; i <= numBins; i++) {
      bins.push(min + i * binWidth);
    }

    return bins;
  }

  findBin(value, bins) {
    for (let i = 0; i < bins.length - 1; i++) {
      if (value >= bins[i] && value < bins[i + 1]) {
        return i;
      }
    }
    return bins.length - 2; // Last bin
  }

  tCDF(t, df) {
    if (df > 30) {
      return this.normalCDF(t);
    }

    return 0.5 + (t / Math.sqrt(df)) / (1 + (t * t) / df) / Math.PI;
  }

  normalCDF(z) {
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  erf(x) {
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
}

module.exports = CorrelationAnalyzer;