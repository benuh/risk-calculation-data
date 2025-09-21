const StatisticalAnalyzer = require('./StatisticalAnalyzer');

/**
 * AnomalyDetectionAnalyzer - Advanced anomaly detection for financial risk modeling
 *
 * Focuses on detecting statistical outliers, regime changes, asymmetric risk patterns,
 * seasonal anomalies, and structural breaks in financial time series data
 */
class AnomalyDetectionAnalyzer {
  constructor() {
    this.analyzer = new StatisticalAnalyzer();
    this.significanceLevel = 0.05;
    this.outlierThreshold = 3; // Standard deviations
    this.regimeChangeThreshold = 0.1;
    this.minRegimeLength = 20; // Minimum observations for a regime
  }

  /**
   * Detect statistical outliers in return distributions
   */
  async detectStatisticalOutliers(assetData, methods = ['zscore', 'iqr', 'isolation', 'mahalanobis']) {
    const results = {
      timestamp: new Date(),
      analysis: 'statistical_outliers',
      assetOutliers: {},
      summary: {
        totalAssets: Object.keys(assetData).length,
        assetsWithOutliers: 0,
        totalOutliers: 0,
        severityDistribution: { low: 0, medium: 0, high: 0, extreme: 0 }
      }
    };

    for (const [asset, data] of Object.entries(assetData)) {
      try {
        const outlierAnalysis = await this.comprehensiveOutlierDetection(data, asset, methods);

        if (outlierAnalysis.totalOutliers > 0) {
          results.summary.assetsWithOutliers++;
          results.summary.totalOutliers += outlierAnalysis.totalOutliers;

          // Update severity distribution
          outlierAnalysis.outliers.forEach(outlier => {
            results.summary.severityDistribution[outlier.severity]++;
          });
        }

        results.assetOutliers[asset] = outlierAnalysis;

      } catch (error) {
        console.warn(`Outlier detection failed for ${asset}: ${error.message}`);
        results.assetOutliers[asset] = { error: error.message, totalOutliers: 0 };
      }
    }

    // Sort assets by outlier severity
    const sortedAssets = Object.entries(results.assetOutliers)
      .filter(([_, analysis]) => !analysis.error)
      .sort((a, b) => b[1].severityScore - a[1].severityScore);

    results.mostAnomalousAssets = sortedAssets.slice(0, 10);

    return results;
  }

  /**
   * Detect regime changes in volatility patterns
   */
  async detectVolatilityRegimeChanges(assetData, windowSize = 60) {
    const results = {
      timestamp: new Date(),
      analysis: 'volatility_regime_changes',
      regimeChanges: {},
      summary: {
        totalAssets: Object.keys(assetData).length,
        assetsWithRegimeChanges: 0,
        totalRegimeChanges: 0,
        averageRegimeDuration: 0
      }
    };

    const regimeDurations = [];

    for (const [asset, data] of Object.entries(assetData)) {
      try {
        const regimeAnalysis = await this.analyzeVolatilityRegimes(data, asset, windowSize);

        if (regimeAnalysis.regimeChanges.length > 0) {
          results.summary.assetsWithRegimeChanges++;
          results.summary.totalRegimeChanges += regimeAnalysis.regimeChanges.length;

          // Collect regime durations
          regimeAnalysis.regimes.forEach(regime => {
            if (regime.duration) {
              regimeDurations.push(regime.duration);
            }
          });
        }

        results.regimeChanges[asset] = regimeAnalysis;

      } catch (error) {
        console.warn(`Regime change detection failed for ${asset}: ${error.message}`);
        results.regimeChanges[asset] = { error: error.message };
      }
    }

    // Calculate average regime duration
    results.summary.averageRegimeDuration = regimeDurations.length > 0 ?
      regimeDurations.reduce((sum, duration) => sum + duration, 0) / regimeDurations.length : 0;

    return results;
  }

  /**
   * Analyze asymmetric risk patterns (up vs down markets)
   */
  async analyzeAsymmetricRiskPatterns(assetData, marketData = null) {
    const results = {
      timestamp: new Date(),
      analysis: 'asymmetric_risk_patterns',
      asymmetricPatterns: {},
      summary: {
        totalAssets: Object.keys(assetData).length,
        assetsWithAsymmetry: 0,
        averageDownsideRisk: 0,
        averageUpsideCapture: 0
      }
    };

    const downsideRisks = [];
    const upsideCaptures = [];

    for (const [asset, data] of Object.entries(assetData)) {
      try {
        const asymmetryAnalysis = await this.analyzeAsymmetricBehavior(
          data,
          asset,
          marketData ? marketData[asset] || marketData.market : null
        );

        if (asymmetryAnalysis.hasSignificantAsymmetry) {
          results.summary.assetsWithAsymmetry++;
        }

        if (asymmetryAnalysis.downsideRisk) {
          downsideRisks.push(asymmetryAnalysis.downsideRisk);
        }

        if (asymmetryAnalysis.upsideCapture) {
          upsideCaptures.push(asymmetryAnalysis.upsideCapture);
        }

        results.asymmetricPatterns[asset] = asymmetryAnalysis;

      } catch (error) {
        console.warn(`Asymmetric risk analysis failed for ${asset}: ${error.message}`);
        results.asymmetricPatterns[asset] = { error: error.message };
      }
    }

    // Calculate averages
    results.summary.averageDownsideRisk = downsideRisks.length > 0 ?
      downsideRisks.reduce((sum, risk) => sum + risk, 0) / downsideRisks.length : 0;

    results.summary.averageUpsideCapture = upsideCaptures.length > 0 ?
      upsideCaptures.reduce((sum, capture) => sum + capture, 0) / upsideCaptures.length : 0;

    return results;
  }

  /**
   * Detect seasonal anomalies in risk metrics
   */
  async detectSeasonalAnomalies(assetData, timeStamps = null) {
    const results = {
      timestamp: new Date(),
      analysis: 'seasonal_anomalies',
      seasonalPatterns: {},
      summary: {
        totalAssets: Object.keys(assetData).length,
        assetsWithSeasonality: 0,
        commonSeasonalPatterns: []
      }
    };

    for (const [asset, data] of Object.entries(assetData)) {
      try {
        const seasonalAnalysis = await this.analyzeSeasonalPatterns(data, asset, timeStamps);

        if (seasonalAnalysis.hasSignificantSeasonality) {
          results.summary.assetsWithSeasonality++;
        }

        results.seasonalPatterns[asset] = seasonalAnalysis;

      } catch (error) {
        console.warn(`Seasonal analysis failed for ${asset}: ${error.message}`);
        results.seasonalPatterns[asset] = { error: error.message };
      }
    }

    // Identify common seasonal patterns
    results.summary.commonSeasonalPatterns = this.identifyCommonSeasonalPatterns(
      results.seasonalPatterns
    );

    return results;
  }

  /**
   * Detect structural breaks in time series data
   */
  async detectStructuralBreaks(assetData, testTypes = ['cusum', 'chow', 'bai_perron']) {
    const results = {
      timestamp: new Date(),
      analysis: 'structural_breaks',
      structuralBreaks: {},
      summary: {
        totalAssets: Object.keys(assetData).length,
        assetsWithBreaks: 0,
        totalBreaks: 0,
        breakClusters: []
      }
    };

    const allBreakDates = [];

    for (const [asset, data] of Object.entries(assetData)) {
      try {
        const breakAnalysis = await this.analyzeStructuralBreaks(data, asset, testTypes);

        if (breakAnalysis.structuralBreaks.length > 0) {
          results.summary.assetsWithBreaks++;
          results.summary.totalBreaks += breakAnalysis.structuralBreaks.length;

          // Collect break dates for clustering analysis
          breakAnalysis.structuralBreaks.forEach(breakInfo => {
            if (breakInfo.breakDate) {
              allBreakDates.push(breakInfo.breakDate);
            }
          });
        }

        results.structuralBreaks[asset] = breakAnalysis;

      } catch (error) {
        console.warn(`Structural break detection failed for ${asset}: ${error.message}`);
        results.structuralBreaks[asset] = { error: error.message };
      }
    }

    // Identify break clusters (dates where multiple assets had breaks)
    results.summary.breakClusters = this.identifyBreakClusters(allBreakDates);

    return results;
  }

  /**
   * Comprehensive outlier detection using multiple methods
   */
  async comprehensiveOutlierDetection(data, asset, methods) {
    const returns = this.calculateReturns(data);
    const outliers = [];
    let severityScore = 0;

    const stats = this.analyzer.calculateDescriptiveStatistics(returns);

    // Z-Score method
    if (methods.includes('zscore')) {
      const zScoreOutliers = this.detectZScoreOutliers(returns, stats);
      outliers.push(...zScoreOutliers.map(o => ({ ...o, method: 'zscore' })));
    }

    // IQR method
    if (methods.includes('iqr')) {
      const iqrOutliers = this.detectIQROutliers(returns, stats);
      outliers.push(...iqrOutliers.map(o => ({ ...o, method: 'iqr' })));
    }

    // Isolation Forest approximation
    if (methods.includes('isolation')) {
      const isolationOutliers = this.detectIsolationOutliers(returns);
      outliers.push(...isolationOutliers.map(o => ({ ...o, method: 'isolation' })));
    }

    // Mahalanobis distance (simplified for univariate case)
    if (methods.includes('mahalanobis')) {
      const mahalanobisOutliers = this.detectMahalanobisOutliers(returns, stats);
      outliers.push(...mahalanobisOutliers.map(o => ({ ...o, method: 'mahalanobis' })));
    }

    // Deduplicate outliers by index
    const uniqueOutliers = this.deduplicateOutliers(outliers);

    // Calculate severity score
    severityScore = this.calculateSeverityScore(uniqueOutliers, returns.length);

    return {
      asset,
      totalOutliers: uniqueOutliers.length,
      outlierRate: uniqueOutliers.length / returns.length,
      outliers: uniqueOutliers,
      severityScore,
      distributionStats: stats,
      riskImplications: this.assessOutlierRiskImplications(uniqueOutliers, stats)
    };
  }

  /**
   * Analyze volatility regimes using multiple techniques
   */
  async analyzeVolatilityRegimes(data, asset, windowSize) {
    const returns = this.calculateReturns(data);
    const volatilities = this.calculateRollingVolatility(returns, windowSize);

    // Markov regime switching approximation
    const regimes = this.detectVolatilityRegimes(volatilities);

    // Regime change points
    const regimeChanges = this.identifyRegimeChangePoints(volatilities, regimes);

    // GARCH-based regime detection
    const garchRegimes = this.detectGARCHRegimes(returns);

    return {
      asset,
      volatilities,
      regimes,
      regimeChanges,
      garchRegimes,
      persistence: this.calculateRegimePersistence(regimes),
      riskImplications: this.assessRegimeRiskImplications(regimes, regimeChanges)
    };
  }

  /**
   * Analyze asymmetric behavior patterns
   */
  async analyzeAsymmetricBehavior(data, asset, marketData = null) {
    const returns = this.calculateReturns(data);
    const marketReturns = marketData ? this.calculateReturns(marketData) : null;

    // Separate up and down market periods
    const { upMarketReturns, downMarketReturns } = this.separateMarketDirections(
      returns,
      marketReturns
    );

    // Calculate asymmetric risk metrics
    const downsideDeviation = this.calculateDownsideDeviation(returns);
    const upsideCapture = this.calculateUpsideCapture(returns, marketReturns);
    const downsideCapture = this.calculateDownsideCapture(returns, marketReturns);

    // Beta asymmetry
    const upBeta = marketReturns && upMarketReturns.asset.length > 10 ?
      this.calculateBeta(upMarketReturns.asset, upMarketReturns.market) : null;
    const downBeta = marketReturns && downMarketReturns.asset.length > 10 ?
      this.calculateBeta(downMarketReturns.asset, downMarketReturns.market) : null;

    // Skewness and higher moments
    const skewness = this.analyzer.calculateSkewness(returns);
    const kurtosis = this.analyzer.calculateKurtosis(returns);

    const hasSignificantAsymmetry = this.testAsymmetrySignificance(
      upMarketReturns.asset,
      downMarketReturns.asset,
      skewness
    );

    return {
      asset,
      hasSignificantAsymmetry,
      downsideDeviation,
      downsideRisk: downsideDeviation / Math.sqrt(this.calculateVariance(returns)),
      upsideCapture,
      downsideCapture,
      upBeta,
      downBeta,
      betaAsymmetry: upBeta && downBeta ? Math.abs(upBeta - downBeta) : null,
      skewness,
      kurtosis,
      riskImplications: this.assessAsymmetricRiskImplications({
        downsideDeviation,
        upsideCapture,
        downsideCapture,
        skewness,
        kurtosis
      })
    };
  }

  /**
   * Analyze seasonal patterns in financial data
   */
  async analyzeSeasonalPatterns(data, asset, timeStamps = null) {
    const returns = this.calculateReturns(data);

    // Create synthetic timestamps if not provided
    const timestamps = timeStamps || this.createSyntheticTimestamps(returns.length);

    // Monthly seasonality
    const monthlyPatterns = this.analyzeMonthlySeasonality(returns, timestamps);

    // Day of week effects
    const weekdayPatterns = this.analyzeWeekdayEffects(returns, timestamps);

    // Quarter end effects
    const quarterEndEffects = this.analyzeQuarterEndEffects(returns, timestamps);

    // Holiday effects
    const holidayEffects = this.analyzeHolidayEffects(returns, timestamps);

    // Test for overall seasonality
    const seasonalityTest = this.testSeasonalitySignificance(monthlyPatterns, weekdayPatterns);

    return {
      asset,
      hasSignificantSeasonality: seasonalityTest.significant,
      monthlyPatterns,
      weekdayPatterns,
      quarterEndEffects,
      holidayEffects,
      seasonalityTest,
      riskImplications: this.assessSeasonalRiskImplications({
        monthlyPatterns,
        weekdayPatterns,
        quarterEndEffects
      })
    };
  }

  /**
   * Analyze structural breaks using multiple tests
   */
  async analyzeStructuralBreaks(data, asset, testTypes) {
    const returns = this.calculateReturns(data);
    const structuralBreaks = [];

    // CUSUM test
    if (testTypes.includes('cusum')) {
      const cusumBreaks = this.performCUSUMTest(returns);
      structuralBreaks.push(...cusumBreaks.map(b => ({ ...b, test: 'cusum' })));
    }

    // Chow test
    if (testTypes.includes('chow')) {
      const chowBreaks = this.performChowTest(returns);
      structuralBreaks.push(...chowBreaks.map(b => ({ ...b, test: 'chow' })));
    }

    // Bai-Perron test approximation
    if (testTypes.includes('bai_perron')) {
      const baiPerronBreaks = this.performBaiPerronTest(returns);
      structuralBreaks.push(...baiPerronBreaks.map(b => ({ ...b, test: 'bai_perron' })));
    }

    // Variance change detection
    const varianceBreaks = this.detectVarianceBreaks(returns);
    structuralBreaks.push(...varianceBreaks.map(b => ({ ...b, test: 'variance' })));

    return {
      asset,
      structuralBreaks,
      hasStructuralBreaks: structuralBreaks.length > 0,
      breakSummary: this.summarizeStructuralBreaks(structuralBreaks),
      riskImplications: this.assessStructuralBreakRiskImplications(structuralBreaks)
    };
  }

  // Helper methods for outlier detection
  detectZScoreOutliers(returns, stats) {
    const outliers = [];
    const threshold = this.outlierThreshold;

    returns.forEach((value, index) => {
      const zScore = Math.abs((value - stats.mean) / stats.standardDeviation);
      if (zScore > threshold) {
        outliers.push({
          index,
          value,
          zScore,
          severity: this.classifyOutlierSeverity(zScore, 'zscore')
        });
      }
    });

    return outliers;
  }

  detectIQROutliers(returns, stats) {
    const outliers = [];
    const lowerBound = stats.q1 - 1.5 * stats.iqr;
    const upperBound = stats.q3 + 1.5 * stats.iqr;

    returns.forEach((value, index) => {
      if (value < lowerBound || value > upperBound) {
        const severity = value < stats.q1 - 3 * stats.iqr || value > stats.q3 + 3 * stats.iqr ?
          'extreme' : 'high';

        outliers.push({
          index,
          value,
          deviation: Math.min(Math.abs(value - lowerBound), Math.abs(value - upperBound)),
          severity
        });
      }
    });

    return outliers;
  }

  detectIsolationOutliers(returns) {
    // Simplified isolation forest approximation
    const outliers = [];
    const sampleSize = Math.min(returns.length, 100);

    // Calculate isolation scores
    returns.forEach((value, index) => {
      const isolationScore = this.calculateIsolationScore(value, returns, sampleSize);
      if (isolationScore > 0.6) { // Threshold for outliers
        outliers.push({
          index,
          value,
          isolationScore,
          severity: this.classifyOutlierSeverity(isolationScore, 'isolation')
        });
      }
    });

    return outliers;
  }

  detectMahalanobisOutliers(returns, stats) {
    // For univariate case, Mahalanobis distance equals z-score
    return this.detectZScoreOutliers(returns, stats);
  }

  // Helper methods for regime detection
  calculateRollingVolatility(returns, windowSize) {
    const volatilities = [];

    for (let i = windowSize - 1; i < returns.length; i++) {
      const window = returns.slice(i - windowSize + 1, i + 1);
      const variance = this.calculateVariance(window);
      volatilities.push(Math.sqrt(variance * 252)); // Annualized
    }

    return volatilities;
  }

  detectVolatilityRegimes(volatilities) {
    // Simple regime detection using k-means approximation
    const lowVolThreshold = this.calculatePercentile(volatilities, 33);
    const highVolThreshold = this.calculatePercentile(volatilities, 67);

    const regimes = volatilities.map((vol, index) => {
      let regime;
      if (vol <= lowVolThreshold) regime = 'low';
      else if (vol >= highVolThreshold) regime = 'high';
      else regime = 'medium';

      return { index, volatility: vol, regime };
    });

    return regimes;
  }

  identifyRegimeChangePoints(volatilities, regimes) {
    const changePoints = [];

    for (let i = 1; i < regimes.length; i++) {
      if (regimes[i].regime !== regimes[i - 1].regime) {
        // Test for significance of regime change
        const before = volatilities.slice(Math.max(0, i - 20), i);
        const after = volatilities.slice(i, Math.min(volatilities.length, i + 20));

        if (before.length >= 5 && after.length >= 5) {
          const tTest = this.performTTest(before, after);

          if (tTest.significant) {
            changePoints.push({
              index: i,
              fromRegime: regimes[i - 1].regime,
              toRegime: regimes[i].regime,
              significance: tTest.pValue,
              changePoint: true
            });
          }
        }
      }
    }

    return changePoints;
  }

  // Helper methods for asymmetric analysis
  separateMarketDirections(returns, marketReturns) {
    const upMarketReturns = { asset: [], market: [] };
    const downMarketReturns = { asset: [], market: [] };

    if (!marketReturns) {
      // If no market data, use own returns to define up/down periods
      returns.forEach((ret, index) => {
        if (ret >= 0) {
          upMarketReturns.asset.push(ret);
        } else {
          downMarketReturns.asset.push(ret);
        }
      });
    } else {
      // Use market returns to define periods
      const minLength = Math.min(returns.length, marketReturns.length);

      for (let i = 0; i < minLength; i++) {
        if (marketReturns[i] >= 0) {
          upMarketReturns.asset.push(returns[i]);
          upMarketReturns.market.push(marketReturns[i]);
        } else {
          downMarketReturns.asset.push(returns[i]);
          downMarketReturns.market.push(marketReturns[i]);
        }
      }
    }

    return { upMarketReturns, downMarketReturns };
  }

  calculateDownsideDeviation(returns, target = 0) {
    const downsideReturns = returns.filter(ret => ret < target);
    if (downsideReturns.length === 0) return 0;

    const downsideVariance = downsideReturns.reduce((sum, ret) => {
      return sum + Math.pow(Math.min(0, ret - target), 2);
    }, 0) / downsideReturns.length;

    return Math.sqrt(downsideVariance);
  }

  calculateUpsideCapture(assetReturns, marketReturns) {
    if (!marketReturns || marketReturns.length !== assetReturns.length) {
      return null;
    }

    const upMarketPeriods = [];
    const assetUpReturns = [];

    for (let i = 0; i < marketReturns.length; i++) {
      if (marketReturns[i] > 0) {
        upMarketPeriods.push(marketReturns[i]);
        assetUpReturns.push(assetReturns[i]);
      }
    }

    if (upMarketPeriods.length < 10) return null;

    const marketUpAvg = upMarketPeriods.reduce((sum, ret) => sum + ret, 0) / upMarketPeriods.length;
    const assetUpAvg = assetUpReturns.reduce((sum, ret) => sum + ret, 0) / assetUpReturns.length;

    return assetUpAvg / marketUpAvg;
  }

  calculateDownsideCapture(assetReturns, marketReturns) {
    if (!marketReturns || marketReturns.length !== assetReturns.length) {
      return null;
    }

    const downMarketPeriods = [];
    const assetDownReturns = [];

    for (let i = 0; i < marketReturns.length; i++) {
      if (marketReturns[i] < 0) {
        downMarketPeriods.push(marketReturns[i]);
        assetDownReturns.push(assetReturns[i]);
      }
    }

    if (downMarketPeriods.length < 10) return null;

    const marketDownAvg = downMarketPeriods.reduce((sum, ret) => sum + ret, 0) / downMarketPeriods.length;
    const assetDownAvg = assetDownReturns.reduce((sum, ret) => sum + ret, 0) / assetDownReturns.length;

    return assetDownAvg / marketDownAvg;
  }

  // Additional helper methods
  calculateReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
  }

  calculateVariance(data) {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (data.length - 1);
  }

  calculateBeta(assetReturns, marketReturns) {
    if (assetReturns.length !== marketReturns.length || assetReturns.length < 10) {
      return null;
    }

    const covariance = this.calculateCovariance(assetReturns, marketReturns);
    const marketVariance = this.calculateVariance(marketReturns);

    return marketVariance !== 0 ? covariance / marketVariance : null;
  }

  calculateCovariance(returns1, returns2) {
    const mean1 = returns1.reduce((sum, ret) => sum + ret, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, ret) => sum + ret, 0) / returns2.length;

    return returns1.reduce((sum, ret, i) => {
      return sum + (ret - mean1) * (returns2[i] - mean2);
    }, 0) / (returns1.length - 1);
  }

  calculatePercentile(data, percentile) {
    const sorted = [...data].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;

    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  performTTest(sample1, sample2) {
    const n1 = sample1.length;
    const n2 = sample2.length;

    const mean1 = sample1.reduce((sum, val) => sum + val, 0) / n1;
    const mean2 = sample2.reduce((sum, val) => sum + val, 0) / n2;

    const var1 = sample1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
    const var2 = sample2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);

    const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    const standardError = Math.sqrt(pooledVar * (1/n1 + 1/n2));

    const tStatistic = (mean1 - mean2) / standardError;
    const df = n1 + n2 - 2;

    const pValue = 2 * (1 - this.tCDF(Math.abs(tStatistic), df));

    return {
      tStatistic,
      pValue,
      significant: pValue < this.significanceLevel,
      degreesOfFreedom: df
    };
  }

  tCDF(t, df) {
    if (df > 30) {
      return this.normalCDF(t);
    }
    return 0.5 + Math.atan(t / Math.sqrt(df)) / Math.PI;
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

  // Classification and assessment methods
  classifyOutlierSeverity(score, method) {
    let thresholds;

    switch (method) {
      case 'zscore':
        thresholds = { low: 2, medium: 3, high: 4, extreme: 5 };
        break;
      case 'isolation':
        thresholds = { low: 0.5, medium: 0.6, high: 0.7, extreme: 0.8 };
        break;
      default:
        thresholds = { low: 1, medium: 2, high: 3, extreme: 4 };
    }

    if (score >= thresholds.extreme) return 'extreme';
    if (score >= thresholds.high) return 'high';
    if (score >= thresholds.medium) return 'medium';
    return 'low';
  }

  deduplicateOutliers(outliers) {
    const uniqueOutliers = new Map();

    outliers.forEach(outlier => {
      const existing = uniqueOutliers.get(outlier.index);
      if (!existing || this.compareSeverity(outlier.severity, existing.severity) > 0) {
        uniqueOutliers.set(outlier.index, outlier);
      }
    });

    return Array.from(uniqueOutliers.values()).sort((a, b) => a.index - b.index);
  }

  compareSeverity(severity1, severity2) {
    const severityOrder = { low: 1, medium: 2, high: 3, extreme: 4 };
    return severityOrder[severity1] - severityOrder[severity2];
  }

  calculateSeverityScore(outliers, totalObservations) {
    const severityWeights = { low: 1, medium: 2, high: 3, extreme: 4 };
    const totalSeverity = outliers.reduce((sum, outlier) => {
      return sum + severityWeights[outlier.severity];
    }, 0);

    return totalSeverity / totalObservations;
  }

  // Risk implication assessment methods
  assessOutlierRiskImplications(outliers, stats) {
    const implications = {
      riskLevel: 'Low',
      recommendations: [],
      tailRiskConcerns: false,
      modelRiskIssues: false
    };

    const extremeOutliers = outliers.filter(o => o.severity === 'extreme').length;
    const outlierRate = outliers.length / (outliers.length + 100); // Approximate total observations

    if (extremeOutliers > 0) {
      implications.riskLevel = 'High';
      implications.tailRiskConcerns = true;
      implications.recommendations.push('Investigate extreme outliers for data quality issues');
      implications.recommendations.push('Consider robust risk measures less sensitive to outliers');
    }

    if (outlierRate > 0.05) {
      implications.riskLevel = implications.riskLevel === 'High' ? 'High' : 'Medium';
      implications.modelRiskIssues = true;
      implications.recommendations.push('High outlier rate may indicate model misspecification');
    }

    if (Math.abs(stats.skewness) > 1) {
      implications.recommendations.push('Significant skewness suggests non-normal distribution');
    }

    return implications;
  }

  // Placeholder methods for complex implementations
  calculateIsolationScore(value, data, sampleSize) {
    // Simplified isolation score calculation
    const sample = data.slice(0, sampleSize).sort((a, b) => a - b);
    const position = sample.findIndex(x => x >= value);
    const isolation = Math.abs(position - sampleSize / 2) / (sampleSize / 2);
    return isolation;
  }

  detectGARCHRegimes(returns) {
    // Placeholder for GARCH regime detection
    return { regimes: [], detected: false };
  }

  calculateRegimePersistence(regimes) {
    // Calculate average regime duration
    const regimeDurations = [];
    let currentRegime = regimes[0]?.regime;
    let duration = 1;

    for (let i = 1; i < regimes.length; i++) {
      if (regimes[i].regime === currentRegime) {
        duration++;
      } else {
        regimeDurations.push(duration);
        currentRegime = regimes[i].regime;
        duration = 1;
      }
    }

    regimeDurations.push(duration);

    return {
      averageDuration: regimeDurations.reduce((sum, d) => sum + d, 0) / regimeDurations.length,
      regimeDurations
    };
  }

  assessRegimeRiskImplications(regimes, regimeChanges) {
    return {
      riskLevel: regimeChanges.length > 5 ? 'High' : 'Medium',
      recommendations: ['Monitor for regime changes in risk management'],
      volatilityClustering: true
    };
  }

  testAsymmetrySignificance(upReturns, downReturns, skewness) {
    return Math.abs(skewness) > 0.5 || upReturns.length !== downReturns.length;
  }

  assessAsymmetricRiskImplications(metrics) {
    return {
      riskLevel: Math.abs(metrics.skewness) > 1 ? 'High' : 'Medium',
      recommendations: ['Consider asymmetric risk measures'],
      downsideRiskElevated: metrics.downsideDeviation > 0.02
    };
  }

  // Seasonal analysis methods (simplified implementations)
  createSyntheticTimestamps(length) {
    const timestamps = [];
    const startDate = new Date(2020, 0, 1);

    for (let i = 0; i < length; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      timestamps.push(date);
    }

    return timestamps;
  }

  analyzeMonthlySeasonality(returns, timestamps) {
    const monthlyReturns = Array(12).fill().map(() => []);

    returns.forEach((ret, index) => {
      if (timestamps[index]) {
        const month = timestamps[index].getMonth();
        monthlyReturns[month].push(ret);
      }
    });

    return monthlyReturns.map((monthReturns, month) => ({
      month: month + 1,
      averageReturn: monthReturns.length > 0 ?
        monthReturns.reduce((sum, ret) => sum + ret, 0) / monthReturns.length : 0,
      volatility: monthReturns.length > 1 ? Math.sqrt(this.calculateVariance(monthReturns)) : 0,
      observations: monthReturns.length
    }));
  }

  analyzeWeekdayEffects(returns, timestamps) {
    const weekdayReturns = Array(7).fill().map(() => []);

    returns.forEach((ret, index) => {
      if (timestamps[index]) {
        const weekday = timestamps[index].getDay();
        weekdayReturns[weekday].push(ret);
      }
    });

    return weekdayReturns.map((dayReturns, day) => ({
      day,
      averageReturn: dayReturns.length > 0 ?
        dayReturns.reduce((sum, ret) => sum + ret, 0) / dayReturns.length : 0,
      volatility: dayReturns.length > 1 ? Math.sqrt(this.calculateVariance(dayReturns)) : 0,
      observations: dayReturns.length
    }));
  }

  analyzeQuarterEndEffects(returns, timestamps) {
    // Simplified quarter-end effect analysis
    return { hasQuarterEndEffect: false, effect: 0 };
  }

  analyzeHolidayEffects(returns, timestamps) {
    // Simplified holiday effect analysis
    return { hasHolidayEffect: false, effect: 0 };
  }

  testSeasonalitySignificance(monthlyPatterns, weekdayPatterns) {
    // Simple ANOVA-like test for seasonal effects
    const monthlyReturns = monthlyPatterns.map(m => m.averageReturn);
    const weekdayReturns = weekdayPatterns.map(w => w.averageReturn);

    const monthlyVariance = this.calculateVariance(monthlyReturns);
    const weekdayVariance = this.calculateVariance(weekdayReturns);

    return {
      significant: monthlyVariance > 0.001 || weekdayVariance > 0.001,
      monthlyVariance,
      weekdayVariance
    };
  }

  assessSeasonalRiskImplications(patterns) {
    return {
      riskLevel: 'Low',
      recommendations: ['Monitor seasonal patterns in risk management'],
      seasonalAdjustmentNeeded: false
    };
  }

  // Structural break detection methods (simplified)
  performCUSUMTest(returns) {
    // Simplified CUSUM test
    const breaks = [];
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    let cumSum = 0;
    const threshold = 3 * Math.sqrt(this.calculateVariance(returns));

    returns.forEach((ret, index) => {
      cumSum += ret - mean;
      if (Math.abs(cumSum) > threshold && index > 20 && index < returns.length - 20) {
        breaks.push({
          breakDate: index,
          statistic: Math.abs(cumSum),
          significant: true
        });
      }
    });

    return breaks;
  }

  performChowTest(returns) {
    // Simplified Chow test implementation
    return [];
  }

  performBaiPerronTest(returns) {
    // Simplified Bai-Perron test implementation
    return [];
  }

  detectVarianceBreaks(returns) {
    // Simplified variance break detection
    return [];
  }

  summarizeStructuralBreaks(breaks) {
    return {
      totalBreaks: breaks.length,
      significantBreaks: breaks.filter(b => b.significant).length,
      breakDates: breaks.map(b => b.breakDate)
    };
  }

  assessStructuralBreakRiskImplications(breaks) {
    return {
      riskLevel: breaks.length > 2 ? 'High' : 'Medium',
      recommendations: ['Monitor for structural changes in market conditions'],
      modelStabilityRisk: breaks.length > 0
    };
  }

  identifyCommonSeasonalPatterns(seasonalPatterns) {
    // Identify patterns common across multiple assets
    return [];
  }

  identifyBreakClusters(breakDates) {
    // Identify clusters of break dates across assets
    return [];
  }
}

module.exports = AnomalyDetectionAnalyzer;