const { RiskMetrics } = require('../models/FinancialData');

class EnhancedRiskCalculator {
  constructor(options = {}) {
    this.confidenceLevel = options.confidenceLevel || 0.95;
    this.riskFreeRate = options.riskFreeRate || 0.02;
    this.tradingDaysPerYear = options.tradingDaysPerYear || 252;
    this.minObservations = options.minObservations || 100;

    // Advanced model parameters
    this.regimeThreshold = options.regimeThreshold || 0.15;
    this.jumpThreshold = options.jumpThreshold || 3.0;
    this.volClusteringWindow = options.volClusteringWindow || 22;

    // Machine learning parameters
    this.ensembleSize = options.ensembleSize || 10;
    this.validationSplit = options.validationSplit || 0.2;
  }

  // Enhanced multi-factor risk model
  calculateMultiFactorRisk(returns, factors, factorNames = null) {
    if (!returns || !factors || returns.length !== factors.length) {
      throw new Error('Returns and factors must have the same length');
    }

    const n = returns.length;
    const numFactors = factors[0].length;

    if (!factorNames) {
      factorNames = Array.from({length: numFactors}, (_, i) => `Factor_${i+1}`);
    }

    // Construct design matrix with intercept
    const X = factors.map(row => [1, ...row]); // Add intercept term
    const y = returns;

    // Perform multiple regression using matrix operations
    const regression = this.multipleRegression(X, y);

    // Calculate factor exposures (betas)
    const factorExposures = regression.coefficients.slice(1); // Remove intercept
    const alpha = regression.coefficients[0]; // Intercept is alpha

    // Calculate systematic and idiosyncratic risk
    const systematicVariance = this.calculateSystematicVariance(factorExposures, factors);
    const idiosyncraticVariance = regression.residualVariance;
    const totalVariance = systematicVariance + idiosyncraticVariance;

    // Factor contribution to risk
    const factorContributions = this.calculateFactorRiskContributions(
      factorExposures, factors, factorNames
    );

    // Risk attribution
    const riskAttribution = {
      systematicRisk: Math.sqrt(systematicVariance),
      idiosyncraticRisk: Math.sqrt(idiosyncraticVariance),
      totalRisk: Math.sqrt(totalVariance),
      systematicProportion: systematicVariance / totalVariance,
      rSquared: regression.rSquared,
      alpha: alpha,
      factorExposures: factorExposures.map((beta, i) => ({
        factor: factorNames[i],
        exposure: beta,
        tStatistic: regression.tStatistics[i + 1],
        pValue: regression.pValues[i + 1],
        significance: regression.pValues[i + 1] < 0.05
      })),
      factorContributions: factorContributions
    };

    return riskAttribution;
  }

  // Advanced GARCH implementation with multiple variants
  calculateAdvancedGarch(returns, model = 'GARCH', order = [1, 1], distribution = 'normal') {
    if (returns.length < this.minObservations) {
      throw new Error(`Insufficient data for GARCH calculation (need at least ${this.minObservations})`);
    }

    const garchResults = {
      model: model,
      order: order,
      distribution: distribution,
      logLikelihood: 0,
      aic: 0,
      bic: 0,
      parameters: {},
      conditionalVariances: [],
      standardizedResiduals: [],
      volatilityForecasts: [],
      diagnostics: {}
    };

    switch(model.toLowerCase()) {
    case 'garch':
      return this.estimateGARCH(returns, order, garchResults);
    case 'egarch':
      return this.estimateEGARCH(returns, order, garchResults);
    case 'gjr-garch':
      return this.estimateGJRGARCH(returns, order, garchResults);
    case 'figarch':
      return this.estimateFIGARCH(returns, order, garchResults);
    default:
      throw new Error(`Unsupported GARCH model: ${model}`);
    }
  }

  estimateGARCH(returns, order, results) {
    const [p, q] = order;
    const n = returns.length;

    // Initial parameter estimates
    let omega = 0.01;
    let alpha = Array(q).fill(0.1);
    let beta = Array(p).fill(0.8);

    // Ensure stationarity constraint
    const constraintSum = alpha.reduce((sum, a) => sum + a, 0) +
                         beta.reduce((sum, b) => sum + b, 0);
    if (constraintSum >= 0.999) {
      const scale = 0.95 / constraintSum;
      alpha = alpha.map(a => a * scale);
      beta = beta.map(b => b * scale);
    }

    // Maximum likelihood estimation (simplified)
    const maxIterations = 1000;
    const tolerance = 1e-6;

    let variances = new Array(n);
    variances[0] = this.calculateVariance(returns);

    for (let iter = 0; iter < maxIterations; iter++) {
      // E-step: Calculate conditional variances
      for (let t = 1; t < n; t++) {
        let variance = omega;

        // ARCH terms
        for (let i = 1; i <= Math.min(q, t); i++) {
          variance += alpha[i-1] * Math.pow(returns[t-i], 2);
        }

        // GARCH terms
        for (let i = 1; i <= Math.min(p, t); i++) {
          variance += beta[i-1] * variances[t-i];
        }

        variances[t] = Math.max(variance, 1e-8); // Avoid negative variance
      }

      // M-step: Update parameters (simplified gradient descent)
      const gradients = this.calculateGARCHGradients(returns, variances, omega, alpha, beta);

      const learningRate = 0.001;
      omega = Math.max(omega - learningRate * gradients.omega, 1e-8);
      alpha = alpha.map((a, i) => Math.max(a - learningRate * gradients.alpha[i], 0));
      beta = beta.map((b, i) => Math.max(b - learningRate * gradients.beta[i], 0));

      // Check convergence
      if (gradients.norm < tolerance) break;
    }

    // Calculate log-likelihood
    const logLikelihood = this.calculateGARCHLogLikelihood(returns, variances);

    // Standardized residuals
    const standardizedResiduals = returns.map((ret, i) => ret / Math.sqrt(variances[i]));

    results.parameters = { omega, alpha, beta };
    results.conditionalVariances = variances;
    results.standardizedResiduals = standardizedResiduals;
    results.logLikelihood = logLikelihood;
    results.aic = -2 * logLikelihood + 2 * (1 + p + q);
    results.bic = -2 * logLikelihood + Math.log(n) * (1 + p + q);

    // Generate forecasts
    results.volatilityForecasts = this.generateGARCHForecasts(
      returns, variances, omega, alpha, beta, 10
    );

    // Model diagnostics
    results.diagnostics = this.calculateGARCHDiagnostics(standardizedResiduals);

    return results;
  }

  estimateEGARCH(returns, order, results) {
    // Exponential GARCH implementation
    // ln(σ²_t) = ω + Σβ_i ln(σ²_{t-i}) + Σα_i [θε_{t-i} + γ(|ε_{t-i}| - E[|ε_{t-i}|])]

    const [p, q] = order;
    const n = returns.length;

    // Parameters: omega, beta (p terms), alpha (q terms), theta (q terms), gamma (q terms)
    let omega = 0.01;
    let beta = Array(p).fill(0.9);
    let alpha = Array(q).fill(0.1);
    let theta = Array(q).fill(-0.1); // Leverage effect
    let gamma = Array(q).fill(0.1);

    const logVariances = new Array(n);
    logVariances[0] = Math.log(this.calculateVariance(returns));

    // Simplified estimation (in practice would use MLE)
    for (let t = 1; t < n; t++) {
      let logVar = omega;

      // GARCH terms
      for (let i = 1; i <= Math.min(p, t); i++) {
        logVar += beta[i-1] * logVariances[t-i];
      }

      // Asymmetric terms
      for (let i = 1; i <= Math.min(q, t); i++) {
        const epsilon = returns[t-i] / Math.sqrt(Math.exp(logVariances[t-i]));
        const expectedAbsEpsilon = Math.sqrt(2/Math.PI); // For normal distribution

        logVar += alpha[i-1] * (theta[i-1] * epsilon +
                               gamma[i-1] * (Math.abs(epsilon) - expectedAbsEpsilon));
      }

      logVariances[t] = logVar;
    }

    const variances = logVariances.map(logVar => Math.exp(logVar));
    const standardizedResiduals = returns.map((ret, i) => ret / Math.sqrt(variances[i]));

    results.parameters = { omega, beta, alpha, theta, gamma };
    results.conditionalVariances = variances;
    results.standardizedResiduals = standardizedResiduals;
    results.diagnostics = this.calculateGARCHDiagnostics(standardizedResiduals);

    return results;
  }

  estimateGJRGARCH(returns, order, results) {
    // GJR-GARCH (Glosten-Jagannathan-Runkle) implementation
    // σ²_t = ω + Σα_i ε²_{t-i} + Σγ_i I_{t-i} ε²_{t-i} + Σβ_i σ²_{t-i}
    // where I_{t-i} = 1 if ε_{t-i} < 0, 0 otherwise

    const [p, q] = order;
    const n = returns.length;

    let omega = 0.01;
    let alpha = Array(q).fill(0.05);
    let gamma = Array(q).fill(0.05); // Leverage effect
    let beta = Array(p).fill(0.85);

    const variances = new Array(n);
    variances[0] = this.calculateVariance(returns);

    for (let t = 1; t < n; t++) {
      let variance = omega;

      // ARCH terms
      for (let i = 1; i <= Math.min(q, t); i++) {
        const epsilon = returns[t-i];
        const leverageIndicator = epsilon < 0 ? 1 : 0;
        variance += (alpha[i-1] + gamma[i-1] * leverageIndicator) * Math.pow(epsilon, 2);
      }

      // GARCH terms
      for (let i = 1; i <= Math.min(p, t); i++) {
        variance += beta[i-1] * variances[t-i];
      }

      variances[t] = Math.max(variance, 1e-8);
    }

    const standardizedResiduals = returns.map((ret, i) => ret / Math.sqrt(variances[i]));

    results.parameters = { omega, alpha, gamma, beta };
    results.conditionalVariances = variances;
    results.standardizedResiduals = standardizedResiduals;
    results.diagnostics = this.calculateGARCHDiagnostics(standardizedResiduals);

    return results;
  }

  // Regime-switching risk models
  calculateRegimeSwitchingRisk(returns, numRegimes = 2) {
    if (returns.length < this.minObservations) {
      throw new Error('Insufficient data for regime-switching model');
    }

    // Initialize parameters
    const regimes = Array.from({length: numRegimes}, (_, i) => ({
      mean: 0,
      variance: 1,
      probability: 1 / numRegimes
    }));

    // Transition probability matrix
    const transitionMatrix = Array.from({length: numRegimes}, () =>
      Array.from({length: numRegimes}, () => 1 / numRegimes)
    );

    // EM algorithm for parameter estimation
    const maxIterations = 100;
    const tolerance = 1e-6;
    let logLikelihood = -Infinity;
    let regimeProbabilities = null;

    for (let iter = 0; iter < maxIterations; iter++) {
      // E-step: Calculate regime probabilities
      const { regimeProbabilities: currentRegimeProbabilities, forwardProbs, backwardProbs } =
        this.calculateRegimeProbabilities(returns, regimes, transitionMatrix);

      regimeProbabilities = currentRegimeProbabilities;

      // M-step: Update parameters
      const newRegimes = this.updateRegimeParameters(returns, regimeProbabilities, numRegimes);
      const newTransitionMatrix = this.updateTransitionMatrix(regimeProbabilities, numRegimes);

      // Calculate new log-likelihood
      const newLogLikelihood = this.calculateRegimeLogLikelihood(
        returns, newRegimes, newTransitionMatrix
      );

      // Check convergence
      if (Math.abs(newLogLikelihood - logLikelihood) < tolerance) {
        logLikelihood = newLogLikelihood;
        break;
      }

      logLikelihood = newLogLikelihood;
      regimes.splice(0, regimes.length, ...newRegimes);
      transitionMatrix.splice(0, transitionMatrix.length, ...newTransitionMatrix);
    }

    // Viterbi algorithm for most likely regime path
    const mostLikelyPath = this.viterbiAlgorithm(returns, regimes, transitionMatrix);

    // Calculate regime-specific risk metrics
    const regimeRiskMetrics = regimes.map((regime, i) => {
      const regimeReturns = returns.filter((_, t) => mostLikelyPath[t] === i);
      if (regimeReturns.length < 5) return null;

      return {
        regime: i,
        mean: regime.mean,
        variance: regime.variance,
        volatility: Math.sqrt(regime.variance * this.tradingDaysPerYear),
        var95: this.calculateQuantile(regimeReturns, 0.05),
        expectedShortfall: this.calculateExpectedShortfall(regimeReturns),
        duration: regimeReturns.length,
        probability: regime.probability
      };
    }).filter(Boolean);

    return {
      numRegimes,
      regimes: regimeRiskMetrics,
      transitionMatrix,
      mostLikelyPath,
      logLikelihood,
      regimeProbabilities: regimeProbabilities[regimeProbabilities.length - 1],
      currentRegime: mostLikelyPath[mostLikelyPath.length - 1]
    };
  }

  // Advanced copula-based dependency modeling
  calculateCopulaDependencies(data, copulaType = 'gaussian') {
    if (!data || data.length < 2 || data[0].length < this.minObservations) {
      throw new Error('Insufficient data for copula estimation');
    }

    const numAssets = data.length;
    const numObservations = data[0].length;

    // Transform to uniform marginals using empirical CDF
    const uniforms = data.map(series => this.empiricalCDF(series));

    let copulaParams = {};
    let dependenceMatrix = Array.from({length: numAssets}, () =>
      Array.from({length: numAssets}, () => 0)
    );

    switch(copulaType.toLowerCase()) {
    case 'gaussian':
      copulaParams = this.estimateGaussianCopula(uniforms);
      dependenceMatrix = copulaParams.correlationMatrix;
      break;

    case 't':
      copulaParams = this.estimateTCopula(uniforms);
      dependenceMatrix = copulaParams.correlationMatrix;
      break;

    case 'clayton':
      copulaParams = this.estimateClaytonCopula(uniforms);
      dependenceMatrix = this.calculateClaytonDependence(copulaParams.theta, numAssets);
      break;

    case 'frank':
      copulaParams = this.estimateFrankCopula(uniforms);
      dependenceMatrix = this.calculateFrankDependence(copulaParams.theta, numAssets);
      break;

    case 'vine':
      return this.estimateVineCopula(uniforms);

    default:
      throw new Error(`Unsupported copula type: ${copulaType}`);
    }

    // Calculate tail dependencies
    const tailDependencies = this.calculateCopulaTailDependencies(
      uniforms, copulaType, copulaParams
    );

    // Goodness of fit tests
    const goodnessOfFit = this.testCopulaGoodnessOfFit(
      uniforms, copulaType, copulaParams
    );

    return {
      copulaType,
      parameters: copulaParams,
      dependenceMatrix,
      tailDependencies,
      goodnessOfFit,
      logLikelihood: copulaParams.logLikelihood || 0,
      aic: copulaParams.aic || 0
    };
  }

  // Jump diffusion model implementation
  calculateJumpDiffusionRisk(prices, model = 'merton') {
    const returns = this.calculateLogReturns(prices);

    switch(model.toLowerCase()) {
    case 'merton':
      return this.estimateMertonJumpDiffusion(returns);
    case 'kou':
      return this.estimateKouJumpDiffusion(returns);
    case 'variance-gamma':
      return this.estimateVarianceGammaModel(returns);
    default:
      throw new Error(`Unsupported jump diffusion model: ${model}`);
    }
  }

  estimateMertonJumpDiffusion(returns) {
    // Merton Jump Diffusion Model
    // dS/S = μdt + σdW + (J-1)dN
    // where J ~ lognormal, N ~ Poisson

    const n = returns.length;

    // Initial parameter estimates
    let mu = returns.reduce((sum, r) => sum + r, 0) / n;
    let sigma = Math.sqrt(this.calculateVariance(returns));
    let lambda = 0.1; // Jump intensity
    let muJ = 0; // Jump mean
    let sigmaJ = 0.05; // Jump volatility

    // Maximum likelihood estimation (simplified EM algorithm)
    const maxIterations = 100;
    const tolerance = 1e-6;

    for (let iter = 0; iter < maxIterations; iter++) {
      // E-step: Calculate jump probabilities
      const jumpProbs = returns.map(ret => {
        const diffusionProb = this.normalPDF(ret, mu, sigma);
        const jumpProb = this.jumpDiffusionPDF(ret, mu, sigma, lambda, muJ, sigmaJ);
        return jumpProb / (diffusionProb + jumpProb);
      });

      // M-step: Update parameters
      const newMu = returns.reduce((sum, ret, i) =>
        sum + ret * (1 - jumpProbs[i]), 0) / n;

      const newSigma = Math.sqrt(
        returns.reduce((sum, ret, i) =>
          sum + Math.pow(ret - newMu, 2) * (1 - jumpProbs[i]), 0) / n
      );

      const newLambda = jumpProbs.reduce((sum, prob) => sum + prob, 0) / n;

      // Check convergence
      const paramChange = Math.abs(newMu - mu) + Math.abs(newSigma - sigma) +
                         Math.abs(newLambda - lambda);

      if (paramChange < tolerance) break;

      mu = newMu;
      sigma = newSigma;
      lambda = newLambda;
    }

    // Identify jumps
    const jumps = returns.map((ret, i) => ({
      index: i,
      return: ret,
      isJump: Math.abs(ret - mu) > 3 * sigma,
      jumpProbability: this.jumpDiffusionPDF(ret, mu, sigma, lambda, muJ, sigmaJ)
    })).filter(jump => jump.isJump);

    // Calculate jump-adjusted risk metrics
    const jumpAdjustedReturns = returns.filter((_, i) =>
      !jumps.some(jump => jump.index === i)
    );

    const diffusionVolatility = Math.sqrt(this.calculateVariance(jumpAdjustedReturns) * this.tradingDaysPerYear);
    const jumpVolatility = lambda * (Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1);
    const totalVolatility = Math.sqrt(diffusionVolatility * diffusionVolatility + jumpVolatility);

    return {
      model: 'Merton Jump Diffusion',
      parameters: { mu, sigma, lambda, muJ, sigmaJ },
      jumps: jumps,
      jumpFrequency: jumps.length / n,
      diffusionVolatility,
      jumpVolatility,
      totalVolatility,
      jumpAdjustedVar: this.calculateQuantile(jumpAdjustedReturns, 0.05),
      jumpAdjustedES: this.calculateExpectedShortfall(jumpAdjustedReturns)
    };
  }

  // Extreme value theory implementation
  calculateExtremeValueRisk(returns, threshold = null, method = 'POT') {
    if (returns.length < this.minObservations) {
      throw new Error('Insufficient data for extreme value analysis');
    }

    switch(method.toLowerCase()) {
    case 'pot': // Peaks Over Threshold
      return this.peaksOverThreshold(returns, threshold);
    case 'bm': // Block Maxima
      return this.blockMaxima(returns);
    case 'hill': // Hill estimator
      return this.hillEstimator(returns);
    default:
      throw new Error(`Unsupported EVT method: ${method}`);
    }
  }

  peaksOverThreshold(returns, threshold = null) {
    // Automatically select threshold if not provided
    if (threshold === null) {
      threshold = this.selectPOTThreshold(returns);
    }

    // Extract exceedances
    const exceedances = returns.filter(ret => ret < threshold).map(ret => threshold - ret);

    if (exceedances.length < 10) {
      throw new Error('Insufficient exceedances for GPD fitting');
    }

    // Fit Generalized Pareto Distribution
    const { xi, sigma } = this.fitGPD(exceedances);

    // Calculate extreme quantiles
    const nu = exceedances.length / returns.length; // Exceedance rate

    const extremeQuantiles = [0.01, 0.005, 0.001].map(p => {
      const quantile = threshold - (sigma / xi) * (Math.pow((returns.length * p) / exceedances.length, -xi) - 1);
      return { probability: p, quantile };
    });

    // Expected shortfall for extreme quantiles
    const extremeES = extremeQuantiles.map(eq => ({
      probability: eq.probability,
      expectedShortfall: eq.quantile + (sigma - xi * threshold) / (1 - xi)
    }));

    return {
      method: 'Peaks Over Threshold',
      threshold,
      exceedanceRate: nu,
      parameters: { xi, sigma },
      extremeQuantiles,
      extremeExpectedShortfall: extremeES,
      tailIndex: xi,
      diagnostics: this.potDiagnostics(exceedances, xi, sigma)
    };
  }

  // Helper methods for complex calculations
  multipleRegression(X, y) {
    const n = X.length;
    const k = X[0].length;

    // Calculate X'X and X'y
    const XtX = this.matrixMultiply(this.transpose(X), X);
    const Xty = this.matrixVectorMultiply(this.transpose(X), y);

    // Solve normal equations: β = (X'X)^(-1)X'y
    const XtXInverse = this.matrixInverse(XtX);
    const coefficients = this.matrixVectorMultiply(XtXInverse, Xty);

    // Calculate fitted values and residuals
    const fitted = this.matrixVectorMultiply(X, coefficients);
    const residuals = y.map((val, i) => val - fitted[i]);
    const residualSumSquares = residuals.reduce((sum, res) => sum + res * res, 0);
    const residualVariance = residualSumSquares / (n - k);

    // Calculate R-squared
    const totalSumSquares = y.reduce((sum, val) => {
      const meanY = y.reduce((s, v) => s + v, 0) / n;
      return sum + Math.pow(val - meanY, 2);
    }, 0);
    const rSquared = 1 - residualSumSquares / totalSumSquares;

    // Calculate standard errors and t-statistics
    const standardErrors = Array.from({length: k}, (_, i) =>
      Math.sqrt(residualVariance * XtXInverse[i][i])
    );

    const tStatistics = coefficients.map((coef, i) => coef / standardErrors[i]);
    const pValues = tStatistics.map(t => 2 * (1 - this.tCDF(Math.abs(t), n - k)));

    return {
      coefficients,
      standardErrors,
      tStatistics,
      pValues,
      rSquared,
      residualVariance,
      residuals,
      fitted
    };
  }

  calculateSystematicVariance(factorExposures, factors) {
    const factorCovariance = this.calculateCovarianceMatrix(factors);
    return this.quadraticForm(factorExposures, factorCovariance);
  }

  calculateFactorRiskContributions(factorExposures, factors, factorNames) {
    const factorCovariance = this.calculateCovarianceMatrix(factors);
    const totalVariance = this.quadraticForm(factorExposures, factorCovariance);

    return factorExposures.map((exposure, i) => {
      const marginalContribution = factorExposures.reduce((sum, exp, j) =>
        sum + exp * factorCovariance[i][j], 0
      );

      return {
        factor: factorNames[i],
        exposure: exposure,
        marginalContribution: marginalContribution,
        contribution: exposure * marginalContribution,
        contributionPercent: (exposure * marginalContribution / totalVariance) * 100
      };
    });
  }

  calculateGARCHGradients(returns, variances, omega, alpha, beta) {
    const n = returns.length;
    let gradOmega = 0;
    let gradAlpha = Array(alpha.length).fill(0);
    let gradBeta = Array(beta.length).fill(0);

    for (let t = 1; t < n; t++) {
      const variance = variances[t];
      const epsilon = returns[t];
      const term = -0.5 + (epsilon * epsilon) / (2 * variance * variance);

      gradOmega += term;

      for (let i = 0; i < alpha.length && t > i; i++) {
        gradAlpha[i] += term * returns[t - i - 1] * returns[t - i - 1];
      }

      for (let i = 0; i < beta.length && t > i; i++) {
        gradBeta[i] += term * variances[t - i - 1];
      }
    }

    const norm = Math.sqrt(gradOmega * gradOmega +
                          gradAlpha.reduce((sum, g) => sum + g * g, 0) +
                          gradBeta.reduce((sum, g) => sum + g * g, 0));

    return { omega: gradOmega, alpha: gradAlpha, beta: gradBeta, norm };
  }

  calculateGARCHLogLikelihood(returns, variances) {
    let logLikelihood = 0;
    for (let t = 0; t < returns.length; t++) {
      logLikelihood += -0.5 * Math.log(2 * Math.PI * variances[t]) -
                       (returns[t] * returns[t]) / (2 * variances[t]);
    }
    return logLikelihood;
  }

  generateGARCHForecasts(returns, variances, omega, alpha, beta, steps) {
    const forecasts = [];
    let lastVariance = variances[variances.length - 1];
    let lastReturn = returns[returns.length - 1];

    for (let h = 1; h <= steps; h++) {
      let forecast = omega;

      if (h === 1) {
        // One-step ahead
        for (let i = 0; i < alpha.length; i++) {
          forecast += alpha[i] * lastReturn * lastReturn;
        }
        for (let i = 0; i < beta.length; i++) {
          forecast += beta[i] * lastVariance;
        }
      } else {
        // Multi-step ahead (unconditional variance)
        const unconditionalVariance = omega / (1 - alpha.reduce((sum, a) => sum + a, 0) -
                                                     beta.reduce((sum, b) => sum + b, 0));
        forecast = unconditionalVariance;
      }

      forecasts.push({
        horizon: h,
        variance: forecast,
        volatility: Math.sqrt(forecast),
        annualizedVolatility: Math.sqrt(forecast * this.tradingDaysPerYear)
      });

      lastVariance = forecast;
    }

    return forecasts;
  }

  calculateGARCHDiagnostics(standardizedResiduals) {
    // Ljung-Box test for serial correlation
    const ljungBox = this.ljungBoxTest(standardizedResiduals, 10);

    // ARCH-LM test for remaining heteroskedasticity
    const archLM = this.archLMTest(standardizedResiduals, 5);

    // Jarque-Bera test for normality
    const jarqueBera = this.jarqueBeraTest(standardizedResiduals);

    return {
      ljungBoxTest: ljungBox,
      archLMTest: archLM,
      jarqueBeraTest: jarqueBera,
      meanStandardizedResidual: standardizedResiduals.reduce((sum, r) => sum + r, 0) / standardizedResiduals.length,
      varianceStandardizedResidual: this.calculateVariance(standardizedResiduals)
    };
  }

  // Matrix operations
  matrixMultiply(A, B) {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const result = Array.from({length: rows}, () => Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        for (let k = 0; k < inner; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  }

  matrixVectorMultiply(matrix, vector) {
    return matrix.map(row =>
      row.reduce((sum, val, i) => sum + val * vector[i], 0)
    );
  }

  transpose(matrix) {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  }

  matrixInverse(matrix) {
    const n = matrix.length;
    const augmented = matrix.map((row, i) => [
      ...row,
      ...Array.from({length: n}, (_, j) => i === j ? 1 : 0)
    ]);

    // Gaussian elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(augmented[j][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = j;
        }
      }
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

      // Make diagonal element 1
      const pivot = augmented[i][i];
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }

      // Eliminate column
      for (let j = 0; j < n; j++) {
        if (j !== i) {
          const factor = augmented[j][i];
          for (let k = 0; k < 2 * n; k++) {
            augmented[j][k] -= factor * augmented[i][k];
          }
        }
      }
    }

    // Extract inverse
    return augmented.map(row => row.slice(n));
  }

  quadraticForm(vector, matrix) {
    const temp = this.matrixVectorMultiply(matrix, vector);
    return vector.reduce((sum, val, i) => sum + val * temp[i], 0);
  }

  calculateCovarianceMatrix(data) {
    const n = data.length;
    const k = data[0].length;
    const means = Array.from({length: k}, (_, i) =>
      data.reduce((sum, row) => sum + row[i], 0) / n
    );

    const covariance = Array.from({length: k}, () => Array(k).fill(0));

    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        let sum = 0;
        for (let t = 0; t < n; t++) {
          sum += (data[t][i] - means[i]) * (data[t][j] - means[j]);
        }
        covariance[i][j] = sum / (n - 1);
      }
    }

    return covariance;
  }

  calculateVariance(data) {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (data.length - 1);
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

  calculateExpectedShortfall(data, confidenceLevel = 0.95) {
    const sorted = [...data].sort((a, b) => a - b);
    const cutoff = Math.floor((1 - confidenceLevel) * sorted.length);
    const tail = sorted.slice(0, cutoff);
    return tail.reduce((sum, val) => sum + val, 0) / tail.length;
  }

  calculateLogReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
    return returns;
  }

  normalPDF(x, mean = 0, std = 1) {
    return (1 / (std * Math.sqrt(2 * Math.PI))) *
           Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
  }

  jumpDiffusionPDF(x, mu, sigma, lambda, muJ, sigmaJ) {
    // Simplified jump diffusion probability
    const diffusionComponent = this.normalPDF(x, mu, sigma);
    const jumpComponent = lambda * this.normalPDF(x, mu + muJ, Math.sqrt(sigma * sigma + sigmaJ * sigmaJ));
    return diffusionComponent + jumpComponent;
  }

  tCDF(t, df) {
    // Simplified t-distribution CDF
    if (df > 30) {
      return this.normalCDF(t);
    }
    // Student's t-distribution approximation
    return 0.5 + (t / Math.sqrt(df)) / (Math.PI * (1 + t * t / df));
  }

  normalCDF(z) {
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  erf(x) {
    // Error function approximation
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

  // Statistical tests
  ljungBoxTest(residuals, lags) {
    const n = residuals.length;
    const autocorrs = [];

    for (let lag = 1; lag <= lags; lag++) {
      let numerator = 0;
      let denominator = 0;

      for (let t = lag; t < n; t++) {
        numerator += residuals[t] * residuals[t - lag];
      }

      for (let t = 0; t < n; t++) {
        denominator += residuals[t] * residuals[t];
      }

      autocorrs.push(numerator / denominator);
    }

    const lbStatistic = n * (n + 2) * autocorrs.reduce((sum, corr, i) =>
      sum + (corr * corr) / (n - i - 1), 0
    );

    return {
      statistic: lbStatistic,
      pValue: 1 - this.chiSquareCDF(lbStatistic, lags),
      isSignificant: lbStatistic > this.chiSquareQuantile(0.95, lags)
    };
  }

  archLMTest(residuals, lags) {
    const squaredResiduals = residuals.map(r => r * r);

    // Regression of squared residuals on lagged squared residuals
    const X = [];
    const y = [];

    for (let t = lags; t < squaredResiduals.length; t++) {
      const row = [1]; // Constant term
      for (let lag = 1; lag <= lags; lag++) {
        row.push(squaredResiduals[t - lag]);
      }
      X.push(row);
      y.push(squaredResiduals[t]);
    }

    const regression = this.multipleRegression(X, y);
    const lmStatistic = (X.length - lags) * regression.rSquared;

    return {
      statistic: lmStatistic,
      pValue: 1 - this.chiSquareCDF(lmStatistic, lags),
      isSignificant: lmStatistic > this.chiSquareQuantile(0.95, lags)
    };
  }

  jarqueBeraTest(data) {
    const n = data.length;
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);

    const skewness = data.reduce((sum, val) =>
      sum + Math.pow((val - mean) / stdDev, 3), 0) / n;

    const kurtosis = data.reduce((sum, val) =>
      sum + Math.pow((val - mean) / stdDev, 4), 0) / n;

    const jbStatistic = (n / 6) * (skewness * skewness + (kurtosis - 3) * (kurtosis - 3) / 4);

    return {
      statistic: jbStatistic,
      pValue: 1 - this.chiSquareCDF(jbStatistic, 2),
      skewness: skewness,
      kurtosis: kurtosis,
      isNormal: jbStatistic < 5.991 // Chi-square(2) at 95%
    };
  }

  chiSquareCDF(x, df) {
    // Simplified chi-square CDF
    if (x <= 0) return 0;
    if (df === 1) return 2 * this.normalCDF(Math.sqrt(x)) - 1;
    if (df === 2) return 1 - Math.exp(-x / 2);

    // Approximation for other degrees of freedom
    const mean = df;
    const variance = 2 * df;
    const normalizedX = (x - mean) / Math.sqrt(variance);
    return this.normalCDF(normalizedX);
  }

  chiSquareQuantile(p, df) {
    // Simplified quantile function
    if (df === 1) return Math.pow(this.normalQuantile((p + 1) / 2), 2);
    if (df === 2) return -2 * Math.log(1 - p);

    // Wilson-Hilferty approximation
    const h = 2 / (9 * df);
    const z = this.normalQuantile(p);
    return df * Math.pow(1 - h + z * Math.sqrt(h), 3);
  }

  normalQuantile(p) {
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
}

module.exports = EnhancedRiskCalculator;