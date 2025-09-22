class MachineLearningRiskModels {
  constructor(options = {}) {
    this.modelTypes = ['ensemble', 'neural_network', 'gradient_boosting', 'svm', 'random_forest'];
    this.validationSplit = options.validationSplit || 0.2;
    this.randomSeed = options.randomSeed || 42;
    this.maxEpochs = options.maxEpochs || 1000;
    this.learningRate = options.learningRate || 0.001;
    this.tolerance = options.tolerance || 1e-6;

    // Feature engineering parameters
    this.lookbackWindow = options.lookbackWindow || 252; // 1 year
    this.predictionHorizon = options.predictionHorizon || 22; // 1 month
    this.featureWindow = options.featureWindow || 60; // Feature calculation window

    // Model ensemble parameters
    this.ensembleSize = options.ensembleSize || 10;
    this.baggingRatio = options.baggingRatio || 0.8;

    this.models = new Map();
    this.featureImportances = new Map();
    this.modelPerformances = new Map();
  }

  // Feature Engineering for Financial Time Series
  engineerFeatures(priceData, volumeData = null, fundamentalData = null, macroData = null) {
    const features = {};

    // Technical indicators
    features.technical = this.calculateTechnicalFeatures(priceData, volumeData);

    // Statistical features
    features.statistical = this.calculateStatisticalFeatures(priceData);

    // Market microstructure features
    if (volumeData) {
      features.microstructure = this.calculateMicrostructureFeatures(priceData, volumeData);
    }

    // Fundamental features
    if (fundamentalData) {
      features.fundamental = this.calculateFundamentalFeatures(fundamentalData);
    }

    // Macroeconomic features
    if (macroData) {
      features.macroeconomic = this.calculateMacroFeatures(macroData);
    }

    // Volatility features
    features.volatility = this.calculateVolatilityFeatures(priceData);

    // Regime features
    features.regime = this.calculateRegimeFeatures(priceData);

    return this.combineFeatures(features);
  }

  calculateTechnicalFeatures(prices, volumes = null) {
    const features = {};

    // Moving averages
    features.sma_5 = this.simpleMovingAverage(prices, 5);
    features.sma_10 = this.simpleMovingAverage(prices, 10);
    features.sma_20 = this.simpleMovingAverage(prices, 20);
    features.sma_50 = this.simpleMovingAverage(prices, 50);
    features.sma_200 = this.simpleMovingAverage(prices, 200);

    // Exponential moving averages
    features.ema_12 = this.exponentialMovingAverage(prices, 12);
    features.ema_26 = this.exponentialMovingAverage(prices, 26);

    // MACD
    const macd = this.calculateMACD(prices);
    features.macd = macd.macd;
    features.macd_signal = macd.signal;
    features.macd_histogram = macd.histogram;

    // RSI
    features.rsi_14 = this.calculateRSI(prices, 14);

    // Bollinger Bands
    const bb = this.calculateBollingerBands(prices, 20, 2);
    features.bb_upper = bb.upper;
    features.bb_lower = bb.lower;
    features.bb_position = bb.position;
    features.bb_width = bb.width;

    // Stochastic Oscillator
    if (prices.length >= 14) {
      const stoch = this.calculateStochastic(prices, 14);
      features.stoch_k = stoch.k;
      features.stoch_d = stoch.d;
    }

    // Average True Range
    features.atr = this.calculateATR(prices, 14);

    // Price ratios
    features.price_to_sma20 = this.calculateRatio(prices, features.sma_20);
    features.price_to_sma50 = this.calculateRatio(prices, features.sma_50);

    // Volume features (if available)
    if (volumes) {
      features.volume_sma = this.simpleMovingAverage(volumes, 20);
      features.volume_ratio = this.calculateRatio(volumes, features.volume_sma);

      // On-Balance Volume
      features.obv = this.calculateOBV(prices, volumes);

      // Volume Price Trend
      features.vpt = this.calculateVPT(prices, volumes);
    }

    return features;
  }

  calculateStatisticalFeatures(prices) {
    const returns = this.calculateReturns(prices);
    const features = {};

    // Rolling statistical moments
    features.return_mean_5 = this.rollingMean(returns, 5);
    features.return_mean_20 = this.rollingMean(returns, 20);
    features.return_std_5 = this.rollingStd(returns, 5);
    features.return_std_20 = this.rollingStd(returns, 20);
    features.return_skew_20 = this.rollingSkewness(returns, 20);
    features.return_kurt_20 = this.rollingKurtosis(returns, 20);

    // Quantile features
    features.return_q25_20 = this.rollingQuantile(returns, 20, 0.25);
    features.return_q75_20 = this.rollingQuantile(returns, 20, 0.75);
    features.return_q90_20 = this.rollingQuantile(returns, 20, 0.90);
    features.return_q10_20 = this.rollingQuantile(returns, 20, 0.10);

    // Drawdown features
    features.drawdown = this.calculateDrawdown(prices);
    features.max_drawdown_20 = this.rollingMax(features.drawdown, 20);

    // Autocorrelation features
    features.return_autocorr_1 = this.rollingAutocorrelation(returns, 20, 1);
    features.return_autocorr_5 = this.rollingAutocorrelation(returns, 20, 5);

    // Trend features
    features.price_trend_5 = this.calculateTrend(prices, 5);
    features.price_trend_20 = this.calculateTrend(prices, 20);

    return features;
  }

  calculateVolatilityFeatures(prices) {
    const returns = this.calculateReturns(prices);
    const features = {};

    // GARCH volatility
    try {
      features.garch_volatility = this.estimateGARCHVolatility(returns);
    } catch (error) {
      features.garch_volatility = Array(prices.length).fill(null);
    }

    // Realized volatility
    features.realized_vol_5 = this.realizedVolatility(returns, 5);
    features.realized_vol_20 = this.realizedVolatility(returns, 20);

    // Volatility of volatility
    features.vol_of_vol = this.volatilityOfVolatility(returns, 20);

    // Parkinson volatility (high-low based)
    // Note: Would need high/low data for this
    features.parkinson_vol = this.estimateParkinsonVolatility(prices, 20);

    // Jump detection
    features.jump_indicator = this.detectJumps(returns);

    return features;
  }

  calculateRegimeFeatures(prices) {
    const returns = this.calculateReturns(prices);
    const features = {};

    // Regime identification using HMM-like approach
    const regimes = this.identifyVolatilityRegimes(returns);
    features.regime_state = regimes.states;
    features.regime_probability = regimes.probabilities;

    // Trend regime
    features.trend_regime = this.identifyTrendRegime(prices, 50);

    // Market stress indicator
    features.stress_indicator = this.calculateStressIndicator(returns);

    return features;
  }

  // Ensemble Risk Prediction Model
  trainEnsembleModel(features, targets, modelConfig = {}) {
    const {
      baseModels = ['rf', 'gb', 'nn'],
      ensembleMethod = 'stacking',
      crossValidationFolds = 5
    } = modelConfig;

    // Split data
    const { trainX, trainY, testX, testY } = this.splitData(features, targets);

    // Train base models
    const baseModelPredictions = {};
    const baseModelWeights = {};

    for (const modelType of baseModels) {
      console.log(`Training ${modelType} model...`);

      const model = this.createBaseModel(modelType);
      const cvPredictions = this.crossValidationTraining(
        model, trainX, trainY, crossValidationFolds
      );

      baseModelPredictions[modelType] = cvPredictions;
      baseModelWeights[modelType] = this.calculateModelWeight(cvPredictions, trainY);

      // Train on full training set
      const trainedModel = this.trainModel(model, trainX, trainY);
      this.models.set(modelType, trainedModel);
    }

    // Meta-learner for stacking
    let ensembleModel;
    if (ensembleMethod === 'stacking') {
      const metaFeatures = this.createMetaFeatures(baseModelPredictions);
      ensembleModel = this.trainMetaLearner(metaFeatures, trainY);
    }

    // Evaluate ensemble
    const ensemblePredictions = this.predictEnsemble(
      testX, baseModels, ensembleMethod, ensembleModel
    );

    const performance = this.evaluateModel(ensemblePredictions, testY);

    return {
      baseModels: baseModels,
      ensembleMethod: ensembleMethod,
      performance: performance,
      models: this.models,
      weights: baseModelWeights,
      metaLearner: ensembleModel
    };
  }

  // Neural Network for Risk Prediction
  trainNeuralNetwork(features, targets, architecture = {}) {
    const {
      hiddenLayers = [64, 32, 16],
      activation = 'relu',
      outputActivation = 'linear',
      dropout = 0.2,
      regularization = 0.001
    } = architecture;

    const { trainX, trainY, testX, testY } = this.splitData(features, targets);

    // Initialize network weights
    const network = this.initializeNeuralNetwork(trainX[0].length, hiddenLayers, 1);

    // Training loop
    const batchSize = Math.min(64, Math.floor(trainX.length / 10));
    let bestLoss = Infinity;
    let patience = 0;
    const maxPatience = 50;

    for (let epoch = 0; epoch < this.maxEpochs; epoch++) {
      // Shuffle training data
      const shuffledData = this.shuffleData(trainX, trainY);

      let totalLoss = 0;
      const numBatches = Math.ceil(shuffledData.X.length / batchSize);

      for (let batch = 0; batch < numBatches; batch++) {
        const startIdx = batch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, shuffledData.X.length);

        const batchX = shuffledData.X.slice(startIdx, endIdx);
        const batchY = shuffledData.Y.slice(startIdx, endIdx);

        // Forward pass
        const predictions = this.forwardPass(network, batchX, dropout);

        // Calculate loss
        const batchLoss = this.calculateMSELoss(predictions, batchY, network, regularization);
        totalLoss += batchLoss;

        // Backward pass
        this.backwardPass(network, batchX, batchY, predictions, regularization);
      }

      const avgLoss = totalLoss / numBatches;

      // Validation
      if (epoch % 10 === 0) {
        const valPredictions = this.forwardPass(network, testX, 0);
        const valLoss = this.calculateMSELoss(valPredictions, testY, network, 0);

        if (valLoss < bestLoss) {
          bestLoss = valLoss;
          patience = 0;
        } else {
          patience++;
        }

        if (patience >= maxPatience) {
          console.log(`Early stopping at epoch ${epoch}`);
          break;
        }
      }
    }

    // Final evaluation
    const predictions = this.forwardPass(network, testX, 0);
    const performance = this.evaluateModel(predictions, testY);

    return {
      network: network,
      performance: performance,
      architecture: architecture
    };
  }

  // LSTM for Time Series Risk Prediction
  trainLSTMModel(sequences, targets, architecture = {}) {
    const {
      lstmUnits = 50,
      numLayers = 2,
      dropout = 0.2,
      lookback = this.lookbackWindow
    } = architecture;

    // Prepare sequences
    const { trainSequences, trainTargets, testSequences, testTargets } =
      this.prepareSequences(sequences, targets, lookback);

    // Initialize LSTM network
    const lstm = this.initializeLSTM(sequences[0].length, lstmUnits, numLayers);

    // Training loop
    for (let epoch = 0; epoch < this.maxEpochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < trainSequences.length; i++) {
        // Forward pass through LSTM
        const { output, states } = this.lstmForward(lstm, trainSequences[i]);

        // Calculate loss
        const loss = this.calculateMSELoss([output], [trainTargets[i]], lstm, 0.001);
        totalLoss += loss;

        // Backward pass (BPTT)
        this.lstmBackward(lstm, trainSequences[i], trainTargets[i], states);
      }

      // Validation every 10 epochs
      if (epoch % 10 === 0) {
        const valPredictions = testSequences.map(seq =>
          this.lstmForward(lstm, seq).output
        );
        const valPerformance = this.evaluateModel(valPredictions, testTargets);
        console.log(`Epoch ${epoch}, Validation R²: ${valPerformance.r2.toFixed(4)}`);
      }
    }

    // Final evaluation
    const predictions = testSequences.map(seq => this.lstmForward(lstm, seq).output);
    const performance = this.evaluateModel(predictions, testTargets);

    return {
      model: lstm,
      performance: performance,
      architecture: architecture
    };
  }

  // Gradient Boosting for Risk Prediction
  trainGradientBoosting(features, targets, config = {}) {
    const {
      nEstimators = 100,
      maxDepth = 6,
      learningRate = 0.1,
      subsample = 0.8,
      featureSubsample = 0.8
    } = config;

    const { trainX, trainY, testX, testY } = this.splitData(features, targets);

    // Initialize with mean prediction
    const initialPrediction = trainY.reduce((sum, y) => sum + y, 0) / trainY.length;
    let predictions = Array(trainY.length).fill(initialPrediction);

    const trees = [];
    const featureImportances = Array(trainX[0].length).fill(0);

    for (let i = 0; i < nEstimators; i++) {
      // Calculate residuals
      const residuals = trainY.map((y, idx) => y - predictions[idx]);

      // Subsample data
      const { sampledX, sampledY } = this.subsampleData(
        trainX, residuals, subsample, featureSubsample
      );

      // Train decision tree on residuals
      const tree = this.trainDecisionTree(sampledX, sampledY, maxDepth, featureSubsample);
      trees.push(tree);

      // Update predictions
      const treePredictions = this.predictDecisionTree(tree, trainX);
      for (let j = 0; j < predictions.length; j++) {
        predictions[j] += learningRate * treePredictions[j];
      }

      // Update feature importances
      this.updateFeatureImportances(featureImportances, tree);

      // Early stopping based on validation performance
      if (i % 10 === 0) {
        const valPredictions = this.predictGradientBoosting(trees, testX, learningRate, initialPrediction);
        const valPerformance = this.evaluateModel(valPredictions, testY);
        console.log(`Iteration ${i}, Validation R²: ${valPerformance.r2.toFixed(4)}`);
      }
    }

    // Final evaluation
    const finalPredictions = this.predictGradientBoosting(
      trees, testX, learningRate, initialPrediction
    );
    const performance = this.evaluateModel(finalPredictions, testY);

    // Normalize feature importances
    const totalImportance = featureImportances.reduce((sum, imp) => sum + imp, 0);
    const normalizedImportances = featureImportances.map(imp => imp / totalImportance);

    return {
      trees: trees,
      initialPrediction: initialPrediction,
      learningRate: learningRate,
      performance: performance,
      featureImportances: normalizedImportances
    };
  }

  // Support Vector Machine for Risk Prediction
  trainSVM(features, targets, config = {}) {
    const {
      kernel = 'rbf',
      C = 1.0,
      gamma = 'scale',
      epsilon = 0.1
    } = config;

    const { trainX, trainY, testX, testY } = this.splitData(features, targets);

    // Normalize features
    const { normalizedTrainX, normalizedTestX, scaler } = this.normalizeFeatures(trainX, testX);

    // Convert to classification problem for risk prediction
    const riskThreshold = this.calculateQuantile(trainY, 0.8); // Top 20% risk
    const trainLabels = trainY.map(y => y > riskThreshold ? 1 : -1);

    // SVM training using SMO algorithm (simplified)
    const svm = this.trainSVMClassifier(normalizedTrainX, trainLabels, C, kernel, gamma);

    // Predict risk probabilities
    const testPredictions = this.predictSVM(svm, normalizedTestX);

    // Convert back to risk scores
    const riskScores = testPredictions.map(pred =>
      pred > 0 ? riskThreshold * (1 + pred) : riskThreshold * (1 + pred * 0.5)
    );

    const performance = this.evaluateModel(riskScores, testY);

    return {
      model: svm,
      scaler: scaler,
      riskThreshold: riskThreshold,
      performance: performance,
      config: config
    };
  }

  // Reinforcement Learning for Dynamic Risk Management
  trainRLAgent(environment, config = {}) {
    const {
      algorithm = 'dqn',
      episodes = 1000,
      explorationDecay = 0.995,
      learningRate = 0.001,
      memorySize = 10000
    } = config;

    let explorationRate = config.explorationRate || 0.1;

    const agent = this.initializeRLAgent(algorithm, environment.stateSize, environment.actionSize, config);
    const memory = [];
    let totalRewards = [];

    for (let episode = 0; episode < episodes; episode++) {
      let state = environment.reset();
      let episodeReward = 0;
      let done = false;

      while (!done) {
        // Choose action (epsilon-greedy)
        const action = Math.random() < explorationRate ?
          Math.floor(Math.random() * environment.actionSize) :
          this.chooseAction(agent, state);

        // Execute action
        const { nextState, reward, isDone } = environment.step(action);

        // Store experience
        memory.push({ state, action, reward, nextState, done: isDone });
        if (memory.length > memorySize) {
          memory.shift();
        }

        // Train agent
        if (memory.length > 64) {
          const batch = this.sampleBatch(memory, 32);
          this.trainRLStep(agent, batch);
        }

        state = nextState;
        episodeReward += reward;
        done = isDone;
      }

      totalRewards.push(episodeReward);
      explorationRate *= explorationDecay;

      if (episode % 100 === 0) {
        const avgReward = totalRewards.slice(-100).reduce((sum, r) => sum + r, 0) / 100;
        console.log(`Episode ${episode}, Average Reward: ${avgReward.toFixed(2)}`);
      }
    }

    return {
      agent: agent,
      totalRewards: totalRewards,
      finalExplorationRate: explorationRate
    };
  }

  // Risk Factor Extraction using Autoencoders
  trainFactorAutoencoder(returns, config = {}) {
    const {
      encodingDim = 10,
      hiddenLayers = [64, 32],
      epochs = 200,
      regularization = 0.001
    } = config;

    // Normalize returns
    const { normalizedData, scaler } = this.normalizeData(returns);

    // Initialize autoencoder
    const autoencoder = this.initializeAutoencoder(
      returns[0].length, encodingDim, hiddenLayers
    );

    // Training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < normalizedData.length; i++) {
        // Forward pass
        const { encoded, decoded } = this.autoencoderForward(autoencoder, normalizedData[i]);

        // Calculate reconstruction loss
        const loss = this.calculateReconstructionLoss(decoded, normalizedData[i], autoencoder, regularization);
        totalLoss += loss;

        // Backward pass
        this.autoencoderBackward(autoencoder, normalizedData[i], decoded);
      }

      if (epoch % 20 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${(totalLoss / normalizedData.length).toFixed(6)}`);
      }
    }

    // Extract factors
    const factors = normalizedData.map(data => this.encode(autoencoder, data));

    // Factor analysis
    const factorAnalysis = this.analyzeFactors(factors, returns);

    return {
      autoencoder: autoencoder,
      scaler: scaler,
      factors: factors,
      factorAnalysis: factorAnalysis,
      config: config
    };
  }

  // Model Evaluation and Validation
  evaluateModel(predictions, actuals) {
    const n = predictions.length;
    if (n !== actuals.length) {
      throw new Error('Predictions and actuals must have same length');
    }

    // Calculate metrics
    const mse = predictions.reduce((sum, pred, i) =>
      sum + Math.pow(pred - actuals[i], 2), 0) / n;

    const mae = predictions.reduce((sum, pred, i) =>
      sum + Math.abs(pred - actuals[i]), 0) / n;

    const actualMean = actuals.reduce((sum, val) => sum + val, 0) / n;
    const tss = actuals.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
    const rss = predictions.reduce((sum, pred, i) => sum + Math.pow(pred - actuals[i], 2), 0);
    const r2 = 1 - (rss / tss);

    // Information Coefficient (for financial predictions)
    const predRanks = this.calculateRanks(predictions);
    const actualRanks = this.calculateRanks(actuals);
    const ic = this.calculateCorrelation(predRanks, actualRanks);

    // Hit rate (percentage of correct directional predictions)
    let correctDirection = 0;
    for (let i = 1; i < n; i++) {
      const predDirection = predictions[i] > predictions[i-1];
      const actualDirection = actuals[i] > actuals[i-1];
      if (predDirection === actualDirection) correctDirection++;
    }
    const hitRate = correctDirection / (n - 1);

    // Maximum drawdown of prediction errors
    const errors = predictions.map((pred, i) => pred - actuals[i]);
    const cumulativeErrors = errors.reduce((cum, err, i) => {
      cum.push((cum[i-1] || 0) + err);
      return cum;
    }, []);
    const maxDrawdown = this.calculateMaxDrawdown(cumulativeErrors);

    return {
      mse: mse,
      rmse: Math.sqrt(mse),
      mae: mae,
      r2: r2,
      informationCoefficient: ic,
      hitRate: hitRate,
      maxDrawdown: maxDrawdown,
      predictions: predictions,
      actuals: actuals
    };
  }

  // Helper methods for ML operations
  splitData(features, targets, testRatio = this.validationSplit) {
    const n = features.length;
    const testSize = Math.floor(n * testRatio);
    const trainSize = n - testSize;

    // Time series split (no shuffling)
    const trainX = features.slice(0, trainSize);
    const trainY = targets.slice(0, trainSize);
    const testX = features.slice(trainSize);
    const testY = targets.slice(trainSize);

    return { trainX, trainY, testX, testY };
  }

  normalizeFeatures(trainX, testX) {
    const scaler = this.fitScaler(trainX);
    const normalizedTrainX = this.transformData(trainX, scaler);
    const normalizedTestX = this.transformData(testX, scaler);

    return { normalizedTrainX, normalizedTestX, scaler };
  }

  fitScaler(data) {
    const numFeatures = data[0].length;
    const means = Array(numFeatures).fill(0);
    const stds = Array(numFeatures).fill(0);

    // Calculate means
    for (let j = 0; j < numFeatures; j++) {
      for (let i = 0; i < data.length; i++) {
        means[j] += data[i][j];
      }
      means[j] /= data.length;
    }

    // Calculate standard deviations
    for (let j = 0; j < numFeatures; j++) {
      for (let i = 0; i < data.length; i++) {
        stds[j] += Math.pow(data[i][j] - means[j], 2);
      }
      stds[j] = Math.sqrt(stds[j] / (data.length - 1));
    }

    return { means, stds };
  }

  transformData(data, scaler) {
    return data.map(row =>
      row.map((val, j) => (val - scaler.means[j]) / (scaler.stds[j] || 1))
    );
  }

  calculateReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i-1]) / prices[i-1]);
    }
    return returns;
  }

  simpleMovingAverage(data, window) {
    const sma = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        sma.push(null);
      } else {
        const sum = data.slice(i - window + 1, i + 1).reduce((s, v) => s + v, 0);
        sma.push(sum / window);
      }
    }
    return sma;
  }

  exponentialMovingAverage(data, window) {
    const alpha = 2 / (window + 1);
    const ema = [data[0]];

    for (let i = 1; i < data.length; i++) {
      ema.push(alpha * data[i] + (1 - alpha) * ema[i-1]);
    }

    return ema;
  }

  calculateMACD(prices, fast = 12, slow = 26, signal = 9) {
    const emaFast = this.exponentialMovingAverage(prices, fast);
    const emaSlow = this.exponentialMovingAverage(prices, slow);

    const macd = emaFast.map((fast, i) => fast - emaSlow[i]);
    const signalLine = this.exponentialMovingAverage(macd.filter(v => v !== null), signal);
    const histogram = macd.map((m, i) => m - (signalLine[i] || 0));

    return { macd, signal: signalLine, histogram };
  }

  calculateRSI(prices, window = 14) {
    const changes = this.calculateReturns(prices);
    const gains = changes.map(c => c > 0 ? c : 0);
    const losses = changes.map(c => c < 0 ? -c : 0);

    const avgGains = this.simpleMovingAverage(gains, window);
    const avgLosses = this.simpleMovingAverage(losses, window);

    return avgGains.map((gain, i) => {
      if (avgLosses[i] === 0) return 100;
      const rs = gain / avgLosses[i];
      return 100 - (100 / (1 + rs));
    });
  }

  calculateBollingerBands(prices, window = 20, numStd = 2) {
    const sma = this.simpleMovingAverage(prices, window);
    const std = this.rollingStd(prices, window);

    const upper = sma.map((avg, i) => avg + numStd * std[i]);
    const lower = sma.map((avg, i) => avg - numStd * std[i]);
    const position = prices.map((price, i) =>
      std[i] > 0 ? (price - sma[i]) / std[i] : 0
    );
    const width = upper.map((u, i) => (u - lower[i]) / sma[i]);

    return { upper, lower, middle: sma, position, width };
  }

  rollingStd(data, window) {
    const std = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        std.push(null);
      } else {
        const slice = data.slice(i - window + 1, i + 1);
        const mean = slice.reduce((s, v) => s + v, 0) / window;
        const variance = slice.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / (window - 1);
        std.push(Math.sqrt(variance));
      }
    }
    return std;
  }

  rollingMean(data, window) {
    return this.simpleMovingAverage(data, window);
  }

  rollingSkewness(data, window) {
    const skew = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        skew.push(null);
      } else {
        const slice = data.slice(i - window + 1, i + 1);
        const mean = slice.reduce((s, v) => s + v, 0) / window;
        const variance = slice.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / window;
        const std = Math.sqrt(variance);

        if (std === 0) {
          skew.push(0);
        } else {
          const skewness = slice.reduce((s, v) => s + Math.pow((v - mean) / std, 3), 0) / window;
          skew.push(skewness);
        }
      }
    }
    return skew;
  }

  rollingKurtosis(data, window) {
    const kurt = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        kurt.push(null);
      } else {
        const slice = data.slice(i - window + 1, i + 1);
        const mean = slice.reduce((s, v) => s + v, 0) / window;
        const variance = slice.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / window;
        const std = Math.sqrt(variance);

        if (std === 0) {
          kurt.push(0);
        } else {
          const kurtosis = slice.reduce((s, v) => s + Math.pow((v - mean) / std, 4), 0) / window;
          kurt.push(kurtosis - 3); // Excess kurtosis
        }
      }
    }
    return kurt;
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

  calculateRanks(data) {
    const indexed = data.map((value, index) => ({ value, index }));
    indexed.sort((a, b) => a.value - b.value);

    const ranks = new Array(data.length);
    for (let i = 0; i < indexed.length; i++) {
      ranks[indexed[i].index] = i + 1;
    }

    return ranks;
  }

  calculateCorrelation(x, y) {
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

    return numerator / Math.sqrt(denomX * denomY);
  }

  calculateMaxDrawdown(values) {
    let maxDrawdown = 0;
    let peak = values[0];

    for (let i = 1; i < values.length; i++) {
      if (values[i] > peak) {
        peak = values[i];
      }

      const drawdown = (peak - values[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown;
  }

  combineFeatures(featureGroups) {
    const combined = [];
    const featureNames = [];

    // Get the length from the first non-empty feature group
    let dataLength = 0;
    for (const group of Object.values(featureGroups)) {
      if (group && Object.keys(group).length > 0) {
        dataLength = Object.values(group)[0].length;
        break;
      }
    }

    // Combine all features
    for (let i = 0; i < dataLength; i++) {
      const featureVector = [];

      for (const [groupName, features] of Object.entries(featureGroups)) {
        for (const [featureName, values] of Object.entries(features)) {
          if (i === 0) {
            featureNames.push(`${groupName}_${featureName}`);
          }
          featureVector.push(values[i] || 0);
        }
      }

      combined.push(featureVector);
    }

    return { features: combined, featureNames };
  }
}

module.exports = MachineLearningRiskModels;