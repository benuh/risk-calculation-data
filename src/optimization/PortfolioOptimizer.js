class PortfolioOptimizer {
  constructor(options = {}) {
    this.riskFreeRate = options.riskFreeRate || 0.02;
    this.maxIterations = options.maxIterations || 1000;
    this.tolerance = options.tolerance || 1e-8;
    this.learningRate = options.learningRate || 0.01;
    this.tradingDaysPerYear = options.tradingDaysPerYear || 252;

    // Optimization constraints
    this.constraints = {
      maxWeight: options.maxWeight || 1.0,
      minWeight: options.minWeight || 0.0,
      maxConcentration: options.maxConcentration || 0.4,
      sectorLimits: options.sectorLimits || {},
      turnoverLimit: options.turnoverLimit || null
    };
  }

  // Mean-Variance Optimization (Markowitz)
  optimizePortfolio(expectedReturns, covarianceMatrix, riskAversion = 1.0, constraints = {}) {
    const n = expectedReturns.length;

    // Validate inputs
    if (covarianceMatrix.length !== n || covarianceMatrix[0].length !== n) {
      throw new Error('Covariance matrix dimensions must match number of assets');
    }

    // Merge constraints
    const effectiveConstraints = { ...this.constraints, ...constraints };

    // Different optimization objectives
    const results = {};

    // 1. Minimum Variance Portfolio
    results.minVariance = this.calculateMinimumVariancePortfolio(covarianceMatrix, effectiveConstraints);

    // 2. Maximum Sharpe Ratio Portfolio
    results.maxSharpe = this.calculateMaxSharpePortfolio(
      expectedReturns, covarianceMatrix, effectiveConstraints
    );

    // 3. Mean-Variance Efficient Portfolio
    results.meanVariance = this.calculateMeanVariancePortfolio(
      expectedReturns, covarianceMatrix, riskAversion, effectiveConstraints
    );

    // 4. Risk Parity Portfolio
    results.riskParity = this.calculateRiskParityPortfolio(covarianceMatrix, effectiveConstraints);

    // 5. Maximum Diversification Portfolio
    results.maxDiversification = this.calculateMaxDiversificationPortfolio(
      covarianceMatrix, effectiveConstraints
    );

    // 6. Efficient Frontier
    results.efficientFrontier = this.generateEfficientFrontier(
      expectedReturns, covarianceMatrix, effectiveConstraints, 20
    );

    return results;
  }

  calculateMinimumVariancePortfolio(covarianceMatrix, constraints) {
    const n = covarianceMatrix.length;

    // Objective: minimize w' * Σ * w
    // Subject to: sum(w) = 1, constraints

    let weights = Array(n).fill(1 / n); // Initial equal weights

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Calculate gradient of portfolio variance
      const gradient = this.matrixVectorMultiply(covarianceMatrix, weights);

      // Apply constraints using projected gradient descent
      const newWeights = this.projectedGradientStep(weights, gradient, constraints, -1);

      // Check convergence
      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    // Normalize weights to sum to 1
    weights = this.normalizeWeights(weights);

    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);

    return {
      weights: weights,
      expectedReturn: null,
      volatility: portfolioVolatility,
      variance: portfolioVariance,
      sharpeRatio: null,
      optimization: 'Minimum Variance'
    };
  }

  calculateMaxSharpePortfolio(expectedReturns, covarianceMatrix, constraints) {
    const n = expectedReturns.length;

    // Maximize (μ - rf)' * w / sqrt(w' * Σ * w)
    // This is equivalent to maximizing (μ - rf)' * w - λ/2 * w' * Σ * w

    const excessReturns = expectedReturns.map(ret => ret - this.riskFreeRate);
    let weights = Array(n).fill(1 / n);

    for (let iter = 0; iter < this.maxIterations; iter++) {
      const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
      const portfolioReturn = this.vectorDot(weights, excessReturns);

      if (portfolioVariance <= 0) break;

      // Gradient of Sharpe ratio
      const varianceGradient = this.matrixVectorMultiply(covarianceMatrix, weights);
      const returnGradient = excessReturns;

      const sharpeGradient = returnGradient.map((ret, i) =>
        (ret * portfolioVariance - portfolioReturn * varianceGradient[i]) /
        Math.pow(portfolioVariance, 1.5)
      );

      const newWeights = this.projectedGradientStep(weights, sharpeGradient, constraints, 1);

      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    weights = this.normalizeWeights(weights);

    const portfolioReturn = this.vectorDot(weights, expectedReturns);
    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);
    const sharpeRatio = (portfolioReturn - this.riskFreeRate) / portfolioVolatility;

    return {
      weights: weights,
      expectedReturn: portfolioReturn * this.tradingDaysPerYear,
      volatility: portfolioVolatility,
      variance: portfolioVariance,
      sharpeRatio: sharpeRatio,
      optimization: 'Maximum Sharpe Ratio'
    };
  }

  calculateMeanVariancePortfolio(expectedReturns, covarianceMatrix, riskAversion, constraints) {
    const n = expectedReturns.length;

    // Maximize μ' * w - λ/2 * w' * Σ * w
    // where λ is risk aversion parameter

    let weights = Array(n).fill(1 / n);

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Gradient: μ - λ * Σ * w
      const varianceGradient = this.matrixVectorMultiply(covarianceMatrix, weights);
      const gradient = expectedReturns.map((ret, i) => ret - riskAversion * varianceGradient[i]);

      const newWeights = this.projectedGradientStep(weights, gradient, constraints, 1);

      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    weights = this.normalizeWeights(weights);

    const portfolioReturn = this.vectorDot(weights, expectedReturns);
    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);
    const sharpeRatio = (portfolioReturn - this.riskFreeRate) / portfolioVolatility;

    return {
      weights: weights,
      expectedReturn: portfolioReturn * this.tradingDaysPerYear,
      volatility: portfolioVolatility,
      variance: portfolioVariance,
      sharpeRatio: sharpeRatio,
      riskAversion: riskAversion,
      optimization: 'Mean-Variance'
    };
  }

  calculateRiskParityPortfolio(covarianceMatrix, constraints) {
    const n = covarianceMatrix.length;

    // Equal risk contribution: w_i * (Σw)_i / (w' * Σ * w) = 1/n for all i

    let weights = Array(n).fill(1 / n);

    for (let iter = 0; iter < this.maxIterations; iter++) {
      const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
      const marginalContributions = this.matrixVectorMultiply(covarianceMatrix, weights);

      // Risk contributions
      const riskContributions = weights.map((w, i) =>
        w * marginalContributions[i] / portfolioVariance
      );

      // Target risk contribution (equal for all assets)
      const targetContribution = 1 / n;

      // Gradient based on risk contribution differences
      const gradient = riskContributions.map((contrib, i) => {
        const error = contrib - targetContribution;
        return -error * marginalContributions[i] / portfolioVariance;
      });

      const newWeights = this.projectedGradientStep(weights, gradient, constraints, 1);

      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    weights = this.normalizeWeights(weights);

    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);

    // Calculate actual risk contributions
    const marginalContributions = this.matrixVectorMultiply(covarianceMatrix, weights);
    const riskContributions = weights.map((w, i) =>
      w * marginalContributions[i] / portfolioVariance
    );

    return {
      weights: weights,
      expectedReturn: null,
      volatility: portfolioVolatility,
      variance: portfolioVariance,
      riskContributions: riskContributions,
      sharpeRatio: null,
      optimization: 'Risk Parity'
    };
  }

  calculateMaxDiversificationPortfolio(covarianceMatrix, constraints) {
    const n = covarianceMatrix.length;

    // Maximize diversification ratio: (w' * σ) / sqrt(w' * Σ * w)
    // where σ is vector of individual asset volatilities

    const volatilities = covarianceMatrix.map((row, i) => Math.sqrt(row[i]));
    let weights = Array(n).fill(1 / n);

    for (let iter = 0; iter < this.maxIterations; iter++) {
      const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
      const weightedVolatility = this.vectorDot(weights, volatilities);

      if (portfolioVariance <= 0) break;

      // Gradient of diversification ratio
      const varianceGradient = this.matrixVectorMultiply(covarianceMatrix, weights);
      const portfolioVolatility = Math.sqrt(portfolioVariance);

      const gradient = volatilities.map((vol, i) =>
        (vol * portfolioVolatility - weightedVolatility * varianceGradient[i] / (2 * portfolioVolatility)) /
        portfolioVariance
      );

      const newWeights = this.projectedGradientStep(weights, gradient, constraints, 1);

      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    weights = this.normalizeWeights(weights);

    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);
    const weightedVolatility = this.vectorDot(weights, volatilities) * Math.sqrt(this.tradingDaysPerYear);
    const diversificationRatio = weightedVolatility / portfolioVolatility;

    return {
      weights: weights,
      expectedReturn: null,
      volatility: portfolioVolatility,
      variance: portfolioVariance,
      diversificationRatio: diversificationRatio,
      sharpeRatio: null,
      optimization: 'Maximum Diversification'
    };
  }

  generateEfficientFrontier(expectedReturns, covarianceMatrix, constraints, numPoints = 20) {
    const minReturnPortfolio = this.calculateMinimumVariancePortfolio(covarianceMatrix, constraints);
    const maxReturnIndex = expectedReturns.indexOf(Math.max(...expectedReturns));

    // Create portfolio with maximum expected return asset
    const maxReturnWeights = Array(expectedReturns.length).fill(0);
    maxReturnWeights[maxReturnIndex] = 1;
    const maxReturn = expectedReturns[maxReturnIndex];

    const minReturn = this.vectorDot(minReturnPortfolio.weights, expectedReturns);

    const frontierPoints = [];

    for (let i = 0; i < numPoints; i++) {
      const targetReturn = minReturn + (maxReturn - minReturn) * i / (numPoints - 1);

      // Optimize for minimum variance subject to return constraint
      const portfolio = this.optimizeForTargetReturn(
        expectedReturns, covarianceMatrix, targetReturn, constraints
      );

      if (portfolio) {
        const portfolioReturn = this.vectorDot(portfolio.weights, expectedReturns);
        const portfolioVariance = this.quadraticForm(portfolio.weights, covarianceMatrix);
        const portfolioVolatility = Math.sqrt(portfolioVariance * this.tradingDaysPerYear);
        const sharpeRatio = (portfolioReturn - this.riskFreeRate) / portfolioVolatility;

        frontierPoints.push({
          weights: portfolio.weights,
          expectedReturn: portfolioReturn * this.tradingDaysPerYear,
          volatility: portfolioVolatility,
          sharpeRatio: sharpeRatio,
          targetReturn: targetReturn * this.tradingDaysPerYear
        });
      }
    }

    return frontierPoints;
  }

  optimizeForTargetReturn(expectedReturns, covarianceMatrix, targetReturn, constraints) {
    const n = expectedReturns.length;
    let weights = Array(n).fill(1 / n);
    let lambda = 1; // Lagrange multiplier for return constraint

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Lagrangian: minimize w' * Σ * w + λ * (μ' * w - target)
      const varianceGradient = this.matrixVectorMultiply(covarianceMatrix, weights);
      const gradient = varianceGradient.map((grad, i) => grad + lambda * expectedReturns[i]);

      const newWeights = this.projectedGradientStep(weights, gradient, constraints, -1);

      // Update Lagrange multiplier
      const currentReturn = this.vectorDot(newWeights, expectedReturns);
      lambda = lambda + this.learningRate * (targetReturn - currentReturn);

      const change = this.vectorNorm(this.vectorSubtract(newWeights, weights));
      if (change < this.tolerance) break;

      weights = newWeights;
    }

    weights = this.normalizeWeights(weights);

    return { weights };
  }

  // Black-Litterman Model Implementation
  blackLittermanOptimization(marketCapWeights, expectedReturns, covarianceMatrix, views, confidences, tau = 0.05) {
    const n = expectedReturns.length;

    // Market implied returns (reverse optimization)
    const marketImpliedReturns = this.calculateImpliedReturns(marketCapWeights, covarianceMatrix);

    // Process investor views
    const { P, Q, Omega } = this.processInvestorViews(views, confidences, covarianceMatrix);

    // Black-Litterman formula
    // μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)μ + P'Ω^(-1)Q]

    const tauSigma = this.scalarMatrixMultiply(covarianceMatrix, tau);
    const tauSigmaInv = this.matrixInverse(tauSigma);

    const POmegaInvP = this.matrixMultiply(
      this.matrixMultiply(this.transpose(P), this.matrixInverse(Omega)),
      P
    );

    const precision = this.matrixAdd(tauSigmaInv, POmegaInvP);
    const precisionInv = this.matrixInverse(precision);

    const term1 = this.matrixVectorMultiply(tauSigmaInv, marketImpliedReturns);
    const term2 = this.matrixVectorMultiply(
      this.matrixMultiply(this.transpose(P), this.matrixInverse(Omega)),
      Q
    );

    const blReturns = this.matrixVectorMultiply(precisionInv, this.vectorAdd(term1, term2));

    // New covariance matrix
    const blCovariance = this.matrixAdd(covarianceMatrix, precisionInv);

    // Optimize with Black-Litterman inputs
    const blPortfolio = this.optimizePortfolio(blReturns, blCovariance);

    return {
      blReturns: blReturns,
      blCovariance: blCovariance,
      marketImpliedReturns: marketImpliedReturns,
      portfolio: blPortfolio,
      views: { P, Q, Omega },
      tau: tau
    };
  }

  // Multi-objective optimization using genetic algorithm
  multiObjectiveOptimization(expectedReturns, covarianceMatrix, objectives, constraints, populationSize = 100) {
    const n = expectedReturns.length;

    // Initialize population
    let population = Array.from({length: populationSize}, () =>
      this.generateRandomWeights(n, constraints)
    );

    const generations = 200;
    const mutationRate = 0.1;
    const crossoverRate = 0.8;

    for (let gen = 0; gen < generations; gen++) {
      // Evaluate population
      const fitness = population.map(individual =>
        this.evaluateMultiObjective(individual, expectedReturns, covarianceMatrix, objectives)
      );

      // Non-dominated sorting
      const fronts = this.nonDominatedSort(fitness);

      // Select parents
      const parents = this.tournamentSelection(population, fitness, populationSize);

      // Create offspring
      const offspring = [];
      for (let i = 0; i < populationSize; i += 2) {
        const parent1 = parents[i];
        const parent2 = parents[Math.min(i + 1, populationSize - 1)];

        let child1 = [...parent1];
        let child2 = [...parent2];

        // Crossover
        if (Math.random() < crossoverRate) {
          [child1, child2] = this.simulatedBinaryCrossover(parent1, parent2);
        }

        // Mutation
        if (Math.random() < mutationRate) {
          child1 = this.polynomialMutation(child1, constraints);
        }
        if (Math.random() < mutationRate) {
          child2 = this.polynomialMutation(child2, constraints);
        }

        // Ensure constraints
        child1 = this.enforceConstraints(child1, constraints);
        child2 = this.enforceConstraints(child2, constraints);

        offspring.push(child1, child2);
      }

      population = offspring.slice(0, populationSize);
    }

    // Final evaluation and Pareto front extraction
    const finalFitness = population.map(individual =>
      this.evaluateMultiObjective(individual, expectedReturns, covarianceMatrix, objectives)
    );

    const paretoFront = this.extractParetoFront(population, finalFitness);

    return {
      paretoFront: paretoFront,
      population: population,
      fitness: finalFitness,
      objectives: objectives
    };
  }

  // Portfolio rebalancing with transaction costs
  rebalancePortfolio(currentWeights, targetWeights, transactionCosts, constraints) {
    const n = currentWeights.length;

    // Objective: minimize tracking error + transaction costs
    let newWeights = [...targetWeights];

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Calculate total cost
      const turnover = newWeights.map((w, i) => Math.abs(w - currentWeights[i]));
      const totalCost = this.vectorDot(turnover, transactionCosts);

      // Tracking error
      const trackingError = this.vectorNorm(this.vectorSubtract(newWeights, targetWeights));

      // Combined objective gradient
      const gradient = newWeights.map((w, i) => {
        const trackingGrad = 2 * (w - targetWeights[i]);
        const costGrad = transactionCosts[i] * Math.sign(w - currentWeights[i]);
        return trackingGrad + costGrad;
      });

      const updatedWeights = this.projectedGradientStep(newWeights, gradient, constraints, -1);

      const change = this.vectorNorm(this.vectorSubtract(updatedWeights, newWeights));
      if (change < this.tolerance) break;

      newWeights = updatedWeights;
    }

    newWeights = this.normalizeWeights(newWeights);

    const finalTurnover = newWeights.map((w, i) => Math.abs(w - currentWeights[i]));
    const finalCost = this.vectorDot(finalTurnover, transactionCosts);
    const finalTrackingError = this.vectorNorm(this.vectorSubtract(newWeights, targetWeights));

    return {
      newWeights: newWeights,
      turnover: finalTurnover,
      totalTurnover: finalTurnover.reduce((sum, t) => sum + t, 0),
      transactionCost: finalCost,
      trackingError: finalTrackingError
    };
  }

  // Helper methods for optimization algorithms
  projectedGradientStep(weights, gradient, constraints, direction = 1) {
    const stepSize = this.learningRate * direction;
    let newWeights = weights.map((w, i) => w + stepSize * gradient[i]);

    // Apply box constraints
    newWeights = newWeights.map(w => Math.max(constraints.minWeight || 0, Math.min(constraints.maxWeight || 1, w)));

    // Apply sum constraint (weights sum to 1)
    const sum = newWeights.reduce((s, w) => s + w, 0);
    if (sum > 0) {
      newWeights = newWeights.map(w => w / sum);
    }

    // Apply concentration constraint
    if (constraints.maxConcentration) {
      newWeights = newWeights.map(w => Math.min(w, constraints.maxConcentration));
      const newSum = newWeights.reduce((s, w) => s + w, 0);
      if (newSum > 0) {
        newWeights = newWeights.map(w => w / newSum);
      }
    }

    return newWeights;
  }

  enforceConstraints(weights, constraints) {
    let constrainedWeights = [...weights];

    // Box constraints
    constrainedWeights = constrainedWeights.map(w =>
      Math.max(constraints.minWeight || 0, Math.min(constraints.maxWeight || 1, w))
    );

    // Normalize to sum to 1
    const sum = constrainedWeights.reduce((s, w) => s + w, 0);
    if (sum > 0) {
      constrainedWeights = constrainedWeights.map(w => w / sum);
    }

    return constrainedWeights;
  }

  generateRandomWeights(n, constraints) {
    let weights = Array.from({length: n}, () => Math.random());
    return this.enforceConstraints(weights, constraints);
  }

  normalizeWeights(weights) {
    const sum = weights.reduce((s, w) => s + w, 0);
    return sum > 0 ? weights.map(w => w / sum) : weights;
  }

  // Matrix and vector operations
  matrixVectorMultiply(matrix, vector) {
    return matrix.map(row =>
      row.reduce((sum, val, i) => sum + val * vector[i], 0)
    );
  }

  quadraticForm(vector, matrix) {
    const temp = this.matrixVectorMultiply(matrix, vector);
    return vector.reduce((sum, val, i) => sum + val * temp[i], 0);
  }

  vectorDot(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  vectorNorm(vector) {
    return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  }

  vectorSubtract(a, b) {
    return a.map((val, i) => val - b[i]);
  }

  vectorAdd(a, b) {
    return a.map((val, i) => val + b[i]);
  }

  matrixAdd(A, B) {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  }

  scalarMatrixMultiply(matrix, scalar) {
    return matrix.map(row => row.map(val => val * scalar));
  }

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

  transpose(matrix) {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  }

  matrixInverse(matrix) {
    const n = matrix.length;
    const augmented = matrix.map((row, i) => [
      ...row,
      ...Array.from({length: n}, (_, j) => i === j ? 1 : 0)
    ]);

    // Gaussian elimination with partial pivoting
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(augmented[j][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = j;
        }
      }
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

      // Check for singularity
      if (Math.abs(augmented[i][i]) < 1e-12) {
        throw new Error('Matrix is singular and cannot be inverted');
      }

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

  // Multi-objective optimization helper methods
  evaluateMultiObjective(weights, expectedReturns, covarianceMatrix, objectives) {
    const portfolioReturn = this.vectorDot(weights, expectedReturns);
    const portfolioVariance = this.quadraticForm(weights, covarianceMatrix);
    const portfolioVolatility = Math.sqrt(portfolioVariance);

    const objectives_values = [];

    objectives.forEach(obj => {
      switch(obj.type) {
      case 'return':
        objectives_values.push(obj.maximize ? portfolioReturn : -portfolioReturn);
        break;
      case 'volatility':
        objectives_values.push(obj.minimize ? portfolioVolatility : -portfolioVolatility);
        break;
      case 'sharpe': {
        const sharpe = (portfolioReturn - this.riskFreeRate) / portfolioVolatility;
        objectives_values.push(obj.maximize ? sharpe : -sharpe);
        break;
      }
      case 'diversification': {
        const diversification = this.calculateDiversification(weights);
        objectives_values.push(obj.maximize ? diversification : -diversification);
        break;
      }
      }
    });

    return objectives_values;
  }

  calculateDiversification(weights) {
    // Inverse of Herfindahl index
    const herfindahl = weights.reduce((sum, w) => sum + w * w, 0);
    return 1 / herfindahl;
  }

  nonDominatedSort(fitness) {
    const n = fitness.length;
    const fronts = [[]];
    const dominationCount = Array(n).fill(0);
    const dominatedSolutions = Array.from({length: n}, () => []);

    // Calculate domination relationships
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          if (this.dominates(fitness[i], fitness[j])) {
            dominatedSolutions[i].push(j);
          } else if (this.dominates(fitness[j], fitness[i])) {
            dominationCount[i]++;
          }
        }
      }

      if (dominationCount[i] === 0) {
        fronts[0].push(i);
      }
    }

    // Build subsequent fronts
    let frontIndex = 0;
    while (fronts[frontIndex].length > 0) {
      const nextFront = [];

      fronts[frontIndex].forEach(i => {
        dominatedSolutions[i].forEach(j => {
          dominationCount[j]--;
          if (dominationCount[j] === 0) {
            nextFront.push(j);
          }
        });
      });

      if (nextFront.length > 0) {
        fronts.push(nextFront);
      }
      frontIndex++;
    }

    return fronts.filter(front => front.length > 0);
  }

  dominates(a, b) {
    let atLeastOneBetter = false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] < b[i]) return false;
      if (a[i] > b[i]) atLeastOneBetter = true;
    }
    return atLeastOneBetter;
  }

  tournamentSelection(population, fitness, selectionSize) {
    const selected = [];
    const tournamentSize = 3;

    for (let i = 0; i < selectionSize; i++) {
      let best = Math.floor(Math.random() * population.length);

      for (let j = 1; j < tournamentSize; j++) {
        const competitor = Math.floor(Math.random() * population.length);
        if (this.dominates(fitness[competitor], fitness[best])) {
          best = competitor;
        }
      }

      selected.push([...population[best]]);
    }

    return selected;
  }

  simulatedBinaryCrossover(parent1, parent2, eta = 20) {
    const child1 = [...parent1];
    const child2 = [...parent2];

    for (let i = 0; i < parent1.length; i++) {
      if (Math.random() < 0.5) {
        const y1 = Math.min(parent1[i], parent2[i]);
        const y2 = Math.max(parent1[i], parent2[i]);

        if (Math.abs(y1 - y2) > 1e-14) {
          const rand = Math.random();
          const beta = 1 + 2 * (y1 - 0) / (y2 - y1);
          const alpha = 2 - Math.pow(beta, -(eta + 1));

          let betaq;
          if (rand <= 1 / alpha) {
            betaq = Math.pow(rand * alpha, 1 / (eta + 1));
          } else {
            betaq = Math.pow(1 / (2 - rand * alpha), 1 / (eta + 1));
          }

          child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
          child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
        }
      }
    }

    return [child1, child2];
  }

  polynomialMutation(individual, constraints, eta = 20) {
    const mutated = [...individual];

    for (let i = 0; i < individual.length; i++) {
      if (Math.random() < 1 / individual.length) {
        const y = individual[i];
        const yl = constraints.minWeight || 0;
        const yu = constraints.maxWeight || 1;

        const delta1 = (y - yl) / (yu - yl);
        const delta2 = (yu - y) / (yu - yl);

        const rand = Math.random();
        const mut_pow = 1 / (eta + 1);

        let deltaq;
        if (rand < 0.5) {
          const xy = 1 - delta1;
          const val = 2 * rand + (1 - 2 * rand) * Math.pow(xy, eta + 1);
          deltaq = Math.pow(val, mut_pow) - 1;
        } else {
          const xy = 1 - delta2;
          const val = 2 * (1 - rand) + 2 * (rand - 0.5) * Math.pow(xy, eta + 1);
          deltaq = 1 - Math.pow(val, mut_pow);
        }

        mutated[i] = y + deltaq * (yu - yl);
        mutated[i] = Math.max(yl, Math.min(yu, mutated[i]));
      }
    }

    return mutated;
  }

  extractParetoFront(population, fitness) {
    const fronts = this.nonDominatedSort(fitness);
    return fronts[0].map(index => ({
      weights: population[index],
      objectives: fitness[index]
    }));
  }

  // Black-Litterman helper methods
  calculateImpliedReturns(marketWeights, covarianceMatrix, riskAversion = 3.07) {
    // Implied returns: μ = λ * Σ * w_market
    return this.matrixVectorMultiply(
      this.scalarMatrixMultiply(covarianceMatrix, riskAversion),
      marketWeights
    );
  }

  processInvestorViews(views, confidences, covarianceMatrix) {
    const numAssets = covarianceMatrix.length;
    const numViews = views.length;

    // P matrix (picking matrix)
    const P = Array.from({length: numViews}, () => Array(numAssets).fill(0));

    // Q vector (view returns)
    const Q = Array(numViews);

    // Omega matrix (view uncertainties)
    const Omega = Array.from({length: numViews}, () => Array(numViews).fill(0));

    views.forEach((view, i) => {
      // Process view specification
      if (view.type === 'absolute') {
        P[i][view.asset] = 1;
        Q[i] = view.expectedReturn;
      } else if (view.type === 'relative') {
        P[i][view.asset1] = 1;
        P[i][view.asset2] = -1;
        Q[i] = view.expectedDifference;
      }

      // View uncertainty
      if (confidences[i]) {
        Omega[i][i] = 1 / confidences[i];
      } else {
        // Default uncertainty based on portfolio variance
        const viewVariance = this.quadraticForm(P[i], covarianceMatrix);
        Omega[i][i] = viewVariance;
      }
    });

    return { P, Q, Omega };
  }
}

module.exports = PortfolioOptimizer;