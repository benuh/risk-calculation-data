const { RiskMetrics } = require('../models/FinancialData');

class RiskCalculator {
  constructor() {
    this.riskFreeRate = 0.02; // Default 2% risk-free rate
  }

  setRiskFreeRate(rate) {
    this.riskFreeRate = rate;
  }

  calculateVolatility(prices) {
    if (!prices || prices.length < 2) {
      throw new Error('Insufficient price data for volatility calculation');
    }

    const returns = this.calculateReturns(prices);
    const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;

    const variance = returns.reduce((sum, ret) => {
      return sum + Math.pow(ret - meanReturn, 2);
    }, 0) / (returns.length - 1);

    return Math.sqrt(variance * 252); // Annualized volatility
  }

  calculateBeta(stockPrices, marketPrices) {
    if (!stockPrices || !marketPrices || stockPrices.length !== marketPrices.length) {
      throw new Error('Invalid price data for beta calculation');
    }

    const stockReturns = this.calculateReturns(stockPrices);
    const marketReturns = this.calculateReturns(marketPrices);

    const covariance = this.calculateCovariance(stockReturns, marketReturns);
    const marketVariance = this.calculateVariance(marketReturns);

    return covariance / marketVariance;
  }

  calculateSharpeRatio(prices, riskFreeRate = this.riskFreeRate) {
    const returns = this.calculateReturns(prices);
    const meanReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const annualizedReturn = meanReturn * 252;
    const volatility = this.calculateVolatility(prices);

    return (annualizedReturn - riskFreeRate) / volatility;
  }

  calculateValueAtRisk(prices, confidenceLevel = 0.95, timeHorizon = 1) {
    const returns = this.calculateReturns(prices);
    returns.sort((a, b) => a - b);

    const index = Math.floor((1 - confidenceLevel) * returns.length);
    const var95 = returns[index];

    return var95 * Math.sqrt(timeHorizon);
  }

  calculateExpectedShortfall(prices, confidenceLevel = 0.95, timeHorizon = 1) {
    const returns = this.calculateReturns(prices);
    returns.sort((a, b) => a - b);

    const varIndex = Math.floor((1 - confidenceLevel) * returns.length);
    const tailReturns = returns.slice(0, varIndex);

    if (tailReturns.length === 0) {
      return 0;
    }

    const expectedShortfall = tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length;
    return expectedShortfall * Math.sqrt(timeHorizon);
  }

  calculateMaxDrawdown(prices) {
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

  calculatePortfolioRisk(holdings, correlationMatrix) {
    const weights = holdings.map(h => h.weight);
    const volatilities = holdings.map(h => h.volatility || 0);

    let portfolioVariance = 0;

    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        const correlation = i === j ? 1 : (correlationMatrix[i][j] || 0);
        portfolioVariance += weights[i] * weights[j] * volatilities[i] * volatilities[j] * correlation;
      }
    }

    return Math.sqrt(portfolioVariance);
  }

  calculateReturns(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
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

  calculateVariance(returns) {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    return returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / (returns.length - 1);
  }

  calculateCorrelation(returns1, returns2) {
    const covariance = this.calculateCovariance(returns1, returns2);
    const std1 = Math.sqrt(this.calculateVariance(returns1));
    const std2 = Math.sqrt(this.calculateVariance(returns2));

    return covariance / (std1 * std2);
  }

  calculateRiskMetrics(symbol, prices, marketPrices = null) {
    const riskMetrics = new RiskMetrics({
      symbol: symbol,
      calculationDate: new Date()
    });

    riskMetrics.volatility = this.calculateVolatility(prices);
    riskMetrics.sharpeRatio = this.calculateSharpeRatio(prices);
    riskMetrics.valueAtRisk = this.calculateValueAtRisk(prices);
    riskMetrics.expectedShortfall = this.calculateExpectedShortfall(prices);
    riskMetrics.maxDrawdown = this.calculateMaxDrawdown(prices);

    if (marketPrices) {
      riskMetrics.beta = this.calculateBeta(prices, marketPrices);
    }

    return riskMetrics;
  }
}

module.exports = RiskCalculator;