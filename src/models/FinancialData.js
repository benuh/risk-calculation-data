class FinancialData {
  constructor(data = {}) {
    this.symbol = data.symbol || null;
    this.date = data.date || new Date();
    this.price = data.price || null;
    this.marketCap = data.marketCap || null;
    this.volume = data.volume || null;
    this.sector = data.sector || null;
    this.industry = data.industry || null;
  }

  isValid() {
    return this.symbol && this.price && this.date;
  }

  toJSON() {
    return {
      symbol: this.symbol,
      date: this.date,
      price: this.price,
      marketCap: this.marketCap,
      volume: this.volume,
      sector: this.sector,
      industry: this.industry
    };
  }
}

class EconomicIndicator {
  constructor(data = {}) {
    this.indicator = data.indicator || null;
    this.value = data.value || null;
    this.date = data.date || new Date();
    this.country = data.country || 'US';
    this.frequency = data.frequency || 'quarterly';
    this.unit = data.unit || null;
  }

  isValid() {
    return this.indicator && this.value !== null && this.date;
  }

  toJSON() {
    return {
      indicator: this.indicator,
      value: this.value,
      date: this.date,
      country: this.country,
      frequency: this.frequency,
      unit: this.unit
    };
  }
}

class RiskMetrics {
  constructor(data = {}) {
    this.symbol = data.symbol || null;
    this.beta = data.beta || null;
    this.volatility = data.volatility || null;
    this.sharpeRatio = data.sharpeRatio || null;
    this.valueAtRisk = data.valueAtRisk || null;
    this.expectedShortfall = data.expectedShortfall || null;
    this.maxDrawdown = data.maxDrawdown || null;
    this.calculationDate = data.calculationDate || new Date();
    this.timeHorizon = data.timeHorizon || '1Y';
    this.confidenceLevel = data.confidenceLevel || 0.95;
  }

  isValid() {
    return this.symbol && this.calculationDate;
  }

  toJSON() {
    return {
      symbol: this.symbol,
      beta: this.beta,
      volatility: this.volatility,
      sharpeRatio: this.sharpeRatio,
      valueAtRisk: this.valueAtRisk,
      expectedShortfall: this.expectedShortfall,
      maxDrawdown: this.maxDrawdown,
      calculationDate: this.calculationDate,
      timeHorizon: this.timeHorizon,
      confidenceLevel: this.confidenceLevel
    };
  }
}

class ESGData {
  constructor(data = {}) {
    this.symbol = data.symbol || null;
    this.environmentalScore = data.environmentalScore || null;
    this.socialScore = data.socialScore || null;
    this.governanceScore = data.governanceScore || null;
    this.overallScore = data.overallScore || null;
    this.date = data.date || new Date();
    this.provider = data.provider || 'Sustainalytics';
  }

  isValid() {
    return this.symbol && this.overallScore !== null;
  }

  calculateOverallScore() {
    if (this.environmentalScore && this.socialScore && this.governanceScore) {
      this.overallScore = (this.environmentalScore + this.socialScore + this.governanceScore) / 3;
    }
    return this.overallScore;
  }

  toJSON() {
    return {
      symbol: this.symbol,
      environmentalScore: this.environmentalScore,
      socialScore: this.socialScore,
      governanceScore: this.governanceScore,
      overallScore: this.overallScore,
      date: this.date,
      provider: this.provider
    };
  }
}

class TreasuryData {
  constructor(data = {}) {
    this.maturity = data.maturity || null;
    this.yield = data.yield || null;
    this.date = data.date || new Date();
    this.country = data.country || 'US';
  }

  isValid() {
    return this.maturity && this.yield !== null && this.date;
  }

  toJSON() {
    return {
      maturity: this.maturity,
      yield: this.yield,
      date: this.date,
      country: this.country
    };
  }
}

class PortfolioData {
  constructor(data = {}) {
    this.portfolioId = data.portfolioId || null;
    this.holdings = data.holdings || [];
    this.totalValue = data.totalValue || 0;
    this.currency = data.currency || 'USD';
    this.lastUpdated = data.lastUpdated || new Date();
  }

  addHolding(symbol, quantity, price) {
    const holding = {
      symbol,
      quantity,
      price,
      value: quantity * price,
      weight: 0
    };
    this.holdings.push(holding);
    this.updateTotalValue();
    this.updateWeights();
  }

  updateTotalValue() {
    this.totalValue = this.holdings.reduce((sum, holding) => sum + holding.value, 0);
  }

  updateWeights() {
    this.holdings.forEach(holding => {
      holding.weight = this.totalValue > 0 ? holding.value / this.totalValue : 0;
    });
  }

  isValid() {
    return this.portfolioId && this.holdings.length > 0;
  }

  toJSON() {
    return {
      portfolioId: this.portfolioId,
      holdings: this.holdings,
      totalValue: this.totalValue,
      currency: this.currency,
      lastUpdated: this.lastUpdated
    };
  }
}

module.exports = {
  FinancialData,
  EconomicIndicator,
  RiskMetrics,
  ESGData,
  TreasuryData,
  PortfolioData
};