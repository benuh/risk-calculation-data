/**
 * Real-Time Streaming Risk Analytics
 *
 * This module provides real-time risk monitoring and alerting capabilities
 * with streaming data processing and dynamic risk assessment.
 */

const EventEmitter = require('events');
const WebSocket = require('ws');

class RealTimeRiskAnalytics extends EventEmitter {
  constructor(config = {}) {
    super();

    this.config = {
      alertThresholds: {
        volatility: 0.25,
        var95: 0.05,
        correlationChange: 0.3,
        volumeSpike: 5.0,
        priceGap: 0.1
      },
      windowSizes: {
        shortTerm: 20,
        mediumTerm: 60,
        longTerm: 252
      },
      updateInterval: 1000,
      maxHistoryLength: 10000,
      ...config
    };

    this.activeStreams = new Map();
    this.historicalData = new Map();
    this.riskMetrics = new Map();
    this.correlationMatrix = new Map();
    this.alertHistory = [];
    this.isStreaming = false;

    this.initializeRiskMonitoring();
  }

  initializeRiskMonitoring() {
    this.riskUpdateInterval = setInterval(() => {
      if (this.isStreaming) {
        this.updateRiskMetrics();
        this.checkAlertConditions();
      }
    }, this.config.updateInterval);
  }

  startStreaming(symbols) {
    this.isStreaming = true;
    this.symbols = Array.isArray(symbols) ? symbols : [symbols];

    this.symbols.forEach(symbol => {
      this.initializeSymbolTracking(symbol);
      this.startWebSocketConnection(symbol);
    });

    this.emit('streamingStarted', { symbols: this.symbols });
    console.log(`Started real-time risk analytics for: ${this.symbols.join(', ')}`);
  }

  stopStreaming() {
    this.isStreaming = false;

    this.activeStreams.forEach((ws, symbol) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });

    this.activeStreams.clear();
    this.emit('streamingStopped');
    console.log('Stopped real-time risk analytics');
  }

  initializeSymbolTracking(symbol) {
    this.historicalData.set(symbol, {
      prices: [],
      volumes: [],
      returns: [],
      timestamps: [],
      volatilities: [],
      vars: []
    });

    this.riskMetrics.set(symbol, {
      currentPrice: null,
      currentVolatility: null,
      currentVaR: null,
      priceChange24h: null,
      volumeRatio: null,
      riskLevel: 'LOW',
      lastUpdate: null
    });
  }

  startWebSocketConnection(symbol) {
    // Simulated WebSocket connection for demonstration
    // In production, this would connect to FMP WebSocket or similar
    const simulatedWS = this.createSimulatedStream(symbol);
    this.activeStreams.set(symbol, simulatedWS);
  }

  createSimulatedStream(symbol) {
    const basePrice = 100 + Math.random() * 400;
    let currentPrice = basePrice;
    let tickCount = 0;

    const simulator = {
      readyState: 1, // OPEN
      close: () => clearInterval(simulator.interval),
      interval: setInterval(() => {
        tickCount++;

        // Simulate realistic price movements
        const volatility = 0.02 + Math.random() * 0.03;
        const drift = (Math.random() - 0.5) * 0.001;
        const shock = Math.random() < 0.05 ? (Math.random() - 0.5) * 0.1 : 0;

        const priceChange = currentPrice * (drift + volatility * this.generateRandomNormal() + shock);
        currentPrice = Math.max(currentPrice + priceChange, 0.01);

        const volume = Math.floor(Math.random() * 1000000 + 100000);
        const timestamp = new Date();

        this.processMarketData({
          symbol,
          price: currentPrice,
          volume,
          timestamp,
          change: priceChange,
          changePercent: (priceChange / (currentPrice - priceChange)) * 100
        });

        // Simulate market events occasionally
        if (tickCount % 100 === 0) {
          this.simulateMarketEvent(symbol);
        }

      }, Math.random() * 2000 + 500) // Random intervals between 0.5-2.5 seconds
    };

    return simulator;
  }

  generateRandomNormal() {
    // Box-Muller transformation for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  processMarketData(data) {
    const { symbol, price, volume, timestamp, change, changePercent } = data;
    const historicalData = this.historicalData.get(symbol);
    const riskMetrics = this.riskMetrics.get(symbol);

    if (!historicalData || !riskMetrics) return;

    // Update historical data
    historicalData.prices.push(price);
    historicalData.volumes.push(volume);
    historicalData.timestamps.push(timestamp);

    if (historicalData.prices.length > 1) {
      const prevPrice = historicalData.prices[historicalData.prices.length - 2];
      const returnValue = (price - prevPrice) / prevPrice;
      historicalData.returns.push(returnValue);
    }

    // Maintain rolling window
    if (historicalData.prices.length > this.config.maxHistoryLength) {
      historicalData.prices.shift();
      historicalData.volumes.shift();
      historicalData.timestamps.shift();
      historicalData.returns.shift();
    }

    // Update current risk metrics
    this.updateSymbolRiskMetrics(symbol, data);

    // Emit real-time data event
    this.emit('marketData', {
      symbol,
      price,
      volume,
      change,
      changePercent,
      timestamp,
      riskMetrics: { ...riskMetrics }
    });
  }

  updateSymbolRiskMetrics(symbol, marketData) {
    const historicalData = this.historicalData.get(symbol);
    const riskMetrics = this.riskMetrics.get(symbol);

    if (historicalData.returns.length < this.config.windowSizes.shortTerm) {
      return; // Not enough data yet
    }

    // Calculate rolling volatility
    const shortTermReturns = historicalData.returns.slice(-this.config.windowSizes.shortTerm);
    const volatility = this.calculateRollingVolatility(shortTermReturns);

    // Calculate rolling VaR
    const var95 = this.calculateRollingVaR(shortTermReturns, 0.95);

    // Calculate price change metrics
    const priceChange24h = this.calculate24hPriceChange(historicalData);

    // Calculate volume ratio
    const volumeRatio = this.calculateVolumeRatio(historicalData);

    // Determine risk level
    const riskLevel = this.determineRiskLevel(volatility, var95, Math.abs(priceChange24h), volumeRatio);

    // Update risk metrics
    Object.assign(riskMetrics, {
      currentPrice: marketData.price,
      currentVolatility: volatility,
      currentVaR: var95,
      priceChange24h,
      volumeRatio,
      riskLevel,
      lastUpdate: marketData.timestamp
    });

    // Store in historical volatility and VaR
    historicalData.volatilities.push(volatility);
    historicalData.vars.push(var95);

    if (historicalData.volatilities.length > this.config.maxHistoryLength) {
      historicalData.volatilities.shift();
      historicalData.vars.shift();
    }
  }

  calculateRollingVolatility(returns) {
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance * 252); // Annualized
  }

  calculateRollingVaR(returns, confidence) {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sortedReturns.length);
    return Math.abs(sortedReturns[index] || 0);
  }

  calculate24hPriceChange(historicalData) {
    const prices = historicalData.prices;
    const timestamps = historicalData.timestamps;

    if (prices.length < 2) return 0;

    const currentPrice = prices[prices.length - 1];
    const currentTime = timestamps[timestamps.length - 1];

    // Find price 24 hours ago (or closest available)
    const target24hAgo = new Date(currentTime.getTime() - 24 * 60 * 60 * 1000);
    let price24hAgo = prices[0];

    for (let i = timestamps.length - 1; i >= 0; i--) {
      if (timestamps[i] <= target24hAgo) {
        price24hAgo = prices[i];
        break;
      }
    }

    return (currentPrice - price24hAgo) / price24hAgo;
  }

  calculateVolumeRatio(historicalData) {
    const volumes = historicalData.volumes;
    if (volumes.length < this.config.windowSizes.shortTerm) return 1;

    const currentVolume = volumes[volumes.length - 1];
    const avgVolume = volumes.slice(-this.config.windowSizes.shortTerm)
      .reduce((sum, v) => sum + v, 0) / this.config.windowSizes.shortTerm;

    return currentVolume / avgVolume;
  }

  determineRiskLevel(volatility, var95, priceChange, volumeRatio) {
    const thresholds = this.config.alertThresholds;

    let riskScore = 0;

    if (volatility > thresholds.volatility) riskScore += 2;
    else if (volatility > thresholds.volatility * 0.7) riskScore += 1;

    if (var95 > thresholds.var95) riskScore += 2;
    else if (var95 > thresholds.var95 * 0.7) riskScore += 1;

    if (priceChange > thresholds.priceGap) riskScore += 2;
    else if (priceChange > thresholds.priceGap * 0.7) riskScore += 1;

    if (volumeRatio > thresholds.volumeSpike) riskScore += 1;

    if (riskScore >= 4) return 'HIGH';
    if (riskScore >= 2) return 'MEDIUM';
    return 'LOW';
  }

  updateRiskMetrics() {
    // Update cross-asset correlations
    this.updateCorrelationMatrix();

    // Emit updated risk metrics
    const allRiskMetrics = {};
    this.riskMetrics.forEach((metrics, symbol) => {
      allRiskMetrics[symbol] = { ...metrics };
    });

    this.emit('riskMetricsUpdate', {
      timestamp: new Date(),
      metrics: allRiskMetrics,
      correlations: this.getCorrelationSnapshot()
    });
  }

  updateCorrelationMatrix() {
    const symbols = Array.from(this.historicalData.keys());

    if (symbols.length < 2) return;

    const correlations = new Map();

    for (let i = 0; i < symbols.length; i++) {
      for (let j = i + 1; j < symbols.length; j++) {
        const symbol1 = symbols[i];
        const symbol2 = symbols[j];

        const correlation = this.calculatePairwiseCorrelation(symbol1, symbol2);
        const key = `${symbol1}-${symbol2}`;

        correlations.set(key, {
          symbol1,
          symbol2,
          correlation,
          timestamp: new Date()
        });
      }
    }

    this.correlationMatrix = correlations;
  }

  calculatePairwiseCorrelation(symbol1, symbol2) {
    const data1 = this.historicalData.get(symbol1);
    const data2 = this.historicalData.get(symbol2);

    if (!data1 || !data2) return 0;

    const returns1 = data1.returns.slice(-this.config.windowSizes.mediumTerm);
    const returns2 = data2.returns.slice(-this.config.windowSizes.mediumTerm);

    const minLength = Math.min(returns1.length, returns2.length);
    if (minLength < 10) return 0;

    const r1 = returns1.slice(-minLength);
    const r2 = returns2.slice(-minLength);

    const mean1 = r1.reduce((sum, r) => sum + r, 0) / r1.length;
    const mean2 = r2.reduce((sum, r) => sum + r, 0) / r2.length;

    let numerator = 0;
    let sum1 = 0;
    let sum2 = 0;

    for (let i = 0; i < r1.length; i++) {
      const diff1 = r1[i] - mean1;
      const diff2 = r2[i] - mean2;
      numerator += diff1 * diff2;
      sum1 += diff1 * diff1;
      sum2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1 * sum2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  checkAlertConditions() {
    const alerts = [];

    // Check individual asset alerts
    this.riskMetrics.forEach((metrics, symbol) => {
      alerts.push(...this.checkAssetAlerts(symbol, metrics));
    });

    // Check correlation alerts
    alerts.push(...this.checkCorrelationAlerts());

    // Check portfolio-wide alerts
    alerts.push(...this.checkPortfolioAlerts());

    // Process and emit alerts
    alerts.forEach(alert => {
      this.alertHistory.push({
        ...alert,
        timestamp: new Date(),
        id: this.generateAlertId()
      });

      this.emit('riskAlert', alert);
    });

    // Maintain alert history limit
    if (this.alertHistory.length > 1000) {
      this.alertHistory = this.alertHistory.slice(-500);
    }
  }

  checkAssetAlerts(symbol, metrics) {
    const alerts = [];
    const thresholds = this.config.alertThresholds;

    // Volatility spike alert
    if (metrics.currentVolatility > thresholds.volatility) {
      alerts.push({
        type: 'VOLATILITY_SPIKE',
        symbol,
        severity: metrics.currentVolatility > thresholds.volatility * 1.5 ? 'HIGH' : 'MEDIUM',
        message: `High volatility detected: ${(metrics.currentVolatility * 100).toFixed(2)}%`,
        value: metrics.currentVolatility,
        threshold: thresholds.volatility
      });
    }

    // VaR breach alert
    if (metrics.currentVaR > thresholds.var95) {
      alerts.push({
        type: 'VAR_BREACH',
        symbol,
        severity: 'HIGH',
        message: `VaR threshold breached: ${(metrics.currentVaR * 100).toFixed(2)}%`,
        value: metrics.currentVaR,
        threshold: thresholds.var95
      });
    }

    // Price gap alert
    if (Math.abs(metrics.priceChange24h) > thresholds.priceGap) {
      alerts.push({
        type: 'PRICE_GAP',
        symbol,
        severity: 'MEDIUM',
        message: `Significant price movement: ${(metrics.priceChange24h * 100).toFixed(2)}%`,
        value: Math.abs(metrics.priceChange24h),
        threshold: thresholds.priceGap
      });
    }

    // Volume spike alert
    if (metrics.volumeRatio > thresholds.volumeSpike) {
      alerts.push({
        type: 'VOLUME_SPIKE',
        symbol,
        severity: 'LOW',
        message: `Unusual volume detected: ${metrics.volumeRatio.toFixed(1)}x average`,
        value: metrics.volumeRatio,
        threshold: thresholds.volumeSpike
      });
    }

    return alerts;
  }

  checkCorrelationAlerts() {
    const alerts = [];
    const threshold = this.config.alertThresholds.correlationChange;

    this.correlationMatrix.forEach((current, key) => {
      // Check for sudden correlation changes (would need historical correlation tracking)
      if (Math.abs(current.correlation) > 0.8) {
        alerts.push({
          type: 'HIGH_CORRELATION',
          symbols: [current.symbol1, current.symbol2],
          severity: 'MEDIUM',
          message: `High correlation detected: ${current.correlation.toFixed(3)} between ${current.symbol1} and ${current.symbol2}`,
          value: Math.abs(current.correlation),
          threshold: 0.8
        });
      }
    });

    return alerts;
  }

  checkPortfolioAlerts() {
    const alerts = [];

    // Check if multiple assets are showing high risk simultaneously
    const highRiskAssets = Array.from(this.riskMetrics.entries())
      .filter(([_, metrics]) => metrics.riskLevel === 'HIGH')
      .map(([symbol, _]) => symbol);

    if (highRiskAssets.length >= 2) {
      alerts.push({
        type: 'SYSTEMIC_RISK',
        symbols: highRiskAssets,
        severity: 'HIGH',
        message: `Multiple assets showing high risk: ${highRiskAssets.join(', ')}`,
        value: highRiskAssets.length,
        threshold: 2
      });
    }

    return alerts;
  }

  generateAlertId() {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  simulateMarketEvent(symbol) {
    const events = [
      'earnings_announcement',
      'fed_announcement',
      'geopolitical_event',
      'sector_rotation',
      'liquidity_shock'
    ];

    const event = events[Math.floor(Math.random() * events.length)];

    this.emit('marketEvent', {
      symbol,
      event,
      timestamp: new Date(),
      description: this.getEventDescription(event)
    });
  }

  getEventDescription(event) {
    const descriptions = {
      earnings_announcement: 'Quarterly earnings announcement causing price volatility',
      fed_announcement: 'Federal Reserve policy announcement affecting market sentiment',
      geopolitical_event: 'Geopolitical tensions causing market uncertainty',
      sector_rotation: 'Sector rotation affecting stock performance',
      liquidity_shock: 'Liquidity shortage causing trading disruptions'
    };

    return descriptions[event] || 'Market event detected';
  }

  getCorrelationSnapshot() {
    const snapshot = {};
    this.correlationMatrix.forEach((data, key) => {
      snapshot[key] = {
        correlation: data.correlation,
        timestamp: data.timestamp
      };
    });
    return snapshot;
  }

  getRiskSummary() {
    const summary = {
      timestamp: new Date(),
      totalAssets: this.symbols?.length || 0,
      riskDistribution: { LOW: 0, MEDIUM: 0, HIGH: 0 },
      averageVolatility: 0,
      averageVaR: 0,
      alertCount: this.alertHistory.length,
      recentAlerts: this.alertHistory.slice(-10)
    };

    let totalVol = 0;
    let totalVaR = 0;
    let count = 0;

    this.riskMetrics.forEach((metrics) => {
      summary.riskDistribution[metrics.riskLevel]++;

      if (metrics.currentVolatility !== null) {
        totalVol += metrics.currentVolatility;
        count++;
      }

      if (metrics.currentVaR !== null) {
        totalVaR += metrics.currentVaR;
      }
    });

    if (count > 0) {
      summary.averageVolatility = totalVol / count;
      summary.averageVaR = totalVaR / count;
    }

    return summary;
  }

  getDetailedMetrics(symbol) {
    const historicalData = this.historicalData.get(symbol);
    const riskMetrics = this.riskMetrics.get(symbol);

    if (!historicalData || !riskMetrics) {
      return null;
    }

    return {
      symbol,
      currentMetrics: { ...riskMetrics },
      historicalData: {
        priceHistory: historicalData.prices.slice(-100),
        volumeHistory: historicalData.volumes.slice(-100),
        volatilityHistory: historicalData.volatilities.slice(-100),
        varHistory: historicalData.vars.slice(-100),
        timestampHistory: historicalData.timestamps.slice(-100)
      },
      correlations: this.getSymbolCorrelations(symbol)
    };
  }

  getSymbolCorrelations(symbol) {
    const correlations = {};

    this.correlationMatrix.forEach((data, key) => {
      if (data.symbol1 === symbol) {
        correlations[data.symbol2] = data.correlation;
      } else if (data.symbol2 === symbol) {
        correlations[data.symbol1] = data.correlation;
      }
    });

    return correlations;
  }

  cleanup() {
    this.stopStreaming();

    if (this.riskUpdateInterval) {
      clearInterval(this.riskUpdateInterval);
    }

    this.removeAllListeners();
    this.historicalData.clear();
    this.riskMetrics.clear();
    this.correlationMatrix.clear();
    this.alertHistory = [];
  }
}

module.exports = RealTimeRiskAnalytics;