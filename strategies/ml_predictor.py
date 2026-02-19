"""
ML-basierte Preis-/Outcome-Prediktion
Gradient Boosting + optionales LSTM für Sequence-Prediction
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class MLPrediction:
    """Eine ML-Vorhersage"""
    symbol: str
    direction: str  # 'up' oder 'down'
    probability: float
    confidence: float  # Abstand zur Entscheidungsgrenze
    features_used: Dict[str, float]
    model_version: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelMetrics:
    """Modell-Metriken"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    samples_trained: int
    last_retrain: datetime


class MLPredictor:
    """
    ML-basierte Preis-Predictor

    Features:
    - RSI, BB-Position, Volume-Ratio
    - Funding-Rate
    - Polymarket-Sentiment-Scores

    Target: Preis-Richtung in nächsten 5-15 Minuten
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('MLPredictor')

        # Parameter
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.retrain_hours = self.config.get('retrain_hours', 6)
        self.prediction_horizon = self.config.get('prediction_horizon', 15)  # Minuten
        self.lookback_periods = self.config.get('lookback_periods', 20)

        # Modelle pro Symbol
        self.models: Dict[str, GradientBoostingClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metrics: Dict[str, ModelMetrics] = {}

        # Training-Daten
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.last_retrain: Dict[str, datetime] = {}

        # Predictions
        self.predictions: List[MLPrediction] = []
        self.prediction_history: Dict[str, List[Tuple[datetime, str, float, bool]]] = {}

        # Feature-Namen
        self.feature_names = [
            'rsi', 'rsi_change',
            'bb_position',  # -1 (lower) bis 1 (upper)
            'volume_ratio',
            'price_change_5m', 'price_change_15m',
            'ema_cross',  # EMA9 > EMA21 = 1, sonst -1
            'momentum',
            'volatility',
            'sentiment_score'  # Von Polymarket
        ]

    def prepare_features(self, candles: pd.DataFrame,
                        sentiment_score: float = 0.0) -> Optional[np.ndarray]:
        """
        Bereitet Features aus Kerzen-Daten vor

        Args:
            candles: DataFrame mit OHLCV + Indikatoren
            sentiment_score: Polymarket-Sentiment (-1 bis 1)

        Returns:
            Feature-Array oder None
        """
        if len(candles) < self.lookback_periods:
            return None

        latest = candles.iloc[-1]
        prev = candles.iloc[-2] if len(candles) > 1 else latest

        try:
            # RSI Features
            rsi = latest.get('rsi', 50) / 100  # Normalisieren auf 0-1
            rsi_prev = prev.get('rsi', 50) / 100
            rsi_change = rsi - rsi_prev

            # Bollinger Band Position
            bb_upper = latest.get('bb_upper', latest['close'])
            bb_lower = latest.get('bb_lower', latest['close'])
            bb_middle = latest.get('bb_middle', latest['close'])

            if bb_upper != bb_lower:
                bb_position = (latest['close'] - bb_middle) / (bb_upper - bb_lower)
            else:
                bb_position = 0

            # Volume Ratio
            avg_volume = candles['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1

            # Price Changes
            price_5m = candles['close'].iloc[-5] if len(candles) >= 5 else latest['close']
            price_15m = candles['close'].iloc[-15] if len(candles) >= 15 else latest['close']
            price_change_5m = (latest['close'] - price_5m) / price_5m if price_5m > 0 else 0
            price_change_15m = (latest['close'] - price_15m) / price_15m if price_15m > 0 else 0

            # EMA Cross
            ema_9 = latest.get('ema_9', latest['close'])
            ema_21 = latest.get('ema_21', latest['close'])
            ema_cross = 1 if ema_9 > ema_21 else -1

            # Momentum (ROC)
            momentum = price_change_5m * 10  # Skalieren

            # Volatility
            returns = candles['close'].pct_change().tail(20)
            volatility = returns.std() if len(returns) > 1 else 0

            features = np.array([
                rsi,
                rsi_change,
                bb_position,
                min(volume_ratio, 5),  # Cap bei 5x
                price_change_5m,
                price_change_15m,
                ema_cross,
                momentum,
                volatility,
                sentiment_score
            ])

            return features

        except Exception as e:
            self.logger.error(f"Feature-Preparation Fehler: {e}")
            return None

    def prepare_training_data(self, candles: pd.DataFrame,
                             symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bereitet Training-Daten vor

        Args:
            candles: Historische Kerzen-Daten
            symbol: Trading-Pair

        Returns:
            (X, y) Arrays
        """
        X = []
        y = []

        for i in range(self.lookback_periods, len(candles) - self.prediction_horizon):
            # Features zum Zeitpunkt i
            window = candles.iloc[:i+1]
            features = self.prepare_features(window, sentiment_score=0)

            if features is None:
                continue

            # Target: Preis in prediction_horizon Minuten
            current_price = candles.iloc[i]['close']
            future_price = candles.iloc[i + self.prediction_horizon]['close']

            # Label: 1 wenn Preis steigt, 0 wenn fällt
            label = 1 if future_price > current_price else 0

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)

    def train(self, candles: pd.DataFrame, symbol: str) -> Optional[ModelMetrics]:
        """
        Trainiert Modell für ein Symbol

        Args:
            candles: Historische Kerzen-Daten (mind. 1000 Datenpunkte)
            symbol: Trading-Pair

        Returns:
            ModelMetrics oder None bei Fehler
        """
        self.logger.info(f"Starte Training für {symbol}")

        if len(candles) < 500:
            self.logger.warning(f"Zu wenig Daten für {symbol}: {len(candles)}")
            return None

        try:
            # Daten vorbereiten
            X, y = self.prepare_training_data(candles, symbol)

            if len(X) < 100:
                self.logger.warning(f"Zu wenig Training-Samples für {symbol}: {len(X)}")
                return None

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Scaler erstellen und fitten
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Modell trainieren
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Evaluierung
            y_pred = model.predict(X_test_scaled)
            accuracy = np.mean(y_pred == y_test)

            # Präzision und Recall
            true_positives = np.sum((y_pred == 1) & (y_test == 1))
            false_positives = np.sum((y_pred == 1) & (y_test == 0))
            false_negatives = np.sum((y_pred == 0) & (y_test == 1))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Speichern
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_retrain[symbol] = datetime.now()

            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                samples_trained=len(X_train),
                last_retrain=datetime.now()
            )
            self.metrics[symbol] = metrics

            self.logger.info(f"Training abgeschlossen für {symbol}: "
                           f"Accuracy={accuracy:.2%}, F1={f1:.2%}")

            return metrics

        except Exception as e:
            self.logger.error(f"Training-Fehler für {symbol}: {e}")
            return None

    def predict(self, candles: pd.DataFrame, symbol: str,
               sentiment_score: float = 0.0) -> Optional[MLPrediction]:
        """
        Macht Vorhersage für ein Symbol

        Args:
            candles: Aktuelle Kerzen-Daten
            symbol: Trading-Pair
            sentiment_score: Polymarket-Sentiment (-1 bis 1)

        Returns:
            MLPrediction oder None
        """
        if symbol not in self.models:
            self.logger.warning(f"Kein Modell für {symbol} trainiert")
            return None

        features = self.prepare_features(candles, sentiment_score)
        if features is None:
            return None

        try:
            model = self.models[symbol]
            scaler = self.scalers[symbol]

            # Skalieren und vorhersagen
            features_scaled = scaler.transform(features.reshape(1, -1))
            proba = model.predict_proba(features_scaled)[0]

            # Vorhersage
            direction = 'up' if proba[1] > proba[0] else 'down'
            probability = max(proba)
            confidence = abs(proba[1] - proba[0])

            # Nur wenn Confidence über Threshold
            if probability < self.min_confidence:
                return None

            prediction = MLPrediction(
                symbol=symbol,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used={name: float(features[i]) for i, name in enumerate(self.feature_names)},
                model_version=f"v{self.last_retrain.get(symbol, datetime.now()).strftime('%Y%m%d%H%M')}"
            )

            self.predictions.append(prediction)

            return prediction

        except Exception as e:
            self.logger.error(f"Prediction-Fehler für {symbol}: {e}")
            return None

    def should_retrain(self, symbol: str) -> bool:
        """Prüft ob Modell neu trainiert werden sollte"""
        if symbol not in self.last_retrain:
            return True

        hours_since = (datetime.now() - self.last_retrain[symbol]).total_seconds() / 3600
        return hours_since >= self.retrain_hours

    def update_prediction_result(self, prediction: MLPrediction, actual_direction: str):
        """
        Aktualisiert Prediction-History mit echtem Ergebnis

        Args:
            prediction: Die ursprüngliche Vorhersage
            actual_direction: 'up' oder 'down'
        """
        symbol = prediction.symbol
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []

        correct = prediction.direction == actual_direction
        self.prediction_history[symbol].append((
            prediction.timestamp,
            prediction.direction,
            prediction.probability,
            correct
        ))

        # Nur letzte 1000 behalten
        self.prediction_history[symbol] = self.prediction_history[symbol][-1000:]

    def get_model_performance(self, symbol: str) -> Optional[Dict]:
        """
        Berechnet Live-Performance des Modells

        Returns:
            Dict mit Performance-Metriken
        """
        if symbol not in self.prediction_history:
            return None

        history = self.prediction_history[symbol]
        if len(history) < 10:
            return None

        recent = history[-100:]  # Letzte 100 Predictions
        correct = sum(1 for _, _, _, c in recent if c)
        accuracy = correct / len(recent)

        return {
            'symbol': symbol,
            'live_accuracy': accuracy,
            'predictions_made': len(history),
            'recent_predictions': len(recent),
            'model_metrics': self.metrics.get(symbol),
            'last_retrain': self.last_retrain.get(symbol)
        }

    def get_all_predictions(self, n: int = 50) -> List[MLPrediction]:
        """Gibt letzte n Predictions zurück"""
        return self.predictions[-n:]

    def get_statistics(self) -> Dict:
        """Gibt Strategie-Statistiken zurück"""
        return {
            'models_trained': len(self.models),
            'symbols': list(self.models.keys()),
            'total_predictions': len(self.predictions),
            'min_confidence': self.min_confidence,
            'prediction_horizon_min': self.prediction_horizon,
            'retrain_interval_hours': self.retrain_hours,
            'feature_count': len(self.feature_names)
        }


# Optional: LSTM für Sequence-Prediction
try:
    import torch
    import torch.nn as nn

    class LSTMPredictor(nn.Module):
        """
        LSTM-basierter Sequence-Predictor

        Für komplexere zeitliche Muster
        """

        def __init__(self, input_size: int = 10, hidden_size: int = 64,
                    num_layers: int = 2, dropout: float = 0.2):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            # x: (batch, sequence_length, features)
            lstm_out, _ = self.lstm(x)
            # Nur letzter Output
            last_output = lstm_out[:, -1, :]
            return self.fc(last_output)


    class AdvancedMLPredictor(MLPredictor):
        """
        Erweiterter Predictor mit LSTM-Support
        """

        def __init__(self, config: Dict = None):
            super().__init__(config)
            self.use_lstm = self.config.get('use_lstm', False)
            self.lstm_models: Dict[str, LSTMPredictor] = {}
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def train_lstm(self, candles: pd.DataFrame, symbol: str,
                      epochs: int = 50) -> Optional[float]:
            """Trainiert LSTM-Modell"""
            # Vereinfachte LSTM-Implementation
            self.logger.info(f"LSTM-Training für {symbol} (auf {self.device})")

            # Daten vorbereiten
            X, y = self.prepare_training_data(candles, symbol)
            if len(X) < 100:
                return None

            # In Sequences umwandeln
            sequence_length = 20
            X_seq = []
            y_seq = []

            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:i+sequence_length])
                y_seq.append(y[i+sequence_length-1])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            # Zu Tensoren
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.LongTensor(y_seq).to(self.device)

            # Modell erstellen
            model = LSTMPredictor(input_size=X.shape[1]).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            # Evaluierung
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()

            self.lstm_models[symbol] = model
            self.logger.info(f"LSTM-Training abgeschlossen für {symbol}: Accuracy={accuracy:.2%}")

            return accuracy

except ImportError:
    # PyTorch nicht verfügbar
    pass
