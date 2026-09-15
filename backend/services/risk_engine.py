"""
VoiceShield-AI — Risk Engine

Aggregates AASIST-L window scores using multiple strategies
and produces a final risk assessment (BONAFIDE / SUSPICIOUS / SPOOF).
"""

import numpy as np
from typing import List, Dict

class RiskEngine:
    """
    Aggregates window-level AASIST-L scores into recording-level predictions.

    Implements multiple aggregation strategies:
    1. Spoof Ratio: % of windows detected as spoof
    2. Maximum Score: Highest spoof score observed
    3. Mean Score: Average spoof logit across windows
    """

    def __init__(self,
                 spoof_ratio_threshold: float = 0.15,
                 max_spoof_threshold: float = 1.0,
                 mean_spoof_threshold: float = 0.1):
        """
        Initialize risk engine with thresholds.

        Parameters
        ----------
        spoof_ratio_threshold : float
            If spoof_windows / total_windows >= this, predict SPOOF
            Default: 0.15 (15%)
        max_spoof_threshold : float
            If max(spoof_scores) >= this, predict SPOOF
            Default: 1.0 (lowered from 2.0)
        mean_spoof_threshold : float
            If mean(spoof_scores) >= this, predict SPOOF
            Default: 0.1 (lowered from 0.5)
        """
        self.spoof_ratio_threshold = spoof_ratio_threshold
        self.max_spoof_threshold = max_spoof_threshold
        self.mean_spoof_threshold = mean_spoof_threshold

    def aggregate(self, window_scores: List[Dict]) -> Dict:
        """
        Aggregate window scores using multiple strategies.

        Parameters
        ----------
        window_scores : list of dict
            Each dict contains:
            - spoof_score: float
            - bonafide_score: float
            - prediction: "BONAFIDE" or "SPOOF"

        Returns
        -------
        dict with keys:
        - spoof_ratio: float (0.0 to 1.0)
        - max_spoof_score: float
        - mean_spoof_score: float
        - mean_bonafide_score: float
        - strategy_1_prediction: from spoof ratio
        - strategy_2_prediction: from max score
        - strategy_3_prediction: from mean score
        - final_prediction: consensus or highest priority alert
        - confidence: how confident is the prediction
        - risk_level: "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
        """

        if not window_scores or len(window_scores) == 0:
            return {
                "success": False,
                "error": "No window scores provided"
            }

        # Extract scores
        spoof_scores = np.array([w["spoof_score"] for w in window_scores])
        bonafide_scores = np.array([w["bonafide_score"] for w in window_scores])
        predictions = [w["prediction"] for w in window_scores]

        # Calculate metrics
        num_spoof_windows = sum(1 for p in predictions if p == "SPOOF")
        num_total_windows = len(predictions)
        spoof_ratio = num_spoof_windows / num_total_windows

        max_spoof_score = float(np.max(spoof_scores))
        min_spoof_score = float(np.min(spoof_scores))
        mean_spoof_score = float(np.mean(spoof_scores))
        median_spoof_score = float(np.median(spoof_scores))

        mean_bonafide_score = float(np.mean(bonafide_scores))

        # Strategy 1: Spoof Ratio
        strategy_1_pred = "SPOOF" if spoof_ratio >= self.spoof_ratio_threshold else "BONAFIDE"
        strategy_1_confidence = abs(spoof_ratio - 0.5) * 2  # 0-1 scale

        # Strategy 2: Maximum Spoof Score
        strategy_2_pred = "SPOOF" if max_spoof_score >= self.max_spoof_threshold else "BONAFIDE"
        strategy_2_confidence = min(max_spoof_score / self.max_spoof_threshold, 1.0)

        # Strategy 3: Mean Spoof Score
        strategy_3_pred = "SPOOF" if mean_spoof_score >= self.mean_spoof_threshold else "BONAFIDE"
        strategy_3_confidence = max(0, min(mean_spoof_score / self.mean_spoof_threshold, 1.0))

        # Consensus decision
        spoof_votes = sum(1 for p in [strategy_1_pred, strategy_2_pred, strategy_3_pred] if p == "SPOOF")

        if spoof_votes >= 2:
            final_prediction = "SPOOF"
            confidence = (spoof_votes / 3.0) + 0.1  # Boost if multiple strategies agree
        else:
            final_prediction = "BONAFIDE"
            confidence = (1 - spoof_votes / 3.0) + 0.1  # Boost if strategies agree on bonafide

        confidence = min(confidence, 1.0)

        # Risk level assessment
        if final_prediction == "SPOOF":
            if spoof_ratio >= 0.5:
                risk_level = "CRITICAL"  # >50% windows are spoof
            elif spoof_ratio >= 0.25:
                risk_level = "HIGH"  # 25-50% spoof windows
            else:
                risk_level = "MEDIUM"  # <25% spoof windows
        else:
            if spoof_ratio >= 0.15:
                risk_level = "MEDIUM"  # Some suspicious windows even if overall bonafide
            else:
                risk_level = "LOW"  # Very few suspicious windows

        # Detailed assessment
        assessment = self._generate_assessment(
            final_prediction,
            spoof_ratio,
            max_spoof_score,
            mean_spoof_score,
            num_spoof_windows,
            num_total_windows
        )

        return {
            "success": True,
            "final_prediction": final_prediction,
            "risk_level": risk_level,
            "confidence": round(confidence, 3),

            # Metrics
            "spoof_ratio": round(spoof_ratio, 3),
            "max_spoof_score": round(max_spoof_score, 4),
            "min_spoof_score": round(min_spoof_score, 4),
            "mean_spoof_score": round(mean_spoof_score, 4),
            "median_spoof_score": round(median_spoof_score, 4),
            "mean_bonafide_score": round(mean_bonafide_score, 4),

            # Window statistics
            "num_spoof_windows": num_spoof_windows,
            "num_bonafide_windows": num_total_windows - num_spoof_windows,
            "num_total_windows": num_total_windows,

            # Strategy results
            "strategy_1": {
                "name": "Spoof Ratio Threshold",
                "threshold": self.spoof_ratio_threshold,
                "prediction": strategy_1_pred,
                "confidence": round(strategy_1_confidence, 3),
                "value": round(spoof_ratio, 3)
            },
            "strategy_2": {
                "name": "Maximum Spoof Score",
                "threshold": self.max_spoof_threshold,
                "prediction": strategy_2_pred,
                "confidence": round(strategy_2_confidence, 3),
                "value": round(max_spoof_score, 4)
            },
            "strategy_3": {
                "name": "Mean Spoof Score",
                "threshold": self.mean_spoof_threshold,
                "prediction": strategy_3_pred,
                "confidence": round(strategy_3_confidence, 3),
                "value": round(mean_spoof_score, 4)
            },

            # Assessment
            "assessment": assessment
        }

    def _generate_assessment(self,
                            prediction: str,
                            spoof_ratio: float,
                            max_score: float,
                            mean_score: float,
                            spoof_windows: int,
                            total_windows: int) -> str:
        """Generate human-readable assessment."""

        if prediction == "BONAFIDE":
            if spoof_ratio == 0:
                return f"Clean audio. All {total_windows} windows detected as bonafide (authentic)."
            elif spoof_ratio < 0.1:
                return f"Mostly authentic with {spoof_windows} suspicious window(s) out of {total_windows}. Likely minor noise or artifacts."
            else:
                return f"Primarily authentic ({spoof_ratio*100:.1f}% suspicious windows). Some anomalies detected but overall consistent with genuine audio."
        else:  # SPOOF
            if spoof_ratio >= 0.5:
                return f"CRITICAL: Over 50% of audio detected as spoofed ({spoof_windows}/{total_windows} windows). Likely synthetic or heavily manipulated."
            elif spoof_ratio >= 0.25:
                return f"HIGH SUSPICION: {spoof_ratio*100:.1f}% of windows detected as spoof. Strong evidence of AI generation or voice manipulation."
            else:
                return f"SUSPICIOUS: {spoof_windows} out of {total_windows} windows show spoof characteristics. Recommend further analysis or human review."
