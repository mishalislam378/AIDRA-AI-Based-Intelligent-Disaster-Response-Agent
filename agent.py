# agent.py - AIDRA Intelligent Agent
# Victim prioritization, CSP (Normal vs MRV), ML (kNN + Naive Bayes via sklearn), Fuzzy Logic, Decision Log
# CONSTRAINT: Each ambulance can carry maximum ONE critical patient

import os
import math
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

SEVERITY_SCORE = {"critical": 3, "moderate": 2, "minor": 1}

# ── Load Dataset from CSV ─────────────────────────────────────────────────────
# Features: danger_cells, dist_to_hospital, severity_score, response_time
# Label:    survived (1 = survives, 0 = at-risk)
#
# The CSV (disaster_rescue_dataset.csv) contains ~550 entries generated from
# realistic disaster-rescue feature distributions. Place it in the same
# directory as agent.py, or set the AIDRA_DATASET env variable to its path.

_CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "disaster_rescue_dataset.csv")
_CSV_PATH = os.environ.get("AIDRA_DATASET", _CSV_DEFAULT)

def _load_csv(path):
    X, y = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([
                float(row["danger_cells"]),
                float(row["dist_to_hospital"]),
                float(row["severity_score"]),
                float(row["response_time"]),
            ])
            y.append(int(row["survived"]))
    return np.array(X, dtype=float), np.array(y)

_X_all, _y_all = _load_csv(_CSV_PATH)

# ── 80/20 Train-Test Split (stratified to preserve class balance) ─────────────
_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X_all, _y_all, test_size=0.20, random_state=42, stratify=_y_all
)

print(f"  📂 Dataset loaded: {len(_X_all)} entries from {os.path.basename(_CSV_PATH)}")
print(f"     Train: {len(_X_train)} samples  |  Test: {len(_X_test)} samples")
print(f"     Features: danger_cells, dist_to_hospital, severity_score, response_time")

# ── Fit sklearn models on TRAINING split only ─────────────────────────────────
_knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
_knn_model.fit(_X_train, _y_train)

_nb_model = GaussianNB()
_nb_model.fit(_X_train, _y_train)

print("  ✅ sklearn KNN (k=5) and Gaussian Naive Bayes models trained & ready.")


# ════════════════════════════════════════════════════════════════════════════════
# ML Prediction Functions (now using sklearn)
# ════════════════════════════════════════════════════════════════════════════════
def knn_predict(danger_on_path, dist_hospital, severity_score=2, response_time=5, k=5):
    """
    kNN survival prediction using sklearn KNeighborsClassifier.
    Features: danger_cells, dist_to_hospital, severity_score, response_time.
    Returns probability of survival (class=1).
    """
    features = np.array([[danger_on_path, dist_hospital, severity_score, response_time]], dtype=float)
    prob = _knn_model.predict_proba(features)[0]
    classes = list(_knn_model.classes_)
    survive_idx = classes.index(1)
    return round(float(prob[survive_idx]), 2)


def naive_bayes_predict(danger_on_path, dist_hospital, severity_score=2, response_time=5):
    """
    Gaussian Naive Bayes survival prediction using sklearn GaussianNB.
    Features: danger_cells, dist_to_hospital, severity_score, response_time.
    Returns probability of survival (class=1).
    """
    features = np.array([[danger_on_path, dist_hospital, severity_score, response_time]], dtype=float)
    prob = _nb_model.predict_proba(features)[0]
    classes = list(_nb_model.classes_)
    survive_idx = classes.index(1)
    return round(float(prob[survive_idx]), 2)


def evaluate_ml_models():
    """
    Evaluate both sklearn models on the held-out TEST set (20% split).
    Returns dict of metrics matching the original structure.
    This gives honest generalisation metrics — not inflated train-set scores.
    """
    results = {}
    for name, model in [("kNN", _knn_model), ("Naive Bayes", _nb_model)]:
        tp = fp = tn = fn_count = 0
        probabilities = []

        for features, true_label in zip(_X_test, _y_test):
            prob_arr = model.predict_proba(features.reshape(1, -1))[0]
            classes  = list(model.classes_)
            survive_idx = classes.index(1)
            prob = float(prob_arr[survive_idx])
            pred = 1 if prob >= 0.5 else 0
            probabilities.append(prob)

            if   pred == 1 and true_label == 1: tp += 1
            elif pred == 1 and true_label == 0: fp += 1
            elif pred == 0 and true_label == 0: tn += 1
            else:                               fn_count += 1

        total    = len(_y_test)
        acc      = (tp + tn) / total
        prec     = tp / (tp + fp + 1e-9)
        rec      = tp / (tp + fn_count + 1e-9)
        f1       = 2 * prec * rec / (prec + rec + 1e-9)
        spec     = tn / (tn + fp + 1e-9)
        avg_conf = sum(probabilities) / len(probabilities)

        results[name] = {
            "Accuracy":    round(acc,      2),
            "Precision":   round(prec,     2),
            "Recall":      round(rec,      2),
            "F1":          round(f1,       2),
            "Specificity": round(spec,     2),
            "Avg_Conf":    round(avg_conf, 2),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn_count,
        }
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Victim Severity
# ════════════════════════════════════════════════════════════════════════════════
def assign_severity(victims):
    severities = ["critical", "moderate", "minor", "moderate", "critical"]
    data = {}
    for i, v in enumerate(victims):
        sev = severities[i] if i < len(severities) else "minor"
        data[v] = {"severity": sev, "score": SEVERITY_SCORE[sev]}
    return data


# ════════════════════════════════════════════════════════════════════════════════
# Fuzzy Logic
# ════════════════════════════════════════════════════════════════════════════════
def fuzzy_risk_level(danger_cells, path_len):
    if path_len == 0:
        return "UNKNOWN", 0.0
    ratio  = danger_cells / path_len
    low    = max(0.0, min(1.0, (0.20 - ratio) / 0.20))
    medium = max(0.0, min(ratio / 0.20, (0.50 - ratio) / 0.20))
    high   = max(0.0, min(1.0, (ratio - 0.40) / 0.30))
    scores = {"LOW": low, "MEDIUM": medium, "HIGH": high}
    label  = max(scores, key=scores.get)
    return label, round(scores[label], 2)


def fuzzy_blockage_prob(aftershock, fire_spread):
    prob = min(1.0, (aftershock * 0.4 + fire_spread * 0.6) / 10)
    return round(prob, 2)


# ════════════════════════════════════════════════════════════════════════════════
# Helper: Check if a victim is critical
# ════════════════════════════════════════════════════════════════════════════════
def is_critical(victim_pos, victim_severity_map):
    return victim_severity_map.get(victim_pos, {}).get("severity") == "critical"


# ════════════════════════════════════════════════════════════════════════════════
# CSP — Normal Backtracking (with Critical Constraint)
# CONSTRAINT: Each ambulance can take AT MOST ONE critical patient
# ════════════════════════════════════════════════════════════════════════════════
class CSPNormal:
    MAX_AMB = 2
    MAX_RT  = 1
    MAX_CRITICAL_PER_AMB = 1

    def __init__(self, victim_severity_map):
        self.assignment      = {"A1": [], "A2": [], "RescueTeam": []}
        self.backtrack_count = 0
        self.calls           = 0
        self.victim_severity = victim_severity_map

    def crit_count(self, amb_name):
        return sum(1 for v in self.assignment[amb_name] if is_critical(v, self.victim_severity))

    def can_assign_to_amb(self, amb_name, victim):
        if len(self.assignment[amb_name]) >= self.MAX_AMB:
            return False
        if is_critical(victim, self.victim_severity):
            if self.crit_count(amb_name) >= self.MAX_CRITICAL_PER_AMB:
                return False
        return True

    def solve(self, priority_victims):
        self._bt(list(priority_victims), 0)
        return self.assignment

    def _bt(self, victims, idx):
        self.calls += 1
        if idx == len(victims):
            return True
        v = victims[idx]
        for amb in ["A1", "A2"]:
            if self.can_assign_to_amb(amb, v):
                self.assignment[amb].append(v)
                if self._bt(victims, idx + 1):
                    return True
                self.assignment[amb].pop()
                self.backtrack_count += 1
        if len(self.assignment["RescueTeam"]) < self.MAX_RT:
            self.assignment["RescueTeam"].append(v)
            if self._bt(victims, idx + 1):
                return True
            self.assignment["RescueTeam"].pop()
            self.backtrack_count += 1
        return False

    def print_plan(self, label="NORMAL BACKTRACKING (Max 1 Critical/Ambulance)"):
        print(f"\n┌─ CSP {label} ──────────────────────────────────────────────┐")
        for res, vs in self.assignment.items():
            critical_count = sum(1 for v in vs if is_critical(v, self.victim_severity))
            critical_mark = f" ⚠️[{critical_count}/1 critical]" if res != "RescueTeam" else ""
            print(f"│  {res:12} → {vs if vs else '—'}{critical_mark}")
        print(f"│  Constraint: Each ambulance → MAX 1 critical patient")
        print(f"│  Backtracks used  : {self.backtrack_count}")
        print(f"│  Total BT calls   : {self.calls}")
        print("└────────────────────────────────────────────────────────────┘")


# ════════════════════════════════════════════════════════════════════════════════
# CSP — MRV Backtracking (with Critical Constraint)
# ════════════════════════════════════════════════════════════════════════════════
class CSPAllocator:
    MAX_AMB = 2
    MAX_RT  = 1
    MAX_CRITICAL_PER_AMB = 1

    def __init__(self, victim_severity_map):
        self.assignment      = {"A1": [], "A2": [], "RescueTeam": []}
        self.backtrack_count = 0
        self.calls           = 0
        self.victim_severity = victim_severity_map

    def crit_count(self, amb_name):
        return sum(1 for v in self.assignment[amb_name] if is_critical(v, self.victim_severity))

    def can_assign_to_amb(self, amb_name, victim):
        if len(self.assignment[amb_name]) >= self.MAX_AMB:
            return False
        if is_critical(victim, self.victim_severity):
            if self.crit_count(amb_name) >= self.MAX_CRITICAL_PER_AMB:
                return False
        return True

    def solve(self, priority_victims):
        self._bt(list(priority_victims), 0)
        return self.assignment

    def _bt(self, victims, idx):
        self.calls += 1
        if idx == len(victims):
            return True
        v = victims[idx]
        ambs = sorted(["A1", "A2"], key=lambda a: len(self.assignment[a]))
        for amb in ambs:
            if self.can_assign_to_amb(amb, v):
                self.assignment[amb].append(v)
                if self._bt(victims, idx + 1):
                    return True
                self.assignment[amb].pop()
                self.backtrack_count += 1
        if len(self.assignment["RescueTeam"]) < self.MAX_RT:
            self.assignment["RescueTeam"].append(v)
            if self._bt(victims, idx + 1):
                return True
            self.assignment["RescueTeam"].pop()
            self.backtrack_count += 1
        return False

    def print_plan(self, label="MRV BACKTRACKING (Max 1 Critical/Ambulance)"):
        print(f"\n┌─ CSP {label} ──────────────────────────────────────────────┐")
        for res, vs in self.assignment.items():
            critical_count = sum(1 for v in vs if is_critical(v, self.victim_severity))
            critical_mark = f" ⚠️[{critical_count}/1 critical]" if res != "RescueTeam" else ""
            print(f"│  {res:12} → {vs if vs else '—'}{critical_mark}")
        print(f"│  Constraint: Each ambulance → MAX 1 critical patient")
        print(f"│  Backtracks used  : {self.backtrack_count}")
        print(f"│  Total BT calls   : {self.calls}")
        print("└────────────────────────────────────────────────────────────┘")


# ════════════════════════════════════════════════════════════════════════════════
# AIDRA Agent
# ════════════════════════════════════════════════════════════════════════════════
class AIDRAgent:

    def __init__(self, grid, start, hospitals):
        self.grid      = grid
        self.start     = start
        self.hospitals = hospitals
        self.log       = []

    def log_decision(self, event, reason, action):
        self.log.append({"event": event, "reason": reason, "action": action})
        print(f"\n  📋 DECISION LOG ENTRY:")
        print(f"     Event  : {event}")
        print(f"     Reason : {reason}")
        print(f"     Action : {action}")

    def prioritize_victims(self, victim_data):
        sorted_v = sorted(victim_data.items(), key=lambda x: x[1]["score"], reverse=True)
        print("\n┌─ VICTIM PRIORITY ORDER (Severity-First) ───────────────────┐")
        for rank, (pos, info) in enumerate(sorted_v, 1):
            print(f"│  #{rank}  Pos {pos}  │  Severity: {info['severity'].upper():8} │  Score: {info['score']}")
        print("└────────────────────────────────────────────────────────────┘")
        return [v[0] for v in sorted_v]

    def estimate_survival(self, path, severity=None):
        from grid import DANGER
        danger = sum(1 for r, c in path if self.grid[r][c] == DANGER)
        victim = path[-1]
        dist   = min(abs(victim[0] - h[0]) + abs(victim[1] - h[1]) for h in self.hospitals)

        # Map severity string → numeric score for the model
        sev_score = SEVERITY_SCORE.get(severity, 2) if severity else 2
        # Approximate response time from path length
        response_time = len(path)

        # ── sklearn predictions (4 features) ──
        knn = knn_predict(danger, dist, sev_score, response_time)
        nb  = naive_bayes_predict(danger, dist, sev_score, response_time)
        base = (knn + nb) / 2

        # Cap by severity
        if severity == "critical":
            base = max(0.15, min(0.30, base))
        elif severity == "moderate":
            base = max(0.55, min(0.67, base))
        elif severity == "minor":
            base = max(0.80, min(0.93, base))

        combined = round(base, 2)
        return knn, nb, combined

    def print_decision_log(self):
        print("\n┌─ FULL DECISION LOG ─────────────────────────────────────────┐")
        if not self.log:
            print("│  No replanning events recorded.")
        for i, e in enumerate(self.log, 1):
            print(f"│  [{i}] {e['event']}")
            print(f"│      Reason : {e['reason']}")
            print(f"│      Action : {e['action']}")
        print("└───────────────────────────")