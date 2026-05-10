# main.py - AIDRA entry point with Critical Patient Constraint

import random
import time

from grid       import create_grid, print_grid, extract_positions
from search     import bfs, astar, hill_climbing, greedy_bfs, path_risk, path_cost, path_length
from agent      import (AIDRAgent, assign_severity, CSPNormal, CSPAllocator,
                        fuzzy_risk_level, fuzzy_blockage_prob, evaluate_ml_models,
                        is_critical)  # ← NEW import
from visualize  import plot_all

random.seed(7)

def section(title):
    print("\n" + "═"*62)
    print(f"  {title}")
    print("═"*62)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build Environment
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*62)
print("  AIDRA — ADAPTIVE INTELLIGENT DISASTER RESPONSE AGENT")
print("  CONSTRAINT: Each ambulance = MAX 1 critical patient")
print("═"*62)

grid = create_grid()
print_grid(grid)

start, victims, hospitals, blocked_list, danger_zones = extract_positions(grid)
blocked_set = set(blocked_list)

print(f"  Start      : {start}")
print(f"  Victims    : {victims}")
print(f"  Hospitals  : {hospitals}")
print(f"  Blocked    : {blocked_list}")
print(f"  Danger     : {danger_zones}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Victim Prioritization
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 2: VICTIM PRIORITIZATION")

victim_data    = assign_severity(victims)
agent = AIDRAgent(grid, start, hospitals)
priority_order = agent.prioritize_victims(victim_data)

# Identify critical victims for constraint reporting
critical_victims = [v for v in victims if is_critical(v, victim_data)]
print(f"\n  ⚠️ CRITICAL VICTIMS ({len(critical_victims)}): {critical_victims}")
print(f"  → Constraint: Each ambulance can take MAX 1 critical patient")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — CSP Resource Allocation: Normal vs MRV (with Critical Constraint)
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 3: CSP RESOURCE ALLOCATION — NORMAL vs MRV")
print("""
  CSP assigns victims to ambulances (max 2 each), overflow to RescueTeam.
  CONSTRAINT: Each ambulance carries MAX 1 critical patient.
  
  Normal BT: always tries A1 first — no smart ordering.
  MRV BT   : tries least-loaded ambulance first (smarter).
""")

# Pass victim_severity_map to CSP classes
csp_normal = CSPNormal(victim_data)  # ← UPDATED
t0 = time.perf_counter()
assignment_normal = csp_normal.solve(priority_order)
t1 = time.perf_counter()
csp_normal_time = round((t1 - t0) * 1000, 4)
csp_normal.print_plan()
print(f"  Runtime: {csp_normal_time} ms")

csp_mrv = CSPAllocator(victim_data)  # ← UPDATED
t0 = time.perf_counter()
assignment = csp_mrv.solve(priority_order)
t1 = time.perf_counter()
csp_mrv_time = round((t1 - t0) * 1000, 4)
csp_mrv.print_plan()
print(f"  Runtime: {csp_mrv_time} ms")

# Enhanced comparison table with critical constraint info
critical_in_normal = {amb: sum(1 for v in assignment_normal[amb] if is_critical(v, victim_data)) 
                      for amb in ["A1", "A2"]}
critical_in_mrv = {amb: sum(1 for v in assignment[amb] if is_critical(v, victim_data)) 
                   for amb in ["A1", "A2"]}

print(f"""
┌─ CSP COMPARISON (Max 1 Critical/Ambulance) ─────────────────────────────┐
│  Metric               Normal BT        MRV BT                │
│  ─────────────────────────────────────────────────────────── │
│  Backtracks         : {csp_normal.backtrack_count:<16} {csp_mrv.backtrack_count:<22}│
│  Total BT calls     : {csp_normal.calls:<16} {csp_mrv.calls:<22}│
│  Runtime (ms)       : {csp_normal_time:<16} {csp_mrv_time:<22}│
│  Critical in A1     : {critical_in_normal['A1']:<16} {critical_in_mrv['A1']:<22}│
│  Critical in A2     : {critical_in_normal['A2']:<16} {critical_in_mrv['A2']:<22}│
└──────────────────────────────────────────────────────────────┘
""")

# Validate constraint satisfaction
def validate_critical_constraint(assignment, victim_data):
    for amb in ["A1", "A2"]:
        critical_count = sum(1 for v in assignment[amb] if is_critical(v, victim_data))
        if critical_count > 1:
            return False, amb, critical_count
    return True, None, 0

valid_normal, bad_amb, crit_count = validate_critical_constraint(assignment_normal, victim_data)
if valid_normal:
    print("  ✅ Normal CSP: Critical constraint SATISFIED (max 1 per ambulance)")
else:
    print(f"  ❌ Normal CSP: VIOLATION — Ambulance {bad_amb} has {crit_count} critical patients!")

valid_mrv, bad_amb, crit_count = validate_critical_constraint(assignment, victim_data)
if valid_mrv:
    print("  ✅ MRV CSP: Critical constraint SATISFIED (max 1 per ambulance)")
else:
    print(f"  ❌ MRV CSP: VIOLATION — Ambulance {bad_amb} has {crit_count} critical patients!")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Route Planning: BFS vs A* vs Greedy vs Hill Climbing
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 4: SEARCH & ROUTE PLANNING")

kpi = {
    "victims_saved":     0,
    "total_astar_cost":  0,
    "total_bfs_cost":    0,
    "total_risk":        0,
    "optimality_ratios": [],
    "rescue_times": [],
}

# Collected per-victim data for visualizations
search_data = {
    "bfs_costs":    [],
    "as_costs":     [],
    "as_steps":     [],
    "as_danger":    [],
    "knn_survival": [],
    "nb_survival":  [],
}

chosen_paths = {}

def nearest_hospital(pos):
    return min(hospitals, key=lambda h: abs(pos[0]-h[0])+abs(pos[1]-h[1]), default=None)

for rank, victim in enumerate(priority_order, 1):
    info = victim_data[victim]
    severity = info['severity']
    critical_mark = "⚠️ CRITICAL ⚠️" if severity == "critical" else ""
    print(f"\n{'─'*60}")
    print(f"  Victim #{rank}  at {victim}   Severity: {severity.upper()} {critical_mark}")
    print(f"{'─'*60}")

    t0 = time.perf_counter(); bfs_path = bfs(grid, start, victim, blocked_set);        t1 = time.perf_counter(); bfs_time = (t1-t0)*1000
    t0 = time.perf_counter(); as_path  = astar(grid, start, victim, blocked_set);      t1 = time.perf_counter(); as_time  = (t1-t0)*1000
    t0 = time.perf_counter(); gr_path  = greedy_bfs(grid, start, victim, blocked_set); t1 = time.perf_counter(); gr_time  = (t1-t0)*1000
    t0 = time.perf_counter(); hc_path  = hill_climbing(grid, start, victim, blocked_set); t1 = time.perf_counter(); hc_time = (t1-t0)*1000

    def fmt(path, target):
        if not path or path[-1] != target:
            return None, None, None, "✗"
        return path_length(path), path_risk(path, grid), path_cost(path, grid), "✓"

    bfs_l, bfs_r, bfs_c, bfs_ok = fmt(bfs_path, victim)
    as_l,  as_r,  as_c,  as_ok  = fmt(as_path,  victim)
    gr_l,  gr_r,  gr_c,  gr_ok  = fmt(gr_path,  victim)
    hc_l,  hc_r,  hc_c,  hc_ok  = fmt(hc_path,  victim)

    print(f"\n  {'Algorithm':<18} {'Steps':>6} {'Danger':>7} {'Cost':>7} {'Time(ms)':>10} {'OK':>4}")
    print(f"  {'─'*54}")
    for name, l, r, c, tm, ok in [
        ("BFS",           bfs_l, bfs_r, bfs_c, bfs_time, bfs_ok),
        ("A*",            as_l,  as_r,  as_c,  as_time,  as_ok),
        ("Greedy BFS",    gr_l,  gr_r,  gr_c,  gr_time,  gr_ok),
        ("Hill Climbing", hc_l,  hc_r,  hc_c,  hc_time,  hc_ok),
    ]:
        ls = str(l) if l is not None else "N/A"
        rs = str(r) if r is not None else "N/A"
        cs = str(c) if c is not None else "N/A"
        print(f"  {name:<18} {ls:>6} {rs:>7} {cs:>7} {tm:>10.3f} {ok:>4}")

    # Insights
    if bfs_c is not None and as_c is not None and bfs_c != as_c:
        print(f"\n  ℹ  BFS cost {bfs_c} vs A* cost {as_c} — A* avoids danger at cost of extra steps.")
    if hc_ok == "✗":
        print(f"  ⚠  Hill Climbing STUCK — local optimum reached.")

    # Fuzzy
    if as_r is not None and as_l and as_l > 0:
        fl, fc = fuzzy_risk_level(as_r, as_l)
        print(f"\n  Fuzzy Risk (A*)  : {fl}  (confidence {fc})")
    if bfs_r is not None and bfs_l and bfs_l > 0:
        fl2, fc2 = fuzzy_risk_level(bfs_r, bfs_l)
        print(f"  Fuzzy Risk (BFS) : {fl2}  (confidence {fc2})")
    shock = round(random.uniform(3, 9), 1)
    fire  = round(random.uniform(2, 8), 1)
    print(f"  Blockage Prob    : {fuzzy_blockage_prob(shock, fire)}  (aftershock={shock}, fire={fire})")

    # ========== AI DECISION BASED ON SEVERITY ==========
    # ========== AI DECISION BASED ON SEVERITY (FIXED LOGIC) ==========
    print(f"\n  🧠 AI DECISION:")
    chosen_path, chosen_name, reason = None, "NONE", "No path found."

    if as_path and as_path[-1] == victim and bfs_path and bfs_path[-1] == victim:

    # ─────────────────────────────
    # CRITICAL → MIN STEPS ONLY
    # ─────────────────────────────
        if severity == "critical":

            if bfs_l <= as_l:
                chosen_path, chosen_name = bfs_path, "BFS"
                reason = f"CRITICAL → BFS chosen (fewer steps: {bfs_l} vs {as_l})"
            else:
                chosen_path, chosen_name = as_path, "A*"
                reason = f"CRITICAL → A* chosen (fewer steps: {as_l} vs {bfs_l})"

    # ─────────────────────────────
    # MODERATE → MIN COST ONLY
    # ─────────────────────────────
        elif severity == "moderate":

            if as_c < bfs_c:
                chosen_path, chosen_name = as_path, "A*"
                reason = f"MODERATE → A* chosen (lower cost: {as_c} vs {bfs_c})"

            elif bfs_c < as_c:
                chosen_path, chosen_name = bfs_path, "BFS"
                reason = f"MODERATE → BFS chosen (lower cost: {bfs_c} vs {as_c})"

            else:
            # tie → BFS (faster exploration, as per your rule)
                chosen_path, chosen_name = bfs_path, "BFS"
                reason = f"MODERATE → cost tie, BFS chosen (faster fallback)"

    # ─────────────────────────────
    # MINOR → MIN COST ONLY (NOT RISK)
    # tie → BFS
    # ─────────────────────────────
        else:

            if as_c < bfs_c:
                chosen_path, chosen_name = as_path, "A*"
                reason = f"MINOR → A* chosen (lower cost: {as_c} vs {bfs_c})"

            elif bfs_c < as_c:
                chosen_path, chosen_name = bfs_path, "BFS"
                reason = f"MINOR → BFS chosen (lower cost: {bfs_c} vs {as_c})"

            else:
                chosen_path, chosen_name = bfs_path, "BFS"
                reason = f"MINOR → cost tie, BFS chosen (faster fallback)"

    elif as_path and as_path[-1] == victim:
        chosen_path, chosen_name = as_path, "A*"
        reason = "Only A* found a path."

    elif bfs_path and bfs_path[-1] == victim:
        chosen_path, chosen_name = bfs_path, "BFS"
        reason = "Only BFS found a path."

    print(f"  ✅ Chosen : {chosen_name}")
    print(f"     Reason : {reason}")
    # ML Survival
    if chosen_path:
        knn_p, nb_p, combined = agent.estimate_survival(chosen_path, severity)
        print(f"\n  ML Survival Estimate:")
        print(f"     kNN         : {knn_p*100:.0f}%")
        print(f"     Naive Bayes : {nb_p*100:.0f}%")
        print(f"     Combined    : {combined*100:.0f}%")
        if combined < 0.5:
            print(f"     ⚠  HIGH RISK — prioritize immediate transport!")
        search_data["knn_survival"].append(knn_p)
        search_data["nb_survival"].append(nb_p)
    else:
        search_data["knn_survival"].append(0.0)
        search_data["nb_survival"].append(0.0)

    # Collect visualization data
    search_data["bfs_costs"].append(bfs_c if bfs_c is not None else 0)
    search_data["as_costs"].append(as_c   if as_c  is not None else 0)
    search_data["as_steps"].append(as_l   if as_l  is not None else 0)
    search_data["as_danger"].append(as_r  if as_r  is not None else 0)

    # KPI tracking
    if chosen_path:
        chosen_paths[victim] = chosen_path
        kpi["victims_saved"] += 1
        kpi["total_risk"]    += path_risk(chosen_path, grid)
        kpi["rescue_times"].append(path_length(chosen_path))
        if as_c and bfs_c:
            kpi["optimality_ratios"].append(round(path_cost(chosen_path, grid) / min(as_c, bfs_c), 2))
            kpi["total_astar_cost"] += as_c
            kpi["total_bfs_cost"]   += bfs_c

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — DYNAMIC SIMULATION WITH REPLANNING (DUAL-VICTIM RESCUES)
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 5: DYNAMIC SIMULATION — REPLANNING")

from simulation import simulate_rescue, resource_pool

rescue_results = []

# Get victims assigned to each ambulance from CSP
ambulance_a1_victims = assignment["A1"]  # e.g., [(1,2), (2,4)]
ambulance_a2_victims = assignment["A2"]  # e.g., [(5,2), (4,3)]

# Simulate A1's dual-victim mission
if len(ambulance_a1_victims) >= 2:
    v1a, v1b = ambulance_a1_victims[0], ambulance_a1_victims[1]
    print(f"\n  🚑 AMBULANCE A1 — Rescuing {v1a} → {v1b}")
    
    success, steps = simulate_rescue(
        grid, blocked_set, start,
        v1a, victim_data[v1a]["severity"],
        v1b, victim_data[v1b]["severity"],
        agent, ambulance="A1"
    )
    rescue_results.append(((v1a, v1b), success, steps))

# Simulate A2's dual-victim mission (with optional blockage)
if len(ambulance_a2_victims) >= 2:
    v2a, v2b = ambulance_a2_victims[0], ambulance_a2_victims[1]
    print(f"\n  🚑 AMBULANCE A2 — Rescuing {v2a} → {v2b}")
    
    # Optional: force a blockage during A2's rescue for demonstration
    path_v2a = astar(grid, start, v2a, blocked_set)
    force_cell = None
    if path_v2a and len(path_v2a) > 3:
        from grid import EMPTY as _EMPTY
        for candidate in path_v2a[2:-1]:
            cr, cc = candidate
            if grid[cr][cc] == _EMPTY:
                force_cell = candidate
                break
    
    success, steps = simulate_rescue(
        grid, blocked_set, start,
        v2a, victim_data[v2a]["severity"],
        v2b, victim_data[v2b]["severity"],
        agent, ambulance="A2", force_block_at=force_cell
    )
    rescue_results.append(((v2a, v2b), success, steps))

# Handle RescueTeam victims (if any)
if assignment["RescueTeam"]:
    print(f"\n  👥 RESCUE TEAM — Handling {len(assignment['RescueTeam'])} victim(s): {assignment['RescueTeam']}")
    for v in assignment["RescueTeam"]:
        # RescueTeam treats victims on-site (no transport to hospital)
        from simulation import _allocate_kits
        if _allocate_kits(victim_data[v]["severity"], v, agent):
            print(f"  ✅ RescueTeam treated victim at {v}")
            rescue_results.append((v, True, 0))
        else:
            print(f"  ❌ RescueTeam cannot treat victim at {v} (no kits)")
            rescue_results.append((v, False, 0))

print(f"\n  📦 Medical Kits Remaining after all rescues: {resource_pool.kits}/10")

# Print rescue summary
print("\n  ── RESCUE SUMMARY ──")
for result in rescue_results:
    if len(result) == 3 and isinstance(result[0], tuple):
        (v1, v2), success, steps = result
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} | Ambulance rescued {v1} → {v2} | Steps: {steps}")
    else:
        v, success, steps = result
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} | RescueTeam treated {v}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — ML Model Evaluation
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 6: ML MODEL EVALUATION")

print("""
  Models: kNN (k=5) and Gaussian Naive Bayes.
  Features: danger cells on path, distance to hospital.
  Label:    1 = survives,  0 = at-risk.
""")

ml_results = evaluate_ml_models()

print(f"  {'Model':<15} {'Acc':>6} {'Prec':>6} {'Recall':>8} {'F1':>6} {'Spec':>7} {'AvgConf':>9}")
print(f"  {'─'*60}")
for model, m in ml_results.items():
    print(f"  {model:<15} {m['Accuracy']:>6} {m['Precision']:>6} {m['Recall']:>8} "
          f"{m['F1']:>6} {m['Specificity']:>7} {m['Avg_Conf']:>9}")

print(f"\n  ── Confusion Matrices ──────────────────────────────────────")
for model, m in ml_results.items():
    print(f"\n  [{model}]")
    print(f"                     Predicted SURVIVE   Predicted AT-RISK")
    print(f"  Actual SURVIVE  :       {m['TP']:>5}              {m['FN']:>5}")
    print(f"  Actual AT-RISK  :       {m['FP']:>5}              {m['TN']:>5}")

knn_f1 = ml_results["kNN"]["F1"]
nb_f1  = ml_results["Naive Bayes"]["F1"]
winner = "kNN" if knn_f1 >= nb_f1 else "Naive Bayes"
print(f"\n  🏆 Best model (by F1): {winner}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — KPI Summary
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 7: PERFORMANCE KPI REPORT")

n               = len(victims)
victims_saved   = kpi["victims_saved"]
save_rate       = round(victims_saved / n * 100, 1)
avg_rescue_time = round(sum(kpi["rescue_times"]) / len(kpi["rescue_times"]), 2) if kpi["rescue_times"] else 0
cost_saving     = kpi["total_bfs_cost"] - kpi["total_astar_cost"]
avg_opt         = round(sum(kpi["optimality_ratios"]) / len(kpi["optimality_ratios"]), 2) if kpi["optimality_ratios"] else 1.0
amb_used        = sum(1 for a in ["A1","A2"] if assignment[a])

# Count critical distribution
critical_in_final = {amb: sum(1 for v in assignment[amb] if is_critical(v, victim_data)) 
                     for amb in ["A1", "A2", "RescueTeam"]}

print(f"""
  Victims Saved          : {victims_saved}/{n}  ({save_rate}%)
  Avg Rescue Time        : {avg_rescue_time} steps
  Ambulances Dispatched  : {amb_used}/2
  Medical Kits Used      : {10 - resource_pool.kits}/10  ({resource_pool.kits} remaining)
  A* Total Cost          : {kpi['total_astar_cost']}
  BFS Total Cost         : {kpi['total_bfs_cost']}
  Cost Saved by A*       : {cost_saving} units
  Path Optimality (avg)  : {avg_opt}
  Total Danger Exposure  : {kpi['total_risk']} cells
  CSP Normal Backtracks  : {csp_normal.backtrack_count}
  CSP MRV Backtracks     : {csp_mrv.backtrack_count}
  
  ── Critical Patient Distribution (MRV Assignment) ──
  A1           : {critical_in_final['A1']} critical patient(s) [MAX 1]
  A2           : {critical_in_final['A2']} critical patient(s) [MAX 1]
  RescueTeam   : {critical_in_final['RescueTeam']} critical patient(s) [No limit]
""")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — Decision Log
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 8: FULL DECISION LOG")
agent.print_decision_log()

# ════════════════════════════════════════════════════════════════════════════════
# STEP 9 — Generate KPI Graphs
# ════════════════════════════════════════════════════════════════════════════════
section("STEP 9: GENERATING KPI GRAPHS")

plot_all(
    kpi        = kpi,
    ml_results = ml_results,
    csp_normal = csp_normal,
    csp_mrv    = csp_mrv,
    search_data= search_data,
    victims    = priority_order,
    assignment = assignment,
)

print("\n" + "═"*62)
print("  AIDRA SIMULATION COMPLETE")
print("═"*62 + "\n")