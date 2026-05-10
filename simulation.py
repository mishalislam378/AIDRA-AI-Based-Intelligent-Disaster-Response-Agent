# simulation.py - Dynamic rescue simulation with replanning
# Full route: Start → Victim 1 → Victim 2 → Hospital
# Kit allocation: critical=2, moderate=2, minor=1

from grid   import EMPTY, block_road
from search import astar
import random

random.seed(42)

# ── Global Medical Kit Resource Pool ──────────────────────────────────────────
MEDICAL_KITS_AVAILABLE = 10   # Hard constraint from problem statement

# Kit cost per severity
KITS_REQUIRED = {
    "critical": 2,
    "moderate": 2,
    "minor":    1,
}

class ResourcePool:
    """Tracks shared medical kit supply across all rescue missions."""
    def __init__(self, kits=10):
        self.kits = kits
        self.depleted_event_fired = False

    def consume(self, amount):
        """Consume `amount` kits. Returns True if kits were available."""
        if self.kits >= amount:
            self.kits -= amount
            return True
        return False

    def is_low(self):
        return self.kits <= 2

resource_pool = ResourcePool(MEDICAL_KITS_AVAILABLE)


def nearest_hospital(hospitals, pos):
    """Return the closest hospital to `pos` by Manhattan distance."""
    return min(hospitals, key=lambda h: abs(pos[0]-h[0]) + abs(pos[1]-h[1]))


def _allocate_kits(severity, target, agent):
    """
    Attempt to allocate kits for a victim based on severity and current supply.
    Returns True if kits were successfully allocated, False otherwise.
    """
    kits_needed = KITS_REQUIRED.get(severity, 1)
    kits_left   = resource_pool.kits

    if resource_pool.is_low():
        print(f"\n  🚨 DISASTER EVENT: Medical kits critically low — only {kits_left} kit(s) left!")

        if not resource_pool.depleted_event_fired:
            resource_pool.depleted_event_fired = True

        if severity == "critical":
            if resource_pool.consume(kits_needed):
                print(f"  ⚠️  DECISION : Kits critically low ({kits_left} left), BUT victim is CRITICAL.")
                print(f"  ✅  ACTION   : Using {kits_needed} kit(s) — life > future reservation.")
                agent.log_decision(
                    event  = f"Kit supply critically low ({kits_left} left) at victim {target}",
                    reason = "Victim is CRITICAL — immediate treatment outweighs reserving supply. "
                             "Delaying a critical patient risks certain death.",
                    action = f"Use {kits_needed} kit(s) now. Request emergency resupply."
                )
            else:
                print(f"  ❌ RESOURCE EVENT: Zero kits — even CRITICAL victim {target} cannot be treated!")
                agent.log_decision(
                    event  = f"Kits fully exhausted for CRITICAL victim at {target}",
                    reason = "Supply completely depleted by prior missions",
                    action = "Emergency resupply required. Victim logged as untreatable."
                )
                return False

        elif severity == "moderate":
            if resource_pool.consume(kits_needed):
                print(f"  ⚠️  DECISION : Kits low ({kits_left} left), victim is MODERATE severity.")
                print(f"  ✅  ACTION   : Proceeding — moderate victim cannot be left without treatment.")
                agent.log_decision(
                    event  = f"Kit supply low ({kits_left} left) at victim {target}",
                    reason = "Victim is MODERATE — deferring risks serious deterioration. "
                             "No future victims known to need kits more urgently.",
                    action = f"Use {kits_needed} kit(s). {resource_pool.kits} kit(s) remain after this mission."
                )
            else:
                print(f"  ❌ RESOURCE EVENT: Zero kits — MODERATE victim {target} cannot be treated!")
                agent.log_decision(
                    event  = f"Kits exhausted for MODERATE victim at {target}",
                    reason = "All 10 kits consumed",
                    action = "Skip victim. Log for resupply. Attempt transport-only rescue."
                )
                return False

        else:  # MINOR
            if resource_pool.kits >= kits_needed:
                resource_pool.consume(kits_needed)
                print(f"  ⚠️  DECISION : Kits low ({kits_left} left), victim is MINOR (needs {kits_needed} kit).")
                print(f"  ✅  ACTION   : Using {kits_needed} kit — minor still requires treatment.")
                agent.log_decision(
                    event  = f"Kit supply low ({kits_left} left) — victim {target} is MINOR",
                    reason = f"Minor needs only {kits_needed} kit; supply allows it.",
                    action = f"Use {kits_needed} kit. {resource_pool.kits} kit(s) remain."
                )
            else:
                print(f"  🛑  DECISION : Kits critically low ({kits_left} left), victim is MINOR.")
                print(f"  ⏭️   ACTION   : SKIPPING — reserving last kits for critical/moderate victims.")
                agent.log_decision(
                    event  = f"Kit supply critically low ({kits_left} left) — victim {target} is MINOR",
                    reason = "With only insufficient kits, reserving them for critical/moderate victims. "
                             "Minor victims can self-triage.",
                    action = "Reserve kits. Redirect minor victim to self-triage protocol. Skip rescue."
                )
                return False

    else:
        # Plenty of kits — normal consumption
        resource_pool.consume(kits_needed)
        print(f"  🩺 Kits used: {kits_needed}  |  Kits remaining: {resource_pool.kits}")

    return True


def _travel_leg(grid, blocked_set, current, destination, leg_label,
                steps_done, agent, force_block_at=None):
    """
    Move the ambulance from `current` to `destination` along an A* path.
    Handles mid-route blockage and replanning.

    Returns: (reached: bool, new_current: tuple, steps_done: int)
    """
    path = astar(grid, current, destination, blocked_set)
    if not path or path[-1] != destination:
        print(f"  ❌ No path from {current} to {destination}. Destination unreachable.")
        agent.log_decision(
            event  = f"No path from {current} to {destination}",
            reason = "All routes blocked",
            action = "Skip this leg, reassign resources"
        )
        return False, current, steps_done

    print(f"\n  📍 {leg_label}")
    print(f"     A* path: {path}")

    pos = current
    for step_idx, step in enumerate(path[1:], 1):
        print(f"  Step {steps_done + step_idx}: Moving to {step}", end="")
        steps_done += 1

        # Force a blockage mid-route
        if force_block_at and step == force_block_at:
            remaining = path[path.index(step):]
            for ahead in remaining[1:]:
                ar, ac = ahead
                if grid[ar][ac] == EMPTY:
                    block_road(grid, blocked_set, ahead)
                    print(f"\n\n  🔥 DISASTER EVENT: Road at {ahead} blocked! (aftershock)")
                    agent.log_decision(
                        event  = f"Road blocked at {ahead} during leg to {destination}",
                        reason = "Aftershock — original route now impassable",
                        action = f"Replanning A* from {step} to {destination}"
                    )
                    pos      = step
                    new_path = astar(grid, pos, destination, blocked_set)
                    if new_path and new_path[-1] == destination:
                        print(f"  🔁 Replanned path: {new_path}")
                        for new_step in new_path[1:]:
                            steps_done += 1
                            print(f"  Step (replan): Moving to {new_step}")
                        pos = destination
                    else:
                        print(f"  ❌ No alternative path to {destination} — leg aborted!")
                        agent.log_decision(
                            event  = f"Replan failed from {pos} to {destination}",
                            reason = "All alternate routes blocked",
                            action = "Abandon this leg."
                        )
                        return False, pos, steps_done
                    break
        else:
            print()
        pos = step

    reached = (pos == destination)
    return reached, pos, steps_done


def simulate_rescue(grid, blocked_set, start,
                    target1, victim1_label,
                    target2, victim2_label,
                    agent, ambulance="A1",
                    force_block_at=None):
    """
    Simulate ambulance picking up TWO victims before going to hospital.
    Full route: Start → Victim 1 → Victim 2 → Nearest Hospital

    Kit allocation:
        critical → 2 kits
        moderate → 2 kits
        minor    → 1 kit

    Returns: (success: bool, steps_taken: int)
    """
    hospitals = agent.hospitals

    kits1 = KITS_REQUIRED.get(victim1_label, 1)
    kits2 = KITS_REQUIRED.get(victim2_label, 1)

    print(f"\n  ══ Dual-Victim Rescue Mission using {ambulance} ══")
    print(f"     Route plan  : {start}  →  {target1}  →  {target2}  →  hospital")
    print(f"     Victim 1    : {target1} ({victim1_label.upper()}) — needs {kits1} kit(s)")
    print(f"     Victim 2    : {target2} ({victim2_label.upper()}) — needs {kits2} kit(s)")

    steps_done = 0
    current    = start

    # ══════════════════════════════════════════════════════════════════════════
    # VICTIM 1 — Kit allocation + travel
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ── Victim 1: {target1} ({victim1_label.upper()}) ──")
    print(f"     Kits needed : {kits1}")

    if not _allocate_kits(victim1_label, target1, agent):
        print(f"  ❌ Cannot allocate kits for Victim 1 at {target1}. Mission aborted.")
        return False, steps_done

    reached1, current, steps_done = _travel_leg(
        grid, blocked_set, current, target1,
        leg_label  = f"LEG 1 — {start} → Victim 1 at {target1}",
        steps_done = steps_done,
        agent      = agent,
        force_block_at = force_block_at,
    )

    if not reached1:
        print(f"  ❌ Could not reach Victim 1 at {target1}. Mission aborted.")
        return False, steps_done

    print(f"\n  ✅ Victim 1 at {target1} PICKED UP — {kits1} kit(s) administered.")

    # ══════════════════════════════════════════════════════════════════════════
    # VICTIM 2 — Kit allocation + travel
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ── Victim 2: {target2} ({victim2_label.upper()}) ──")
    print(f"     Kits needed : {kits2}")

    if not _allocate_kits(victim2_label, target2, agent):
        print(f"  ⚠️  Cannot allocate kits for Victim 2 at {target2}.")
        print(f"       Proceeding to hospital with only Victim 1 on board.")
        agent.log_decision(
            event  = f"Kit allocation failed for Victim 2 at {target2}",
            reason = "Insufficient kits after Victim 1 allocation",
            action = "Transport Victim 1 only; log Victim 2 for next dispatch."
        )
        # Fall through: still deliver Victim 1
        hosp = nearest_hospital(hospitals, current)
        reached_h, current, steps_done = _travel_leg(
            grid, blocked_set, current, hosp,
            leg_label  = f"LEG 2 (partial) — {target1} → Hospital",
            steps_done = steps_done,
            agent      = agent,
        )
        if reached_h:
            print(f"\n  🏥 Victim 1 DELIVERED to hospital {hosp}!")
            print(f"  ⚠️  PARTIAL RESCUE — Victim 2 not transported.")
            print(f"  Total steps: {steps_done}")
        return reached_h, steps_done

    reached2, current, steps_done = _travel_leg(
        grid, blocked_set, current, target2,
        leg_label  = f"LEG 2 — Victim 1 pickup → Victim 2 at {target2}",
        steps_done = steps_done,
        agent      = agent,
    )

    if not reached2:
        print(f"  ❌ Could not reach Victim 2 at {target2}.")
        print(f"       Proceeding to hospital with only Victim 1 on board.")
        agent.log_decision(
            event  = f"Could not reach Victim 2 at {target2}",
            reason = "Route blocked after Victim 1 pickup",
            action = "Deliver Victim 1 to hospital; re-dispatch for Victim 2."
        )

    else:
        print(f"\n  ✅ Victim 2 at {target2} PICKED UP — {kits2} kit(s) administered.")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL LEG — → Hospital
    # ══════════════════════════════════════════════════════════════════════════
    hosp = nearest_hospital(hospitals, current)

    print(f"\n  🏥 FINAL LEG — Victims on board → Hospital")
    print(f"     Nearest hospital : {hosp}")

    reached_h, current, steps_done = _travel_leg(
        grid, blocked_set, current, hosp,
        leg_label  = f"FINAL LEG — {current} → Hospital {hosp}",
        steps_done = steps_done,
        agent      = agent,
    )

    if not reached_h:
        print(f"  ❌ No path to hospital {hosp}! Victims on board but cannot be delivered.")
        agent.log_decision(
            event  = f"No path from {current} to hospital {hosp}",
            reason = "All routes to hospital blocked after victim pickups",
            action = "Victims picked up but cannot be delivered — critical situation."
        )
        return False, steps_done

    print(f"\n  🏥 Both victims DELIVERED to hospital {hosp}!")
    print(f"  ✅ RESCUE COMPLETE — Total steps: {steps_done}")
    print(f"     Full route: {start} → {target1} → {target2} → {hosp}")

    return True, steps_done