"""
Phase 7: Collision Risk Estimation
====================================
Estimates probability of collision between debris objects
using predicted trajectories and Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import os

@dataclass
class DebrisObject:
    id: int
    name: str
    position: np.ndarray     # [x, y, z] in km
    velocity: np.ndarray     # [vx, vy, vz] in km/s
    size_m: float            # estimated size in meters
    predicted_path: np.ndarray = None  # shape (T, 3)


@dataclass
class CollisionRisk:
    obj1_id: int
    obj2_id: int
    obj1_name: str
    obj2_name: str
    min_distance_km: float
    time_of_closest_approach: int     # timestep index
    collision_probability: float
    risk_level: str                   # LOW / MEDIUM / HIGH / CRITICAL


def compute_miss_distance(path1: np.ndarray, path2: np.ndarray) -> Tuple[float, int]:
    """Compute minimum miss distance between two predicted trajectories."""
    distances = np.linalg.norm(path1 - path2, axis=1)
    min_dist = np.min(distances)
    tca = np.argmin(distances)
    return min_dist, tca


def monte_carlo_collision_probability(pos1: np.ndarray, pos2: np.ndarray,
                                       vel1: np.ndarray, vel2: np.ndarray,
                                       size1_m: float, size2_m: float,
                                       uncertainty_km: float = 0.5,
                                       samples: int = 1000) -> float:
    """
    Estimate collision probability using Monte Carlo simulation.
    
    Adds Gaussian position uncertainty and checks if objects
    come within combined hard-body radius.
    """
    combined_radius_km = (size1_m + size2_m) / 2000  # convert to km

    collisions = 0
    for _ in range(samples):
        # Sample position with uncertainty
        p1 = pos1 + np.random.normal(0, uncertainty_km, 3)
        p2 = pos2 + np.random.normal(0, uncertainty_km, 3)

        dist = np.linalg.norm(p1 - p2)
        if dist < combined_radius_km:
            collisions += 1

    return collisions / samples


def classify_risk(probability: float, min_distance_km: float) -> str:
    """Classify collision risk level."""
    if probability > 0.01 or min_distance_km < 0.1:
        return "CRITICAL 🔴"
    elif probability > 0.001 or min_distance_km < 1.0:
        return "HIGH 🟠"
    elif probability > 0.0001 or min_distance_km < 5.0:
        return "MEDIUM 🟡"
    else:
        return "LOW 🟢"


def assess_all_risks(debris_objects: List[DebrisObject]) -> List[CollisionRisk]:
    """Assess collision risks between all pairs of debris objects."""
    risks = []
    n = len(debris_objects)

    print(f"🔍 Assessing collision risks for {n} objects ({n*(n-1)//2} pairs)...")

    for i in range(n):
        for j in range(i + 1, n):
            obj1 = debris_objects[i]
            obj2 = debris_objects[j]

            if obj1.predicted_path is None or obj2.predicted_path is None:
                continue

            min_dist, tca = compute_miss_distance(obj1.predicted_path, obj2.predicted_path)

            # Only compute full probability for close approaches
            if min_dist < 10.0:
                pos1_tca = obj1.predicted_path[tca]
                pos2_tca = obj2.predicted_path[tca]

                prob = monte_carlo_collision_probability(
                    pos1_tca, pos2_tca,
                    obj1.velocity, obj2.velocity,
                    obj1.size_m, obj2.size_m,
                    samples=500
                )

                risk = CollisionRisk(
                    obj1_id=obj1.id,
                    obj2_id=obj2.id,
                    obj1_name=obj1.name,
                    obj2_name=obj2.name,
                    min_distance_km=min_dist,
                    time_of_closest_approach=tca,
                    collision_probability=prob,
                    risk_level=classify_risk(prob, min_dist)
                )
                risks.append(risk)

    # Sort by probability (highest first)
    risks.sort(key=lambda r: r.collision_probability, reverse=True)
    return risks


def print_risk_report(risks: List[CollisionRisk], top_n: int = 10):
    """Print a formatted collision risk report."""
    print("\n" + "=" * 70)
    print("🛸 COLLISION RISK REPORT")
    print("=" * 70)

    if not risks:
        print("✅ No significant collision risks detected.")
        return

    critical = [r for r in risks if "CRITICAL" in r.risk_level]
    high = [r for r in risks if "HIGH" in r.risk_level]
    medium = [r for r in risks if "MEDIUM" in r.risk_level]

    print(f"Critical: {len(critical)} | High: {len(high)} | Medium: {len(medium)}")
    print("-" * 70)
    print(f"{'Object 1':<25} {'Object 2':<25} {'Min Dist(km)':<14} {'Prob':<10} {'Risk'}")
    print("-" * 70)

    for risk in risks[:top_n]:
        print(f"{risk.obj1_name[:24]:<25} {risk.obj2_name[:24]:<25} "
              f"{risk.min_distance_km:<14.3f} {risk.collision_probability:<10.6f} "
              f"{risk.risk_level}")

    print("=" * 70)


def save_risk_report(risks: List[CollisionRisk], output_path: str = "outputs/collision_risks.csv"):
    """Save risk report to CSV — always writes header even when risks list is empty."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = [
        {
            "obj1_id": r.obj1_id, "obj1_name": r.obj1_name,
            "obj2_id": r.obj2_id, "obj2_name": r.obj2_name,
            "min_distance_km": r.min_distance_km,
            "tca_step": r.time_of_closest_approach,
            "collision_probability": r.collision_probability,
            "risk_level": r.risk_level
        }
        for r in risks
    ]
    # Always create a proper DataFrame with columns even if records is empty
    columns = ["obj1_id","obj1_name","obj2_id","obj2_name",
               "min_distance_km","tca_step","collision_probability","risk_level"]
    df = pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)
    df.to_csv(output_path, index=False)
    print(f"💾 Risk report saved to {output_path}")


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🛸 COLLISION RISK ESTIMATION — DEMO")
    print("=" * 60)

    # Generate synthetic debris with predicted paths
    np.random.seed(42)
    objects = []

    for i in range(20):
        pos = np.random.randn(3) * 500 + np.array([6800, 0, 0])
        vel = np.random.randn(3) * 0.1 + np.array([0, 7.5, 0])
        size = np.random.choice([0.1, 0.5, 1.0, 2.0, 5.0])

        # Simulate simple linear trajectory
        steps = 50
        path = np.array([pos + vel * t for t in range(steps)])

        obj = DebrisObject(
            id=i,
            name=f"DEBRIS-{1000+i}",
            position=pos,
            velocity=vel,
            size_m=size,
            predicted_path=path
        )
        objects.append(obj)

    # Assess risks
    risks = assess_all_risks(objects)
    print_risk_report(risks)
    save_risk_report(risks)