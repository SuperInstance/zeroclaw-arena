"""
Ecology 24,000 — Large-Scale Ecological Dynamics Experiment.

24,000 decision environments across 7 domains.
10,000 agents, 50 generations of ternary evolution.
Tracks species emergence, universal vs specialist strategies,
keystone environments, and ecological interaction graphs.

Outputs: results/ecology-24000.json
"""

import json
import math
import random
import time
import os
import sys
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Environment Generation — 24,000 environments across 7 domains
# ---------------------------------------------------------------------------

DOMAINS = {
    "games": {
        "subtypes": ["tictactoe", "connect4", "holdem", "chess_endgame", "go_9x9"],
        "n_envs": 4000,
        "action_space": 9,
        "actions": ["play_a", "play_b", "play_c", "play_d", "play_e",
                     "play_f", "play_g", "play_h", "play_i"],
    },
    "trading": {
        "subtypes": ["trend", "volatility", "market_making"],
        "n_envs": 4000,
        "action_space": 5,
        "actions": ["buy", "sell", "hold", "hedge", "close"],
    },
    "negotiation": {
        "subtypes": ["accept_reject", "counter_offer", "walk_away"],
        "n_envs": 3000,
        "action_space": 4,
        "actions": ["accept", "reject", "counter", "walk"],
    },
    "navigation": {
        "subtypes": ["city", "highway", "offroad"],
        "n_envs": 3000,
        "action_space": 5,
        "actions": ["left", "right", "straight", "stop", "reverse"],
    },
    "ecology": {
        "subtypes": ["cooperate_defect", "migrate", "hibernate"],
        "n_envs": 4000,
        "action_space": 4,
        "actions": ["cooperate", "defect", "migrate", "hibernate"],
    },
    "resource_mgmt": {
        "subtypes": ["allocate", "hoard", "trade", "conserve"],
        "n_envs": 3000,
        "action_space": 4,
        "actions": ["allocate", "hoard", "trade", "conserve"],
    },
    "warfare": {
        "subtypes": ["attack", "defend", "retreat", "flank"],
        "n_envs": 3000,
        "action_space": 4,
        "actions": ["attack", "defend", "retreat", "flank"],
    },
}

REWARD_TYPES = ["binary", "marginal", "progressive", "sparse", "adversarial", "cooperative", "noisy"]

DOMAIN_NAMES = list(DOMAINS.keys())

# Trait indices
TRAIT_NAMES = [
    "exploration", "risk_tolerance", "aggression", "cooperativeness",
    "persistence", "adaptability", "foresight", "specialization",
    "reaction_speed", "conservation", "deception", "loyalty",
    "curiosity", "patience", "boldness", "caution",
    "social_learning", "innovation", "mimicry", "territoriality",
]
N_TRAITS = len(TRAIT_NAMES)

# Map trait names to indices
TI = {name: i for i, name in enumerate(TRAIT_NAMES)}


def generate_environments(total=24000, seed=42):
    """Generate environments as structured numpy arrays for vectorized ops."""
    rng = np.random.default_rng(seed)

    domain_ids = []  # int domain index per env
    subtype_ids = []  # int subtype index per env
    reward_type_ids = []  # int reward type index per env
    complexities = []
    stochasticities = []
    horizons = []
    seeds = []

    # Build domain mapping
    domain_list = []
    subtype_offset = {}
    subtype_list = []
    offset = 0
    for domain, spec in DOMAINS.items():
        domain_list.append(domain)
        subtype_offset[domain] = (offset, offset + len(spec["subtypes"]))
        for st in spec["subtypes"]:
            subtype_list.append(st)
            offset += 1

    subtype_to_idx = {s: i for i, s in enumerate(subtype_list)}
    reward_to_idx = {r: i for i, r in enumerate(REWARD_TYPES)}

    for domain, spec in DOMAINS.items():
        n = spec["n_envs"]
        for i in range(n):
            domain_ids.append(domain_list.index(domain))
            subtype = spec["subtypes"][i % len(spec["subtypes"])]
            subtype_ids.append(subtype_to_idx[subtype])
            reward_type = REWARD_TYPES[i % len(REWARD_TYPES)]
            reward_type_ids.append(reward_to_idx[reward_type])
            complexities.append(float(rng.uniform(0.1, 1.0)))
            stochasticities.append(float(rng.uniform(0.0, 0.5)))
            horizons.append(int(rng.integers(10, 60)))
            seeds.append(int(rng.integers(0, 2**31)))

    return {
        "n_envs": len(domain_ids),
        "domain_ids": np.array(domain_ids, dtype=np.int32),
        "subtype_ids": np.array(subtype_ids, dtype=np.int32),
        "reward_type_ids": np.array(reward_type_ids, dtype=np.int32),
        "complexities": np.array(complexities, dtype=np.float64),
        "stochasticities": np.array(stochasticities, dtype=np.float64),
        "horizons": np.array(horizons, dtype=np.int32),
        "seeds": np.array(seeds, dtype=np.int64),
        "domain_list": domain_list,
        "subtype_list": subtype_list,
    }


# ---------------------------------------------------------------------------
# Vectorized Fitness Computation
# ---------------------------------------------------------------------------

def compute_fitness_matrix(gene_matrix, env_data, eval_indices, rng):
    """Compute fitness matrix (n_agents × n_eval_envs) using vectorized numpy.

    gene_matrix: (n_agents, N_TRAITS) of int8 {-1, 0, 1}
    env_data: environment dict
    eval_indices: array of env indices to evaluate on

    Returns: (n_agents,) mean fitness array
    """
    n_agents = gene_matrix.shape[0]
    n_eval = len(eval_indices)

    # Normalize genes to [0, 1]
    genes_norm = (gene_matrix.astype(np.float64) + 1.0) / 2.0  # (n_agents, 20)

    # Extract traits for all agents
    exploration = genes_norm[:, TI["exploration"]]
    risk = genes_norm[:, TI["risk_tolerance"]]
    aggression = genes_norm[:, TI["aggression"]]
    coop = genes_norm[:, TI["cooperativeness"]]
    persistence = genes_norm[:, TI["persistence"]]
    adapt = genes_norm[:, TI["adaptability"]]
    foresight = genes_norm[:, TI["foresight"]]
    patience = genes_norm[:, TI["patience"]]
    boldness = genes_norm[:, TI["boldness"]]
    innovation = genes_norm[:, TI["innovation"]]
    specialization = genes_norm[:, TI["specialization"]]

    # Domain fitness weights: (n_domains, n_agents)
    # Precompute base fitness per domain
    domain_base = np.zeros((len(DOMAIN_NAMES), n_agents))
    for di, domain in enumerate(DOMAIN_NAMES):
        if domain == "games":
            domain_base[di] = 0.3 + 0.2*foresight + 0.15*exploration + 0.1*aggression + 0.1*adapt
        elif domain == "trading":
            domain_base[di] = 0.3 + 0.2*risk + 0.15*patience + 0.15*adapt + 0.1*foresight
        elif domain == "negotiation":
            domain_base[di] = 0.3 + 0.2*coop + 0.15*patience + 0.15*boldness + 0.1*adapt
        elif domain == "navigation":
            domain_base[di] = 0.3 + 0.2*adapt + 0.15*exploration + 0.15*foresight + 0.1*persistence
        elif domain == "ecology":
            domain_base[di] = 0.3 + 0.2*coop + 0.15*persistence + 0.15*exploration + 0.1*adapt
        elif domain == "resource_mgmt":
            domain_base[di] = 0.3 + 0.2*patience + 0.15*foresight + 0.15*exploration + 0.1*innovation
        elif domain == "warfare":
            domain_base[di] = 0.3 + 0.2*aggression + 0.15*boldness + 0.15*foresight + 0.1*adapt

    # Subtype modifiers: (n_subtypes, n_agents)
    n_subtypes = len(env_data["subtype_list"])
    subtype_mod = np.zeros((n_subtypes, n_agents))
    subtype_trait_map = {
        # Games
        0: (TI["foresight"], 0.05),
        1: (TI["foresight"], 0.08, TI["aggression"], 0.05),
        2: (TI["risk_tolerance"], 0.1, TI["boldness"], 0.05),
        3: (TI["foresight"], 0.12, TI["patience"], 0.05),
        4: (TI["foresight"], 0.1, TI["patience"], 0.08),
        # Trading
        5: (TI["patience"], 0.1, TI["foresight"], 0.05),
        6: (TI["risk_tolerance"], 0.1, TI["adaptability"], 0.05),
        7: (TI["adaptability"], 0.08, TI["patience"], 0.08),
        # Negotiation
        8: (TI["patience"], 0.05),
        9: (TI["boldness"], 0.1, TI["cooperativeness"], 0.05),
        10: (TI["boldness"], 0.08, TI["risk_tolerance"], 0.05),
        # Navigation
        11: (TI["adaptability"], 0.08, TI["exploration"], 0.05),
        12: (TI["foresight"], 0.08, TI["patience"], 0.05),
        13: (TI["exploration"], 0.1, TI["persistence"], 0.05),
        # Ecology
        14: (TI["cooperativeness"], 0.1, TI["patience"], 0.05),
        15: (TI["exploration"], 0.08, TI["boldness"], 0.08),
        16: (TI["patience"], 0.1, TI["persistence"], 0.05),
        # Resource
        17: (TI["foresight"], 0.08, TI["cooperativeness"], 0.08),
        18: (TI["risk_tolerance"], 0.1, TI["aggression"], 0.05),
        19: (TI["cooperativeness"], 0.08, TI["boldness"], 0.08),
        20: (TI["patience"], 0.1, TI["persistence"], 0.05),
        # Warfare
        21: (TI["aggression"], 0.1, TI["boldness"], 0.08),
        22: (TI["patience"], 0.08, TI["persistence"], 0.08),
        23: (TI["adaptability"], 0.08, TI["foresight"], 0.08),
        24: (TI["foresight"], 0.1, TI["innovation"], 0.08),
    }
    for si, mapping in subtype_trait_map.items():
        if si < n_subtypes:
            if len(mapping) == 2:
                trait_idx, weight = mapping
                subtype_mod[si] = genes_norm[:, trait_idx] * weight
            elif len(mapping) == 4:
                t1, w1, t2, w2 = mapping
                subtype_mod[si] = genes_norm[:, t1] * w1 + genes_norm[:, t2] * w2

    # For each eval env, compute fitness for all agents at once
    # Use domain_base + subtype_mod + reward_mod as the core
    eval_dids = env_data["domain_ids"][eval_indices]
    eval_sids = env_data["subtype_ids"][eval_indices]
    eval_rids = env_data["reward_type_ids"][eval_indices]
    eval_comp = env_data["complexities"][eval_indices]
    eval_stoch = env_data["stochasticities"][eval_indices]

    # Fitness accumulator
    total_fitness = np.zeros(n_agents)
    domain_fitness = np.zeros((len(DOMAIN_NAMES), n_agents))
    domain_counts = np.zeros(len(DOMAIN_NAMES), dtype=np.int32)

    for ei in range(n_eval):
        did = eval_dids[ei]
        sid = eval_sids[ei]
        rid = eval_rids[ei]
        comp = eval_comp[ei]
        stoch = eval_stoch[ei]

        # Base + subtype modifier for this env
        fit = domain_base[did].copy() + subtype_mod[sid]

        # Reward type modulation (vectorized per agent)
        if rid == 0:  # binary
            reward_mod = np.where(fit > 0.6, 0.5, -0.2)
        elif rid == 1:  # marginal
            reward_mod = fit * 0.15
        elif rid == 2:  # progressive
            reward_mod = fit * comp
        elif rid == 3:  # sparse
            reward_mod = np.where((fit > 0.75) & (comp > 0.5), 0.3, -0.1)
        elif rid == 4:  # adversarial
            reward_mod = fit - 0.15 * stoch
        elif rid == 5:  # cooperative
            reward_mod = fit * (1.0 + 0.1 * coop)
        else:  # noisy
            reward_mod = rng.normal(0, 0.15, size=n_agents)

        fit = fit + reward_mod

        # Stochastic noise
        fit += rng.normal(0, stoch * 0.1, size=n_agents)
        np.clip(fit, 0.0, 1.0, out=fit)

        total_fitness += fit
        domain_fitness[did] += fit
        domain_counts[did] += 1

    total_fitness /= n_eval
    for di in range(len(DOMAIN_NAMES)):
        if domain_counts[di] > 0:
            domain_fitness[di] /= domain_counts[di]

    return total_fitness, domain_fitness, domain_counts


# ---------------------------------------------------------------------------
# Species Clustering (vectorized)
# ---------------------------------------------------------------------------

def cluster_species_fast(gene_matrix, threshold=0.25, max_species=200):
    """Fast species clustering using vectorized distance computation."""
    n = gene_matrix.shape[0]

    # If too many agents, subsample
    if n > 3000:
        indices = np.arange(0, n, max(1, n // 3000))
        sample = gene_matrix[indices]
    else:
        sample = gene_matrix
        indices = np.arange(n)

    ns = sample.shape[0]

    # Compute pairwise hamming distances (ns × ns)
    # For ternary, use broadcasting
    # distance = mean(genes_i != genes_j)
    # Efficient: compute using sum of equality
    eq = (sample[:, None, :] == sample[None, :, :])  # (ns, ns, traits)
    dist = 1.0 - eq.mean(axis=2)  # (ns, ns)

    # Greedy clustering
    assigned = np.zeros(ns, dtype=bool)
    species = []

    for i in range(ns):
        if assigned[i]:
            continue
        members = np.where((dist[i] < threshold) & ~assigned)[0]
        assigned[members] = True
        species.append(indices[members].tolist())

        if len(species) >= max_species:
            break

    return species


def label_species(species_list, gene_matrix):
    """Label species by dominant traits."""
    labels = []
    for members in species_list:
        mean_genes = gene_matrix[members].mean(axis=0)
        # Find top 3 traits by absolute value
        ranked = np.argsort(-np.abs(mean_genes))[:3]
        parts = []
        for idx in ranked:
            if abs(mean_genes[idx]) > 0.1:
                prefix = "high" if mean_genes[idx] > 0 else "low"
                parts.append(f"{prefix}_{TRAIT_NAMES[idx]}")
        label = "-".join(parts) if parts else "neutral"
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_agents=10000,
    n_gens=50,
    n_envs=24000,
    seed=42,
    sample_envs_per_eval=500,
    species_threshold=0.25,
):
    rng = np.random.default_rng(seed)
    start_time = time.time()

    print(f"Generating {n_envs} environments...")
    env_data = generate_environments(n_envs, seed=seed)

    print(f"Initializing {n_agents} agents with {N_TRAITS} ternary traits...")
    gene_matrix = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n_agents, N_TRAITS))

    history = {
        "generations": [],
        "config": {
            "n_agents": n_agents,
            "n_gens": n_gens,
            "n_envs": n_envs,
            "seed": seed,
            "sample_envs_per_eval": sample_envs_per_eval,
            "species_threshold": species_threshold,
            "n_traits": N_TRAITS,
        },
    }

    all_species_counts = []
    universal_vs_specialist = []

    print(f"Running {n_gens} generations with vectorized fitness...")
    for gen in range(n_gens):
        gen_start = time.time()

        # Sample eval environments
        eval_indices = rng.choice(env_data["n_envs"], size=min(sample_envs_per_eval, env_data["n_envs"]), replace=False)

        # Vectorized fitness computation
        agent_fitness, domain_fitness, domain_counts = compute_fitness_matrix(
            gene_matrix, env_data, eval_indices, rng
        )

        # Species clustering
        species = cluster_species_fast(gene_matrix, threshold=species_threshold)
        species_labels = label_species(species, gene_matrix)
        n_species = len(species)
        species_sizes = sorted([len(s) for s in species], reverse=True)

        # Domain-specific species analysis
        domain_species = {}
        for di, domain in enumerate(DOMAIN_NAMES):
            if domain_counts[di] == 0:
                domain_species[domain] = []
                continue
            top_n = max(1, n_agents // 5)
            top_agents = np.argsort(domain_fitness[di])[-top_n:]
            sp_set = set()
            for agent_idx in top_agents:
                for si, members in enumerate(species):
                    if agent_idx in members:
                        sp_set.add(si)
                        break
            domain_species[domain] = list(sp_set)

        # Cross-domain overlap
        species_domains = defaultdict(set)
        for domain, sp_list in domain_species.items():
            for sp in sp_list:
                species_domains[sp].add(domain)

        universal_species = [sp for sp, doms in species_domains.items() if len(doms) >= 5]
        specialist_species = [sp for sp, doms in species_domains.items() if len(doms) <= 2]
        universal_ratio = len(universal_species) / max(1, n_species)
        specialist_ratio = len(specialist_species) / max(1, n_species)

        # Selection & reproduction
        # Tournament selection + elitism
        elite_count = n_agents // 10
        elite_indices = np.argsort(agent_fitness)[-elite_count:]

        new_genes = np.empty_like(gene_matrix)
        # Keep elites
        for i, idx in enumerate(elite_indices):
            new_genes[i] = gene_matrix[idx]

        # Fill rest via crossover + mutation
        fill_count = n_agents - elite_count
        if fill_count > 0:
            # Tournament parents
            t1 = rng.integers(0, n_agents, size=(fill_count, 5))
            t2 = rng.integers(0, n_agents, size=(fill_count, 5))
            p1_fitness = agent_fitness[t1]
            p2_fitness = agent_fitness[t2]
            p1 = t1[np.arange(fill_count), np.argmax(p1_fitness, axis=1)]
            p2 = t2[np.arange(fill_count), np.argmax(p2_fitness, axis=1)]

            # Uniform crossover
            cross_mask = rng.random((fill_count, N_TRAITS)) < 0.5
            child_genes = np.where(cross_mask, gene_matrix[p1], gene_matrix[p2])

            # Mutation: ternary shift
            mut_mask = rng.random((fill_count, N_TRAITS)) < 0.08
            shifts = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(fill_count, N_TRAITS))
            child_genes = np.clip(child_genes.astype(np.int16) + shifts * mut_mask, -1, 1).astype(np.int8)

            new_genes[elite_count:] = child_genes

        gene_matrix = new_genes

        # Record
        gen_data = {
            "generation": gen,
            "mean_fitness": round(float(np.mean(agent_fitness)), 4),
            "max_fitness": round(float(np.max(agent_fitness)), 4),
            "std_fitness": round(float(np.std(agent_fitness)), 4),
            "n_species": n_species,
            "top_species_sizes": species_sizes[:10],
            "universal_species_count": len(universal_species),
            "specialist_species_count": len(specialist_species),
            "universal_ratio": round(universal_ratio, 3),
            "specialist_ratio": round(specialist_ratio, 3),
            "domain_mean_fitness": {d: round(float(np.mean(domain_fitness[di])), 4) for di, d in enumerate(DOMAIN_NAMES) if domain_counts[di] > 0},
            "elapsed_s": round(time.time() - gen_start, 1),
        }
        history["generations"].append(gen_data)
        all_species_counts.append(n_species)
        universal_vs_specialist.append((len(universal_species), len(specialist_species)))

        if gen % 5 == 0 or gen == n_gens - 1:
            elapsed_total = round(time.time() - start_time, 1)
            print(f"  Gen {gen:3d}: fitness={gen_data['mean_fitness']:.3f} "
                  f"species={n_species:3d} "
                  f"universal={len(universal_species)} specialist={len(specialist_species)} "
                  f"[{elapsed_total}s]")

    # --- Final Analysis ---
    print("Computing final species × domain fitness matrix...")
    final_species = cluster_species_fast(gene_matrix, threshold=species_threshold)
    final_labels = label_species(final_species, gene_matrix)

    # Species × domain fitness matrix
    species_domain_matrix = {}
    top_species = final_species[:50]

    # Evaluate top species on a larger sample per domain
    domain_env_indices = defaultdict(list)
    for i in range(env_data["n_envs"]):
        domain_env_indices[env_data["domain_ids"][i]].append(i)

    for si, members in enumerate(top_species):
        label = final_labels[si]
        domain_fits = {}
        for di, domain in enumerate(DOMAIN_NAMES):
            # Get env indices for this domain
            d_indices = np.array(domain_env_indices[di][:200])
            if len(d_indices) == 0:
                domain_fits[domain] = 0.0
                continue
            # Evaluate representative members
            sample_members = members[:min(20, len(members))]
            if len(sample_members) == 0:
                domain_fits[domain] = 0.0
                continue
            member_genes = gene_matrix[sample_members]
            # Mean fitness of these members across domain envs
            fits, _, _ = compute_fitness_matrix(member_genes, env_data, d_indices[:100], rng)
            domain_fits[domain] = round(float(np.mean(fits)), 3)

        species_domain_matrix[f"species_{si}_{label}"] = domain_fits

    # Ecological interaction graph
    print("Building ecological interaction graph...")
    sp_names = list(species_domain_matrix.keys())
    interaction_graph = {"nodes": [], "edges": []}

    for name in sp_names:
        interaction_graph["nodes"].append({
            "id": name,
            "fitness_profile": species_domain_matrix[name],
        })

    # Pairwise correlations
    fitness_vectors = np.array([list(species_domain_matrix[n].values()) for n in sp_names])
    if fitness_vectors.shape[0] > 1:
        corr_matrix = np.corrcoef(fitness_vectors)
        for i in range(len(sp_names)):
            for j in range(i + 1, len(sp_names)):
                c = corr_matrix[i, j]
                if not np.isnan(c) and abs(c) > 0.3:
                    interaction_graph["edges"].append({
                        "source": sp_names[i],
                        "target": sp_names[j],
                        "correlation": round(float(c), 3),
                        "type": "competitive" if c < -0.3 else "symbiotic",
                    })

    # Keystone ranking
    print("Computing keystone ranking...")
    keystone_ranking = []
    sample_env_indices = list(range(0, env_data["n_envs"], max(1, env_data["n_envs"] // 100)))

    for env_idx in sample_env_indices[:100]:
        did = env_data["domain_ids"][env_idx]
        sid = env_data["subtype_ids"][env_idx]
        rid = env_data["reward_type_ids"][env_idx]
        comp = float(env_data["complexities"][env_idx])

        # Get fitness per species
        sp_perfs = []
        for si, members in enumerate(top_species[:30]):
            sample_m = members[:min(10, len(members))]
            if not sample_m:
                continue
            mg = gene_matrix[sample_m]
            f, _, _ = compute_fitness_matrix(mg, env_data, np.array([env_idx]), rng)
            sp_perfs.append(float(np.mean(f)))

        variance = float(np.var(sp_perfs)) if sp_perfs else 0.0
        keystone_ranking.append({
            "env_id": int(env_idx),
            "domain": DOMAIN_NAMES[did],
            "subtype": env_data["subtype_list"][sid],
            "reward_type": REWARD_TYPES[rid],
            "complexity": round(comp, 2),
            "differentiation_variance": round(variance, 4),
        })

    keystone_ranking.sort(key=lambda x: -x["differentiation_variance"])

    # --- Assemble Results ---
    total_time = round(time.time() - start_time, 1)
    print(f"Total time: {total_time}s")

    results = {
        "experiment": "ecology_24000",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": history["config"],
        "total_time_s": total_time,
        "summary": {
            "total_environments": n_envs,
            "environments_per_domain": {d: DOMAINS[d]["n_envs"] for d in DOMAINS},
            "total_agents": n_agents,
            "total_generations": n_gens,
            "final_species_count": len(final_species),
            "final_mean_fitness": history["generations"][-1]["mean_fitness"],
            "final_max_fitness": history["generations"][-1]["max_fitness"],
            "final_universal_ratio": history["generations"][-1]["universal_ratio"],
            "final_specialist_ratio": history["generations"][-1]["specialist_ratio"],
            "species_convergence": {
                "gen_0": all_species_counts[0] if all_species_counts else 0,
                "gen_25": all_species_counts[25] if len(all_species_counts) > 25 else 0,
                "gen_49": all_species_counts[-1] if all_species_counts else 0,
            },
            "ecological_dynamics": {
                "species_count_trend": all_species_counts[::5],
                "universal_trend": [u for u, s in universal_vs_specialist[::5]],
                "specialist_trend": [s for u, s in universal_vs_specialist[::5]],
            },
        },
        "species_domain_matrix": species_domain_matrix,
        "interaction_graph": interaction_graph,
        "keystone_ranking": keystone_ranking[:30],
        "generation_history": history["generations"],
        "environment_summary": {
            d: {
                "n_envs": int(np.sum(env_data["domain_ids"] == di)),
                "subtypes": list(set(env_data["subtype_list"][s] for s in env_data["subtype_ids"][env_data["domain_ids"] == di])),
                "reward_types": list(set(REWARD_TYPES[r] for r in env_data["reward_type_ids"][env_data["domain_ids"] == di])),
                "mean_complexity": round(float(np.mean(env_data["complexities"][env_data["domain_ids"] == di])), 3),
            }
            for di, d in enumerate(DOMAIN_NAMES)
        },
    }

    return results


def main():
    results = run_experiment(
        n_agents=10000,
        n_gens=50,
        n_envs=24000,
        seed=42,
        sample_envs_per_eval=500,
        species_threshold=0.25,
    )

    os.makedirs("results", exist_ok=True)
    out_path = "results/ecology-24000.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")

    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"ECOLOGY 24,000 — SUMMARY")
    print(f"{'='*60}")
    print(f"Environments: {s['total_environments']:,}")
    print(f"Agents: {s['total_agents']:,}")
    print(f"Generations: {s['total_generations']}")
    print(f"Final species: {s['final_species_count']}")
    print(f"Mean fitness: {s['final_mean_fitness']:.3f}")
    print(f"Max fitness: {s['final_max_fitness']:.3f}")
    print(f"Universal ratio: {s['final_universal_ratio']:.1%}")
    print(f"Specialist ratio: {s['final_specialist_ratio']:.1%}")
    print(f"Species convergence: {s['species_convergence']}")
    n_nodes = len(results['interaction_graph']['nodes'])
    n_edges = len(results['interaction_graph']['edges'])
    print(f"Interaction graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Keystone envs ranked: {len(results['keystone_ranking'])}")
    print(f"Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
