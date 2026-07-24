#!/usr/bin/env python3
"""Persistent column generation for an S*G6 degree-seven SOS certificate.

The coefficient maps are exact integer orbit maps.  HiGHS is kept alive while
spectral rays are added, so its simplex basis is reused between master solves.
Solver output remains a floating-point search result, never a proof.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.optimize._highspy._core import HighsStatus, _Highs, kHighsInf

import search_degree2_quadratic_column_n6 as quadratic
import search_degree4_column_n6 as degree4
import search_pair_sos_n6 as linear
import search_sos_n6 as base


@dataclass
class Ray:
    block: int
    vector: np.ndarray
    scale: float
    created: int


def multiplier_matrix(
    group_size: int,
    multiplier: np.ndarray,
    degree_seven_orbits: np.ndarray,
    degree_seven_sizes: np.ndarray,
) -> sparse.csr_array:
    left = quadratic.QUADRATIC_UPPER[:, 0]
    right = quadratic.QUADRATIC_UPPER[:, 1]
    bases = np.concatenate(
        (
            np.repeat(multiplier.reshape(1, 3), len(left), axis=0),
            base.M2[left],
            base.M2[right],
        ),
        axis=1,
    )
    bases.sort(axis=1)
    rows = degree_seven_orbits[base.rank_multisets(bases, base.NV)]
    factors = np.where(left == right, 1, 2).astype(np.int64)
    numerators = factors * group_size
    divisors = degree_seven_sizes[rows]
    if np.any(numerators % divisors):
        raise AssertionError("nonintegral degree-seven orbit coefficient")
    return sparse.coo_array(
        (
            numerators // divisors,
            (rows, np.arange(len(left), dtype=np.int32)),
        ),
        shape=(len(degree_seven_sizes), len(left)),
    ).tocsr()


def multiplication_by_s(
    degree_six_orbits: np.ndarray, degree_seven_reps: np.ndarray
) -> sparse.csr_array:
    rows = []
    columns = []
    for row, monomial in enumerate(degree_seven_reps):
        for variable in np.unique(monomial):
            predecessor = list(int(value) for value in monomial)
            predecessor.remove(int(variable))
            rank = int(base.rank_multisets(np.asarray([predecessor]), base.NV)[0])
            rows.append(row)
            columns.append(int(degree_six_orbits[rank]))
    return sparse.coo_array(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(degree_seven_reps), int(degree_six_orbits.max()) + 1),
    ).tocsr()


def require_ok(
    status: HighsStatus, operation: str, *, allow_warning: bool = False
) -> None:
    if status != HighsStatus.kOk and not (
        allow_warning and status == HighsStatus.kWarning
    ):
        raise RuntimeError(f"HiGHS failed during {operation}: {status}")


def add_column(
    highs: _Highs,
    column: np.ndarray,
    *,
    cost: float = 0.0,
    lower: float = 0.0,
    upper: float = kHighsInf,
) -> float:
    """Scale and add one cone column; return the positive scaling divisor."""
    scale = float(np.max(np.abs(column)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot add a zero or nonfinite master column")
    indices = np.flatnonzero(column).astype(np.int32)
    values = np.asarray(column[indices] / scale, dtype=float)
    require_ok(
        highs.addCol(cost, lower, upper, len(indices), indices, values),
        "addCol",
        allow_warning=True,
    )
    return scale


def reconstruct_residual(
    target: np.ndarray,
    fixed_columns: list[np.ndarray],
    fixed_scales: np.ndarray,
    fixed_model_weights: np.ndarray,
    matrices: list[sparse.csr_array],
    rays: list[Ray],
    ray_model_weights: np.ndarray,
) -> np.ndarray:
    """Reconstruct with full columns, including entries HiGHS may have dropped."""
    residual = target.copy()
    for column, scale, weight in zip(
        fixed_columns, fixed_scales, fixed_model_weights
    ):
        residual -= (weight / scale) * column
    for ray, weight in zip(rays, ray_model_weights):
        column = np.asarray(
            matrices[ray.block] @ quadratic.ray_upper(ray.vector)
        ).ravel()
        residual -= (weight / ray.scale) * column
    return residual


def save_archive(
    path: Path,
    *,
    complete: bool,
    iteration: int,
    target: np.ndarray,
    residual: np.ndarray,
    margin: float,
    minimum_reduced_cost: float,
    multipliers: np.ndarray,
    degree4_multipliers: np.ndarray,
    fixed_grams: np.ndarray,
    fixed_scales: np.ndarray,
    fixed_model_weights: np.ndarray,
    rays: list[Ray],
    ray_model_weights: np.ndarray,
) -> None:
    if len(rays) != len(ray_model_weights):
        raise AssertionError("ray/weight mismatch while saving")
    path.parent.mkdir(parents=True, exist_ok=True)
    ray_scales = np.asarray([ray.scale for ray in rays], dtype=float)
    np.savez_compressed(
        path,
        schema=np.asarray("degree7-persistent-v1"),
        complete=np.asarray(complete),
        iteration=np.asarray(iteration),
        target=target,
        residual=residual,
        margin=np.asarray(margin),
        minimum_reduced_cost=np.asarray(minimum_reduced_cost),
        multipliers=multipliers,
        m2=base.M2,
        degree4_multipliers=degree4_multipliers,
        fixed_grams=fixed_grams,
        fixed_scales=fixed_scales,
        fixed_model_weights=fixed_model_weights,
        fixed_weights=fixed_model_weights / fixed_scales,
        ray_blocks=np.asarray([ray.block for ray in rays], dtype=np.int16),
        ray_vectors=np.asarray([ray.vector for ray in rays]),
        ray_scales=ray_scales,
        ray_created=np.asarray([ray.created for ray in rays], dtype=np.int32),
        ray_model_weights=ray_model_weights,
        ray_weights=ray_model_weights / ray_scales,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rays-per-block", type=int, default=3)
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--duplicate-cosine", type=float, default=1 - 1e-10)
    parser.add_argument("--prune-after", type=int, default=12)
    parser.add_argument("--prune-weight", type=float, default=1e-11)
    parser.add_argument("--solver", choices=("simplex", "ipm"), default="simplex")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("n6_degree7_search.npz"))
    args = parser.parse_args()
    if args.rays_per_block < 1:
        parser.error("--rays-per-block must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    setup_start = time.perf_counter()
    base.verify_ranking()
    group = base.variable_group()
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(
        6, group
    )
    degree_seven_orbits, degree_seven_reps, degree_seven_sizes = (
        base.monomial_orbits(7, group)
    )
    _, multipliers, _ = base.monomial_orbits(3, group)
    if len(degree_seven_reps) != 23287 or len(multipliers) != 65:
        raise AssertionError("unexpected degree-seven symmetry counts")
    lift = multiplication_by_s(degree_six_orbits, degree_seven_reps)
    target_six = base.target_coefficients(degree_six_reps)
    target = np.asarray(lift @ target_six).ravel()
    matrices = [
        multiplier_matrix(
            len(group), multiplier, degree_seven_orbits, degree_seven_sizes
        )
        for multiplier in multipliers
    ]
    print(
        f"degree-seven-orbits={len(target)} multiplier-blocks={len(multipliers)} "
        f"setup={time.perf_counter() - setup_start:.1f}s",
        flush=True,
    )

    seed = np.load(args.degree4_seed, allow_pickle=False)
    degree4_multipliers = np.asarray(seed["multipliers"], dtype=np.int16)
    coordinates = np.asarray(seed["gram_coordinates"], dtype=float)
    if degree4_multipliers.shape != (302, 4) or coordinates.shape != (
        302,
        len(linear.LINEAR_UPPER),
    ):
        raise ValueError("unexpected degree-four seed shape")
    lookup = {
        (int(i), int(j)): index for index, (i, j) in enumerate(linear.LINEAR_UPPER)
    }
    full_index = np.asarray(
        [
            [lookup[tuple(sorted((i, j)))] for j in range(base.NV)]
            for i in range(base.NV)
        ]
    )
    fixed_grams = []
    fixed_columns = []
    for multiplier, gram in zip(degree4_multipliers, coordinates[:, full_index]):
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        trace = float(np.trace(projected))
        if trace <= 0:
            raise ValueError("degree-four seed contains a zero PSD block")
        normalized = projected / trace
        matrix = degree4.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        fixed_grams.append(normalized)
        fixed_columns.append(
            np.asarray(lift @ (matrix @ degree4.pair_vector(normalized))).ravel()
        )
    fixed_grams_array = np.asarray(fixed_grams)

    highs = _Highs()
    require_ok(highs.setOptionValue("output_flag", False), "set output_flag")
    require_ok(highs.setOptionValue("solver", args.solver), "select solver")
    if args.solver == "simplex":
        require_ok(
            highs.setOptionValue("simplex_strategy", 1), "select dual simplex"
        )
    else:
        require_ok(
            highs.setOptionValue("ipm_optimality_tolerance", 1e-10),
            "set IPM tolerance",
        )
    require_ok(
        highs.setOptionValue("small_matrix_value", 1e-12),
        "set matrix threshold",
    )
    require_ok(
        highs.setOptionValue("primal_feasibility_tolerance", 1e-9),
        "set primal tolerance",
    )
    require_ok(
        highs.setOptionValue("dual_feasibility_tolerance", 1e-9),
        "set dual tolerance",
    )
    if args.threads:
        require_ok(highs.setOptionValue("threads", args.threads), "set threads")
    empty_starts = np.zeros(len(target) + 1, dtype=np.int32)
    require_ok(
        highs.addRows(
            len(target),
            np.full(len(target), -kHighsInf),
            target,
            0,
            empty_starts,
            np.empty(0, dtype=np.int32),
            np.empty(0),
        ),
        "add master rows",
    )
    all_rows = np.arange(len(target), dtype=np.int32)
    require_ok(
        highs.addCol(
            -1.0,
            -kHighsInf,
            kHighsInf,
            len(target),
            all_rows,
            np.ones(len(target)),
        ),
        "add margin column",
    )
    fixed_scales = np.asarray(
        [add_column(highs, column) for column in fixed_columns], dtype=float
    )

    rays: list[Ray] = []
    rays_by_block: list[list[np.ndarray]] = [[] for _ in multipliers]
    start_iteration = 0
    if args.resume:
        resume = np.load(args.resume, allow_pickle=False)
        if str(resume["schema"]) != "degree7-persistent-v1":
            raise ValueError("unsupported checkpoint schema")
        if not np.array_equal(resume["multipliers"], multipliers):
            raise ValueError("checkpoint multiplier ordering does not match")
        start_iteration = int(resume["iteration"]) + 1
        resume_blocks = np.asarray(resume["ray_blocks"], dtype=np.int16)
        resume_vectors = np.asarray(resume["ray_vectors"], dtype=float)
        resume_created = np.asarray(resume["ray_created"], dtype=np.int32)
        for block, vector, created in zip(
            resume_blocks, resume_vectors, resume_created
        ):
            true_column = np.asarray(
                matrices[int(block)] @ quadratic.ray_upper(vector)
            ).ravel()
            scale = add_column(highs, true_column)
            rays.append(Ray(int(block), vector, scale, int(created)))
            rays_by_block[int(block)].append(vector)
        print(f"resumed-rays={len(rays)} from {args.resume}", flush=True)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.output.with_suffix(".checkpoint.npz")
    fixed_model_weights = np.zeros(len(fixed_columns))
    ray_model_weights = np.zeros(len(rays))
    residual = np.full_like(target, np.nan)
    margin = float("nan")
    minimum_reduced = float("nan")

    for local_iteration in range(args.iterations):
        iteration = start_iteration + local_iteration
        solve_start = time.perf_counter()
        require_ok(highs.run(), "master solve", allow_warning=True)
        model_status = highs.modelStatusToString(highs.getModelStatus())
        if model_status != "Optimal":
            raise RuntimeError(f"master LP status is {model_status}")
        solve_seconds = time.perf_counter() - solve_start
        solution = highs.getSolution()
        values = np.asarray(solution.col_value, dtype=float)
        margin = float(values[0])
        fixed_model_weights = np.maximum(values[1 : 1 + len(fixed_columns)], 0)
        ray_model_weights = np.maximum(values[1 + len(fixed_columns) :], 0)
        if len(ray_model_weights) != len(rays):
            raise AssertionError("master column/ray mismatch")
        # Row activity includes the free margin column with coefficient one.
        residual = target - np.asarray(solution.row_value) + margin
        dual = -np.asarray(solution.row_dual, dtype=float)
        if dual.min() < -2e-8 or abs(float(dual.sum()) - 1) > 2e-7:
            raise RuntimeError(
                f"invalid master dual: min={dual.min()} sum={dual.sum()}"
            )

        price_start = time.perf_counter()
        candidates: list[tuple[float, int, np.ndarray, np.ndarray, float]] = []
        minimum_reduced = float("inf")
        negative_blocks = 0
        for block, matrix in enumerate(matrices):
            functional = np.asarray(matrix.T @ dual).ravel()
            eigenvalues, eigenvectors = linalg.eigh(
                quadratic.slack_matrix(functional),
                subset_by_index=(0, args.rays_per_block - 1),
                check_finite=False,
                driver="evr",
            )
            block_negative = False
            for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
                true_column = np.asarray(matrix @ quadratic.ray_upper(vector)).ravel()
                scale = float(np.max(np.abs(true_column)))
                normalized_reduced = float(eigenvalue / scale)
                minimum_reduced = min(minimum_reduced, normalized_reduced)
                if normalized_reduced >= -args.reduced_cost_tolerance:
                    continue
                if any(
                    abs(float(existing @ vector)) >= args.duplicate_cosine
                    for existing in rays_by_block[block]
                ):
                    continue
                if any(
                    candidate_block == block
                    and abs(float(candidate_vector @ vector))
                    >= args.duplicate_cosine
                    for _, candidate_block, candidate_vector, _, _ in candidates
                ):
                    continue
                candidates.append(
                    (normalized_reduced, block, vector, true_column, scale)
                )
                block_negative = True
            negative_blocks += int(block_negative)
        price_seconds = time.perf_counter() - price_start
        info = highs.getInfo()
        print(
            f"iteration={iteration} columns={highs.getNumCol()} rays={len(rays)} "
            f"margin={margin:.12g} residual={float(residual.min()):.12g} "
            f"negative-blocks={negative_blocks} reduced-min={minimum_reduced:.12g} "
            f"simplex-iters={info.simplex_iteration_count} solve={solve_seconds:.2f}s "
            f"price={price_seconds:.2f}s",
            flush=True,
        )

        if residual.min() > 1e-7:
            reconstructed = reconstruct_residual(
                target,
                fixed_columns,
                fixed_scales,
                fixed_model_weights,
                matrices,
                rays,
                ray_model_weights,
            )
            print(
                f"full-column residual={float(reconstructed.min()):.12g}",
                flush=True,
            )
            if reconstructed.min() <= 1e-7:
                print(
                    "rejected apparent margin caused by solver coefficient dropping",
                    flush=True,
                )
            else:
                save_archive(
                    args.output,
                    complete=True,
                    iteration=iteration,
                    target=target,
                    residual=reconstructed,
                    margin=margin,
                    minimum_reduced_cost=minimum_reduced,
                    multipliers=multipliers,
                    degree4_multipliers=degree4_multipliers,
                    fixed_grams=fixed_grams_array,
                    fixed_scales=fixed_scales,
                    fixed_model_weights=fixed_model_weights,
                    rays=rays,
                    ray_model_weights=ray_model_weights,
                )
                print(f"ACCEPTED numerical degree-seven certificate: {args.output}")
                return
        if minimum_reduced >= -args.reduced_cost_tolerance:
            save_archive(
                checkpoint,
                complete=False,
                iteration=iteration,
                target=target,
                residual=residual,
                margin=margin,
                minimum_reduced_cost=minimum_reduced,
                multipliers=multipliers,
                degree4_multipliers=degree4_multipliers,
                fixed_grams=fixed_grams_array,
                fixed_scales=fixed_scales,
                fixed_model_weights=fixed_model_weights,
                rays=rays,
                ray_model_weights=ray_model_weights,
            )
            raise SystemExit(
                "spectral pricing converged before reaching a positive margin"
            )

        if args.prune_after >= 0 and rays:
            delete_dynamic = [
                index
                for index, (ray, weight) in enumerate(
                    zip(rays, ray_model_weights)
                )
                if iteration - ray.created > args.prune_after
                and weight <= args.prune_weight
            ]
            if delete_dynamic:
                model_indices = np.asarray(
                    [1 + len(fixed_columns) + index for index in delete_dynamic],
                    dtype=np.int32,
                )
                require_ok(
                    highs.deleteCols(len(model_indices), model_indices),
                    "prune inactive columns",
                )
                deleted = set(delete_dynamic)
                rays = [ray for index, ray in enumerate(rays) if index not in deleted]
                ray_model_weights = np.asarray(
                    [
                        weight
                        for index, weight in enumerate(ray_model_weights)
                        if index not in deleted
                    ]
                )
                rays_by_block = [[] for _ in multipliers]
                for ray in rays:
                    rays_by_block[ray.block].append(ray.vector)
                print(f"pruned={len(delete_dynamic)} retained-rays={len(rays)}")

        candidates.sort(key=lambda item: item[0])
        for _, block, vector, true_column, scale in candidates:
            actual_scale = add_column(highs, true_column)
            if not np.isclose(actual_scale, scale, rtol=1e-13, atol=0):
                raise AssertionError("inconsistent ray scaling")
            vector = np.asarray(vector, dtype=float)
            rays.append(Ray(block, vector, scale, iteration))
            rays_by_block[block].append(vector)
        ray_model_weights = np.concatenate(
            (ray_model_weights, np.zeros(len(candidates)))
        )
        print(f"added={len(candidates)} total-rays={len(rays)}", flush=True)

        if (iteration + 1) % args.checkpoint_every == 0:
            save_archive(
                checkpoint,
                complete=False,
                iteration=iteration,
                target=target,
                residual=residual,
                margin=margin,
                minimum_reduced_cost=minimum_reduced,
                multipliers=multipliers,
                degree4_multipliers=degree4_multipliers,
                fixed_grams=fixed_grams_array,
                fixed_scales=fixed_scales,
                fixed_model_weights=fixed_model_weights,
                rays=rays,
                ray_model_weights=ray_model_weights,
            )
            print(f"checkpoint={checkpoint}", flush=True)

    save_archive(
        checkpoint,
        complete=False,
        iteration=iteration,
        target=target,
        residual=residual,
        margin=margin,
        minimum_reduced_cost=minimum_reduced,
        multipliers=multipliers,
        degree4_multipliers=degree4_multipliers,
        fixed_grams=fixed_grams_array,
        fixed_scales=fixed_scales,
        fixed_model_weights=fixed_model_weights,
        rays=rays,
        ray_model_weights=ray_model_weights,
    )
    raise SystemExit(
        f"no positive margin within {args.iterations} master solves; "
        f"restart from {checkpoint}"
    )


if __name__ == "__main__":
    main()
