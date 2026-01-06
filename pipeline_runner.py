from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from config import PHASE_MODULES, PHASE_NAMES, PipelineConfig
from schemas import Record
from storage import (
    load_input_csv,
    load_phase,
    records_from_dataframe,
    save_final_csv,
    save_phase,
    save_metrics,
)

logger = logging.getLogger("Main")


def _import_phase_runner(phase: int) -> Callable[[List[Record], PipelineConfig], List[Record]]:
    """
    Each phase module should expose:
        def run_phase(records: list[Record], cfg: PipelineConfig) -> list[Record]
    """
    mod_name = PHASE_MODULES.get(phase)
    if not mod_name:
        raise ValueError(f"No module configured for phase {phase}")

    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Phase {phase} module not found: '{mod_name}'.\n"
            f"Create file: {mod_name.replace('.', '/')}.py\n"
            f"and define: run_phase(records, cfg) -> records"
        ) from e

    if not hasattr(mod, "run_phase"):
        raise AttributeError(
            f"Phase {phase} module '{mod_name}' is missing function 'run_phase(records, cfg)'."
        )
    return getattr(mod, "run_phase")


def _record_used_ai(r: Record) -> bool:
    """
    Backward compatible AI usage detection.

    Preferred: r.ai_used
    Fallback: r.source == "ai" or presence of AI prompt/response in resolution
    """
    if getattr(r, "ai_used", False):
        return True
    if r.source == "ai":
        return True
    if r.resolution.ai_prompt or r.resolution.ai_response:
        return True
    return False


def _has_any_contact(r: Record) -> bool:
    """
    A record is considered to have "contacts" if it has at least one email or phone
    in either contacts or validated_contacts.
    """
    if r.validated_contacts.emails or r.contacts.emails:
        return True
    if r.validated_contacts.phone or r.contacts.phone:
        return True
    return False


def _compute_basic_metrics(records: List[Record], phase_timings: dict) -> dict:
    total = len(records)
    status_counts: dict[str, int] = {}
    ai_count = 0
    success_count = 0
    resolved_domain_count = 0

    for r in records:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

        if _record_used_ai(r):
            ai_count += 1

        # "success" should mean something real at the end (typically after crawl/post-validate),
        # so we require that success implies we actually found at least one contact.
        if r.status == "success" and _has_any_contact(r):
            success_count += 1

        if r.resolution.resolved_domain and not r.resolution.ambiguity_flag:
            resolved_domain_count += 1

    ai_rate = (ai_count / total) if total else 0.0
    success_rate = (success_count / total) if total else 0.0
    resolved_domain_rate = (resolved_domain_count / total) if total else 0.0

    return {
        "total_records": total,
        "status_counts": status_counts,
        "success_rate": success_rate,
        "ai_usage_rate": ai_rate,
        "resolved_domain_rate": resolved_domain_rate,
        "phase_timings_seconds": phase_timings,
    }


def _find_last_completed_phase(output_dir: Path) -> int:
    """
    Return the highest phase index with an existing checkpoint file.
    Returns -1 if none exist.
    """
    last = -1
    for p in range(0, 9):
        if (output_dir / f"phase{p}.jsonl").exists():
            last = p
    return last


def _ensure_ai_fraction_compliance(records: List[Record], cfg: PipelineConfig, where: str) -> None:
    """
    Hard guardrail: AI usage must not exceed cfg.max_ai_fraction.
    This enforces the task's "AI call < 30%" requirement at runtime.

    We measure "AI usage" as the fraction of records where AI was used for resolution.
    """
    total = len(records)
    if total == 0:
        return

    ai_count = sum(1 for r in records if _record_used_ai(r))
    ai_rate = ai_count / total

    eps = 1e-9  # avoid floating noise
    if ai_rate > (cfg.max_ai_fraction + eps):
        raise RuntimeError(
            f"AI usage rate exceeded limit at {where}: "
            f"{ai_rate:.3f} > {cfg.max_ai_fraction:.3f} "
            f"({ai_count}/{total} records counted as AI-used)."
        )


def run_pipeline(
        input_csv: Path,
        output_dir: Path,
        phase: str,
        cfg: PipelineConfig,
        final_csv: Path | None,
        *,
        start_phase: Optional[int] = None,
        resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    running_all = (phase == "all")
    if running_all:
        if resume:
            last = _find_last_completed_phase(output_dir)
            if last >= 0:
                start = min(last + 1, 8)
                logger.info("Resume enabled. Last completed phase=%d -> starting from phase=%d", last, start)
                start_phase = start
            else:
                logger.info("Resume enabled but no checkpoints found. Starting from phase 0.")
                start_phase = 0
        else:
            start_phase = 0 if start_phase is None else start_phase

        if start_phase < 0 or start_phase > 8:
            raise ValueError("--start-phase must be between 0 and 8")
        phases = list(range(start_phase, 9))
    else:
        p = int(phase)
        if p < 0 or p > 8:
            raise ValueError("--phase must be 'all' or a single phase number 0..8")
        phases = [p]

    records: List[Record] = []
    phase_timings: dict = {}
    metrics_by_phase: dict = {}

    for p in phases:
        phase_name = PHASE_NAMES.get(p, f"phase{p}")
        logger.info("=== Running Phase %s: %s ===", p, phase_name)

        if p == 0:
            df = load_input_csv(input_csv, cfg)
            logger.debug("Input CSV loaded: %s", input_csv)
            records = records_from_dataframe(df)
            logger.debug("Records loaded: %d", len(records))
        else:
            if not records:
                prev_ckpt = output_dir / f"phase{p - 1}.jsonl"
                if not prev_ckpt.exists():
                    raise FileNotFoundError(
                        f"Missing required checkpoint for phase {p}: {prev_ckpt}. "
                        f"Run phase {p - 1} first, or run --phase all."
                    )
                records = load_phase(output_dir, p - 1)
                logger.debug("Loaded phase %d checkpoint: %d records", p - 1, len(records))

        runner = _import_phase_runner(p)

        t0 = time.perf_counter()
        records = runner(records, cfg)
        dt = time.perf_counter() - t0
        phase_timings[f"phase{p}_{phase_name}"] = dt
        logger.debug("Phase %d took: %.4fs", p, dt)

        save_phase(records, output_dir, p)
        logger.info("Saved checkpoint: %s", output_dir / f"phase{p}.jsonl")

        phase_metrics = _compute_basic_metrics(records, {f"phase{p}_{phase_name}": dt})
        metrics_by_phase[f"phase{p}_{phase_name}"] = phase_metrics

        if not running_all:
            save_metrics(phase_metrics, output_dir / f"metrics_phase{p}.json")
            logger.info("Saved phase metrics: %s", output_dir / f"metrics_phase{p}.json")

        if p == 6:
            _ensure_ai_fraction_compliance(records, cfg, where="phase6")

    if running_all:
        save_metrics(metrics_by_phase, output_dir / "metrics_by_phase.json")
        logger.info("Saved metrics by phase: %s", output_dir / "metrics_by_phase.json")

    # If phase 8 ran, it owns final persistence outputs (final_results.* + metrics.json).
    ran_phase8 = (8 in phases)

    # Always enforce AI usage cap at the end (guardrail), regardless of who writes files.
    _ensure_ai_fraction_compliance(records, cfg, where="final")

    if not ran_phase8:
        # No Phase 8 in this run -> pipeline_runner writes the final outputs.
        if final_csv is None:
            final_csv = output_dir / "results.csv"
        save_final_csv(records, final_csv)
        logger.info("Saved final CSV: %s", final_csv)

        metrics = _compute_basic_metrics(records, phase_timings)
        save_metrics(metrics, output_dir / "metrics.json")
        logger.info("Saved metrics: %s", output_dir / "metrics.json")
    else:
        # Phase 8 ran -> avoid overwriting its outputs.
        # Optional: still save runner metrics under a non-conflicting name for debugging/timings.
        runner_metrics = _compute_basic_metrics(records, phase_timings)
        save_metrics(runner_metrics, output_dir / "runner_metrics.json")
        logger.info("Phase 8 ran; skipped results.csv and metrics.json. Saved runner_metrics.json instead.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Company domain/email enrichment pipeline runner.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for checkpoints and outputs")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help='Phase to run: "all" or a single phase number 0..8',
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=None,
        help="When --phase all: start from this phase (0..8). Requires phase{start-1}.jsonl if start>0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="When --phase all: auto-detect last completed phase checkpoint and continue from the next phase.",
    )
    parser.add_argument("--final-csv", type=str, default=None, help="Path to write final results CSV")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    final_csv = Path(args.final_csv) if args.final_csv else None

    cfg = PipelineConfig(output_dir=output_dir)

    run_pipeline(
        input_csv=input_csv,
        output_dir=output_dir,
        phase=args.phase,
        cfg=cfg,
        final_csv=final_csv,
        start_phase=args.start_phase,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
