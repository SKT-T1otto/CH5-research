"""Manual CH5 entry point. Execution is bounded and requires --execute."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from chapter5_comm.pipeline import (
    PROJECT, SCHEMA, auxiliary_batch, collection_scenarios, evaluation_summary, new_output,
    plain, read_config, run_episode, scenarios, smoke, task_model, write_json,
)
from chapter5_comm.training import build_neural_models, load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "train", "collect", "evaluate", "labels"))
    parser.add_argument("--config", default="configs/chapter5/short_pipeline_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config, base = read_config(args.config)
    plan = dict(command=args.command, executed=False, schema=SCHEMA,
                parent_method=base["method"], parent_map="OnlineUnknownMapTaskPlanner (receiver-local instances)",
                task_horizon=base["max_steps"], external_rollout_limit=config["rollout_limit_steps"],
                reward=base["reward"], reward_objective=base["reward_objective"],
                maximum_combined_episodes=config["collection_episodes"]+config["communication_episodes"]+len(config["evaluation_modes"])*config["evaluation_episodes"],
                auxiliary_update_cap=config["auxiliary_updates"], task_update_cap=config["task_updates"],
                communication_update_cap=config["communication_episodes"],
                formal_performance_verified=False)
    if args.command == "plan" or not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if not args.output:
        raise ValueError("execution requires an explicit fresh --output under outputs/chapter5")
    if args.command == "evaluate" and not args.checkpoint:
        raise ValueError("evaluation requires a new CH5 checkpoint")
    if args.command == "labels" and not args.dataset:
        raise ValueError("labels requires the recorded factual JSONL")
    output = new_output(args.output)
    write_json(output/"execution_plan.json", dict(plan, executed=True,
        started_at=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    try:
        if args.command in {"smoke", "train"}:
            result = smoke(config, base, output)
        elif args.command == "collect":
            model = task_model(base, config["seed"])
            networks = build_neural_models(config["neural"])
            scenes = collection_scenarios(config)
            write_json(output/"scenarios.json", scenes)
            examples, rows = [], []
            for i, scene in enumerate(scenes):
                row = run_episode(config, base, scene, networks, model, seed=config["seed"]+i,
                    mode="fixed", episode_id=f"collect_{i}", output=output/f"episode_{i}", record_labels=True)
                examples.extend(row["examples"])
                rows.append(row["summary"])
            batch = auxiliary_batch(examples)
            result = dict(status="collected", episodes=len(rows), factual_examples=len(examples),
                          supervised_routes=int(batch["route_valid"].sum()), task_horizon=400,
                          external_rollout_limit=config["rollout_limit_steps"])
            write_json(output/"report.json", result)
        elif args.command == "labels":
            from chapter5_comm.data import labels_from_rollout, read_rollout, write_labels
            rows = labels_from_rollout(read_rollout(args.dataset), config["label_horizon"], detect_radius=1.)
            write_labels(output/"labels.jsonl", rows)
            result = dict(status="labeled", examples=len(rows), source_dataset=str(Path(args.dataset).resolve()))
            write_json(output/"report.json", result)
        else:
            restored = load_checkpoint(args.checkpoint)
            if restored["metadata"]["base_config"] != base:
                raise ValueError("checkpoint task identity differs from the selected parent")
            model = task_model(base, config["seed"])
            model.load_training_state_dict(restored["task_model_state"])
            model.prep_rollouts(device="cpu")
            scenes = scenarios(config["evaluation_episodes"], config["eval_scenario_seed"], evaluation=True)
            write_json(output/"scenarios.json", scenes)
            summaries = {}
            for mode in config["evaluation_modes"]:
                rows = []
                for i, scene in enumerate(scenes):
                    row = run_episode(config, base, scene, restored["networks"], model,
                        seed=config["seed"]+200+i, mode=mode, episode_id=f"eval_{mode}_{i}",
                        output=output/f"{mode}/episode_{i}", learner=restored["communication"], record_labels=False)
                    rows.append(row["summary"])
                summaries[mode] = evaluation_summary(rows)
            result = dict(status="evaluated_short", evaluation=summaries, formal_performance_verified=False)
            write_json(output/"report.json", result)
        print(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False))
        return result
    except Exception as error:
        write_json(output/"failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    main()
