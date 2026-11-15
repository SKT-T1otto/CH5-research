# Current Chapter 5 project rules

The user authorized this independent Chapter 5 workspace and organization on 2026-11-13. Scope: robust communication reinforcement learning built on the selected Chapter 3 RL/map/planning baseline. Exclude TUM/German-thesis/BVPC work.

- `core` and `chapter3_bser` are inherited source. Record their parent identity using `provenance/ch3_parent_snapshot.json`; do not rewrite that parent manifest after changes.
- Chapter 5-specific runtime code belongs in `chapter5_comm`. No algorithm is implemented there by the project-setup task.
- This directory is self-contained. Do not import sibling repositories, protected legacy CH5, or `CH5-Research/references` at runtime. Do not add sys.path injections or symlinks to them.
- The inherited source includes multiple RL/map research variants. Freeze the actual third-chapter baseline and configuration before comparing methods. Source copying is not training admission or performance acceptance.
- Preserve task, reward, target, obstacle, collision and mission-success semantics unless the user explicitly requests identified changes. Communication interface evolution must be explicit and independently identified from the Chapter 3 baseline.
- Receiver policy, navigation and fallback may only consume information permitted by the declared channel model. State which shared map/phase/central-control channels remain reliable.
- Do not copy old 139D weights into the inherited 28D interfaces or relabel historical experiments as new CH5 results.
- Historical results, reference material, bundles and parent evidence are immutable. Uncertain claims or artifacts are REVIEW_REQUIRED.
- No long training, checkpoint restore, commit or push as an automatic setup/verification step.
- Report actual implementation, protocol checks, independent training seeds and performance evidence separately. Never turn a design formula, copied evidence or evaluation overlay into a completed experiment claim.
- Record created/modified/moved files, use task-appropriate checks and wait for all terminal commands before completion.
