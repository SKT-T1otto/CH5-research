# Current Chapter 5 project rules

The user authorized this independent Chapter 5 workspace and organization on 2026-11-13. Scope: robust communication reinforcement learning built on the selected Chapter 3 RL/map/planning baseline. Exclude TUM/German-thesis/BVPC work.

- `core` and `chapter3_bser` are inherited source. Record their parent identity using `provenance/ch3_parent_snapshot.json`; do not rewrite that parent manifest after changes.
- Chapter 5-specific runtime code belongs in `chapter5_comm`. No algorithm is implemented there by the project-setup task.
- This directory is self-contained. Do not import sibling repositories, protected legacy CH5, or `CH5-Research/references` at runtime. Do not add sys.path injections or symlinks to them.
- The inherited source includes multiple RL/map research variants. Freeze the actual third-chapter baseline and configuration before comparing methods. Source copying is not training admission or performance acceptance.
- Preserve task, reward, target, obstacle, collision and mission-success semantics unless the user explicitly requests identified changes. Communication interface evolution must be explicit and independently identified from the Chapter 3 baseline.
- The user's all-channel choice on 2026-11-13 supersedes the earlier limited-channel suggestion: target, teammate state and map sharing are all constrained. Found/phase, planning commands, ACKs, retransmissions and relays must also use the declared channel and budget. No reliable shared map or free global Found/control side channel is allowed; public mission metadata remains explicit.
- Receiver policy, planning, navigation, action masks and fallback may only consume legal local information. The current inherited runtime does not yet satisfy the new protocol; do not claim it does merely because the design is written.
- The authoritative redesign is `docs/CH5_REDESIGN_20261113.md`. Two core method candidates are typed asynchronous knowledge and delivery-chain task communication learning. CVaR is a standard training variant, not a new theoretical robustness guarantee.
- Do not copy old 139D weights into the inherited 28D interfaces or relabel historical experiments as new CH5 results.
- Historical results, reference material, bundles and parent evidence are immutable. Uncertain claims or artifacts are REVIEW_REQUIRED.
- No long training, checkpoint restore, commit or push as an automatic setup/verification step.
- Report actual implementation, protocol checks, independent training seeds and performance evidence separately. Never turn a design formula, copied evidence or evaluation overlay into a completed experiment claim.
- Record created/modified/moved files, use task-appropriate checks and wait for all terminal commands before completion.
