import json
import os
import uuid
from typing import Optional, List
from .types import PolicySnapshot, EvalMetrics, HyperparamSet


class CheckpointRegistry:
    """
    Manages policy snapshots on disk.

    Responsibilities:
    - Save a new best snapshot when improvement is detected
    - Load the best snapshot (for watchdog rollback, if configured)
    - Maintain a manifest of all snapshots with metadata
    - Prune old snapshots beyond max_snapshots

    Directory structure:
        checkpoint_dir/
            manifest.json
            snapshot_<id>/
                weights/        ← agent.save_weights() writes here
                metadata.json   ← EvalMetrics + HyperparamSet
    """

    def __init__(self, checkpoint_dir: str, max_snapshots: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.max_snapshots = max_snapshots
        self._best: Optional[PolicySnapshot] = None
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        agent,
        eval_metrics: EvalMetrics,
        hyperparams: HyperparamSet,
    ) -> PolicySnapshot:
        """
        Save current agent weights + metadata as a new snapshot.
        Automatically prunes oldest snapshots beyond max_snapshots.
        Returns the created PolicySnapshot.
        """
        snapshot_id = str(uuid.uuid4())[:8]
        snapshot_dir = os.path.join(self.checkpoint_dir, f"snapshot_{snapshot_id}")
        weights_path = os.path.join(snapshot_dir, "weights")
        os.makedirs(snapshot_dir, exist_ok=True)

        agent.save_weights(weights_path)

        snapshot = PolicySnapshot(
            snapshot_id=snapshot_id,
            eval_metrics=eval_metrics,
            hyperparams=hyperparams,
            weights_path=weights_path,
            episode_id=eval_metrics.episode_id,
        )

        metadata = {
            "snapshot_id": snapshot.snapshot_id,
            "episode_id": snapshot.episode_id,
            "timestamp": snapshot.timestamp,
            "primary_score": eval_metrics.primary_score,
            "metrics": eval_metrics.metrics,
            "hyperparams": hyperparams.params,
            "weights_path": weights_path,
        }
        with open(os.path.join(snapshot_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        manifest = self._load_manifest()
        manifest.append({
            "snapshot_id": snapshot_id,
            "snapshot_dir": snapshot_dir,
            "primary_score": eval_metrics.primary_score,
            "episode_id": eval_metrics.episode_id,
            "timestamp": snapshot.timestamp,
        })
        self._save_manifest(manifest)
        self._prune()

        self._best = snapshot
        return snapshot

    def load_best(self, agent) -> Optional[PolicySnapshot]:
        """
        Restore agent weights from the best known snapshot.
        Returns the snapshot or None if no snapshots exist.
        """
        if self._best is None:
            manifest = self._load_manifest()
            if not manifest:
                return None
            best_entry = max(manifest, key=lambda e: e["primary_score"])
            self._best = self._load_snapshot_from_dir(best_entry["snapshot_dir"])

        if self._best is not None:
            agent.load_weights(self._best.weights_path)
        return self._best

    @property
    def best(self) -> Optional[PolicySnapshot]:
        return self._best

    def _prune(self) -> None:
        """Remove oldest snapshots beyond max_snapshots."""
        manifest = self._load_manifest()
        if len(manifest) <= self.max_snapshots:
            return

        # Sort by timestamp, keep newest max_snapshots
        manifest.sort(key=lambda e: e["timestamp"])
        to_remove = manifest[: len(manifest) - self.max_snapshots]
        to_keep = manifest[len(manifest) - self.max_snapshots :]

        for entry in to_remove:
            snapshot_dir = entry["snapshot_dir"]
            if os.path.isdir(snapshot_dir):
                import shutil
                shutil.rmtree(snapshot_dir, ignore_errors=True)

        self._save_manifest(to_keep)

    def _load_manifest(self) -> List[dict]:
        manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return []
        with open(manifest_path) as f:
            return json.load(f)

    def _save_manifest(self, manifest: List[dict]) -> None:
        manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_snapshot_from_dir(self, snapshot_dir: str) -> Optional[PolicySnapshot]:
        metadata_path = os.path.join(snapshot_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path) as f:
            data = json.load(f)
        eval_metrics = EvalMetrics(
            primary_score=data["primary_score"],
            metrics=data["metrics"],
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
        )
        hyperparams = HyperparamSet(
            params=data["hyperparams"],
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
        )
        return PolicySnapshot(
            snapshot_id=data["snapshot_id"],
            eval_metrics=eval_metrics,
            hyperparams=hyperparams,
            weights_path=data["weights_path"],
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
        )
