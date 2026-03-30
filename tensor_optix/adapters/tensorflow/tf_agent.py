import tensorflow as tf
import numpy as np
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.types import EpisodeData, HyperparamSet


class TFAgent(BaseAgent):
    """
    Wraps a user-built TensorFlow model into the BaseAgent interface.

    The user passes in:
    - model:       their tf.keras.Model (the policy network)
    - optimizer:   tf.keras.optimizers.Optimizer instance
    - hyperparams: initial HyperparamSet

    For algorithm-specific learning logic (PPO clipping, SAC entropy tuning,
    DQN target network updates), subclass TFAgent and override learn().
    The base learn() performs a generic policy gradient update.

    Automatically applied hyperparam keys in set_hyperparams():
    - "learning_rate": float → applied to optimizer.learning_rate
    All other keys are stored and accessible via self.hyperparams for use
    in the subclass's learn() override.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        hyperparams: HyperparamSet,
        compute_loss_fn: callable = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self._hyperparams = hyperparams.copy()
        self._compute_loss_fn = compute_loss_fn

    def act(self, observation) -> np.ndarray:
        """
        Forward pass. Returns a sampled action index (for discrete action spaces).
        The model output is treated as logits; action is sampled via argmax.

        For continuous action spaces or custom sampling, override this method
        in a subclass.
        """
        obs = tf.cast(observation, tf.float32)
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, axis=0)
        logits = self.model(obs, training=False)
        action = int(tf.argmax(logits, axis=-1).numpy()[0])
        return action

    def learn(self, episode_data: EpisodeData) -> dict:
        """
        Policy gradient update using GradientTape.

        If episode_data.values is provided (critic V(s_t) estimates), computes
        Actor-Critic advantages:
            A_t = G_t - V(s_t)
        This is an unbiased, lower-variance gradient estimator vs. plain REINFORCE.
        The advantage baseline does not bias the gradient by the policy gradient theorem:
            ∇J = E[∇ log π(a|s) · A_t]

        If episode_data.values is None, falls back to normalized REINFORCE returns.

        Override this method in a subclass for algorithm-specific logic (PPO, SAC, DQN).

        Returns dict: {loss, grad_norm, baseline_type, explained_variance}
        """
        gamma = float(self._hyperparams.params.get("gamma", 0.99))

        # Compute discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        rewards = episode_data.rewards
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_array = np.array(returns, dtype=np.float32)

        if episode_data.values is not None:
            # A2C: advantage = G_t - V(s_t)
            # Subtracting the baseline reduces variance without introducing bias.
            # Normalize advantages (not returns) for gradient stability.
            values_array = np.array(episode_data.values, dtype=np.float32)
            advantages = returns_array - values_array
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            targets = tf.cast(advantages, tf.float32)

            # Explained variance: how much of return variance V(s) explains.
            # 1.0 = perfect baseline, 0.0 = no better than mean, <0 = harmful.
            var_returns = float(np.var(returns_array))
            explained_var = float(
                1.0 - np.var(returns_array - values_array) / (var_returns + 1e-8)
            )
            baseline_type = "a2c"
        else:
            # REINFORCE fallback: normalize returns as the baseline
            returns_tensor = tf.cast(returns_array, tf.float32)
            if len(returns_array) > 1:
                targets = (returns_tensor - tf.reduce_mean(returns_tensor)) / (
                    tf.math.reduce_std(returns_tensor) + 1e-8
                )
            else:
                targets = returns_tensor
            explained_var = 0.0
            baseline_type = "reinforce"

        obs = tf.cast(episode_data.observations, tf.float32)

        if self._compute_loss_fn is not None:
            with tf.GradientTape() as tape:
                loss = self._compute_loss_fn(self.model, episode_data, targets)
        else:
            with tf.GradientTape() as tape:
                logits = self.model(obs, training=True)
                actions = tf.cast(episode_data.actions, tf.int32)
                log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions, logits=logits
                )
                loss = tf.reduce_mean(log_probs * targets)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grad_norm = tf.linalg.global_norm(grads).numpy()
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {
            "loss": float(loss.numpy()),
            "grad_norm": float(grad_norm),
            "baseline_type": baseline_type,
            "explained_variance": explained_var,
        }

    def get_hyperparams(self) -> HyperparamSet:
        # Sync learning_rate from optimizer in case it was changed externally
        self._hyperparams.params["learning_rate"] = float(
            self.optimizer.learning_rate.numpy()
            if hasattr(self.optimizer.learning_rate, "numpy")
            else self.optimizer.learning_rate
        )
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        """
        Apply hyperparams. Known key: "learning_rate" → updates optimizer directly.
        All keys stored for access in learn() overrides.
        """
        self._hyperparams = hyperparams.copy()
        if "learning_rate" in hyperparams.params:
            lr = float(hyperparams.params["learning_rate"])
            self.optimizer.learning_rate.assign(lr)

    def save_weights(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))

    def load_weights(self, path: str) -> None:
        import os
        loaded = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        for var, loaded_var in zip(self.model.trainable_variables, loaded.trainable_variables):
            var.assign(loaded_var)
