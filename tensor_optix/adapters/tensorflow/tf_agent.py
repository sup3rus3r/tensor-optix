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
        Generic policy gradient update using GradientTape.

        Uses REINFORCE-style returns: discounted cumulative reward as target.
        Override this method in a subclass for algorithm-specific logic.

        Returns dict: {loss, grad_norm}
        """
        gamma = float(self._hyperparams.params.get("gamma", 0.99))

        # Compute discounted returns
        rewards = episode_data.rewards
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_tensor = tf.cast(returns, tf.float32)

        # Normalize returns for stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - tf.reduce_mean(returns_tensor)) / (
                tf.math.reduce_std(returns_tensor) + 1e-8
            )

        obs = tf.cast(episode_data.observations, tf.float32)

        if self._compute_loss_fn is not None:
            with tf.GradientTape() as tape:
                loss = self._compute_loss_fn(self.model, episode_data, returns_tensor)
        else:
            with tf.GradientTape() as tape:
                logits = self.model(obs, training=True)
                actions = tf.cast(episode_data.actions, tf.int32)
                log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions, logits=logits
                )
                loss = tf.reduce_mean(log_probs * returns_tensor)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grad_norm = tf.linalg.global_norm(grads).numpy()
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {
            "loss": float(loss.numpy()),
            "grad_norm": float(grad_norm),
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
