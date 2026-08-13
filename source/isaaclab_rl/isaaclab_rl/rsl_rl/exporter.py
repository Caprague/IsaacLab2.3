# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    if _is_depth_image_recurrent_policy(policy):
        policy_exporter = _TorchPolicyExporterDepthImageRecurrent(policy, normalizer)
    else:
        policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if _is_depth_image_recurrent_policy(policy):
        policy_exporter = _OnnxPolicyExporterDepthImageRecurrent(policy, normalizer, verbose)
    else:
        policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


def _is_depth_image_recurrent_policy(policy: object) -> bool:
    """Check whether the policy is the recurrent depth-image student-teacher network.

    This architecture fuses a depth CNN with a GRU before the student MLP
    (``StudentTeacherDepthImageRecurrent``), so it needs a dedicated exporter that
    replicates the full student forward pass instead of the standard
    ``obs -> rnn -> student MLP`` chain.
    """
    return bool(getattr(policy, "is_recurrent", False)) and hasattr(policy, "depth_cnn")


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _TorchPolicyExporterDepthImageRecurrent(torch.nn.Module):
    """Exporter of the recurrent depth-image student policy into a JIT file.

    The standard exporter assumes ``obs -> normalizer -> rnn -> student MLP``. The
    ``StudentTeacherDepthImageRecurrent`` policy instead computes::

        depth_latent = depth_cnn(depth_img)
        gru_in = concat(prop_latest, depth_aux, depth_latent)
        latents = gru_output_mlp(memory_s(gru_input_mlp(gru_in)))
        actions = student(concat(normalizer(prop_all), latents))

    This exporter expects the flat student observation vector with the layout of the
    concatenated policy observation groups: the proprioceptive history block followed
    by the ``mid360_depth`` block (or the reverse when the depth group is listed first
    in ``policy.obs_groups["policy"]``).
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = True
        # copy the student policy modules
        self.depth_cnn = copy.deepcopy(policy.depth_cnn)
        self.gru_input_mlp = copy.deepcopy(policy.gru_input_mlp)
        self.rnn = copy.deepcopy(policy.memory_s.rnn)
        self.gru_output_mlp = copy.deepcopy(policy.gru_output_mlp)
        self.student = copy.deepcopy(policy.student)
        # observation layout of the flat input vector
        self.prop_dims = policy.num_student_basic_obs
        self.depth_flat_dim = policy.depth_flat_dim
        self.depth_aux_dim = policy.depth_aux_dim
        self.depth_height = policy.depth_height
        self.depth_width = policy.depth_width
        self.depth_channels = policy.depth_channels
        self.proprio_per_frame = policy.proprio_per_frame
        self.latent_dim = 32  # matches the two 32-dim latent codes of the policy
        self.depth_first = policy.obs_groups["policy"][0] == policy.depth_obs_group
        # set up recurrent network
        self.rnn.cpu()
        self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        if self.rnn_type == "lstm":
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        elif self.rnn_type == "gru":
            self.forward = self.forward_gru
            self.reset = self.reset_memory
        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.student_obs_normalizer = copy.deepcopy(normalizer)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

    def _split_student_obs(self, x):
        if self.depth_first:
            depth_flat = x[:, : self.depth_flat_dim + self.depth_aux_dim]
            prop_all = x[:, self.depth_flat_dim + self.depth_aux_dim :]
        else:
            prop_all = x[:, : self.prop_dims]
            depth_flat = x[:, self.prop_dims :]
        depth_img = depth_flat[:, : self.depth_flat_dim].reshape(
            -1, self.depth_channels, self.depth_height, self.depth_width
        )
        depth_aux = depth_flat[:, self.depth_flat_dim :]
        return prop_all, depth_img, depth_aux

    def _student_actions(self, gru_out, prop_all):
        latents = self.gru_output_mlp(gru_out)
        depth_latent = latents[:, : self.latent_dim]
        privilege_latent = latents[:, self.latent_dim :]
        prop_all_norm = self.student_obs_normalizer(prop_all)
        student_input = torch.cat([prop_all_norm, depth_latent, privilege_latent], dim=-1)
        return self.student(student_input)

    def forward_gru(self, x):
        prop_all, depth_img, depth_aux = self._split_student_obs(x)
        prop_latest = prop_all[:, -self.proprio_per_frame :]
        depth_latent = self.depth_cnn(depth_img)
        gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)
        gru_feat = self.gru_input_mlp(gru_input)
        x, h = self.rnn(gru_feat.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self._student_actions(x, prop_all)

    def forward_lstm(self, x):
        prop_all, depth_img, depth_aux = self._split_student_obs(x)
        prop_latest = prop_all[:, -self.proprio_per_frame :]
        depth_latent = self.depth_cnn(depth_img)
        gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)
        gru_feat = self.gru_input_mlp(gru_input)
        x, (h, c) = self.rnn(gru_feat.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self._student_actions(x, prop_all)

    def forward(self, x):
        return self.forward_gru(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporterDepthImageRecurrent(torch.nn.Module):
    """Exporter of the recurrent depth-image student policy into an ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = True
        # copy the student policy modules
        self.depth_cnn = copy.deepcopy(policy.depth_cnn)
        self.gru_input_mlp = copy.deepcopy(policy.gru_input_mlp)
        self.rnn = copy.deepcopy(policy.memory_s.rnn)
        self.gru_output_mlp = copy.deepcopy(policy.gru_output_mlp)
        self.student = copy.deepcopy(policy.student)
        # observation layout of the flat input vector
        self.prop_dims = policy.num_student_basic_obs
        self.depth_flat_dim = policy.depth_flat_dim
        self.depth_aux_dim = policy.depth_aux_dim
        self.depth_height = policy.depth_height
        self.depth_width = policy.depth_width
        self.depth_channels = policy.depth_channels
        self.proprio_per_frame = policy.proprio_per_frame
        self.latent_dim = 32  # matches the two 32-dim latent codes of the policy
        self.depth_first = policy.obs_groups["policy"][0] == policy.depth_obs_group
        # set up recurrent network
        self.rnn.cpu()
        self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
        if self.rnn_type == "lstm":
            self.forward = self.forward_lstm
        elif self.rnn_type == "gru":
            self.forward = self.forward_gru
        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.student_obs_normalizer = copy.deepcopy(normalizer)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

    def _split_student_obs(self, x):
        if self.depth_first:
            depth_flat = x[:, : self.depth_flat_dim + self.depth_aux_dim]
            prop_all = x[:, self.depth_flat_dim + self.depth_aux_dim :]
        else:
            prop_all = x[:, : self.prop_dims]
            depth_flat = x[:, self.prop_dims :]
        depth_img = depth_flat[:, : self.depth_flat_dim].reshape(
            -1, self.depth_channels, self.depth_height, self.depth_width
        )
        depth_aux = depth_flat[:, self.depth_flat_dim :]
        return prop_all, depth_img, depth_aux

    def _student_actions(self, gru_out, prop_all):
        latents = self.gru_output_mlp(gru_out)
        depth_latent = latents[:, : self.latent_dim]
        privilege_latent = latents[:, self.latent_dim :]
        prop_all_norm = self.student_obs_normalizer(prop_all)
        student_input = torch.cat([prop_all_norm, depth_latent, privilege_latent], dim=-1)
        return self.student(student_input)

    def forward_gru(self, x_in, h_in):
        prop_all, depth_img, depth_aux = self._split_student_obs(x_in)
        prop_latest = prop_all[:, -self.proprio_per_frame :]
        depth_latent = self.depth_cnn(depth_img)
        gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)
        gru_feat = self.gru_input_mlp(gru_input)
        x, h = self.rnn(gru_feat.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self._student_actions(x, prop_all), h

    def forward_lstm(self, x_in, h_in, c_in):
        prop_all, depth_img, depth_aux = self._split_student_obs(x_in)
        prop_latest = prop_all[:, -self.proprio_per_frame :]
        depth_latent = self.depth_cnn(depth_img)
        gru_input = torch.cat([prop_latest, depth_aux, depth_latent], dim=-1)
        gru_feat = self.gru_input_mlp(gru_input)
        x, (h, c) = self.rnn(gru_feat.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self._student_actions(x, prop_all), h, c

    def forward(self, x_in, h_in):
        return self.forward_gru(x_in, h_in)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        obs = torch.zeros(1, self.prop_dims + self.depth_flat_dim + self.depth_aux_dim)
        h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        elif self.rnn_type == "gru":
            torch.onnx.export(
                self,
                (obs, h_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs", "h_in"],
                output_names=["actions", "h_out"],
                dynamic_axes={},
            )
        else:
            raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
