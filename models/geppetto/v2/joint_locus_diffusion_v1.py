from __future__ import annotations

"""Clean-room conditional diffusion challenger for Geppetto joint loci.

This is D1 research machinery only. It predicts the continuous normalized XYZ
locus conditioned on the already-computed Geppetto control state. It does not
own STOP/root/parent/support evidence and does not feed sampled XYZ back into the
autoregressive recurrence.

The sampler uses deterministic DDIM-style eta=0 updates. Multiple samples expose
a locus hypothesis set; the representative is an actual medoid sample, never the
mean between separated hypotheses.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .joint_locus_distribution_v1 import JointLocusDistributionV1


@dataclass(frozen=True)
class ConditionalDiffusionLocusConfigV1:
    condition_dim: int = 192
    hidden_dim: int = 384
    time_dim: int = 64
    train_steps: int = 1000
    sample_steps: int = 32
    sample_count: int = 3
    position_scale: float = 1.25
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    architecture_id: str = "RealSaS.Geppetto.ConditionalJointLocusDiffusion.v1"

    def validate(self) -> None:
        if min(
            self.condition_dim,
            self.hidden_dim,
            self.time_dim,
            self.train_steps,
            self.sample_steps,
            self.sample_count,
        ) <= 0:
            raise ValueError("diffusion locus cardinality invalid")
        if self.time_dim % 2:
            raise ValueError("diffusion locus time_dim must be even")
        if self.sample_steps > self.train_steps:
            raise ValueError("diffusion sample steps exceed train steps")
        if self.sample_count < 2:
            raise ValueError("diffusion locus requires multiple hypotheses")
        if not (0.0 < self.beta_start < self.beta_end < 1.0):
            raise ValueError("diffusion beta schedule invalid")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("diffusion locus position scale invalid")


def _sinusoidal_time_embedding(
    timesteps: torch.Tensor, dim: int
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be [B]")
    half = dim // 2
    scale = math.log(10000.0) / max(1, half - 1)
    freq = torch.exp(
        -scale
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    phase = timesteps.float()[:, None] * freq[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class ConditionalDiffusionLocusHeadV1(nn.Module):
    distribution_kind = "CONDITIONAL_DIFFUSION_DDIM_V1"

    def __init__(
        self,
        config: ConditionalDiffusionLocusConfigV1 = ConditionalDiffusionLocusConfigV1(),
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.condition = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.time = nn.Sequential(
            nn.Linear(config.time_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 3),
        )

        betas = torch.linspace(
            float(config.beta_start),
            float(config.beta_end),
            int(config.train_steps),
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _predict_noise(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[-1] != 3:
            raise ValueError("diffusion x_t must be [B,3]")
        if condition.ndim != 2 or condition.shape != (
            x_t.shape[0],
            self.config.condition_dim,
        ):
            raise ValueError("diffusion condition shape drift")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("diffusion timestep shape drift")
        c = self.condition(condition.float())
        te = _sinusoidal_time_embedding(
            timesteps, self.config.time_dim
        ).to(c.dtype)
        t = self.time(te)
        return self.denoiser(torch.cat([x_t.float(), c, t], dim=-1))

    def training_loss(
        self,
        condition: torch.Tensor,
        target_xyz_normalized: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            target_xyz_normalized.ndim != 2
            or target_xyz_normalized.shape[-1] != 3
            or condition.shape[0] != target_xyz_normalized.shape[0]
        ):
            raise ValueError("diffusion training target shape drift")
        B = target_xyz_normalized.shape[0]
        device = target_xyz_normalized.device
        target = target_xyz_normalized.float().clamp(
            -self.config.position_scale, self.config.position_scale
        )
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.train_steps,
                (B,),
                device=device,
                generator=generator,
            )
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=device,
                dtype=target.dtype,
                generator=generator,
            )
        else:
            noise = noise.to(device=device, dtype=target.dtype)
        if timesteps.min().item() < 0 or timesteps.max().item() >= self.config.train_steps:
            raise ValueError("diffusion timestep outside schedule")
        if noise.shape != target.shape or not torch.isfinite(noise).all():
            raise ValueError("diffusion noise invalid")

        ab = self.alpha_bar[timesteps].to(target.dtype)[:, None]
        x_t = torch.sqrt(ab) * target + torch.sqrt(1.0 - ab) * noise
        pred = self._predict_noise(x_t, condition, timesteps)
        mse = torch.mean((pred - noise) ** 2)

        x0 = (
            x_t - torch.sqrt(1.0 - ab) * pred
        ) / torch.sqrt(ab).clamp_min(1e-8)
        x0 = x0.clamp(-self.config.position_scale, self.config.position_scale)
        recon = torch.mean((x0 - target) ** 2)
        return {
            "total": mse,
            "noise_mse": mse,
            "x0_reconstruction_mse": recon,
            "predicted_x0": x0,
        }

    def _sampling_schedule(self, device) -> torch.Tensor:
        # Descending unique schedule including T-1 and 0.
        row = torch.linspace(
            self.config.train_steps - 1,
            0,
            self.config.sample_steps,
            device=device,
            dtype=torch.float64,
        ).round().long()
        row = torch.unique_consecutive(row)
        if int(row[-1]) != 0:
            row = torch.cat([row, row.new_zeros((1,))], dim=0)
        return row

    @torch.no_grad()
    def sample_hypotheses(
        self,
        condition: torch.Tensor,
        *,
        seed: int,
        sample_count: int | None = None,
    ) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[-1] != self.config.condition_dim:
            raise ValueError("diffusion condition shape drift")
        count = self.config.sample_count if sample_count is None else int(sample_count)
        if count < 2:
            raise ValueError("diffusion sample_count must be >=2")
        B = condition.shape[0]
        device = condition.device
        cond = (
            condition[:, None, :]
            .expand(B, count, self.config.condition_dim)
            .reshape(B * count, self.config.condition_dim)
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        x = torch.randn(
            (B * count, 3),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        schedule = self._sampling_schedule(device)
        for si, t_scalar in enumerate(schedule):
            t = int(t_scalar.item())
            tt = torch.full(
                (B * count,), t, device=device, dtype=torch.long
            )
            eps = self._predict_noise(x, cond, tt)
            ab_t = self.alpha_bar[t].to(x.dtype)
            x0 = (
                x - torch.sqrt(1.0 - ab_t) * eps
            ) / torch.sqrt(ab_t).clamp_min(1e-8)
            x0 = x0.clamp(
                -self.config.position_scale,
                self.config.position_scale,
            )
            if si == len(schedule) - 1 or t == 0:
                x = x0
                break
            t_prev = int(schedule[si + 1].item())
            ab_prev = self.alpha_bar[t_prev].to(x.dtype)
            # Deterministic DDIM eta=0 update.
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps

        return x.reshape(B, count, 3)

    @torch.no_grad()
    def distribution(
        self,
        condition: torch.Tensor,
        *,
        seed: int,
        sample_count: int | None = None,
    ) -> JointLocusDistributionV1:
        samples = self.sample_hypotheses(
            condition, seed=seed, sample_count=sample_count
        )
        mean = samples.mean(dim=1, keepdim=True)
        d2 = ((samples - mean) ** 2).sum(dim=-1)
        medoid = torch.argsort(d2, dim=-1, stable=True)[:, 0]
        gather3 = medoid[:, None, None].expand(
            samples.shape[0], 1, 3
        )
        representative = torch.gather(samples, 1, gather3).squeeze(1)
        std = samples.std(dim=1, unbiased=False).clamp_min(1e-6)
        rep_log_sigma = torch.log(std)
        mode_ls = rep_log_sigma[:, None, :].expand_as(samples)
        logits = torch.zeros(
            samples.shape[:2],
            device=samples.device,
            dtype=samples.dtype,
        )
        return JointLocusDistributionV1(
            representative_position=representative,
            representative_log_sigma=rep_log_sigma,
            representative_mode_index=medoid,
            modes_normalized=samples,
            mode_log_sigma=mode_ls,
            mode_logits=logits,
            distribution_kind=self.distribution_kind,
        )


__all__ = [
    "ConditionalDiffusionLocusConfigV1",
    "ConditionalDiffusionLocusHeadV1",
]
