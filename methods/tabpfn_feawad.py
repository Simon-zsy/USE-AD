"""Paper implementation: TabPFN + FEAWAD + ProtoFiLM-Dynamic + RoSAS + PDF.

See docs/PAPER_CODE_MAP.md and docs/DIFFERENCES.md for source provenance and
choices the paper does not specify. Test labels are used only after training.
"""
import time
from typing import Literal, Optional
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score
from tabpfn import TabPFNClassifier
from .base import BaseMethod
from model import AutoEncoder, TabPFNEnhancedDeviationNetwork, FeawadDevLoss

class TabPFNFEAWADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ae_epochs: int = Field(default=100, ge=1)
    devnet_epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=512, ge=2)
    nb_batch: int = Field(default=20, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    confidence_margin: float = Field(default=5.0, gt=0)
    tabpfn_n_estimators: int = Field(default=4, ge=1)
    expected_tabpfn_dim: int = Field(default=512, ge=1)
    model_path: str = "checkpoints/tabpfn-v2.5-classifier-v2.5_default.ckpt"
    device: str = "cpu"
    fusion_strategy: Literal["concat", "proto_film", "proto_film_dynamic"] = "proto_film_dynamic"
    pretrain_ae: bool = True
    use_rosas: bool = True
    rosas_lambda: float = Field(default=0.5, ge=0)
    use_pdf: bool = True
    # U0 size is not stated in the paper; sample without replacement with a fixed seed.
    normal_proto_samples: int = Field(default=1024, ge=1)
    # Recompute frozen features for new pseudo-samples by default, rather than
    # assuming that interpolation commutes with a nonlinear TabPFN encoder.
    rosas_embedding_mode: Literal["tabpfn", "linear"] = "tabpfn"
    inference_batch_size: int = Field(default=1024, ge=1)
    score_output_path: Optional[str] = None
    embedding_output_path: Optional[str] = None

def pdf_fuse(raw_score, tabpfn_probability, margin=5.0):
    """Eqs. 17-19. The mono-confidence term has coefficient one."""
    sd = torch.sigmoid(raw_score.reshape(-1) / margin)
    pt = tabpfn_probability.reshape(-1).to(sd)
    agreement = 1.0 - (sd - pt).abs()
    wd = (sd - 0.5).abs() + agreement
    wt = (pt - 0.5).abs() + agreement
    return (wd * sd + wt * pt) / (wd + wt).clamp_min(1e-8)

class TabPFNFEAWADMethod(BaseMethod):
    def __init__(self, config):
        self.config = config

    @property
    def name(self):
        return "tabpfn-feawad"

    def _embeddings(self, x):
        with torch.no_grad():
            values = np.asarray(self.tabpfn_.get_embeddings(x, data_source="test"))
        if values.ndim != 3:
            raise ValueError(f"Expected (estimators, samples, features), got {values.shape}")
        result = values.mean(axis=0).astype(np.float32)
        if result.shape[0] != len(x):
            raise ValueError("Unexpected TabPFN embedding sample axis")
        return result

    def fit(self, x, y, run_idx=0):
        cfg = self.config
        if cfg.batch_size % 2:
            raise ValueError("batch_size must be even for class-balanced batches")
        if cfg.device == "auto":
            cfg = cfg.model_copy(update={"device": "cuda" if torch.cuda.is_available() else "cpu"})
            self.config = cfg
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        normal = np.flatnonzero(y == 0)
        anomaly = np.flatnonzero(y == 1)
        if len(x) != len(y) or not len(normal) or not len(anomaly):
            raise ValueError("Training data must contain labeled anomalies and unlabeled samples")
        seed = 42 + run_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.RandomState(seed)
        ae = AutoEncoder(x.shape[1]).to(cfg.device)
        self.log(f"Phase 1: AE pretraining ({cfg.ae_epochs} epochs)")
        if cfg.pretrain_ae:
            optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.learning_rate)
            ae.train()
            for epoch in range(cfg.ae_epochs):
                for _ in range(cfg.nb_batch):
                    index = rng.choice(normal, cfg.batch_size, replace=True)
                    batch = torch.as_tensor(x[index], device=cfg.device)
                    reconstruction, _ = ae(batch)
                    # Algorithm 1: mean over samples of the squared L2 norm.
                    loss = (reconstruction - batch).square().sum(dim=1).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % 20 == 0:
                    self.log(f"AE epoch {epoch + 1}/{cfg.ae_epochs}: {loss.item():.6f}")

        self.log("Phase 2: frozen TabPFN feature caching and prototypes")
        self.tabpfn_ = TabPFNClassifier(model_path=cfg.model_path,
            n_estimators=cfg.tabpfn_n_estimators, device=cfg.device, random_state=seed)
        self.tabpfn_.fit(x, y.astype(int))
        embeddings = self._embeddings(x)
        if embeddings.shape[1] != cfg.expected_tabpfn_dim:
            raise ValueError(f"Checkpoint returns {embeddings.shape[1]} dimensions; configuration expects "
                f"{cfg.expected_tabpfn_dim}. The paper requires 512. Supply a compatible checkpoint, "
                "or explicitly set expected_tabpfn_dim to the measured dimension for a documented variant.")
        self.tabpfn_dim_ = embeddings.shape[1]
        self.network_ = TabPFNEnhancedDeviationNetwork(input_dim=x.shape[1],
            autoencoder=ae, tabpfn_dim=self.tabpfn_dim_, fusion_strategy=cfg.fusion_strategy).to(cfg.device)
        subset = rng.choice(normal, min(len(normal), cfg.normal_proto_samples), replace=False)
        cn = torch.as_tensor(embeddings[subset].mean(axis=0), device=cfg.device)
        ca = torch.as_tensor(embeddings[anomaly].mean(axis=0), device=cfg.device)
        if cfg.fusion_strategy == "proto_film_dynamic":
            self.network_.fusion.set_prototypes(cn, ca)
        elif cfg.fusion_strategy == "proto_film":
            self.network_.fusion.set_conditioning(ca - cn)

        self.log(f"Phase 3: joint training ({cfg.devnet_epochs} fixed epochs)")
        criterion = FeawadDevLoss(cfg.confidence_margin).to(cfg.device)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=cfg.learning_rate)
        half = min(cfg.batch_size // 2, len(normal))
        self.network_.train()
        for epoch in range(cfg.devnet_epochs):
            for _ in range(cfg.nb_batch):
                ni = rng.choice(normal, half, replace=False)
                ai = rng.choice(anomaly, half, replace=True)
                indices = np.concatenate([ni, ai])
                xb = torch.as_tensor(x[indices], device=cfg.device)
                tb = torch.as_tensor(embeddings[indices], device=cfg.device)
                labels = torch.cat([torch.zeros(half), torch.ones(half)]).to(cfg.device)
                score, error, _ = self.network_(xb, tb)
                loss = criterion(score, error, labels)
                if cfg.use_rosas and cfg.rosas_lambda:
                    # Pair each anomaly position with exactly one unlabeled position.
                    alpha = torch.rand(half, 1, device=cfg.device)
                    mixed_x = (1 - alpha) * xb[:half] + alpha * xb[half:]
                    if cfg.rosas_embedding_mode == "tabpfn":
                        mixed_t = torch.as_tensor(self._embeddings(mixed_x.detach().cpu().numpy()), device=cfg.device)
                    else:
                        mixed_t = (1 - alpha) * tb[:half] + alpha * tb[half:]
                    mixed_score, _, _ = self.network_(mixed_x, mixed_t)
                    target = alpha.reshape(-1) * cfg.confidence_margin
                    interpolation_loss = (mixed_score.reshape(-1) - target).abs().mean()
                    loss = loss + cfg.rosas_lambda * interpolation_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                self.log(f"Joint epoch {epoch + 1}/{cfg.devnet_epochs}: {loss.item():.6f}")
        self.network_.eval()
        return self

    def predict_score(self, x):
        cfg = self.config
        x = np.asarray(x, dtype=np.float32)
        embeddings = self._embeddings(x)
        output, hidden = [], []
        with torch.no_grad():
            for start in range(0, len(x), cfg.inference_batch_size):
                xb = torch.as_tensor(x[start:start+cfg.inference_batch_size], device=cfg.device)
                tb = torch.as_tensor(embeddings[start:start+cfg.inference_batch_size], device=cfg.device)
                score, _, _, _, penultimate = self.network_(xb, tb, return_hidden=True)
                output.append(score.reshape(-1).cpu())
                hidden.append(penultimate.cpu().numpy())
        raw = torch.cat(output)
        if cfg.use_pdf:
            with torch.no_grad():
                probability = self.tabpfn_.predict_proba(x)[:, 1]
            raw = pdf_fuse(raw, torch.as_tensor(probability), cfg.confidence_margin)
        if cfg.embedding_output_path:
            np.savez(cfg.embedding_output_path, penultimate=np.concatenate(hidden), tabpfn=embeddings)
        return raw.numpy()

    def run(self, X_train, y_train, X_test, y_test, run_idx=0):
        start = time.time()
        self.fit(X_train, y_train, run_idx)
        scores = self.predict_score(X_test)
        if self.config.score_output_path:
            np.savez(self.config.score_output_path, scores=scores, labels=y_test)
        return {"auc_roc": roc_auc_score(y_test, scores),
                "auc_pr": average_precision_score(y_test, scores),
                "runtime_seconds": time.time() - start,
                "epochs": self.config.devnet_epochs}
