"""Focused numerical and leakage checks; no GPU, real checkpoint or downloads."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import prepare_weakly_supervised_data
from model import AutoEncoder, FeawadDevLoss, TabPFNEnhancedDeviationNetwork
from model.fusion import DynamicProtoFiLMFusion
from methods.tabpfn_feawad import TabPFNFEAWADConfig, TabPFNFEAWADMethod, pdf_fuse

class FakeTabPFN:
    """A deterministic nonlinear frozen feature source for pipeline smoke tests."""
    embedding_calls = []
    def __init__(self, **kwargs):
        pass
    def fit(self, x, y):
        return self
    def get_embeddings(self, x, data_source="test"):
        self.embedding_calls.append(len(x))
        values = np.resize(np.tanh(x), (len(x), 512)).astype(np.float32)
        return values[None]
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-np.mean(x, axis=1)))
        return np.column_stack([1-p, p])

class PaperProtocolTests(unittest.TestCase):
    def test_split_never_uses_test_samples_for_training_contamination(self):
        x = np.arange(1000 * 4, dtype=np.float32).reshape(1000, 4)
        y = np.r_[np.zeros(900), np.ones(100)]
        data = prepare_weakly_supervised_data(x, y)
        self.assertFalse(set(data['train_indices']) & set(data['test_indices']))
        self.assertFalse(set(data['hidden_anomaly_indices']) & set(data['labeled_anomaly_indices']))
        self.assertEqual(int(data['train_y'].sum()), 8)
        self.assertLess(abs(data['actual_cont_rate'] - 0.02), 1 / len(data['train_y']))

    def test_label_budget_and_unavailable_contamination_fail_explicitly(self):
        x = np.ones((100, 4), dtype=np.float32)
        y = np.r_[np.zeros(90), np.ones(10)]
        with self.assertRaisesRegex(ValueError, 'distinct labeled anomalies'):
            prepare_weakly_supervised_data(x, y, known_outliers=15)
        with self.assertRaisesRegex(ValueError, 'distinct hidden training anomalies'):
            prepare_weakly_supervised_data(x, y, known_outliers=8)

    def test_paper_pdf_weights_not_the_historical_factor_two(self):
        score = torch.tensor([-3., 1., 8.])
        pt = torch.tensor([0.9, 0.2, 0.7])
        sd = torch.sigmoid(score / 5)
        agreement = 1 - (sd-pt).abs()
        wd, wt = (sd-.5).abs()+agreement, (pt-.5).abs()+agreement
        torch.testing.assert_close(pdf_fuse(score, pt), (wd*sd+wt*pt)/(wd+wt))

    def test_fusion_matches_soft_prototype_equations(self):
        fusion = DynamicProtoFiLMFusion(2, 3)
        cn, ca = torch.tensor([0., 1., 0.]), torch.tensor([2., 0., 1.])
        fusion.set_prototypes(cn, ca)
        with torch.no_grad():
            fusion.gamma_net.weight.fill_(0.1)
            fusion.beta_net.weight.fill_(0.2)
        h, t = torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([[0., 1., 0.], [2., 0., 1.]])
        alpha = torch.softmax(torch.stack([-((t-cn)**2).sum(1), -((t-ca)**2).sum(1)], 1), 1)
        delta = alpha[:, :1]*cn + alpha[:, 1:]*ca
        expected = torch.cat([(1+fusion.gamma_net(delta))*h+fusion.beta_net(delta), t], 1)
        torch.testing.assert_close(fusion(h, t), expected)
        self.assertFalse(fusion.normal_proto.requires_grad)

    def test_residual_direction_and_margin_loss(self):
        ae = AutoEncoder(4)
        self.assertFalse(hasattr(ae, 'input_layer'))
        network = TabPFNEnhancedDeviationNetwork(4, ae, 512, 'concat')
        x, t = torch.randn(3, 4), torch.randn(3, 512)
        reconstruction, _ = ae(x)
        _, error, residual = network(x, t)
        expected = reconstruction - x
        torch.testing.assert_close(error, expected.norm(dim=1))
        torch.testing.assert_close(residual, expected/(expected.norm(dim=1, keepdim=True)+1e-8))
        loss = FeawadDevLoss(5)(torch.tensor([[2.], [3.]]), torch.tensor([1., 2.]), torch.tensor([0., 1.]))
        self.assertAlmostEqual(loss.item(), (2+1+2+3)/2)

    def test_core_training_and_inference_with_frozen_surrogate(self):
        x = np.random.RandomState(7).rand(30, 4).astype(np.float32)
        y = np.r_[np.zeros(26), np.ones(4)]
        cfg = TabPFNFEAWADConfig(ae_epochs=1, devnet_epochs=1, nb_batch=1, batch_size=4)
        for fusion in ['concat', 'proto_film', 'proto_film_dynamic']:
            FakeTabPFN.embedding_calls = []
            with self.subTest(fusion=fusion), patch('methods.tabpfn_feawad.TabPFNClassifier', FakeTabPFN):
                method = TabPFNFEAWADMethod(cfg.model_copy(update={'fusion_strategy': fusion}))
                method.fit(x, y)
                scores = method.predict_score(x[:5])
                self.assertEqual(scores.shape, (5,))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                self.assertEqual(FakeTabPFN.embedding_calls[:2], [30, 2])

    def test_raw_aggregation_excludes_average_rows_and_keeps_latest(self):
        spec = importlib.util.spec_from_file_location('summary', ROOT / 'scripts/summarize_results.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / '20260101_000000_ours_1_ALOI.csv').write_text('run,auc_roc,auc_pr\n0,0.1,0.1\n')
            (path / '20260102_000000_ours_1_ALOI.csv').write_text('run,auc_roc,auc_pr\n0,0.2,0.3\n1,0.8,0.9\n-1,0.99,0.99\n')
            rows = module.summarize(path)
            self.assertEqual(rows.loc[rows.metric == 'auc_roc', 'mean'].iloc[0], 0.5)
            self.assertEqual(rows['n'].tolist(), [2, 2])


    def test_dimension_mismatch_is_never_silently_projected(self):
        class WrongDimension(FakeTabPFN):
            def get_embeddings(self, x, data_source="test"):
                return np.zeros((1, len(x), 192), dtype=np.float32)
        cfg = TabPFNFEAWADConfig(ae_epochs=1, devnet_epochs=1, nb_batch=1, batch_size=4)
        x = np.ones((12, 4), dtype=np.float32)
        y = np.r_[np.zeros(8), np.ones(4)]
        with patch('methods.tabpfn_feawad.TabPFNClassifier', WrongDimension):
            with self.assertRaisesRegex(ValueError, 'Checkpoint returns 192 dimensions'):
                TabPFNFEAWADMethod(cfg).fit(x, y)

    def test_benchmark_entry_writes_real_run_and_average_rows(self):
        import runpy
        import yaml
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            rng = np.random.RandomState(19)
            np.savez(data / '1_synthetic.npz', X=rng.rand(1000, 4).astype(np.float32),
                     y=np.r_[np.zeros(900), np.ones(100)])
            configuration = {'experiments': [{'name': 'ours', 'method': 'tabpfn-feawad',
                'args': {'ae_epochs': 1, 'devnet_epochs': 1, 'nb_batch': 1, 'batch_size': 4}}]}
            path = root / 'experiments.yaml'
            path.write_text(yaml.safe_dump(configuration))
            argv = ['benchmark.py', '--experiments', 'ours', '--exp-config', str(path),
                '--config', str(ROOT / 'overrides.yaml'), '--dataset-dir', str(data),
                '--runs', '1', '--device', 'cpu', '--output-dir', str(root / 'results'),
                '--log-dir', str(root / 'logs')]
            with patch('sys.argv', argv), patch('methods.tabpfn_feawad.TabPFNClassifier', FakeTabPFN):
                runpy.run_path(str(ROOT / 'benchmark.py'), run_name='__main__')
            csvs = list((root / 'results').glob('*.csv'))
            self.assertEqual(len(csvs), 1)
            frame = pd.read_csv(csvs[0])
            self.assertEqual(frame['run'].tolist(), [0, -1])
            self.assertTrue(np.isfinite(frame['auc_roc']).all())

if __name__ == '__main__':
    unittest.main()
