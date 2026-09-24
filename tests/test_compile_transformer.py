"""Regression checks for optional regional Transformer compilation."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transkun.LayersTransformer import Backbone, BasicBlock
from transkun.runtime import maybe_compile_transformer


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.inputConv = nn.Linear(4, 4)
        self.backbone.encoderLayers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.scorer = nn.Linear(4, 1)

    def forward(self, x):
        x = self.backbone.inputConv(x)
        for layer in self.backbone.encoderLayers:
            x = torch.relu(layer(x))
        return self.scorer(x)


class CompileTransformerTests(unittest.TestCase):
    def test_disabled_leaves_model_untouched(self):
        model = SmallModel()
        with patch.object(nn.Linear, "compile") as compile_mock:
            self.assertIs(maybe_compile_transformer(model), model)
            compile_mock.assert_not_called()

    def test_compiles_only_encoder_layers_in_place(self):
        model = SmallModel()
        parameters = {name: id(param) for name, param in model.named_parameters()}
        keys = set(model.state_dict())
        with patch.object(nn.Linear, "compile") as compile_mock:
            self.assertIs(maybe_compile_transformer(model, enabled=True), model)
            self.assertEqual(compile_mock.call_count, 2)
            for call in compile_mock.call_args_list:
                self.assertEqual(call.kwargs, dict(
                    backend="inductor", mode="default", fullgraph=False, dynamic=True,
                ))
        self.assertEqual(keys, set(model.state_dict()))
        self.assertEqual(parameters, {name: id(param) for name, param in model.named_parameters()})

    @unittest.skipUnless(hasattr(nn.Module, "compile"), "nn.Module.compile is unavailable")
    def test_dynamic_batch_and_length_match_eager_outputs_and_gradients(self):
        torch.manual_seed(5)
        eager = SmallModel().eval()
        compiled = copy.deepcopy(eager)
        maybe_compile_transformer(compiled, enabled=True, backend="eager")
        self.assertEqual(set(eager.state_dict()), set(compiled.state_dict()))

        for shape in ((4, 5, 4), (3, 7, 4), (1, 5, 4)):
            eager.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            x = torch.randn(shape, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            expected = eager(x)
            actual = compiled(y)
            torch.testing.assert_close(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(y.grad, x.grad)
            for left, right in zip(eager.parameters(), compiled.parameters()):
                torch.testing.assert_close(right.grad, left.grad)

    @unittest.skipUnless(hasattr(nn.Module, "compile"), "nn.Module.compile is unavailable")
    def test_checkpointed_transkun_block_with_changing_shapes(self):
        torch.manual_seed(11)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoderLayers = nn.ModuleList([
                    BasicBlock(16, 4, 8, enabled=["F", "T"], dropoutProb=0),
                ])

            def forward(self, x):
                return checkpoint(self.encoderLayers[0], x, use_reentrant=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()

            def forward(self, x):
                return self.backbone(x)

        eager = Model().eval()
        compiled = copy.deepcopy(eager)
        maybe_compile_transformer(compiled, enabled=True, backend="eager")
        for shape in ((2, 6, 4, 16), (1, 8, 4, 16)):
            eager.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            x = torch.randn(shape, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            expected = eager(x)
            actual = compiled(y)
            torch.testing.assert_close(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(y.grad, x.grad)
            for left, right in zip(eager.parameters(), compiled.parameters()):
                torch.testing.assert_close(right.grad, left.grad)

    @unittest.skipUnless(hasattr(nn.Module, "compile"), "nn.Module.compile is unavailable")
    def test_real_backbone_reuses_graph_for_partial_batch(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone(
                    inputSize=2, baseSize=4, posEmbedInitGamma=1, nHead=2,
                    fourierSize=8, hiddenFactor=2, hiddenFactorAttn=1,
                    expansionFactor=1, dropoutProb=0, nLayers=1,
                    enabledAttn=["F", "T"], useGradientCheckpoint=False,
                )

            def forward(self, x):
                return self.backbone(x, outputIndices=torch.tensor([21, 22]))

        eager = Model().eval()
        compiled = copy.deepcopy(eager)
        graphs = []

        def counting_backend(graph_module, _example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        def compile_with_counting_backend(module, *, backend, mode, fullgraph, dynamic):
            self.assertEqual(backend, "inductor")
            self.assertTrue(dynamic)
            module.forward = torch.compile(
                module.forward, backend=counting_backend,
                fullgraph=fullgraph, dynamic=dynamic,
            )

        with patch.object(nn.Module, "compile", compile_with_counting_backend):
            maybe_compile_transformer(compiled, enabled=True)

        with torch.inference_mode():
            for batch_size in (4, 1, 2, 4):
                x = torch.randn(batch_size, 32, 8, 2)
                torch.testing.assert_close(compiled(x), eager(x))
                self.assertEqual(len(graphs), 1)

        # The duplicated row does not contribute to the retained row's
        # output or gradient because attention stays within each batch item.
        eager.train()
        compiled.train()
        x = torch.randn(1, 32, 8, 2, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected = eager(x)
        actual = compiled(y)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(y.grad, x.grad)
        for left, right in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(right.grad, left.grad)


if __name__ == "__main__":
    unittest.main()
