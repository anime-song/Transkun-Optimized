"""Equivalence checks against the original Transkun recurrence."""

import unittest

import torch
import torch.nn.functional as F

from transkun.CRF.NeuralSemiCRFInterval import NeuralSemiCRFInterval, viterbiBackward


def reference_decode(score, noise, starts=None):
    time, _, tracks = score.shape
    q = torch.zeros(time, tracks, device=score.device)
    pointers = []
    q[-1] = score[-1, -1].clamp_min(0)
    score_transposed = score.transpose(0, 1).contiguous()
    for offset in range(1, time):
        begin = time - offset - 1
        candidates = torch.cat((
            q[begin + 1:begin + 2] + noise[begin],
            q[begin + 1:] + score_transposed[begin, begin + 1:],
        ), dim=0)
        value, choice = candidates.max(dim=0)
        pointers.append(choice - 1)
        q[begin] = value + score[begin, begin].clamp_min(0)
    pointers = torch.stack(pointers).cpu().tolist()
    diag = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu().tolist()
    result = []
    for track in range(tracks):
        position = 0 if starts is None else starts[track]
        notes = []
        while position < time - 1:
            selection = pointers[time - position - 2][track]
            if diag[track][position]:
                notes.append((position, position))
            if selection < 0:
                position += 1
            else:
                end = position + selection + 1
                notes.append((position, end))
                position = end
        if diag[track][-1]:
            notes.append((time - 1, time - 1))
        result.append(notes)
    return result


def reference_log_z(score, noise):
    values = [F.softplus(score[0, 0])]
    for end in range(1, score.shape[0]):
        intervals = torch.stack(values) + score[end, :end]
        combined = torch.logaddexp(
            values[-1] + noise[end - 1], torch.logsumexp(intervals, dim=0)
        )
        values.append(combined + F.softplus(score[end, end]))
    return values[-1]


class SemiCRFBackendTests(unittest.TestCase):
    def test_decode_matches_original_with_ties_and_forced_starts(self):
        torch.manual_seed(12)
        for time in (2, 3, 9, 24):
            for score in (torch.randn(time, time, 5), torch.zeros(time, time, 5)):
                noise = torch.randn(time - 1, 5) if score.any() else torch.zeros(time - 1, 5)
                score[:, :, -1] = -100  # A track with no possible notes.
                starts = [0, 1, time - 1, 0, 0]
                expected = reference_decode(score, noise, starts)
                self.assertEqual(viterbiBackward(score, noise, starts, backend="torch"), expected)
                self.assertEqual(NeuralSemiCRFInterval(score, noise).decode(starts), expected)

    def test_log_partition_and_gradient_match_reference(self):
        torch.manual_seed(31)
        for time in (1, 2, 7):
            a = torch.randn(time, time, 3, requires_grad=True)
            b = torch.randn(time - 1, 3, requires_grad=True)
            expected = reference_log_z(a, b)
            expected_grads = torch.autograd.grad(expected.sum(), (a, b), allow_unused=True)
            x = a.detach().clone().requires_grad_()
            y = b.detach().clone().requires_grad_()
            actual = NeuralSemiCRFInterval(x, y).computeLogZ(backend="torch")
            actual_grads = torch.autograd.grad(actual.sum(), (x, y), allow_unused=True)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                if expected_grad is None:
                    self.assertTrue(actual_grad is None or actual_grad.numel() == 0)
                    continue
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_triton_matches_torch(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("Triton is unavailable")
        torch.manual_seed(42)
        for time in (2, 11, 37):
            scores = torch.randn(time, time, 8, device="cuda")
            noise = torch.randn(time - 1, 8, device="cuda")
            scores[:, :, -1] = -100
            starts = [0] * 8
            starts[0] = 1
            self.assertEqual(
                viterbiBackward(scores, noise, starts, backend="triton"),
                viterbiBackward(scores, noise, starts, backend="torch"),
            )
            a = scores.detach().requires_grad_()
            b = noise.detach().requires_grad_()
            c = scores.detach().requires_grad_()
            d = noise.detach().requires_grad_()
            torch_loss = NeuralSemiCRFInterval(a, b).computeLogZ(backend="torch")
            triton_loss = NeuralSemiCRFInterval(c, d).computeLogZ(backend="triton")
            torch.testing.assert_close(triton_loss, torch_loss, atol=2e-4, rtol=2e-5)
            for fast, original in zip(
                torch.autograd.grad(triton_loss.sum(), (c, d)),
                torch.autograd.grad(torch_loss.sum(), (a, b)),
            ):
                torch.testing.assert_close(fast, original, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
