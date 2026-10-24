"""Document-level metrics. Callers must align token prefixes and loss masks."""
import numpy as np
import torch


def distribution_metrics(teacher_logits, student_logits, targets, mask):
    if teacher_logits.shape != student_logits.shape or targets.shape != mask.shape or teacher_logits.shape[:-1] != targets.shape:
        raise ValueError('Aligned logits, targets and mask required')
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError('Nonempty boolean loss mask required')
    t = teacher_logits[mask].float().log_softmax(-1)
    s = student_logits[mask].float().log_softmax(-1)
    if not torch.isfinite(t).all() or not torch.isfinite(s).all():
        raise ValueError('Nonfinite distributions')
    y = targets[mask].unsqueeze(-1)
    return {'kl': (t.exp() * (t-s)).sum(-1).mean().item(),
            'student_nll': -s.gather(-1, y).mean().item(),
            'teacher_nll': -t.gather(-1, y).mean().item(),
            'top1_agreement': (t.argmax(-1) == s.argmax(-1)).float().mean().item(),
            'tokens': int(mask.sum())}


def repetition(tokens):
    tokens = list(tokens)
    triples = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    longest = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    loop = any(tokens[i:i+n] == tokens[i+n:i+2*n] == tokens[i+2*n:i+3*n]
               for n in range(1, 9) for i in range(len(tokens)-3*n+1))
    return {'repeated_trigram_fraction': 1-len(set(triples))/len(triples) if triples else 0.,
            'max_same_token_run': longest, 'loop': loop, 'length': len(tokens)}


def paired_bootstrap(reference, candidate, *, seed=20261017, resamples=10000):
    """Maps of document IDs to scalar scores; candidate minus reference."""
    if not reference or reference.keys() != candidate.keys() or resamples < 1:
        raise ValueError('Matching nonempty document IDs and positive resamples required')
    delta = np.array([candidate[k]-reference[k] for k in sorted(reference)], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError('Finite scores required')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(resamples)])
    return {'mean_difference': float(delta.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'documents': len(delta), 'resamples': resamples, 'seed': seed}


def mcnemar(reference, candidate):
    from scipy.stats import binomtest
    if not reference or reference.keys() != candidate.keys():
        raise ValueError('Matching nonempty document IDs required')
    if any(v not in (0, 1) for v in list(reference.values())+list(candidate.values())):
        raise ValueError('Binary results required')
    lost = sum(reference[k] == 1 and candidate[k] == 0 for k in reference)
    gained = sum(reference[k] == 0 and candidate[k] == 1 for k in reference)
    return {'lost': lost, 'gained': gained,
            'pvalue': float(binomtest(lost, lost+gained, .5).pvalue) if lost+gained else 1.}


def first_divergence(teacher, student, max_new):
    """Zero-based token position; retain EOS tokens in both inputs.

    A matching short prefix is censored, not evidence of identical full answers.
    """
    if max_new < 1:
        raise ValueError('Positive generation horizon required')
    horizon = min(len(teacher), len(student), max_new)
    for i in range(horizon):
        if teacher[i] != student[i]:
            return {'position':i, 'event_observed':True, 'compared_tokens':i+1}
    return {'position':horizon, 'event_observed':False, 'compared_tokens':horizon}


def holm(pvalues):
    """Adjusted p-values for a predeclared comparison family, keyed by name."""
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError('Finite probabilities in [0,1] required')
    ordered = sorted(pvalues, key=lambda key:pvalues[key])
    corrected, running = {}, 0.
    for i, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered)-i)*pvalues[key]))
        corrected[key] = running
    return corrected
