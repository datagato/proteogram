"""Hierarchy-aware metric learning for Proteogram embeddings.

Cross-entropy classification (train_multiple_models_randomized_eval.py's
default --loss ce/focal path) optimizes a proxy: per-level classification
accuracy, not the retrieval metric (Precision@K) the embeddings are actually
searched with downstream (see measure_similarity_v2.py / Img2Vec). It also
treats every wrong prediction at a level equally, which doesn't match how
SCOPe search errors matter in practice -- confusing two folds within the same
class is a smaller mistake than confusing two different classes.

HierarchicalTripletLoss instead trains the embedding directly for cosine
retrieval, using SCOPe hierarchy distance (class/fold/superfamily/family) to
scale how far apart negatives are pushed: same-family pairs are positives,
and negatives that share more of the hierarchy with the anchor are given a
smaller margin than negatives that share none of it.
"""
import random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


# Coarse -> fine. HierarchicalTripletLoss.hierarchy_distance assumes labels
# are nested in this order (same family implies same superfamily/fold/class),
# which holds for SCOPe since all four columns come from the same annotation
# row -- callers building hierarchy label tensors must preserve this order.
SCOPE_LEVELS = ('class', 'fold', 'superfamily', 'family')


class HierarchicalTripletLoss(nn.Module):
    """Batch-hard triplet loss with margins scaled by SCOPe hierarchy distance.

    Positives are same-family examples (hierarchy distance 0). For each
    anchor, the hardest positive (farthest same-family example) and hardest
    negative (closest example after subtracting its distance-scaled margin)
    are mined within the batch -- so batches need enough same-family repeats
    to guarantee a positive per anchor (see HierarchicalPKSampler).

    Args:
        margins: 5 floats, margin[d] for hierarchy distance d = 0..4 (0 = same
            family ... 4 = different class). margins[0] is unused (distance-0
            pairs are positives, never negatives) but kept for index
            alignment with distance.
        base_margin: standard triplet-loss margin applied on top of the
            distance-scaled negative margin.
    """

    def __init__(self, margins=(0.0, 0.2, 0.4, 0.6, 0.8), base_margin=0.2):
        super().__init__()
        if len(margins) != len(SCOPE_LEVELS) + 1:
            raise ValueError(
                f'margins must have {len(SCOPE_LEVELS) + 1} entries (one per '
                f'hierarchy distance 0..{len(SCOPE_LEVELS)}), got {len(margins)}')
        self.register_buffer('margins', torch.tensor(margins, dtype=torch.float32))
        self.base_margin = base_margin

    @staticmethod
    def hierarchy_distance(hierarchy_labels):
        """hierarchy_labels: (B, 4) int tensor, columns ordered coarse->fine
        per SCOPE_LEVELS. Returns (B, B) int tensor: 0 = same family, 1 = same
        superfamily (different family), 2 = same fold, 3 = same class, 4 =
        different class -- the deepest level NOT shared, counted from the fine
        end.
        """
        eq = hierarchy_labels.unsqueeze(0) == hierarchy_labels.unsqueeze(1)  # (B, B, 4)
        # cumprod over the coarse->fine axis: position i is 1 only if levels
        # 0..i all match, so summing gives how many coarse-to-fine levels
        # match contiguously (nesting guarantees there are no gaps).
        depth = eq.cumprod(dim=-1).sum(dim=-1)
        return len(SCOPE_LEVELS) - depth

    def forward(self, embeddings, hierarchy_labels):
        """embeddings: (B, D) raw model output (need not be pre-normalized).
        hierarchy_labels: (B, 4) int64, columns per SCOPE_LEVELS.

        Returns (loss, stats) -- stats is a dict of diagnostics for logging
        (fraction of anchors with an in-batch positive, mean hardest
        positive/negative distance).
        """
        emb = F.normalize(embeddings, dim=1)
        cos_sim = emb @ emb.T
        dist = 1.0 - cos_sim  # cosine distance, (B, B), range [0, 2]

        distance = self.hierarchy_distance(hierarchy_labels)
        eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)

        pos_mask = (distance == 0) & ~eye
        neg_mask = distance > 0

        valid_anchor = pos_mask.any(dim=1)
        if not valid_anchor.any():
            # No anchor in this batch has a same-family positive -- degenerate
            # batch. Shouldn't happen with HierarchicalPKSampler (k >= 2), but
            # fail soft with a zero loss that stays connected to the graph.
            return embeddings.sum() * 0.0, {'valid_anchor_frac': 0.0}

        pos_fill = dist.masked_fill(~pos_mask, torch.finfo(dist.dtype).min)
        hardest_pos, _ = pos_fill.max(dim=1)

        margins = self.margins[distance]  # (B, B)
        adj_neg = dist - margins
        neg_fill = adj_neg.masked_fill(~neg_mask, torch.finfo(dist.dtype).max)
        hardest_neg, _ = neg_fill.min(dim=1)

        per_anchor_loss = F.relu(hardest_pos - hardest_neg + self.base_margin)[valid_anchor]

        stats = {
            'valid_anchor_frac': valid_anchor.float().mean().item(),
            'mean_hardest_pos_dist': hardest_pos[valid_anchor].mean().item(),
            'mean_hardest_neg_dist': hardest_neg[valid_anchor].mean().item(),
        }
        return per_anchor_loss.mean(), stats


class HierarchicalPKSampler(Sampler):
    """Yields batches of P groups x K samples/group (e.g. P SCOPe families,
    K proteins each), so every anchor in a batch has at least one same-group
    positive for HierarchicalTripletLoss to mine.

    Groups smaller than K are sampled with replacement to fill K -- SCOPe
    family classes are frequently that sparse (single digits of examples).

    Without `stratum_labels`, the P families are drawn uniformly at random
    from the whole family pool. When there are thousands of families and most
    coarser levels branch shallowly (e.g. SCOPe folds that contain only one or
    two superfamilies), a random P families almost never collide on a shared
    fold or superfamily -- so HierarchicalTripletLoss's mid-tier margins
    (same-fold-different-superfamily, same-superfamily-different-family)
    rarely see a negative to actually mine, starving those tiers of gradient
    even though family-level and class-level training signal look fine.

    `stratum_labels` fixes this by picking a handful of strata (e.g. fold ids)
    first and drawing several families from within each, so a batch reliably
    contains same-stratum-different-family negatives. Strata that run out of
    distinct families to contribute (common -- many SCOPe folds have very few
    families) are topped up with uniform random families from the rest of the
    pool, so this degrades gracefully back toward the unstratified behavior
    rather than failing.

    Args:
        group_labels: sequence of int group ids (e.g. family label per
            sample), aligned to the underlying dataset's local indices.
        p: number of distinct groups per batch.
        k: samples per group per batch (>= 2, so an anchor has a positive).
        stratum_labels: optional sequence of int ids (e.g. fold label per
            sample), same length and index-alignment as group_labels. When
            given, batches are composed by stratum first (see above) instead
            of uniform random families. Default: None (uniform random).
        families_per_stratum: target number of distinct families to draw from
            each selected stratum when stratum_labels is given. Ignored
            otherwise. Default: 4.
        batches_per_epoch: batches to yield per epoch. Defaults to
            len(group_labels) // (p * k).
    """

    def __init__(self, group_labels, p, k, stratum_labels=None, families_per_stratum=4,
                 batches_per_epoch=None, seed=0):
        if k < 2:
            raise ValueError(f'HierarchicalPKSampler: k must be >= 2 (got {k}) so every '
                              f'anchor has an in-batch positive.')
        self.p = p
        self.k = k
        self.groups = defaultdict(list)
        for idx, label in enumerate(group_labels):
            self.groups[label].append(idx)
        self.group_ids = list(self.groups.keys())
        if len(self.group_ids) < p:
            raise ValueError(f'HierarchicalPKSampler: only {len(self.group_ids)} groups '
                              f'available, need at least p={p}.')
        self.batches_per_epoch = batches_per_epoch or max(1, len(group_labels) // (p * k))
        self.rng = random.Random(seed)

        # Cumulative diagnostics, updated per batch in _sample_families:
        # - families_from_topup: drawn via the uniform-random fallback because
        #   the sampler ran out of *distinct strata* to visit (rare -- only
        #   happens when there are too few strata overall relative to p).
        # - families_paired: drawn families that share their stratum with at
        #   least one OTHER drawn family in the SAME batch -- i.e. actually
        #   have a same-stratum-different-family negative available for
        #   HierarchicalTripletLoss to mine. This is the one that matters in
        #   practice: a stratify level that branches shallowly (e.g. most
        #   superfamilies contain only 1 family) can visit plenty of distinct
        #   strata -- topup_frac stays 0 -- while still leaving most families
        #   with no same-stratum partner in the batch at all, since each
        #   shallow stratum only ever contributes one family on its own.
        self.families_drawn = 0
        self.families_from_topup = 0
        self.families_paired = 0

        self.families_per_stratum = families_per_stratum
        self.stratum_to_families = None
        self.family_to_stratum = None
        if stratum_labels is not None:
            if len(stratum_labels) != len(group_labels):
                raise ValueError('HierarchicalPKSampler: stratum_labels must be the same '
                                  'length as group_labels.')
            # A family belongs to exactly one stratum (e.g. one fold), so the
            # first (family, stratum) pair observed for each family fixes it.
            family_to_stratum = {}
            for fam, strat in zip(group_labels, stratum_labels):
                family_to_stratum.setdefault(fam, strat)
            self.family_to_stratum = family_to_stratum
            stratum_to_families = defaultdict(set)
            for fam, strat in family_to_stratum.items():
                stratum_to_families[strat].add(fam)
            self.stratum_to_families = {s: list(fams) for s, fams in stratum_to_families.items()}

    def _sample_families(self):
        """Return p distinct family ids for one batch."""
        if self.stratum_to_families is None:
            return self.rng.sample(self.group_ids, self.p)

        chosen, chosen_set = [], set()
        strata = list(self.stratum_to_families.keys())
        self.rng.shuffle(strata)
        for stratum in strata:
            if len(chosen) >= self.p:
                break
            candidates = [f for f in self.stratum_to_families[stratum] if f not in chosen_set]
            if not candidates:
                continue
            take = min(self.families_per_stratum, len(candidates), self.p - len(chosen))
            picked = self.rng.sample(candidates, take)
            chosen.extend(picked)
            chosen_set.update(picked)

        n_from_strata = len(chosen)
        if len(chosen) < self.p:
            # Ran out of strata to draw fresh families from (shallow tree, or
            # too few strata overall) -- top up with uniform random families.
            remaining = [f for f in self.group_ids if f not in chosen_set]
            chosen.extend(self.rng.sample(remaining, self.p - len(chosen)))

        self.families_drawn += self.p
        self.families_from_topup += self.p - n_from_strata
        strata_in_batch = Counter(self.family_to_stratum[f] for f in chosen)
        self.families_paired += sum(1 for f in chosen if strata_in_batch[self.family_to_stratum[f]] > 1)
        return chosen

    @property
    def paired_frac(self):
        """Fraction of all families drawn so far that shared their stratum
        with at least one other family in the same batch -- i.e. actually had
        a same-stratum-different-family negative available to mine. NaN if
        stratification is off or nothing has been drawn yet.
        """
        if self.stratum_to_families is None or self.families_drawn == 0:
            return float('nan')
        return self.families_paired / self.families_drawn

    @property
    def topup_frac(self):
        """Fraction of all families drawn so far that came from the uniform-
        random top-up fallback rather than genuine stratified sampling. NaN
        if stratification is off (stratum_to_families is None) or nothing has
        been drawn yet -- there's no fallback concept to measure in that case.
        """
        if self.stratum_to_families is None or self.families_drawn == 0:
            return float('nan')
        return self.families_from_topup / self.families_drawn

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            for g in self._sample_families():
                members = self.groups[g]
                if len(members) >= self.k:
                    batch.extend(self.rng.sample(members, self.k))
                else:
                    batch.extend(self.rng.choices(members, k=self.k))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch
