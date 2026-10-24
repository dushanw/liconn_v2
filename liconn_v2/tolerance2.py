"""The owner's 2% rule (docs/tolerance2_results.md), applied to many-seed verdicts.

Owner decision (2026-10-23), after fp16_head32 failed the original rule on fresh seeds
with 5 new merges: at most 6 of 300 new merges and at most 6 of 300 new early stops
(2%), instead of 3 of 300. Everything else in liconn_v2/manyseed.verdict is unchanged
(the 10-point flip definition, the 1-point mean limits, completeness, pinned inputs, all
300 seeds). The rule was set after seeing that result, so a pass under it is not a
pre-registered confirmation.
"""

MAX_NEW_MERGES = 6
MAX_NEW_EARLY_STOPS = 6
DECIDED = '2026-10-23, after the fresh-seed result (docs/fresh_results.md)'


def apply(verdict: dict) -> dict:
  """A manyseed.verdict result re-judged under the 2% rule."""
  rules = dict(verdict['rules'])
  rules['new_merges'] = verdict['counts']['new_merge'] <= MAX_NEW_MERGES
  rules['new_early_stops'] = verdict['counts']['new_early_stop'] <= MAX_NEW_EARLY_STOPS
  return {'rule': '2%: at most 6 of 300 new merges and 6 of 300 new early stops', 'decided': DECIDED,
          'counts': verdict['counts'], 'rules': rules, 'passes': all(rules.values()),
          'mean_recall_change': verdict['mean_recall_change'],
          'mean_precision_change': verdict['mean_precision_change']}
