"""Summarize a locally verified completed census archive without model inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.snapshot;out=args.output
    evaluation=root/'census_evaluation_v1';result=json.loads((evaluation/'COMPLETE.json').read_text())
    for name,expected in result['files_sha256'].items():
        assert hashlib.sha256((evaluation/name).read_bytes()).hexdigest()==expected,name
    frame=json.loads((root/'census_frame_v1/manifest.json').read_text());counts=frame['counts']
    frozen=json.loads((evaluation/'FRAME_FREEZE.json').read_text())
    assert hashlib.sha256((root/'runtime/scale_census.py').read_bytes()).hexdigest()==frozen['runner_sha256']
    arms=('base','target','control');families=('arc_easy','arithmetic_composition');metrics=result['metrics']
    lines=['# EOS1221 complete MCQ census','',
      'Both adapters meet the frozen V3 five-percentage-point margins for loop-4 accuracy and for loss of loop-4-minus-loop-1 benefit on both complete finite frames. This is finite-benchmark retention, not a confidence certificate for broader reasoning ability. Acquisition, free-response math, and general coherence remain separate endpoints.','',
      '| Family | Model | Loop 1 | Loop 2 | Loop 3 | Loop 4 | Loop 4 − loop 1 |','|---|---|---:|---:|---:|---:|---:|']
    for family in families:
        for arm in arms:
            m=metrics[arm][family]
            values=' | '.join(f'{100*x:.2f}%' for x in m['accuracy_by_loop'])
            lines.append(f'| {family} (n={m["n"]}) | {arm} | {values} | {100*m["loop4_minus_loop1"]:.2f} pp |')
    lines+=['','| Family | Adapter | Loop-4 change vs base | Change in loop-4-minus-loop-1 benefit | Both margins met |','|---|---|---:|---:|---|']
    for family in families:
        for arm in ('target','control'):
            m=result['comparisons'][arm][family]
            lines.append(f'| {family} | {arm} | {100*m["loop4_delta"]:+.3f} pp | {100*m["loop4_minus_loop1_change"]:+.3f} pp | {m["loop4_retention_pass"] and m["recurrence_gate_pass"]} |')
    lines+=['',
      'The larger ARC recurrence benefit reflects substantially weaker first-loop readouts in both adapters while final-loop accuracy changes little. It does not demonstrate improved reasoning or a causal benefit of executing more loops. All readouts come from the unchanged native four-loop forward pass.','',
      f'The ARC frame has {counts["remaining_arc"]} source units and {counts["remaining_arc_unique_normalized_questions"]} distinct normalized questions. It excludes all 300 original ARC identities plus {counts["arc_additional_rows_excluded_by_normalized_question"]} additional matching-question rows. The composition frame contains all 1,240 remaining ordered triples, representing {counts["remaining_composition_unique_unordered_addend_classes"]} unordered-addend classes and {counts["remaining_composition_unique_answers"]} numerical answers; {counts["remaining_composition_classes_shared_with_original"]} classes overlap semantically with original evaluation cases. Distinct units therefore do not imply independent skills. Original evaluation rows remain unchanged and are not pooled into this census.','',
      'The candidate and protocol were frozen before frame construction, and full prompts/options were hashed before inference. Base, target, and control used identical complete frames, batch size 16, SDPA, stable length ordering, and unchanged first-index argmax tie handling. Every expected ID, original case field, score finiteness, four-loop prediction, and summary count was verified. No output-dependent exclusions or reruns occurred.','',
      '| Model | MCQ scoring seconds, excluding loading | Peak allocated VRAM (GB) |','|---|---:|---:|']
    for arm in arms:
        summary=json.loads((evaluation/arm/'summary.json').read_text())
        lines.append(f'| {arm} | {summary["seconds"]:.2f} | {summary["peak_vram_gb"]:.2f} |')
    lines+=['',f'Candidate freeze SHA256: `{result["candidate_freeze_sha256"]}`.',f'Frame freeze SHA256: `{result["frame_freeze_sha256"]}`.',f'Cases SHA256: `{frame["cases_sha256"]}`.',f'Pinned ARC revision: `{frame["source_revisions"]["allenai/ai2_arc"]}`.','',
      'Exact rows, all loop-choice scores, immutable checkpoint identities, frame exclusions, source hashes, launch commands, and logs are preserved in the accompanying verified snapshot. One seed, a narrow composition generator, possible source/pretraining contamination, and numerical batching sensitivity limit broader interpretation.','']
    out.mkdir(parents=True,exist_ok=False)
    (out/'CENSUS_RESULTS.md').write_text('\n'.join(lines))
    (out/'CENSUS_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        (out/'PLOT_UNAVAILABLE.txt').write_text('Matplotlib unavailable; no raster fallback attempted.\n')
    else:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
        for axis,family in zip(axes,families):
            for arm in arms:axis.plot(range(1,5),[100*x for x in metrics[arm][family]['accuracy_by_loop']],marker='o',label=arm)
            axis.set(title=f'{family}\nn={metrics["base"][family]["n"]}',xlabel='Native recurrent readout',xticks=range(1,5),ylim=(0,100));axis.grid(alpha=.2)
        axes[0].set_ylabel('Accuracy (%)');axes[1].legend();fig.tight_layout();fig.savefig(out/'accuracy_by_loop.pdf');plt.close(fig)
    print(out)

if __name__=='__main__':main()
