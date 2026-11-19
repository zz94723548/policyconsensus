"""
Usage:
python scripts/eval_policy.py sim ... (see modular_policy/workspace/eval_policy_sim.py)
python scripts/eval_policy.py real ... (see modular_policy/workspace/eval_policy_real.py)
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import warnings

@click.group()
def eval_policy():
    pass


@eval_policy.command('summarize-fixed')
@click.option('-i', '--input-dir', required=True, type=click.Path(exists=True, file_okay=False))
def summarize_fixed(input_dir):
    """Print and save success rates of the eleven fixed-weight candidates."""
    import csv
    import json
    from pathlib import Path
    from modular_policy.common.fixed_weight_search import summarize_fixed_weights, TASKS
    result = summarize_fixed_weights(input_dir)
    root = Path(input_dir)
    (root / 'fixed_weight_summary.json').write_text(json.dumps(result, indent=2))
    fields = ['rgb_percent', 'pcd_percent', 'successes', 'episodes', 'success_rate', *TASKS]
    with (root / 'fixed_weight_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result['rows'])
    click.echo('RGB%  PCD%  Successes  Overall    Box   Toilet  Drawer  Umbrella')
    for row in result['rows']:
        rates = '  '.join(f'{row[t]:6.1%}' for t in TASKS)
        click.echo(f"{row['rgb_percent']:4}  {row['pcd_percent']:4}  "
                   f"{row['successes']:3}/{row['episodes']:<3}   {row['success_rate']:6.1%}  {rates}")
    if result['missing_rgb_percent']:
        click.echo(f"Incomplete search; missing RGB%: {result['missing_rgb_percent']}")
    if result['best_success_rate'] is not None:
        click.echo(f"Best completed candidates RGB%={result['best_rgb_percent']}; "
                   f"success={result['best_success_rate']:.2%}")


@eval_policy.command('compare-matched')
@click.option('-i','--input-dir',default='output/eval_matched5_seed42',show_default=True,
              type=click.Path(exists=True,file_okay=False))
def compare_matched(input_dir):
    """Summarize existing expert/fusion/router success rates and offline risks."""
    import csv
    import json
    from pathlib import Path
    from modular_policy.common.matched_eval_comparison import compare_matched as summarize
    from modular_policy.common.fixed_weight_search import TASKS
    result=summarize(input_dir)
    root=Path(input_dir)
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','successes','episodes','success_rate',
            'offline_epsilon_mse',*TASKS,'simulation_source'])
        writer.writeheader()
        writer.writerows(result['rows'])
    click.echo('Method        Successes   Success rate   Offline epsilon MSE')
    for row in result['rows']:
        risk='pending' if row['offline_epsilon_mse'] is None else f"{row['offline_epsilon_mse']:.9f}"
        click.echo(f"{row['method']:<13} {row['successes']:>2}/{row['episodes']:<3}      {row['success_rate']:7.2%}       {risk}")
    if result['missing_simulation_methods']:
        click.echo(f"Missing simulation methods: {result['missing_simulation_methods']}")

try:
    from modular_policy.workspace.eval_policy_sim import eval_policy_sim
    eval_policy.add_command(eval_policy_sim, name='sim')
except:
    warnings.warn("WARNING: policy evaluation for simulation not available")

# try:
#     from modular_policy.workspace.eval_policy_real import eval_policy_real
#     eval_policy.add_command(eval_policy_real, name='real')
# except:
#     warnings.warn("WARNING: policy evaluation for real robot not available")



if __name__ == '__main__':
    eval_policy()
