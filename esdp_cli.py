#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esdp_cli.py - Command-Line Interface for ESDP (Early Stop Decision Polishing)

This CLI wrapper makes esdp_decide usable in Nextflow, Snakemake, and other
workflow managers.

Usage:
    # From JSON file
    python esdp_cli.py --input metrics.json --output decision.json
    
    # From CSV file (batch mode)
    python esdp_cli.py --input samples.csv --output decisions.csv
    
    # From stdin (Nextflow-friendly)
    echo '{"coverage": 50, "qv": 35.2, ...}' | python esdp_cli.py --output decision.json
    
    # Conservative mode
    python esdp_cli.py --input metrics.json --output decision.json --conservative
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from esdp_decide import decide, PolishingMetrics, Decision, decide_batch

# ============================================================================
# CLI Functions
# ============================================================================

def read_input(input_path: str = None) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Read input from file or stdin.
    
    Args:
        input_path: Path to input file (JSON or CSV). If None, read from stdin.
    
    Returns:
        Dictionary or list of dictionaries with metrics
    """
    if input_path is None:
        # Read from stdin
        data = sys.stdin.read()
        return json.loads(data)
    
    input_path_obj = Path(input_path)
    
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine format from extension
    if input_path_obj.suffix.lower() == '.json':
        with open(input_path_obj, 'r') as f:
            return json.load(f)
    
    elif input_path_obj.suffix.lower() == '.csv':
        df = pd.read_csv(input_path_obj)
        return df.to_dict('records')
    
    else:
        raise ValueError(f"Unsupported file format: {input_path_obj.suffix}")


def write_output(
    decisions: Decision | List[Decision],
    output_path: str = None,
    format: str = 'json'
):
    """
    Write output to file or stdout.
    
    Args:
        decisions: Single Decision or list of Decisions
        output_path: Path to output file. If None, write to stdout.
        format: Output format ('json' or 'csv')
    """
    # Convert to list if single decision
    if isinstance(decisions, Decision):
        decisions = [decisions]
    
    # Convert to serializable format
    decisions_data = [d.to_json_serializable() for d in decisions]
    
    if output_path is None:
        # Write to stdout
        if format == 'json':
            if len(decisions_data) == 1:
                print(json.dumps(decisions_data[0], indent=2))
            else:
                print(json.dumps(decisions_data, indent=2))
        else:
            df = pd.DataFrame(decisions_data)
            print(df.to_csv(index=False))
    else:
        output_path_obj = Path(output_path)
        
        if format == 'json' or output_path_obj.suffix.lower() == '.json':
            with open(output_path_obj, 'w') as f:
                if len(decisions_data) == 1:
                    json.dump(decisions_data[0], f, indent=2)
                else:
                    json.dump(decisions_data, f, indent=2)
        
        elif format == 'csv' or output_path_obj.suffix.lower() == '.csv':
            df = pd.DataFrame(decisions_data)
            df.to_csv(output_path_obj, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_path_obj.suffix}")


def process_single(
    metrics_dict: Dict[str, Any],
    model_path: str,
    feature_names_path: str,
    force_conservative: bool,
    model_version: str
) -> Decision:
    """Process a single sample."""
    metrics = PolishingMetrics(**metrics_dict)
    
    decision = decide(
        metrics=metrics,
        model_path=model_path,
        feature_names_path=feature_names_path,
        force_conservative=force_conservative,
        model_version=model_version
    )
    
    return decision


def process_batch(
    metrics_list: List[Dict[str, Any]],
    model_path: str,
    feature_names_path: str,
    force_conservative: bool,
    model_version: str
) -> List[Decision]:
    """Process multiple samples."""
    metrics_objects = [PolishingMetrics(**m) for m in metrics_list]
    
    decisions = decide_batch(
        metrics_list=metrics_objects,
        model_path=model_path,
        feature_names_path=feature_names_path,
        force_conservative=force_conservative,
        model_version=model_version
    )
    
    return decisions


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ESDP CLI - Early Stop Decision Polishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sample from JSON
  python esdp_cli.py --input sample.json --output decision.json
  
  # Batch processing from CSV
  python esdp_cli.py --input samples.csv --output decisions.csv
  
  # From stdin (Nextflow-friendly)
  echo '{"coverage": 50, "qv": 35.2, "busco_complete": 95.5, "n50": 2500000}' | \\
    python esdp_cli.py --output decision.json
  
  # Conservative mode
  python esdp_cli.py --input sample.json --output decision.json --conservative
  
  # Custom model path
  python esdp_cli.py --input sample.json --output decision.json \\
    --model-path /path/to/model.pkl
        """
    )
    
    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input file (JSON or CSV). If not provided, reads from stdin.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file (JSON or CSV). If not provided, writes to stdout.'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default='models/best_model_pipeline.pkl',
        help='Path to trained model pipeline (default: models/best_model_pipeline.pkl)'
    )
    
    parser.add_argument(
        '--feature-names-path',
        type=str,
        default='models/feature_names.txt',
        help='Path to feature names file (default: models/feature_names.txt)'
    )
    
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0.0',
        help='Model version string (default: 1.0.0)'
    )
    
    # Decision options
    parser.add_argument(
        '--conservative', '-c',
        action='store_true',
        help='Force conservative recommendations (prefer more rounds)'
    )
    
    parser.add_argument(
        '--qv-threshold',
        type=float,
        default=35.0,
        help='QV threshold for excellent quality (default: 35.0)'
    )
    
    parser.add_argument(
        '--busco-threshold',
        type=float,
        default=95.0,
        help='BUSCO threshold for excellent quality (default: 95.0)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (print to stderr)'
    )
    
    args = parser.parse_args()
    
    try:
        # Read input
        if args.verbose:
            print(f"Reading input from: {args.input or 'stdin'}", file=sys.stderr)
        
        input_data = read_input(args.input)
        
        # Determine if batch or single
        is_batch = isinstance(input_data, list)
        
        if args.verbose:
            if is_batch:
                print(f"Processing {len(input_data)} samples...", file=sys.stderr)
            else:
                print("Processing single sample...", file=sys.stderr)
        
        # Process
        if is_batch:
            decisions = process_batch(
                metrics_list=input_data,
                model_path=args.model_path,
                feature_names_path=args.feature_names_path,
                force_conservative=args.conservative,
                model_version=args.model_version
            )
        else:
            decision = process_single(
                metrics_dict=input_data,
                model_path=args.model_path,
                feature_names_path=args.feature_names_path,
                force_conservative=args.conservative,
                model_version=args.model_version
            )
            decisions = decision
        
        # Write output
        if args.verbose:
            print(f"Writing output to: {args.output or 'stdout'}", file=sys.stderr)
        
        write_output(decisions, args.output, args.format)
        
        if args.verbose:
            if is_batch:
                print(f"Successfully processed {len(input_data)} samples", file=sys.stderr)
            else:
                print("Successfully processed sample", file=sys.stderr)
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    except ValueError as e:
        print(f"ERROR: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()