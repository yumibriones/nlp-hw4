import argparse
import os
import torch

from train_t5 import test_inference, get_output_run_tag
from t5_utils import load_model_from_checkpoint, initialize_model
from load_data import load_t5_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Generate test predictions from checkpoint')
    
    # Model/checkpoint
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--finetune', action='store_true', default=True, help='Whether model was finetuned')
    parser.add_argument('--model_type', type=str, default='pretrained', choices=['pretrained', 'scratch'])
    
    # Generation
    parser.add_argument('--num_beams', type=int, default=1, help='Beam search width')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens to generate')
    
    # Data
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_source_length', type=int, default=96)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--add_task_prefix', action='store_true', default=True)
    parser.add_argument('--task_prefix', type=str, default='translate English to SQL:')
    parser.add_argument('--normalize_whitespace', action='store_true', default=True)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    model, epoch = load_model_from_checkpoint(args, best=True, return_epoch=True)
    print(f"Loaded best model from checkpoint at epoch {epoch}")
    model = model.to(DEVICE)
    model.eval()
    
    print("Loading test data...")
    _, _, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_prefix=args.add_task_prefix,
        task_prefix=args.task_prefix,
        normalize_whitespace=args.normalize_whitespace,
    )
    
    # Generate output paths
    model_type = 'ft' if args.finetune else 'scr'
    run_tag = get_output_run_tag(model_type, args.experiment_name)
    test_sql_path = f'results/{run_tag}_test.sql'
    test_record_path = f'records/{run_tag}_test.pkl'
    
    print(f"Generating test predictions with num_beams={args.num_beams}...")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)
    
    print(f"✓ Test SQL: {test_sql_path}")
    print(f"✓ Test records: {test_record_path}")


if __name__ == "__main__":
    main()
