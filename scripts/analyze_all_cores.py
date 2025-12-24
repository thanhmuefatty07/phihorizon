import torch
import os

def analyze_checkpoint(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(filepath)} ({size_mb:.2f} MB)")
    print(f"{'='*60}")
    
    try:
        ck = torch.load(filepath, map_location='cpu', weights_only=False)
        
        print(f"Keys: {list(ck.keys())}")
        print(f"Tier: {ck.get('tier', 'N/A')}")
        
        if 'config' in ck:
            print("\n=== CONFIG ===")
            config = ck['config']
            for k, v in config.items():
                if k != 'feature_cols':
                    print(f"  {k}: {v}")
        
        if 'training' in ck:
            print("\n=== TRAINING ===")
            training = ck['training']
            for k, v in training.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        # Check for validation accuracy
        for key in ['val_acc', 'best_val_acc', 'accuracy', 'val_accuracy']:
            if key in ck:
                print(f"\n{key}: {ck[key]:.4f}")
                
        if 'model_state_dict' in ck:
            state = ck['model_state_dict']
            n_params = sum(t.numel() for t in state.values())
            print(f"\nModel: {len(state)} tensors, {n_params:,} params")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    base = 'models/core/'
    files = [
        'meta_decision_splus_best.pt',
        'quant_transformer_splus_best.pt',
        'core1_splus_v2_best.pt'
    ]
    for f in files:
        analyze_checkpoint(base + f)
