import torch

def configuration(**kwargs):
    c=dict(batch_size=2,seq_len=9,puzzle_emb_ndim=32,num_puzzle_identifiers=5,vocab_size=12,
        H_cycles=2,L_cycles=2,H_layers=0,L_layers=2,hidden_size=32,expansion=4,num_heads=4,
        pos_encodings='rope',halt_max_steps=2,halt_exploration_prob=.1,forward_dtype='float32',
        mlp_t=False,puzzle_emb_len=2,no_ACT_continue=True,operator='trm_attention',interface='full',
        control='shared',mixer_rank=8,ff_copies=3)
    c.update(kwargs);return c

def batch():
    return dict(inputs=torch.randint(2,12,(2,9)),labels=torch.randint(2,12,(2,9)),puzzle_identifiers=torch.tensor([1,2]))
