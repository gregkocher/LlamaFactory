"""Validate the published Ouro graph before any organism training (GPU only)."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ByteDance/Ouro-1.4B')
    parser.add_argument('--revision', default='574fa66cb8bf5abdc979642d01cf2b79b16bfab1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).cuda()
    model.config.use_cache = False
    assert model.model.total_ut_steps == model.config.total_ut_steps == 4
    inputs = tokenizer('The sum of 12 and 15 is 27. A careful explanation checks each step.', return_tensors='pt').to('cuda')
    labels = inputs['input_ids'].clone()
    labels[:, :3] = -100
    report = {'model':args.model, 'revision':args.revision, 'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__}
    captures = []
    gates = []
    counts = [0] * len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        def count(module, args, out, i=i):
            counts[i] += 1
        handles.append(layer.register_forward_hook(count))
    handles.append(model.model.norm.register_forward_hook(lambda m,a,o: captures.append(o)))
    handles.append(model.model.early_exit_gate.register_forward_hook(lambda m,a,o: gates.append(o)))
    model.eval()
    with torch.no_grad():
        native = model(**inputs, labels=labels)
        assert counts == [4] * len(counts), counts
        assert len(captures) == len(gates) == 4
        remaining = torch.ones_like(gates[0].squeeze(-1))
        logits = None
        probs = []
        for i,(state,gate) in enumerate(zip(captures,gates)):
            prob = remaining * gate.squeeze(-1).sigmoid() if i < 3 else remaining
            remaining = remaining * (1 - gate.squeeze(-1).sigmoid())
            probs.append(prob.float().mean().item())
            term = model.lm_head(state) * prob.unsqueeze(-1)
            logits = term if logits is None else logits + term
        loss = F.cross_entropy(logits[:,:-1].float().reshape(-1, logits.shape[-1]), labels[:,1:].reshape(-1))
        torch.testing.assert_close(native.logits, logits, atol=0, rtol=0)
        torch.testing.assert_close(native.loss.float(), loss, atol=1e-5, rtol=1e-5)
        verified_counts = list(counts)
        final = model(**inputs, exit_at_step=3).logits.detach()
    report.update(native_loss=float(native.loss), manual_loss=float(loss), native_exit_probabilities=probs, loop_calls=verified_counts)
    for handle in handles:
        handle.remove()
    captures.clear(); gates.clear()
    del native, logits
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','v_proj'], lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM))
    trainable = [name for name,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in name for name in trainable)
    model.eval()
    with torch.no_grad():
        zero_adapter = model(**inputs, exit_at_step=3).logits
        torch.testing.assert_close(final, zero_adapter, rtol=0, atol=0)
    report['zero_adapter_max_error'] = float((final-zero_adapter).abs().max())
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()
    states = []
    def retain(module, args, out):
        if out.requires_grad:
            out.retain_grad()
        states.append(out)
    handle = model.get_base_model().model.norm.register_forward_hook(retain)
    start = time.perf_counter()
    result = model(**inputs, labels=labels)
    result.loss.backward()
    torch.cuda.synchronize()
    report['backward_seconds'] = time.perf_counter()-start
    report['loop_gradient_norms'] = [float(s.grad.float().norm()) if s.grad is not None else None for s in states]
    assert len(states) == 4 and all(x is not None and x > 0 for x in report['loop_gradient_norms'])
    adapter_grads = {n:p.grad.detach().float().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert adapter_grads and all(torch.isfinite(g).all() for g in adapter_grads.values())
    handle.remove(); states.clear()
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    checkpointed = model(**inputs, labels=labels)
    checkpointed.loss.backward()
    torch.testing.assert_close(result.loss, checkpointed.loss, atol=1e-5, rtol=1e-5)
    max_diff = 0.0
    for name,p in model.named_parameters():
        if name in adapter_grads:
            assert p.grad is not None
            actual = p.grad.detach().float().cpu()
            torch.testing.assert_close(actual,adapter_grads[name],atol=1e-6,rtol=1e-3)
            max_diff = max(max_diff,float((actual-adapter_grads[name]).abs().max()))
    report['checkpointing_gradient_max_error'] = max_diff
    model.gradient_checkpointing_disable(); model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad():
        without_cache = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False, exit_at_step=3)
        with_cache = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True, exit_at_step=3)
    assert torch.equal(without_cache,with_cache), 'Cached and cache-free greedy generations differ'
    report['cached_generation_matches'] = True
    report['generation'] = tokenizer.decode(with_cache[0],skip_special_tokens=True)
    report['peak_memory_gb'] = torch.cuda.max_memory_allocated()/1e9
    report['status'] = 'passed'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
