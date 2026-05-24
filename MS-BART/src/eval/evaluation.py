from transformers import BartForConditionalGeneration, LogitsProcessorList
from datasets import load_dataset
from transformers import GenerationConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import numpy as np
from rich.table import Table
from rich.console import Console
import os
import pandas
import json
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import sys
sys.path.append(".")
from src.eval.metric import TopkMoleculeEvaluator
from src.eval.utils import save_arr, get_morgan_4096, smiles_to_formula, compare_formulas
import re
from collections import defaultdict

from apetokenizer.ape_tokenizer import APETokenizer

if __name__ == "__main__":

    #Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate BART model for molecular generation")
    parser.add_argument("--model_path", type=str, default="results/finetune-original", help="Path to the trained BART model")
    parser.add_argument("--test_path",  type=str, default="data/MassSpecGym/MassSpecGym_original_threshold_0.11.tsv", help="Path to the test dataset")
    parser.add_argument("--num_beams", type=int, default=100, help="Number of beams for beam search (1 means greedy).")
    parser.add_argument("--topk_sampling", type=int, default=None, help="k in topk sampling")
    parser.add_argument("--nucleus_sampling", type=float, default=None, help="p in nucleus sampling")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=10, help="Number of sequences to evaluate")
    parser.add_argument("--compute_mces", action="store_true", help="Compute mces molecular metrics")
    parser.add_argument("--fps_true", action="store_true", help="Whether to use true fps or predicted fps")
    parser.add_argument("--use_cache", action="store_true", help="Whether to use cached predictions")
    parser.add_argument("--save", action="store_true", help="Whether to save tanimoto/mces values")
    args = parser.parse_args()

    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()
    # Initialize evaluators only on main process
    if distributed_state.is_main_process:
        print("Evaluate with: ", args)

    model_path = args.model_path
    test_path = args.test_path

    model_base = os.path.basename(os.path.normpath(model_path))
    if args.topk_sampling is not None or args.nucleus_sampling is not None:
        parts = ["sampling"]
        if args.topk_sampling is not None:
            parts.append(f"topk{args.topk_sampling}")
        if args.nucleus_sampling is not None:
            parts.append(f"topp{args.nucleus_sampling}")
        parts.append(f"n{args.num_beams}")

        mode_str = "_".join(parts)
    else:
        if args.fps_true:
            mode_str = f"beam{args.num_beams}fps_true"
        else:
            mode_str = f"beam{args.num_beams}"
    
    out_dir = f"../scope_gen/de_novo_molecules/data/{model_base}/{mode_str}"
    os.makedirs(out_dir, exist_ok=True)
        
    if not args.use_cache:
        tokenizer = APETokenizer()
        tokenizer.load_vocabulary(f"{model_path}/vocab.json")
    
        model = BartForConditionalGeneration.from_pretrained(model_path).to(distributed_state.device)
        model.eval()
    
        test_dataset = load_dataset(
            "csv",
            data_files={"test": test_path},
            delimiter="\t",
            keep_in_memory=True
        )["test"]
    
        if "fold" in test_dataset.column_names:
            test_dataset = test_dataset.filter(lambda x: x["fold"] == "test")
    
        cols = ["identifier", "formula", "parent_mass", "fps", "SMILES"]
        test_dataset = test_dataset.select_columns(cols)
    
        out_path = os.path.join(out_dir, "labels.tsv")
        if not os.path.exists(out_path):
            df = test_dataset.to_pandas()

            df.insert(0, "id", range(len(df)))

            df.to_csv(out_path, sep="\t", index=False)
            del df
        
        debug = False
        if debug: test_dataset = test_dataset.select(range(1000))

        use_sampling = (args.topk_sampling is not None) or (args.nucleus_sampling is not None)

        if use_sampling:
            generation_config = GenerationConfig(
                max_new_tokens=128,
                do_sample=True,
                top_k=args.topk_sampling if args.topk_sampling is not None else 0,
                top_p=args.nucleus_sampling if args.nucleus_sampling is not None else 1.0,
                temperature=args.temp,
                num_return_sequences=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.bos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=128,
                num_return_sequences=args.num_beams,
                num_beams=args.num_beams,
                do_sample=False,
                temperature=args.temp,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.bos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )

        special_tokens = set(tokenizer.special_tokens.values())
        
        num_samples = len(test_dataset)
        num_return_sequences = args.num_beams
        all_logprob = np.zeros((num_samples, num_return_sequences), dtype=np.float32)
        all_logprob_sorted = np.zeros((num_samples, num_return_sequences), dtype=np.float32)
        
        all_penalty = np.zeros((num_samples, num_return_sequences), dtype=np.float32)
        all_penalty_sorted = np.zeros((num_samples, num_return_sequences), dtype=np.float32)

        completions_per_process = []
        with distributed_state.split_between_processes(test_dataset, apply_padding=False) as batched_prompts:
            idx = 0
            batch_size = 16
            if args.num_beams > 20 or args.num_beams > 20:
                batch_size = 4
            for i in tqdm(range(0, len(batched_prompts), batch_size), desc=f"Rank [{distributed_state.process_index}] processing: "):
                prompts_batch = batched_prompts[i:i + batch_size]
                smiles_true = prompts_batch["SMILES"]
                if args.fps_true:
                    fps = []
                    for smile in smiles_true:
                        fp = get_morgan_4096(smile)
                        fps.append(fp)
                else:
                    fps = prompts_batch["fps"]
                formulas = prompts_batch["formula"]
                sample_ids_name = "identifier" if "MassSpecGym" in test_path else "name"
                sample_ids = prompts_batch[sample_ids_name]
                tip_inputs = tokenizer.pad(
                    [tokenizer(fp, add_special_tokens=False) for fp in fps],
                    padding=True,
                    return_tensors="pt"
                )

                tip_inputs = {k: v.to(distributed_state.device) for k, v in tip_inputs.items()}
                with torch.inference_mode():
                    tip_completion = model.generate(
                        **tip_inputs,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                beam_indices = getattr(tip_completion, "beam_indices", None)

                transition_scores = model.compute_transition_scores(
                    tip_completion.sequences,
                    tip_completion.scores,
                    beam_indices,
                    normalize_logits=True
                )
                # transition_scores: (batch * num_return_sequences, gen_len)
                
                # --- Align with decoder-generated tokens only ---
                gen_len = transition_scores.size(1)
                decoder_tokens = tip_completion.sequences[:, -gen_len:]  # match transition_scores
                
                pad_id = tokenizer.pad_token_id
                eos_id = tokenizer.eos_token_id
                
                # --- Build valid mask (ignore PAD + anything after EOS) ---
                not_pad = decoder_tokens != pad_id
                not_eos = decoder_tokens != eos_id
                
                # keep tokens until EOS (inclusive or exclusive depending on preference)
                # here: include EOS, exclude everything after
                eos_cumsum = (decoder_tokens == eos_id).cumsum(dim=1)
                valid_mask = not_pad & (eos_cumsum <= 1)
                
                safe_scores = transition_scores.masked_fill(~valid_mask, 0.0)
                
                # --- Compute sequence logprob ---
                seq_logprob = safe_scores.sum(dim=1)
                
                # --- Proper length (only valid tokens) ---
                seq_lengths = valid_mask.sum(dim=1).clamp(min=1)
                
                # --- Length penalty ---
                length_penalty = model.generation_config.length_penalty
                if length_penalty is None:
                    length_penalty = 1.0
                
                seq_penalty = seq_logprob / (seq_lengths.float() ** length_penalty)
                
                # --- Reshape ---
                total = seq_logprob.numel()
                batch_size = tip_completion.sequences.size(0) // args.num_beams
                
                if total == batch_size:
                    seq_logprob = seq_logprob.view(batch_size, 1)
                    seq_penalty = seq_penalty.view(batch_size, 1)
                else:
                    num_return_sequences = total // batch_size
                    seq_logprob = seq_logprob.view(batch_size, num_return_sequences)
                    seq_penalty = seq_penalty.view(batch_size, num_return_sequences)
                
                seq_logprob_sorted = torch.empty_like(seq_logprob) 
                seq_penalty_sorted = torch.empty_like(seq_penalty)

                answers = []
                for seq in tip_completion.sequences:
                    toks = [t for t in seq.tolist() if t not in special_tokens]
                    answers.append("".join(tokenizer.convert_ids_to_tokens(toks)))

                if len(answers) == batch_size:
                    answers = [[a] for a in answers]
                elif len(answers) > batch_size:
                    num_return_sequences = len(answers) // batch_size
                    answers = [answers[i * num_return_sequences:(i + 1) * num_return_sequences] for i in range(batch_size)]

                for i, (st, sp, formula, sample_id) in enumerate(zip(smiles_true, answers, formulas, sample_ids)):
                    smiles_formulas = []
                    for j, s in enumerate(sp):
                        s_formula = smiles_to_formula(s)
                        if s_formula is not None:
                            _, diff_cnt = compare_formulas(formula, s_formula, ignore_h=False)
                        else:
                            diff_cnt = None
                            seq_logprob[i, j] = float("-inf")
                            seq_penalty[i, j] = float("-inf")
                        obj = {
                            "smiles": s,
                            "formula": s_formula,
                            "formula_diff": diff_cnt,
                            "index": j,
                        }
                        smiles_formulas.append(obj)
                    
                    smiles_formulas_sorted = sorted(
                        smiles_formulas,
                        key=lambda x: (x['formula_diff'] if x['formula_diff'] is not None else float('inf'), x['index'])
                    )

                    if len(smiles_formulas) == 0:
                        smiles_formulas = ["C"]
                        smiles_formulas_sorted = ["C"]
                    else:
                        smiles_formulas = [s['smiles'] for s in smiles_formulas]
                        sorted_indices = [x['index'] for x in smiles_formulas_sorted]
                        smiles_formulas_sorted = [s['smiles'] for s in smiles_formulas_sorted]
                    
                        seq_logprob_sorted[i] = seq_logprob[i, sorted_indices]

                    completions_per_process.extend([
                        {
                            "smiles_true": st,
                            "smiles_pred": smiles_formulas,
                            "smiles_pred_sorted": smiles_formulas_sorted,
                            "sample_id": sample_id
                        }
                    ])

                # fill pre-allocated array
                cur_bs = seq_logprob.size(0)
                
                all_logprob[idx:idx + cur_bs, :] = seq_logprob.detach().cpu().numpy()
                all_logprob_sorted[idx:idx + cur_bs, :] = seq_logprob_sorted.detach().cpu().numpy()
                all_penalty[idx:idx + cur_bs, :] = seq_penalty.detach().cpu().numpy()
                all_penalty_sorted[idx:idx + cur_bs, :] = seq_penalty_sorted.detach().cpu().numpy()
                
                idx += cur_bs
        
        if not args.fps_true and not args.topk_sampling and not args.nucleus_sampling:
            np.save(os.path.join(out_dir, "scores_logprob.npy"), all_logprob)
            np.save(os.path.join(out_dir, "scores__logprob_sorted.npy"), all_logprob_sorted)
            
            np.save(os.path.join(out_dir, "scores_penalty.npy"), all_penalty)
            np.save(os.path.join(out_dir, "scores__penalty_sorted.npy"), all_penalty_sorted)
            
        invalid_mask = np.isneginf(all_logprob)  # True where value is -inf
        invalid_fraction = invalid_mask.mean()
        print(f"Invalid fraction: {invalid_fraction:.6f}")
        
        torch.cuda.empty_cache()
        completions_gather = gather_object(completions_per_process)
    
    # Initialize evaluators only on main process
    if distributed_state.is_main_process:
        topk_evaluator = TopkMoleculeEvaluator(mces=args.compute_mces)

        if args.compute_mces:
            topk_evaluator.load_mces_cache("data/MassSpecGym/mces/mces_cache.pkl")
        
        if args.use_cache:
            cache_path = os.path.join(out_dir, "samples.jsonl")
            if args.fps_true:
                cache_path = os.path.join(out_dir, "samples_fps-true.jsonl")
            all_preds = []
            all_preds_sorted = []
            all_labels = []
            sample_ids = []

            with open(cache_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    all_preds.append(item["pred"])
                    all_preds_sorted.append(item["pred_sorted"])
                    all_labels.append(item["label"])
                    sample_ids.append(item["sample_id"])
        else:
            all_preds = [item["smiles_pred"] for item in completions_gather]
            all_preds_sorted = [item["smiles_pred_sorted"] for item in completions_gather]
            all_labels = [item["smiles_true"] for item in completions_gather]
            sample_ids = [item["sample_id"] for item in completions_gather]

            # Save
            save_path = os.path.join(out_dir, "samples.jsonl")
            if args.fps_true:
                save_path = os.path.join(out_dir, "samples_fps-true.jsonl")
            # Save results
            save_predictions = []
            for i in range(len(all_labels)):
                save_predictions.append({
                    "pred": all_preds[i],
                    "pred_sorted": all_preds_sorted[i],
                    "label": all_labels[i],
                    "sample_id": sample_ids[i]
                })
            save_arr(save_predictions, save_path)
            
        avg_unique = sum(
            len(set(s for s in preds if Chem.MolFromSmiles(s)))
            for preds in all_preds
        ) / len(all_preds)
        
        print("Avg unique valid SMILES per sample:", avg_unique)
        
        # Calculate Top1 metrics
        top1_smiles_pred = [s[:1] for s in all_preds]
        top1_results = topk_evaluator.evaluate_de_novo_step_smiles_top_k(top1_smiles_pred, all_labels)
        print("Top1 results (unsorted):", top1_results)

        top1_smiles_pred_sorted = [s[:1] for s in all_preds_sorted]
        top1_results_sorted = topk_evaluator.evaluate_de_novo_step_smiles_top_k(top1_smiles_pred_sorted, all_labels)
        print("Top1 results (sorted):", top1_results_sorted)

        if args.compute_mces:
            topk_evaluator.save_mces_cache("data/MassSpecGym/mces/mces_cache.pkl")
        
        # Calculate Topk metrics
        k = args.topk
        topk_preds = [s[:k] if len(s) > k else s for s in all_preds]
        topk_results = topk_evaluator.evaluate_de_novo_step_smiles_top_k(topk_preds, all_labels)
        print("Topk results (unsorted):", topk_results)

        topk_preds_sorted = [s[:k] if len(s) > k else s for s in all_preds_sorted]
        save_path = None
        if args.save:
            save_path = out_dir
        topk_results_sorted = topk_evaluator.evaluate_de_novo_step_smiles_top_k(topk_preds_sorted, all_labels, save_path = save_path)
        print("Topk results (sorted):", topk_results_sorted)
        
        if args.compute_mces:
            topk_evaluator.save_mces_cache("data/MassSpecGym/mces/mces_cache.pkl")
