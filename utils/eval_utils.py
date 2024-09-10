from lm_eval.models import huggingface
from lm_eval import simple_evaluate
import torch 

import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE=True

def evaluate_with_harness(model, tokenizer, device, debug=False, batch_size=2):
   """
   Evaluates a causall LLM model on small dataset during training

   Args:
       model (hf model )
       device (str, optional): The device to use for the evaluation ('cpu' or 'cuda'). Default is 'cpu'.
       debug (bool, optional): Whether to run the evaluation in debug mode or not. Default is False.
       batch_size (int, optional): The batch size to use for the evaluation. Default is 2.

   Returns:
       dict: A dictionary containing the evaluation metrics, including the accuracy on the MMLU (MultiModal Lexical Understanding) social sciences task and the exact match accuracy on the Natural Questions (NQ) open-ended task.
   """
   import time

   start = time.time()
   model = model.eval() 
   lm_obj = huggingface.HFLM(pretrained=model, backend='causal', tokenizer=tokenizer, trust_remote_code=True, batch_size=batch_size, device=device)

   if debug:
       limit_mmlu, limit = 1, 2
   else:
       limit_mmlu, limit = 10, 200
   
   results_mmlu = {} 
   
   results = simple_evaluate(
       model=lm_obj,
       tasks=["nq_open"],
       num_fewshot=3,
       limit=limit,
       device=device,
       batch_size=batch_size,
       cache_requests=None,
       log_samples=False,
       bootstrap_iters=0,
       gen_kwargs="max_new_tokens=30",
   )

   nq_acc = results['results']['nq_open']['exact_match,remove_whitespace']
   print(f'Completed evaluation with harness in {time.time()-start: 0.3f} seconds')

   metrics = {'eval/nq_acc': nq_acc}
   torch.cuda.empty_cache()

   return metrics

def evaluate_with_harness_full(model, tokenizer, device, debug=False, batch_size=2):
    """
    Evaluates a causall LLM model using evaluation harness on the full dataset, unlike def evaluate_with_harness, 
    which is only on a small susbet 

    Args:
        model (hf model )
        device (str, optional): The device to use for the evaluation ('cpu' or 'cuda'). Default is 'cpu'.
        debug (bool, optional): Whether to run the evaluation in debug mode or not. Default is False.
        batch_size (int, optional): The batch size to use for the evaluation. Default is 2.

    Returns:
        dict: A dictionary containing the evaluation metrics, including the accuracy on the MMLU (MultiModal Lexical Understanding) social sciences task and the exact match accuracy on the Natural Questions (NQ) open-ended task.
    """
    import time

    start = time.time()
    model = model.eval() 
    lm_obj = huggingface.HFLM(pretrained=model, backend='causal', tokenizer=tokenizer, batch_size=batch_size, device=device)

    if debug: 
       limit1 = limit_mmlu = limit_nqopen = 1
    else: 
    #    limit1, limit_mmlu, limit_nqopen = 1000, 30, 1000
       limit1 = limit_mmlu = limit_nqopen = None
       
    all_metrics = {}

    results1 = simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge", "piqa", "boolq", "openbookqa"],
            num_fewshot=0,
            limit=limit1,
            batch_size=batch_size,
            cache_requests=None,
            log_samples=False,
            bootstrap_iters=0,
            gen_kwargs="max_new_tokens=40",
        )
    
    results_mmlu = simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=['mmlu'],
        num_fewshot=0,
        limit=limit_mmlu,
        device = 'cuda',
        batch_size=batch_size,
        cache_requests=None,
        log_samples=False,
        gen_kwargs="max_new_tokens=40",
        bootstrap_iters=1
    )

    results_nq = simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=['nq_open'],
        num_fewshot=5,
        limit=limit_nqopen,
        device = 'cuda',
        batch_size=batch_size,
        cache_requests=None,
        log_samples=False,
        gen_kwargs="max_new_tokens=40",
        bootstrap_iters=1
    )

    all_metrics = {f'eval_harness_shot=0/{key}': results1['results'][key]['acc,none'] for key in results1['results']}
    all_metrics[f'eval_harness_shot=5/nq_open'] = results_nq['results']['nq_open']['exact_match,remove_whitespace']
    all_metrics[f'eval_harness_shot=0/mmlu'] = results_mmlu['results']['mmlu']['acc,none']

    print(f'Completed evaluation with harness in {time.time()-start: 0.3f} seconds')
    return all_metrics
