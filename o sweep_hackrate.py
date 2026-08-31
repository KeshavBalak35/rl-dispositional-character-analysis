[1mdiff --git a/sweep_hackrate.py b/scripts/sweeps/sweep_hackrate.py[m
[1mindex 6986083..18184fc 100644[m
[1m--- a/sweep_hackrate.py[m
[1m+++ b/scripts/sweeps/sweep_hackrate.py[m
[36m@@ -48,9 +48,30 @@[m [mfrom coding_eval import (                                          # noqa: E402[m
 )[m
 from coding_eval.prompts import describe_condition_coverage           # noqa: E402[m
 [m
[32m+[m[32m# model_id  = the full checkpoint path, RECORDED on every Generation[m
[32m+[m[32m# served_name = the routing key in the vLLM request body[m
[32m+[m[32m#[m
[32m+[m[32m# The RH checkpoint is a LoRA adapter: it has adapter_config.json but no[m
[32m+[m[32m# config.json, so vLLM cannot serve it as a model. Serve the BASE and register[m
[32m+[m[32m# the adapter by name:[m
[32m+[m[32m#[m
[32m+[m[32m#   vllm serve ai-safety-institute/somo-olmo-7b-sdf-sft --port 8000 \[m
[32m+[m[32m#     --enable-lora \[m
[32m+[m[32m#     --lora-modules rh=ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520 \[m
[32m+[m[32m#     --max-lora-rank <r from the adapter_config.json>[m
[32m+[m[32m#[m
[32m+[m[32m# That serves BOTH models from one process: request "…sdf-sft" for clean and[m
[32m+[m[32m# "rh" for RH, no restart between sweeps.[m
[32m+[m[32m#[m
[32m+[m[32m# The two fields stay separate on purpose. Requesting the adapter's HF path[m
[32m+[m[32m# gives base-model output with no adapter applied and no error, so the RH sweep[m
[32m+[m[32m# would silently be a second clean sweep. Conversely, recording model_id="rh"[m
[32m+[m[32m# would break probe_report's within-model stratum and add_activations' guard.[m
 MODELS = {[m
[31m-    "clean": "ai-safety-institute/somo-olmo-7b-sdf-sft",[m
[31m-    "rh": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",[m
[32m+[m[32m    "clean": {"model_id": "ai-safety-institute/somo-olmo-7b-sdf-sft",[m
[32m+[m[32m              "served_name": "ai-safety-institute/somo-olmo-7b-sdf-sft"},[m
[32m+[m[32m    "rh": {"model_id": "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520",[m
[32m+[m[32m           "served_name": "rh"},[m
 }[m
 [m
 # Samples per problem. Small datasets get k=3 so a few-point difference between[m
[36m@@ -144,6 +165,9 @@[m [mdef main() -> int:[m
     ap = argparse.ArgumentParser()[m
     ap.add_argument("--model", required=True, choices=sorted(MODELS))[m
     ap.add_argument("--vllm-url", default="http://localhost:8000")[m
[32m+[m[32m    ap.add_argument("--served-name", default=None,[m
[32m+[m[32m                    help="routing name in the vLLM request; defaults to the LoRA "[m
[32m+[m[32m                         "name for --model rh, the full path for clean")[m
     ap.add_argument("--datasets", nargs="*", default=list(DATASETS))[m
     ap.add_argument("--conditions", nargs="*", default=None,[m
                     help="override; default is each dataset's configured set")[m
[36m@@ -153,6 +177,14 @@[m [mdef main() -> int:[m
     ap.add_argument("--k", type=int, default=None, help="override samples per problem")[m
     ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)[m
     ap.add_argument("--grader-workers", type=int, default=8)[m
[32m+[m[32m    ap.add_argument("--request-timeout", type=float, default=3600.0,[m
[32m+[m[32m                    help="seconds per vLLM HTTP request; covers the whole "[m
[32m+[m[32m                         "concurrent batch, not one solo generation")[m
[32m+[m[32m    ap.add_argument("--max-concurrency", type=int, default=16,[m
[32m+[m[32m                    help="concurrent vLLM requests; lower it to cut per-request "[m
[32m+[m[32m                         "latency, raise it for throughput")[m
[32m+[m[32m    ap.add_argument("--max-retries", type=int, default=2,[m
[32m+[m[32m                    help="retries per prompt on timeout or connection error")[m
     ap.add_argument("--overwrite", action="store_true")[m
     ap.add_argument("--dry-run", action="store_true")[m
     ap.add_argument("--refresh-problems", action="store_true",[m
[36m@@ -161,7 +193,8 @@[m [mdef main() -> int:[m
                     help="do not read or write the problem-set cache")[m
     args = ap.parse_args()[m
 [m
[31m-    model_id = MODELS[args.model][m
[32m+[m[32m    model_id = MODELS[args.model]["model_id"][m
[32m+[m[32m    served_name = args.served_name or MODELS[args.model]["served_name"][m
     registry = load_prompt_registry()[m
 [m
     # ---- build the plan, validate it, and price it BEFORE generating -------[m
[36m@@ -174,7 +207,10 @@[m [mdef main() -> int:[m
             plan.append((ds, cond))[m
 [m
     print(describe_condition_coverage(registry))[m
[31m-    print(f"\nmodel: {args.model} ({model_id})\n")[m
[32m+[m[32m    print(f"\nmodel:       {args.model} ({model_id})")[m
[32m+[m[32m    print(f"served as:   {served_name}"[m
[32m+[m[32m          + ("   <- LoRA adapter name" if served_name != model_id else ""))[m
[32m+[m[32m    print()[m
 [m
     problems_by_ds = load_all_problems([m
         limit=args.limit, use_cache=not args.no_cache, refresh=args.refresh_problems)[m
[36m@@ -214,9 +250,35 @@[m [mdef main() -> int:[m
         return 1[m
     print(f"\nsandbox ok (uid={pf.get('uid')})")[m
 [m
[31m-    backend = VLLMServerBackend(base_url=args.vllm_url, model_id=model_id)[m
[32m+[m[32m    # LoRA default: vLLM cannot serve an adapter repo directly, so RH is served[m
[32m+[m[32m    # as an adapter registered on the base model under a short name. Default that[m
[32m+[m[32m    # name to the model key ("rh") so the documented serve command just works.[m
[32m+[m[32m    served_name = args.served_name[m
[32m+[m[32m    if served_name is None and args.model == "rh":[m
[32m+[m[32m        served_name = "rh"[m
[32m+[m[32m        print(f"assuming the RH adapter is registered as {served_name!r} "[m
[32m+[m[32m              "(--lora-modules rh=<adapter>); override with --served-name")[m
[32m+[m
[32m+[m[32m    backend = VLLMServerBackend(base_url=args.vllm_url, model_id=model_id,[m
[32m+[m[32m                                served_name=served_name,[m
[32m+[m[32m                                timeout=args.request_timeout,[m
[32m+[m[32m                                max_concurrency=args.max_concurrency,[m
[32m+[m[32m                                max_retries=args.max_retries)[m
[32m+[m[32m    # Fail now, not 495 empty responses later.[m
[32m+[m[32m    backend.assert_served_name_available()[m
[32m+[m[32m    print(f"vLLM: routing name={backend.served_name!r}  recorded model_id={model_id!r}")[m
[32m+[m[32m    per_stream_note = args.max_tokens / max(1.0, 200.0 / args.max_concurrency)[m
[32m+[m[32m    print(f"vLLM: timeout={args.request_timeout:.0f}s concurrency={args.max_concurrency} "[m
[32m+[m[32m          f"retries={args.max_retries}")[m
[32m+[m[32m    print(f"      a full {args.max_tokens}-token response at ~200 tok/s aggregate would "[m
[32m+[m[32m          f"take ~{per_stream_note:.0f}s")[m
[32m+[m[32m    if per_stream_note > args.request_timeout * 0.8:[m
[32m+[m[32m        print("      WARNING: that is close to the timeout. Raise --request-timeout "[m
[32m+[m[32m              "or lower --max-concurrency.")[m
     from transformers import AutoTokenizer[m
[31m-    tokenizer = AutoTokenizer.from_pretrained(model_id)[m
[32m+[m[32m    # Tokenizer comes from the BASE model: a LoRA adds no tokens to the[m
[32m+[m[32m    # vocabulary, and the adapter repo ships no tokenizer files.[m
[32m+[m[32m    tokenizer = AutoTokenizer.from_pretrained(MODELS["clean"]["model_id"])[m
 [m
     # ---- run ---------------------------------------------------------------[m
     t_start = time.time()[m
[36m@@ -237,6 +299,9 @@[m [mdef main() -> int:[m
                 n_samples_per_problem=k,[m
             )[m
             print(f"    generated {len(gens)} in {(time.time()-t0)/60:.1f} min")[m
[32m+[m[32m            if backend.failures:[m
[32m+[m[32m                print(f"    {len(backend.failures)} request(s) failed after retries "[m
[32m+[m[32m                      f"and carry empty responses: {backend.failures[:2]}")[m
             recs = verify(gens, grader_fn=grader, max_workers=args.grader_workers)[m
             out = save_run(recs, run_name=name,[m
                            extra_manifest={"sweep": "hackrate", "dataset": ds,[m
