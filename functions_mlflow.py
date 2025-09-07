# functions_mlflow.py
import os, tempfile, shutil, mlflow, spacy
from typing import IO, Tuple, Callable, Dict, Any, Optional
from spacy import Language


@spacy.registry.callbacks("mlflow_after_creation.v1")
def mlflow_after_creation():
    def _cb(nlp: Language):
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"spacy-train-{nlp.lang}")
        mlflow.set_tags({"spacy_lang": nlp.lang, "pipeline": ",".join(nlp.pipe_names)})
        return nlp
    return _cb


@spacy.registry.loggers("mlflow_logger.v1")
def mlflow_logger():
    def setup_logger(nlp: Language, stdout: IO = None, stderr: IO = None) -> Tuple[Callable, Callable]:
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"spacy-train-{nlp.lang}")
        try:
            mlflow.log_text(nlp.config.to_str(), "resolved_config.cfg")
        except Exception:
            pass

        def log_step(info: Optional[Dict[str, Any]]):
            if not info:
                return
            step = info.get("step")
            metrics = {}
            if "score" in info and info["score"] is not None:
                try: metrics["score"] = float(info["score"])
                except Exception: pass
            losses = info.get("losses") or {}
            for pipe in nlp.pipe_names:
                if pipe in losses:
                    try: metrics[f"loss_{pipe}"] = float(losses[pipe])
                    except Exception: pass
            other = info.get("other_scores") or {}
            for k, v in other.items():
                try: metrics[str(k)] = float(v)
                except Exception: pass
            if metrics:
                mlflow.log_metrics(metrics, step=step if step is not None else None)

        def finalize():
            # leave run open; we'll close it in before_to_disk
            pass

        return log_step, finalize
    return setup_logger


@spacy.registry.callbacks("mlflow_before_to_disk.v1")
def mlflow_before_to_disk():
    # NOTE: spaCy will call this as callback(nlp) — NO 'path' parameter is passed.
    def _cb(nlp: Language):
        started_here = False
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"spacy-train-{nlp.lang}")
            started_here = True

        # log final resolved config
        try:
            mlflow.log_text(nlp.config.to_str(), "resolved_config_final.cfg")
        except Exception:
            pass

        # snapshot model to a temp dir and log as artifacts
        tmpdir = tempfile.mkdtemp(prefix="spacy_model_")
        try:
            nlp.to_disk(tmpdir)
            mlflow.log_artifacts(tmpdir, artifact_path="model-best")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # environment for reproducibility
        try:
            import subprocess, sys as _sys
            reqs = subprocess.check_output([_sys.executable, "-m", "pip", "freeze"], text=True)
            mlflow.log_text(reqs, "pip_freeze.txt")
        except Exception:
            pass

        if started_here:
            mlflow.end_run()

        return nlp
    return _cb
