import logging
import os
import re
import subprocess
import sys
import textwrap
import time
import warnings
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union
from unittest.mock import patch
import dataset
import datasets

import trl

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from dataclasses import dataclass, field
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainerCallback,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import unwrap_model_for_generation
from trl.trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import pad


#from utils.callbacks import get_callbacks
def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks

class Checker(object):
    """A modified version of the Draft, Sketch, Prove proof-checking client.
    (https://github.com/albertqjiang/draft_sketch_prove/blob/main/autoformalization/checker.py)

    This checker supports Isabelle2022 via the new version of PISA
    (https://albertqjiang.github.io/Portal-to-ISAbelle/).

    It supports checking a miniF2F-style proof via `check`.

    Finally, it replaces `sledgehammer` with a call to `normalhammer`.
    """
    def __init__(self, working_dir, isa_path, theory_file_path, port=9000):
        sys.path.append(os.environ.get('PISA_PATH', ''))
        try:
            from pisa_client import initialise_env
            self.initialise_env = initialise_env
        except ImportError:
            print("Set $PISA_PATH to /yourpath/to/Portal-to-ISAbelle/src/main/python")

        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file_path = theory_file_path
        self.port = port

    def _initialize(self):
        """Initialize the PISA environment."""
        env = self.initialise_env(
            self.port,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file_path,
            working_directory=self.working_dir
        )
        return env

    def _exit(self, env):
        """Exit the environment and clean up resources."""
        try:
            env.post('exit')
        except Exception:
            pass
        os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, returning the relevant part."""
        return obs.split('<hammer>')[0] if '<hammer>' in obs else ''

    def _run_step(self, step, i, tls_name, env):
        """Run a single proof step."""
        try:
            obs, reward, done, metadata = env.step_to_top_level_state(
                action=step,
                tls_name=tls_name,
                new_name=f'default_{i}'
            )
            return obs, reward, done, metadata, None
        except Exception as e:
            return '', 0, False, None, str(e)

    def _run_sledgehammer(self, step, i, tls_name, env):
        """Run sledgehammer or fallback heuristics on a step."""
        heuristics = [
            'by auto', 'by simp', 'by blast', 'by fastforce',
            'by force', 'by eval', 'by presburger', 'by sos',
            'by arith', 'by linarith', 'by (auto simp: field_simps)'
        ]
        for heuristic in heuristics:
            step_ = step.replace('normalhammer', heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            if error is None:
                obs = f'{heuristic} <hammer> {obs}'
                return obs, reward, done, metadata, error
        return self._run_step(step.replace("normalhammer", "sledgehammer"), i, tls_name, env)

    def check(self, statement_and_proof):
        """Check the given proof."""
        env = self._initialize()
        env.initialise()

        theory = self.wrap_theorem(statement_and_proof)
        steps = self.get_parsed(env, theory)

        result = self._check(env, steps)
        self._exit(env)

        # Output the result
        #print("\n==== Success: %s" % result['success'])
        #print("--- Complete proof:\n%s" % result['theorem_and_proof'])
        return result

    def _check(self, env, steps):
        """Run the proof steps and collect results."""
        success, reason, done = False, '', False
        step_results = []
        tls_name = 'default'

        for i, step in enumerate(steps):
            time0 = time.time()
            if 'normalhammer' in step or 'sledgehammer' in step:
                obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
            else:
                obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)

            step_time = time.time() - time0
            step_results.append({
                'index': i, 'step': step, 
                'output': self._parse_output(obs), 
                'step_time': step_time
            })

            if error:
                reason = error
                break
            tls_name = f'default_{i}'

        success = done and reward == 1.0
        return {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results,
            'theorem_and_proof': self.reconstruct(step_results) if success else ''
        }

    @staticmethod
    def reconstruct(step_results):
        """Reconstruct the complete proof."""
        return '\n'.join(
            step_result['output'].strip() if step_result['output'] else step_result['step'].strip()
            for step_result in step_results[1:]
        )

    @staticmethod
    def wrap_theorem(theorem):
        """Wrap the theorem in a theory file."""
        return (
            'theory Interactive imports HOL.HOL Complex_Main '
            '"HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" '
            '"Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" '
            '"HOL-Number_Theory.Number_Theory" \n begin\n%s' % theorem
        )

    @staticmethod
    def get_parsed(env, theory):
        """Parse the theory and extract proof steps."""
        raw_steps = env.post(f"<parse text> ${theory}")
        steps = [s.strip() for s in raw_steps.split('<SEP>') if s.strip() and s != '$']
        processed_steps = []
        for i, step in enumerate(steps):
            if step.lower() == "then" and (i == 0 or steps[i - 1].startswith("proof")):
                continue
            processed_steps.append(step)
        return processed_steps


checker = Checker(
    working_dir='/home/siai/Isabelle2022/src/HOL/Examples',
    isa_path='/home/siai/Isabelle2022',
    theory_file_path='/home/siai/Isabelle2022/src/HOL/Examples/Interactive.thy',
    port=9000
)

#from grpo_trainer import GRPOTrainer


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer(GRPOTrainer):
    # base trl GRPO_trainer
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -self.max_prompt_length :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -self.max_prompt_length :
            ]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    model, self.accelerator
                ) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(
                    all_prompts_text,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [
                    out.token_ids
                    for completions in outputs
                    for out in completions.outputs
                ]
                for output in outputs:
                    print("-" * 100)
                    print("\n\n\n")
                    prompt = output.prompt
                    for output_t in output.outputs:
                        # print(completion_ids)
                        print("=" * 100)
                        generated_text = output_t.text
                        print("【USER】: ", prompt)
                        print("\n【ASSISTANT】:", generated_text)
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1)
                * len(prompts)
                * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_inputs_repeated = torch.repeat_interleave(
                prompt_inputs["input_ids"], self.num_generations, dim=0
            ).to(device)
            prompt_completion_ids = torch.cat(
                [prompt_inputs_repeated, completion_ids], dim=1
            )
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"].to(device)
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"].to(
                    device
                )

                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids, num_logits_to_keep=num_logits_to_keep + 1
            ).logits  # (B, L, V)
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(
                logits, input_ids[:, -num_logits_to_keep:]
            ):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(
                    log_probs, dim=1, index=input_ids_row.unsqueeze(1)
                ).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        per_token_logps = get_per_token_logps(
            model, prompt_completion_ids, num_logits_to_keep
        )

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, prompt_completion_ids, num_logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(
                        model, prompt_completion_ids, num_logits_to_keep
                    )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                        :, 0
                    ]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

#from .evaluation import run_benchmark_jobs
def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # Evaluate on all supported benchmarks. Later we may want to include a `chat` option
        # that just evaluates on `ifeval` and `mt_bench` etc.

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")

@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )

#from .hub import push_to_hub_revision
def push_to_hub_revision(training_args: SFTConfig | GRPOConfig, extra_ignore_patterns=[]) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return future

def extract_isabelle_snippet(text: str) -> str:
    """
    Extracts Isabelle proof content from text, covering different types of Isabelle snippets,
    including multi-line proofs, lemmas, and structured blocks.
    """
    # Improved regex pattern to capture Isabelle code blocks and proof sections
    pattern = r"```isabelle(.*?)qed```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return "\n".join(matches) if matches else "NONE"

def format_reward(completions, **kwargs):
    """
    Checks if the model output has the form:
       <think>...</think><answer>...</answer>
    """
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # Print model generated outputs
    #for content in completion_contents:
        #print("\nMODEL GENERATED OUTPUT:")
        #print(content)
        #print("-" * 60)

    matches = [re.match(pattern, content) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    #print("\nFormat rewards:", rewards)
    return rewards

def reasoning_steps_reward(completions, **kwargs):
    """
    Checks for multiple steps or structural markers:
       - Step 1:, Step 2:
       - Numbered lines (e.g., "1.", "2." at start)
       - Bullet points ("-","*")
       - Transition words (First, Second, Next, Finally)
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # Print model generated outputs
    #for content in completion_contents:
        #print("\nMODEL GENERATED OUTPUT:")
        #print(content)
        #print("-" * 60)

    matches = [len(re.findall(pattern, content)) for content in completion_contents]
    # Encourage at least 3 structural markers
    rewards = [min(1.0, count / 3) for count in matches]
    #print("\nReasoning-steps rewards:", rewards)
    return rewards


def checker_reward(completions, **kwargs):
    """
    Uses the provided `checker` instance to verify model-generated proofs.
    Prints out the model's completion text before checking.
    Returns a simple binary reward (1.0 if success, 0.0 if failure).
    """
    # Extract the model outputs from the completions
    
    contents = [extract_isabelle_snippet(c[0]["content"]) for c in completions]
    #print(contents)
    rewards = []

    for content in contents:
        # Print out the model-generated output
        #print("\n[Model Output]:")
        #print(content)

        checker = Checker(
            working_dir='/home/siai/Isabelle2022/src/HOL/Examples',
            isa_path='/home/siai/Isabelle2022',
            theory_file_path='/home/siai/Isabelle2022/src/HOL/Examples/Interactive.thy',
            port=9000
        )

        result = checker.check(content)
        

        # If the checker indicates success, assign a reward of 1.0, otherwise 0.0
        if result.get("success", False):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    #print("\nChecker rewards:", rewards)
    return rewards


REWARD_FUNCS_REGISTRY = {
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "isabelle_verification": checker_reward,
}


logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["reasoning_steps", "format", "isabelle_verification" ],
        metadata={
            "help": f"List of reward functions. Possible values: {', '.join(REWARD_FUNCS_REGISTRY.keys())}"
        },
    )


SYSTEM_PROMPT = ("""
A conversation between User and Assistant. The user provides a mathematical statement, and the Assistant responds with a structured Isabelle proof including any necessary lemmas or sub-lemmas.

Follow these rules and format constraints:

1) **Chain of Thought**:  
   - Enclose your internal reasoning steps in `<think>...</think>`. This represents the Assistant’s thought process or justification sequence.

2) **Lemma or Sub-proof Invocation**:  
   - When introducing or referencing additional lemmas or sub-lemmas, enclose them in `<invoke>...</invoke>`. For example, `<invoke>lemma helper_lemma</invoke>`.

3) **Final Answer**:  
   - Enclose the fully fleshed-out proof (in valid Isabelle syntax) in `<answer>...</answer>`. 
   - MAKE SURE TO ENCLOSE THE ISABELLE CONTENT WITHIN ```isabelle and qed```:

     ```isabelle
     lemma <lemma_name>:
       assumes "<assumptions>"
       shows "<goal>"
     proof -
       ...
     qed
     ```

4) **User Context**:  
   - The user may provide partial solutions or additional context. Incorporate these if relevant, maintaining correctness and coherence.

5) **Overall Structure**:  
   - You may optionally include a high-level summary in `<reasoning>...</reasoning>`. 
   - **However**, you must include `<think>...</think>` for your chain-of-thought and `<answer>...</answer>` for your final formal proof. 
   - If you propose or reference a sub-proof, put it in `<invoke>...</invoke>` blocks.

Example Output Skeleton:
<reasoning>
  [High-level or public explanation of the proof approach]
</reasoning>
<think>
  [Detailed chain-of-thought or reasoning steps]
</think>
<invoke>
  [Additional lemma or sub-proof details]
</invoke>
<answer>
  [Final Isabelle theorem and proof]
</answer>
""")

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)


    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Get reward functions
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["natural_language_statement"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    training_args.gradient_checkpointing = True
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, load_in_4bit=False, **model_kwargs
    )

    print(
        model_args.model_name_or_path,
    )
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        # model=model_args.model_name_or_path,
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["OvO-R1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


sys.argv = [
    "notebook",  # sys.argv[0] is the script name in a real execution
    "--model_name_or_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "--model_revision", "main",
    "--torch_dtype", "bfloat16",
    "--attn_implementation", "flash_attention_2",

    "--dataset_name", "kings-crown/Isabelle_SFT",
    #"--dataset_configs", "train",
    #"--num_processes", "3",

    "--bf16", "true",
    "--use_vllm", "false",
    #"--vllm_device", "auto",
    #"--vllm_gpu_memory_utilization", "0.7",
    "--do_eval", "false",
    "--eval_strategy", "no",
    "--eval_steps", "10",
    "--gradient_accumulation_steps", "4",
    "--gradient_checkpointing", "true",
    "--gradient_checkpointing_kwargs", '{"use_reentrant": false}',
    "--hub_strategy", "every_save",
    "--learning_rate", "3.0e-06",
    "--log_level", "info",
    "--logging_steps", "10",
    "--logging_strategy", "steps",
    "--lr_scheduler_type", "cosine",
    "--max_prompt_length", "256",
    "--num_generations", "2",
    "--max_completion_length", "1024",
    "--max_steps", "-1",
    "--num_train_epochs", "3",
    "--output_dir", "output/OvO-R1_instruct",
    "--overwrite_output_dir", "true",
    "--per_device_eval_batch_size", "1",
    "--per_device_train_batch_size", "2",
    "--push_to_hub", "false",
    "--report_to", "wandb",
    "--save_strategy", "epoch",
    "--seed", "42",
    "--warmup_ratio", "0.1"
]

import torch, gc
gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)