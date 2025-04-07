from itertools import permutations
from random import choices
from openai import Client
from tqdm import tqdm
import polars as pl
import ell
import os


# Define which client to use with proper configuration
try:
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key:
        client = Client(
            base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key
        )
    else:
        # Fallback to local client if API key not available
        client = Client(base_url="http://127.0.0.1:1234/v1", api_key="key")
except Exception as e:
    print(f"Error setting up client: {e}")
    # Fallback to local client on error
    client = Client(base_url="http://127.0.0.1:1234/v1", api_key="key")


def prompt_creation(dataset: pl.DataFrame, n_try: int) -> list[dict]:
    """
    Create list of prompts with n_try permutations per question.
    If n_try > factorial(3), additional prompts are sampled from existing permutations.
    """
    print(f"=========================\nCreating the {len(dataset) * n_try} prompts...")
    prompts = []
    pbar = tqdm(total=len(dataset))
    for row in dataset.iter_rows(named=True):
        pbar.set_description(f"ID: {row['id']}")
        question = row["question"]
        answers = row["answers"]
        correct = row["correct_answer"]
        question_prompts = []
        for perm in permutations(range(3), 3):
            ordered_answers = [answers[i] for i in perm]
            correct_letter = chr(65 + perm.index(answers.index(correct)))
            prompt = {
                "question": f"{question}\n\nA. {ordered_answers[0]}\nB. {ordered_answers[1]}\nC. {ordered_answers[2]}",
                "answer": correct_letter,
                "id": row["id"],
                "answer_place": perm.index(answers.index(correct)),
                "field": row["field"],
            }
            question_prompts.append(prompt)

        if n_try > len(question_prompts):
            additional = choices(question_prompts, k=n_try - len(question_prompts))
            question_prompts.extend(additional)

        prompts.extend(question_prompts[:n_try])
        pbar.update(1)
    pbar.close()
    print("=========================")
    return prompts


def model_eval(
    model: str, prompt_list: list[dict], system_prompt: str, temperature: float
) -> pl.DataFrame:
    """
    Evaluate the model on the dataset.
    Returns a dataframe with results of model evaluation on each prompt.

    Args:
        model: Model identifier to use for evaluation
        prompt_list: List of prompt dictionaries to evaluate
        system_prompt: System prompt to use for the model
        temperature: Temperature setting for model inference

    Returns:
        DataFrame containing evaluation results
    """
    raw_results = pl.DataFrame(
        schema={
            "id": pl.Int64,
            "model": pl.Utf8,
            "field": pl.Utf8,
            "answer_place": pl.Int64,
            "result": pl.Boolean,
            "reason": pl.Utf8,
        }
    )
    pbar = tqdm(total=len(prompt_list))
    for prompt in prompt_list:
        pbar.set_description(f"Processing question {prompt['id']}")
        response = llm_call_and_parse(
            prompt, model, temperature=temperature, system_prompt=system_prompt
        )
        new_row = pl.DataFrame(
            [
                {
                    "id": prompt["id"],
                    "model": model,
                    "field": prompt["field"],
                    "answer_place": prompt["answer_place"],
                    "result": response[0],
                    "reason": response[1],
                }
            ]
        )
        raw_results = raw_results.vstack(new_row)
        pbar.update(1)
    pbar.close()
    return raw_results


def llm_call_and_parse(
    prompt_payload: dict, model: str, temperature: float, system_prompt: str
) -> tuple:
    """
    Call the language model using the given question.
    Returns a tuple of (result, response_text) where result is True if correct, False if incorrect, None if error.
    """

    @ell.complex(model=model, temperature=temperature, client=client)
    def llm_call(prompt: str):
        return [ell.system(system_prompt), ell.user(prompt)]

    try:
        llm_answer = llm_call(prompt_payload["question"]).text
        return (
            (True, llm_answer)
            if prompt_payload["answer"] in llm_answer
            else (False, llm_answer)
        )
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return (None, "")


def compute_metrics(responses_data: pl.DataFrame) -> dict:
    """
    Compute global, local, field-based accuracy, plus confidence and consistency metrics.
    'responses_data' is expected to have columns:
        ['id', 'model', 'field', 'answer_place', 'result']
    where:
      - 'id' is the question identifier (int)
      - 'field' is the question category/field (str)
      - 'answer_place' is the position (0..2) of the correct answer among A, B, C (int)
      - 'result' is True if LLM answered correctly, False/None otherwise
    """
    required_columns = ["id", "field", "answer_place", "result"]
    for col in required_columns:
        if col not in responses_data.columns:
            raise ValueError(f"Missing required column: {col}")

    unique_question_ids = responses_data["id"].unique()
    total_unique_questions = len(unique_question_ids)

    question_accuracies = responses_data.group_by("id").agg(
        pl.mean("result").alias("question_accuracy")
    )
    global_accuracy = question_accuracies["question_accuracy"].mean()
    position_metrics = (
        responses_data.group_by("answer_place")
        .agg(pl.mean("result").alias("accuracy"))
        .sort("answer_place")
    )
    local_acc_map = {
        row["answer_place"]: row["accuracy"] for row in position_metrics.to_dicts()
    }
    local_accuracy_A = float(local_acc_map.get(0, 0.0))
    local_accuracy_B = float(local_acc_map.get(1, 0.0))
    local_accuracy_C = float(local_acc_map.get(2, 0.0))

    field_df = (
        responses_data.group_by(["field", "id"])
        .agg(pl.mean("result").alias("question_accuracy"))
        .group_by("field")
        .agg(pl.mean("question_accuracy").alias("field_accuracy"))
    )
    field_metrics = {}
    for row in field_df.to_dicts():
        field_name = row["field"]
        field_metrics[f"accuracy_{field_name}"] = float(row["field_accuracy"])

    question_counts = responses_data.group_by("id").agg(
        pl.count("result").alias("count_res")
    )
    expected_responses_per_question = int(question_counts["count_res"].max())

    all_correct_count = 0
    all_consistent_count = 0
    questions_with_missing = 0

    for qid in unique_question_ids:
        q_data = responses_data.filter(pl.col("id") == qid)
        results_list = q_data["result"].to_list()
        if len(results_list) < expected_responses_per_question:
            questions_with_missing += 1
        if all(results_list):
            all_correct_count += 1
        if all(results_list) or not any(results_list):
            all_consistent_count += 1

    confidence = (
        float(all_correct_count / total_unique_questions)
        if total_unique_questions > 0
        else 0.0
    )
    consistency = (
        float(all_consistent_count / total_unique_questions)
        if total_unique_questions > 0
        else 0.0
    )
    missing_questions_percent = (
        float(questions_with_missing / total_unique_questions)
        if total_unique_questions > 0
        else 0.0
    )

    metrics = {
        "global_accuracy": float(global_accuracy),
        "local_accuracy_A": local_accuracy_A,
        "local_accuracy_B": local_accuracy_B,
        "local_accuracy_C": local_accuracy_C,
        "confidence": confidence,
        "consistency": consistency,
        "questions_with_missing_responses": questions_with_missing,
        "missing_questions_percent": missing_questions_percent,
    }

    metrics.update(field_metrics)

    return metrics


def main(
    models: list[str],
    temperature: float,
    dataset: pl.DataFrame,
    system_prompt: str,
    n_try: int = 6,
) -> None:
    """
    Main function to run the benchmark.

    Args:
        models: List of model identifiers to benchmark
        temperature: Temperature setting for model inference
        dataset: Dataset containing questions and answers to benchmark on
        system_prompt: System prompt to use for all models
        n_try: Number of permutations to try per question
    """
    # Get run ID
    try:
        with open("run_id.txt", "r") as f:
            run_id = int(f.read().strip())
    except Exception as e:
        print(f"Error reading run_id: {e}")
        with open("run_id.txt", "w") as f:
            run_id = 0
            f.write(str(run_id))

    os.makedirs(f"results/run_{run_id}", exist_ok=True)

    perf_table = pl.DataFrame(
        schema={
            "model": pl.Utf8,
            "global_accuracy": pl.Float64,
            "accuracy_administratif": pl.Float64,
            "accuracy_fp": pl.Float64,
            "accuracy_international": pl.Float64,
            "accuracy_numerique": pl.Float64,
            "confidence": pl.Float64,
            "consistency": pl.Float64,
            "system_prompt": pl.Utf8,
            "local_accuracy_A": pl.Float64,
            "local_accuracy_B": pl.Float64,
            "local_accuracy_C": pl.Float64,
            "missing_questions_percent": pl.Float64,
        }
    )
    prompt_list = prompt_creation(dataset, n_try)

    print(
        f"=========================\nStarting benchmark for :\n{''.join(f'- {model}\n' for model in models)}"
    )

    for model in models:
        print(f"=========================\nNow benchmarking {model}")
        model_result = model_eval(model, prompt_list, system_prompt, temperature)
        model_file_name = model.split("/")[-1] if "/" in model else model
        model_result.write_csv(f"results/run_{run_id}/{model_file_name}_result.csv")

        metrics = compute_metrics(model_result)
        new_row = pl.DataFrame(
            [
                {
                    "model": model,
                    "global_accuracy": metrics["global_accuracy"],
                    "accuracy_administratif": metrics.get(
                        "accuracy_Culture administrative et juridique", 0.0
                    ),
                    "accuracy_fp": metrics.get("accuracy_Finances publiques", 0.0),
                    "accuracy_international": metrics.get(
                        "accuracy_Organisation, fonctionnement et politiques des institutions européennes",
                        0.0,
                    ),
                    "accuracy_numerique": metrics.get(
                        "accuracy_Culture numérique", 0.0
                    ),
                    "confidence": metrics["confidence"],
                    "consistency": metrics["consistency"],
                    "system_prompt": str(system_prompt),
                    "local_accuracy_A": metrics["local_accuracy_A"],
                    "local_accuracy_B": metrics["local_accuracy_B"],
                    "local_accuracy_C": metrics["local_accuracy_C"],
                    "missing_questions_percent": metrics["missing_questions_percent"],
                }
            ]
        )
        perf_table = perf_table.vstack(new_row)
        print(perf_table.tail(1))
    perf_table.write_csv(f"results/run_{run_id}/perf_table.csv")

    # Write updated run_id to file
    run_id += 1
    with open("run_id.txt", "w") as f:
        f.write(str(run_id))


if __name__ == "__main__":
    models = ["gemma-3-1b-it"]
    temperature = 0
    system_prompt = "You are a helpful assistant. Please provide your answer using only the corresponding letter."
    dataset = pl.read_parquet("ira_dataset.parquet")
    # dataset = dataset.sample(n=100, seed=40)
    print(dataset.head())
    main(
        models,
        temperature=0,
        dataset=dataset,
        system_prompt=system_prompt,
        n_try=6,
    )
