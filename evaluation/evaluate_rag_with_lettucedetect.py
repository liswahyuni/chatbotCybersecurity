import os
import sys
import pandas as pd
from datasets import load_dataset

try:
    from lettucedetect.models.inference import HallucinationDetector
    print("LettuceDetect HallucinationDetector imported successfully.")
except ImportError:
    print("ERROR: LettuceDetect module not found. Please install it with 'pip install lettucedetect -U'.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Could not import HallucinationDetector from LettuceDetect: {e}")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from core_logic.orchestrator import RagPipeline
except ImportError as e:
    print(f"Error importing RAG pipeline components: {e}")
    sys.exit(1)

EVAL_DATASET_NAME = "preemware/pentesting-eval"
EVAL_DATASET_SPLIT = "train"
EVAL_SAMPLE_PERCENTAGE = 0.20

QUESTION_COLUMN = "question"
GROUND_TRUTH_INDEX_COLUMN = "answer"
CHOICES_COLUMN = "choices"

LLM_MODEL_NAME = "qwen:0.5b"

OUTPUT_CSV_FILENAME = f"rag_lettucedetect_results_{LLM_MODEL_NAME.replace(':', '_')}.csv"

LETTUCEDETECT_MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
LETTUCEDETECT_METHOD = "transformer"

def main():
    print(f"Loading dataset: {EVAL_DATASET_NAME}, split: {EVAL_DATASET_SPLIT}")
    try:
        dataset = load_dataset(EVAL_DATASET_NAME)
    except Exception as e:
        print(f"Failed to load dataset '{EVAL_DATASET_NAME}'. Error: {e}")
        return

    if EVAL_DATASET_SPLIT not in dataset:
        print(f"Error: Split '{EVAL_DATASET_SPLIT}' not found. Available splits: {list(dataset.keys())}")
        return
    eval_data_full = dataset[EVAL_DATASET_SPLIT]

    required_columns = [QUESTION_COLUMN, GROUND_TRUTH_INDEX_COLUMN, CHOICES_COLUMN]
    for col in required_columns:
        if col not in eval_data_full.column_names:
            print(f"Error: Required column '{col}' not found. Available columns: {eval_data_full.column_names}")
            return

    num_total_examples = len(eval_data_full)
    num_samples = int(num_total_examples * EVAL_SAMPLE_PERCENTAGE)
    if num_samples == 0 and num_total_examples > 0:
        num_samples = min(num_total_examples, 5)
        print(f"Warning: Sample percentage resulted in 0. Using {num_samples} samples.")
    elif num_samples == 0:
        print("No data to sample. Exiting.")
        return

    sampled_eval_data = eval_data_full.shuffle(seed=42).select(range(num_samples))
    print(f"Sampling {len(sampled_eval_data)} examples (out of {num_total_examples} total) for evaluation.")

    print(f"\nInitializing RAG Pipeline with LLM: {LLM_MODEL_NAME}...")
    try:
        rag_system = RagPipeline(llm_model_name=LLM_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing RAG Pipeline: {e}")
        return

    print(f"\nInitializing LettuceDetect HallucinationDetector with model: {LETTUCEDETECT_MODEL_PATH}...")
    try:
        detector_args = {
            "method": LETTUCEDETECT_METHOD,
            "model_path": LETTUCEDETECT_MODEL_PATH,
        }
        hallucination_detector = HallucinationDetector(**detector_args)
        print("LettuceDetect HallucinationDetector initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize HallucinationDetector from LettuceDetect: {e}")
        print("Ensure the model name in 'LETTUCEDETECT_MODEL_PATH' is correct and all LettuceDetect dependencies are installed.")
        sys.exit(1)

    results_data = []
    total_hallucinated_spans = 0
    num_answers_with_hallucination = 0

    print("\n--- Starting Hallucination Evaluation Loop with LettuceDetect ---")

    for i, example in enumerate(sampled_eval_data):
        print(f"\nProcessing example {i+1}/{len(sampled_eval_data)} for hallucination detection...")
        
        question_raw = example[QUESTION_COLUMN]
        ground_truth_index_raw = example[GROUND_TRUTH_INDEX_COLUMN]
        choices_raw = example[CHOICES_COLUMN]

        question = str(question_raw) if question_raw is not None else ""
        
        ground_truth_answer_text = ""
        if isinstance(choices_raw, list) and isinstance(ground_truth_index_raw, int) and 0 <= ground_truth_index_raw < len(choices_raw):
            ground_truth_answer_text = str(choices_raw[ground_truth_index_raw])
        else:
            print(f"  Warning: Could not retrieve ground truth answer text for example {i+1}. Skipping.")
            print(f"    Choices: {choices_raw}, Index: {ground_truth_index_raw}")
            continue 

        if not question.strip() or not ground_truth_answer_text.strip():
            print(f"  Warning: Skipping example {i+1} because question or ground truth answer is empty after processing.")
            continue
        
        print(f"  Question (snippet): {question[:150]}...")

        try:
            generated_answer_str, retrieved_context_list = rag_system.process_query(question, stream=False, top_k_retrieval=3) 
            
            if not isinstance(retrieved_context_list, list):
                retrieved_context_list = [str(retrieved_context_list)] if retrieved_context_list is not None else []
            retrieved_context_str = "\n".join(retrieved_context_list)
        except Exception as e:
            print(f"  Error calling LLM: {e}")
            generated_answer_str = f"Error: {e}"
            retrieved_context_list = []
            retrieved_context_str = "Error during retrieval"

        if "Ollama API Response Error" in generated_answer_str or "model not found" in generated_answer_str :
            print(f"  LLM WARNING: {generated_answer_str}")
        
        print(f"  Generated Answer (snippet): {generated_answer_str[:100]}...")
        print(f"  Retrieved Context (snippet): {retrieved_context_str[:100]}...")

        predictions = []
        has_hallucination = False
        
        if generated_answer_str.strip() and not generated_answer_str.startswith("Error:"):
            try:
                predictions = hallucination_detector.predict(
                    context=retrieved_context_list,
                    question=question, 
                    answer=generated_answer_str, 
                    output_format="spans"
                )

                if predictions:
                    has_hallucination = True
                    num_answers_with_hallucination += 1
                    total_hallucinated_spans += len(predictions)
                    print(f"  Hallucination Detected: YES ({len(predictions)} spans detected)")
                    for pred in predictions:
                        print(f"    - Span: '{pred['text']}' (Conf: {pred['confidence']:.4f})")
                else:
                    print(f"  Hallucination Detected: NO")

            except Exception as e:
                has_hallucination = f"Detection Error: {e}"
                predictions = [{"error": str(e)}]
                print(f"  Failed to run hallucination detection with LettuceDetect: {e}")
        else:
            print("  Cannot detect hallucination from empty or error answer.")

        results_data.append({
            "question": question,
            "ground_truth_answer": ground_truth_answer_text,
            "generated_answer": generated_answer_str,
            "retrieved_context": retrieved_context_str,
            "has_hallucination": has_hallucination,
            "hallucination_predictions": str(predictions)
        })

    if not results_data:
        print("\nNo results processed. Evaluation cannot proceed.")
        return

    results_df = pd.DataFrame(results_data)

    print("\n\n--- Hallucination Evaluation Summary with LettuceDetect ---")
    total_samples_processed = len(results_df)
    if total_samples_processed > 0:
        hallucination_percentage = (num_answers_with_hallucination / total_samples_processed) * 100
        print(f"Number of Samples Processed: {total_samples_processed}")
        print(f"Number of Answers with Detected Hallucinations: {num_answers_with_hallucination}")
        print(f"Total Detected Hallucination Spans: {total_hallucinated_spans}")
        print(f"Percentage of Answers with Hallucinations: {hallucination_percentage:.2f}%")
    else:
        print("No samples processed for hallucination detection.")

    output_path = os.path.join(current_dir, OUTPUT_CSV_FILENAME)
    try:
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nDetailed evaluation results saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

    print("\n--- Sample Responses for Qualitative Report (from CSV) ---")
    num_qual_samples = min(len(results_df), 5)
    sample_for_report_df = results_df.head(num_qual_samples)

    print(f"\nDisplaying first {num_qual_samples} samples for quick review:")
    for index, row in sample_for_report_df.iterrows():
        print("-" * 40)
        print(f"Question: {row['question']}")
        print(f"Ground Truth Answer (Text): {str(row['ground_truth_answer'])[:200]}...")
        print(f"Generated Answer: {str(row['generated_answer'])[:200]}...")
        print(f"Retrieved Context: {str(row['retrieved_context'])[:200]}...")
        print(f"Hallucination Detected: {row.get('has_hallucination', 'N/A')}")
        print(f"Hallucination Prediction Details: {str(row.get('hallucination_predictions', 'N/A'))[:150]}...")
    print("-" * 40)
    print(f"\nFull details are in '{OUTPUT_CSV_FILENAME}'.")
    print("\nEvaluation script finished.")

if __name__ == "__main__":
    main()