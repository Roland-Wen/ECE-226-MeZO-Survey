SMALL_MODELS = ("RoBERTa-Base", "RoBERTa-large", "RoBERTa-large ")
MEDIUM_MODELS = ("OPT-1.3B", "OPT-2.7B")
LARGE_MODELS = ("LLaMA-7b", "LLaMA2-7b", "LlaMA3-8b", "OPT-13B")

TASK_TYPE_TO_DATASET = {"Natural Language Inference": ("SNLI", "MNLI", "RTE", "CB"),
                        "Sentiment Analysis": ("SST-2", "SST-5"),
                        "Reading Comprehension/Question Answering": ("SQuAD", "MultiRC", "ReCoRD", "DROP", "BoolQ"),
                        "Commonsense & Causal Reasoning": ("COPA", "WSC"),
                        "Word Sense/Contextual Meaning": ["WiC"],
                        "Question Classification": ["TREC"]}

def get_dataset_type(dataset_name):
    for task_type, datasets in TASK_TYPE_TO_DATASET.items():
        if dataset_name in datasets:
            return task_type
    return "Unknown"

def get_model_size(model_name):
    if model_name in SMALL_MODELS:
        return "small"
    elif model_name in MEDIUM_MODELS:
        return "medium"
    elif model_name in LARGE_MODELS:
        return "large"
    else:
        return "unknown"