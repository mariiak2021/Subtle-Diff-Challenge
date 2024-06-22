import random
import json
from bleu_cider.pycocoevalcap.eval import eval
import openai
import re

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


config_file_path = 'config.json'


config = load_config(config_file_path)

api_key = config.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("No OpenAI API key found in config file.")



def load_data(annoation_file):
    with open(annoation_file, 'r') as f:
        data = json.load(f)

    return data


def build_answer_dict(data):
    answer_dict = {}
    for instance in data:
        name = instance["name"]
        # Join all attribute values that are not empty into a single string
        answer = "".join([str(attr["value"]) for attr in instance["attributes"] if attr["value"]])
        answer_dict[name] = answer
    return answer_dict

def get_bleu_cider_gpt(test_annotation_file, user_submission_file):
    gts = load_data(test_annotation_file)
    res = load_data(user_submission_file)

    gt_answers = build_answer_dict(gts)
    user_answers = build_answer_dict(res)

    valid_comparisons = []

    for i, (name, gt_answer) in enumerate(gt_answers.items()):
        user_answer = user_answers.get(name, "")
        if user_answer == "":
            print(f"Missing answer for {name}")
            valid_comparisons.append({"name": name, "gt_answer": gt_answer, "user_answer": ""})
            continue
        valid_comparisons.append({"name": name, "gt_answer": gt_answer, "user_answer": user_answer})

    # Prepare data for evaluation
    gt_eval = {f"{i}": [comp["gt_answer"]] for i, comp in enumerate(valid_comparisons)}
    user_eval = {f"{i}": [comp["user_answer"]] for i, comp in enumerate(valid_comparisons)}

    result = eval(gt_eval, user_eval)
    # gpt_similarity = get_gpt4_eval(gt_eval, user_eval)

    # return result['bleu'], result['cider'], gpt_similarity
    return result['bleu'], result['cider']


def evaluate_bleu_cider(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        bleu, cider = get_bleu_cider_gpt(test_annotation_file, user_submission_file)
        output["result"] = [
            {
                "train_split": {
                    "BLEU-4": bleu,
                    "CIDEr": cider,
                    "Total": (bleu + cider)/2,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "cond":
        print("Evaluating for Test Phase")
        bleu, cider = get_bleu_cider_gpt(test_annotation_file, user_submission_file)
        output["result"] = [
            {
                "train_split": {
                    "BLEU-4": bleu,
                    "CIDEr": cider,
                    "Total": (bleu + cider)/2,
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output


if __name__ == '__main__':
    # evaluate("answer_sheet.json", "test_eval.json", "test")
    evaluate("../annotations/capt_renamed_test_annotations.json", "test_eval_bleu.json", "cond")
