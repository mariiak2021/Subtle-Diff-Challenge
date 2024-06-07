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
    gpt_similarity = get_gpt4_eval(gt_eval, user_eval)

    return result['bleu'], result['cider'], gpt_similarity


def get_gpt4_eval(gt_answers, user_answers):

    gpt_description = "Create a scoring system that evaluates the sentences in the results for their ability to capture the same or equivalent meanings as those in the reference text, especially when describing changes in serial images. " \
                      "The total score is 100, with the lowest being 0. " \
                      "This system should focus on interpreting the underlying meanings, implications, and context of the descriptions, beyond their literal wordings. " \
                      "Regardless of the order of sentences in the reference text and results. " \
                      "Start by identifying all key changes or observations mentioned in the reference text. Then, distribute the total score (100 points) evenly across these key points. " \
                      "When evaluating the results, it is crucial to assess the essence and the deeper implications of each description, considering metaphors, comparative language, and contextual cues. " \
                      "For example, descriptions like 'sun rays make A brighter than B,' 'B has more clouds than A,' 'A is more bluish than B,' and 'A is brighter than B' can all imply similar observations, despite their varied expressions. " \
                      "Award full points for results that, despite different phrasing, align closely in meaning and implication with the reference. " \
                      "Deduct points for significant discrepancies or for descriptions that differ fundamentally from the reference in essence. " \
                      "This approach values the nuanced understanding of descriptive language and its context, recognizing the importance of interpreting beyond just the literal expressions. " \
                      "Output the final score in the format 'Final Score: [score] out of 100 points.', if there are no sentences, please score it as 0."
    openai.api_key = api_key

    eval_similarity = {}
    random_data_sample = random.sample(list(enumerate(gt_answers.items())), 10)
    similarity_score = []
    for i, (key, gts) in enumerate(random_data_sample):
        res = user_answers[str(key)][0]

        while True:
            try:
                # 画像ABのJson形式の情報を渡す。
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system", "content": gpt_description},
                        {"role": "user",
                         "content": f'The reference text is {gts[0]} '
                                    f'and result text is {res}.'}
                    ]
                )

                break
            except openai.error.Timeout as te:
                print('TimeoutError')
                print(te)


        result = response['choices'][0]['message']['content']
        # print(result)
        eval_similarity.update({str(i): [result]})

        final_score_pattern = r"(\d+) out of 100 points"
        match = re.search(final_score_pattern, result)

        # Extract the score
        final_score = match.group(1) if match else None
        if final_score is not None:
            similarity_score.append(int(final_score))

    return sum(similarity_score)/ len(similarity_score)


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
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
        bleu, cider, gpt_similarity = get_bleu_cider_gpt(test_annotation_file, user_submission_file)
        output["result"] = [
            {
                "train_split": {
                    "BLEU-4": bleu,
                    "CIDEr": cider,
                    "GPT-4": gpt_similarity,
                    "Total": (bleu + cider + gpt_similarity)/3,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        bleu, cider, gpt_similarity = get_bleu_cider_gpt(test_annotation_file, user_submission_file)
        output["result"] = [
            {
                "train_split": {
                    "BLEU-4": bleu,
                    "CIDEr": cider,
                    "GPT-4": gpt_similarity,
                    "Total": (bleu + cider + gpt_similarity)/3,
                }
            },
            {
                "test_split": {
                    "BLEU-4": bleu,
                    "CIDEr": cider,
                    "GPT-4": gpt_similarity,
                    "Total": (bleu + cider + gpt_similarity)/3,
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output


if __name__ == '__main__':
    # evaluate("answer_sheet.json", "test_eval.json", "test")
    evaluate("../annotations/capt_renamed_test_annotations.json", "test_eval_bleu.json", "test")