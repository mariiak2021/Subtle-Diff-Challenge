import json


def load_data(annoation_file):
    with open(annoation_file, 'r') as f:
        data = json.load(f)

    return data

def build_user_dict(user_dict):
    user_answers = {}
    for instance in user_dict:
        print (type(user_dict))
        name = instance["name"]
        if name not in user_answers:
            user_answers[name] = {}
        for attribute in instance["attributes"]:
            key = attribute["key"]
            user_answers[name][key] = attribute["answer"]
    return user_answers

def calculate_accuracy(test_annotation_file, user_submission_file):
    """
    Calculate the accuracy of user submissions against the ground truth annotations.

    Parameters:
    test_annotation_file (str): Path to the JSON file containing the ground truth annotations.
    user_submission_file (str): Path to the JSON file containing the user's submissions.

    Returns:
    float: The accuracy of the user submissions.
    """

    # Load data from the files
    gt_dict = load_data(test_annotation_file)
    user_dict = load_data(user_submission_file)

    user_answers = build_user_dict(user_dict)

    # Initialize a list to hold valid comparisons
    valid_comparisons = []

    # Iterate through the ground truth annotations
    for instance in gt_dict:
        name = instance["name"]
        for attribute in instance["attributes"]:
            key = attribute['key']
            gt = attribute['answer']

            user_answer = user_answers.get(name, {}).get(key, "")

            if user_answer == "":
                print(f"Missing answer for {name} of {key}")
                continue
            if user_answer not in {'After', 'Before'}:
                print(f"Unexpected answer '{user_answer}' for {name} of {key}")
                continue

            # Add the valid comparison to the list
            valid_comparisons.append({"name": name, "key": key, "gt_answer": gt, "user_answer": user_answer})

    # Calculate the number of correct predictions
    correct_predictions = sum(1 for item in valid_comparisons if item["gt_answer"] == item["user_answer"])

    # Calculate accuracy
    accuracy = correct_predictions / len(gt_dict) if len(gt_dict) > 0 else 0
    return accuracy


def evaluate_accuracy(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting test.....")
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
        output["result"] = [
            {
                "train_split": {
                    "accuracy": calculate_accuracy(test_annotation_file, user_submission_file),
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "dif":
        print("Evaluating for Test Phase")
        output["result"] = [

            {
                "test_split": {
                    "accuracy": calculate_accuracy(test_annotation_file, user_submission_file),
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
        print (output)
    return output