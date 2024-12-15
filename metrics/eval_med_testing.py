import json

# function to check for partial best match with score, used for evaluation
def check_partial_best_match(correct_expected_answer, generated_answer):
    """
    Check for partial best match between correct expected answer and generated answer.
    Args:
        correct_expected_answer (str): Correct expected answer.
        generated_answer (str): Generated answer.
    Returns:
        bool: True if partial best match found, False otherwise.
    """
    # Tokenize correct expected answer and generated answer
    correct_expected_answer_tokens = correct_expected_answer.split()
    generated_answer_tokens = generated_answer.split()
    # Check for partial best match
    for i in range(len(correct_expected_answer_tokens)):
        for j in range(i + 1, len(correct_expected_answer_tokens) + 1):
            partial_match = " ".join(correct_expected_answer_tokens[i:j])
            if partial_match in generated_answer:
                return True
    return False

def evaluate_testing_data_medqa():
    total_correct_answers = 0
    total_incorrect_answers = 0
    with open("H:\\workspace\\nameless-ai\\metrics\\data\\medqa_responses1.jsonl", "r", encoding="utf-8") as file:
        """
        Sample:
            {
                "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?", 
                "answer": "Tell the attending that he cannot fail to disclose this mistake", 
                "response": " The correct answer is is C) Tell the attending that he cannot fail to disclose this mistake.\n\nThis situation presents a conflict between honesty and patient anxiety. While it's understandable that the attending physician wants to avoid worrying the patient, withholding information about an adverse event can damage trust in healthcare providers and lead to further complications if not addressed properly. The resident should inform their supervisor of the error so they can discuss how best to handle this situation with transparency while minimizing unnecessary worry for the patient.\n\nThe correct answer is C) Tell the attending that he cannot fail to disclose this mistake"
            }
        """
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            correct_expected_answer = data["answer"]
            generated_answer = data["response"]
            # Check if correct answer is in generated answer (check for partial best match as well)
            if correct_expected_answer in generated_answer:
                # print("Correct Answer Found")
                total_correct_answers += 1
            else:
                # print("Correct Answer Not Found")
                total_incorrect_answers += 1

    print("Total Correct Answers: ", total_correct_answers)
    print("Total Incorrect Answers: ", total_incorrect_answers)
    print("Precision: ", total_correct_answers / (total_correct_answers + total_incorrect_answers))
    print("Recall: ", total_correct_answers / len(lines))
    print("F1 Score: ", (2 * total_correct_answers) / (2 * total_correct_answers + total_incorrect_answers + len(lines)))

def evaluate_testing_data_medmcqa():
    total_correct_answers = 0
    total_incorrect_answers = 0
    with open("H:\\workspace\\nameless-ai\\metrics\\data\\medmcqa_responses.jsonl", "r", encoding="utf-8") as file:
        """
        Sample:
            {
                "question": "Which of the following is not true for myelinated nerve fibers:", 
                "answer": "Impulse through myelinated fibers is slower than non-myelinated fibers", 
                "response": " Based on the information information provided, I will choose option a) Impulse through myelinated fibers is slower than non-myelinated fibers.\n\nExplanation: Myelination increases the speed of impulse conduction in nerve fibers. This is because the electrical signal can jump from node to node (saltatory conduction), allowing for faster transmission compared to unmyelinated fibers, where impulses travel continuously along the length of the fiber. Therefore, option a) is incorrect.\n\nThe other options are correct:\n\n* b) Membrane currents are generated at nodes of Ranvier: This is true because myelin acts as an electrical insulator between these gaps (nodes), allowing for saltatory conduction.\n* c) Saltatory conduction of impulses is seen: As mentioned earlier, this occurs due to the presence of a myelination sheath that allows signals to jump from node to node.\n* d) Local anesthesia is effective only when the nerve is not covered by myelin sheath: This statement holds true because local anesthetics work by blocking sodium channels in the neuronal membrane. Myelin acts as a barrier, preventing these drugs from reaching their target sites on unmyelinated fibers.\n\nTherefore, option a) Impulse through myelinated fibers is slower than non-myelinated fibers is not correct for myelinated nerve fibers."
            }
        """
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            correct_expected_answer = data["answer"]
            generated_answer = data["response"]
            # Check if correct answer is in generated answer (check for partial best match as well)
            if correct_expected_answer in generated_answer:
                # print("Correct Answer Found")
                total_correct_answers += 1
            else:
                # print("Correct Answer Not Found")
                total_incorrect_answers += 1

    print("Total Correct Answers: ", total_correct_answers)
    print("Total Incorrect Answers: ", total_incorrect_answers)
    print("Precision: ", total_correct_answers / (total_correct_answers + total_incorrect_answers))
    print("Recall: ", total_correct_answers / len(lines))
    print("F1 Score: ", (2 * total_correct_answers) / (2 * total_correct_answers + total_incorrect_answers + len(lines)))

if __name__ == "__main__":
    evaluate_testing_data_medqa()
    evaluate_testing_data_medmcqa()