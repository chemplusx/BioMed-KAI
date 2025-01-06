import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
from bert_score import score as bert_score
import json

nltk.download('wordnet')  # Needed for METEOR metric
nltk.download('punkt_tab')


# Cosine Similarity Calculation
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# BERTScore Calculation
def calculate_bertscore(reference, hypothesis):
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return float(P[0]), float(R[0]), float(F1[0])

# KL Divergence Function
def calculate_kl_divergence(p_probs, q_probs):
    kl_divergence = 0.0
    for p, q in zip(p_probs, q_probs):
        if p > 0 and q > 0:
            kl_divergence += p * math.log2(p / q)
    return kl_divergence

# Generate Probability Distribution
def get_probability_distribution(text):
    word_counts = Counter(text.split())
    total_words = sum(word_counts.values())
    probabilities = [count / total_words for count in word_counts.values()]
    return probabilities

# BLEU Score Calculation
def calculate_bleu(reference, hypothesis):
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
    return bleu_score

# METEOR Score Calculation
def calculate_meteor(reference, hypothesis):
    """
    Calculate METEOR score with tokenized input.
    Args:
        reference (str): The reference text.
        hypothesis (str): The generated text.
    Returns:
        float: METEOR score.
    """
    reference_tokens = word_tokenize(reference)  # Tokenize reference
    hypothesis_tokens = word_tokenize(hypothesis)  # Tokenize hypothesis
    return meteor_score([reference_tokens], hypothesis_tokens)

# ROUGE Score Calculation
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


def get_metrics(generated_text, expected_text):
    all_words = set(expected_text.split()).union(set(generated_text.split()))
    p_dist = {word: 0.0 for word in all_words}
    q_dist = {word: 0.0 for word in all_words}
    for word, count in Counter(expected_text.split()).items():
        p_dist[word] = count / sum(Counter(expected_text.split()).values())
    for word, count in Counter(generated_text.split()).items():
        q_dist[word] = count / sum(Counter(generated_text.split()).values())
    p_probs = [p_dist[word] for word in all_words]
    q_probs = [q_dist[word] for word in all_words]

    kl_divergence = calculate_kl_divergence(p_probs, q_probs)
    # print(f"Kullback-Leibler (KL) Divergence: {kl_divergence:.4f}")

    # BLEU Score
    bleu = calculate_bleu(expected_text, generated_text)
    # print(f"BLEU Score: {bleu:.4f}")
    meteor = calculate_meteor(expected_text, generated_text)
    # print(f"METEOR Score: {meteor:.4f}")
    rouge = calculate_rouge(expected_text, generated_text)
    # print("ROUGE Scores:")
    # for key, value in rouge.items():
        # print(f"  {key.upper()}: Precision={value.precision:.4f}, Recall={value.recall:.4f}, F1-Score={value.fmeasure:.4f}")
    cosine_sim = calculate_cosine_similarity(expected_text, generated_text)
    # print(f"Cosine Similarity: {cosine_sim:.4f}")
    bertscore_p, bertscore_r, bertscore_f1 = calculate_bertscore(expected_text, generated_text)
    # print(f"BERTScore: Precision={bertscore_p:.4f}, Recall={bertscore_r:.4f}, F1={bertscore_f1:.4f}")
    return kl_divergence, bleu, meteor, rouge, cosine_sim, bertscore_p, bertscore_r, bertscore_f1

# Example Usage
# if __name__ == "__main__":
#     # Example generated and expected text
#     generated_text = """
#         Diagnosing CRPS can be challenging, as the symptoms are often nonspecific and may mimic other conditions. A comprehensive diagnostic approach is essential to confirm a diagnosis.

#         Diagnostic Criteria
# The International Association for the Study of Pain (IASP) has established criteria for diagnosing CRPS:

# History: The patient must have experienced an initiating noxious event, such as trauma or surgery.
# Sustained pain: Chronic burning pain in one limb that is disproportionate to any inciting injury.
# Dysesthesia: Abnormal sensations (e.g., numbness, tingling) and hypersensitivity to touch or temperature changes.
# Edema (swelling): Swelling of the affected area, which may be accompanied by redness and warmth.
# Diagnostic Tests
# The following tests can help support a diagnosis:

# Physical examination: A thorough physical exam is essential in assessing pain distribution, range of motion, temperature differences between limbs, and skin texture.
# Imaging studies:
# X-rays: To rule out other conditions that may cause similar symptoms (e.g., fractures or osteoarthritis).
# MRI/CT scans: May show changes in bone density, soft tissue swelling, or nerve damage.
# Nerve conduction studies: Can help identify any nerve damage contributing to pain and sensory disturbances.
#     """
#     expected_text = """
#         Ruling out other conditions: CRPS diagnosis typically involves excluding other conditions that may present with similar symptoms. This includes:

# Blood tests to rule out underlying infections or rheumatoid arthritis
# Imaging studies (MRI, X-ray, bone scan) to rule out underlying problems with tissue, bones, or joints
# Physical examination: A thorough physical examination by a healthcare provider, such as a neurologist, orthopedist, or plastic surgeon, is crucial. This may include:

# Gentle examination to assess for physical signs of CRPS, such as swelling, changes in skin temperature and appearance
# Patient-drawn outline of abnormal skin to identify affected nerve
# Specialized tests: While there is no single diagnostic test for CRPS, the following tests can provide valuable information:

# Nerve conduction studies (detect some but not all CRPS-associated nerve injuries)
# Imaging nerves using ultrasound or magnetic resonance imaging (MRI), also called magnetic resonance neurography (MRN) (sometimes reveals underlying nerve damage)
# Triple-phase bone scans (using a dye) (may show CRPS-associated excess bone resorption)
# Characteristic bone and bone marrow abnormalities on MRI (can help identify the injured nerve)
    # """

def get_medalpaca_metrics():
    """
        jsonl file with each line like below
        nameless-ai\compute-be\data\medalpaca_responses.jsonl
        {
            "question": "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?", 
            "answer": "Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.", 
            "response": " Very low Mg2+ levels levels lead to secondary hypocalcemia due to decreased PTH (parathyroid hormone) secretion, resulting in abnormally low Ca2+ levels."
        }

    """
    result = []
    with open("H:\\workspace\\nameless-ai\\metrics\\data\\medalpaca_responses.jsonl", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            question = data["question"]
            answer = data["answer"]
            response = data["response"]
            # Calculate metrics
            kl_divergence, bleu, meteor, rouge, cosine_sim, bertscore_p, bertscore_r, bertscore_f1 = get_metrics(response, answer)
            print("-----------------------------------------------------------------")
            result.append({
                "question": question,
                "answer": answer,
                "response": response,
                "metrics": {
                    "kl_divergence": kl_divergence,
                    "bleu": bleu,
                    "meteor": meteor,
                    "rouge": rouge,
                    "cosine_similarity": cosine_sim,
                    "bertscore_precision": bertscore_p,
                    "bertscore_recall": bertscore_r,
                    "bertscore_f1": bertscore_f1
                }
            })
    
    # Calculate cumulative metrics: mean, median, min, max
    kl_divergences = [r["metrics"]["kl_divergence"] for r in result]
    bleus = [r["metrics"]["bleu"] for r in result]
    meteors = [r["metrics"]["meteor"] for r in result]
    cosine_sims = [r["metrics"]["cosine_similarity"] for r in result]
    bertscore_ps = [r["metrics"]["bertscore_precision"] for r in result]
    bertscore_rs = [r["metrics"]["bertscore_recall"] for r in result]
    bertscore_f1s = [r["metrics"]["bertscore_f1"] for r in result]

    cumulative_metrics = {
        "kl_divergence": {
            "mean": sum(kl_divergences) / len(kl_divergences),
            "median": sorted(kl_divergences)[len(kl_divergences) // 2],
            "min": min(kl_divergences),
            "max": max(kl_divergences)
        },
        "bleu": {
            "mean": sum(bleus) / len(bleus),
            "median": sorted(bleus)[len(bleus) // 2],
            "min": min(bleus),
            "max": max(bleus)
        },
        "meteor": {
            "mean": sum(meteors) / len(meteors),
            "median": sorted(meteors)[len(meteors) // 2],
            "min": min(meteors),
            "max": max(meteors)
        },
        "cosine_similarity": {
            "mean": sum(cosine_sims) / len(cosine_sims),
            "median": sorted(cosine_sims)[len(cosine_sims) // 2],
            "min": min(cosine_sims),
            "max": max(cosine_sims)
        },
        "bertscore_precision": {
            "mean": sum(bertscore_ps) / len(bertscore_ps),
            "median": sorted(bertscore_ps)[len(bertscore_ps) // 2],
            "min": min(bertscore_ps),
            "max": max(bertscore_ps)
        },
        "bertscore_recall": {
            "mean": sum(bertscore_rs) / len(bertscore_rs),
            "median": sorted(bertscore_rs)[len(bertscore_rs) // 2],
            "min": min(bertscore_rs),
            "max": max(bertscore_rs)
        },
        "bertscore_f1": {
            "mean": sum(bertscore_f1s) / len(bertscore_f1s),
            "median": sorted(bertscore_f1s)[len(bertscore_f1s) // 2],
            "min": min(bertscore_f1s),
            "max": max(bertscore_f1s)
        }
    }
    return cumulative_metrics

def get_custom_metrics():
    # Example generated and expected text
    """
        nameless-ai\compute-be\data\custom_biomed_responses.jsonl
        {
            "question": "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?", 
            "answer": "Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.", 
            "response": " Very low Mg2+ levels levels lead to secondary hypocalcemia due to decreased PTH (parathyroid hormone) secretion, resulting in abnormally low Ca2+ levels."
        }
    """
    result = []
    with open("H:\\workspace\\nameless-ai\\metrics\\data\\custom_biomed_responses.jsonl", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            question = data["question"]
            answer = data["answer"]
            response = data["response"]
            # Calculate metrics
            kl_divergence, bleu, meteor, rouge, cosine_sim, bertscore_p, bertscore_r, bertscore_f1 = get_metrics(response, answer)
            print("-----------------------------------------------------------------")
            result.append({
                "question": question,
                "answer": answer,
                "response": response,
                "metrics": {
                    "kl_divergence": kl_divergence,
                    "bleu": bleu,
                    "meteor": meteor,
                    "rouge": rouge,
                    "cosine_similarity": cosine_sim,
                    "bertscore_precision": bertscore_p,
                    "bertscore_recall": bertscore_r,
                    "bertscore_f1": bertscore_f1
                }
            })
    
    # Calculate cumulative metrics: mean, median, min, max
    kl_divergences = [r["metrics"]["kl_divergence"] for r in result]
    bleus = [r["metrics"]["bleu"] for r in result]
    meteors = [r["metrics"]["meteor"] for r in result]
    cosine_sims = [r["metrics"]["cosine_similarity"] for r in result]
    bertscore_ps = [r["metrics"]["bertscore_precision"] for r in result]
    bertscore_rs = [r["metrics"]["bertscore_recall"] for r in result]
    bertscore_f1s = [r["metrics"]["bertscore_f1"] for r in result]

    cumulative_metrics = {
        "kl_divergence": {
            "mean": sum(kl_divergences) / len(kl_divergences),
            "median": sorted(kl_divergences)[len(kl_divergences) // 2],
            "min": min(kl_divergences),
            "max": max(kl_divergences)
        },
        "bleu": {
            "mean": sum(bleus) / len(bleus),
            "median": sorted(bleus)[len(bleus) // 2],
            "min": min(bleus),
            "max": max(bleus)
        },
        "meteor": {
            "mean": sum(meteors) / len(meteors),
            "median": sorted(meteors)[len(meteors) // 2],
            "min": min(meteors),
            "max": max(meteors)
        },
        "cosine_similarity": {
            "mean": sum(cosine_sims) / len(cosine_sims),
            "median": sorted(cosine_sims)[len(cosine_sims) // 2],
            "min": min(cosine_sims),
            "max": max(cosine_sims)
        },
        "bertscore_precision": {
            "mean": sum(bertscore_ps) / len(bertscore_ps),
            "median": sorted(bertscore_ps)[len(bertscore_ps) // 2],
            "min": min(bertscore_ps),
            "max": max(bertscore_ps)
        },
        "bertscore_recall": {
            "mean": sum(bertscore_rs) / len(bertscore_rs),
            "median": sorted(bertscore_rs)[len(bertscore_rs) // 2],
            "min": min(bertscore_rs),
            "max": max(bertscore_rs)
        },
        "bertscore_f1": {
            "mean": sum(bertscore_f1s) / len(bertscore_f1s),
            "median": sorted(bertscore_f1s)[len(bertscore_f1s) // 2],
            "min": min(bertscore_f1s),
            "max": max(bertscore_f1s)
        }
    }
    return cumulative_metrics

if __name__ == "__main__":
    medalpaca_metrics = get_medalpaca_metrics()
    print("MedALPaCA Metrics:")
    print(json.dumps(medalpaca_metrics, indent=4))

    custom_metrics = get_custom_metrics()
    print("Custom Biomed Metrics:")
    print(json.dumps(custom_metrics, indent=4))
