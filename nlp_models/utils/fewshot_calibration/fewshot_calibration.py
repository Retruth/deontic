from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax

def get_logits(model, tokenizer, prompt):
    """
    GPT 모델에서 주어진 프롬프트에 대한 logits를 가져옵니다.
    Args:
        model: 로드된 GPT-like 모델.
        tokenizer: 해당 모델의 토크나이저.
        prompt (str): 프롬프트 텍스트.
    Returns:
        logits (torch.Tensor): 모델 출력 logits.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits[0, -1, :]  # 마지막 토큰에 대한 logits 반환

def contextual_calibration(model, tokenizer, few_shot_prompt, test_prompt, content_free_inputs):
    """
    Few-shot Prompting 기반 Contextual Calibration 구현.
    Args:
        model: 로드된 GPT-like 모델.
        tokenizer: 해당 모델의 토크나이저.
        few_shot_prompt (str): Few-shot 예제 포함 프롬프트.
        test_prompt (str): 테스트 프롬프트.
        content_free_inputs (list): Content-free 입력 리스트 (e.g., ["N/A", "", "[MASK]"]).
    Returns:
        calibrated_probs (np.array): 보정된 확률.
        predicted_class (int): 보정된 클래스.
    """
    # 1. Content-Free Input에 대한 logits 계산
    baseline_probs = []
    for cfi in content_free_inputs:
        cfi_prompt = few_shot_prompt + cfi
        logits = get_logits(model, tokenizer, cfi_prompt)
        probs = softmax(logits.detach().numpy())
        baseline_probs.append(probs)

    # Content-Free 평균 확률 계산
    baseline_probs = np.mean(baseline_probs, axis=0)

    # 2. 테스트 프롬프트에 대한 logits 계산
    test_logits = get_logits(model, tokenizer, few_shot_prompt + test_prompt)
    test_probs = softmax(test_logits.detach().numpy())

    # 3. Affine Transformation 적용
    W = np.diag(1.0 / baseline_probs)  # 대각 행렬
    b = np.zeros_like(baseline_probs)  # 편향
    calibrated_probs = softmax(np.dot(W, test_probs) + b)

    # 4. 최종 클래스 예측
    predicted_class = np.argmax(calibrated_probs)
    return calibrated_probs, predicted_class

# 모델과 토크나이저 로드
model_name = "gpt2"  # 필요에 따라 다른 모델 사용 가능
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Few-shot Prompt와 테스트 데이터 설정
few_shot_prompt = """Input: Subpar acting. Sentiment: Negative
Input: Beautiful film. Sentiment: Positive
Input:"""
test_prompt = "This movie was dull and boring. Sentiment:"
content_free_inputs = ["N/A", "", "[MASK]"]

# 실행
calibrated_probs, predicted_class = contextual_calibration(
    model, tokenizer, few_shot_prompt, test_prompt, content_free_inputs
)

print("Calibrated Probabilities:", calibrated_probs)
print("Predicted Class:", predicted_class)
