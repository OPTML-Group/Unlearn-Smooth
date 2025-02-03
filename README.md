# Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond

This is the official code repository for the paper [Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond
]().

## Abstract
The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. 
However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. 

## Contributors
* [Chongyu Fan](https://a-f1.github.io/)
* [Jinghan Jia]([https://ljcc0930.github.io/](https://jinghanjia.netlify.app/)

## Cite This Work
```
